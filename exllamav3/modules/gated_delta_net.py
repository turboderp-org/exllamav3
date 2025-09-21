from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm, LayerNorm
from ..constants import PAGE_SIZE
from ..cache import Cache
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from ..util import profile_opt
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
import torch.distributed as dist
from .gated_rmsnorm import GatedRMSNorm
from ..cache import CacheableState

"""
batch_shape: tuple of (bsz, _)
past_len: int (default: 0)

*OR*

cache_seqlens: shape (bsz) 
"""

# Fallback functions if causal_conv1d not installed
# TODO: Custom kernels

def torch_causal_conv1d_update_function(
    x,
    conv_state,
    weight,
    bias = None,
    activation = True,
    cache_seqlens = None,
    conv_state_indices = None
):
    bsz, dim, seq_len = x.shape
    state_len = conv_state.shape[-1]

    y = torch.cat([conv_state, x], dim = -1).to(weight.dtype)
    conv_state.copy_(y[:, :, -state_len:])
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y


def torch_causal_conv1d_fwd_function(
    x,
    weight,
    bias,
    seq_idx = None,
    initial_states = None,
    final_states_out = None,
    silu_activation = True,
):
    # Differs from Qwen3-Next Transformers impl. but corresponds better to causal_conv1d which uses zeros
    # as the initial state
    bsz, dim, seq_len = x.shape
    zero_state = torch.zeros((bsz, dim, weight.shape[-1]), dtype = x.dtype, device = x.device)
    y = torch.cat([zero_state, x], dim = -1).to(weight.dtype)
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y


try:
    from causal_conv1d.cpp_functions import causal_conv1d_update_function, causal_conv1d_fwd_function
except ModuleNotFoundError:
    causal_conv1d_update_function = torch_causal_conv1d_update_function
    causal_conv1d_fwd_function = torch_causal_conv1d_fwd_function


class GDN_RecurrentState(CacheableState):
    def __init__(
        self,
        position: int | None = 0,
        positions: list[int] | None = None,
        last_conv_state: torch.Tensor = None,
        last_recurrent_state: torch.Tensor = None,
        batched = False
    ):
        super().__init__()
        self.position = position
        self.positions = positions
        self.last_conv_state = last_conv_state
        self.last_recurrent_state = last_recurrent_state
        self.batched = batched

    @override
    def stash(self):
        # TODO: Option to preallocate and pin space for stashed states
        return GDN_RecurrentState(
            self.position,
            self.positions,
            self.last_conv_state.cpu(),
            self.last_recurrent_state.cpu()
        )

    @override
    def unstash(self, device):
        return GDN_RecurrentState(
            self.position,
            self.positions,
            self.last_conv_state.to(device, non_blocking = True),
            self.last_recurrent_state.to(device, non_blocking = True),
        )

    @override
    def get_size(self):
        if self.last_conv_state is None:
            return 0
        return (
            self.last_conv_state.element_size() * self.last_conv_state.numel() +
            self.last_recurrent_state.element_size() * self.last_recurrent_state.numel()
        )

    def collect_batch(self, batch: list[GDN_RecurrentState]):
        bsz = len(batch)
        lcs = torch.cat([b.last_conv_state for b in batch], dim = 0)
        lrs = torch.cat([b.last_recurrent_state for b in batch], dim = 0)
        positions = [b.position for b in batch]
        return GDN_RecurrentState(None, positions, lcs, lrs, True)

    def distribute_batch(self, batch: list[GDN_RecurrentState]):
        for i, b in enumerate(batch):
            b.last_conv_state.copy_(self.last_conv_state[i:i+1, ...])
            b.last_recurrent_state.copy_(self.last_recurrent_state[i:i+1, ...])
            b.position = self.positions[i]


def prepare_for_recurrence(input_ids: torch.Tensor, params: dict, model) -> torch.Tensor:
    """
    Add linear attn parameters to state
    """
    batch_shape = params.get("batch_shape")
    cache_seqlens = params.get("cache_seqlens")

    if batch_shape is not None:
        bsz, _ = batch_shape
        past_len = params.get("past_len", 0)
        if past_len > 0:
            rs = params.get("recurrent_states")
            if rs is None:
                raise ValueError(f"Past length given, but no previous state for linear attn in params")
            for k, v in rs.items():
                if not v.batched and v.position != past_len:
                    raise ValueError(f"recurrent states don't match input past_len")
        else:
            rl = model.get_recurrent_layers()
            rs = {attn.layer_idx: GDN_RecurrentState() for attn in rl}
            params["recurrent_states"] = rs

    elif cache_seqlens is not None:
        # (Empty) states must be provided with cache_seqlens
        pass

    else:
        if "recurrent_states" in params:
            raise ValueError(f"recurrent_states given without bsz and seqlens")


class GatedDeltaNet(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        rms_norm_eps: float,
        conv_kernel_size: int,
        key_a_log: str | None = None,
        key_dt_bias: str | None = None,
        key_conv1d: str | None = None,
        key_fused_ba: str | None = None,
        key_fused_qkvz: str | None = None,
        key_norm: str | None = None,
        key_o: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_v_groups = num_v_heads // num_k_heads
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size

        self.out_dtype = out_dtype

        fdim_qkvz = 2 * self.num_k_heads * self.k_head_dim + 2 * self.v_head_dim * self.num_v_heads
        fdim_ba = 2 * self.num_v_heads

        if key_fused_qkvz:
            self.qkvz_proj = Linear(config, f"{key}.{key_fused_qkvz}", hidden_size, fdim_qkvz, qmap = qmap + ".input", out_dtype = torch.float)
            self.register_submodule(self.qkvz_proj)
        else:
            self.qkvz_proj = None

        if key_fused_ba:
            self.ba_proj = Linear(config, f"{key}.{key_fused_ba}", hidden_size, fdim_ba, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.register_submodule(self.ba_proj)
        else:
            self.ba_proj = None

        self.o_proj = Linear(config, f"{key}.{key_o}", 2 * hidden_size, hidden_size, qmap = qmap + ".output", out_dtype = self.out_dtype)
        self.register_submodule(self.o_proj)

        self.norm = GatedRMSNorm(config, f"{key}.{key_norm}", self.rms_norm_eps, out_dtype = torch.half)
        self.register_submodule(self.norm)

        self.a_log = None
        self.a_log_f_exp = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.key_a_log = f"{key}.{key_a_log}"
        self.key_dt_bias = f"{key}.{key_dt_bias}"
        self.key_conv1d_weight = f"{key}.{key_conv1d}.weight"
        self.key_conv1d_bias = f"{key}.{key_conv1d}.bias"

        self.conv_dim = self.k_head_dim * self.num_k_heads

        self.caps.update({
            "recurrent_cache": True
        })

        # self.cache_layers = []
        # self.tp_cache_lookup = {}
        # self.multi_kv = None
        # self.tp_reduce = False
        # self.has_split_cache = False


    @override
    def optimizer_targets(self):
        qkvz = self.qkvz_proj.optimizer_targets()
        return [[qkvz]]


    def load_local(self, device, **kwargs):
        pass


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device)
        self.a_log = self.config.stc.get_tensor(self.key_a_log, self.device, optional = False, allow_bf16 = True)
        self.dt_bias = self.config.stc.get_tensor(self.key_dt_bias, self.device, optional = False, allow_bf16 = True)
        self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device, optional = False, allow_bf16 = True)
        self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True, allow_bf16 = True)
        self.norm.load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        self.a_log = None
        self.a_log_f_exp = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.norm.unload()
        super().unload()


    def split_fused_inputs(self, mixed_qkvz, mixed_ba):
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.k_head_dim + 2 * self.v_head_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.k_head_dim,
            self.k_head_dim,
            (self.num_v_groups * self.v_head_dim),
            (self.num_v_groups * self.v_head_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim = 3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim = 3)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.v_head_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.v_head_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        # TODO: Profile, optimize

        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        bsz, seqlen, _ = x.shape

        # Previous state
        rs = params.get("recurrent_states")
        if rs is not None:
            rs = rs[self.layer_idx]
            conv_state = rs.last_conv_state
            recurrent_state = rs.last_recurrent_state
            save_state = True
        else:
            conv_state = None
            recurrent_state = None
            save_state = False

        # Projections
        qkvz = self.qkvz_proj.forward(x, params).to(torch.bfloat16)
        ba = self.ba_proj.forward(x, params).to(torch.bfloat16)
        q, k, v, z, b, a = self.split_fused_inputs(qkvz, ba)
        q = q.reshape(bsz, seqlen, -1)
        k = k.reshape(bsz, seqlen, -1)
        v = v.reshape(bsz, seqlen, -1)

        # Convolution
        mixed_qkv = torch.cat((q, k, v), dim = -1).transpose(1, 2)

        if conv_state is None:
            if save_state:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                rs.last_conv_state = conv_state
            mixed_qkv = causal_conv1d_fwd_function(
                mixed_qkv,
                self.conv1d_weight.squeeze(1),
                self.conv1d_bias,
                None,
                None,
                None,
                True,
            )

        else:
            mixed_qkv = causal_conv1d_update_function(
                mixed_qkv,
                conv_state,  # Updated inplace
                self.conv1d_weight.squeeze(1),
                self.conv1d_bias,
                True,
                None,
                None,
            )

        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Gate
        k_dim = self.k_head_dim * self.num_k_heads
        v_dim = self.v_head_dim * self.num_v_heads
        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
        q = q.view(bsz, seqlen, -1, self.k_head_dim)
        k = k.view(bsz, seqlen, -1, self.k_head_dim)
        v = v.view(bsz, seqlen, -1, self.v_head_dim)

        if self.a_log_f_exp is None:
            self.a_log_f_exp = -self.a_log.float().exp()
        g = self.a_log_f_exp * F.softplus(a.float() + self.dt_bias)
        beta = b.sigmoid()

        # Grouped attn
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_groups, dim = 2)
            k = k.repeat_interleave(self.num_v_groups, dim = 2)

        # Use chunked rule when advantageous
        if seqlen >= 32:
            core_attn_out, recurrent_state = chunk_gated_delta_rule(
                q, k, v,
                g = g,
                beta = beta,
                initial_state = recurrent_state,
                output_final_state = True,  # cache_params is not None,
                use_qk_l2norm_in_kernel = True,
            )
        else:
            core_attn_out, recurrent_state = fused_recurrent_gated_delta_rule(
                q, k, v,
                g = g,
                beta = beta,
                initial_state = recurrent_state,
                output_final_state = save_state,
                use_qk_l2norm_in_kernel = True,
            )

        # Update cache
        if save_state:
            rs.last_recurrent_state = recurrent_state
            if not rs.batched:
                rs.position += seqlen
            else:
                rs.positions = [r + seqlen for r in rs.positions]

        # Norm
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm.forward(core_attn_out, params, gate = z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        # Output projection
        x = self.o_proj.forward(core_attn_out, params)

        return to2(x, out_dtype, self.out_dtype)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.a_log, self.key_a_log),
            (self.dt_bias, self.key_dt_bias),
            (self.conv1d_weight, self.key_conv1d_weight),
            (self.conv1d_bias, self.key_conv1d_bias),
        ]:
            if x is not None:
                t[k] = x
        return t


    def new_recurrent_state(self):
        return GDN_RecurrentState()


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError()


    def tp_export(self, plan, producer):
        raise NotImplementedError()


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        raise NotImplementedError()