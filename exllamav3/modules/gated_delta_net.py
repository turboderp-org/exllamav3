from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import get_for_device, to2
from . import Module, Linear
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
from .gated_rmsnorm import GatedRMSNorm
from ..cache import CacheableState
from ..util.tensor import g_tensor_cache
from ..util import profile_opt
from ..model.model_tp_shared import TPTensorWrapper

"""
causal_conv1d wrappers and fallback functions 
"""

def causal_conv1d_update_function_torch(
    x,
    conv_state,
    weight,
    bias = None,
):
    bsz, dim, seq_len = x.shape
    state_len = conv_state.shape[-1]

    y = torch.cat([conv_state, x], dim = -1).to(weight.dtype)
    conv_state.copy_(y[:, :, -state_len:])
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y


def causal_conv1d_fwd_function_torch(
    x,
    weight,
    bias,
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


def causal_conv1d_update_with_history(
    x,
    conv_state,
    weight,
    bias,
):
    # TODO: Could use static buffer and skip the concat, investigate if causal_conv1d_cu can return longer state
    bsz, dim, seq_len = x.shape
    state_len = conv_state.shape[-1]

    y = torch.cat([conv_state, x], dim = -1).to(weight.dtype)
    conv_state.copy_(y[:, :, -state_len:])
    conv_state_history = y
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y, conv_state_history


def causal_conv1d_update_function_cu(
    x,
    conv_state,
    weight,
    bias = None,
):
    y = torch.empty_like(x)
    causal_conv1d_cuda.causal_conv1d_update(x, conv_state, weight, bias, y, True, None, None)
    return y


def causal_conv1d_fwd_function_cu(
    x,
    weight,
    bias,
):
    y = torch.empty_like(x)
    causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, None, None, y, None, True)
    return y


try:
    import causal_conv1d_cuda
    causal_conv1d_update_function = causal_conv1d_update_function_cu
    causal_conv1d_fwd_function = causal_conv1d_fwd_function_cu
except ModuleNotFoundError:
    causal_conv1d_update_function = causal_conv1d_update_function_torch
    causal_conv1d_fwd_function = causal_conv1d_fwd_function_torch

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ModuleNotFoundError:
    chunk_gated_delta_rule = None

"""
fla wrapper, reduce overhead by bypassing input_guard and torch custom ops stuff
"""

def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd

    scale = k.shape[-1] ** -0.5
    with torch.cuda.device(q.device.index):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v.contiguous(),
            g,
            None,
            None,
            beta,
            scale,
            initial_state.contiguous() if initial_state is not None else None,
            output_final_state,
            use_qk_l2norm_in_kernel,
            None,
        )
    return o, final_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        inv_norm = 1 / torch.sqrt(
            (x * x).sum(dim = dim, keepdim = True)
            + eps
        )
        return x * inv_norm

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape

    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query

    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()
    g = g.float()

    for i in range(sequence_length):
        q_t = query[:, i, :]
        k_t = key[:, i, :]
        v_t = value[:, i, :]
        g_t = g[:, i, :].exp().unsqueeze(-1)
        beta_t = beta[:, i, :].unsqueeze(-1)
        kv_mem = last_recurrent_state * k_t.unsqueeze(-1)
        kv_mem = kv_mem.sum(dim = -2)
        v_t = v_t - kv_mem * g_t
        upd = k_t.unsqueeze(-1) * v_t.unsqueeze(-2) * beta_t.unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t.unsqueeze(-1) + upd
        core_attn_out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2) * scale

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out, last_recurrent_state


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
        self.batch = None
        self.history = None
        self.conv_history = None

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
    def unstash(self, device, trim_position):
        assert self.position == trim_position
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

    @override
    def get_cachable_interval(self):
        # Recurrent state tracks only one token
        return 0

    def collect_batch(self, batch: list[GDN_RecurrentState]):
        lcs = torch.cat([b.last_conv_state for b in batch], dim = 0)
        lrs = torch.cat([b.last_recurrent_state for b in batch], dim = 0)
        positions = [b.position for b in batch]
        return GDN_RecurrentState(None, positions, lcs, lrs, True)

    def distribute_batch(self, batch: list[GDN_RecurrentState]):
        for i, b in enumerate(batch):
            b.last_conv_state.copy_(self.last_conv_state[i:i+1, ...])
            b.last_recurrent_state.copy_(self.last_recurrent_state[i:i+1, ...])
            if self.history is not None:
                b.history = self.history[i:i+1]
                b.conv_history = self.conv_history[i:i+1]
            b.position = self.positions[i]

    @override
    def reset(self):
        self.last_conv_state = None
        self.last_recurrent_state = None

    @override
    def force_position(self, position: int):
        self.position = position

    @override
    def clone(self):
        return GDN_RecurrentState(
            self.position,
            self.positions,
            self.last_conv_state.clone() if self.last_conv_state is not None else None,
            self.last_recurrent_state.clone() if self.last_recurrent_state is not None else None,
            self.batched,
        )

    @override
    def rewind(self, count: int):
        assert self.history is not None
        assert not self.batched and self.batch is None
        if count == 0: return
        self.last_recurrent_state[0].copy_(self.history[0, -count])
        cdim = self.last_conv_state.shape[-1]
        self.last_conv_state[0].copy_(self.conv_history[0, :, -count - cdim : -count])
        self.position -= count

    @override
    def drop_history(self):
        self.history = None
        self.conv_history = None

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
        beta_scale: float = 1.0,
        key_a_log: str | None = None,
        key_dt_bias: str | None = None,
        key_conv1d: str | None = None,
        key_conv1d_q: str | None = None,
        key_conv1d_k: str | None = None,
        key_conv1d_v: str | None = None,
        key_fused_ba: str | None = None,
        key_fused_qkvz: str | None = None,
        key_qkv: str | None = None,
        key_qkv_alt: list | None = None,
        key_z: str | None = None,
        key_b: str | None = None,
        key_a: str | None = None,
        key_norm: str | None = None,
        key_o: str | None = None,
        a_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        conv1d_weight: torch.Tensor | None = None,
        qkv_proj: Linear | None = None,
        z_proj: Linear | None = None,
        b_proj: Linear | None = None,
        a_proj: Linear | None = None,
        norm: GatedRMSNorm | None = None,
        o_proj: Linear | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)
        self.module_name = "GatedDeltaNet"

        self.q_priority = 1 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_v_groups = num_v_heads // num_k_heads
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads
        self.beta_scale = beta_scale

        self.out_dtype = out_dtype

        self.fdim_qkvz = 2 * self.num_k_heads * self.k_head_dim + 2 * self.num_v_heads * self.v_head_dim
        self.fdim_ba = 2 * self.num_v_heads
        self.fdim_qkv = 2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim

        if key_qkv or key_z:
            assert key_qkv and key_z, \
                "GatedDeltaNet split qkv/z projections require both key_qkv and key_z"
        if key_b or key_a:
            assert key_b and key_a, \
                "GatedDeltaNet split b/a projections require both key_b and key_a"

        if key_fused_qkvz:
            self.qkvz_proj = Linear(
                config,
                f"{key}.{key_fused_qkvz}",
                hidden_size,
                self.fdim_qkvz,
                qmap = qmap + ".input",
                out_dtype = torch.float,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkvz",
            )
            self.qkv_proj = None
            self.z_proj = None
            self.register_submodule(self.qkvz_proj)
        else:
            self.qkvz_proj = None
            self.qkv_proj = None
            self.z_proj = None

        if qkv_proj:
            self.qkv_proj = qkv_proj
            self.z_proj = z_proj
            self.qkvz_proj = None
            self.register_submodule(self.qkv_proj)
            self.register_submodule(self.z_proj)
        elif key_qkv:
            self.qkv_proj = Linear(
                config,
                f"{key}.{key_qkv}",
                hidden_size,
                self.fdim_qkv,
                qmap = qmap + ".input",
                out_dtype = torch.float,
                alt_key = None if not key_qkv_alt else [f"{key}.{x}" for x in key_qkv_alt],
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkvz",
            )
            self.z_proj = Linear(
                config,
                f"{key}.{key_z}",
                hidden_size,
                self.v_dim,
                qmap = qmap + ".input",
                out_dtype = torch.float,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkvz",
            )
            self.qkvz_proj = None
            self.register_submodule(self.qkv_proj)
            self.register_submodule(self.z_proj)
        else:
            self.qkvz_proj = None
            self.qkv_proj = None
            self.z_proj = None

        if key_fused_ba:
            self.ba_proj = Linear(config, f"{key}.{key_fused_ba}", hidden_size, self.fdim_ba, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.b_proj = None
            self.a_proj = None
            self.register_submodule(self.ba_proj)
        else:
            self.b_proj = None
            self.a_proj = None
            self.ba_proj = None

        if b_proj:
            self.b_proj = b_proj
            self.a_proj = a_proj
            self.ba_proj = None
            self.register_submodule(self.b_proj)
            self.register_submodule(self.a_proj)
        elif key_b:
            self.b_proj = Linear(config, f"{key}.{key_b}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.a_proj = Linear(config, f"{key}.{key_a}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.ba_proj = None
            self.register_submodule(self.b_proj)
            self.register_submodule(self.a_proj)
        else:
            self.b_proj = None
            self.a_proj = None
            self.ba_proj = None

        if o_proj:
            self.o_proj = o_proj
            self.register_submodule(self.o_proj)
        else:
            self.o_proj = Linear(
                config,
                f"{key}.{key_o}",
                self.v_head_dim * self.num_v_heads,
                hidden_size,
                qmap = qmap + ".output",
                out_dtype = self.out_dtype,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".o",
            )
            self.register_submodule(self.o_proj)

        if norm is not None:
            self.norm = norm
            self.register_submodule(self.norm)
        else:
            self.norm = GatedRMSNorm(config, f"{key}.{key_norm}", self.rms_norm_eps, out_dtype = torch.half)
            self.register_submodule(self.norm)

        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.conv1d_q_weight = None
        self.conv1d_k_weight = None
        self.conv1d_v_weight = None

        if dt_bias is not None:
            self.a_log = a_log
            self.dt_bias = dt_bias
            self.key_a_log = None
            self.key_dt_bias = None
        else:
            self.key_a_log = f"{key}.{key_a_log}"
            self.key_dt_bias = f"{key}.{key_dt_bias}"
        if conv1d_weight is not None:
            self.conv1d_weight = conv1d_weight
            self.key_conv1d_weight = None,
            self.key_conv1d_bias = None,
            self.key_conv1d_q_weight = None,
            self.key_conv1d_k_weight = None,
            self.key_conv1d_v_weight = None,
        else:
            self.key_conv1d_weight = f"{key}.{key_conv1d}.weight"
            self.key_conv1d_bias = f"{key}.{key_conv1d}.bias"
            self.key_conv1d_q_weight = f"{key}.{key_conv1d_q}.weight" if key_conv1d_q else None
            self.key_conv1d_k_weight = f"{key}.{key_conv1d_k}.weight" if key_conv1d_k else None
            self.key_conv1d_v_weight = f"{key}.{key_conv1d_v}.weight" if key_conv1d_v else None

        self.conv_dim = self.k_head_dim * self.num_k_heads

        self.caps.update({
            "recurrent_cache": True
        })

        self.bc = None
        self.bsz1_pa_args = []

        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        if self.qkvz_proj is not None:
            return [[self.qkvz_proj.optimizer_targets()]]

        targets = []
        if self.qkv_proj is not None:
            targets += self.qkv_proj.optimizer_targets()
        if self.z_proj is not None:
            targets += self.z_proj.optimizer_targets()
        return [targets]


    def load_local(self, device, **kwargs):
        is_quantized = (
            self.qkvz_proj is not None and self.qkvz_proj.quant_format_id() == "exl3" and
            self.ba_proj is not None and self.ba_proj.quant_format_id() is None and
            self.o_proj is not None and self.o_proj.quant_format_id() == "exl3"
        )

        if is_quantized:
            self.bsz1_pa_args = [
                (device, (1, self.fdim_qkv, 1), torch.bfloat16),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "a"),
                (device, (1, 1, self.num_v_heads), torch.bfloat16),
                (device, (1, 1, self.num_v_heads), torch.float),
                (device, (1, 1, self.fdim_qkvz), torch.float),
                (device, (1, 1, self.fdim_ba), torch.float),
                (device, (1, self.fdim_qkv, self.conv_kernel_size + 1), torch.bfloat16, "a"),
                (device, (1, self.fdim_qkv, 2), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads * self.v_head_dim), torch.half),
            ]

            self.bc = ext.BC_GatedDeltaNet(
                *(g_tensor_cache.get(*arg) for arg in self.bsz1_pa_args),
                self.qkvz_proj.inner.bc,
                self.ba_proj.inner.bc,
                self.dt_bias,
                self.a_log,
                self.num_k_heads,
                self.num_v_heads,
                self.k_head_dim,
                self.v_head_dim,
                self.conv1d_weight,
                self.conv1d_bias,
                self.norm.bc,
                self.o_proj.inner.bc,
                self.beta_scale
            )

    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        if self.key_a_log is not None:
            self.a_log = self.config.stc.get_tensor(self.key_a_log, self.device, optional = False, allow_bf16 = True)
            self.dt_bias = self.config.stc.get_tensor(self.key_dt_bias, self.device, optional = False, allow_bf16 = True)
        if self.key_conv1d_weight is not None:
            self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device, optional = True, allow_bf16 = True)
            self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True, allow_bf16 = True)
            if self.conv1d_weight is None:
                self.conv1d_q_weight = self.config.stc.get_tensor(self.key_conv1d_q_weight, self.device, optional = False, allow_bf16 = True)
                self.conv1d_k_weight = self.config.stc.get_tensor(self.key_conv1d_k_weight, self.device, optional = False, allow_bf16 = True)
                self.conv1d_v_weight = self.config.stc.get_tensor(self.key_conv1d_v_weight, self.device, optional = False, allow_bf16 = True)
        self.norm.load(device, **kwargs)
        self.load_local(device, **kwargs)

    @override
    def unload(self):
        if self.bc is not None:
            # for arg in self.bsz1_pa_args:
            #     g_tensor_cache.drop(*arg)
            self.bc = None
            self.bsz1_pa_args = []
        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None
        self.conv1d_q_weight = None
        self.conv1d_k_weight = None
        self.conv1d_v_weight = None
        self.norm.unload()
        super().unload()


    def split_fused_inputs(self, mixed_qkvz, mixed_ba):
        # mixed_qkvz and mixed_ba have same (bsz, seqlen)
        # both are contiguous
        bsz, seqlen, _ = mixed_qkvz.shape

        mixed_qkvz = mixed_qkvz.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.k_head_dim + 2 * self.v_head_dim * self.num_v_heads // self.num_k_heads,
        )
        mixed_ba = mixed_ba.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.num_v_heads // self.num_k_heads
        )

        split_arg_list_qkvz = [
            self.k_head_dim,
            self.k_head_dim,
            (self.num_v_groups * self.v_head_dim),
            (self.num_v_groups * self.v_head_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]
        q, k, v, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim = 3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim = 3)

        q = q.reshape(bsz, seqlen, -1)
        k = k.reshape(bsz, seqlen, -1)
        v = v.reshape(bsz, seqlen, -1)
        z = z.reshape(bsz, seqlen, -1, self.v_head_dim)
        b = b.reshape(bsz, seqlen, self.num_v_heads)
        a = a.reshape(bsz, seqlen, self.num_v_heads)
        mixed_qkv = torch.cat((q, k, v), dim = -1)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        return mixed_qkv, z, b, a


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.num_k_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(x, False)
            return to2(x, out_dtype, self.out_dtype)

        bsz, seqlen, _ = x.shape
        save_history = params.get("recurrent_history", False)

        # Post load, fuse conv1d weights if needed
        if self.conv1d_weight is None:
            self.conv1d_weight = torch.cat([
                self.conv1d_q_weight,
                self.conv1d_k_weight,
                self.conv1d_v_weight,
            ], dim = 0)
            self.conv1d_q_weight = None
            self.conv1d_k_weight = None
            self.conv1d_v_weight = None

        # Previous state
        rs = params.get("recurrent_states")
        if rs is not None:
            rs = rs[self.layer_idx, params.get("layer_instance", 0)]
            conv_state = rs.last_conv_state if rs.last_conv_state is not None else \
                torch.zeros((bsz, self.fdim_qkv, self.conv_kernel_size), dtype = torch.bfloat16, device = x.device)
            recurrent_state = rs.last_recurrent_state if rs.last_recurrent_state is not None else \
                torch.zeros(
                    (bsz, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                    dtype = torch.float,
                    device = self.device
                )

            save_state = True
        else:
            conv_state = None
            recurrent_state = None
            save_state = False
            save_history = False  # no SD without prior state, for simplicity

        # C++ path
        if self.bc is not None and bsz == 1 and seqlen == 1 and save_state and not save_history:
            y = torch.empty_like(x)
            mixed_qkv = self.bc.run_bsz1_a(x)
            mixed_qkv = causal_conv1d_update_function(
                mixed_qkv,
                conv_state,  # Updated inplace
                self.conv1d_weight.squeeze(1),
                self.conv1d_bias,
            )
            self.bc.run_bsz1_b(mixed_qkv, y, recurrent_state)
            x = y

        # Torch path
        else:
            # Projections
            #
            # NOTE:
            # Qwen3.5 uses split projections (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a),
            # while Qwen3-Next uses fused projections. The fused C++ helper expects the
            # packed layout used by fused projections; applying it to split qkv tensors
            # causes incorrect head ordering and broken generations.
            if self.qkvz_proj is not None and self.ba_proj is not None:
                qkvz = self.qkvz_proj.forward(x, params)
                ba = self.ba_proj.forward(x, params)

                mixed_qkv = torch.empty((bsz, self.fdim_qkv, seqlen), dtype = torch.bfloat16, device = self.device)
                z = torch.empty((bsz, seqlen, self.num_v_heads, self.v_head_dim), dtype = torch.bfloat16, device = self.device)
                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

                ext.gated_delta_net_fused_op(
                    qkvz, ba,
                    self.dt_bias,
                    self.a_log,
                    mixed_qkv, z, beta, g,
                    self.num_k_heads,
                    self.num_v_heads,
                    self.k_head_dim,
                    self.v_head_dim,
                    self.beta_scale
                )
            else:
                # TODO: Bound class and/or graph for this part
                qkv = self.qkv_proj.forward(x, params)
                z = self.z_proj.forward(x, params).view(bsz, seqlen, self.num_v_heads, self.v_head_dim)
                b = self.b_proj.forward(x, params)
                a = self.a_proj.forward(x, params)

                mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()

                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

                ext.gated_delta_net_fused_op_2(
                    b, a,
                    self.dt_bias,
                    self.a_log,
                    beta, g,
                    self.beta_scale
                )

            # Convolution
            # TODO: Figure out an alternative or write a new kernel that won't require transposing qkv back and forth
            if conv_state is None:
                if save_state:
                    conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                    rs.last_conv_state = conv_state
                mixed_qkv = causal_conv1d_fwd_function(
                    mixed_qkv,
                    self.conv1d_weight.squeeze(1),
                    self.conv1d_bias,
                )
            else:
                if save_history:
                    mixed_qkv, conv_state_history = causal_conv1d_update_with_history(
                        mixed_qkv,
                        conv_state,  # Updated inplace
                        self.conv1d_weight.squeeze(1),
                        self.conv1d_bias,
                    )
                else:
                    mixed_qkv = causal_conv1d_update_function(
                        mixed_qkv,
                        conv_state,  # Updated inplace
                        self.conv1d_weight.squeeze(1),
                        self.conv1d_bias,
                    )

            # Use chunked rule when advantageous and available
            # TODO: Replace chunked fn with non-Triton implementation
            if seqlen >= self.num_v_heads and chunk_gated_delta_rule is not None:
                mixed_qkv = mixed_qkv.transpose(1, 2)

                q, k, v = torch.split(mixed_qkv, [self.k_dim, self.k_dim, self.v_dim], dim = -1)
                q = q.view(bsz, seqlen, -1, self.k_head_dim)
                k = k.view(bsz, seqlen, -1, self.k_head_dim)
                v = v.view(bsz, seqlen, -1, self.v_head_dim)

                # Grouped attn
                if self.num_v_heads // self.num_k_heads > 1:
                    q = q.repeat_interleave(self.num_v_groups, dim = 2)
                    k = k.repeat_interleave(self.num_v_groups, dim = 2)

                core_attn_out, recurrent_state = chunk_gated_delta_rule(
                    q, k, v,
                    g = g,
                    beta = beta,
                    initial_state = recurrent_state,
                    output_final_state = save_state,
                    use_qk_l2norm_in_kernel = True,
                )

            else:
                core_attn_out = torch.empty(
                    (bsz, seqlen, self.num_v_heads, self.v_head_dim),
                    dtype = torch.bfloat16,
                    device = self.device,
                )

                mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()
                if recurrent_state is None:
                    recurrent_state = torch.zeros(
                        (bsz, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                        dtype = torch.float,
                        device = self.device
                    )

                history = torch.empty(
                    (bsz, seqlen - 1, self.num_v_heads, self.k_head_dim, self.v_head_dim),
                    dtype = torch.float,
                    device = self.device
                ) if save_history else None

                ext.cuda_recurrent_gated_delta_rule(
                    mixed_qkv,
                    g,
                    beta,
                    recurrent_state,
                    core_attn_out,
                    self.num_k_heads,
                    self.num_v_heads,
                    self.k_head_dim,
                    self.v_head_dim,
                    history,
                )

            # Norm
            core_attn_out = self.norm.forward(core_attn_out, params, gate = z)
            core_attn_out = core_attn_out.view(bsz, seqlen, self.num_v_heads * self.v_head_dim)

            # Output projection
            x = self.o_proj.forward(core_attn_out, params)

        # Update cache
        if save_state:
            rs.last_recurrent_state = recurrent_state
            rs.last_conv_state = conv_state
            if not rs.batched:
                rs.position += seqlen
                if save_history:
                    rs.history = history
                    rs.conv_history = conv_state_history
            else:
                rs.positions = [r + seqlen for r in rs.positions]
                if save_history:
                    rs.history = history
                    rs.conv_history = conv_state_history

        # TP reduction
        if self.tp_reduce:
            params["backend"].all_reduce(x)

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
        assert self.qkv_proj is not None
        assert self.z_proj is not None
        assert self.b_proj is not None
        assert self.a_proj is not None
        storage = 0
        storage += self.qkv_proj.storage_size()
        storage += self.z_proj.storage_size()
        storage += self.b_proj.storage_size()
        storage += self.a_proj.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        overhead_s += 2 * self.num_k_heads * self.k_head_dim * torch.half.itemsize
        overhead_s += 2 * self.num_v_heads * self.v_head_dim * torch.half.itemsize
        recons = max(
            self.qkv_proj.recons_size(),
            self.z_proj.recons_size(),
        )
        channel_width = 1
        channels_to_split = self.num_k_heads
        assert self.num_v_heads % self.num_k_heads == 0, \
            "num_k_heads doesn't divide num_v_heads"
        while channel_width * self.k_head_dim < 128:
            assert channels_to_split % 2 == 0, \
                "Model's K/V heads cannot divide into 128-channel tensors"
            channel_width *= 2
            channels_to_split //= 2
        assert (channel_width * self.k_head_dim) % 128 == 0 and (channel_width * self.v_head_dim) % 128 == 0, \
            "Model's K/V heads cannot divide into 128-channel tensors"
        tpa = TPAllocation(
            key = self.key,
            channel_width = channel_width,
            channel_unit = "K-heads",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = channels_to_split,
            limit_key = "attn"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            if child is None:
                return None
            if isinstance(child, torch.Tensor):
                return TPTensorWrapper.tp_export(child, plan, producer)
            else:
                return child.tp_export(plan, producer)

        return {
            "cls": GatedDeltaNet,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "k_head_dim": self.k_head_dim,
                "v_head_dim": self.k_head_dim,
                "rms_norm_eps": self.rms_norm_eps,
                "conv_kernel_size": self.conv_kernel_size,
                "beta_scale": self.beta_scale,
                "out_dtype": self.out_dtype,
            },
            "num_k_heads": self.num_k_heads,
            "num_v_heads": self.num_v_heads,
            "num_kv_group": self.num_v_heads // self.num_k_heads,
            **{name: _export(getattr(self, name, None)) for name in (
                "qkv_proj",
                "z_proj",
                "b_proj",
                "a_proj",
                "o_proj",
                "norm",
                "conv1d_weight",
                "a_log",
                "dt_bias",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        k_head_dim = exported["kwargs"]["k_head_dim"]
        v_head_dim = exported["kwargs"]["v_head_dim"]
        G = exported["num_kv_group"]
        global_num_k_heads = exported["num_k_heads"]
        global_num_v_heads = exported["num_v_heads"]
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "K-heads"
        num_k_heads = last - first
        num_v_heads = (last - first) * G

        q_split = (True, first * k_head_dim, last * k_head_dim) \
            if num_k_heads else None
        k_split = (True, (global_num_k_heads + first) * k_head_dim, (global_num_k_heads + last) * k_head_dim) \
            if num_k_heads else None
        v_split = (True, (global_num_k_heads * 2 + first * G) * v_head_dim, (global_num_k_heads * 2 + last * G) * v_head_dim) \
            if num_k_heads else None
        z_split = (True, first * v_head_dim * G, last * v_head_dim * G) \
            if num_k_heads else None
        o_split = (False, first * v_head_dim * G, last * v_head_dim * G) \
            if num_k_heads else None
        a_split = (True, first * G, last * G) \
            if num_k_heads else None
        b_split = (True, first * G, last * G) \
            if num_k_heads else None

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        def _import_split_3(name, split_0, split_1, split_2):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split_3(local_context, exported[name], plan, split_0, split_1, split_2) \
                if split_0 and exported.get(name) else None

        module = GatedDeltaNet(
            config = None,
            **exported["kwargs"],
            num_k_heads = num_k_heads,
            num_v_heads = num_v_heads,
            conv1d_weight = _import_split_3("conv1d_weight", q_split, k_split, v_split),
            qkv_proj = _import_split_3("qkv_proj", q_split, k_split, v_split),
            z_proj = _import_split("z_proj", z_split),
            o_proj = _import_split("o_proj", o_split),
            b_proj = _import_split("b_proj", b_split),
            a_proj = _import_split("a_proj", a_split),
            norm = _import("norm"),
            a_log = _import_split("a_log", a_split),
            dt_bias = _import_split("dt_bias", a_split),
        )

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True

        module.load_local(device)
        torch.cuda.synchronize()
        return module
