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
from ..cache import Cache
from ..util.tensor import g_tensor_cache
from ..model.model_tp_shared import TPTensorWrapper
from .gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn
from ..cache.recurrent import (
    mp_cache_recurrent_stash,
    mp_cache_recurrent_unstash,
    mp_cache_recurrent_clear
)
from ..util import profile_opt

next_checkpoint_handle = 0

def mp_cache_recurrent_rewind(local_context: dict, cache_id: int, slot: int, last_history, num_tokens):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        l.rewind(slot, last_history, num_tokens)


class GDNState:

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict = None,
        test_state: bool = False,
        exported: bool = False,
    ):
        self.slot = slot
        self.position = position
        self.cache = cache
        self.last_history = 0
        self.exported = exported

        if not exported:
            assert test_state or position == 0 or stashed is not None, \
                "State must be new, restored from checkpoint or marked as a test state."

            if clear and stashed is None:
                if not self.cache.model.loaded_tp:
                    for l in self.cache.get_all_recurrent_layers().values():
                        l.clear(slot)
                else:
                    self.cache.model.tp_dispatch_all(mp_cache_recurrent_clear, (id(self.cache), self.slot))

            if stashed is not None:
                self.unstash(stashed)

            self.checkpoint_size = sum(
                l.get_checkpoint_size()
                for l in self.cache.get_all_recurrent_layers().values()
            )


    def free(self):
        self.cache.release_state(self)


    def rewind(self, num_tokens: int):
        if not self.cache.model.loaded_tp:
            for l in self.cache.get_all_recurrent_layers().values():
                l.rewind(self.slot, self.last_history, num_tokens)
        else:
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_rewind, (id(self.cache), self.slot, self.last_history, num_tokens))
        self.position -= num_tokens
        self.last_history = 0


    def stash(self):
        global next_checkpoint_handle
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size
        }
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                stashed[k] = l.stash(self.slot)
        else:
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_stash, (id(self.cache), next_checkpoint_handle, self.slot))
            stashed["tp_handle"] = next_checkpoint_handle
            next_checkpoint_handle += 1
        return stashed


    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                l.unstash(self.slot, stashed[k])
        else:
            cp_handle = stashed["tp_handle"]
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_unstash, (id(self.cache), cp_handle, self.slot))


    def post_advance(self):
        pass


    def tp_export(self):
        return GDNState(
            cache = id(self.cache),
            slot = self.slot,
            position = self.position,
            exported = True,
        )


    def reset(self):
        self.position = 0


class GDNLayerState:

    def __init__(
        self,
        module: GatedDeltaNet,
        max_batch_size: int,
        max_history: int,
        cache_id: int,
    ):
        self.module = module
        self.conv_state = torch.empty(
            (max_batch_size, module.fdim_qkv, module.conv_kernel_size + max_history),
            dtype = torch.bfloat16,
            device = "meta"
        )
        self.recurrent_state = torch.empty(
            (max_batch_size, max_history + 1, module.num_v_heads, module.k_head_dim, module.v_head_dim),
            dtype = torch.float,
            device = "meta"
        )
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id


    def get_checkpoint_size(self):
        return (
            self.module.fdim_qkv * self.module.conv_kernel_size * 2 +
            self.module.num_v_heads * self.module.k_head_dim * self.module.v_head_dim * 4
        )


    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in [self.conv_state, self.recurrent_state])


    def alloc(self, device):
        self.conv_state = torch.empty_like(self.conv_state, device = device)
        self.recurrent_state = torch.empty_like(self.recurrent_state, device = device)
        self.conv_state.zero_()
        self.recurrent_state.zero_()
        self.device = device


    def free(self):
        self.conv_state = torch.empty_like(self.conv_state, device = "meta")
        self.recurrent_state = torch.empty_like(self.recurrent_state, device = "meta")
        self.device = None


    def clear(self, idx: int):
        if self.device is not None:
            self.conv_state[idx].zero_()
            self.recurrent_state[idx].zero_()


    def get_state_tensors(self):
        return (
            self.conv_state,
            self.recurrent_state,
        )


    def rewind(self, slot: int, last_history: int, num_tokens: int):
        assert num_tokens <= last_history
        if num_tokens > 0:
            r_state = self.recurrent_state[slot, 0]
            r_state_rewind = self.recurrent_state[slot, last_history + 1 - num_tokens]
            r_state.copy_(r_state_rewind)
        cdim = self.module.conv_kernel_size
        if last_history > 0:
            c_state = self.conv_state[slot, :, :cdim]
            p = self.conv_state.shape[-1] - num_tokens
            c_state_rewind = self.conv_state[slot, :, p - cdim : p]
            temp = c_state_rewind.clone()
            c_state.copy_(temp)


    def stash(self, slot):
        cdim = self.module.conv_kernel_size
        return (
            self.recurrent_state[slot, :1].cpu(),
            self.conv_state[slot, :, :cdim].cpu()
        )


    def unstash(self, slot, stashed):
        cdim = self.module.conv_kernel_size
        s, c = stashed
        self.recurrent_state[slot, :1].copy_(s)
        self.conv_state[slot, :, :cdim].copy_(c)


    def tp_export(self, plan):
        return {
            "cls": GDNLayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            }
        }


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
        conv1d_bias: torch.Tensor | None = None,
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
        self.num_v_groups = num_v_heads // num_k_heads if num_k_heads else 0
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads
        self.beta_scale = beta_scale

        self.out_dtype = out_dtype

        self.fdim_qkvz = 2 * self.num_k_heads * self.k_head_dim + 2 * self.num_v_heads * self.v_head_dim
        self.fdim_ba = 2 * self.num_v_heads
        self.fdim_qkv = 2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim

        if self.num_k_heads == 0:
            return

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
        elif qkv_proj:
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
        elif b_proj:
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
            self.conv1d_bias = conv1d_bias
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
        self.layer_state_cls = GDNLayerState

        self.bc = None
        self.bsz1_pa_args = []

        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False
        self.has_split_cache = False


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

        if self.num_k_heads == 0:
            return

        # Recurrent states
        for rl in self.recurrent_layers:
            rl.alloc(device)

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
        for cl in self.recurrent_layers:
            cl.free()
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
        rsg = params.get("recurrent_states")
        if rsg:
            recurrent_slots = get_for_device(params, "recurrent_slots", self.device)
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            if rsg[0].exported:
                rsl = self.tp_recurrent_lookup[rsg[0].cache]
            else:
                rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            conv_state, recurrent_state = rsl.get_state_tensors()
            save_state = True
        else:
            recurrent_slots = None
            conv_state, recurrent_state = None, None
            save_state = False
            save_history = False  # no SD without prior state, for simplicity

        # C++ path (currently disabled pending testing)
        # if self.bc is not None and bsz == 1 and seqlen == 1 and save_state and not save_history:
        #     y = torch.empty_like(x)
        #     mixed_qkv = self.bc.run_bsz1_a(x)
        #     mixed_qkv = causal_conv1d_update_function(
        #         mixed_qkv,
        #         conv_state,  # Updated inplace
        #         self.conv1d_weight.squeeze(1),
        #         self.conv1d_bias,
        #     )
        #     self.bc.run_bsz1_b(mixed_qkv, y, recurrent_state)
        #     x = y

        # Torch path
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
        mixed_qkv = causal_conv1d_update(
            mixed_qkv = mixed_qkv,
            conv_state = conv_state,
            recurrent_slots = recurrent_slots,
            conv1d_weight = self.conv1d_weight.squeeze(1).contiguous(),
            conv1d_bias = self.conv1d_bias,
            history = save_history,
            params = params,
        )

        # Delta rule
        core_attn_out = gated_delta_rule_fn(
            mixed_qkv = mixed_qkv,
            beta = beta,
            g = g,
            recurrent_state = recurrent_state,
            recurrent_slots = recurrent_slots,
            history = save_history,
            save_state = save_state,
            num_k_heads = self.num_k_heads,
            num_v_heads = self.num_v_heads,
            k_dim = self.k_dim,
            v_dim = self.v_dim,
            k_head_dim = self.k_head_dim,
            v_head_dim = self.v_head_dim,
            params = params,
        )

        # Norm
        core_attn_out = self.norm.forward(core_attn_out, params, gate = z)
        core_attn_out = core_attn_out.view(bsz, seqlen, self.num_v_heads * self.v_head_dim)

        # Output projection
        x = self.o_proj.forward(core_attn_out, params)

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
        for cl in self.recurrent_layers:
            storage += cl.storage_size()
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
            limit_key = "linear_attn"
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
                "v_head_dim": self.v_head_dim,
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
                "conv1d_bias",
                "a_log",
                "dt_bias",
            )},
            "device": self.device,
            "recurrent_layers": [
                rl.tp_export(plan) for rl in self.recurrent_layers
            ]
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
            conv1d_bias = _import_split_3("conv1d_bias", q_split, k_split, v_split),
            qkv_proj = _import_split_3("qkv_proj", q_split, k_split, v_split),
            z_proj = _import_split("z_proj", z_split),
            o_proj = _import_split("o_proj", o_split),
            b_proj = _import_split("b_proj", b_split),
            a_proj = _import_split("a_proj", a_split),
            norm = _import("norm"),
            a_log = _import_split("a_log", a_split),
            dt_bias = _import_split("dt_bias", a_split),
        )

        if num_k_heads:
            recurrent_layers = exported["recurrent_layers"]
            if len(recurrent_layers):
                module.has_split_cache = True
                for rl in exported["recurrent_layers"]:
                    rli = rl["cls"](module, **rl["args"])
                    module.recurrent_layers.append(rli)
                    module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True

        module.load_local(device)
        torch.cuda.synchronize()
        return module
