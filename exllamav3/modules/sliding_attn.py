from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2, g_tensor_cache
from . import Module, Linear, RMSNorm, LayerNorm
from ..constants import PAGE_SIZE
from .attention_fn.triton_paged import paged_attn_triton_decode, paged_attn_triton_prefill
from .attention_fn.bc_attn import bc_attn_enable as _bc_attn_enable, build_bc_swa, MAX_BSZ as _bc_max_bsz, MAX_QLEN as _bc_max_qlen
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from ..cache import Cache
from ..cache.recurrent import (
    mp_cache_recurrent_stash,
    mp_cache_recurrent_unstash,
    new_checkpoint_handle,
)
from ..model.model_tp_alloc import TPAllocation
from ..util import profile_opt


class SWAExportedState:
    """Lightweight stand-in for an SWAState in TP worker processes: carries the fields the module
    reads during a forward plus the wshift it writes, which the parent reads back after the pass
    (the pseudo worker for the output device shares the parent's address space)"""

    exported = True

    def __init__(self, cache: int, slot: int, position: int, window_beg: int):
        self.cache = cache
        self.slot = slot
        self.position = position
        self.window_beg = window_beg
        self.wshift = 0

class SWAState:

    exported = False

    # The state ring stores at least one page of history before the attention window at all times (kv_state_size
    # rounds window + overprocessing up to whole pages), so in-place rollback of at least this many tokens is
    # always possible
    guaranteed_rollback = PAGE_SIZE

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict = None,
        test_state: bool = False,
    ):
        assert test_state or position == 0 or stashed is not None, \
            "State must be new, restored from checkpoint or marked as a test state."

        self.slot = slot
        self.position = position
        self.cache = cache
        self.last_history = 0
        self.window_beg = position // PAGE_SIZE * PAGE_SIZE
        self.wshift = 0

        if stashed is not None:
            self.unstash(stashed)

        self.checkpoint_size = sum(
            l.get_checkpoint_size()
            for l in self.cache.get_all_recurrent_layers().values()
        )


    def free(self):
        self.cache.release_state(self)


    def rewind(self, num_tokens: int):
        assert num_tokens <= self.position - self.window_beg
        self.position -= num_tokens
        self.last_history = 0


    def rollback_capacity(self):
        """
        Number of tokens the state can rewind in place. K/V for all positions from window_beg are still stored in
        the ring, so rewinding is just moving the position back; overwritten future slots don't matter.
        """
        return self.position - self.window_beg


    def stash(self):
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
            "window_beg": self.window_beg,
        }
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                stashed[k] = l.stash(self.slot, self.position)
        else:
            cp_handle = new_checkpoint_handle()
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_stash, (id(self.cache), cp_handle, self.slot, self.position))
            stashed["tp_handle"] = cp_handle
        return stashed


    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        self.window_beg = stashed["window_beg"]
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                l.unstash(self.slot, stashed[k], self.position)
        else:
            cp_handle = stashed["tp_handle"]
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_unstash, (id(self.cache), cp_handle, self.slot, self.position))


    def post_advance(self):
        self.window_beg += self.wshift


    def tp_export(self):
        return SWAExportedState(
            cache = id(self.cache),
            slot = self.slot,
            position = self.position,
            window_beg = self.window_beg,
        )


    def tp_readback(self, exported: SWAExportedState):
        # The forward pass decides the per-step window shift; under TP it runs on the exported
        # handle (in the parent's address space via the output device's pseudo worker), and
        # post_advance() applies the shift parent-side afterwards
        self.wshift = exported.wshift


    def reset(self):
        self.position = 0
        self.window_beg = 0
        self.wshift = 0

class SWALayerState:

    def __init__(
        self,
        module: SlidingAttention,
        max_batch_size: int,
        max_history: int,
        cache_id: int,
    ):
        assert module.kv_state_size % 256 == 0
        self.module = module
        self.k_state = torch.empty(
            (max_batch_size, module.kv_state_size, module.num_kv_heads, module.head_dim),
            dtype = torch.half,
            device = "meta"
        )
        self.v_state = torch.empty(
            (max_batch_size, module.kv_state_size, module.num_kv_heads, module.head_dim),
            dtype = torch.half,
            device = "meta"
        )
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id


    def get_checkpoint_size(self):
        return (
            self.module.sliding_window * self.module.num_kv_heads * self.module.head_dim * 2 +
            self.module.sliding_window * self.module.num_kv_heads * self.module.head_dim * 2
        )


    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in (self.k_state, self.v_state))


    def tp_export(self, plan):
        return {
            "cls": SWALayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            }
        }


    def alloc(self, device):
        self.k_state = torch.empty_like(self.k_state, device = device)
        self.v_state = torch.empty_like(self.v_state, device = device)
        self.device = device


    def free(self):
        self.k_state = torch.empty_like(self.k_state, device = "meta")
        self.v_state = torch.empty_like(self.v_state, device = "meta")
        self.device = None


    def clear(self, idx: int):
        pass


    def get_state_tensors(self):
        return (
            self.k_state,
            self.v_state,
        )


    def rewind(self, slot: int, last_history: int, num_tokens: int):
        pass


    def stash(self, slot, position):
        b = min(self.module.kv_state_size, position)
        a = max(0, b - self.module.sliding_window)
        return (
            self.k_state[slot, a:b].cpu(),
            self.v_state[slot, a:b].cpu()
        )


    def unstash(self, slot, stashed, position):
        b = min(self.module.kv_state_size, position)
        a = max(0, b - self.module.sliding_window)
        k, v = stashed
        self.k_state[slot, a:b].copy_(k)
        self.v_state[slot, a:b].copy_(v)


class SlidingAttention(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        rope_settings: RopeSettings | None,
        sm_scale: float | None = None,
        key_q: str | None = None,
        key_k: str | None = None,
        key_v: str | None = None,
        key_o: str | None = None,
        key_g: str | None = None,
        key_sinks: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        sliding_window: int = -1,
        sliding_window_overp: int = 2 * PAGE_SIZE,
        logit_softcapping: float = 0.0,
        q_norm: RMSNorm | LayerNorm | None = None,
        k_norm: RMSNorm | LayerNorm | None = None,
        v_norm: RMSNorm | LayerNorm | None = None,
        q_proj: Linear | Module | None = None,
        k_proj: Linear | Module | None = None,
        v_proj: Linear | Module | None = None,
        o_proj: Linear | Module | None = None,
        g_proj: Linear | Module | None = None,
        post_rope_norm: bool = False,
        full_gate: bool = False,
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)
        assert sliding_window > 0

        self.q_priority = 2 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.gqa = (num_q_heads != num_kv_heads)
        self.sm_scale = sm_scale or self.head_dim ** (-0.5)
        self.rope_settings = rope_settings
        self.rope = None
        self.out_dtype = out_dtype
        self.sliding_window = sliding_window
        self.sliding_window_overp = sliding_window_overp
        # The state buffer holds the window plus overprovisioned history so speculative decoding can
        # roll back rejected drafts without recompute; the shift logic drops up to a page at a time,
        # so round UP to whole pages to preserve at least the requested slack after a shift
        self.kv_state_size = -(-(sliding_window + sliding_window_overp) // PAGE_SIZE) * PAGE_SIZE
        self.logit_softcapping = logit_softcapping
        self.post_rope_norm = post_rope_norm
        self.full_gate = full_gate
        self.bt_cache = {}

        # Set before the zero-heads early return: forward()/unload() and the TP import touch these
        # on ranks that hold none of this layer's heads
        self.multi_kv = None
        self.multi_qg = None
        self.q_norm_tensor = None
        self.k_norm_tensor = None
        self.has_split_cache = False
        self.prealloc_qgh_1 = None
        self.prealloc_qg_1 = None
        self.prealloc_kvh_1 = None
        self.prealloc_kv_1 = None
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False
        self.bc_attn = {}
        self.headwise_gate = False
        self.g_proj = None
        self.key_sinks = key_sinks
        self.sinks = None

        if post_rope_norm:
            assert q_norm is None and k_norm is None, \
                "Post-RoPE norm only supported without weights"

        if self.num_kv_heads == 0:
            return

        # Create q, k, v projections
        fkey, frange_q, frange_k, frange_v = None, None, None, None

        if key_q or frange_q:
            f = 1
            self.q_proj = Linear(
                config,
                f"{key}.{key_q}",
                hidden_size,
                num_q_heads * head_dim * f,
                qmap = qmap + ".input" if qmap is not None else None,
                fkey = fkey,
                frange = frange_q,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkv",
            )
            self.register_submodule(self.q_proj)
        else:
            assert q_proj
            self.q_proj = q_proj
            self.register_submodule(self.q_proj)

        if key_k or frange_k:
            self.k_proj = Linear(
                config,
                f"{key}.{key_k}",
                hidden_size,
                num_kv_heads * head_dim,
                qmap =  qmap + ".input" if qmap is not None else None,
                fkey = fkey,
                frange = frange_k,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkv",
            )
            self.v_proj = Linear(
                config,
                f"{key}.{key_v}",
                hidden_size,
                num_kv_heads * head_dim,
                qmap =  qmap + ".input" if qmap is not None else None,
                fkey = fkey,
                frange = frange_v,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkv",
            )
            self.register_submodule(self.k_proj)
            self.register_submodule(self.v_proj)
        else:
            assert k_proj and v_proj
            self.k_proj = k_proj
            self.v_proj = v_proj
            self.register_submodule(self.k_proj)
            self.register_submodule(self.v_proj)

        # Create o proj
        if key_o:
            self.o_proj = Linear(
                config,
                f"{key}.{key_o}",
                num_q_heads * head_dim,
                hidden_size,
                qmap =  qmap + ".o" if qmap is not None else None,
                out_dtype = out_dtype,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".o" if qmap is not None else None,
                trim_padded_out = True,
            )
            self.register_submodule(self.o_proj)
        else:
            assert o_proj
            self.o_proj = o_proj
            self.register_submodule(self.o_proj)

        # Register q/k norms
        if q_norm:
            assert k_norm, "Must have both Q and K norms, or neither"
            self.q_norm = q_norm
            self.k_norm = k_norm
            self.register_submodule(self.q_norm)
            self.register_submodule(self.k_norm)
            if isinstance(q_norm, RMSNorm):
                self.norm_eps = q_norm.rms_norm_eps
                self.norm_constant_bias = q_norm.constant_bias
                assert self.norm_eps == k_norm.rms_norm_eps
            else:
                self.norm_eps = q_norm.layernorm_eps
                self.norm_constant_bias = 0.0
        else:
            self.q_norm = None
            self.k_norm = None
            self.norm_eps = 1e-6
            self.norm_constant_bias = 0.0

        # Register v norm
        if v_norm:
            self.v_norm = v_norm
            self.register_submodule(self.v_norm)
        else:
            self.v_norm = None

        # Register headwise gate
        if key_g:
            gate_features = num_q_heads * head_dim if full_gate else num_q_heads
            _qmap = ".input" if full_gate else None
            self.g_proj = Linear(
                config,
                f"{key}.{key_g}",
                hidden_size,
                gate_features,
                qmap = _qmap,
                out_dtype = torch.half,
                pad_to = 1,
                select_hq_bits = select_hq_bits,
            )
            self.headwise_gate = not full_gate
            self.register_submodule(self.g_proj)
        else:
            if g_proj:
                self.g_proj = g_proj
                self.headwise_gate = not full_gate
                self.register_submodule(self.g_proj)
            else:
                self.g_proj = None
                self.headwise_gate = False

        self.caps.update({
            "recurrent_cache": True,
            "sliding_window_overp": sliding_window_overp
        })
        self.layer_state_cls = SWALayerState


    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        v = self.v_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k + v, o]]


    def load_local(self, device, **kwargs):

        if self.num_kv_heads == 0:
            return

        # Recurrent states
        for rl in self.recurrent_layers:
            rl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(
                device,
                self.rope_settings,
            )

        if self.key_sinks:
            self.sinks = self.config.stc.get_tensor(
                f"{self.key}.{self.key_sinks}", device, no_defer = True
            ).float().contiguous()

        # Test if K and V proj can be fused
        if (
            not self.config.infer_params.no_reconstruct and
            device != torch.device("cpu") and
            self.k_proj.quant_type == "exl3" and
            self.v_proj.quant_type == "exl3" and
            self.k_proj.out_features == self.v_proj.out_features and
            self.k_proj.inner.K == self.v_proj.inner.K and
            self.k_proj.inner.bias is None and
            self.v_proj.inner.bias is None and
            self.config.infer_params.use_mgemm(
                self.k_proj.inner.K, self.k_proj.out_features,
                self.k_proj.inner.mul1 and self.v_proj.inner.mul1,
                device,
            )
        ):
            self.multi_kv = MultiLinear(self. device, [self.k_proj, self.v_proj])
            self.prealloc_kvh_1 = g_tensor_cache.get(device, (2, 1, self.hidden_size), torch.half, "kvh_1")
            self.prealloc_kv_1 = g_tensor_cache.get(device, (2, 1, self.num_kv_heads * self.head_dim), torch.half, "kv_1")

        # Test if Q and G proj can be fused
        if (
            not self.config.infer_params.no_reconstruct and
            self.g_proj is not None and
            device != torch.device("cpu") and
            self.q_proj.quant_type == "exl3" and
            self.g_proj.quant_type == "exl3" and
            self.q_proj.out_features == self.g_proj.out_features and
            self.q_proj.inner.K == self.g_proj.inner.K and
            self.q_proj.inner.bias is None and
            self.g_proj.inner.bias is None and
            self.config.infer_params.use_mgemm(
                self.q_proj.inner.K, self.q_proj.out_features,
                self.q_proj.inner.mul1 and self.g_proj.inner.mul1,
                device,
            )
        ):
            self.multi_qg = MultiLinear(self. device, [self.q_proj, self.g_proj])
            self.prealloc_qgh_1 = g_tensor_cache.get(device, (2, 1, self.hidden_size), torch.half, "qgh_1")
            self.prealloc_qg_1 = g_tensor_cache.get(device, (2, 1, self.num_q_heads * self.head_dim), torch.half, "qg_1")

        # Head norm
        if self.q_norm and isinstance(self.q_norm, RMSNorm) and not self.q_norm.span_heads:
            self.q_norm_tensor = self.q_norm.weight.data
            self.k_norm_tensor = self.k_norm.weight.data



    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        super().unload()

        self.bc_attn = {}
        self.rope = None
        self.sinks = None

        if self.multi_kv is not None:
            self.multi_kv.unload()
            self.multi_kv = None

        if self.multi_qg is not None:
            self.multi_qg.unload()
            self.multi_qg = None

        self.q_norm_tensor = None
        self.k_norm_tensor = None

        self.prealloc_qgh_1 = None
        self.prealloc_qg_1 = None
        self.prealloc_kvh_1 = None
        self.prealloc_kv_1 = None
        self.bt_cache = {}


    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.sinks is not None:
            # bf16 -> fp16 is exact at sink-logit magnitudes; stored as loaded
            t[f"{self.key}.{self.key_sinks}"] = self.sinks.half().contiguous()
        return t


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.num_kv_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(x, False)
        else:
            bsz, seqlen, _ = x.shape
            attn_mode = params.get("attn_mode", "flash_attn_nc")
            match attn_mode:
                case "flash_attn":
                    x = self.decode_flash_attn(x, bsz, seqlen, params)
                case "flash_attn_nc":
                    x = self.decode_flash_attn_nc(x, bsz, seqlen, params)
                case _:
                    raise ValueError(f"Unknown attn_mode: {attn_mode}")
            if self.tp_reduce:
                params["backend"].all_reduce(x)

        return to2(x, out_dtype, self.out_dtype)


    def project_qkv(self, x: torch.Tensor, params: dict) -> tuple:
        bsz, q_len, dim = x.shape

        if self.multi_qg is None or bsz * q_len > 32:
            q = self.q_proj.forward(x, params)
            if self.g_proj:
                g = self.g_proj.forward(x, params)
            else:
                g = None
        else:
            x = x.view(1, bsz * q_len, dim)
            if bsz * q_len == 1:
                qgh = self.prealloc_qgh_1
                qg = self.prealloc_qg_1
            else:
                qgh = torch.empty((2, bsz * q_len, dim), dtype = torch.half, device = x.device)
                qg = torch.empty((2, bsz * q_len, self.num_q_heads * self.head_dim), dtype = torch.half, device = x.device)
            ext.exl3_mgemm(
                x,
                self.multi_qg.ptrs_trellis,
                qg,
                self.multi_qg.ptrs_suh,
                qgh,
                self.multi_qg.ptrs_svh,
                None,
                None,
                self.multi_qg.K,
                -1,
                self.multi_qg.mcg,
                self.multi_qg.mul1,
                -1,
                -1,
                0,
                1
            )
            q = qg[0].view(bsz, q_len, self.num_q_heads * self.head_dim)
            g = qg[1].view(bsz, q_len, self.num_q_heads * self.head_dim)

        if self.multi_kv is None or bsz * q_len > 32:
            k = self.k_proj.forward(x, params)
            v = self.v_proj.forward(x, params)

        else:
            x = x.view(1, bsz * q_len, dim)
            if bsz * q_len == 1:
                kvh = self.prealloc_kvh_1
                kv = self.prealloc_kv_1
            else:
                kvh = torch.empty((2, bsz * q_len, dim), dtype = torch.half, device = x.device)
                kv = torch.empty((2, bsz * q_len, self.num_kv_heads * self.head_dim), dtype = torch.half, device = x.device)
            ext.exl3_mgemm(
                x,
                self.multi_kv.ptrs_trellis,
                kv,
                self.multi_kv.ptrs_suh,
                kvh,
                self.multi_kv.ptrs_svh,
                None,
                None,
                self.multi_kv.K,
                -1,
                self.multi_kv.mcg,
                self.multi_kv.mul1,
                -1,
                -1,
                0,
                1
            )
            k = kv[0].view(bsz, q_len, self.num_kv_heads * self.head_dim)
            v = kv[1].view(bsz, q_len, self.num_kv_heads * self.head_dim)

        q = q.view(bsz, q_len, self.num_q_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        if self.v_norm is not None:
            v = self.v_norm.forward(v, params, out_dtype = torch.half)

        return q, k, v, g


    def project_o(self, o: torch.Tensor, bsz: int, seqlen: int, params: dict) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.num_q_heads * self.head_dim)
        x = self.o_proj.forward(o, params)
        return x


    def _decode_state_prep(self, rsg, k_states, v_states, seqlen):
        """Per-row page shift when the window run reaches the end of the state buffer, cached
        per-slot block table, and the in-state sequence lengths for a decode step."""
        S = self.kv_state_size
        pps = S // PAGE_SIZE
        positions_l = []
        for rs in rsg:
            cache_pos = rs.position - rs.window_beg
            if cache_pos + seqlen > S:
                shift = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
                k_states[rs.slot, :-shift].copy_(k_states[rs.slot, shift:].clone())
                v_states[rs.slot, :-shift].copy_(v_states[rs.slot, shift:].clone())
                rs.wshift = shift
                cache_pos -= shift
            else:
                rs.wshift = 0
            positions_l.append(cache_pos)

        slots = tuple(rs.slot for rs in rsg)
        bt = self.bt_cache.get(slots)
        if bt is None:
            bt = torch.tensor(
                [[slot * pps + j for j in range(pps)] for slot in slots],
                dtype = torch.int32, device = self.device
            )
            self.bt_cache[slots] = bt
        cache_seqlens = torch.tensor(positions_l, dtype = torch.int32).to(self.device, non_blocking = True)
        return bt, cache_seqlens


    def bc_swa_step(self, x, rsg, params, seqlen):
        """Graph-captured decode step over the sliding-window state (projections through
        o_proj as one replayed CUDA graph). Returns the block output, or None when the
        configuration is unsupported."""
        if x.dtype != torch.float16 or not x.is_contiguous():
            return None
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        if rsg[0].exported:
            rsl = self.tp_recurrent_lookup[rsg[0].cache]
        else:
            rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
        key = id(rsl)
        bca = self.bc_attn.get(key)
        if bca is None:
            bca = self.bc_attn[key] = (build_bc_swa(self, rsl) or False)
        if bca is False:
            return None

        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        position = params.get("position", 0)

        k_states, v_states = rsl.get_state_tensors()
        bt, cache_seqlens = self._decode_state_prep(rsg, k_states, v_states, seqlen)
        return bca.step(x, cache_seqlens, bt, position, positions, position_ids, inv_freq)


    def decode_flash_attn_nc(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        causal = params.get("causal", True)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)

        q, k, v, g = self.project_qkv(x, params)

        if self.q_norm:
            if not self.rope or self.q_norm_tensor is None:
                q = self.q_norm.forward(q, params, out_dtype = torch.half)
                k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor,
                self.k_norm_tensor,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq,
                self.post_rope_norm
            )

        o = paged_attn_triton_prefill(
            q, None, None, None, None, None, None,
            causal = causal,
            softmax_scale = self.sm_scale,
            window_size = (self.sliding_window, self.sliding_window),
            softcap = self.logit_softcapping,
            sinks = self.sinks,
            k_new = k,
            v_new = v,
        )

        if self.headwise_gate: o *= g.sigmoid().unsqueeze(-1)
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.full_gate: o *= g.sigmoid()
        o = self.project_o(o, bsz, seqlen, params)
        return o


    def decode_flash_attn(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        causal = params.get("causal", True)
        non_causal_spans = params.get("non_causal_spans")

        # Graph-captured C++ path for the whole decode step
        if (
            _bc_attn_enable and causal and non_causal_spans is None and
            bsz <= _bc_max_bsz and seqlen <= _bc_max_qlen
        ):
            rsg = params.get("recurrent_states")
            if rsg is not None:
                o = self.bc_swa_step(x, rsg, params, seqlen)
                if o is not None:
                    return o

        q, k, v, g = self.project_qkv(x, params)

        if self.q_norm:
            if not self.rope or self.q_norm_tensor is None:
                q = self.q_norm.forward(q, params, out_dtype = torch.half)
                k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor,
                self.k_norm_tensor,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq,
                self.post_rope_norm
            )

        # Recurrent state: a short contiguous fp16 K/V span per sequence, viewed as a paged
        # cache (kv_state_size is a multiple of the page size) so the batch runs as one kernel
        # call with per-row block tables and sequence lengths
        rsg = params.get("recurrent_states")
        assert rsg is not None, "SlidingAttention.forward() called in flash_attn mode with no recurrent states"
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        if rsg[0].exported:
            rsl = self.tp_recurrent_lookup[rsg[0].cache]
        else:
            rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
        k_states, v_states = rsl.get_state_tensors()

        S = self.kv_state_size
        pps = S // PAGE_SIZE
        k_pages = k_states.view(-1, PAGE_SIZE, self.num_kv_heads, self.head_dim)
        v_pages = v_states.view(-1, PAGE_SIZE, self.num_kv_heads, self.head_dim)
        sw = self.sliding_window

        def pad(z):
            return (z + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE

        if seqlen <= 16 and not non_causal_spans:
            # Decode: append into the state (shifting it left first on the rare step where the
            # window run reaches the end of the buffer) and attend over the pages
            bt, cache_seqlens = self._decode_state_prep(rsg, k_states, v_states, seqlen)

            o = paged_attn_triton_decode(
                q, k, v, k_pages, v_pages, bt, cache_seqlens,
                causal = causal,
                softmax_scale = self.sm_scale,
                window_size = (sw, 0),
                softcap = self.logit_softcapping,
                sinks = self.sinks,
                max_kv_len = S,
            )

        else:
            # Prefill: attend over [hot window of the state || new K/V], reading the new tokens
            # straight from the projection output (nothing staged, the state is untouched until
            # after), then rebase the state onto the tail
            hots, wposs = [], []
            for rs in rsg:
                cache_pos = rs.position - rs.window_beg
                cache_wpos = max(cache_pos - sw, 0) // PAGE_SIZE * PAGE_SIZE
                wposs.append(cache_wpos)
                hots.append(cache_pos - cache_wpos)

            bt = torch.tensor(
                [[rs.slot * pps + wposs[i] // PAGE_SIZE + j for j in range(pps)] for i, rs in enumerate(rsg)],
                dtype = torch.int32, device = self.device
            )
            cache_seqlens = torch.tensor(hots, dtype = torch.int32).to(self.device, non_blocking = True)

            if not non_causal_spans:
                o = paged_attn_triton_prefill(
                    q, None, None, k_pages, v_pages, bt, cache_seqlens,
                    causal = causal,
                    softmax_scale = self.sm_scale,
                    window_size = (sw, 0),
                    softcap = self.logit_softcapping,
                    sinks = self.sinks,
                    k_new = k,
                    v_new = v,
                )
            else:
                # Spans tile the chunk; each attends over the state plus the new tokens up to its
                # own end, bidirectionally within itself when flagged non-causal
                o = []
                for span in non_causal_spans:
                    a, b, c = span[:3]
                    # pre-chunk extent of a re-fed span suffix: widen the left window as if the
                    # whole span were in the chunk
                    pre = span[3] if len(span) > 3 else 0
                    l = b - a
                    o_ = paged_attn_triton_prefill(
                        q[:, a : b].contiguous(), None, None, k_pages, v_pages, bt, cache_seqlens,
                        causal = not c,
                        softmax_scale = self.sm_scale,
                        window_size = (max(sw, l + pre), l - 1) if c else (sw, 0),
                        softcap = self.logit_softcapping,
                        sinks = self.sinks,
                        k_new = k[:, : b].contiguous(),
                        v_new = v[:, : b].contiguous(),
                    )
                    o.append(o_)
                o = torch.cat(o, dim = 1)

            # State update: keep the last window (page-aligned base) of [old || new] per row
            for i, rs in enumerate(rsg):
                cache_pos = rs.position - rs.window_beg
                if cache_pos + seqlen <= S:
                    k_states[rs.slot, cache_pos : cache_pos + seqlen].copy_(k[i])
                    v_states[rs.slot, cache_pos : cache_pos + seqlen].copy_(v[i])
                    rs.wshift = 0
                else:
                    cache_wpos, cache_hot = wposs[i], hots[i]
                    shift = pad(cache_hot + seqlen) - S
                    rs.wshift = shift + cache_wpos
                    n_keep = cache_hot - shift
                    if n_keep > 0:
                        k_states[rs.slot, :n_keep].copy_(k_states[rs.slot, cache_wpos + shift : cache_wpos + cache_hot].clone())
                        v_states[rs.slot, :n_keep].copy_(v_states[rs.slot, cache_wpos + shift : cache_wpos + cache_hot].clone())
                        k_states[rs.slot, n_keep : n_keep + seqlen].copy_(k[i])
                        v_states[rs.slot, n_keep : n_keep + seqlen].copy_(v[i])
                    else:
                        skip = shift - cache_hot
                        k_states[rs.slot, : seqlen - skip].copy_(k[i, skip:])
                        v_states[rs.slot, : seqlen - skip].copy_(v[i, skip:])

        if self.headwise_gate: ext.mul_sigmoid_broadcast_(o, g)
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.full_gate: ext.mul_sigmoid_(o, g)

        o = self.project_o(o, bsz, seqlen, params)
        return o


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.q_proj.storage_size()
        storage += self.k_proj.storage_size()
        storage += self.v_proj.storage_size()
        storage += self.o_proj.storage_size()
        if self.g_proj is not None:
            storage += self.g_proj.storage_size()
        for rl in self.recurrent_layers:
            storage += rl.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        overhead_s += 2 * self.num_q_heads * self.head_dim * torch.half.itemsize  # q, o
        overhead_s += 2 * self.num_kv_heads * self.head_dim * torch.half.itemsize  # k, v
        recons = max(
            self.q_proj.recons_size(),
            self.k_proj.recons_size(),
            self.v_proj.recons_size(),
            self.o_proj.recons_size(),
        )
        channel_width = 1
        channels_to_split = self.num_kv_heads
        while channel_width * self.head_dim < 128:
            assert channels_to_split % 2 == 0, \
                "Model's K/V heads cannot divide into 128-channel tensors"
            channel_width *= 2
            channels_to_split //= 2
        assert (channel_width * self.head_dim) % 128 == 0, \
            "Model's K/V heads cannot divide into 128-channel tensors"
        tpa = TPAllocation(
            key = self.key,
            channel_width = channel_width,
            channel_unit = "heads",
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
        assert not (self.q_norm is not None and isinstance(self.q_norm, RMSNorm) and self.q_norm.span_heads), \
            "TP export of SlidingAttention with span_heads norms is not implemented"

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": SlidingAttention,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "head_dim": self.head_dim,
                "rope_settings": self.rope_settings,
                "sm_scale": self.sm_scale,
                "out_dtype": self.out_dtype,
                "sliding_window": self.sliding_window,
                "sliding_window_overp": self.sliding_window_overp,
                "logit_softcapping": self.logit_softcapping,
                "post_rope_norm": self.post_rope_norm,
                "full_gate": self.full_gate,
            },
            "num_kv_heads": self.num_kv_heads,
            "n_gqa": self.num_q_heads // self.num_kv_heads,
            **{name: _export(getattr(self, name, None)) for name in (
                "q_norm",
                "k_norm",
                "v_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "g_proj",
            )},
            # Learned attention sinks (gpt-oss): one logit per query head, sliced to the local heads on import
            "sinks": producer.send(self.sinks) if self.sinks is not None else None,
            "device": self.device,
            "recurrent_layers": [
                rl.tp_export(plan) for rl in self.recurrent_layers
            ],
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        head_dim = exported["kwargs"]["head_dim"]
        full_gate = exported["kwargs"]["full_gate"]
        n_gqa = exported["n_gqa"]
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "heads"
        num_kv_heads = last - first
        num_q_heads = num_kv_heads * n_gqa

        q_split = (True, first * head_dim * n_gqa, last * head_dim * n_gqa) \
            if num_kv_heads else None
        kv_split = (True, first * head_dim, last * head_dim) \
            if num_kv_heads else None
        o_split = (False, first * head_dim * n_gqa, last * head_dim * n_gqa) \
            if num_kv_heads else None
        # Full gate spans head_dim channels per q head, headwise gate is one channel per q head
        if full_gate:
            g_split = (True, first * head_dim * n_gqa, last * head_dim * n_gqa) \
                if num_kv_heads else None
        else:
            g_split = (True, first * n_gqa, last * n_gqa) \
                if num_kv_heads else None

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        module = SlidingAttention(
            config = None,
            **exported["kwargs"],
            num_q_heads = num_q_heads,
            num_kv_heads = num_kv_heads,
            # Head-dim-wide norm weights are shared across heads; span_heads is rejected at export
            q_norm = _import("q_norm"),
            k_norm = _import("k_norm"),
            v_norm = _import("v_norm"),
            q_proj = _import_split("q_proj", q_split),
            k_proj = _import_split("k_proj", kv_split),
            v_proj = _import_split("v_proj", kv_split),
            o_proj = _import_split("o_proj", o_split),
            g_proj = _import_split("g_proj", g_split),
        )

        # Attention sinks are one logit per query head; each rank keeps its local head range
        if exported.get("sinks") is not None and num_kv_heads:
            consumer = local_context["consumer"]
            module.sinks = consumer.recv(
                exported["sinks"], cuda = True, slice_dim = 0, first = first * n_gqa, last = last * n_gqa
            )

        if num_kv_heads:
            recurrent_layers = exported["recurrent_layers"]
            if len(recurrent_layers):
                module.has_split_cache = True
                for rl in recurrent_layers:
                    rli = rl["cls"](module, **rl["args"])
                    module.recurrent_layers.append(rli)
                    module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True

        module.load_local(device)
        torch.cuda.synchronize()
        return module
