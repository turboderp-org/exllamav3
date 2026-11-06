"""
Query-only attention over another model's K/V cache.

A drafter whose layers share the target's keys and values (the Gemma4 assistant drafters) has no K or V
projections of its own: every step projects a query, rotates it to the position of the token it consumes and
attends over the target's cache for one of the target's layers. Nothing is written, so the drafter needs no
cache, no prefill and no rewinds, and every step of a drafting round sees the same keys.

The source layer is bound at attach time: a paged layer of the target's Cache, or the window ring of one of
its sliding-window layers (a SWAState the generator carries per sequence, passed under its own key,
params["source_recurrent_states"], because params["recurrent_states"] names a model's OWN states, which
Model.forward advances afterwards). Both are read exactly the way the target's own attention reads them in a
decode step, through the paged decode kernel with no appended K/V: natively (attention_fn/bc_shared_kv.py,
one C++ call per step, graph-replayed where graphs are available) or through the python path.
"""
from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config
from ..util.device_copy import host_to_device
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm
from ..constants import PAGE_SIZE
from .attention_fn.triton_paged import paged_attn_triton_decode
from .attention_fn.bc_attn import bc_attn_enable as _bc_attn_enable, MAX_BSZ as _bc_max_bsz, MAX_QLEN as _bc_max_qlen
from .attention_fn.bc_shared_kv import build_bc_shared_kv


class SharedKVAttention(Module):

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
        sliding_window: int = -1,
        key_q: str = "q_proj",
        key_o: str = "o_proj",
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        sm_scale: float | None = None,
        logit_softcapping: float = 0.0,
        q_norm: RMSNorm | None = None,
        qbits_key: str = "bits",
    ):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.rope_settings = rope_settings
        self.rope = None
        self.sliding_window = sliding_window
        self.out_dtype = out_dtype
        self.sm_scale = sm_scale or head_dim ** (-0.5)
        self.logit_softcapping = logit_softcapping
        self.q_norm = q_norm
        self.q_proj = Linear(
            config = config,
            key = f"{key}.{key_q}",
            in_features = hidden_size,
            out_features = num_q_heads * head_dim,
            qmap = qmap + ".input" if qmap else None,
            qbits_key = qbits_key,
        )
        self.o_proj = Linear(
            config = config,
            key = f"{key}.{key_o}",
            in_features = num_q_heads * head_dim,
            out_features = hidden_size,
            qmap = qmap + ".o" if qmap else None,
            qbits_key = qbits_key,
        )
        self.register_submodule(self.q_norm)
        self.register_submodule(self.q_proj)
        self.register_submodule(self.o_proj)
        # Bound by attach_source(): the target layer whose K/V this layer reads
        self.source_kind = None       # "paged" (target Cache layer) or "ring" (target SlidingAttention state)
        self.source_layer_idx = None
        self.source_module = None
        self._bt_cache = {}
        # Native step objects per source tensor pair (see attention_fn/bc_shared_kv.py); False = declined
        self.bc_attn = {}


    def attach_source(self, kind: str, layer_idx: int, module = None):
        assert kind in ("paged", "ring")
        if kind == "ring":
            assert module is not None, "ring source needs the target's SlidingAttention module"
            assert module.num_kv_heads == self.num_kv_heads and module.head_dim == self.head_dim, \
                f"{self.key}: K/V geometry differs from target layer {layer_idx}"
        self.source_kind = kind
        self.source_layer_idx = layer_idx
        self.source_module = module


    @override
    def optimizer_targets(self):
        return [[self.q_proj.optimizer_targets(), [], self.o_proj.optimizer_targets()]]


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.rope_settings:
            self.rope = RoPE(device, self.rope_settings)


    @override
    def unload(self):
        super().unload()
        self.rope = None
        self._bt_cache.clear()
        self.bc_attn.clear()


    def _bc_step(self, x, cache_k, cache_v, block_table, cache_seqlens, params):
        """
        Native path: projections, q-norm/RoPE and the paged attention over the source pages as one C++ call
        (one replayed CUDA graph per batch shape after warmup). None when the configuration is unsupported
        """
        key = (cache_k.data_ptr(), cache_v.data_ptr())
        bca = self.bc_attn.get(key)
        if bca is None:
            bca = self.bc_attn[key] = (build_bc_shared_kv(self, cache_k, cache_v) or False)
        if bca is False:
            return None
        return bca.step(
            x, cache_seqlens, block_table, params.get("position", 0),
            get_for_device(params, "positions", self.device, None),
            get_for_device(params, "position_ids", self.device, None),
        )


    @override
    def weights_numel(self):
        return self.q_proj.weights_numel() + self.o_proj.weights_numel()


    def _ring_pages(self, rsg: list, instance: int):
        """The target's window ring viewed as a paged cache, and each row's in-ring length"""
        tmod = self.source_module
        rsl = rsg[0].cache.get_recurrent_layer((self.source_layer_idx, instance))
        k_states, v_states = rsl.get_state_tensors()
        S = tmod.kv_state_size
        pps = S // PAGE_SIZE
        k_pages = k_states.view(-1, PAGE_SIZE, tmod.num_kv_heads, tmod.head_dim)
        v_pages = v_states.view(-1, PAGE_SIZE, tmod.num_kv_heads, tmod.head_dim)
        slots = tuple(rs.slot for rs in rsg)
        bt = self._bt_cache.get(slots)
        if bt is None:
            bt = torch.tensor([[slot * pps + j for j in range(pps)] for slot in slots], dtype = torch.int32, device = self.device)
            self._bt_cache[slots] = bt
        seqlens = host_to_device(torch.tensor([rs.position - rs.window_beg for rs in rsg], dtype = torch.int32), self.device)
        return k_pages, v_pages, bt, seqlens, S


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        assert self.source_kind is not None, f"{self.key}: not attached to a target layer"
        assert params.get("attn_mode", "flash_attn") == "flash_attn", "SharedKVAttention runs in decode mode only"
        bsz, seqlen, _ = x.shape
        # Source pages and their addressing, shared by the native and the python path
        if self.source_kind == "paged":
            cache = params["cache"]
            block_table = get_for_device(params, "block_table", self.device)
            cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
            layer = cache.layers[self.source_layer_idx, params.get("layer_instance") or 0]
            k_cache, v_cache = layer.get_kv(cache_seqlens, block_table, -1)
            max_kv_len = None
        else:
            rsg = params.get("source_recurrent_states")
            assert rsg is not None, f"{self.key}: ring source needs the target's recurrent states"
            k_cache, v_cache, block_table, cache_seqlens, max_kv_len = self._ring_pages(rsg, params.get("layer_instance") or 0)
        if (
            _bc_attn_enable and x.dtype == torch.float16 and x.is_contiguous() and
            bsz <= _bc_max_bsz and seqlen <= _bc_max_qlen
        ):
            o = self._bc_step(x, k_cache, v_cache, block_table, cache_seqlens, params)
            if o is not None:
                return to2(o, out_dtype, self.out_dtype)
        q = self.q_proj.forward(x, params, out_dtype = torch.half)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm.forward(q, params, out_dtype = torch.half)
        if self.rope is not None:
            q, _ = self.rope.apply(
                q, None,
                params.get("position", 0),
                get_for_device(params, "positions", self.device, None),
                get_for_device(params, "position_ids", self.device, None),
                True,
            )
        o = paged_attn_triton_decode(
            q, None, None, k_cache, v_cache, block_table, cache_seqlens,
            causal = True,
            softmax_scale = self.sm_scale,
            window_size = (self.sliding_window, 0) if self.sliding_window > 0 else None,
            softcap = self.logit_softcapping,
            max_kv_len = max_kv_len,
        )
        o = o.reshape(bsz, seqlen, self.num_q_heads * self.head_dim)
        o = self.o_proj.forward(o, params)
        return to2(o, out_dtype, self.out_dtype)
