from __future__ import annotations
from typing_extensions import override
import math
import torch
import os
import torch.nn.functional as F
from ..model.config import Config
from ..model.model_tp_alloc import TPAllocation
from .module import Module
from .linear import Linear
from .rmsnorm import RMSNorm
from ..ext import exllamav3_ext as ext
from ..util.rope import RopeStyle, yarn_inv_freq
from ..util.tensor import g_tensor_cache, get_for_device
from .quant.exl3 import LinearEXL3
from ..cache.dsa import DSV4LayerState, CacheLayer_dsa
from .multilinear import MultiLinear
from .attention_fn.dsa_triton import dsa_attn, dsa_indexer_scores, dsa_debug_bounds
from .attention_fn.bc_dsa import bc_dsa_enable, build_bc_dsa, build_bc_dsa_batch

# Multi-job decode via the batched path: one projection pass, one MULTIROW attention
# call and one grouped o_proj over all rows (per-slot state addressed through slot-id /
# per-job position arrays in-kernel). EXL3_DSV4_BATCH_EAGER=0 restores the loop
dsv4_batch_eager = os.environ.get("EXL3_DSV4_BATCH_EAGER", "1") != "0"
# Capture the batched step as one CUDA graph per (cache layer, B, S) after two warmup
# runs; EXL3_DSV4_BATCH_GRAPH=0 keeps the eager batched chain
dsv4_batch_graph = os.environ.get("EXL3_DSV4_BATCH_GRAPH", "1") != "0"
from ..constants import PAGE_SIZE

# Reference: transformers models/deepseek_v4 (paper §2)

def _ext_rope(x, inv_freq, position = 0, position_ids = None):
    # In-place GPT-J rotation of a (bsz, seq, heads, rope_dim) tensor via ext.rope. x may be
    # a trailing-slice VIEW of wider heads. De-rotation (paper eq. 26) uses a negated inv_freq
    # table. attn_factor is 1.0 by V4 semantics (yarn mscale never applied to cos/sin).
    ext.rope(
        x, x, None, None, inv_freq, position, None, position_ids,
        int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0,
    )


class DSV4CompressorState:
    """
    Cross-chunk state interface for one compressor (one entry name in HF terms). The
    compressor calls, in order:

    - entry_count (positioning)
    - get_buffer (rows carried from previous chunks)
    - store_rows (persist this chunk's projected rows)
    - advance_entries (windows emitted; consume rows)
    - get_overlap / set_overlap (Ca slice, overlapping scheme only).

    This base class is the simple in-memory implementation (whole-tensor buffers, no rewind); the
    cache layer provides a ring-backed subclass whose bookkeeping is derived from the absolute
    position so that rewind is pure cursor arithmetic.
    """

    def __init__(self):
        self._rows_kv = None       # (bsz, n, proj_width) fp32: rows not yet consumed
        self._rows_gate = None
        self._overlap = None       # (kv (bsz, m, hd), gate (bsz, m, hd)) fp32 or None
        self._count = 0

    @property
    def entry_count(self):
        return self._count

    def get_buffer(self):
        if self._rows_kv is None or self._rows_kv.shape[1] == 0:
            return None
        return self._rows_kv, self._rows_gate

    def store_rows(self, kv_new, gate_new):
        if self._rows_kv is None or self._rows_kv.shape[1] == 0:
            self._rows_kv, self._rows_gate = kv_new, gate_new
        else:
            self._rows_kv = torch.cat([self._rows_kv, kv_new], dim = 1)
            self._rows_gate = torch.cat([self._rows_gate, gate_new], dim = 1)

    def advance_entries(self, nw, m):
        self._count += nw
        if nw and self._rows_kv is not None:
            self._rows_kv = self._rows_kv[:, nw * m:].contiguous()
            self._rows_gate = self._rows_gate[:, nw * m:].contiguous()

    def get_overlap(self):
        return self._overlap

    def set_overlap(self, kv, gate):
        self._overlap = (kv.clone(), gate.clone())


class DSV4Compressor:
    """
    Torch-composed compressor shared by HCA (width = head_dim, non-overlapping) and CSA /
    indexer (width = 2 * head_dim: the Ca / Cb overlapping-window scheme). Stateless when
    state is None (complete windows within the chunk only, remainder discarded), stateful
    otherwise (buffer + overlap carried across chunks).

    Not a Module: owned by DSV4Attention, which registers the child Linears/norm.
    """

    # BC scratch row budget; must match BC_DSV4Compressor::MAX_QLEN (dsv4_compressor.h)
    BC_MAX_QLEN = 32

    def __init__(
        self,
        attn,
        key,
        head_dim,
        compress_rate,
        overlapping,
        qmap,
        select_hq_bits,
        wkv: Linear | None = None,
        wgate: Linear | None = None,
        norm: RMSNorm | None = None,
    ):
        cfg = attn.config
        proj_width = 2 * head_dim if overlapping else head_dim
        self.head_dim = head_dim
        self.rope_dim = attn.rope_head_dim if head_dim > attn.rope_head_dim else head_dim
        self.compress_rate = compress_rate
        self.overlapping = overlapping
        self.key = key
        self.wkv = wkv if wkv is not None else Linear(
            cfg, f"{key}.wkv", attn.hidden_size, proj_width, qmap = qmap, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.wgate = wgate if wgate is not None else Linear(
            cfg, f"{key}.wgate", attn.hidden_size, proj_width, qmap = qmap, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.norm = norm if norm is not None else RMSNorm(cfg, f"{key}.norm", attn.rms_norm_eps)
        self.ape = None  # (compress_rate, proj_width) fp32, loaded by DSV4Attention.load
        self.bc = None   # BC_DSV4Compressor companion (cached path), built lazily
        self.fused_ready = False
        self.fused_inv_freq = None
        self.fused_norm_w = None


    def make_bc(self, inv_freq):
        # Arm the fused cached path. The BC companion itself is built lazily on the first
        # forward_fused call: at load time weight tensors may still be DEFERRED (stloader fills
        # them asynchronously), so any snapshot/copy taken here could capture zeros.
        self.fused_inv_freq = inv_freq


    def _build_fused(self):
        # Build the C++ bound class: whole forward (projections + fused compress kernels) in
        # one transition. Requires unpadded projections (always true for the real V4 dims);
        # quantized and fp16 checkpoints both supported.

        self.fused_ready = True

        # bf16 checkpoints: norm weights are O(1), bf16 -> fp16 is exact
        self.fused_norm_w = self.norm.weight.data
        if self.fused_norm_w.dtype != torch.half:
            self.fused_norm_w = self.fused_norm_w.half().contiguous()
        wkv_i, wgate_i = self.wkv.inner, self.wgate.inner
        if (
            self.wkv.out_features != self.wkv.out_features_unpadded or
            self.wgate.out_features != self.wgate.out_features_unpadded
        ):
            # padded projections: python fallback path
            return

        W = self.wkv.out_features
        device = self.ape.device

        kv_scratch = torch.empty((self.BC_MAX_QLEN, W), dtype = torch.half, device = device)           # TODO: tensor cache
        gate_scratch = torch.empty((self.BC_MAX_QLEN, W), dtype = torch.half, device = device)
        # 2 slabs: the batched wkv+wgate mgemm writes one transformed input per expert
        xh_scratch = torch.empty((2 * self.BC_MAX_QLEN, self.wkv.in_features), dtype = torch.half, device = device)

        args = dict(
            wkv_exl3 = None, wkv_fp16 = None, wgate_exl3 = None, wgate_fp16 = None,
            ape = self.ape, norm_w = self.fused_norm_w, rms_norm_eps = self.norm.rms_norm_eps,
            inv_freq = self.fused_inv_freq, m = self.compress_rate,
            kv_scratch = kv_scratch, gate_scratch = gate_scratch, xh_scratch = xh_scratch,
        )
        for tag, inner in [("wkv", wkv_i), ("wgate", wgate_i)]:
            if isinstance(inner, LinearEXL3):
                args[f"{tag}_exl3"] = inner.bc
            elif hasattr(inner, "weight") and inner.weight.dtype == torch.half:
                args[f"{tag}_fp16"] = ext.BC_LinearFP16(inner.weight, getattr(inner, "bias", None))
            else:
                return

        # Batched wkv+wgate projection: one 2-expert exl3_mgemm when formats match
        if (
            isinstance(wkv_i, LinearEXL3) and
            isinstance(wgate_i, LinearEXL3) and
            wkv_i.K == wgate_i.K and
            wkv_i.mcg == wgate_i.mcg and
            wkv_i.mul1 == wgate_i.mul1
        ):
            args["mg_trellis"] = torch.tensor(
                [wkv_i.trellis.data_ptr(), wgate_i.trellis.data_ptr()],
                dtype = torch.long, device = device
            )
            args["mg_suh"] = torch.tensor(
                [wkv_i.suh.data_ptr(), wgate_i.suh.data_ptr()],
                dtype = torch.long, device = device
            )
            args["mg_svh"] = torch.tensor(
                [wkv_i.svh.data_ptr(), wgate_i.svh.data_ptr()],
                dtype = torch.long, device = device
            )
            args["mg_indices"] = torch.arange(2, dtype = torch.long, device = device).unsqueeze(0)

        self.bc = ext.BC_DSV4Compressor(**args)


    def unmake_bc(self):
        self.bc = None
        self.fused_ready = False
        self.fused_inv_freq = None
        self.fused_norm_w = None


    def forward_fused(self, x, params, buf_kv, buf_gate, ovl, dest_a, dest_b, position,
                      pool_bt = None, pool_epp = 0):
        """Cached-path forward (bsz 1): project + window-pool + norm + rope, writing emitted
        entries straight into the per-slot pools and updating the ring/snapshot state. One
        C++ transition via the BC companion when the chunk fits its scratch, else the two
        Linear forwards + the fused kernels."""
        bsz, seq, _ = x.shape
        if not self.fused_ready:
            self._build_fused()
        # The BC companion bypasses Linear.forward, which must run during conversion so the
        # capture/override machinery sees the projection inputs
        use_bc = self.bc is not None and seq <= self.BC_MAX_QLEN and \
            not any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        if use_bc:
            self.bc.run(x[0], buf_kv, buf_gate, ovl, dest_a, dest_b, position, None, None,
                        pool_bt, pool_epp)
        else:
            kv = self.wkv.forward(x, params)[0]
            gate = self.wgate.forward(x, params)[0]
            ext.dsv4_compress(
                kv, gate, buf_kv, buf_gate, ovl, self.ape, self.fused_norm_w,
                self.norm.rms_norm_eps, self.fused_inv_freq, dest_a, dest_b, position,
                None, self.compress_rate, None, pool_bt, pool_epp,
            )


    def forward(self, x, params, inv_freq, state: DSV4CompressorState | None = None):
        """x (bsz, seq, hidden) half. Returns newly emitted compressed entries
        (bsz, n_windows, head_dim) half, roped at their window positions
        (w + entry_count) * compress_rate. With a state, sub-window remainders are buffered
        for the next chunk and the Ca overlap slice is carried; entry_count is advanced."""
        bsz, seq, _ = x.shape
        m = self.compress_rate
        kv_new = self.wkv.forward(x, params).float()
        gate_new = self.wgate.forward(x, params).float()
        kv, gate = kv_new, gate_new
        fwp = 0
        if state is not None:
            fwp = state.entry_count * m
            buf = state.get_buffer()
            if buf is not None:
                kv = torch.cat([buf[0], kv], dim = 1)
                gate = torch.cat([buf[1], gate], dim = 1)
            state.store_rows(kv_new, gate_new)
        usable = (kv.shape[1] // m) * m
        if usable == 0:
            return x.new_zeros((bsz, 0, self.head_dim))
        kv = kv[:, :usable].view(bsz, -1, m, kv.shape[-1])
        gate = gate[:, :usable].view(bsz, -1, m, gate.shape[-1]) + self.ape
        nw = kv.shape[1]
        if state is not None:
            state.advance_entries(nw, m)
        if self.overlapping:
            hd = self.head_dim
            new_kv = kv.new_zeros((bsz, nw, 2 * m, hd))
            new_gate = gate.new_full((bsz, nw, 2 * m, hd), -float("inf"))
            new_kv[:, :, m:] = kv[..., hd:]
            new_gate[:, :, m:] = gate[..., hd:]
            if nw > 1:
                new_kv[:, 1:, :m] = kv[:, :-1, :, :hd]
                new_gate[:, 1:, :m] = gate[:, :-1, :, :hd]
            if state is not None:
                ovl = state.get_overlap()
                if ovl is not None:
                    new_kv[:, 0, :m] = ovl[0]
                    new_gate[:, 0, :m] = ovl[1]
                # NOTE: the saved gate slice already carries the position bias (ape); it is
                # not re-added when restored into the next window's first half (HF semantics)
                state.set_overlap(kv[:, -1, :, :hd], gate[:, -1, :, :hd])
            kv, gate = new_kv, new_gate
        comp = (kv * gate.softmax(dim = 2)).sum(dim = 2)
        comp = self.norm.forward(comp.half(), params)
        wpos = (torch.arange(nw, device = x.device, dtype = torch.int) * m + fwp)
        _ext_rope(comp.view(bsz, nw, 1, self.head_dim)[..., -self.rope_dim:], inv_freq,
                  position_ids = wpos.unsqueeze(0).expand(bsz, -1).contiguous())
        return comp


    def modules(self):
        return [self.wkv, self.wgate, self.norm]


    def tp_export(self, plan, producer):
        # The compressor produces headless shared pool entries: fully replicated per rank
        return {
            "args": {
                "key": self.key,
                "head_dim": self.head_dim,
                "compress_rate": self.compress_rate,
                "overlapping": self.overlapping,
            },
            "wkv": self.wkv.tp_export(plan, producer),
            "wgate": self.wgate.tp_export(plan, producer),
            "norm": self.norm.tp_export(plan, producer),
            "ape": producer.send(self.ape),
        }


    @staticmethod
    def tp_import(local_context, exported, plan, attn):
        consumer = local_context["consumer"]
        def _imp(name):
            e = exported[name]
            return e["cls"].tp_import(local_context, e, plan)
        comp = DSV4Compressor(
            attn,
            qmap = None,
            select_hq_bits = 0,
            wkv = _imp("wkv"),
            wgate = _imp("wgate"),
            norm = _imp("norm"),
            **exported["args"],
        )
        comp.ape = consumer.recv(exported["ape"], cuda = True)
        return comp


class DSV4Attention(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        layer_type: str,  # "sliding" | "csa" | "hca"
        hidden_size: int,
        num_q_heads: int,
        head_dim: int,
        rope_head_dim: int,
        q_lora_rank: int,
        o_groups: int,
        o_lora_rank: int,
        sliding_window: int,
        compress_rate: int | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        rope_scaling: dict | None = None,
        rms_norm_eps: float = 1e-6,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        qbits_key: str = "bits",
        select_hq_bits: int = 0,
        q_a: Linear | None = None,
        q_norm: RMSNorm | None = None,
        q_b: Linear | None = None,
        wkv: Linear | None = None,
        kv_norm: RMSNorm | None = None,
        wo_a: list | None = None,
        wo_b: Linear | None = None,
        idx_wq_b: Linear | None = None,
        idx_weights: Linear | None = None,
        tp_defer_compressors: bool = False,
    ):
        super().__init__(config = config, key = key, qmap = None)
        self.q_priority = 2 + select_hq_bits
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.compress_rate = compress_rate
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.out_dtype = out_dtype
        self.sm_scale = head_dim ** -0.5

        # For the model-level allocator conventions (not a KV cache module yet -- M2)
        self.num_kv_heads = 1

        # TP bookkeeping; set before the zero-width early return below (forward() and the
        # TP import glue touch these on ranks that hold none of this layer's o_groups)
        self.tp_reduce = False
        self.tp_mode = False
        self.has_split_cache = False
        self.tp_recurrent_lookup = {}
        self.tp_cache_lookup = {}
        self.recurrent_layers = []
        self.cache_layers = []
        self.sinks = None  # (num_q_heads,) fp32
        self.compressor = None
        self.indexer = None
        self.idx_wq_b = None
        self.idx_weights = None
        self.inv_freq_main = None
        self.inv_freq_compress = None
        self.wo_a_multi = None
        self.woa_indices = None
        self.woa_multi_ready = False
        self.x_fan = None
        self.q_fan = None
        self.x_fan_ready = False
        self._fan_scratch = {}
        self._bgraph_state = {}

        if num_q_heads == 0:
            # Zero-width TP shard: no weights, no state, no caps (keeps the rank out of
            # the child's kv/recurrent module lists); forward() contributes zeros
            self.q_a = self.q_norm = self.q_b = self.wkv = self.kv_norm = None
            self.wo_a = []
            self.wo_b = None
            return

        self.q_a = q_a if q_a is not None else Linear(
            config,
            f"{key}.wq_a",
            hidden_size,
            q_lora_rank,
            qmap = qmap,
            out_dtype = torch.half,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.q_norm = q_norm if q_norm is not None else RMSNorm(config, f"{key}.q_norm", rms_norm_eps)
        self.q_b = q_b if q_b is not None else Linear(
            config,
            f"{key}.wq_b",
            q_lora_rank,
            num_q_heads * head_dim,
            qmap = f"{key}.q_b",
            out_dtype = torch.half,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.wkv = wkv if wkv is not None else Linear(
            config,
            f"{key}.wkv",
            hidden_size,
            head_dim,
            qmap = qmap,
            out_dtype = torch.half,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.kv_norm = kv_norm if kv_norm is not None else RMSNorm(config, f"{key}.kv_norm", rms_norm_eps)

        group_width = num_q_heads * head_dim // o_groups
        self.wo_a = wo_a if wo_a is not None else [
            Linear(
                config,
                f"{key}.wo_a.slice.{g}",
                group_width,
                o_lora_rank,
                fkey = f"{key}.wo_a",
                frange = [g * o_lora_rank,
                (g + 1) * o_lora_rank],
                frange_dim = 0,
                qmap = f"{key}.o.{g}",
                out_dtype = torch.half,
                qbits_key = qbits_key,
                trim_padded_out = True,
                select_hq_bits = select_hq_bits,
            )
            for g in range(o_groups)
        ]
        self.wo_b = wo_b if wo_b is not None else Linear(
            config,
            f"{key}.wo_b",
            o_groups * o_lora_rank,
            hidden_size,
            qmap = f"{key}.o_b",
            out_dtype = torch.float,
            qbits_key = qbits_key,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.idx_wq_b = idx_wq_b
        self.idx_weights = idx_weights
        if layer_type in ("csa", "hca") and not tp_defer_compressors:
            self.compressor = DSV4Compressor(
                self,
                f"{key}.compressor",
                head_dim,
                compress_rate,
                overlapping = (layer_type == "csa"),
                qmap = qmap,
                select_hq_bits = select_hq_bits
            )
        if layer_type == "csa" and not tp_defer_compressors:
            self.indexer = DSV4Compressor(
                self,
                f"{key}.indexer.compressor",
                index_head_dim,
                compress_rate,
                overlapping = True,
                qmap = qmap,
                select_hq_bits = select_hq_bits
            )
        if layer_type == "csa" and self.idx_wq_b is None:
            self.idx_wq_b = Linear(
                config,
                f"{key}.indexer.wq_b",
                q_lora_rank,
                index_n_heads * index_head_dim,
                qmap = f"{key}.q_b",
                out_dtype = torch.half,
                qbits_key = qbits_key,
                trim_padded_out = True,
                select_hq_bits = select_hq_bits,
            )

            # Router-like scoring head: 4096 -> 64 (one logit per indexer head), unquantized
            self.idx_weights = Linear(
                config,
                f"{key}.indexer.weights_proj",
                hidden_size,
                index_n_heads,
                qmap = None,
                out_dtype = torch.half,
                pad_to = 1,
            )

        for m in [
            self.q_a,
            self.q_norm,
            self.q_b,
            self.wkv,
            self.kv_norm,
            *self.wo_a,
            self.wo_b,
            self.idx_wq_b,
            self.idx_weights
        ]:
            self.register_submodule(m)

        for comp in [
            self.compressor,
            self.indexer
        ]:
            if comp is not None:
                for m in comp.modules():
                    self.register_submodule(m)

        self.inv_freq_main = None
        self.inv_freq_compress = None

        # Batched wo_a slice projection (exl3_mgemm over the 8 group slices), built lazily
        # on first use; falls back to the per-slice loop for fp16 slices or mixed K
        self.wo_a_multi = None
        self.woa_indices = None
        self.woa_multi_ready = False

        # x-side / q_res-side projection fans (eager cached path), built lazily
        self.x_fan = None
        self.q_fan = None
        self.x_fan_ready = False
        self._fan_scratch = {}
        self._bgraph_state = {}

        # Hybrid cache pattern: fixed-size per-slot recurrent state (SWA ring, compressor
        # sub-window buffers, overlap snapshots) plus, for CSA/HCA layers, paged pools that
        # alias the token page table (CacheLayer_dsa)
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = DSV4LayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.cache_layers = []
        if self.layer_type in ("csa", "hca"):
            self.caps.update({"kv_cache": True})


    def cache_layer_type(self, default, kwargs: dict):
        """DSA pools are always CacheLayer_dsa regardless of the requested transformer cache
        layer type; a quantized Cache quantizes only its transformer/MLA layers (pools stay
        fp16 in v1, like recurrent state)."""
        return CacheLayer_dsa, {}


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.sinks = stc.get_tensor(f"{self.key}.attn_sink", device, no_defer = True).float().contiguous()

        self.inv_freq_main = yarn_inv_freq(self.rope_head_dim, self.rope_theta, device)
        self.inv_freq_compress = yarn_inv_freq(
            self.rope_head_dim, self.compress_rope_theta, device, rope_scaling = self.rope_scaling)
        self.inv_freq_main_neg = -self.inv_freq_main
        self.inv_freq_compress_neg = -self.inv_freq_compress

        # Head norms fused into the qkv rope call: unweighted q head norm = ones weight # (same dtype
        # as the kv norm weight; the kernel requires matching dtypes). .data references only
        self.kv_norm_w = self.kv_norm.weight.data
        self.q_ones = torch.ones(self.head_dim, dtype = self.kv_norm_w.dtype, device = device)

        if self.compressor is not None:
            self.compressor.ape = stc.get_tensor(f"{self.compressor.key}.ape", device, no_defer = True).float().contiguous()
            self.compressor.make_bc(self.inv_freq_compress)
        if self.indexer is not None:
            self.indexer.make_bc(self.inv_freq_compress)
            self.indexer.ape = stc.get_tensor(f"{self.indexer.key}.ape", device, no_defer = True).float().contiguous()
        for rl in self.recurrent_layers:
            rl.alloc(device)
        for cl in self.cache_layers:
            cl.alloc(device)


    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        for rl in self.recurrent_layers:
            rl.free()
        self.sinks = None
        if self.compressor is not None:
            self.compressor.ape = None
            self.compressor.unmake_bc()
        if self.indexer is not None:
            self.indexer.ape = None
            self.indexer.unmake_bc()
        self.inv_freq_main = self.inv_freq_compress = None
        self.inv_freq_main_neg = self.inv_freq_compress_neg = None
        self.q_ones = self.kv_norm_w = None
        self.x_fan = None
        self.q_fan = None
        self.x_fan_ready = False
        self._fan_scratch = {}
        self._bgraph_state = {}
        self._bc_dsa_batch = {}
        self._bc_dsa = {}
        self.wo_a_multi = None
        self.woa_indices = None
        self.woa_multi_ready = False


    @override
    def get_tensors(self):
        t = {f"{self.key}.attn_sink": self.sinks.contiguous()}
        if self.compressor is not None:
            t[f"{self.compressor.key}.ape"] = self.compressor.ape.contiguous()
        if self.indexer is not None:
            t[f"{self.indexer.key}.ape"] = self.indexer.ape.contiguous()
        return t


    @override
    def weights_numel(self):
        n = sum(m.weights_numel() for m in self.modules)
        n += self.num_q_heads
        for comp in [self.compressor, self.indexer]:
            if comp is not None:
                n += comp.compress_rate * (2 if comp.overlapping else 1) * comp.head_dim
        return n


    @override
    def optimizer_targets(self):
        return [t for m in self.modules for t in m.optimizer_targets()]


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        # Split unit = o_group (hpg heads + one wo_a slice + o_lora_rank rows of wo_b):
        # the grouped o_proj consumes whole 8-head slabs and the attention kernel stores
        # output group-major, so groups are the natural quantum. Everything KV-side is
        # shared-MQA and replicated per rank (q_a, wkv, norms, compressor, indexer, pools,
        # rings); only q_b / wo_a / wo_b / sinks split
        storage_split = self.q_b.storage_size() + self.wo_b.storage_size() + \
            sum(l.storage_size() for l in self.wo_a)
        storage_dev = self.q_a.storage_size() + self.wkv.storage_size()
        for l in (self.idx_wq_b, self.idx_weights):
            if l is not None:
                storage_dev += l.storage_size()
        for comp in (self.compressor, self.indexer):
            if comp is not None:
                storage_dev += comp.wkv.storage_size() + comp.wgate.storage_size()
        for rl in self.recurrent_layers:
            storage_dev += rl.storage_size()
        for cl in self.cache_layers:
            storage_dev += cl.storage_size()
        overhead_d = self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = self.num_q_heads * self.head_dim * torch.half.itemsize  # q rows
        overhead_s += self.o_groups * self.o_lora_rank * torch.half.itemsize  # o intermediates
        recons = max(
            self.q_a.recons_size(),
            self.q_b.recons_size(),
            self.wo_b.recons_size(),
            max((l.recons_size() for l in self.wo_a), default = 0),
        )
        tpa = TPAllocation(
            key = self.key,
            channel_width = 1,
            channel_unit = "heads",
            storage_per_device = storage_dev,
            storage_to_split = storage_split,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.o_groups,
            limit_key = "attn",
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": DSV4Attention,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "layer_type": self.layer_type,
                "hidden_size": self.hidden_size,
                "head_dim": self.head_dim,
                "rope_head_dim": self.rope_head_dim,
                "q_lora_rank": self.q_a.out_features_unpadded,
                "o_lora_rank": self.o_lora_rank,
                "sliding_window": self.sliding_window,
                "compress_rate": self.compress_rate,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
                "rope_theta": self.rope_theta,
                "compress_rope_theta": self.compress_rope_theta,
                "rope_scaling": self.rope_scaling,
                "rms_norm_eps": self.rms_norm_eps,
                "out_dtype": self.out_dtype,
            },
            "num_q_heads": self.num_q_heads,
            "o_groups": self.o_groups,
            **{name: _export(getattr(self, name, None)) for name in (
                "q_a",
                "q_norm",
                "q_b",
                "wkv",
                "kv_norm",
                "wo_b",
                "idx_wq_b",
                "idx_weights",
            )},
            "wo_a": [l.tp_export(plan, producer) for l in self.wo_a],
            "sinks": producer.send(self.sinks),
            "compressor": self.compressor.tp_export(plan, producer) if self.compressor is not None else None,
            "indexer": self.indexer.tp_export(plan, producer) if self.indexer is not None else None,
            "recurrent_layers": [rl.tp_export(plan) for rl in self.recurrent_layers],
            "cache_layers": [cl.tp_export(plan) for cl in self.cache_layers],
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        kw = exported["kwargs"]
        key = kw["key"]
        head_dim = kw["head_dim"]
        o_lora_rank = kw["o_lora_rank"]
        full_heads = exported["num_q_heads"]
        full_groups = exported["o_groups"]
        hpg = full_heads // full_groups
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "heads"
        num_groups = last - first
        num_q_heads = num_groups * hpg

        # Column split of q_b by the local head range; row (input-dim) split of wo_b by
        # the local group blocks -- the trailing all-reduce sums the partial products.
        # Everything else is replicated (shared-KV MQA + headless compressor/indexer)
        q_b_split = (True, first * hpg * head_dim, last * hpg * head_dim) \
            if num_groups else None
        wo_b_split = (False, first * o_lora_rank, last * o_lora_rank) \
            if num_groups else None

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if num_groups and exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        wo_a = [
            e["cls"].tp_import(local_context, e, plan)
            for e in exported["wo_a"][first : last]
        ] if num_groups else []

        module = DSV4Attention(
            config = None,
            **kw,
            num_q_heads = num_q_heads,
            o_groups = num_groups,
            q_a = _import("q_a"),
            q_norm = _import("q_norm"),
            q_b = _import_split("q_b", q_b_split),
            wkv = _import("wkv"),
            kv_norm = _import("kv_norm"),
            wo_a = wo_a,
            wo_b = _import_split("wo_b", wo_b_split),
            idx_wq_b = _import("idx_wq_b"),
            idx_weights = _import("idx_weights"),
            tp_defer_compressors = True,
        )
        module.tp_mode = True

        if num_groups:
            consumer = local_context["consumer"]
            # Per-head sinks: each rank keeps its local head range
            module.sinks = consumer.recv(
                exported["sinks"], cuda = True, slice_dim = 0,
                first = first * hpg, last = last * hpg,
            )
            if exported.get("compressor") is not None:
                module.compressor = DSV4Compressor.tp_import(
                    local_context, exported["compressor"], plan, module)
            if exported.get("indexer") is not None:
                module.indexer = DSV4Compressor.tp_import(
                    local_context, exported["indexer"], plan, module)
            for rl in exported["recurrent_layers"]:
                rli = rl["cls"](module, **rl["args"])
                module.recurrent_layers.append(rli)
                module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli
            if len(exported["cache_layers"]):
                module.has_split_cache = True
                for cl in exported["cache_layers"]:
                    cli = cl["cls"](None, module, **cl["args"])
                    module.cache_layers.append(cli)
                    module.tp_cache_lookup[cl["args"]["cache_id"]] = cli

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True

        module.load_local(device)
        torch.cuda.synchronize()
        return module


    def load_local(self, device, **kwargs):
        if self.num_q_heads == 0:
            return
        self.inv_freq_main = yarn_inv_freq(self.rope_head_dim, self.rope_theta, device)
        self.inv_freq_compress = yarn_inv_freq(
            self.rope_head_dim, self.compress_rope_theta, device, rope_scaling = self.rope_scaling)
        self.inv_freq_main_neg = -self.inv_freq_main
        self.inv_freq_compress_neg = -self.inv_freq_compress
        self.kv_norm_w = self.kv_norm.weight.data
        self.q_ones = torch.ones(self.head_dim, dtype = self.kv_norm_w.dtype, device = device)
        if self.compressor is not None:
            self.compressor.make_bc(self.inv_freq_compress)
        if self.indexer is not None:
            self.indexer.make_bc(self.inv_freq_compress)
        for rl in self.recurrent_layers:
            rl.alloc(device)
        for cl in self.cache_layers:
            cl.alloc(device)


    def _rope_type(self):
        return self.inv_freq_main if self.layer_type == "sliding" else self.inv_freq_compress


    def _rope_type_neg(self):
        return self.inv_freq_main_neg if self.layer_type == "sliding" else self.inv_freq_compress_neg


    def _project_qkv(self, x, params, position):
        """Shared front: q_a/q_norm/q_b and wkv, then ONE in-place ext.rope call that also
        applies both head norms (unweighted per-head q norm via a ones weight, weighted
        kv_norm) before rotating the trailing rope slice (rotate_offset)."""
        bsz, seq, _ = x.shape
        rd = self.rope_head_dim
        q_res = self.q_norm.forward(self.q_a.forward(x, params), params, out_dtype = torch.half)
        q = self.q_b.forward(q_res, params).view(bsz, seq, self.num_q_heads, self.head_dim)
        kv = self.wkv.forward(x, params).view(bsz, seq, 1, self.head_dim)
        ext.rope(
            q, q, kv, kv,
            self._rope_type(), position, None, None,
            int(RopeStyle.GPTJ), 1.0, self.q_ones, self.kv_norm_w,
            self.rms_norm_eps, 0.0, 0.0, 0, 1, self.head_dim - rd,
        )
        return q_res, q, kv.view(bsz, seq, self.head_dim)


    def _build_x_fan(self):
        """x-side projection fan for the eager cached path: q_a / wkv / comp wkv+wgate /
        idx wkv+wgate as ONE per-matrix-N exl3_mgemm (uniform bits/format required, q_a
        widest). Mirrors the BC_DSV4Attention fan. A second fan pairs q_b with idx_wq_b
        (both consume q_res) for the top-k regime."""
        self.x_fan_ready = True
        if os.environ.get("EXL3_DSV4_NO_XFAN", "0") != "0":
            return
        device = torch.device(self.device)

        def mk_fan(lins):
            inner = [l.inner for l in lins]
            if not all(l.quant_type == "exl3" for l in lins):
                return None
            if not all(l.out_features == l.out_features_unpadded for l in lins):
                return None
            if len({(i.K, i.mcg, i.mul1) for i in inner}) != 1:
                return None
            ns = [l.out_features for l in lins]
            if max(ns) != ns[0]:
                return None    # first output is the dtype/max-width carrier
            return dict(
                trellis = torch.tensor([i.trellis.data_ptr() for i in inner], dtype = torch.long, device = device),
                suh = torch.tensor([i.suh.data_ptr() for i in inner], dtype = torch.long, device = device),
                svh = torch.tensor([i.svh.data_ptr() for i in inner], dtype = torch.long, device = device),
                n = torch.tensor(ns, dtype = torch.int32, device = device),
                idx = torch.arange(len(lins), dtype = torch.long, device = device).unsqueeze(0),
                ns = ns, K = inner[0].K, mcg = inner[0].mcg, mul1 = inner[0].mul1,
            )

        lins = [self.q_a, self.wkv]
        if self.compressor is not None:
            self.compressor._build_fused() if not self.compressor.fused_ready else None
            lins += [self.compressor.wkv, self.compressor.wgate]
        if self.indexer is not None:
            self.indexer._build_fused() if not self.indexer.fused_ready else None
            lins += [self.indexer.wkv, self.indexer.wgate]
        self.x_fan = mk_fan(lins)
        self.q_fan = mk_fan([self.q_b, self.idx_wq_b]) if self.indexer is not None else None
        self._fan_scratch = {}

        from .multilinear import MultiLinear
        self.qb_multi = self.wob_multi = None
        self._one_idx = None
        try:
            if self.q_fan is None and self.q_b.quant_type == "exl3":
                self.qb_multi = MultiLinear(self.device, [self.q_b])
            if self.wo_b.quant_type == "exl3":
                self.wob_multi = MultiLinear(self.device, [self.wo_b])
            self._one_idx = torch.zeros((1, 1), dtype = torch.long, device = device)
        except AssertionError:
            self.qb_multi = self.wob_multi = None


    def _mgemm1(self, mu, x2, out_dtype, tag):
        """Single-matrix exl3_mgemm (one 'expert'): capture-safe multi-row linear for the
        graphed batched body. x2 (1, R, k) -> (1, R, n)."""
        R = x2.shape[1]
        C = g_tensor_cache.get(x2.device, (1, R, mu.out_features), out_dtype, tag + "_c")
        ah = g_tensor_cache.get(x2.device, (1, R, x2.shape[-1]), torch.half, tag)
        ext.exl3_mgemm(
            x2, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh,
            self._one_idx, None, mu.K, -1, mu.mcg, mu.mul1, -1, -1, 0, 1, None, None)
        return C

    def _fan_outs(self, fan, tag, seq, shapes):
        """Per-(seq) fan output scratch + pointer array, cached (g_tensor_cache reuses the
        same storage per shape, so the pointer arrays stay valid)."""
        key = (tag, seq)
        ent = self._fan_scratch.get(key)
        if ent is None:
            device = torch.device(self.device)
            outs = [g_tensor_cache.get(device, (seq,) + sh, torch.half, f"dsv4_fan_{tag}{i}")
                    for i, sh in enumerate(shapes)]
            cptr = torch.tensor([o.data_ptr() for o in outs], dtype = torch.long, device = device)
            ahad = g_tensor_cache.get(device, (len(outs), seq, self.hidden_size if tag == "x"
                                               else self.q_a.out_features),
                                      torch.half, f"dsv4_fan_{tag}_ah")
            ent = (outs, cptr, ahad)
            self._fan_scratch[key] = ent
        return ent


    def _build_woa_multi(self):
        self.woa_multi_ready = True
        try:
            if all(l.quant_type == "exl3" for l in self.wo_a):
                self.wo_a_multi = MultiLinear(self.device, self.wo_a)
                self.woa_indices = torch.arange(
                    self.o_groups, dtype = torch.long, device = self.device).unsqueeze(0)
        except AssertionError:
            self.wo_a_multi = None    # mixed K/format across slices: per-slice loop


    def _project_o_grouped(self, o, params, out_dtype, mgemm_out = False):
        """
        Grouped output projection: o (G, bsz, seq, hpg * head_dim) fp16 contiguous,
        rope slice already de-rotated. Short rows: ONE exl3_mgemm over the G slices (each
        group is an "expert" with its own input slice A[g]); at seq == 1 the expert-major
        output (G, 1, n) is memory-identical to the concatenated (1, G * n) row, so the cat
        is a free view. Long rows / fp16 / conversion: per-slice wo_a loop + cat.
        """
        G = self.o_groups
        bsz, seq = o.shape[1], o.shape[2]
        if not self.woa_multi_ready and self.device is not None:
            self._build_woa_multi()

        use_mg = (
            self.wo_a_multi is not None and bsz == 1 and seq <= 32
            and not any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        )
        if use_mg:
            mu = self.wo_a_multi
            A = o[:, 0]                                   # (G, seq, hpg * head_dim)
            ah = g_tensor_cache.get(o.device, tuple(A.shape), torch.half, "dsv4_woa_had")
            C = g_tensor_cache.get(o.device, (G, seq, mu.out_features), torch.half,
                                   "dsv4_woa_c") if mgemm_out else \
                torch.empty((G, seq, mu.out_features), dtype = torch.half, device = o.device)
            ext.exl3_mgemm(
                A, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh,
                self.woa_indices, None, mu.K, -1, mu.mcg, mu.mul1,
                -1, -1, 0, 1, None, None
            )
            if seq == 1:
                o2 = C.view(1, 1, G * mu.out_features)
            elif mgemm_out:
                o2 = g_tensor_cache.get(o.device, (seq, G * mu.out_features), torch.half,
                                        "dsv4_woa_t")
                o2.view(seq, G, mu.out_features).copy_(C.transpose(0, 1))
                o2 = o2.unsqueeze(0)
            else:
                o2 = C.permute(1, 0, 2).reshape(seq, G * mu.out_features).unsqueeze(0)
            if mgemm_out and self.wob_multi is not None and seq <= 32:
                return self._mgemm1(self.wob_multi, o2.contiguous(),
                                    out_dtype or self.out_dtype,
                                    f"dsv4_wob1_L{self.layer_idx}")
            return self.wo_b.forward(o2, params, out_dtype = out_dtype or self.out_dtype)

        o = torch.cat([self.wo_a[g].forward(o[g], params) for g in range(self.o_groups)], dim = -1)
        return self.wo_b.forward(o, params, out_dtype = out_dtype or self.out_dtype)


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        if self.num_q_heads == 0:
            # Zero-width TP shard: contribute nothing, keep the collective aligned
            y = torch.zeros_like(x, dtype = out_dtype or self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(y, False)
            return y
        mode = params.get("attn_mode", "flash_attn_nc")
        if mode == "flash_attn":
            y = self._forward_cached(x, params, out_dtype)
        else:
            assert mode == "flash_attn_nc", f"DSV4Attention: unsupported attn_mode {mode}"
            y = self._forward_nc(x, params, out_dtype)
        if self.tp_reduce:
            params["backend"].all_reduce(y)
        return y


    def _forward_nc(self, x, params, out_dtype):
        """
        Stateless single-shot pass (HF cache = None semantics: every complete compressor
        window in the chunk is compressed, the remainder discarded). Same kernels as the
        cached path:

        - fused compressor into throwaway scratch pools
        - indexer top-k index lists,
        - dsa_attn with in-chunk window indices and the fused derot/grouped epilogue

        No masks or eager attention are ever materialized.
        """
        bsz, seq, _ = x.shape
        device = x.device
        position = params.get("position", 0)

        q_res, q, kv = self._project_qkv(x, params, position)

        # Window rows come from the chunk itself (win_floor == q_pos0: no prior rows in nc)
        w = self.sliding_window

        m = self.compress_rate if self.compressor is not None else 1
        T = seq // m if self.compressor is not None else 0
        hpg = self.num_q_heads // self.o_groups
        hd = self.head_dim
        D_c, D_r = hd - self.rope_head_dim, self.rope_head_dim

        if self.compressor is not None:
            cap = max(T, 1)
            rows = min(seq, PAGE_SIZE + m)
            pool_c = g_tensor_cache.get(device, (cap, D_c), torch.half, "dsv4_nc_pool_c")
            pool_r = g_tensor_cache.get(device, (cap, D_r), torch.half, "dsv4_nc_pool_r")
            ring_kv = g_tensor_cache.get(device, (rows, self.compressor.wkv.out_features_unpadded),
                                         torch.half, "dsv4_nc_ring_kv")
            ring_gate = g_tensor_cache.get(device, ring_kv.shape, torch.half, "dsv4_nc_ring_gate")
            ovl = g_tensor_cache.get(device, (1, 2, m, hd), torch.float, "dsv4_nc_ovl") \
                if self.layer_type == "csa" else None
            bt = torch.arange(-(-cap // PAGE_SIZE), dtype = torch.int32, device = device).unsqueeze(0)
        else:
            pool_c = x.new_empty((1, D_c), dtype = torch.half)
            pool_r = x.new_empty((1, D_r), dtype = torch.half)
            bt = torch.zeros((1, 1), dtype = torch.int32, device = device)

        outs = []
        for b in range(bsz):
            indices = None
            k_len = 0
            if self.compressor is not None:
                # Window positions count from 0 in the stateless path (fwp = 0), matching
                # the HF cache = None reference; causal bounds use the absolute positions
                self.compressor.forward_fused(
                    x[b:b + 1], params, ring_kv, ring_gate, ovl, pool_c, pool_r, 0)
                if self.layer_type == "csa":
                    idx_hd = self.index_head_dim
                    pool_idx = g_tensor_cache.get(device, (max(T, 1), idx_hd), torch.half, "dsv4_nc_pool_idx")
                    iring_kv = g_tensor_cache.get(device, (rows, self.indexer.wkv.out_features_unpadded), torch.half, "dsv4_nc_iring_kv")
                    iring_gate = g_tensor_cache.get(device, iring_kv.shape, torch.half, "dsv4_nc_iring_gate")
                    iovl = g_tensor_cache.get(device, (1, 2, m, idx_hd), torch.float, "dsv4_nc_iovl")
                    self.indexer.forward_fused(x[b:b + 1], params, iring_kv, iring_gate, iovl, pool_idx, None, 0)
                    if T > self.index_topk:
                        indices, k_len = self._indexer_topk(
                            x[b:b + 1], params, q_res[b:b + 1], pool_idx[:T], T, position)

            out = dsa_attn(
                q[b].half().contiguous(), pool_c, pool_r, bt, sinks = self.sinks,
                kv_chunk = kv[b].contiguous(), win_len = w, win_floor = position,
                indices = indices, k_len = k_len, pool_len = T, q_pos0 = position,
                compress_rate = m, scale = self.sm_scale,
                derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
                out = torch.empty((self.o_groups, seq, hpg * hd), dtype = torch.half, device = device),
            )
            outs.append(self._project_o_grouped(out.unsqueeze(1), params, out_dtype))
        return torch.cat(outs, dim = 0) if bsz > 1 else outs[0]


    def _indexer_topk(
        self,
        x,
        params,
        q_res,
        idx_pool,
        ec,
        pos0,
        q_idx_pre = None,
        block_table = None,
        epp = 0,
    ):
        """Lightning-indexer scoring + top-k selection over the indexer key pool (ec valid
        rows). The indexer query rope uses the compress table at the query positions == this
        layer's own cos/sin (CSA layers rope with the compress table). Causal bounds and the
        head-weight scale live in the scoring kernel; the -1-padded int32 index list comes
        from the pack kernel. Returns (indices (seq, K_pad) int32, k_len)."""
        _, seq, _ = x.shape
        if q_idx_pre is not None:
            q_idx = q_idx_pre.view(1, seq, self.index_n_heads, self.index_head_dim)
        else:
            q_idx = self.idx_wq_b.forward(q_res, params).view(1, seq, self.index_n_heads, self.index_head_dim).contiguous()
        _ext_rope(q_idx[..., -self.rope_head_dim:], self.inv_freq_compress, position = pos0)
        wts = self.idx_weights.forward(x, params)
        scores = dsa_indexer_scores(q_idx[0], wts[0], idx_pool, pos0, self.compress_rate, ec,
                                    block_table = block_table, epp = epp)
        k = min(self.index_topk, ec)
        K_pad = -(-k // 32) * 32
        indices = torch.empty((seq, K_pad), dtype = torch.int32, device = x.device)
        ext.dsa_topk(scores, indices, k, None, 0)
        return indices, k


    def _forward_cached(self, x, params, out_dtype):
        """attn_mode flash_attn: kernel-based path over the per-job ring state and the paged
        pools. Each batch row appends to its slot's ring and, through its block-table row,
        to its pool pages, then attends over [sliding ring ++ selected/dense pool entries]
        via dsa_attn. Block-table rows correspond 1:1 to batch rows (generator batches and
        batch_shape identity tables alike)."""
        rsg = params["recurrent_states"]
        bsz = x.shape[0]
        assert len(rsg) >= bsz
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        kl = bt = None
        if self.compressor is not None:
            # In TP workers rs.cache is an opaque id; resolve to this rank's replicated
            # pool layer before anything dereferences it
            kl = self.tp_cache_lookup[rsg[0].cache] if self.tp_mode \
                else rsg[0].cache.layers[layer_instance]
            if "block_table" in params:
                bt = get_for_device(params, "block_table", self.device)
            else:
                # Direct-forward use without a page table (tests/benchmarks): slot-
                # partitioned identity, one row per batch row via each state's slot
                assert not self.tp_mode, "DSV4Attention: TP forward requires a block table"
                sbt = kl.slot_bt(rsg[0].cache.num_slots)
                bt = sbt[[rsg[i].slot for i in range(bsz)]]
            if dsa_debug_bounds:
                bmax, bmin = int(bt.max()), int(bt.min())
                assert 0 <= bmin and bmax < kl.num_pages, \
                    f"DSA block table content OOB: [{bmin}, {bmax}] vs {kl.num_pages} pages " \
                    f"(layer {self.layer_idx})"
        if bsz == 1:
            return self._forward_cached_one(
                x, params, rsg[0], self._get_rsl(rsg[0], layer_instance), out_dtype,
                kl = kl, bt_row = bt[:1] if bt is not None else None)
        if not dsv4_batch_eager:
            # Per-job loop: each job dispatches into its per-slot whole-step graph (or the
            # eager core). Measured faster than the batched-eager path below at bsz 2-8 --
            # graph replays beat batched GEMVs + per-job eager cores. The batched path is
            # the capture substrate for the planned bszN whole-batch graphs
            outs = [
                self._forward_cached_one(
                    x[i:i + 1], params, rsg[i],
                    self._get_rsl(rsg[i], layer_instance),
                    out_dtype, copy_static = True,
                    kl = kl, bt_row = bt[i:i + 1] if bt is not None else None)
                for i in range(bsz)
            ]
            return torch.cat(outs, dim = 0)
        return self._forward_cached_batch(x, params, rsg, layer_instance, out_dtype, kl, bt)


    def _get_rsl(self, rs, layer_instance):
        """Per-slot layer state: direct on the parent, via the id-keyed lookup in TP
        workers (rs is a DSV4ExportedState there and rs.cache an opaque id)."""
        if self.tp_mode:
            return self.tp_recurrent_lookup[rs.cache]
        return rs.cache.get_recurrent_layer(layer_instance)


    def _forward_cached_batch(self, x, params, rsg, layer_instance, out_dtype, kl, bt):
        """Multi-job cached path: the projections that don't touch per-slot state (x fan or
        Linears, q_norm, q_b / q-fan, fused-norm rope with per-job base positions) run ONCE
        over all rows, the per-slot attention core (ring/pools/indexer/attention) runs per
        job on row slices, and the grouped o_proj runs once over the collected outputs.
        Falls back to the per-job loop when the rows can't share a projection pass."""
        B, S, _ = x.shape
        eligible = (
            S <= 16 and x.dtype == torch.half and x.is_contiguous()
            and not any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        )
        if not eligible:
            outs = [
                self._forward_cached_one(x[i:i + 1], params, rsg[i],
                                         self._get_rsl(rsg[i], layer_instance),
                                         out_dtype, copy_static = True,
                                         kl = kl, bt_row = bt[i:i + 1] if bt is not None else None)
                for i in range(B)
            ]
            return torch.cat(outs, dim = 0)

        device = x.device
        R = B * S
        rsl = self._get_rsl(rsg[0], layer_instance)
        for rs in rsg[1:B]:
            assert self._get_rsl(rs, layer_instance) is rsl, \
                "batched DSA path requires all jobs in one cache"
        m = self.compress_rate if self.compressor is not None else 1
        w = self.sliding_window
        kp = -(-self.index_topk // 32) * 32 if self.indexer is not None else 32

        # Host prepass (never captured): ring shifts BEFORE attention (the body then
        # addresses the post-shift ring via the effective ring_beg; appends are in-body,
        # after attention reads), then the per-job state rows into a pinned staging write
        pos_l, floor_l, beg_l, ec_l, slot_l = [], [], [], [], []
        for i in range(B):
            rs = rsg[i]
            pos0 = rs.position
            slot = rs.slot
            offset = pos0 - rs.window_beg
            shift = 0
            if offset + S > rsl.ring_rows:
                need = offset + S - rsl.ring_rows
                shift = -(-need // PAGE_SIZE) * PAGE_SIZE
                ring_j = rsl.ring[slot]
                ring_j[:rsl.ring_rows - shift].copy_(ring_j[shift:].clone())
                rs.wshift = shift
            ec = (pos0 + S) // m if self.compressor is not None else 0
            assert self.compressor is None or ec <= bt.shape[1] * kl.epp, \
                f"DSA pool overflow: entry {ec} beyond block table ({bt.shape[1]} pages)"
            pos_l.append(pos0)
            floor_l.append(pos0 - min(w - 1, offset, pos0))
            beg_l.append(rs.window_beg + shift)
            ec_l.append(ec)
            slot_l.append(slot)

        # Whole-batch BC graph (house pattern): all per-job state device-driven through the
        # owner's (6, MAX_B) array and block-table static, the input pointer is the only
        # patched parameter. Declines (no fan / non-exl3 projections) fall through to the
        # eager batched body below
        if dsv4_batch_graph and B <= 8 and S <= 16 and R <= 32:
            if not hasattr(self, "_bc_dsa_batch"):
                self._bc_dsa_batch = {}
            bcd = self._bc_dsa_batch.get(id(rsl))
            if bcd is None:
                bcd = build_bc_dsa_batch(self, rsl, kl)
                self._bc_dsa_batch[id(rsl)] = bcd if bcd is not None else False
            if bcd:
                return bcd.run(x, B, S, pos_l, floor_l, beg_l, ec_l, slot_l, bt)

        # Eager batched body (capture reference / fallback): per-job state in a device
        # array, same kernels as the graphs
        gkey = (id(rsl), B, S)
        st = self._bgraph_state.get(gkey)
        if st is None:
            st = dict(
                # Pinned staging is a RING: the host may run several steps ahead of the
                # GPU, so a single pin would be rewritten while a previous step's async
                # H2D from it is still queued
                pins = [torch.empty((6, B), dtype = torch.int32, pin_memory = True)
                        for _ in range(8)],
                pin_i = 0,
                arr = torch.empty((6, B), dtype = torch.int32, device = device),
                bt_st = torch.zeros((B, kl.num_pages), dtype = torch.int32, device = device)
                    if kl is not None else None,
            )
            self._bgraph_state[gkey] = st
        pin = st["pins"][st["pin_i"]]
        st["pin_i"] = (st["pin_i"] + 1) % len(st["pins"])
        pin.copy_(torch.tensor([pos_l, floor_l, beg_l, ec_l, slot_l, [kp] * B],
                               dtype = torch.int32))
        st["arr"].copy_(pin, non_blocking = True)
        if kl is not None:
            npr = min(bt.shape[1], kl.num_pages)
            st["bt_st"][:, :npr].copy_(bt[:B, :npr])
        return self._batch_body(x, st["arr"], B, S, R, rsl, kl, st["bt_st"], m, w, kp,
                                params, out_dtype)

    def _batch_body(self, x, arr, B, S, R, rsl, kl, bt_st, m, w, kp, params, out_dtype):
        """Eager batched step, the capture reference for BC_DSV4BatchAttention: everything
        is device-driven (per-job state from `arr`, slot/block-table indirection
        in-kernel)."""
        from .attention_fn.dsa_triton import dsa_attn, _dsa_indexer_fewq_kernel
        import triton
        device = x.device
        a_pos, a_floor, a_beg, a_ec, a_slots, a_klen = arr.unbind(0)

        q_res, q, kv, comp_kv, comp_gate, idx_kv, idx_gate, q_idx = \
            self._project_batch(x, params, None, a_pos = a_pos)

        indices = None
        if self.compressor is not None:
            comp, idx = self.compressor, self.indexer
            ext.dsv4_compress(
                comp_kv, comp_gate, rsl.comp_buf_kv, rsl.comp_buf_gate,
                rsl.comp_ovl, comp.ape, comp.fused_norm_w, comp.norm.rms_norm_eps,
                comp.fused_inv_freq, kl.pool_c.view(-1, kl.D_c), kl.pool_r.view(-1, kl.D_r),
                0, a_pos, m, a_slots, bt_st, kl.epp)
            if self.layer_type == "csa":
                ext.dsv4_compress(
                    idx_kv, idx_gate, rsl.idx_buf_kv, rsl.idx_buf_gate,
                    rsl.idx_ovl, idx.ape, idx.fused_norm_w, idx.norm.rms_norm_eps,
                    idx.fused_inv_freq, kl.pool_idx.view(-1, kl.D_i), None,
                    0, a_pos, m, a_slots, bt_st, kl.epp)

        if self.compressor is not None and self.layer_type == "csa":
            if True:
                # Unified selection for ALL rows: for jobs still under index_topk the
                # bounded top-k IS the causal identity set (ascending) -- no dense/gathered
                # regime split
                Hi, Di = self.index_n_heads, self.index_head_dim
                qi = q_idx.view(B, S, Hi, Di)
                ext.rope(
                    qi[..., -self.rope_head_dim:], qi[..., -self.rope_head_dim:],
                    None, None, self.inv_freq_compress, 0, a_pos, None,
                    int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, 1, 0)
                wts = g_tensor_cache.get(device, (R, Hi), torch.half, "dsv4_b_wts")
                self.idx_weights.inner.bc.run(x.view(R, -1), wts)
                s_max = -(-kl.capacity // 128) * 128
                scores = g_tensor_cache.get(device, (R, s_max), torch.half, "dsv4_b_scores")
                with torch.cuda.device(device):
                    _dsa_indexer_fewq_kernel[(R, triton.cdiv(s_max, 128))](
                        qi.view(R, Hi, Di), wts.view(R, Hi), kl.pool_idx.view(-1, Di),
                        scores, a_ec, R, a_pos, a_ec, bt_st,
                        bt_st.stride(0),
                        H_i = Hi, H_pad = max(triton.next_power_of_2(Hi), 16), D_i = Di,
                        S_stride = s_max, compress_rate = m,
                        scale = Di ** -0.5 * Hi ** -0.5, BLOCK_N = 128,
                        SEQ = S, MULTIROW = 1, EPP = kl.epp,
                        DEBUG_BOUNDS = 1 if dsa_debug_bounds else 0,
                        DEBUG_PAGES = kl.num_pages if dsa_debug_bounds else 0,
                        num_warps = 8, num_stages = 2)
                indices = g_tensor_cache.get(device, (R, kp), torch.int32, "dsv4_b_idx")
                # Per-row scan bound from the state array: each row reads only its own
                # causal region, so the score buffer needs no -inf backfill
                ext.dsa_topk(scores, indices, self.index_topk, a_ec, S)

        # Paged pools: the split kernel reads one block-table row per job from the fixed
        # (B, num_pages) static
        if self.compressor is not None:
            bt = bt_st
            epp = kl.epp
            pool_c = kl.pool_c.view(-1, self.head_dim - self.rope_head_dim)
            pool_r = kl.pool_r.view(-1, self.rope_head_dim)
        else:
            epp = 256
            pool_c = g_tensor_cache.get(device, (1, self.head_dim - self.rope_head_dim),
                                        torch.half, "dsv4_b_pc0")
            pool_r = g_tensor_cache.get(device, (1, self.rope_head_dim), torch.half, "dsv4_b_pr0")
            bt = g_tensor_cache.get(device, (1, 1), torch.int32, "dsv4_b_bt0")

        hpg = self.num_q_heads // self.o_groups
        out = dsa_attn(
            q.view(R, self.num_q_heads, self.head_dim).half().contiguous(),
            pool_c, pool_r, bt, sinks = self.sinks,
            ring = rsl.ring, kv_chunk = kv.reshape(R, self.head_dim),
            win_len = w, indices = indices,
            compress_rate = m, scale = self.sm_scale,
            derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
            page_size = epp,
            out = g_tensor_cache.get(device, (self.o_groups, R, hpg * self.head_dim),
                                     torch.half, "dsv4_b_out"),
            multirow = dict(
                q_pos = a_pos, win_floor = a_floor, ring_beg = a_beg, pool_len = a_ec,
                k_len = a_klen, slot_ids = a_slots,
                ring_stride = rsl.ring_rows * self.head_dim, seq = S,
            ),
        )

        # Ring appends (shift already applied in the prepass): batched, slot-indexed
        ext.dsv4_ring_append(kv.reshape(R, self.head_dim), rsl.ring, a_pos, a_beg, a_slots)
        return self._project_o_grouped(out.unsqueeze(1), params, out_dtype,
                                       mgemm_out = True).view(B, S, -1)

    def _project_batch(self, x, params, rsg, a_pos = None):
        """One projection pass over all B * S rows for the batched cached path. Returns
        (q_res (1, R, q_lora), q (B, S, H, hd) roped, kv (B, S, 1, hd) roped+normed,
        comp_kv/gate (R, Wc), idx_kv/gate (R, Wi), q_idx (1, R, Hi * Di) or None), or None
        to decline (mixed-format fans handle their own decline; plain Linears always work)."""
        B, S, _ = x.shape
        rows = B * S
        if not self.x_fan_ready:
            self._build_x_fan()
        xf = x.view(1, rows, -1)
        comp_kv = comp_gate = idx_kv = idx_gate = q_idx = None
        need_q_idx = self.indexer is not None

        if self.x_fan is not None and rows <= 32:
            f = self.x_fan
            shapes = [(self.q_a.out_features,), (self.head_dim,)]
            if self.compressor is not None:
                shapes += [(self.compressor.wkv.out_features,)] * 2
            if self.indexer is not None:
                shapes += [(self.indexer.wkv.out_features,)] * 2
            fouts, cptr, ahad = self._fan_outs(f, "x", rows, shapes)
            ext.exl3_mgemm(
                xf, f["trellis"], fouts[0].view(1, rows, -1), f["suh"], ahad, f["svh"],
                f["idx"], None, f["K"], -1, f["mcg"], f["mul1"], -1, -1, 0, 1,
                f["n"], cptr)
            q_res = g_tensor_cache.get(x.device, (rows, self.q_a.out_features),
                                       torch.half, "dsv4_b_qres")
            ext.rms_norm(fouts[0], self.q_norm.weight, q_res, self.q_norm.rms_norm_eps,
                         self.q_norm.constant_bias, self.q_norm.constant_scale,
                         False, False)
            q_res = q_res.view(1, rows, -1)
            kv = fouts[1]
            if self.compressor is not None:
                comp_kv, comp_gate = fouts[2], fouts[3]
            if self.indexer is not None:
                idx_kv, idx_gate = fouts[4], fouts[5]
        else:
            q_res = self.q_norm.forward(self.q_a.forward(xf, params), params, out_dtype = torch.half)
            kv = self.wkv.forward(xf, params)[0]
            if self.compressor is not None:
                if not self.compressor.fused_ready:
                    self.compressor._build_fused()
                comp_kv = self.compressor.wkv.forward(xf, params)[0]
                comp_gate = self.compressor.wgate.forward(xf, params)[0]
            if self.indexer is not None:
                if not self.indexer.fused_ready:
                    self.indexer._build_fused()
                idx_kv = self.indexer.wkv.forward(xf, params)[0]
                idx_gate = self.indexer.wgate.forward(xf, params)[0]

        if need_q_idx and self.q_fan is not None and rows <= 32:
            f2 = self.q_fan
            o2, cptr2, ahad2 = self._fan_outs(f2, "q", rows, [
                (self.num_q_heads * self.head_dim,),
                (self.index_n_heads * self.index_head_dim,)
            ])
            ext.exl3_mgemm(
                q_res, f2["trellis"], o2[0].view(1, rows, -1), f2["suh"], ahad2, f2["svh"],
                f2["idx"], None, f2["K"], -1, f2["mcg"], f2["mul1"], -1, -1, 0, 1,
                f2["n"], cptr2)
            q = o2[0]
            q_idx = o2[1].view(1, rows, -1)
        elif self.qb_multi is not None and rows <= 32:
            q = self._mgemm1(self.qb_multi, q_res, torch.half, "dsv4_qb1_ah")
        else:
            q = self.q_b.forward(q_res, params)
            if need_q_idx:
                q_idx = self.idx_wq_b.forward(q_res, params)

        q = q.view(B, S, self.num_q_heads, self.head_dim)
        kv = kv.view(B, S, 1, self.head_dim)
        positions = a_pos if a_pos is not None else \
            torch.tensor([rs.position for rs in rsg[:B]], dtype = torch.int32,
                         device = x.device)
        ext.rope(
            q, q, kv, kv,
            self._rope_type(), 0, positions, None,
            int(RopeStyle.GPTJ), 1.0, self.q_ones, self.kv_norm_w,
            self.rms_norm_eps, 0.0, 0.0, 0, 1,
            self.head_dim - self.rope_head_dim,
        )
        return q_res, q, kv, comp_kv, comp_gate, idx_kv, idx_gate, q_idx


    def _forward_cached_one(
        self,
        x,
        params,
        rs,
        rsl,
        out_dtype,
        copy_static = False,
        pre = None,
        return_o = False,
        kl = None,
        bt_row = None,           # (1, npr) i32 device block-table row of this job
    ):
        _, seq, _ = x.shape

        # Whole-step graph path (EXL3_BC_DSA=1); not used when the batched path already
        # projected this job's rows (pre)
        if pre is None and \
                bc_dsa_enable and seq <= 16 and x.dtype == torch.half and x.is_contiguous():
            if not hasattr(self, "_bc_dsa"):
                self._bc_dsa = {}
            key = (id(rsl), rs.slot)
            bcd = self._bc_dsa.get(key)
            if bcd is None:
                bcd = build_bc_dsa(self, rs, rsl, kl)
                self._bc_dsa[key] = bcd if bcd is not None else False
            if bcd:
                y = bcd.run(x, rs, rsl, bt_row)
                if y is not None:
                    # y is a shared static, overwritten by the next slot's replay; batch
                    # rows are assembled only after all slots have run
                    return y.clone() if copy_static else y
        device = x.device
        pos0 = rs.position
        slot = rs.slot

        if not self.x_fan_ready:
            self._build_x_fan()

        converting = any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))
        use_fan = self.x_fan is not None and seq <= 32 and not converting
        ec = (pos0 + seq) // self.compress_rate if self.compressor is not None else 0
        topk_regime = self.indexer is not None and ec > self.index_topk
        q_idx_pre = None
        fouts = None
        if pre is not None:
            # Batched path already ran the shared projection pass (rope applied, per-job
            # positions); the compress section consumes the row slices directly
            q_res, q, kv = pre["q_res"], pre["q"], pre["kv"]
            fouts = [None, None, pre["comp_kv"], pre["comp_gate"], pre["idx_kv"], pre["idx_gate"]]
            q_idx_pre = pre["q_idx"]
            use_fan = True
        elif use_fan:

            # One per-matrix-N mgemm covers the whole x-side projection fan (q_a, wkv and
            # both compressors' kv/gate); a second one pairs q_b with idx_wq_b over q_res
            # in the top-k regime. Head norms fold into the rope kernel as in _project_qkv
            f = self.x_fan
            shapes = [(self.q_a.out_features,), (self.head_dim,)]
            if self.compressor is not None:
                shapes += [(self.compressor.wkv.out_features,)] * 2
            if self.indexer is not None:
                shapes += [(self.indexer.wkv.out_features,)] * 2
            fouts, cptr, ahad = self._fan_outs(f, "x", seq, shapes)
            ext.exl3_mgemm(
                x,
                f["trellis"],
                fouts[0].view(1, seq, -1),
                f["suh"],
                ahad,
                f["svh"],
                f["idx"],
                None,
                f["K"],
                -1,
                f["mcg"], f["mul1"],
                -1, -1, 0, 1,
                f["n"],
                cptr
            )
            q_res = self.q_norm.forward(fouts[0].view(1, seq, -1), params, out_dtype = torch.half)
            kv = fouts[1].view(1, seq, 1, self.head_dim)

            if topk_regime and self.q_fan is not None:
                f2 = self.q_fan
                o2, cptr2, ahad2 = self._fan_outs(f2, "q", seq, [
                    (self.num_q_heads * self.head_dim,),
                    (self.index_n_heads * self.index_head_dim,)
                ])
                ext.exl3_mgemm(
                    q_res, f2["trellis"], o2[0].view(1, seq, -1), f2["suh"], ahad2, f2["svh"],
                    f2["idx"], None, f2["K"], -1, f2["mcg"], f2["mul1"], -1, -1, 0, 1,
                    f2["n"], cptr2
                )
                q = o2[0].view(1, seq, self.num_q_heads, self.head_dim)
                q_idx_pre = o2[1]
            else:
                q = self.q_b.forward(q_res, params).view(1, seq, self.num_q_heads, self.head_dim)

            ext.rope(
                q, q, kv, kv,
                self._rope_type(), pos0, None, None,
                int(RopeStyle.GPTJ), 1.0, self.q_ones, self.kv_norm_w,
                self.rms_norm_eps, 0.0, 0.0, 0, 1,
                self.head_dim - self.rope_head_dim,
            )
            kv = kv.view(1, seq, self.head_dim)
        else:
            q_res, q, kv = self._project_qkv(x, params, pos0)

        # Window sources for the kernel: this chunk's kv rows plus prior rows read from the
        # ring at abs - window_beg; the kernel derives all per-query addressing from the
        # positions, so no temp/index tensors are built. The ring itself is updated AFTER
        # attention: the shift/rebase branches move rows the kernel reads in place
        w = self.sliding_window
        n_prev = min(w - 1, pos0 - rs.window_beg, pos0)
        win_floor = pos0 - n_prev
        ring = rsl.ring[slot]
        ring_beg = rs.window_beg

        indices = None
        k_len = 0
        pool_len = 0
        dense_m = 1
        epp = 256
        if self.compressor is not None:
            m = self.compress_rate
            dense_m = m
            epp = kl.epp
            assert ec <= bt_row.shape[1] * epp, \
                f"DSA pool overflow: entry {ec} beyond block table ({bt_row.shape[1]} pages)"

            # Fused compressor step: projections + window pooling + norm + rope, entries
            # written into the paged pools through the job's block table, ring/snapshot
            # state updated. With the fan the projections are already done: feed the
            # compress kernels directly
            pool_c_flat = kl.pool_c.view(-1, kl.D_c)
            pool_r_flat = kl.pool_r.view(-1, kl.D_r)
            if use_fan:
                comp = self.compressor
                ext.dsv4_compress(
                    fouts[2], fouts[3], rsl.comp_buf_kv[slot], rsl.comp_buf_gate[slot],
                    rsl.comp_ovl[slot] if rsl.comp_ovl is not None else None,
                    comp.ape, comp.fused_norm_w, comp.norm.rms_norm_eps, comp.fused_inv_freq,
                    pool_c_flat, pool_r_flat, pos0, None, m, None, bt_row, epp)
                if self.layer_type == "csa":
                    idx = self.indexer
                    ext.dsv4_compress(
                        fouts[4], fouts[5], rsl.idx_buf_kv[slot], rsl.idx_buf_gate[slot],
                        rsl.idx_ovl[slot], idx.ape, idx.fused_norm_w, idx.norm.rms_norm_eps,
                        idx.fused_inv_freq, kl.pool_idx.view(-1, kl.D_i), None, pos0,
                        None, m, None, bt_row, epp)
            else:
                self.compressor.forward_fused(
                    x, params, rsl.comp_buf_kv[slot], rsl.comp_buf_gate[slot],
                    rsl.comp_ovl[slot] if rsl.comp_ovl is not None else None,
                    pool_c_flat, pool_r_flat, pos0, bt_row, epp)
                if self.layer_type == "csa":
                    self.indexer.forward_fused(
                        x, params, rsl.idx_buf_kv[slot], rsl.idx_buf_gate[slot],
                        rsl.idx_ovl[slot], kl.pool_idx.view(-1, kl.D_i), None, pos0,
                        bt_row, epp)
            pool_len = ec

            # Selection is only non-trivial once the pool exceeds index_topk: below that,
            # top-k keeps every entry under the causal bound, which is exactly DENSE_POOL
            # mode. Indexer scoring chain is skipped (key pool is still maintained for later)
            if topk_regime:
                indices, k_len = self._indexer_topk(
                    x, params, q_res, kl.pool_idx.view(-1, kl.D_i), ec, pos0, q_idx_pre,
                    block_table = bt_row, epp = epp)

        if self.compressor is not None:
            pool_c, pool_r = kl.pool_c, kl.pool_r
            bt = bt_row
        else:
            pool_c = x.new_empty((1, self.head_dim - self.rope_head_dim), dtype = torch.half)
            pool_r = x.new_empty((1, self.rope_head_dim), dtype = torch.half)
            bt = torch.zeros((1, 1), dtype = torch.int32, device = device)

        # eq. 26 de-rotation and the group-major store for the grouped o_proj are fused into
        # the kernel epilogue: output is (G, seq, hpg * D), fp16
        hpg = self.num_q_heads // self.o_groups
        out = dsa_attn(
            q[0].half().contiguous(), pool_c, pool_r, bt, sinks = self.sinks,
            ring = ring, kv_chunk = kv[0], win_len = self.sliding_window,
            win_floor = win_floor, ring_beg = ring_beg,
            indices = indices, k_len = k_len, pool_len = pool_len, q_pos0 = pos0,
            compress_rate = dense_m, scale = self.sm_scale,
            derot_inv_freq = self._rope_type_neg(), groups = self.o_groups, group_major = True,
            page_size = epp,
            out = torch.empty((self.o_groups, seq, hpg * self.head_dim), dtype = torch.half, device = device),
        )

        # Ring update after attention: the shift/rebase branches move rows the kernel
        # reads in place (the kernel addresses the PRE-update ring via ring_beg). Keep the
        # trailing window resident for the next forward: in-place append while the chunk
        # fits; SWA-style page shift for small appends near the ring end; a window rebase
        # for chunks that overflow the ring outright. All layers compute the same wshift
        # for the same forward, so setting it is idempotent
        offset = pos0 - rs.window_beg
        pos_end = pos0 + seq
        if offset + seq <= rsl.ring_rows:
            ring[offset : offset + seq].copy_(kv[0])
        elif seq < PAGE_SIZE:
            need = offset + seq - rsl.ring_rows
            shift = -(-need // PAGE_SIZE) * PAGE_SIZE
            ring[:rsl.ring_rows - shift].copy_(ring[shift:].clone())
            rs.wshift = shift
            ring[offset - shift : offset - shift + seq].copy_(kv[0])
        else:
            new_beg = max(pos_end - (w - 1), 0) // PAGE_SIZE * PAGE_SIZE
            n_keep = pos_end - new_beg
            n_from_kv = min(n_keep, seq)
            n_from_ring = n_keep - n_from_kv          # rows below pos0 still needed
            assert 0 < n_keep <= rsl.ring_rows
            if n_from_ring > 0:
                src0 = pos0 - n_from_ring - ring_beg
                ring[:n_from_ring].copy_(ring[src0 : src0 + n_from_ring].clone())
            ring[n_from_ring : n_keep].copy_(kv[0, seq - n_from_kv:])
            rs.wshift = new_beg - rs.window_beg

        if return_o:
            return out
        return self._project_o_grouped(out.unsqueeze(1), params, out_dtype)

