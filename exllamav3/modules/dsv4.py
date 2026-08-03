from __future__ import annotations
from typing_extensions import override
import math
import torch
import os
import torch.nn.functional as F
from ..model.config import Config
from .module import Module
from .linear import Linear
from .rmsnorm import RMSNorm
from ..ext import exllamav3_ext as ext
from ..util.rope import RopeStyle, yarn_inv_freq
from ..util.tensor import g_tensor_cache
from .quant.exl3 import LinearEXL3
from ..cache.dsa import DSV4LayerState
from .multilinear import MultiLinear
from .attention_fn.dsa_triton import dsa_attn, dsa_indexer_scores
from .attention_fn.bc_dsa import bc_dsa_enable, build_bc_dsa
from ..constants import PAGE_SIZE

# Reference: transformers models/deepseek_v4 (paper §2)

def _ext_rope(x, inv_freq, position = 0, position_ids = None):
    # In-place GPT-J rotation of a (bsz, seq, heads, rope_dim) tensor via ext.rope. x may be
    # a trailing-slice VIEW of wider heads. De-rotation (paper eq. 26) uses a negated inv_freq
    # table. attn_factor is 1.0 by V4 semantics (yarn mscale never applied to cos/sin).
    ext.rope(
        x, x, None, None, inv_freq, position, None, position_ids,
        int(RopeStyle.GPTJ), 1.0, None, None, 1e-6, 0.0, 0.0, 0, False, 1, 0,
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
    ):
        cfg = attn.config
        proj_width = 2 * head_dim if overlapping else head_dim
        self.head_dim = head_dim
        self.rope_dim = attn.rope_head_dim if head_dim > attn.rope_head_dim else head_dim
        self.compress_rate = compress_rate
        self.overlapping = overlapping
        self.key = key
        self.wkv = Linear(
            cfg, f"{key}.wkv", attn.hidden_size, proj_width, qmap = qmap, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.wgate = Linear(
            cfg, f"{key}.wgate", attn.hidden_size, proj_width, qmap = qmap, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.norm = RMSNorm(cfg, f"{key}.norm", attn.rms_norm_eps)
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


    def forward_fused(self, x, params, buf_kv, buf_gate, ovl, dest_a, dest_b, position):
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
            self.bc.run(x[0], buf_kv, buf_gate, ovl, dest_a, dest_b, position, None, None)
        else:
            kv = self.wkv.forward(x, params)[0]
            gate = self.wgate.forward(x, params)[0]
            ext.dsv4_compress(
                kv, gate, buf_kv, buf_gate, ovl, self.ape, self.fused_norm_w,
                self.norm.rms_norm_eps, self.fused_inv_freq, dest_a, dest_b, position,
                None, self.compress_rate,
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

        self.q_a = Linear(
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
        self.q_norm = RMSNorm(config, f"{key}.q_norm", rms_norm_eps)
        self.q_b = Linear(
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
        self.wkv = Linear(
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
        self.kv_norm = RMSNorm(config, f"{key}.kv_norm", rms_norm_eps)

        group_width = num_q_heads * head_dim // o_groups
        self.wo_a = [
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
        self.wo_b = Linear(
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
        self.sinks = None  # (num_q_heads,) fp32

        self.compressor = None
        self.indexer = None
        self.idx_wq_b = None
        self.idx_weights = None
        if layer_type in ("csa", "hca"):
            self.compressor = DSV4Compressor(
                self,
                f"{key}.compressor",
                head_dim,
                compress_rate,
                overlapping = (layer_type == "csa"),
                qmap = qmap,
                select_hq_bits = select_hq_bits
            )
        if layer_type == "csa":
            self.indexer = DSV4Compressor(
                self,
                f"{key}.indexer.compressor",
                index_head_dim,
                compress_rate,
                overlapping = True,
                qmap = qmap,
                select_hq_bits = select_hq_bits
            )
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

        # Recurrent-state cache pattern (SWA/GDN): per-slot rings and pools, no paged KV
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = DSV4LayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}


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


    @override
    def unload(self):
        super().unload()
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
            self.rms_norm_eps, 0.0, 0.0, 0, False, 1, self.head_dim - rd,
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


    def _project_o_grouped(self, o, params, out_dtype):
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
            C = torch.empty((G, seq, mu.out_features), dtype = torch.half, device = o.device)
            ext.exl3_mgemm(
                A, mu.ptrs_trellis, C, mu.ptrs_suh, ah, mu.ptrs_svh,
                self.woa_indices, None, mu.K, -1, mu.mcg, mu.mul1,
                -1, -1, 0, 1, None, None
            )
            if seq == 1:
                o2 = C.view(1, 1, G * mu.out_features)
            else:
                o2 = C.permute(1, 0, 2).reshape(seq, G * mu.out_features).unsqueeze(0)
            return self.wo_b.forward(o2, params, out_dtype = out_dtype or self.out_dtype)

        o = torch.cat([self.wo_a[g].forward(o[g], params) for g in range(self.o_groups)], dim = -1)
        return self.wo_b.forward(o, params, out_dtype = out_dtype or self.out_dtype)


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        mode = params.get("attn_mode", "flash_attn_nc")
        if mode == "flash_attn":
            return self._forward_cached(x, params, out_dtype)
        assert mode == "flash_attn_nc", f"DSV4Attention: unsupported attn_mode {mode}"
        return self._forward_nc(x, params, out_dtype)


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
                derot_inv_freq = self._rope_type_neg(), groups = self.o_groups,
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
        q_idx_pre = None
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
        scores = dsa_indexer_scores(q_idx[0], wts[0], idx_pool, pos0, self.compress_rate, ec)
        k = min(self.index_topk, ec)
        K_pad = -(-k // 32) * 32
        indices = torch.empty((seq, K_pad), dtype = torch.int32, device = x.device)
        ext.dsa_topk(scores, indices, k)
        return indices, k


    def _forward_cached(self, x, params, out_dtype):
        """attn_mode flash_attn: kernel-based path over the per-job ring/pool state. Batch
        rows are processed per sequence (v1); each row appends to its slot's ring and pools,
        then attends over [sliding ring ++ selected/dense pool entries] via dsa_attn."""
        rsg = params["recurrent_states"]
        bsz = x.shape[0]
        assert len(rsg) >= bsz
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        outs = [
            self._forward_cached_one(
                x[i:i + 1],
                params,
                rsg[i],
                rsg[i].cache.get_recurrent_layer(layer_instance),
                out_dtype,
                copy_static = bsz > 1)
            for i in range(bsz)
        ]
        return torch.cat(outs, dim = 0) if bsz > 1 else outs[0]


    def _forward_cached_one(
        self,
        x,
        params,
        rs,
        rsl,
        out_dtype,
        copy_static = False
    ):
        _, seq, _ = x.shape

        # Whole-step graph path (EXL3_BC_DSA=1)
        if bc_dsa_enable and seq <= 16 and x.dtype == torch.half and x.is_contiguous():
            if not hasattr(self, "_bc_dsa"):
                self._bc_dsa = {}
            key = (id(rsl), rs.slot)
            bcd = self._bc_dsa.get(key)
            if bcd is None:
                bcd = build_bc_dsa(self, rs, rsl)
                self._bc_dsa[key] = bcd if bcd is not None else False
            if bcd:
                y = bcd.run(x, rs, rsl)
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
        if use_fan:

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
                self.rms_norm_eps, 0.0, 0.0, 0, False, 1,
                self.head_dim - self.rope_head_dim,
            )
            kv = kv.view(1, seq, self.head_dim)
        else:
            q_res, q, kv = self._project_qkv(x, params, pos0)

        # Window sources for the kernel: this chunk's kv rows plus prior rows read from the
        # ring at abs - window_beg; the kernel derives all per-query addressing from the
        # positions, so no temp/index tensors are built. The attention reads only rows below
        # pos0 from the ring, so the ring update below can happen in either order
        w = self.sliding_window
        n_prev = min(w - 1, pos0 - rs.window_beg, pos0)
        win_floor = pos0 - n_prev
        ring = rsl.ring[slot]
        ring_beg = rs.window_beg

        # Ring update: keep the trailing window resident for the next forward. In-place append
        # while the chunk fits; SWA-style page shift for small appends near the ring end; a
        # window rebase for chunks that overflow the ring outright. All layers compute the
        # same wshift for the same forward, so setting it is idempotent
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

        indices = None
        k_len = 0
        pool_len = 0
        dense_m = 1
        if self.compressor is not None:
            m = self.compress_rate
            dense_m = m
            assert ec <= rsl.pool_capacity, \
                f"DSA pool overflow: {ec} > {rsl.pool_capacity} (raise max_dsa_tokens)"

            # Fused compressor step: projections + window pooling + norm + rope, entries
            # written straight into the per-slot pools, ring/snapshot state updated. With
            # the fan the projections are already done: feed the compress kernels directly
            if use_fan:
                comp = self.compressor
                ext.dsv4_compress(
                    fouts[2], fouts[3], rsl.comp_buf_kv[slot], rsl.comp_buf_gate[slot],
                    rsl.comp_ovl[slot] if rsl.comp_ovl is not None else None,
                    comp.ape, comp.fused_norm_w, comp.norm.rms_norm_eps, comp.fused_inv_freq,
                    rsl.pool_c[slot], rsl.pool_r[slot], pos0, None, m)
                if self.layer_type == "csa":
                    idx = self.indexer
                    ext.dsv4_compress(
                        fouts[4], fouts[5], rsl.idx_buf_kv[slot], rsl.idx_buf_gate[slot],
                        rsl.idx_ovl[slot], idx.ape, idx.fused_norm_w, idx.norm.rms_norm_eps,
                        idx.fused_inv_freq, rsl.pool_idx[slot], None, pos0, None, m)
            else:
                self.compressor.forward_fused(
                    x, params, rsl.comp_buf_kv[slot], rsl.comp_buf_gate[slot],
                    rsl.comp_ovl[slot] if rsl.comp_ovl is not None else None,
                    rsl.pool_c[slot], rsl.pool_r[slot], pos0)
                if self.layer_type == "csa":
                    self.indexer.forward_fused(
                        x, params, rsl.idx_buf_kv[slot], rsl.idx_buf_gate[slot],
                        rsl.idx_ovl[slot], rsl.pool_idx[slot], None, pos0)
            pool_len = ec

            # Selection is only non-trivial once the pool exceeds index_topk: below that,
            # top-k keeps every entry under the causal bound, which is exactly DENSE_POOL
            # mode. Indexer scoring chain is skipped (key pool is still maintained for later)
            if topk_regime:
                indices, k_len = self._indexer_topk(
                    x, params, q_res, rsl.pool_idx[slot, :ec], ec, pos0, q_idx_pre)

        if self.compressor is not None:
            pool_c, pool_r = rsl.pool_c[slot], rsl.pool_r[slot]
            bt = rsl._identity_bt
            if bt is None or bt.device != device:
                bt = torch.arange(rsl.pool_capacity // PAGE_SIZE, dtype = torch.int32,
                                  device = device).unsqueeze(0)
                rsl._identity_bt = bt
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
            derot_inv_freq = self._rope_type_neg(), groups = self.o_groups,
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

        return self._project_o_grouped(out.unsqueeze(1), params, out_dtype)

