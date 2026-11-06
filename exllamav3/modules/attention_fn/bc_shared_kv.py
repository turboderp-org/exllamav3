"""
Native decode path for SharedKVAttention (query-only attention over another model's cache): q_proj, fused
q_norm/RoPE, the paged flash-decoding kernels and o_proj as one C++ call per step, replayed as one CUDA graph per
(bsz, q_len) slot after a warmup run, or issued stage by stage where graphs are unavailable. The split kernel is
compiled with no appended tokens: the source cache already holds every key the query attends to.

Mirrors bc_attn.BCAttn; statics come from g_tensor_cache under their own tags and are shared by the drafter's
layers of the same geometry.
"""
from __future__ import annotations
import torch
from ...ext import exllamav3_ext as ext
from ...util.tensor import g_tensor_cache
from ...util.backend import attn_decode_config
from ...constants import PAGE_SIZE
from .bc_attn import (
    bc_attn_enable, MAX_BSZ, MAX_QLEN, MAX_R, BCKernelTooLarge, _compile_kernel, _get_sm_count, _trace_build,
)


class BCSharedKVAttn:

    def __init__(self, module, cache_k: torch.Tensor, cache_v: torch.Tensor):
        self.module = module
        self.device = torch.device(module.device)
        self.head_dim = module.head_dim
        self.num_q_heads = module.num_q_heads
        self.num_kv_heads = module.num_kv_heads
        self.hidden_size = module.hidden_size
        self.hidden_padded = module.q_proj.in_features
        self.sm_scale = module.sm_scale
        self.window_size = module.sliding_window
        self.softcap = module.logit_softcapping
        self.cache_k, self.cache_v = cache_k, cache_v
        self.o_dtype = module.o_proj.inner.default_out_dtype
        self.slot_keys = {}
        rope = module.rope
        w = max(self.hidden_padded, self.num_q_heads * self.head_dim)
        xh = g_tensor_cache.get(self.device, (2 * MAX_R * w,), torch.half, "bcs_xh")
        h32 = g_tensor_cache.get(self.device, (1,), torch.half, "bcs_dummy")
        self.bc = ext.BC_SharedKVAttention(
            num_q_heads = self.num_q_heads,
            num_kv_heads = self.num_kv_heads,
            head_dim = self.head_dim,
            hidden_size = self.hidden_size,
            hidden_size_padded = self.hidden_padded,
            page_size = PAGE_SIZE,
            q_proj = module.q_proj.inner.bc,
            o_proj = module.o_proj.inner.bc,
            q_norm = module.q_norm.weight.data if module.q_norm is not None else None,
            norm_eps = module.q_norm.rms_norm_eps if module.q_norm is not None else 1e-6,
            norm_constant_bias = module.q_norm.constant_bias if module.q_norm is not None else 0.0,
            inv_freq = rope.inv_freq,
            rope_style = int(rope.rope_settings.rope_style),
            attn_factor = rope.attn_factor,
            l4_scaling_beta = rope.llama_4_scaling_beta,
            l4_scaling_original = rope.llama_4_scaling_original,
            rotate_dims = rope.rope_settings.rotate_dims,
            cache_k = cache_k,
            cache_v = cache_v,
            xh = xh,
            h32 = h32,
        )


    def _configure(self, bsz: int, q_len: int):
        import triton
        from .triton_paged import (
            combine_subtiles, decode_row_layout, _paged_attn_decode_split_kernel,
            _paged_attn_decode_combine_kernel, _normalize_window,
        )
        dev = self.device
        hd = self.head_dim
        hd_pad = triton.next_power_of_2(hd)
        qh, kvh = self.num_q_heads, self.num_kv_heads
        block_n, split_warps, split_stages = attn_decode_config(dev, hd_pad)
        block_rows, h_blocks = decode_row_layout(q_len, qh // kvh, hd_pad)
        programs = bsz * kvh * h_blocks
        splits_cap = max(1, min(2 * _get_sm_count(dev) // programs, 128))
        window_left, window_right = _normalize_window(self.window_size)
        sig = {
            "q": "*fp16:16", "k_cache": "*fp16:16", "v_cache": "*fp16:16",
            "block_table": "*i32:16", "cache_seqlens": "*i32", "out": "*fp16:16",
            "partial_o": "*fp32:16", "partial_ml": "*fp32:16",
            "k_scales": "*fp16:16", "v_scales": "*fp16:16", "h32": "*fp16:16",
            "split_len": "i32", "num_pages_per_seq": "i32", "num_splits": "i32",
            "sinks": "*fp32",
        } | {n: "constexpr" for n in (
            "QCK", "QCV", "q_len", "kv_append_len", "n_q_heads", "n_kv_heads",
            "page_size", "head_dim", "HD_PAD", "scale", "CAUSAL", "WINDOW_LEFT", "WINDOW_RIGHT",
            "SOFTCAP", "FINAL", "HAS_SINKS", "BLOCK_ROWS", "BLOCK_N")}
        # kv_append_len 0: the query attends to the cached keys only, nothing of its own is appended
        consts = dict(
            QCK = 0, QCV = 0,
            q_len = q_len, kv_append_len = 0, n_q_heads = qh, n_kv_heads = kvh,
            page_size = PAGE_SIZE, head_dim = hd, HD_PAD = hd_pad, scale = float(self.sm_scale),
            CAUSAL = True, WINDOW_LEFT = window_left, WINDOW_RIGHT = window_right,
            SOFTCAP = float(self.softcap or 0.0), FINAL = False, HAS_SINKS = False,
            BLOCK_ROWS = block_rows, BLOCK_N = block_n,
        )
        k_split = _compile_kernel(dev, _paged_attn_decode_split_kernel, sig, consts, split_warps, split_stages)
        sig_c = {
            "partial_o": "*fp32:16", "partial_ml": "*fp32:16", "out": "*fp16:16", "h32": "*fp16:16",
            "num_splits": "i32", "sinks": "*fp32",
        } | {n: "constexpr" for n in (
            "QCV", "HAS_SINKS", "q_len", "n_q_heads", "n_kv_heads", "head_dim", "HD_PAD", "V_DIM",
            "BLOCK_ROWS", "ROWS_SUB", "D_SUB")}
        rows_sub, d_sub = combine_subtiles(block_rows, hd_pad)
        consts_c = dict(
            QCV = 0, HAS_SINKS = False, q_len = q_len,
            n_q_heads = qh, n_kv_heads = kvh, head_dim = hd, HD_PAD = hd_pad, V_DIM = hd,
            BLOCK_ROWS = block_rows, ROWS_SUB = rows_sub, D_SUB = d_sub,
        )
        k_combine = _compile_kernel(dev, _paged_attn_decode_combine_kernel, sig_c, consts_c, 4, 1)
        k_combine.grid_y = (block_rows // rows_sub) * (hd_pad // d_sub)
        R = bsz * q_len
        q = g_tensor_cache.get(dev, (bsz, q_len, qh, hd), torch.half, "bcs_q")
        o = g_tensor_cache.get(dev, (bsz, q_len, qh, hd), torch.half, "bcs_o")
        partial_o = g_tensor_cache.get_bucketed(dev, programs * splits_cap * block_rows * hd_pad, torch.float, "bcs_po")
        partial_ml = g_tensor_cache.get_bucketed(dev, programs * splits_cap * block_rows * 2, torch.float, "bcs_ml")
        xp = yp = None
        if self.hidden_padded != self.hidden_size:
            xp = g_tensor_cache.get(dev, (R, self.hidden_padded), torch.half, "bcs_xp")
            xp.zero_()
            yp = g_tensor_cache.get(dev, (R, self.hidden_padded), self.o_dtype or torch.half, "bcs_yp")
        self.bc.configure_slot(
            bsz, q_len, q, o, partial_o, partial_ml, k_split, k_combine, block_n, splits_cap, programs, xp, yp,
        )


    def step(
        self,
        x: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        position: int,
        positions: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        bsz, q_len, _ = x.shape
        if (bsz, q_len) not in self.slot_keys:
            try:
                self._configure(bsz, q_len)
            except BCKernelTooLarge:
                return None
            self.slot_keys[(bsz, q_len)] = True
        assert block_table.data_ptr() % 16 == 0, \
            "BC_SharedKVAttention: block_table must be 16-byte aligned (pass a whole tensor, not a sliced view)"
        y = torch.empty((bsz, q_len, self.hidden_size), dtype = self.o_dtype, device = x.device)
        self.bc.run(bsz, q_len, x, y, cache_seqlens, block_table, position, positions, position_ids)
        return y


def build_bc_shared_kv(module, cache_k: torch.Tensor, cache_v: torch.Tensor):
    """A BCSharedKVAttn for the module over the given source pages, or None when unsupported (the module then
    takes its python path): quantized q/o projections with native kernels, RoPE present (q_norm is fused into
    the rope stage), fp16 source pages on the module's device"""
    m = module
    dev = torch.device(m.device)
    ok = (
        bc_attn_enable and
        m.rope is not None and
        m.q_proj.quant_type == "exl3" and m.q_proj.inner.bc is not None and
        m.o_proj.quant_type == "exl3" and m.o_proj.inner.bc is not None and
        (m.q_norm is None or (type(m.q_norm).__name__ == "RMSNorm" and m.q_norm.weight is not None)) and
        cache_k.dtype == torch.float16 and cache_v.dtype == torch.float16 and
        cache_k.device == dev and cache_v.device == dev and
        cache_k.is_contiguous() and cache_v.is_contiguous()
    )
    if not ok:
        _trace_build(m, None, "shared_kv")
        return None
    bca = BCSharedKVAttn(m, cache_k, cache_v)
    _trace_build(m, bca, "shared_kv")
    return bca
