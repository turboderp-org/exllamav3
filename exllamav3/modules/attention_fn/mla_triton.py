"""
Triton kernels for multi-head latent attention (DeepSeek-V2/V3, Kimi-Linear).

The cache holds the compressed latent and the shared RoPE key, one "kv head" each:

    ckv  (num_pages, page_size, 1, kv_lora_rank)      e.g. 512
    kpe  (num_pages, page_size, 1, qk_rope_head_dim)  e.g. 64

Attention runs in absorbed form throughout: queries are multiplied by W_UK before they get here,
so they live in the latent space and nothing per-head is ever materialized for K or V.

    scores = q_lat @ ckv^T + q_pe @ kpe^T
    o_lat  = softmax(scale * scores) @ ckv

Three differences from the regular paged kernels in triton_paged.py:

  - The QK contraction spans kv_lora_rank + qk_rope_head_dim, which is not a power of two (576 for
    every current model) and so cannot index with a single tl.arange. It runs as two tl.dots
    accumulating into one score tile, which also keeps the RoPE half in its own tensor where it can
    stay fp16 while the latent half is quantized later.
  - V *is* K. In fp16 the latent tile is loaded once in the (BLOCK_N, D_c) orientation and
    transposed for the score dot; the quantized path instead runs the shared plane loaders from
    triton_paged.py once per orientation -- still fewer DRAM bytes than one fp16 load at any width.
  - There is one kv head, so the head axis carries the tl.dot M dimension in the decode kernel and
    BLOCK_H (heads per program) decides whether the kernel is compute- or bandwidth-bound.

The quantized cache (CacheLayer_MLA_quant) reuses the whole MHA cache-quant stack: the packed
format is exactly CacheLayer_quant's (32-value groups, absmax midpoint grid, power-of-two bit
planes, values stored in the H32-rotated domain), written by the same CUDA quantizer
(quant_cache_cont) and read by the same Triton plane loaders (_qc_load_kt/_qc_load_v with one kv
head). The H32 fold works as in the MHA kernels: q_lat is pre-rotated in-kernel (32-blocks never
cross the 512-wide latent's group boundaries), scores come out exact because H32 is orthogonal,
and the PV accumulator lives in the rotated domain until a single post-rotation at the final
normalization (split partials stay rotated; the combine kernel rotates). The RoPE key is small
and shared by every head, so it stays fp16 unconditionally.

Everything else follows triton_paged.py: cache_seqlens is read on the device, the split count and
block-table width are runtime arguments, and the grid derives from allocation-time constants, so
these kernels stay CUDA-graph-safe.
"""

import math
import os

import torch

# Debug aid: synchronize and error-check after every MLA kernel launch, so an async illegal
# memory access is attributed to the kernel that caused it instead of a later sync point
_debug_sync = os.environ.get("EXL3_MLA_DEBUG_SYNC", "0") != "0"

# Temporary debug overrides for the qc decode config (unset = tuned defaults)
_qc_bn_override = int(os.environ.get("EXL3_MLA_QC_BN", "0") or "0")
_qc_trans_override = os.environ.get("EXL3_MLA_QC_TRANS")

def _dbg_sync(tag, device):
    # NOTE: must sync the device the kernel launched on -- torch.cuda.synchronize() without an
    # argument syncs the ambient current device, which in a multi-device layer split is usually
    # NOT the layer's device, and the attribution becomes meaningless
    if _debug_sync:
        try:
            torch.cuda.synchronize(device)
        except Exception as e:
            raise RuntimeError(f"MLA debug sync failed after {tag} on {device}: {e}") from e

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False
    triton = None
    tl = None


if has_triton:

    from .triton_paged import _qc_load_kt, _qc_load_v, _rot_h32

    @triton.jit
    def _mla_kv_quant_scatter_kernel(
        tmp_q,               # (rows, N_G * BITS) int32, packed by quant_cache_cont
        tmp_s,               # (rows, N_G) fp16 group scales
        kpe_new,             # (rows, D_r) fp16
        qk,                  # (pages, page_size, N_G * BITS) int32
        sk,                  # (pages, page_size, N_G) fp16
        kpe_cache,           # (pages, page_size, 1, D_r) fp16
        block_table,
        cache_seqlens,
        num_pages_per_seq,
        append_len: tl.constexpr,
        page_size: tl.constexpr,
        W_TOT: tl.constexpr,     # N_G * BITS packed words per row
        W_PAD: tl.constexpr,     # next pow2 (tl.arange needs it; W_TOT = 48/80/96/112 for odd widths)
        N_G: tl.constexpr,
        D_r: tl.constexpr,
    ):
        """Scatter contiguously quantized latent rows plus their fp16 rope keys into the paged
        cache. Quantization itself runs in the CUDA kernel (same packed format as the MHA cache);
        this only places the rows."""
        row = tl.program_id(0)
        batch = row // append_len
        pos = row - batch * append_len

        abs_pos = tl.load(cache_seqlens + batch) + pos
        page = abs_pos // page_size
        page_off = abs_pos - page * page_size
        phys = tl.load(block_table + batch * num_pages_per_seq + page)
        tok = phys * page_size + page_off

        w = tl.arange(0, W_PAD)
        mask_w = w < W_TOT
        tl.store(qk + tok * W_TOT + w, tl.load(tmp_q + row * W_TOT + w, mask = mask_w), mask = mask_w)
        g = tl.arange(0, N_G)
        tl.store(sk + tok * N_G + g, tl.load(tmp_s + row * N_G + g))
        r = tl.arange(0, D_r)
        tl.store(kpe_cache + tok * D_r + r, tl.load(kpe_new + row * D_r + r))


    @triton.jit
    def _mla_kv_update_kernel(
        ckv_new,             # (bsz, len, D_c)
        kpe_new,             # (bsz, len, D_r)
        ckv_cache,           # (pages, page_size, D_c)
        kpe_cache,           # (pages, page_size, D_r)
        block_table,
        cache_seqlens,
        num_pages_per_seq,
        append_len: tl.constexpr,
        page_size: tl.constexpr,
        D_c: tl.constexpr,
        D_r: tl.constexpr,
    ):
        """Append one chunk of latent + rope rows to the paged cache. Both widths move in one
        launch; the ext fp16 append kernel requires K and V to be the same shape, which they are
        not here."""
        row = tl.program_id(0)
        batch = row // append_len
        pos = row - batch * append_len

        abs_pos = tl.load(cache_seqlens + batch) + pos
        page = abs_pos // page_size
        page_off = abs_pos - page * page_size
        phys = tl.load(block_table + batch * num_pages_per_seq + page)
        tok = phys * page_size + page_off

        offs_c = tl.arange(0, D_c)
        tl.store(ckv_cache + tok * D_c + offs_c,
                 tl.load(ckv_new + row * D_c + offs_c))
        offs_r = tl.arange(0, D_r)
        tl.store(kpe_cache + tok * D_r + offs_r,
                 tl.load(kpe_new + row * D_r + offs_r))


    @triton.jit
    def _mla_decode_split_kernel(
        q_lat,               # (H, bsz * q_len, D_c) absorbed queries, head-major
        q_pe,                # (H, bsz * q_len, D_r)
        ckv_cache,           # fp16 latent pages, or packed int32 when QC > 0
        kpe_cache,           # always fp16
        ckv_scales,          # fp16 group scales when QC > 0 (dummy otherwise)
        h32,                 # 32x32 Hadamard/sqrt(32) when QC > 0 (dummy otherwise)
        block_table,
        cache_seqlens,
        out,                 # (H, bsz * q_len, D_c) latent output, head-major
        partial_o,
        partial_ml,
        split_len,           # runtime: derived from the block-table bound, changes as it grows
        num_pages_per_seq,   # runtime: block-table width can grow without recompiling
        num_splits,          # runtime: the grid may be launched wider; extra splits idle
        QC: tl.constexpr,    # latent cache bits (2-8), 0 = fp16
        QC_TRANS: tl.constexpr,  # 0: expand both orientations; 1: expand V once, trans for K^T
        bsz: tl.constexpr,
        q_len: tl.constexpr,
        pre_appended_len: tl.constexpr,
        n_q_heads: tl.constexpr,
        page_size: tl.constexpr,
        D_c: tl.constexpr,
        D_r: tl.constexpr,
        scale: tl.constexpr,
        CAUSAL: tl.constexpr,
        FINAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Flash-decoding phase 1: one program per (batch, head block, kv split). Query positions
        and sibling heads share the row axis so the latent tile is read once per program."""
        pid = tl.program_id(0)
        split = tl.program_id(1)

        h_blocks = tl.cdiv(n_q_heads, BLOCK_H)
        h_block = pid % h_blocks
        batch = pid // h_blocks

        rows = tl.arange(0, BLOCK_ROWS)
        row_q = rows % BLOCK_M
        row_h = h_block * BLOCK_H + (rows // BLOCK_M)
        valid_row = (row_q < q_len) & (row_h < n_q_heads)

        offs_c = tl.arange(0, D_c)
        offs_r = tl.arange(0, D_r)

        # Head-major: the absorbed queries arrive straight from a batched GEMM over heads, and the
        # latent output feeds the next one, so neither D_c-wide tensor is ever permuted
        q_row = row_h * (bsz * q_len) + batch * q_len + row_q
        q_c = tl.load(q_lat + q_row[:, None] * D_c + offs_c[None, :],
                      mask = valid_row[:, None], other = 0.0)
        q_r = tl.load(q_pe + q_row[:, None] * D_r + offs_r[None, :],
                      mask = valid_row[:, None], other = 0.0)
        if QC > 0:
            # Packed values live in the rotated domain; rotating q here keeps the score dot exact
            # (H32 is orthogonal and block-diagonal per 32, so dot(Hq, Hk) == dot(q, k))
            q_c = _rot_h32(q_c, h32, BLOCK_ROWS, D_c)

        total_k_len = tl.load(cache_seqlens + batch) + pre_appended_len
        q_abs = total_k_len - q_len + row_q

        n_start = split * split_len
        n_end = tl.minimum(n_start + split_len, total_k_len)

        m = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
        l = tl.full((BLOCK_ROWS,), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_ROWS, D_c), tl.float32)

        for n0 in range(n_start, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            in_range = offs_n < n_end
            page = offs_n // page_size
            page_off = offs_n - page * page_size
            phys = tl.load(block_table + batch * num_pages_per_seq + page, mask = in_range, other = 0)
            tok = phys * page_size + page_off

            if QC > 0:
                # V is K in the packed cache too: both orientations come from the same words.
                # Either expand twice (QC_TRANS 0) or expand the V orientation once and transpose
                # the dequantized fp16 tile for the score dot (QC_TRANS 1) -- the transpose is of
                # a plain fp16 value, not of loader interleave output (the miscompiling shape)
                if QC_TRANS:
                    v_tile = _qc_load_v(ckv_cache, ckv_scales, tok, 0, offs_c, in_range, QC, 1, D_c)
                    kt = tl.trans(v_tile)
                else:
                    kt = _qc_load_kt(ckv_cache, ckv_scales, tok, 0, offs_c, in_range, QC, 1, D_c)
                    v_tile = _qc_load_v(ckv_cache, ckv_scales, tok, 0, offs_c, in_range, QC, 1, D_c)
            else:
                # V is K: load the latent tile once, transpose it for the score dot
                v_tile = tl.load(ckv_cache + tok[:, None] * D_c + offs_c[None, :],
                                 mask = in_range[:, None], other = 0.0)
                kt = tl.trans(v_tile)
            kt_pe = tl.load(kpe_cache + tok[None, :] * D_r + offs_r[:, None],
                            mask = in_range[None, :], other = 0.0)

            scores = tl.dot(q_c, kt)
            scores = tl.dot(q_r, kt_pe, acc = scores) * scale

            valid = valid_row[:, None] & in_range[None, :]
            if CAUSAL:
                valid = valid & (offs_n[None, :] <= q_abs[:, None])
            scores = tl.where(valid, scores, -float("inf"))

            m_new = tl.maximum(m, tl.max(scores, axis = 1))
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(scores - m_exp[:, None])
            p = tl.where(valid, p, 0.0)
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
            l = l * alpha + tl.sum(p, axis = 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m = m_new

        if FINAL:
            out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
            if QC > 0:
                # The accumulator is in the rotated domain (rotated V); one inverse rotation at
                # the end recovers the latent (H32 is symmetric, so the same helper serves)
                out_tile = _rot_h32(out_tile, h32, BLOCK_ROWS, D_c)
            tl.store(out + q_row[:, None] * D_c + offs_c[None, :], out_tile,
                     mask = valid_row[:, None])
        else:
            if split < num_splits:
                po_base = (pid * num_splits + split) * BLOCK_ROWS * D_c
                tl.store(partial_o + po_base + rows[:, None] * D_c + offs_c[None, :], acc)
                ml_base = (pid * num_splits + split) * BLOCK_ROWS * 2
                tl.store(partial_ml + ml_base + rows * 2, m)
                tl.store(partial_ml + ml_base + rows * 2 + 1, l)


    @triton.jit
    def _mla_decode_combine_kernel(
        partial_o,
        partial_ml,
        out,
        h32,
        num_splits,
        QC: tl.constexpr,
        bsz: tl.constexpr,
        q_len: tl.constexpr,
        n_q_heads: tl.constexpr,
        D_c: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Flash-decoding phase 2: reduce the per-split partial accumulators."""
        pid = tl.program_id(0)
        h_blocks = tl.cdiv(n_q_heads, BLOCK_H)
        h_block = pid % h_blocks
        batch = pid // h_blocks

        rows = tl.arange(0, BLOCK_ROWS)
        row_q = rows % BLOCK_M
        row_h = h_block * BLOCK_H + (rows // BLOCK_M)
        valid_row = (row_q < q_len) & (row_h < n_q_heads)
        offs_c = tl.arange(0, D_c)

        m_max = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
        for s in range(num_splits):
            ml_base = (pid * num_splits + s) * BLOCK_ROWS * 2
            m_max = tl.maximum(m_max, tl.load(partial_ml + ml_base + rows * 2))

        m_safe = tl.where(m_max == -float("inf"), 0.0, m_max)
        l_sum = tl.zeros((BLOCK_ROWS,), tl.float32)
        acc = tl.zeros((BLOCK_ROWS, D_c), tl.float32)
        for s in range(num_splits):
            ml_base = (pid * num_splits + s) * BLOCK_ROWS * 2
            m_s = tl.load(partial_ml + ml_base + rows * 2)
            l_s = tl.load(partial_ml + ml_base + rows * 2 + 1)
            w = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_safe))
            po_base = (pid * num_splits + s) * BLOCK_ROWS * D_c
            acc += tl.load(partial_o + po_base + rows[:, None] * D_c + offs_c[None, :]) * w[:, None]
            l_sum += l_s * w

        out_tile = acc / tl.where(l_sum[:, None] == 0.0, 1.0, l_sum[:, None])
        if QC > 0:
            # Split partials stay in the rotated domain; the single inverse rotation happens here
            out_tile = _rot_h32(out_tile, h32, BLOCK_ROWS, D_c)
        q_row = row_h * (bsz * q_len) + batch * q_len + row_q
        tl.store(out + q_row[:, None] * D_c + offs_c[None, :], out_tile, mask = valid_row[:, None])


    @triton.jit
    def _mla_prefill_kernel(
        q_lat,
        q_pe,
        ckv_cache,
        kpe_cache,
        ckv_scales,
        h32,
        block_table,
        cache_seqlens,
        out,
        num_pages_per_seq,
        q_len,
        n_rows,
        QC: tl.constexpr,
        QC_TRANS: tl.constexpr,
        pre_appended_len: tl.constexpr,
        n_q_heads: tl.constexpr,
        page_size: tl.constexpr,
        D_c: tl.constexpr,
        D_r: tl.constexpr,
        scale: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Long-query MLA attention, still absorbed. One program per (q block, batch, head); the
        accumulator is D_c wide rather than a normal head dim, so BLOCK_M has to stay small."""
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        batch = bh // n_q_heads
        head = bh - batch * n_q_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_row = offs_m < q_len

        offs_c = tl.arange(0, D_c)
        offs_r = tl.arange(0, D_r)

        q_row = head * n_rows + batch * q_len + offs_m
        q_c = tl.load(q_lat + q_row[:, None] * D_c + offs_c[None, :],
                      mask = valid_row[:, None], other = 0.0)
        q_r = tl.load(q_pe + q_row[:, None] * D_r + offs_r[None, :],
                      mask = valid_row[:, None], other = 0.0)
        if QC > 0:
            q_c = _rot_h32(q_c, h32, BLOCK_M, D_c)

        past = tl.load(cache_seqlens + batch) + pre_appended_len - q_len
        total_k_len = past + q_len
        q_abs = past + offs_m

        n_end = total_k_len
        if CAUSAL:
            # No kv tile past the last query row of this block can contribute
            n_end = tl.minimum(n_end, past + (pid_m + 1) * BLOCK_M)

        m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l = tl.full((BLOCK_M,), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_M, D_c), tl.float32)

        for n0 in range(0, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            in_range = offs_n < n_end
            page = offs_n // page_size
            page_off = offs_n - page * page_size
            phys = tl.load(block_table + batch * num_pages_per_seq + page, mask = in_range, other = 0)
            tok = phys * page_size + page_off

            if QC > 0:
                if QC_TRANS:
                    v_tile = _qc_load_v(ckv_cache, ckv_scales, tok, 0, offs_c, in_range, QC, 1, D_c)
                    kt = tl.trans(v_tile)
                else:
                    kt = _qc_load_kt(ckv_cache, ckv_scales, tok, 0, offs_c, in_range, QC, 1, D_c)
                    v_tile = _qc_load_v(ckv_cache, ckv_scales, tok, 0, offs_c, in_range, QC, 1, D_c)
            else:
                v_tile = tl.load(ckv_cache + tok[:, None] * D_c + offs_c[None, :],
                                 mask = in_range[:, None], other = 0.0)
                kt = tl.trans(v_tile)
            kt_pe = tl.load(kpe_cache + tok[None, :] * D_r + offs_r[:, None],
                            mask = in_range[None, :], other = 0.0)

            scores = tl.dot(q_c, kt)
            scores = tl.dot(q_r, kt_pe, acc = scores) * scale

            valid = valid_row[:, None] & in_range[None, :]
            if CAUSAL:
                valid = valid & (offs_n[None, :] <= q_abs[:, None])
            scores = tl.where(valid, scores, -float("inf"))

            m_new = tl.maximum(m, tl.max(scores, axis = 1))
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(scores - m_exp[:, None])
            p = tl.where(valid, p, 0.0)
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
            l = l * alpha + tl.sum(p, axis = 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m = m_new

        out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
        if QC > 0:
            out_tile = _rot_h32(out_tile, h32, BLOCK_M, D_c)
        tl.store(out + q_row[:, None] * D_c + offs_c[None, :], out_tile, mask = valid_row[:, None])


_sm_count = {}


def _get_sm_count(device: torch.device) -> int:
    idx = device.index if hasattr(device, "index") else device
    if idx not in _sm_count:
        _sm_count[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _sm_count[idx]


def mla_kv_append(
    ckv_new: torch.Tensor,      # (bsz, length, D_c)
    kpe_new: torch.Tensor,      # (bsz, length, D_r)
    ckv_cache: torch.Tensor,    # (pages, page_size, 1, D_c)
    kpe_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
):
    bsz, length, D_c = ckv_new.shape
    D_r = kpe_new.shape[-1]
    page_size = ckv_cache.shape[1]
    if length == 0:
        return
    with torch.cuda.device(ckv_new.device):
        _mla_kv_update_kernel[(bsz * length,)](
            ckv_new, kpe_new, ckv_cache, kpe_cache, block_table, cache_seqlens,
            block_table.shape[1], length, page_size, D_c, D_r,
            num_warps = 4, num_stages = 2,
        )
    _dbg_sync("mla_kv_append", ckv_new.device)


def mla_kv_quant_append(
    ckv_new: torch.Tensor,      # (bsz, length, D_c) fp16
    kpe_new: torch.Tensor,      # (bsz, length, D_r) fp16
    qk: torch.Tensor,           # (pages, page_size, D_c // 32 * bits) int32
    sk: torch.Tensor,           # (pages, page_size, D_c // 32) fp16
    kpe_cache: torch.Tensor,    # (pages, page_size, 1, D_r) fp16
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    bits: int,
    scratch: dict | None = None,
):
    """Quantize new latent rows straight into the paged cache and copy their rope keys.

    The quantization runs through the same CUDA kernel as the MHA quantized cache
    (quant_cache_cont: H32 rotation, absmax midpoint grid, bit-plane packing), into contiguous
    row temporaries that a small Triton kernel then scatters to pages. That keeps the packed
    format bit-identical to CacheLayer_quant's, so the shared plane loaders read it as-is."""
    from ...ext import exllamav3_ext as ext
    bsz, length, D_c = ckv_new.shape
    D_r = kpe_new.shape[-1]
    page_size = qk.shape[1]
    if length == 0:
        return
    rows = bsz * length
    groups = D_c // 32
    w_tot = groups * bits

    key = (rows, bits, groups, D_r)
    buf = scratch.get(key) if scratch is not None else None
    if buf is None:
        buf = (
            torch.empty((rows, w_tot), dtype = torch.int, device = ckv_new.device),
            torch.empty((rows, groups), dtype = torch.half, device = ckv_new.device),
        )
        if scratch is not None:
            scratch[key] = buf
    tmp_q, tmp_s = buf

    _dbg_sync(f"upstream-of-append rows={rows} (not an MLA kernel)", ckv_new.device)
    ext.quant_cache_cont(ckv_new.reshape(rows, D_c).contiguous(), tmp_q, tmp_s, 0.0)
    _dbg_sync(f"quant_cache_cont rows={rows} bits={bits}", ckv_new.device)
    with torch.cuda.device(ckv_new.device):
        _mla_kv_quant_scatter_kernel[(rows,)](
            tmp_q, tmp_s, kpe_new.reshape(rows, D_r).contiguous(),
            qk, sk, kpe_cache, block_table, cache_seqlens,
            block_table.shape[1], length, page_size,
            w_tot, triton.next_power_of_2(w_tot), groups, D_r,
            num_warps = 2, num_stages = 2,
        )
    _dbg_sync("mla_kv_quant_scatter", ckv_new.device)


def mla_attn_triton_decode(
    q_lat: torch.Tensor,        # (H, bsz * q_len, D_c) fp16, absorbed, head-major
    q_pe: torch.Tensor,         # (H, bsz * q_len, D_r) fp16, head-major
    ckv_cache: torch.Tensor,    # (pages, page_size, 1, D_c)
    kpe_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    bsz: int,
    q_len: int,
    causal: bool = True,
    softmax_scale: float | None = None,
    pre_appended_len: int = 0,
    out: torch.Tensor | None = None,
    block_h: int | None = None,
    block_n: int | None = None,
    num_splits: int | None = None,
    max_kv_len: int | None = None,
    scratch: dict | None = None,
    qc: tuple | None = None,    # (scales, bits): ckv_cache is the packed int32 tensor
    qc_trans: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    n_q_heads, n_rows, D_c = q_lat.shape
    D_r = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    assert n_rows == bsz * q_len

    if qc is not None:
        from .triton_paged import _get_h32
        ckv_scales, qc_bits = qc
        assert ckv_cache.shape[-1] == D_c // 32 * qc_bits
        h32 = _get_h32(q_lat.device)
        if _qc_trans_override is not None:
            qc_trans = _qc_trans_override != "0"
    else:
        ckv_scales, qc_bits, h32 = q_lat, 0, q_lat

    if out is None:
        out = torch.empty_like(q_lat)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_c + D_r)

    block_m = triton.next_power_of_2(q_len)
    if block_h is None:
        # 16 rows fills the tl.dot M axis; more heads per program means fewer redundant passes
        # over the latent cache, which is what keeps large-head-count models compute-bound
        block_h = max(16 // block_m, 1)
    block_h = min(block_h, max(1, triton.next_power_of_2(n_q_heads)))
    block_rows = block_m * block_h
    if block_n is None:
        # qc: one big expanded tile per iteration, no pipelining (the expansion is ALU work the
        # scheduler overlaps anyway); swept best on Ampere/Ada/Blackwell at 512-wide latents.
        # fp16: smaller tiles, deeper pipeline
        block_n = (32 if D_c <= 512 else 16) if qc is None else (_qc_bn_override or 64)
    if num_stages is None:
        num_stages = 3 if qc is None else 1
    if num_warps is None:
        num_warps = 4 if qc is None else 8
    h_blocks = triton.cdiv(n_q_heads, block_h)
    num_pages_per_seq = block_table.shape[1]

    max_k_len = num_pages_per_seq * page_size
    if max_kv_len is not None:
        max_k_len = min(max_k_len, max_kv_len)

    programs = bsz * h_blocks
    if num_splits is None:
        target = 2 * _get_sm_count(q_lat.device)
        num_splits = max(1, min(target // programs, triton.cdiv(max_k_len, 4 * block_n), 128))
    split_len = triton.cdiv(triton.cdiv(max_k_len, num_splits), block_n) * block_n

    if num_splits > 1:
        n_o = programs * num_splits * block_rows * D_c
        n_ml = programs * num_splits * block_rows * 2
        if scratch is None:
            partial_o = torch.empty(n_o, dtype = torch.float32, device = q_lat.device)
            partial_ml = torch.empty(n_ml, dtype = torch.float32, device = q_lat.device)
        else:
            buf = scratch.get(n_o)
            if buf is None:
                buf = scratch[n_o] = (
                    torch.empty(n_o, dtype = torch.float32, device = q_lat.device),
                    torch.empty(n_ml, dtype = torch.float32, device = q_lat.device),
                )
            partial_o, partial_ml = buf
    else:
        partial_o = partial_ml = q_lat

    if _debug_sync:
        _dbg_sync("upstream-of-decode (not an MLA kernel)", q_lat.device)
        _dbg_snap = (cache_seqlens.cpu().tolist(), block_table.cpu())
    with torch.cuda.device(q_lat.device):
        _mla_decode_split_kernel[(programs, num_splits)](
            q_lat, q_pe, ckv_cache, kpe_cache, ckv_scales, h32, block_table, cache_seqlens, out,
            partial_o, partial_ml,
            split_len, num_pages_per_seq, num_splits,
            qc_bits, bool(qc_trans), bsz, q_len, pre_appended_len, n_q_heads, page_size, D_c, D_r, float(softmax_scale),
            bool(causal), num_splits == 1,
            block_m, block_h, block_rows, block_n,
            num_warps = num_warps, num_stages = num_stages,
        )
        try:
            _dbg_sync("mla_decode_split", q_lat.device)
        except RuntimeError:
            print(f" !! mla_decode_split fault: grid=({programs},{num_splits}) bsz={bsz} q_len={q_len} "
                  f"H={n_q_heads} D_c={D_c} D_r={D_r} qc_bits={qc_bits} qc_trans={qc_trans} "
                  f"block m/h/rows/n={block_m}/{block_h}/{block_rows}/{block_n} "
                  f"split_len={split_len} npps={num_pages_per_seq} max_k_len={max_k_len} "
                  f"warps={num_warps} stages={num_stages} dev={q_lat.device} "
                  f"pre_app={pre_appended_len} causal={causal} "
                  f"cache={tuple(ckv_cache.shape)} kpe={tuple(kpe_cache.shape)} "
                  f"bt={tuple(block_table.shape)} out={tuple(out.shape)}")
            # The device context is poisoned, but host-side copies made BEFORE the launch may
            # still be readable on some driver states; try to dump the invariants
            try:
                sl, bt = _dbg_snap
                print(f" !! seqlens={sl} (+pre_app={pre_appended_len}; capacity npps*ps={num_pages_per_seq * page_size})")
                print(f" !! bt min={int(bt.min())} max={int(bt.max())} total_pages={ckv_cache.shape[0]}")
                print(f" !! invariant seqlen: {all(x + pre_appended_len <= num_pages_per_seq * page_size for x in sl)}")
                used = [bt[i, : (sl[i] + pre_appended_len + page_size - 1) // page_size] for i in range(bsz)]
                print(f" !! used-page ids in range: "
                      f"{all(int(u.min()) >= 0 and int(u.max()) < ckv_cache.shape[0] for u in used if u.numel())}")
                print(f" !! used pages per row: {[u.tolist() for u in used]}")
            except Exception as e2:
                print(f" !! dump failed: {e2}")
            raise
        if num_splits > 1:
            _mla_decode_combine_kernel[(programs,)](
                partial_o, partial_ml, out, h32, num_splits,
                qc_bits, bsz, q_len, n_q_heads, D_c, block_m, block_h, block_rows,
                num_warps = 4, num_stages = 1,
            )
            _dbg_sync("mla_decode_combine", q_lat.device)
    return out


def mla_attn_triton_prefill(
    q_lat: torch.Tensor,        # (H, bsz * q_len, D_c) head-major
    q_pe: torch.Tensor,         # (H, bsz * q_len, D_r) head-major
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    bsz: int,
    q_len: int,
    causal: bool = True,
    softmax_scale: float | None = None,
    pre_appended_len: int = 0,
    out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    qc: tuple | None = None,    # (scales, bits): ckv_cache is the packed int32 tensor
    qc_trans: bool = True,
    num_warps: int = 8,
    num_stages: int | None = None,
) -> torch.Tensor:
    n_q_heads, n_rows, D_c = q_lat.shape
    D_r = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    assert n_rows == bsz * q_len

    if qc is not None:
        from .triton_paged import _get_h32
        ckv_scales, qc_bits = qc
        assert ckv_cache.shape[-1] == D_c // 32 * qc_bits
        h32 = _get_h32(q_lat.device)
    else:
        ckv_scales, qc_bits, h32 = q_lat, 0, q_lat

    if block_m is None:
        block_m = 32 if qc is None else 16
    if block_n is None:
        block_n = 32 if qc is None else 64
    if num_stages is None:
        num_stages = 2 if qc is None else 1

    if out is None:
        out = torch.empty_like(q_lat)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D_c + D_r)

    with torch.cuda.device(q_lat.device):
        _mla_prefill_kernel[(triton.cdiv(q_len, block_m), bsz * n_q_heads)](
            q_lat, q_pe, ckv_cache, kpe_cache, ckv_scales, h32, block_table, cache_seqlens, out,
            block_table.shape[1], q_len, n_rows,
            qc_bits, bool(qc_trans), pre_appended_len, n_q_heads, page_size, D_c, D_r, float(softmax_scale),
            bool(causal), block_m, block_n,
            num_warps = num_warps, num_stages = num_stages,
        )
    _dbg_sync("mla_prefill", q_lat.device)
    return out
