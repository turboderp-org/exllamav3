"""
Triton kernels for the QSA indexer stages of the graph-captured decode path (BC_Attention,
Qwen3.8-Flash-Next). The selection machinery is shared with the DSA/kpool stack: the raw-key
plane append reuses _mla_plane_update_kernel, scoring reuses _dsa_indexer_fewq_kernel over the
pooled plane (QSA's relu(q.k).sum(heads) * dk**-0.5 is the DSA weighted-relu score with uniform
head weights), top-k is the capture-safe radix top-k, and the block->token expansion reuses
_dsa_pool_expand_kernel. Its tail convention (the query's incomplete pool as raw tokens)
matches QSA's forced tail block.

QSA-specific:

  - _qsa_stage_kernel: split the fused index_qk_proj output into compact per-head RMS-normed
    queries and compact raw keys (the projection emits [q_heads || raw_k] in one row).
  - _qsa_pool_update_kernel: (re)build the pooled block keys touched by an append: fp32 mean
    over the block's raw keys, RMS norm, partial NEOX rope at the block START position, computed
    on the fly from inv_freq (no cached sin/cos tables). Partial blocks are written but never
    selected (the scoring bound admits only complete blocks), and the write that completes a
    block sees all its members. Every write is idempotent and capture/warmup-safe.
  - _qsa_sparse_split_kernel: gathered GQA flash-decoding over the selected token indices.
    Structure and partial layout follow _paged_attn_decode_split_kernel (one program per
    (batch, kv_head, h_block, split), same partial_o/partial_ml layout), so the standard
    _paged_attn_decode_combine_kernel reduces the splits unchanged. Causality lives in
    the selection: -1 indices mask out, everything else was emitted <= the query position.
    PAGED = 0 drops the block-table indirection: indices address flat (rows, kvh, hd) K/V
    directly (the nc path, where each "batch" is one query row of a from-scratch forward).

The sparse kernels assume one index list per query row (rows cannot share K/V tiles): the
paged/BC form runs decode rows (q_len == 1), the flat form any (B * S) row set. All bounds are
runtime arguments or derived on device, so the kernels are CUDA-graph-safe.
"""

import torch

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

if has_triton:

    @triton.jit(do_not_specialize = ["R"])
    def _qsa_stage_kernel(
        qk,                  # (R, (H_i + 1) * D) fp16 fused index_qk_proj output
        q_norm_w,            # (D,) fp16
        q_out,               # (R, H_i, D) fp16, RMS-normed (rope is applied afterwards)
        k_out,               # (R, D) fp16 raw keys, unnormed/unroped
        R,
        eps: tl.constexpr,
        H_i: tl.constexpr,
        D: tl.constexpr,
    ):
        """One program per (row, head): heads 0..H_i-1 are query heads (normed with the
        RMSNorm's +1 constant bias), head H_i is the raw key (plain copy-out)."""
        row = tl.program_id(0)
        h = tl.program_id(1)
        offs = tl.arange(0, D)
        x = tl.load(qk + row * ((H_i + 1) * D) + h * D + offs)
        if h < H_i:
            xf = x.to(tl.float32)
            var = tl.sum(xf * xf, axis = 0) / D
            y = xf * tl.rsqrt(var + eps) * (tl.load(q_norm_w + offs).to(tl.float32) + 1.0)
            tl.store(q_out + (row * H_i + h) * D + offs, y.to(tl.float16))
        else:
            tl.store(k_out + row * D + offs, x)


    @triton.jit(do_not_specialize = ["num_pages_per_row", "append_len"])
    def _qsa_pool_update_kernel(
        raw_plane,           # flat (pages * page_size, D) fp16 raw indexer keys
        pool_plane,          # flat (pages * page_size // P, D) fp16 pooled keys
        k_norm_w,            # (D,) fp16
        inv_freq,            # (ROPE_R // 2,) fp32
        block_table,         # (bsz, num_pages_per_row) i32
        cache_seqlens,       # (bsz,) i32, pre-append counts
        num_pages_per_row,
        append_len,
        page_size: tl.constexpr,
        P: tl.constexpr,
        D: tl.constexpr,
        ROPE_R: tl.constexpr,        # partial rotary width (the main attention's rotate_dims)
        attn_factor: tl.constexpr,
        eps: tl.constexpr,
        MAXPOOLS: tl.constexpr,      # grid height: append_len // P + 1
    ):
        """(Re)build the pooled keys of the blocks touched by this append: fp32 mean of the
        present raw keys, RMS norm (+1 constant bias), partial NEOX rope at the block start.
        The fp16 rounding points match the eager path (mean -> fp16, norm -> fp16, rope math in
        fp16 with fp16-rounded sin/cos). The three segments (rope lo/hi halves, pass-through)
        are separate register vectors so the NEOX pair rotation needs no in-register shuffle."""
        b = tl.program_id(0)
        pi = tl.program_id(1)
        pos0 = tl.load(cache_seqlens + b)
        t_end = pos0 + append_len
        pool = pos0 // P + pi
        if pool * P >= t_end:
            return

        HALF: tl.constexpr = ROPE_R // 2
        PASS: tl.constexpr = D - ROPE_R
        offs_h = tl.arange(0, HALF)
        bt = block_table + b * num_pages_per_row

        acc_lo = tl.zeros((HALF,), tl.float32)
        acc_hi = tl.zeros((HALF,), tl.float32)
        if PASS > 0:
            acc_ps = tl.zeros((PASS,), tl.float32)
            offs_p = ROPE_R + tl.arange(0, PASS)
        cnt = 0.0
        for j in range(P):
            tok = pool * P + j
            if tok < t_end:
                phys = tl.load(bt + tok // page_size)
                row = raw_plane + (phys * page_size + tok % page_size) * D
                acc_lo += tl.load(row + offs_h).to(tl.float32)
                acc_hi += tl.load(row + HALF + offs_h).to(tl.float32)
                if PASS > 0:
                    acc_ps += tl.load(row + offs_p).to(tl.float32)
                cnt += 1.0

        # fp32 mean -> fp16 (the norm reads the rounded values, like the eager path)
        m_lo = (acc_lo / cnt).to(tl.float16).to(tl.float32)
        m_hi = (acc_hi / cnt).to(tl.float16).to(tl.float32)
        ssq = tl.sum(m_lo * m_lo, axis = 0) + tl.sum(m_hi * m_hi, axis = 0)
        if PASS > 0:
            m_ps = (acc_ps / cnt).to(tl.float16).to(tl.float32)
            ssq += tl.sum(m_ps * m_ps, axis = 0)
        rstd = tl.rsqrt(ssq / D + eps)
        y_lo = (m_lo * rstd * (tl.load(k_norm_w + offs_h).to(tl.float32) + 1.0)).to(tl.float16)
        y_hi = (m_hi * rstd * (tl.load(k_norm_w + HALF + offs_h).to(tl.float32) + 1.0)).to(tl.float16)

        # Partial NEOX rope at the block start: pair (d, d + HALF) shares frequency d
        pos_f = (pool * P).to(tl.float32)
        fr = tl.load(inv_freq + offs_h) * pos_f
        cosv = (tl.cos(fr) * attn_factor).to(tl.float16)
        sinv = (tl.sin(fr) * attn_factor).to(tl.float16)
        r_lo = y_lo * cosv - y_hi * sinv
        r_hi = y_hi * cosv + y_lo * sinv

        phys0 = tl.load(bt + (pool * P) // page_size)
        prow = pool_plane + (phys0 * (page_size // P) + pool % (page_size // P)) * D
        tl.store(prow + offs_h, r_lo)
        tl.store(prow + HALF + offs_h, r_hi)
        if PASS > 0:
            y_ps = (m_ps * rstd * (tl.load(k_norm_w + offs_p).to(tl.float32) + 1.0)).to(tl.float16)
            tl.store(prow + offs_p, y_ps)


    @triton.jit(do_not_specialize = ["k_len", "num_pages_per_seq", "num_splits", "split_len"])
    def _qsa_sparse_split_kernel(
        q,                   # (bsz, 1, n_q_heads, head_dim) fp16, normed + roped
        k_cache,             # fp16 (pages, page_size, n_kv_heads, head_dim)
        v_cache,
        block_table,
        indices,             # (bsz, K_pad) i32 selected token positions, -1 padded
        partial_o,
        partial_ml,
        k_len,               # runtime: valid width of the index rows
        num_pages_per_seq,
        num_splits,
        split_len,
        n_q_heads: tl.constexpr,
        n_kv_heads: tl.constexpr,
        page_size: tl.constexpr,
        head_dim: tl.constexpr,
        K_pad: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        PAGED: tl.constexpr = 1,
    ):
        """Gathered GQA flash-decoding phase 1 over an index list, q_len == 1: one program per
        (batch, kv_head, h_block, split), iterating the row's indices instead of the sequential
        kv range. Same partial layout as _paged_attn_decode_split_kernel, so the standard
        combine kernel reduces the splits."""
        pid = tl.program_id(0)
        split = tl.program_id(1)

        group_size = n_q_heads // n_kv_heads
        h_blocks = tl.cdiv(group_size, BLOCK_H)
        h_block = pid % h_blocks
        bh = pid // h_blocks
        batch = bh // n_kv_heads
        kv_head = bh - batch * n_kv_heads

        rows = tl.arange(0, BLOCK_H)
        row_h_local = h_block * BLOCK_H + rows
        q_head = kv_head * group_size + row_h_local
        valid_row = row_h_local < group_size

        offs_d = tl.arange(0, head_dim)
        q_base = (batch * n_q_heads + q_head) * head_dim
        q_tile = tl.load(q + q_base[:, None] + offs_d[None, :], mask = valid_row[:, None], other = 0.0)

        n_start = split * split_len
        n_end = tl.minimum(n_start + split_len, k_len)

        m = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l = tl.full((BLOCK_H,), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_H, head_dim), tl.float32)

        for n0 in range(n_start, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            idx = tl.load(indices + batch * K_pad + offs_n, mask = offs_n < n_end, other = -1)
            valid_n = (idx >= 0) & (offs_n < n_end)
            idx_c = tl.where(valid_n, idx, 0)
            if PAGED:
                phys = tl.load(block_table + batch * num_pages_per_seq + idx_c // page_size,
                               mask = valid_n, other = 0)
                tok = phys * page_size + idx_c % page_size
            else:
                tok = idx_c

            k_ptrs = k_cache + ((tok[None, :] * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
            k_tile = tl.load(k_ptrs, mask = valid_n[None, :], other = 0.0)
            scores = tl.dot(q_tile, k_tile) * scale

            valid = valid_row[:, None] & valid_n[None, :]
            scores = tl.where(valid, scores, -float("inf"))

            m_new = tl.maximum(m, tl.max(scores, axis = 1))
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(scores - m_exp[:, None])
            p = tl.where(valid, p, 0.0)
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
            l = l * alpha + tl.sum(p, axis = 1)

            v_ptrs = v_cache + ((tok[:, None] * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
            v_tile = tl.load(v_ptrs, mask = valid_n[:, None], other = 0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
            m = m_new

        if split < num_splits:
            po_base = (pid * num_splits + split) * BLOCK_H * head_dim
            tl.store(partial_o + po_base + rows[:, None] * head_dim + offs_d[None, :], acc)
            ml_base = (pid * num_splits + split) * BLOCK_H * 2
            tl.store(partial_ml + ml_base + rows * 2, m)
            tl.store(partial_ml + ml_base + rows * 2 + 1, l)


    _sm_counts = {}

    def _get_sms(dev):
        if dev.index not in _sm_counts:
            _sm_counts[dev.index] = torch.cuda.get_device_properties(dev.index).multi_processor_count
        return _sm_counts[dev.index]


    def qsa_sparse_attend_rows(
        q: torch.Tensor,               # (R, n_q_heads, head_dim) fp16, normed + roped
        k: torch.Tensor,               # (rows, n_kv_heads, head_dim) fp16 (paged: flat cache view)
        v: torch.Tensor,
        indices: torch.Tensor,         # (R, K_pad) int32, -1 padded: flat k/v row indices, or
                                       # per-sequence cache positions in the paged form
        sm_scale: float,
        block_table: torch.Tensor | None = None,   # (R, num_pages) int32, one row per query row
        page_size: int = 0,
    ) -> torch.Tensor:
        """Eager gathered GQA attention, one index list per query row: the sparse prefill /
        eager-fallback form of the BC sparse decode kernels. block_table = None runs the flat
        (non-paged) variant over contiguous K/V (the nc path); with a block table, indices are
        cache positions mapped through it (the cached path). Splits size to the grid, so few
        rows (decode fallback) still fill the device. Returns (R, n_q_heads, head_dim) fp16."""
        from .triton_paged import _paged_attn_decode_combine_kernel
        R, H, hd = q.shape
        kvh = k.shape[1]
        group = H // kvh
        BLOCK_H = 16
        BLOCK_N = 32
        h_blocks = triton.cdiv(group, BLOCK_H)
        programs = R * kvh * h_blocks
        K_pad = indices.shape[1]
        dev = q.device
        paged = block_table is not None
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() \
            and indices.is_contiguous()
        assert not paged or (block_table.is_contiguous() and block_table.shape[0] == R)

        splits = max(1, min(2 * _get_sms(dev) // programs, -(-K_pad // (4 * BLOCK_N)), 128))
        per_split = -(-K_pad // splits)
        split_len = -(-per_split // BLOCK_N) * BLOCK_N
        partial_o = torch.empty((programs * splits * BLOCK_H * hd,), dtype = torch.float, device = dev)
        partial_ml = torch.empty((programs * splits * BLOCK_H * 2,), dtype = torch.float, device = dev)
        o = torch.empty((R, H, hd), dtype = torch.half, device = dev)

        # Triton JIT launches on the CURRENT device; with a model split across GPUs this
        # layer's device need not be current
        with torch.cuda.device(dev):
            _qsa_sparse_split_kernel[(programs, splits)](
                q, k, v, block_table if paged else indices, indices, partial_o, partial_ml,
                K_pad, block_table.shape[1] if paged else 0, splits, split_len,
                n_q_heads = H, n_kv_heads = kvh, page_size = page_size if paged else 1,
                head_dim = hd, K_pad = K_pad, scale = float(sm_scale),
                BLOCK_H = BLOCK_H, BLOCK_N = BLOCK_N, PAGED = 1 if paged else 0,
                num_warps = 4, num_stages = 2,
            )
            _paged_attn_decode_combine_kernel[(programs,)](
                partial_o, partial_ml, o, partial_ml, splits, partial_ml,
                QCV = 0, HAS_SINKS = False, q_len = 1, n_q_heads = H, n_kv_heads = kvh,
                head_dim = hd, BLOCK_M = 1, BLOCK_H = BLOCK_H, BLOCK_ROWS = BLOCK_H,
                num_warps = 4, num_stages = 1,
            )
        return o
