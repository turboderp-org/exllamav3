"""
Production DSA (DeepSeek Sparse Attention) Triton kernels.

Constexpr toggles on the core sparse attention kernel:

  HAS_WINDOW   include a sliding-window phase over raw K=V rows before the gathered phase
               (V4: window 128 from the ring; V3.2-on-MLA: off)
  HAS_SINKS    per-head learnable sink logit folded into the softmax state init (gpt-oss
               style; V4: on)
  DENSE_POOL   attend to ALL pool entries with a per-query causal entry bound instead of a
               gathered index list (V4 HCA; also the short-context fast path for CSA when
               the pool is smaller than index_topk)
  DEROTATE     eq. 26 output de-rotation fused into the epilogue: the output's trailing
               D_r slice is GPT-J-rotated at the query's absolute position using a caller
               frequency table (pass the NEGATED table for de-rotation). One program = one
               query row, so this is a single theta vector broadcast over heads
  HPG          heads-per-group > 0 stores the output group-major, (H / HPG, R, HPG * D),
               so each group's wo_a GEMM reads a contiguous slice with no host permute
               (V4 grouped output projection); 0 = token-major (R, H, D)

The score is always two-dot: D_c-wide nope part + D_r-wide rope part, V IS K. For V4 the
cache rows are (448 nope | 64 rotated rope) split across two tensors exactly like the MLA
ckv/kpe layout; for V3.2-on-MLA they are (512 latent | 64 rope) and the same kernel serves
with DEROTATE=0, HAS_WINDOW=0.

The lightning-indexer scoring kernel is shared as-is between raw-token (V3.2) and pooled
(V4 CSA) keys -- only the key tensor differs.
"""

import os
import torch
from ...util.tensor import g_tensor_cache

# EXL3_DSA_DEBUG_BOUNDS=1: compile the JIT DSA kernels with device-side bounds asserts on
# every block-table page read and gathered pool index (names the kernel and traps at the
# bad index instead of faulting downstream). Triton only emits device_assert when kernels
# compile in debug mode, so force TRITON_DEBUG before any compilation
dsa_debug_bounds = os.environ.get("EXL3_DSA_DEBUG_BOUNDS", "0") != "0"

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False


if has_triton:

    @triton.jit(do_not_specialize = [
        "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0", "R",
        "win_floor", "ring_beg",
    ], debug = dsa_debug_bounds)
    def _dsa_attn_kernel(
        q,                   # (R, H, D_c + D_r) fp16 token-major, rope slice pre-rotated
        ring,                # (ring_rows, D_c + D_r) fp16: rows at abs < q_pos0, linearly
                             # addressed as abs - ring_beg (HAS_WINDOW)
        kv_chunk,            # (R, D_c + D_r) fp16: this chunk's K = V rows, abs >= q_pos0
        pool_c,              # paged pool, nope part: (pages * page_size, D_c) fp16
        pool_r,              # paged pool, rope part: (pages * page_size, D_r) fp16
        block_table,         # (R, num_pages_per_row) int32
        indices,             # (R, K_pad) int32 pool-entry indices, -1 padded (not DENSE_POOL)
        sinks,               # (H,) fp32 (HAS_SINKS)
        derot_inv_freq,      # (D_r // 2,) fp32 epilogue frequency table (DEROTATE)
        out,                 # (R, H, D_c + D_r) fp16, or (H / HPG, R, HPG * D) when HPG > 0
        k_len,               # runtime: valid gathered entries per row
        win_len,             # runtime: sliding window width
        pool_len,            # runtime: pool entry count (DENSE_POOL bound base)
        num_pages_per_row,   # runtime
        q_pos0,              # runtime: absolute position of query row 0 (window/DENSE_POOL
                             # bounds and DEROTATE query position base)
        R,                   # runtime: query row count (HPG group stride)
        win_floor,           # runtime: lowest absolute position visible to the window
        ring_beg,            # runtime: absolute position of ring row 0
        H: tl.constexpr,
        page_size: tl.constexpr,
        D_c: tl.constexpr,
        D_c_pad: tl.constexpr,
        D_r: tl.constexpr,
        K_pad: tl.constexpr,
        compress_rate: tl.constexpr,   # DENSE_POOL causal bound: entry w < (pos + 1) // m
        scale: tl.constexpr,
        HAS_WINDOW: tl.constexpr,
        HAS_SINKS: tl.constexpr,
        DENSE_POOL: tl.constexpr,
        DEROTATE: tl.constexpr,
        HPG: tl.constexpr,             # heads per output group; 0 = token-major store
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_W: tl.constexpr,         # window tile; smaller than BLOCK_N (two-source smem)
        DEBUG_BOUNDS: tl.constexpr = 0,
        DEBUG_PAGES: tl.constexpr = 0,
        NC_BLOCK: tl.constexpr = 0,    # DSpark draft mode: every row sees the SAME range
                                       # [win_floor, q_pos0 + R) (window history ++ whole
                                       # chunk, non-causal); history rows are PAGED, read
                                       # via block_table at absolute-position pages
    ):
        """One program per (query row, head block); heads are the MMA M dim. Consecutive
        programs cover one query's head blocks so gathers stay L2-resident. Two KV phases:
        the sliding ring (dense, short) and the pool (gathered via per-query index list, or
        dense with a causal entry bound when DENSE_POOL)."""
        pid = tl.program_id(0)
        h_blocks = tl.cdiv(H, BLOCK_H)
        row = pid // h_blocks
        h_block = pid % h_blocks

        offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        valid_h = offs_h < H
        offs_c = tl.arange(0, D_c_pad)
        valid_c = offs_c < D_c
        offs_r = tl.arange(0, D_r)
        D = D_c + D_r

        q_base = q + (row * H + offs_h)[:, None] * D
        qc = tl.load(q_base + offs_c[None, :], mask = valid_h[:, None] & valid_c[None, :], other = 0.0)
        qr = tl.load(q_base + D_c + offs_r[None, :], mask = valid_h[:, None], other = 0.0)

        if HAS_SINKS:
            sink = tl.load(sinks + offs_h, mask = valid_h, other = -float("inf"))
            m_state = sink
            l = tl.full((BLOCK_H,), 1.0, tl.float32)
        else:
            m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
            l = tl.zeros((BLOCK_H,), tl.float32)
        acc_c = tl.zeros((BLOCK_H, D_c_pad), tl.float32)
        acc_r = tl.zeros((BLOCK_H, D_r), tl.float32)

        # Phase 1: sliding-window rows, addressed by absolute position: query row sees
        # positions [q_abs - win_len + 1, q_abs] clipped to win_floor; rows at abs >= q_pos0
        # come from this chunk's kv rows, older rows from the ring at abs - ring_beg
        if HAS_WINDOW:
            q_abs = q_pos0 + row
            if NC_BLOCK:
                top = q_pos0 + R - 1
            else:
                top = q_abs
            for n0 in tl.range(0, win_len, BLOCK_W, num_stages = 1):
                offs_j = n0 + tl.arange(0, BLOCK_W)
                abs_pos = top - offs_j
                in_range = (offs_j < win_len) & (abs_pos >= win_floor)
                mc = in_range & (abs_pos >= q_pos0)
                mr = in_range & (abs_pos < q_pos0)
                idx_c = tl.where(mc, abs_pos - q_pos0, 0)
                if NC_BLOCK:
                    ap = tl.where(mr, abs_pos, 0)
                    w_phys = tl.load(block_table + row * num_pages_per_row + ap // page_size,
                                     mask = mr, other = 0)
                    idx_r = w_phys * page_size + ap % page_size
                else:
                    idx_r = tl.where(mr, abs_pos - ring_beg, 0)
                vc = tl.load(kv_chunk + idx_c[:, None] * D + offs_c[None, :],
                             mask = mc[:, None] & valid_c[None, :], other = 0.0) \
                   + tl.load(ring + idx_r[:, None] * D + offs_c[None, :],
                             mask = mr[:, None] & valid_c[None, :], other = 0.0)
                vr = tl.load(kv_chunk + idx_c[:, None] * D + D_c + offs_r[None, :],
                             mask = mc[:, None], other = 0.0) \
                   + tl.load(ring + idx_r[:, None] * D + D_c + offs_r[None, :],
                             mask = mr[:, None], other = 0.0)
                scores = tl.dot(qc, tl.trans(vc))
                scores = tl.dot(qr, tl.trans(vr), acc = scores) * scale
                scores = tl.where(in_range[None, :], scores, -float("inf"))
                m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
                m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
                p = tl.exp(scores - m_exp[:, None])
                p = tl.where(in_range[None, :], p, 0.0)
                alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
                l = l * alpha + tl.sum(p, axis = 1)
                pv = p.to(vc.dtype)
                acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
                acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
                m_state = m_new

        # Phase 2: pool entries -- gathered by index list, or dense with causal bound
        if DENSE_POOL:
            bound = (q_pos0 + row + 1) // compress_rate
            n_end = tl.minimum(bound, pool_len)
        else:
            n_end = k_len
        for n0 in range(0, n_end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            if DENSE_POOL:
                idx = tl.where(offs_n < n_end, offs_n, -1)
            else:
                idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
            in_range = idx >= 0
            idx_s = tl.where(in_range, idx, 0)
            page = idx_s // page_size
            phys = tl.load(block_table + row * num_pages_per_row + page, mask = in_range, other = 0)
            if DEBUG_BOUNDS:
                tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_attn: entry idx >= pool_len")
                tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_attn: pool page OOB")
            tok = phys * page_size + idx_s % page_size
            vc = tl.load(pool_c + tok[:, None] * D_c + offs_c[None, :],
                         mask = in_range[:, None] & valid_c[None, :], other = 0.0)
            vr = tl.load(pool_r + tok[:, None] * D_r + offs_r[None, :],
                         mask = in_range[:, None], other = 0.0)
            scores = tl.dot(qc, tl.trans(vc))
            scores = tl.dot(qr, tl.trans(vr), acc = scores) * scale
            scores = tl.where(in_range[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(scores - m_exp[:, None])
            p = tl.where(in_range[None, :], p, 0.0)
            alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
            l = l * alpha + tl.sum(p, axis = 1)
            pv = p.to(vc.dtype)
            acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
            acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
            m_state = m_new

        denom = tl.where(l == 0.0, 1.0, l)
        oc = acc_c / denom[:, None]
        o_r = acc_r / denom[:, None]

        if DEROTATE:
            # eq. 26: rotate the output's rope slice at the query's absolute position (the
            # caller passes a negated table for de-rotation). Rotation is linear, so applying
            # it after the softmax-weighted sum equals summing rotated rows. Standard GPT-J
            # pairs (2i, 2i+1) via split/rotate/interleave, all in registers, fp32
            theta = tl.load(derot_inv_freq + tl.arange(0, D_r // 2)) * (q_pos0 + row)
            cos = tl.cos(theta)[None, :]
            sin = tl.sin(theta)[None, :]
            o_e, o_o = tl.split(tl.reshape(o_r, (BLOCK_H, D_r // 2, 2)))
            o_r = tl.interleave(o_e * cos - o_o * sin, o_o * cos + o_e * sin)

        D_out = D_c + D_r
        if HPG > 0:
            # Group-major: out[h // HPG, row, (h % HPG) * D + d]
            base_h = (offs_h // HPG) * (R * HPG * D_out) + row * (HPG * D_out) + (offs_h % HPG) * D_out
        else:
            base_h = (row * H + offs_h) * D_out
        out_base = out + base_h[:, None]
        tl.store(out_base + offs_c[None, :], oc.to(tl.float16), mask = valid_h[:, None] & valid_c[None, :])
        tl.store(out_base + D_c + offs_r[None, :], o_r.to(tl.float16), mask = valid_h[:, None])


    @triton.jit(do_not_specialize = [
        "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0",
        "win_floor", "ring_beg", "ring_stride",
    ], debug = dsa_debug_bounds)
    def _dsa_attn_split_kernel(
        q,                   # (R, H, D_c + D_r) fp16
        ring,                # (ring_rows, D) fp16, rows at abs - ring_beg
        kv_chunk,            # (R, D) fp16, rows at abs - q_pos0
        pool_c,              # (pages * page_size, D_c) fp16
        pool_r,              # (pages * page_size, D_r) fp16
        block_table,         # (R, num_pages_per_row) int32
        indices,             # (R, K_pad) int32, -1 padded (not DENSE_POOL)
        ws_ml,               # (R * HB * S * BLOCK_H * 2) fp32 partial m / l
        ws_acc,              # (R * HB * S * BLOCK_H * D) fp32 partial numerators
        k_len,
        win_len,
        pool_len,
        num_pages_per_row,
        q_pos0,
        win_floor,
        ring_beg,
        slot_ids,            # MULTIROW: (B,) i32 ring slot per job (else ignored, pass 0)
        ring_stride,         # MULTIROW: ring slot stride in elements
        H: tl.constexpr,
        page_size: tl.constexpr,
        D_c: tl.constexpr,
        D_c_pad: tl.constexpr,
        D_r: tl.constexpr,
        K_pad: tl.constexpr,
        compress_rate: tl.constexpr,
        scale: tl.constexpr,
        HAS_WINDOW: tl.constexpr,
        DENSE_POOL: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_W: tl.constexpr,
        SEQ: tl.constexpr = 1,
        MULTIROW: tl.constexpr = 0,
        DEBUG_BOUNDS: tl.constexpr = 0,
        DEBUG_PAGES: tl.constexpr = 0,
    ):
        """Flash-decoding split phase: each program covers one (query row, head block) and a
        contiguous slice of that row's VIRTUAL key sequence [window keys ++ pool entries],
        writing softmax partials (m, l, acc) to the workspace. Sinks, normalization, the
        eq. 26 de-rotation and the output store all live in the combine kernel."""
        pid = tl.program_id(0)
        split = tl.program_id(1)
        n_splits = tl.num_programs(1)
        h_blocks: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
        row = pid // h_blocks

        # MULTIROW: rows are B jobs x SEQ; the position/window/ring state args are per-job
        # i32 arrays, the ring is the stacked (slots, rows, D) tensor addressed by slot, and
        # the block table holds one row per JOB (paged pools)
        if MULTIROW:
            job = row // SEQ
            loc = row % SEQ
            q_pos0 = tl.load(q_pos0 + job)
            win_floor = tl.load(win_floor + job)
            ring_beg = tl.load(ring_beg + job)
            pool_len = tl.load(pool_len + job)
            k_len = tl.load(k_len + job)
            slot = tl.load(slot_ids + job)
            ring = ring + slot.to(tl.int64) * ring_stride
            bt_row = job
            cbase = job * SEQ
        else:
            loc = row
            cbase = 0
            bt_row = row

        offs_h = (pid % h_blocks) * BLOCK_H + tl.arange(0, BLOCK_H)
        valid_h = offs_h < H
        offs_c = tl.arange(0, D_c_pad)
        valid_c = offs_c < D_c
        offs_r = tl.arange(0, D_r)
        D = D_c + D_r

        q_base = q + (row * H + offs_h)[:, None] * D
        qc = tl.load(q_base + offs_c[None, :], mask = valid_h[:, None] & valid_c[None, :], other = 0.0)
        qr = tl.load(q_base + D_c + offs_r[None, :], mask = valid_h[:, None], other = 0.0)

        # This row's virtual key range for this split
        if DENSE_POOL:
            n_pool = tl.minimum((q_pos0 + loc + 1) // compress_rate, pool_len)
        else:
            n_pool = k_len
        n_win = win_len if HAS_WINDOW else 0
        n_tot = n_win + n_pool
        chunk = (n_tot + n_splits - 1) // n_splits
        j0 = split * chunk
        j1 = tl.minimum(j0 + chunk, n_tot)

        m_state = tl.full((BLOCK_H,), -float("inf"), tl.float32)
        l = tl.zeros((BLOCK_H,), tl.float32)
        acc_c = tl.zeros((BLOCK_H, D_c_pad), tl.float32)
        acc_r = tl.zeros((BLOCK_H, D_r), tl.float32)

        if HAS_WINDOW:
            q_abs = q_pos0 + loc
            w1 = tl.minimum(j1, n_win)
            for n0 in tl.range(j0, w1, BLOCK_W, num_stages = 1):
                offs_j = n0 + tl.arange(0, BLOCK_W)
                abs_pos = q_abs - offs_j
                in_range = (offs_j < w1) & (abs_pos >= win_floor)
                mc = in_range & (abs_pos >= q_pos0)
                mr = in_range & (abs_pos < q_pos0)
                idx_c = tl.where(mc, cbase + abs_pos - q_pos0, 0)
                idx_r = tl.where(mr, abs_pos - ring_beg, 0)
                vc = tl.load(kv_chunk + idx_c[:, None] * D + offs_c[None, :],
                             mask = mc[:, None] & valid_c[None, :], other = 0.0) \
                   + tl.load(ring + idx_r[:, None] * D + offs_c[None, :],
                             mask = mr[:, None] & valid_c[None, :], other = 0.0)
                vr = tl.load(kv_chunk + idx_c[:, None] * D + D_c + offs_r[None, :],
                             mask = mc[:, None], other = 0.0) \
                   + tl.load(ring + idx_r[:, None] * D + D_c + offs_r[None, :],
                             mask = mr[:, None], other = 0.0)
                scores = tl.dot(qc, tl.trans(vc))
                scores = tl.dot(qr, tl.trans(vr), acc = scores) * scale
                scores = tl.where(in_range[None, :], scores, -float("inf"))
                m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
                m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
                p = tl.exp(scores - m_exp[:, None])
                p = tl.where(in_range[None, :], p, 0.0)
                alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
                l = l * alpha + tl.sum(p, axis = 1)
                pv = p.to(vc.dtype)
                acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
                acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
                m_state = m_new

        p0 = tl.maximum(j0 - n_win, 0)
        p1 = j1 - n_win
        for n0 in range(p0, p1, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            if DENSE_POOL:
                idx = tl.where(offs_n < p1, offs_n, -1)
            else:
                idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < p1, other = -1)
            in_range = idx >= 0
            idx_s = tl.where(in_range, idx, 0)
            page = idx_s // page_size
            phys = tl.load(block_table + bt_row * num_pages_per_row + page, mask = in_range, other = 0)
            if DEBUG_BOUNDS:
                tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_split: entry idx >= pool_len")
                tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_split: pool page OOB")
            tok = phys * page_size + idx_s % page_size
            vc = tl.load(pool_c + tok[:, None] * D_c + offs_c[None, :],
                         mask = in_range[:, None] & valid_c[None, :], other = 0.0)
            vr = tl.load(pool_r + tok[:, None] * D_r + offs_r[None, :],
                         mask = in_range[:, None], other = 0.0)
            scores = tl.dot(qc, tl.trans(vc))
            scores = tl.dot(qr, tl.trans(vr), acc = scores) * scale
            scores = tl.where(in_range[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(scores - m_exp[:, None])
            p = tl.where(in_range[None, :], p, 0.0)
            alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
            l = l * alpha + tl.sum(p, axis = 1)
            pv = p.to(vc.dtype)
            acc_c = acc_c * alpha[:, None] + tl.dot(pv, vc)
            acc_r = acc_r * alpha[:, None] + tl.dot(pv, vr)
            m_state = m_new

        # Partials out (fp32)
        hloc = tl.arange(0, BLOCK_H)
        base = ((pid * n_splits + split) * BLOCK_H + hloc) * 2
        tl.store(ws_ml + base, m_state)
        tl.store(ws_ml + base + 1, l)
        abase = ((pid * n_splits + split) * BLOCK_H + hloc)[:, None] * D
        tl.store(ws_acc + abase + offs_c[None, :], acc_c, mask = valid_c[None, :])
        tl.store(ws_acc + abase + D_c + offs_r[None, :], acc_r)

    @triton.jit(do_not_specialize = ["q_pos0", "R", "n_splits"])
    def _dsa_attn_combine_kernel(
        ws_ml,               # (R * HB * S * BLOCK_H * 2) fp32
        ws_acc,              # (R * HB * S * BLOCK_H * D) fp32
        sinks,               # (H,) fp32 (HAS_SINKS)
        derot_inv_freq,      # (D_r // 2,) fp32 (DEROTATE)
        out,                 # (R, H, D) fp16 or (H / HPG, R, HPG * D) when HPG > 0
        q_pos0,
        R,
        n_splits,
        H: tl.constexpr,
        D_c: tl.constexpr,
        D_r: tl.constexpr,
        HAS_SINKS: tl.constexpr,
        DEROTATE: tl.constexpr,
        HPG: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,       # even; D_c is even so rope pairs never straddle tiles
        SEQ: tl.constexpr = 1,
        MULTIROW: tl.constexpr = 0,  # q_pos0 is a per-job i32 array when set
    ):
        """Combine phase: merge the split partials (sink folded in as one more partial),
        normalize, de-rotate (identity rotation, theta = 0, below the rope columns) and
        store. One program per (row, head block, D tile); the m/l merge is recomputed per D
        tile, which is trivial next to the acc traffic."""
        pid = tl.program_id(0)
        dtile = tl.program_id(1)
        h_blocks: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
        row = pid // h_blocks
        offs_h = (pid % h_blocks) * BLOCK_H + tl.arange(0, BLOCK_H)
        valid_h = offs_h < H
        hloc = tl.arange(0, BLOCK_H)
        D: tl.constexpr = D_c + D_r
        offs_d = dtile * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_d = offs_d < D

        if HAS_SINKS:
            m_run = tl.load(sinks + offs_h, mask = valid_h, other = -float("inf"))
            l_run = tl.full((BLOCK_H,), 1.0, tl.float32)
        else:
            m_run = tl.full((BLOCK_H,), -float("inf"), tl.float32)
            l_run = tl.zeros((BLOCK_H,), tl.float32)
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        for s in range(n_splits):
            base = ((pid * n_splits + s) * BLOCK_H + hloc)
            m_s = tl.load(ws_ml + base * 2)
            l_s = tl.load(ws_ml + base * 2 + 1)
            a_s = tl.load(ws_acc + base[:, None] * D + offs_d[None, :], mask = valid_d[None, :], other = 0.0)
            m_new = tl.maximum(m_run, m_s)
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.where(m_run == -float("inf"), 0.0, tl.exp(m_run - m_exp))
            beta = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_exp))
            l_run = l_run * alpha + l_s * beta
            acc = acc * alpha[:, None] + a_s * beta[:, None]
            m_run = m_new

        denom = tl.where(l_run == 0.0, 1.0, l_run)
        o = acc / denom[:, None]

        if DEROTATE:
            # Uniform pair rotation: theta = 0 (identity) below the rope columns
            offs_p = (dtile * BLOCK_D) // 2 + tl.arange(0, BLOCK_D // 2)
            col_e = offs_p * 2
            in_rope = col_e >= D_c
            fr = tl.load(derot_inv_freq + tl.where(in_rope, (col_e - D_c) // 2, 0),
                         mask = in_rope, other = 0.0)
            if MULTIROW:
                qp = tl.load(q_pos0 + row // SEQ) + row % SEQ
            else:
                qp = q_pos0 + row
            theta = fr * qp
            cos = tl.cos(theta)[None, :]
            sin = tl.sin(theta)[None, :]
            o_e, o_o = tl.split(tl.reshape(o, (BLOCK_H, BLOCK_D // 2, 2)))
            o = tl.interleave(o_e * cos - o_o * sin, o_o * cos + o_e * sin)

        if HPG > 0:
            base_h = (offs_h // HPG) * (R * HPG * D) + row * (HPG * D) + (offs_h % HPG) * D
        else:
            base_h = (row * H + offs_h) * D
        tl.store(out + base_h[:, None] + offs_d[None, :], o.to(tl.float16),
                 mask = valid_h[:, None] & valid_d[None, :])

    @triton.jit(do_not_specialize = ["T", "R", "q_pos0", "bound_max"],
                debug = dsa_debug_bounds)
    def _dsa_indexer_kernel(
        q_idx,               # (R, H_i, D_i) fp16, rope applied
        w,                   # (R, H_i) fp16 raw head weights (scales folded into `scale`)
        k_idx,               # (T, D_i) fp16 indexer keys, rope applied; paged when EPP > 0
        scores,              # (R, S_stride) fp16 out
        T,                   # runtime: valid keys
        R,                   # runtime: valid query rows
        q_pos0,              # runtime: absolute position of query row 0
        bound_max,           # runtime: entry count clamp; bound[r] =
                             #   min((q_pos0 + r + 1) // compress_rate, bound_max)
        block_table,         # EPP > 0: (npr,) i32 page table of the (single) job
        H_i: tl.constexpr,
        D_i: tl.constexpr,
        S_stride: tl.constexpr,
        compress_rate: tl.constexpr,
        scale: tl.constexpr,             # D_i ** -0.5 * H_i ** -0.5
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EPP: tl.constexpr = 0,           # pool entries per page; 0 = contiguous k_idx
        DEBUG_BOUNDS: tl.constexpr = 0,
        DEBUG_PAGES: tl.constexpr = 0,
    ):
        """Lightning-indexer scoring: scores[r, s] = sum_h w[r, h] * relu(q[r, h] . k[s]) * scale.
        GEMM-shaped with a per-head ReLU epilogue; the head loop runs H_i full MMA dots against
        a resident key tile. Shared between raw-token keys (V3.2-on-MLA) and pooled keys (V4
        CSA), only the key tensor differs. Causal entry bound applied in the epilogue."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D_i)
        valid_m = offs_m < R
        valid_n = offs_n < T

        if EPP > 0:
            phys = tl.load(block_table + offs_n // EPP, mask = valid_n, other = 0)
            if DEBUG_BOUNDS:
                tl.device_assert(tl.where(valid_n, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_indexer: pool page OOB")
            k_rows = phys * EPP + offs_n % EPP
        else:
            k_rows = offs_n
        kt = tl.load(k_idx + k_rows[None, :] * D_i + offs_d[:, None],
                     mask = valid_n[None, :], other = 0.0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for h in range(H_i):
            qh = tl.load(q_idx + (offs_m * H_i + h)[:, None] * D_i + offs_d[None, :],
                         mask = valid_m[:, None], other = 0.0)
            logits = tl.dot(qh, kt)
            wh = tl.load(w + offs_m * H_i + h, mask = valid_m, other = 0.0)
            acc += tl.maximum(logits, 0.0) * wh[:, None].to(tl.float32)

        acc = acc * scale
        bound = tl.minimum((q_pos0 + offs_m + 1) // compress_rate, bound_max)
        acc = tl.where(offs_n[None, :] < bound[:, None], acc, -float("inf"))
        tl.store(scores + offs_m[:, None] * S_stride + offs_n[None, :], acc.to(tl.float16),
                 mask = valid_m[:, None] & valid_n[None, :])


    @triton.jit(do_not_specialize = ["T", "R", "q_pos0", "bound_max", "num_pages_per_row"],
                debug = dsa_debug_bounds)
    def _dsa_indexer_fewq_kernel(
        q_idx,               # (R, H_i, D_i) fp16, rope applied
        w,                   # (R, H_i) fp16 raw head weights
        k_idx,               # (T, D_i) fp16 indexer keys, rope applied; paged when EPP > 0
        scores,              # (R, S_stride) fp16 out
        T,
        R,
        q_pos0,
        bound_max,
        block_table,         # EPP > 0: i32 page table, one row per job (row 0 if not MULTIROW)
        num_pages_per_row,   # EPP > 0, MULTIROW: block table row stride
        H_i: tl.constexpr,
        H_pad: tl.constexpr,
        D_i: tl.constexpr,
        S_stride: tl.constexpr,
        compress_rate: tl.constexpr,
        scale: tl.constexpr,             # D_i ** -0.5 * H_i ** -0.5
        BLOCK_N: tl.constexpr,
        SEQ: tl.constexpr = 1,
        MULTIROW: tl.constexpr = 0,      # T/q_pos0/bound_max are per-job i32 arrays
        EPP: tl.constexpr = 0,           # pool entries per page; 0 = contiguous k_idx
        DEBUG_BOUNDS: tl.constexpr = 0,
        DEBUG_PAGES: tl.constexpr = 0,
    ):
        """Few-query variant (decode): one program per (query row, key tile) with HEADS as
        the MMA M dim. A single dot replaces the head loop, which is a serial latency chain
        when the row tile is nearly empty."""
        r = tl.program_id(0)
        pid_n = tl.program_id(1)
        if MULTIROW:
            job = r // SEQ
            loc = r % SEQ
            T = tl.load(T + job)
            q_pos0 = tl.load(q_pos0 + job)
            bound_max = tl.load(bound_max + job)
            block_table = block_table + job * num_pages_per_row
        else:
            loc = r
        # In a captured graph the grid is sized for the FULL score buffer (pool capacity)
        # with T patched per replay; tiles past T retire immediately (their region of the
        # scores buffer holds the required -inf from the one-time fill)
        if pid_n * BLOCK_N >= T:
            return
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D_i)
        offs_h = tl.arange(0, H_pad)
        valid_n = offs_n < T
        valid_h = offs_h < H_i

        qh = tl.load(q_idx + (r * H_i + offs_h)[:, None] * D_i + offs_d[None, :],
                     mask = valid_h[:, None], other = 0.0)                     # (H_pad, D)
        if EPP > 0:
            phys = tl.load(block_table + offs_n // EPP, mask = valid_n, other = 0)
            if DEBUG_BOUNDS:
                tl.device_assert(tl.where(valid_n, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_fewq: pool page OOB")
            k_rows = phys * EPP + offs_n % EPP
        else:
            k_rows = offs_n
        kt = tl.load(k_idx + k_rows[None, :] * D_i + offs_d[:, None],
                     mask = valid_n[None, :], other = 0.0)                     # (D, N)
        logits = tl.dot(qh, kt)                                                # (H_pad, N)
        wh = tl.load(w + r * H_i + offs_h, mask = valid_h, other = 0.0)
        acc = tl.sum(tl.maximum(logits, 0.0) * wh[:, None].to(tl.float32), axis = 0) * scale

        bound = tl.minimum((q_pos0 + loc + 1) // compress_rate, bound_max)
        acc = tl.where(offs_n < bound, acc, -float("inf"))
        tl.store(scores + r * S_stride + offs_n, acc.to(tl.float16), mask = valid_n)

def dsa_attn(
    q,                       # (R, H, D_c + D_r) fp16, rope slice pre-rotated
    pool_c,                  # (pages, page_size, D_c) or (rows, D_c) fp16
    pool_r,                  # matching rope-part tensor, D_r wide
    block_table,             # (R, num_pages_per_row) int32
    sinks = None,            # (H,) fp32 or None
    ring = None,             # (ring_rows, D_c + D_r) fp16: rows at abs - ring_beg (window)
    kv_chunk = None,         # (R, D_c + D_r) fp16: this chunk's rows at abs - q_pos0
    win_len = 0,
    win_floor = 0,           # lowest absolute position visible to the window
    ring_beg = 0,            # absolute position of ring row 0
    indices = None,          # (R, K_pad) int32, -1 padded (gathered mode)
    k_len = 0,
    pool_len = 0,            # dense mode: pool entry count
    q_pos0 = 0,              # dense mode: absolute position of query row 0
    compress_rate = 1,       # dense mode causal bound divisor
    scale = None,
    derot_inv_freq = None,   # (D_r // 2,) fp32: fuse eq. 26 output rotation at q_pos0 + row
                             # (pass the NEGATED frequency table for de-rotation)
    groups = 1,              # > 1: store output group-major (groups, R, (H // groups) * D)
    group_major = None,      # force group-major layout even at groups == 1 (TP shard
                             # holding a single o_group); default = (groups > 1)
    out = None,
    n_splits = 0,            # flash-decoding splits; 0 = auto (few queries -> split path)
    block_h = 32,
    block_n = 32,
    num_warps = 4,
    num_stages = 3,
    page_size = 256,         # pool entries per block-table page (PAGE_SIZE // m for the
                             # paged pools; 256 with an identity table for contiguous pools)
    nc_block = False,        # DSpark draft mode: non-causal chunk + paged window history
                             # (single job per call; forces the one-shot kernel)
    multirow = None,         # batched jobs: dict(q_pos, win_floor, ring_beg, pool_len,
                             # k_len (B,) i32; slot_ids (B,) i32; ring_stride int; seq int)
                             # -- scalar args of the same names are ignored, ring is the
                             # stacked (slots, rows, D) tensor, block_table holds one row
                             # per JOB, R = B * seq
):
    """Core DSA attention over [sliding ring ++ pool entries], V IS K. Gathered mode when
    `indices` is given, dense-pool mode otherwise (per-query causal entry bound). With
    derot_inv_freq the eq. 26 output de-rotation is fused into the epilogue (otherwise the
    output's rope slice is still rotated and de-rotation is the caller's epilogue); with
    groups > 1 the output is written group-major for the grouped o_proj."""
    R, H, D = q.shape
    D_r = pool_r.shape[-1]
    D_c = D - D_r
    if scale is None:
        scale = D ** -0.5
    dense_pool = indices is None
    assert H % groups == 0
    if group_major is None:
        group_major = groups > 1
    hpg = H // groups if group_major else 0
    out_shape = (groups, R, hpg * D) if group_major else (R, H, D)
    if out is None:
        out = g_tensor_cache.get(q.device, out_shape, torch.half, "dsa_out")
    else:
        assert out.shape == out_shape

    dummy_i = block_table  # any valid int32 pointer for unused int args
    has_window = win_len > 0 and kv_chunk is not None
    if not has_window:
        ring, kv_chunk = q, q
    elif ring is None:
        ring = kv_chunk    # window fully inside the chunk (win_floor >= q_pos0)
    if indices is None:
        indices, K_pad = dummy_i, 32
    else:
        K_pad = indices.shape[1]
        k_len = k_len or K_pad
    if sinks is None:
        sinks_t, has_sinks = q, False
    else:
        sinks_t, has_sinks = sinks, True
    if derot_inv_freq is None:
        derot_t, derotate = q, False   # any valid fp pointer for the unused arg
    else:
        derot_t, derotate = derot_inv_freq, True

    # A single-row block table is shared by every query row (contiguous per-slot pools):
    # stride 0 makes the kernel's bt[row * npr + page] hit row 0 for all rows
    npr = block_table.shape[1] if block_table.shape[0] > 1 else 0
    dbg = 1 if dsa_debug_bounds else 0
    if dsa_debug_bounds:
        # Debug codegen inflates the one-shot kernel's smem footprint past 99 KB devices;
        # halve the head tile (perf is irrelevant with asserts on)
        num_stages = min(num_stages, 2)
        block_h = min(block_h, 16)
    dbg_pages = -(-(pool_c.numel() // max(D_c, 1)) // max(page_size, 1)) if dsa_debug_bounds else 0
    if nc_block:
        n_splits = 1
    if multirow is not None:
        n_splits = n_splits or 8
    if n_splits == 0:
        if R <= 8:
            est = (win_len if has_window else 0) + \
                  (k_len if indices is not dummy_i else min(pool_len, (q_pos0 + R) // max(compress_rate, 1)))
            n_splits = 16 if est > 256 else 8
        else:
            n_splits = 1

    if n_splits > 1:
        block_h = min(block_h, 16)   # more head-blocks: the split path wants parallelism
        # Flash-decoding split: the monolithic kernel at R = 1 is 2 blocks walking the key
        # tiles serially (pure latency); the split phase spreads the keys over
        # R * h_blocks * n_splits programs and the combine folds sinks/derot/store
        hb = triton.cdiv(H, block_h)
        D_out = D
        ws_ml = g_tensor_cache.get(q.device, (R * hb * n_splits * block_h * 2,),
                                   torch.float, "dsa_ws_ml")
        ws_acc = g_tensor_cache.get(q.device, (R * hb * n_splits * block_h * D_out,),
                                    torch.float, "dsa_ws_acc")
        if multirow is not None:
            mr = multirow
            a_klen, a_pool = mr["k_len"], mr["pool_len"]
            a_qpos, a_floor, a_beg = mr["q_pos"], mr["win_floor"], mr["ring_beg"]
            a_slots, a_rstride, a_seq = mr["slot_ids"], mr["ring_stride"], mr["seq"]
            npr = block_table.shape[1]     # one block-table row per job
        else:
            a_klen, a_pool, a_qpos, a_floor, a_beg = k_len, pool_len, q_pos0, win_floor, ring_beg
            a_slots, a_rstride, a_seq = 0, 0, 1
        with torch.cuda.device(q.device):
            _dsa_attn_split_kernel[(R * hb, n_splits)](
                q, ring, kv_chunk, pool_c.reshape(-1, D_c), pool_r.reshape(-1, D_r),
                block_table, indices, ws_ml, ws_acc,
                a_klen, win_len, a_pool, npr, a_qpos, a_floor, a_beg,
                a_slots, a_rstride,
                H = H, page_size = page_size, D_c = D_c, D_c_pad = triton.next_power_of_2(D_c),
                D_r = D_r, K_pad = K_pad, compress_rate = compress_rate, scale = scale,
                HAS_WINDOW = has_window, DENSE_POOL = dense_pool,
                BLOCK_H = block_h, BLOCK_N = block_n, BLOCK_W = 16,
                SEQ = a_seq, MULTIROW = multirow is not None,
                DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
                num_warps = num_warps, num_stages = 2,
            )
            _dsa_attn_combine_kernel[(R * hb, triton.cdiv(D, 128))](
                ws_ml, ws_acc, sinks_t, derot_t, out,
                a_qpos, R, n_splits,
                H = H, D_c = D_c, D_r = D_r,
                HAS_SINKS = has_sinks, DEROTATE = derotate, HPG = hpg,
                BLOCK_H = block_h, BLOCK_D = 128,
                SEQ = a_seq, MULTIROW = multirow is not None,
                num_warps = 4, num_stages = 2,
            )
        return out

    grid = (R * triton.cdiv(H, block_h),)
    with torch.cuda.device(q.device):   # layer split: launch on the tensor's device
        _dsa_attn_kernel[grid](
            q, ring, kv_chunk, pool_c.reshape(-1, D_c), pool_r.reshape(-1, D_r),
            block_table, indices, sinks_t, derot_t, out,
            k_len, win_len, pool_len, npr, q_pos0, R, win_floor, ring_beg,
            H = H, page_size = page_size, D_c = D_c, D_c_pad = triton.next_power_of_2(D_c),
            D_r = D_r, K_pad = K_pad, compress_rate = compress_rate,
            scale = scale,
            HAS_WINDOW = has_window,
            HAS_SINKS = has_sinks,
            DENSE_POOL = dense_pool,
            DEROTATE = derotate,
            HPG = hpg,
            BLOCK_H = block_h, BLOCK_N = block_n, BLOCK_W = 16,
            DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
            NC_BLOCK = 1 if nc_block else 0,
            num_warps = num_warps, num_stages = num_stages,
        )
    return out


def dsa_indexer_scores(
    q_idx,                   # (R, H_i, D_i) fp16, rope applied
    weights,                 # (R, H_i) fp16 raw head weights (H_i ** -0.5 folded in-kernel)
    k_idx,                   # (T, D_i) fp16 key pool, rope applied; contiguous rows, or the
                             # flat paged pool when block_table is given
    q_pos0,                  # absolute position of query row 0
    compress_rate,
    bound_max,               # entry count clamp (valid pool entries)
    scores = None,
    block_m = 64,
    block_n = 128,
    num_warps = 8,
    num_stages = 2,
    block_table = None,      # (npr,) or (1, npr) i32 page table of the (single) job
    epp = 0,                 # pool entries per page (paged mode)
):
    """Indexer scores (R, T) fp16 with -inf past each query's causal entry bound
    min((q_pos0 + r + 1) // compress_rate, bound_max); feed to topk."""
    R, H_i, D_i = q_idx.shape
    T = bound_max
    if block_table is None:
        T = k_idx.shape[0]
        bt, epp = 0, 0
    else:
        bt = block_table.reshape(-1)
    dbg = 1 if (dsa_debug_bounds and epp) else 0
    dbg_pages = -(-k_idx.shape[0] // epp) if dbg else 0
    S_stride = triton.cdiv(max(T, 1), block_n) * block_n
    if scores is None:
        scores = torch.empty((R, S_stride), dtype = torch.half, device = q_idx.device)
    with torch.cuda.device(q_idx.device):
        if R <= 4:
            # Few-query (decode) shape: heads as the MMA M dim, one dot per key tile --
            # the query-tiled kernel degenerates to a serial head loop over padding here
            grid = (R, triton.cdiv(max(T, 1), block_n))
            _dsa_indexer_fewq_kernel[grid](
                q_idx, weights, k_idx, scores, T, R, q_pos0, bound_max, bt, 0,
                H_i = H_i, H_pad = max(triton.next_power_of_2(H_i), 16), D_i = D_i,
                S_stride = S_stride, compress_rate = compress_rate,
                scale = D_i ** -0.5 * H_i ** -0.5,
                BLOCK_N = block_n, EPP = epp,
                DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
                num_warps = num_warps, num_stages = num_stages,
            )
        else:
            grid = (triton.cdiv(R, block_m), triton.cdiv(max(T, 1), block_n))
            _dsa_indexer_kernel[grid](
                q_idx, weights, k_idx, scores, T, R, q_pos0, bound_max, bt,
                H_i = H_i, D_i = D_i, S_stride = S_stride, compress_rate = compress_rate,
                scale = D_i ** -0.5 * H_i ** -0.5,
                BLOCK_M = block_m, BLOCK_N = block_n, EPP = epp,
                DEBUG_BOUNDS = dbg, DEBUG_PAGES = dbg_pages,
                num_warps = num_warps, num_stages = num_stages,
            )
    return scores[:, :T]
