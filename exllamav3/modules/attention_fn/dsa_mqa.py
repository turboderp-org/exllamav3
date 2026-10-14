"""
MQA-specialized DSA attention kernels for RDNA. The upstream kernels give every program BLOCK_H heads and the whole
output width, so the fp32 accumulator spans BLOCK_H x D next to a resident q tile, which RDNA3's WMMA replicates
across the two half-waves as its A operand. Both compile at the VGPR ceiling with heavy spilling to scratch on these
parts. Here one program owns HP heads (the MMA M dimension) and one BD-wide column block of the output:

  scores = sum_kc q[HP, kc:kc+KC] . K[BN, kc:kc+KC]^T     the score reduction over D is a RUNTIME loop of KC-wide
                                                           chunks with q re-read per chunk (cache hits): a loop-
                                                           invariant q would be kept resident as the A operand,
                                                           which is exactly what spilled
  acc   += p[HP, BN] . V[BN, BD]                           V = K, only this program's columns

The virtual key row is [c (D_c) | r (D_r)]: ring / chunk rows are contiguous, pool rows are pool_c ++ pool_r.
Scores are recomputed per column block. Packed pools (QC > 0) work in upstream's H32-rotated domain: q and window
tiles are rotated per 32-group on the latent columns, decode partials stay rotated (the combine rotates back), and
the prefill epilogue rotates back itself.

Decode grid contract (dsv4_attn.cpp / dsa_attn): (rows * H / BLOCK_H, n_splits), BLOCK_H being the combine's head
block; a program reads its index modulo H / BLOCK_H as (head group, column block) with (H / HP) * (D / BD) ==
H / BLOCK_H. m / l are identical in every column block of a head group; column block 0 writes them. The prefill
kernel takes one program per (query row, head group, column block).

Not covered (the callers keep the upstream kernels): Q_SPLIT and OUT_LATENT (DSA on MLA), non-power-of-two D,
head counts the tiling does not divide, D_r == 0.
"""

import triton
import triton.language as tl

from .triton_paged import _rot_h32


@triton.jit
def _qc_plane_cols(qw, row_words, mask_n, g0, n_g, pbase,
                   W: tl.constexpr, BITS: tl.constexpr, NG: tl.constexpr):
    """One bit plane of 32-groups [g0, g0 + NG) of packed rows, (BN, NG * 32) int32;
    groups at or past n_g (relative) read as zero."""
    VPW: tl.constexpr = 32 // W
    garr = tl.arange(0, NG * W)
    grp = garr // W
    cols = (g0 + grp) * BITS + pbase + (garr % W)
    w = tl.load(qw + row_words[:, None] + cols[None, :],
                mask = mask_n[:, None] & (grp < n_g)[None, :], other = 0)
    nib = (w[:, :, None] >> (tl.arange(0, VPW) * W)[None, None, :]) & ((1 << W) - 1)
    return tl.reshape(nib, (w.shape[0], NG * 32))


@triton.jit
def _qc_load_cols(qwords, scales, tok, mask_n, g0, G: tl.constexpr,
                  BITS: tl.constexpr, NG: tl.constexpr):
    """(BN, NG * 32) fp16 tile of columns [g0 * 32, (g0 + NG) * 32) of a packed pool with G
    groups per row (rotated domain, same grid as triton_paged._qc_load_v); zero past G."""
    n_g = G - g0
    row_words = tok * (G * BITS)
    raw = tl.zeros((1, 1), tl.int32)
    pbase = 0
    first = True
    if BITS & 8:
        raw = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 8, BITS, NG)
        pbase += 8
        first = False
    if BITS & 4:
        p = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 4, BITS, NG)
        raw = p if first else (raw << 4) | p
        pbase += 4
        first = False
    if BITS & 2:
        p = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 2, BITS, NG)
        raw = p if first else (raw << 2) | p
        pbase += 2
        first = False
    if BITS & 1:
        p = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 1, BITS, NG)
        raw = p if first else (raw << 1) | p
    garr = tl.arange(0, NG)
    sc = tl.load(scales + tok[:, None] * G + (g0 + garr)[None, :],
                 mask = mask_n[:, None] & (garr < n_g)[None, :], other = 0.0)
    scx = tl.reshape(tl.broadcast_to(sc[:, :, None], (sc.shape[0], NG, 32)), (sc.shape[0], NG * 32))
    mh = (1 << (BITS - 1)) - 0.5
    inv_m = 1.0 / (1 << (BITS - 1))
    return ((raw.to(tl.float32) - mh) * (scx.to(tl.float32) * inv_m)).to(tl.float16)


@triton.jit
def _rot_cols(x, h32, cols, D_c: tl.constexpr, ROWS: tl.constexpr, W: tl.constexpr):
    """H32-rotate the latent columns (cols < D_c) of an fp16 tile; rope columns pass."""
    xr = _rot_h32(x, h32, ROWS, W)
    return tl.where((cols < D_c)[None, :], xr, x)


@triton.jit
def _pool_tile(pool_c, pool_r, pool_s, tok, in_range, cols, c0,
               D_c: tl.constexpr, D_r: tl.constexpr, QC: tl.constexpr, NC: tl.constexpr):
    """(BN, NC) fp16 tile of virtual pool rows [c | r], columns c0 + [0, NC) (cols = c0 +
    arange(NC)); latent part from the fp16 or packed pool, rope part from pool_r."""
    if QC > 0:
        t = _qc_load_cols(pool_c, pool_s, tok, in_range, c0 // 32, D_c // 32, QC, NC // 32)
    else:
        t = tl.load(pool_c + tok[:, None] * D_c + cols[None, :],
                    mask = in_range[:, None] & (cols < D_c)[None, :], other = 0.0)
    t += tl.load(pool_r + tok[:, None] * D_r + (cols - D_c)[None, :],
                 mask = in_range[:, None] & (cols >= D_c)[None, :], other = 0.0)
    return t


@triton.jit
def _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, cols, D: tl.constexpr):
    """(BW, len(cols)) fp16 tile of window rows: this step's chunk rows or ring rows."""
    return tl.load(kv_chunk + idx_c[:, None] * D + cols[None, :], mask = mc[:, None], other = 0.0) \
         + tl.load(ring + idx_r[:, None] * D + cols[None, :], mask = mr[:, None], other = 0.0)


@triton.jit
def _softmax_step(scores, in_range, m_state, l, acc, v, scale):
    scores = scores * scale
    scores = tl.where(in_range[None, :], scores, -float("inf"))
    m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
    m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp(scores - m_exp[:, None])
    p = tl.where(in_range[None, :], p, 0.0)
    alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
    l = l * alpha + tl.sum(p, axis = 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
    return m_new, l, acc


@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0",
    "win_floor", "ring_beg", "ring_stride",
])
def _dsa_decode_mqa_kernel(
    q, ring, kv_chunk, pool_c, pool_r, block_table, indices, ws_ml, ws_acc,
    k_len, win_len, pool_len, num_pages_per_row, q_pos0, win_floor, ring_beg,
    slot_ids, ring_stride, pool_s, h32,
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    D_c_pad: tl.constexpr,         # unused (upstream signature)
    D_r: tl.constexpr,
    K_pad: tl.constexpr,
    compress_rate: tl.constexpr,
    scale: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    DENSE_POOL: tl.constexpr,
    BLOCK_H: tl.constexpr,         # combine head block: workspace layout + grid basis
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SEQ: tl.constexpr = 1,
    MULTIROW: tl.constexpr = 0,
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
    Q_SPLIT: tl.constexpr = 0,     # must be 0
    OUT_LATENT: tl.constexpr = 0,  # must be 0
    QC: tl.constexpr = 0,
    HP: tl.constexpr = 32,         # heads per program (MMA M)
    BD: tl.constexpr = 256,        # output columns per program
    KC: tl.constexpr = 64,         # score reduction chunk over D (runtime loop)
    KSTAGES: tl.constexpr = 1,     # software-pipeline depth of the KC loop
):
    D: tl.constexpr = D_c + D_r
    HB_C: tl.constexpr = H // BLOCK_H
    NDB: tl.constexpr = D // BD
    tl.static_assert((H // HP) * NDB == HB_C)
    tl.static_assert(Q_SPLIT == 0 and OUT_LATENT == 0)

    pid = tl.program_id(0)
    split = tl.program_id(1)
    n_splits = tl.num_programs(1)
    row = pid // HB_C
    sub = pid % HB_C
    hgrp = sub // NDB
    dblk = sub % NDB

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

    offs_h = hgrp * HP + tl.arange(0, HP)
    c0 = dblk * BD
    vcols = c0 + tl.arange(0, BD)

    q_rows = q + (row * H + offs_h)[:, None] * D
    kcols0 = tl.arange(0, KC)

    # This row's virtual key range [window ++ pool] for this split (same split as upstream)
    if DENSE_POOL:
        n_pool = tl.minimum((q_pos0 + loc + 1) // compress_rate, pool_len)
    else:
        n_pool = k_len
    n_win = win_len if HAS_WINDOW else 0
    n_tot = n_win + n_pool
    chunk = (n_tot + n_splits - 1) // n_splits
    j0 = split * chunk
    j1 = tl.minimum(j0 + chunk, n_tot)

    m_state = tl.full((HP,), -float("inf"), tl.float32)
    l = tl.zeros((HP,), tl.float32)
    acc = tl.zeros((HP, BD), tl.float32)

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
            # Score reduction streamed over D in a RUNTIME loop: a loop-invariant q would be
            # kept resident as the WMMA A operand (replicated across half-waves on RDNA3:
            # 16 x 512 per warp = 256 VGPRs), which is what spilled the upstream kernel
            scores = tl.zeros((HP, BLOCK_W), tl.float32)
            for kc in tl.range(0, D, KC, num_stages = KSTAGES):
                kcols = kc + kcols0
                qk = tl.load(q_rows + kcols[None, :])
                k = _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, kcols, D)
                if QC > 0:
                    qk = _rot_cols(qk, h32, kcols, D_c, HP, KC)
                    k = _rot_cols(k, h32, kcols, D_c, BLOCK_W, KC)
                scores = tl.dot(qk, tl.trans(k), acc = scores)
            v = _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, vcols, D)
            if QC > 0:
                v = _rot_cols(v, h32, vcols, D_c, BLOCK_W, BD)
            m_state, l, acc = _softmax_step(scores, in_range, m_state, l, acc, v, scale)

    p0 = tl.maximum(j0 - n_win, 0)
    p1 = j1 - n_win
    for n0 in tl.range(p0, p1, BLOCK_N, num_stages = 1):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if DENSE_POOL:
            idx = tl.where(offs_n < p1, offs_n, -1)
        else:
            idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < p1, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        phys = tl.load(block_table + bt_row * num_pages_per_row + idx_s // page_size,
                       mask = in_range, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_mqa: entry idx >= pool_len")
            tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_mqa: pool page OOB")
        tok = phys * page_size + idx_s % page_size
        scores = tl.zeros((HP, BLOCK_N), tl.float32)
        for kc in tl.range(0, D, KC, num_stages = KSTAGES):
            kcols = kc + kcols0
            qk = tl.load(q_rows + kcols[None, :])
            if QC > 0:
                qk = _rot_cols(qk, h32, kcols, D_c, HP, KC)
            k = _pool_tile(pool_c, pool_r, pool_s, tok, in_range, kcols, kc, D_c, D_r, QC, KC)
            scores = tl.dot(qk, tl.trans(k), acc = scores)
        v = _pool_tile(pool_c, pool_r, pool_s, tok, in_range, vcols, c0, D_c, D_r, QC, BD)
        m_state, l, acc = _softmax_step(scores, in_range, m_state, l, acc, v, scale)

    # Partials in the combine's layout: ((row * HB_C + h // BLOCK_H) * S + split) * BLOCK_H
    # + h % BLOCK_H
    pidc = row * HB_C + offs_h // BLOCK_H
    base = (pidc * n_splits + split) * BLOCK_H + offs_h % BLOCK_H
    tl.store(ws_ml + base * 2, m_state, mask = dblk == 0)
    tl.store(ws_ml + base * 2 + 1, l, mask = dblk == 0)
    tl.store(ws_acc + base[:, None] * D + vcols[None, :], acc)




@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0", "R",
    "win_floor", "ring_beg",
])
def _dsa_prefill_mqa_kernel(
    q, ring, kv_chunk, pool_c, pool_r, block_table, indices, sinks, derot_inv_freq, out,
    k_len, win_len, pool_len, num_pages_per_row, q_pos0, R, win_floor, ring_beg,
    pool_s, h32,
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    D_c_pad: tl.constexpr,         # unused (upstream signature)
    D_r: tl.constexpr,
    K_pad: tl.constexpr,
    compress_rate: tl.constexpr,
    scale: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    DENSE_POOL: tl.constexpr,
    DEROTATE: tl.constexpr,
    HPG: tl.constexpr,
    BLOCK_H: tl.constexpr,         # unused (upstream signature)
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
    NC_BLOCK: tl.constexpr = 0,
    NC_CHUNK: tl.constexpr = 0,
    NC_HIST: tl.constexpr = 0,
    Q_SPLIT: tl.constexpr = 0,     # must be 0
    OUT_LATENT: tl.constexpr = 0,  # must be 0
    QC: tl.constexpr = 0,
    HP: tl.constexpr = 32,         # heads per program (MMA M)
    BD: tl.constexpr = 256,        # output columns per program
    KC: tl.constexpr = 64,         # score reduction chunk over D (runtime loop)
    KSTAGES: tl.constexpr = 1,
):
    D: tl.constexpr = D_c + D_r
    NHG: tl.constexpr = H // HP
    NDB: tl.constexpr = D // BD
    tl.static_assert(Q_SPLIT == 0 and OUT_LATENT == 0)

    pid = tl.program_id(0)
    row = pid // (NHG * NDB)
    sub = pid % (NHG * NDB)
    hgrp = sub // NDB
    dblk = sub % NDB

    offs_h = hgrp * HP + tl.arange(0, HP)
    c0 = dblk * BD
    vcols = c0 + tl.arange(0, BD)
    q_rows = q + (row * H + offs_h)[:, None] * D
    kcols0 = tl.arange(0, KC)

    if HAS_SINKS:
        m_state = tl.load(sinks + offs_h)
        l = tl.full((HP,), 1.0, tl.float32)
    else:
        m_state = tl.full((HP,), -float("inf"), tl.float32)
        l = tl.zeros((HP,), tl.float32)
    acc = tl.zeros((HP, BD), tl.float32)

    # Phase 1: sliding-window rows by absolute position (same addressing as upstream)
    if HAS_WINDOW:
        q_abs = q_pos0 + row
        if NC_BLOCK or NC_CHUNK:
            top = q_pos0 + R - 1
        else:
            top = q_abs
        for n0 in tl.range(0, win_len, BLOCK_W, num_stages = 1):
            offs_j = n0 + tl.arange(0, BLOCK_W)
            abs_pos = top - offs_j
            in_range = (offs_j < win_len) & (abs_pos >= win_floor)
            if NC_CHUNK:
                in_range = in_range & ((abs_pos >= q_pos0) | (abs_pos > q_abs - NC_HIST))
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
            # Score reduction streamed over D in a runtime loop (see module docstring)
            scores = tl.zeros((HP, BLOCK_W), tl.float32)
            for kc in tl.range(0, D, KC, num_stages = KSTAGES):
                kcols = kc + kcols0
                qk = tl.load(q_rows + kcols[None, :])
                k = _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, kcols, D)
                if QC > 0:
                    qk = _rot_cols(qk, h32, kcols, D_c, HP, KC)
                    k = _rot_cols(k, h32, kcols, D_c, BLOCK_W, KC)
                scores = tl.dot(qk, tl.trans(k), acc = scores)
            v = _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, vcols, D)
            if QC > 0:
                v = _rot_cols(v, h32, vcols, D_c, BLOCK_W, BD)
            m_state, l, acc = _softmax_step(scores, in_range, m_state, l, acc, v, scale)

    # Phase 2: pool entries -- gathered by index list, or dense with causal bound
    if DENSE_POOL:
        n_end = tl.minimum((q_pos0 + row + 1) // compress_rate, pool_len)
    else:
        n_end = k_len
    for n0 in tl.range(0, n_end, BLOCK_N, num_stages = 1):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if DENSE_POOL:
            idx = tl.where(offs_n < n_end, offs_n, -1)
        else:
            idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size,
                       mask = in_range, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_prefill: entry idx >= pool_len")
            tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_prefill: pool page OOB")
        tok = phys * page_size + idx_s % page_size
        scores = tl.zeros((HP, BLOCK_N), tl.float32)
        for kc in tl.range(0, D, KC, num_stages = KSTAGES):
            kcols = kc + kcols0
            qk = tl.load(q_rows + kcols[None, :])
            if QC > 0:
                qk = _rot_cols(qk, h32, kcols, D_c, HP, KC)
            k = _pool_tile(pool_c, pool_r, pool_s, tok, in_range, kcols, kc, D_c, D_r, QC, KC)
            scores = tl.dot(qk, tl.trans(k), acc = scores)
        v = _pool_tile(pool_c, pool_r, pool_s, tok, in_range, vcols, c0, D_c, D_r, QC, BD)
        m_state, l, acc = _softmax_step(scores, in_range, m_state, l, acc, v, scale)

    # Epilogue: normalize, rotate packed-pool columns back, de-rotate the rope pairs, store
    denom = tl.where(l == 0.0, 1.0, l)
    o = acc / denom[:, None]
    if QC > 0:
        # Latent columns were accumulated in the H32 domain (involutory); rope columns pass
        o = _rot_cols(o.to(tl.float16), h32, vcols, D_c, HP, BD).to(tl.float32)
    if DEROTATE:
        # GPT-J pairs (2i, 2i+1) of the rope columns at the query's absolute position;
        # latent pairs get theta 0 (identity)
        pc = c0 + 2 * tl.arange(0, BD // 2)
        is_r = pc >= D_c
        fi = tl.where(is_r, (pc - D_c) // 2, 0)
        theta = tl.load(derot_inv_freq + fi, mask = is_r, other = 0.0) * (q_pos0 + row)
        cos = tl.cos(theta)[None, :]
        sin = tl.sin(theta)[None, :]
        o_e, o_o = tl.split(tl.reshape(o, (HP, BD // 2, 2)))
        o = tl.interleave(o_e * cos - o_o * sin, o_o * cos + o_e * sin)

    if HPG > 0:
        base_h = (offs_h // HPG) * (R * HPG * D) + row * (HPG * D) + (offs_h % HPG) * D
    else:
        base_h = (row * H + offs_h) * D
    tl.store(out + base_h[:, None] + vcols[None, :], o.to(tl.float16))


# Production tilings (see the kernels): the decode split kernel and the one-shot prefill kernel
DECODE_BLOCK_H = 16     # the combine's head block: workspace layout and grid basis
DECODE_HP = 32
DECODE_BLOCK_N = 32
DECODE_BLOCK_W = 32
DECODE_KC = 64
DECODE_KSTAGES = 1
DECODE_WARPS = 4
DECODE_SPLITS = 8       # graphed decode (bc_dsa)

PREFILL_HP = 64
PREFILL_BD = 512
PREFILL_KC = 32
PREFILL_BLOCK_N = 64
PREFILL_BLOCK_W = 64
PREFILL_WARPS = 16
PREFILL_KSTAGES = 1


def _decode_tiling(H, D, block_h, hp):
    """(HP, BD) for a combine head block, or None when the decode kernel cannot tile the shape"""
    if D & (D - 1) or H % block_h or hp > H or H % hp:
        return None
    hb_c = H // block_h
    nhg = H // hp
    if hb_c % nhg:
        return None
    ndb = hb_c // nhg
    if D % ndb or (D // ndb) % 32 or D // ndb < 16:
        return None
    return hp, D // ndb


def decode_eligible(consts: dict):
    """(HP, BD) when a _dsa_attn_split_kernel constexpr set (with its BLOCK_H) can run on the MQA decode kernel"""
    if consts.get("Q_SPLIT", 0) or consts.get("OUT_LATENT", 0):
        return None
    H, D_c, D_r = consts["H"], consts["D_c"], consts["D_r"]
    if D_r <= 0 or (consts.get("QC", 0) and D_c % 32):
        return None
    for hp in (min(DECODE_HP, H), 16):
        if hp <= H:
            t = _decode_tiling(H, D_c + D_r, consts["BLOCK_H"], hp)
            if t is not None:
                return t
    return None


def decode_consts(consts: dict, t) -> dict:
    """The decode kernel's constexprs for an upstream split-kernel constexpr set and its tiling"""
    c = dict(consts)
    c.update(HP = t[0], BD = t[1], KC = DECODE_KC, KSTAGES = DECODE_KSTAGES,
             BLOCK_N = DECODE_BLOCK_N, BLOCK_W = DECODE_BLOCK_W)
    return c


def prefill_eligible(consts: dict):
    """(HP, BD) when a _dsa_attn_kernel constexpr set can run on the MQA prefill kernel"""
    if consts.get("Q_SPLIT", 0) or consts.get("OUT_LATENT", 0):
        return None
    H, D_c, D_r = consts["H"], consts["D_c"], consts["D_r"]
    D = D_c + D_r
    if D_r <= 0 or D & (D - 1) or (consts.get("QC", 0) and D_c % 32):
        return None
    hp = min(PREFILL_HP, H)
    bd = min(PREFILL_BD, D)
    if H % hp or hp < 16 or D % bd or bd % 32 or D % PREFILL_KC:
        return None
    return hp, bd


def prefill_consts(consts: dict, t) -> dict:
    c = dict(consts)
    c.update(HP = t[0], BD = t[1], KC = PREFILL_KC, KSTAGES = PREFILL_KSTAGES,
             BLOCK_N = PREFILL_BLOCK_N, BLOCK_W = PREFILL_BLOCK_W)
    return c
