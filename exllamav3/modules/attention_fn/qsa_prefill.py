"""Opt-in bounded Q8 QSA prefill for the validated Flash-Next/SM121 layout."""

import torch
import triton
import triton.language as tl

from .triton_paged import (
    _get_h32, _qc_load_kt, _qc_load_v, _rot_h32,
    _paged_attn_decode_combine_kernel, combine_subtiles,
)

MAX_CONTEXT_TOKENS = 262144
_ZEROS = {}


def try_attend(q, k, v, indices, sm_scale, block_table=None, page_size=0,
               qc=None, n_kv_heads=None, *, context=None):
    # Autosplit can probe the whole shared pool, larger than a request window.
    # Fall back before any CUDA query or staging allocation, never clip indices.
    if context is None:
        return None
    rows = q.shape[0]
    length = context["length"]
    if not 256 <= rows <= length <= MAX_CONTEXT_TOKENS:
        return None
    if (q.shape[1:] != (24, 256) or n_kv_heads != 2 or qc is None
            or tuple(qc[2:]) != (8, 8) or indices.shape != (rows, 2080)
            or block_table is None or block_table.shape[0] != rows or page_size <= 0):
        return None
    table = context["block_table"]
    if table.ndim != 1 or table.numel() < triton.cdiv(length, page_size):
        return None
    tensors = (q, k, v, indices, block_table, table, qc[0], qc[1])
    if any(not t.is_cuda or t.device != q.device or not t.is_contiguous() for t in tensors):
        return None
    if (q.dtype != torch.float16 or k.dtype != torch.int32 or v.dtype != torch.int32
            or indices.dtype != torch.int32 or table.dtype != torch.int32):
        return None
    if torch.cuda.get_device_capability(q.device) != (12, 1):
        return None
    with torch.cuda.device(q.device):
        return _attend(q, k, v, indices, sm_scale, block_table, page_size, qc,
                       n_kv_heads, context)


def _attend(q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads, context):
    rows = q.shape[0]
    if context["length"] != rows:
        return _sparse_attend(q, k, v, indices, sm_scale, block_table, page_size,
                              qc, n_kv_heads, context)
    from .triton_paged import paged_attn_triton_prefill
    if q.device not in _ZEROS:
        _ZEROS[q.device] = torch.zeros((1,), dtype=torch.int32, device=q.device)
    count = min(2048, rows)
    # Keep the native rotated Q8 domain, avoiding a second precision boundary.
    # A call-local override avoids changing another request's staging policy.
    prefix = paged_attn_triton_prefill(
        q[:count].unsqueeze(0), None, None, k, v, block_table[:1], _ZEROS[q.device],
        causal=True, softmax_scale=sm_scale, qc=qc, pre_appended_len=count,
        max_kv_len=0, n_kv_heads_override=n_kv_heads, qc_staging=False)
    if rows == count:
        return prefix.squeeze(0)
    rest = _sparse_attend(q[count:], k, v, indices[count:], sm_scale, block_table[count:],
                          page_size, qc, n_kv_heads, context)
    return torch.cat((prefix.squeeze(0), rest), dim=0)


@triton.jit
def _prefix_extent(indices, extent, K_PAD: tl.constexpr, PREFIX: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, PREFIX)
    ix = tl.load(indices + row * K_PAD + col)
    last = tl.max(tl.where(ix >= 0, col + 1, 0), 0)
    tl.store(extent + row, tl.cdiv(last, 32) * 32)


@triton.jit
def _stage_qkv(k, v, ks, vs, bt, sk, sv, length,
               PAGE: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr,
               KB: tl.constexpr, VB: tl.constexpr):
    logical = tl.program_id(0) * 32 + tl.arange(0, 32)
    head = tl.program_id(1)
    valid = logical < length
    page = tl.load(bt + logical // PAGE, valid, 0)
    physical = page * PAGE + logical % PAGE
    d = tl.arange(0, HD)
    kt = _qc_load_kt(k, ks, physical, head, d, valid, KB, KVH, HD, HD)
    vt = _qc_load_v(v, vs, physical, head, d, valid, VB, KVH, HD, HD)
    offset = (logical[:, None] * KVH + head) * HD + d[None, :]
    tl.store(sk + offset, tl.trans(kt), valid[:, None])
    tl.store(sv + offset, vt, valid[:, None])


def stage_qkv(k, v, qc, context, page_size, kvh, hd):
    length = context["length"]
    if not 0 < length <= 262144:
        raise ValueError("QSA staging exceeds the bounded context window")
    sk = torch.empty((length, kvh, hd), device=k.device, dtype=torch.float16)
    sv = torch.empty_like(sk)
    ks, vs, kb, vb = qc
    _stage_qkv[(triton.cdiv(length, 32), kvh)](
        k, v, ks, vs, context["block_table"], sk, sv, length,
        page_size, kvh, hd, kb, vb, num_warps=4,
    )
    return sk, sv


@triton.jit(do_not_specialize=["k_len", "num_pages", "splits", "split_len"])
def _sparse_kernel(
    q, k, v, bt, indices, partial_o, partial_ml, output, extent,
    k_len, num_pages, splits, split_len, ks, vs, h32,
    H: tl.constexpr, KVH: tl.constexpr, PAGE: tl.constexpr, HD: tl.constexpr,
    KP: tl.constexpr, SCALE: tl.constexpr, BH: tl.constexpr, BN: tl.constexpr,
    PAGED: tl.constexpr, QCK: tl.constexpr, QCV: tl.constexpr,
    DIRECT: tl.constexpr, CLIP: tl.constexpr, STAGED: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    group = H // KVH
    blocks = tl.cdiv(group, BH)
    hb = pid % blocks
    bh = pid // blocks
    batch = bh // KVH
    kh = bh - batch * KVH
    rows = tl.arange(0, BH)
    local_h = hb * BH + rows
    qh = kh * group + local_h
    valid_row = local_h < group
    d = tl.arange(0, HD)
    qbase = (batch * H + qh) * HD
    qt = tl.load(q + qbase[:, None] + d[None, :], valid_row[:, None], 0.0)
    if QCK > 0:
        qt = _rot_h32(qt, h32, BH, HD)
    start = split * split_len
    end = tl.minimum(start + split_len, k_len)
    if CLIP:
        # Expanded pools occupy [0,2048); an incomplete tail stays at 2048,
        # even on the very first query. Never truncate away that tail.
        prefix_end = tl.load(extent + batch)
        end = prefix_end + BN
    m = tl.full((BH,), -float("inf"), tl.float32)
    l = tl.full((BH,), 0.0, tl.float32)
    acc = tl.zeros((BH, HD), tl.float32)
    for step in range(start, end, BN):
        n0 = step
        if CLIP:
            n0 = tl.where(step < prefix_end, step, 2048)
        n = n0 + tl.arange(0, BN)
        idx = tl.load(indices + batch * KP + n, n < k_len, -1)
        valid_n = (idx >= 0) & (n < k_len)
        safe_idx = tl.where(valid_n, idx, 0)
        if PAGED:
            phys = tl.load(bt + batch * num_pages + safe_idx // PAGE, valid_n, 0)
            tok = phys * PAGE + safe_idx % PAGE
        else:
            tok = safe_idx
        if QCK > 0 and not STAGED:
            kt = _qc_load_kt(k, ks, tok, kh, d, valid_n, QCK, KVH, HD, HD)
        else:
            kt = tl.load(k + (tok[None, :] * KVH + kh) * HD + d[:, None], valid_n[None, :], 0.0)
        scores = tl.dot(qt, kt) * SCALE
        valid = valid_row[:, None] & valid_n[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m, tl.max(scores, 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.where(valid, tl.exp(scores - m_exp[:, None]), 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l = l * alpha + tl.sum(p, 1)
        if QCV > 0 and not STAGED:
            vt = _qc_load_v(v, vs, tok, kh, d, valid_n, QCV, KVH, HD, HD)
        else:
            vt = tl.load(v + (tok[:, None] * KVH + kh) * HD + d[None, :], valid_n[:, None], 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
        m = m_new
    if DIRECT:
        result = acc / tl.where(l == 0.0, 1.0, l)[:, None]
        if QCV > 0:
            result = _rot_h32(result, h32, BH, HD)
        tl.store(output + qbase[:, None] + d[None, :], result, valid_row[:, None])
    else:
        base = (pid * splits + split) * BH
        tl.store(partial_o + (base + rows[:, None]) * HD + d[None, :], acc)
        tl.store(partial_ml + (base + rows) * 2, m)
        tl.store(partial_ml + (base + rows) * 2 + 1, l)


def _sparse_attend(q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads, context):
    from .qsa_triton import _get_sms, _qsa_sparse_attend_rows_native
    rows, heads, hd = q.shape
    if rows < 256:
        return _qsa_sparse_attend_rows_native(
            q, k, v, indices, sm_scale, block_table, page_size, qc, n_kv_heads)
    ks, vs, kb, vb = qc
    kvh, h32 = n_kv_heads, _get_h32(q.device)
    k, v = stage_qkv(k, v, qc, context, page_size, kvh, hd)
    bh, bn = 16, 32
    programs = rows * kvh * triton.cdiv(heads // kvh, bh)
    kp = indices.shape[1]
    splits = max(1, min(2 * _get_sms(q.device) // programs, triton.cdiv(kp, bn), 128))
    direct = splits == 1
    clip = splits == 1 and kp == 2080
    out = torch.empty_like(q)
    po = q if direct else torch.empty(programs * splits * bh * hd, device=q.device, dtype=torch.float32)
    ml = q if direct else torch.empty(programs * splits * bh * 2, device=q.device, dtype=torch.float32)
    extent = indices
    if clip:
        extent = torch.empty(rows, device=q.device, dtype=torch.int32)
        _prefix_extent[(rows,)](indices, extent, kp, 2048)
    _sparse_kernel[(programs, splits)](
        q, k, v, indices, indices, po, ml, out, extent, kp, 0,
        splits, triton.cdiv(triton.cdiv(kp, splits), bn) * bn, ks, vs, h32,
        heads, kvh, 1, hd, kp, float(sm_scale), bh, bn,
        False, kb, vb, direct, clip, True, num_warps=4, num_stages=2)
    if not direct:
        rs, ds = combine_subtiles(bh, hd)
        _paged_attn_decode_combine_kernel[(programs, (bh // rs) * (hd // ds))](
            po, ml, out, h32, splits, ml, QCV=vb, HAS_SINKS=False, q_len=1,
            n_q_heads=heads, n_kv_heads=kvh, head_dim=hd, HD_PAD=hd, V_DIM=hd,
            BLOCK_M=1, BLOCK_H=bh, BLOCK_ROWS=bh, ROWS_SUB=rs, D_SUB=ds,
            num_warps=4, num_stages=1)
    return out
