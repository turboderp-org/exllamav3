import math

import torch

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

    # Triton dummy functions so import doesn't break if Triton is unavailable
    class _DummyTritonLanguage:
        constexpr = object()

    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn

    triton = _DummyTriton()
    tl = _DummyTritonLanguage()

from .common import AttnArgs, get_non_causal_span_arglist


def _is_power_of_2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@triton.jit
def _paged_kv_update_kernel(
    k,
    v,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    num_pages_per_seq,   # runtime: block-table width can grow without recompiling
    kv_append_len: tl.constexpr,
    n_kv_heads: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    batch = pid_bt // kv_append_len
    t = pid_bt - batch * kv_append_len
    logical_t = tl.load(cache_seqlens + batch) + t
    page = logical_t // page_size
    page_off = logical_t - page * page_size
    phys = tl.load(block_table + batch * num_pages_per_seq + page)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    src = (((batch * kv_append_len + t) * n_kv_heads + pid_h) * head_dim + offs_d)
    dst = (((phys * page_size + page_off) * n_kv_heads + pid_h) * head_dim + offs_d)
    mask = offs_d < head_dim
    tl.store(k_cache + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
    tl.store(v_cache + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)


@triton.jit
def _paged_attn_splitdv_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    out,
    sinks,
    q_len: tl.constexpr,
    kv_append_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    num_pages_per_seq: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bhm = tl.program_id(0)
    pid_dv = tl.program_id(1)

    q_block = pid_bhm % tl.cdiv(q_len, BLOCK_M)
    bh = pid_bhm // tl.cdiv(q_len, BLOCK_M)
    batch = bh // n_q_heads
    q_head = bh - batch * n_q_heads
    group_size = n_q_heads // n_kv_heads
    kv_head = q_head // group_size

    offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)

    q_ptrs = q + (((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :])
    q_tile = tl.load(q_ptrs, mask=offs_m[:, None] < q_len, other=0.0)

    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len
    q_abs = total_k_len - q_len + offs_m

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DV), tl.float32)

    for n0 in range(0, total_k_len, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        page = offs_n // page_size
        page_off = offs_n - page * page_size
        phys = tl.load(
            block_table + batch * num_pages_per_seq + page,
            mask=offs_n < total_k_len,
            other=0,
        )

        k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
        k_tile = tl.load(k_ptrs, mask=offs_n[None, :] < total_k_len, other=0.0)
        scores = tl.dot(q_tile, k_tile) * scale
        if SOFTCAP > 0.0:
            scores_scaled = scores / SOFTCAP
            scores = (2.0 / (1.0 + tl.exp(-2.0 * scores_scaled)) - 1.0) * SOFTCAP

        valid = (offs_m[:, None] < q_len) & (offs_n[None, :] < total_k_len)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_abs[:, None])
        if WINDOW_LEFT >= 0:
            valid = valid & (offs_n[None, :] >= q_abs[:, None] - WINDOW_LEFT)
        if WINDOW_RIGHT >= 0:
            valid = valid & (offs_n[None, :] <= q_abs[:, None] + WINDOW_RIGHT)
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l_new = l * alpha + tl.sum(p, axis=1)

        v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_dv[None, :])
        v_tile = tl.load(
            v_ptrs,
            mask=(offs_n[:, None] < total_k_len) & (offs_dv[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new
        l = l_new

    if HAS_SINKS:
        # Learned per-head sink: one extra exp(sink - m) term in the softmax denominator,
        # contributing no value (gpt-oss style)
        sink = tl.load(sinks + q_head).to(tl.float32)
        m_top = tl.maximum(m, sink)
        alpha_s = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_top))
        acc = acc * alpha_s[:, None]
        l = l * alpha_s + tl.exp(sink - m_top)
    out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
    out_ptrs = out + (((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_dv[None, :])
    tl.store(out_ptrs, out_tile, mask=(offs_m[:, None] < q_len) & (offs_dv[None, :] < head_dim))


@triton.jit
def _paged_attn_longq_grouped_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    out,
    sinks,
    q_len: tl.constexpr,
    kv_append_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    num_pages_per_seq: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_dv = tl.program_id(1)

    group_size = n_q_heads // n_kv_heads
    h_blocks = tl.cdiv(group_size, BLOCK_H)
    q_blocks = tl.cdiv(q_len, BLOCK_M)

    h_block = pid % h_blocks
    q_block = (pid // h_blocks) % q_blocks
    bh = pid // (h_blocks * q_blocks)
    batch = bh // n_kv_heads
    kv_head = bh - batch * n_kv_heads

    rows = tl.arange(0, BLOCK_ROWS)
    row_q = q_block * BLOCK_M + (rows % BLOCK_M)
    row_h_local = h_block * BLOCK_H + (rows // BLOCK_M)
    q_head = kv_head * group_size + row_h_local

    offs_d = tl.arange(0, head_dim)
    offs_dv = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    valid_row = (row_q < q_len) & (row_h_local < group_size)

    q_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
    q_tile = tl.load(q + q_base[:, None] + offs_d[None, :], mask=valid_row[:, None], other=0.0)

    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len
    q_abs = total_k_len - q_len + row_q

    m = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_ROWS,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_ROWS, BLOCK_DV), tl.float32)

    for n0 in range(0, total_k_len, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        page = offs_n // page_size
        page_off = offs_n - page * page_size
        phys = tl.load(
            block_table + batch * num_pages_per_seq + page,
            mask=offs_n < total_k_len,
            other=0,
        )

        k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
        k_tile = tl.load(k_ptrs, mask=offs_n[None, :] < total_k_len, other=0.0)
        scores = tl.dot(q_tile, k_tile) * scale
        if SOFTCAP > 0.0:
            scores_scaled = scores / SOFTCAP
            scores = (2.0 / (1.0 + tl.exp(-2.0 * scores_scaled)) - 1.0) * SOFTCAP

        valid = valid_row[:, None] & (offs_n[None, :] < total_k_len)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_abs[:, None])
        if WINDOW_LEFT >= 0:
            valid = valid & (offs_n[None, :] >= q_abs[:, None] - WINDOW_LEFT)
        if WINDOW_RIGHT >= 0:
            valid = valid & (offs_n[None, :] <= q_abs[:, None] + WINDOW_RIGHT)
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l_new = l * alpha + tl.sum(p, axis=1)

        v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_dv[None, :])
        v_tile = tl.load(
            v_ptrs,
            mask=(offs_n[:, None] < total_k_len) & (offs_dv[None, :] < head_dim),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new
        l = l_new

    if HAS_SINKS:
        sink = tl.load(sinks + q_head, mask=valid_row, other=0.0).to(tl.float32)
        m_top = tl.maximum(m, sink)
        alpha_s = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_top))
        acc = acc * alpha_s[:, None]
        l = l * alpha_s + tl.exp(sink - m_top)
    out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
    out_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
    tl.store(
        out + out_base[:, None] + offs_dv[None, :],
        out_tile,
        mask=valid_row[:, None] & (offs_dv[None, :] < head_dim),
    )


def _normalize_window(window_size):
    if window_size is None:
        return -1, -1
    if isinstance(window_size, int):
        return (-1, -1) if window_size < 0 else (window_size, 0)
    if len(window_size) != 2:
        raise ValueError("window_size must be None, an int, or a (left, right) tuple")
    return int(window_size[0]), int(window_size[1])


def _check_tensor(name: str, tensor: torch.Tensor, dtype: torch.dtype | None = torch.float16):
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _prep_sinks(sinks: torch.Tensor | None, n_q_heads: int, dummy: torch.Tensor):
    """Validate a learned attention-sinks tensor (one logit per q head, gpt-oss style) and
    return (pointer arg, HAS_SINKS). The caller's dummy stands in when sinks are absent."""
    if sinks is None:
        return dummy, False
    if sinks.shape != (n_q_heads,):
        raise ValueError("sinks must have shape (n_q_heads,)")
    if sinks.dtype != torch.float32 or not sinks.is_contiguous():
        sinks = sinks.float().contiguous()
    return sinks, True


def _same_device(*tensors: torch.Tensor | None) -> bool:
    device = None
    for tensor in tensors:
        if tensor is None:
            continue
        if device is None:
            device = tensor.device
        elif tensor.device != device:
            return False
    return True


def paged_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: int | tuple[int, int] | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int = 64,
    block_dv: int | None = None,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Paged KV-cache attention with GQA and optional in-place cache append.

    Tensor layouts match flash_attn_with_kvcache for the supported path:
    q/k/v are [batch, seq, heads, dim], caches are [pages, page_size, kv_heads, dim],
    block_table is [batch, pages_per_seq], and cache_seqlens is the pre-append length.
    """
    if not has_triton:
        raise RuntimeError("paged_attn_triton requires Triton, but Triton is not available")

    _check_tensor("q", q)
    _check_tensor("k_cache", k_cache)
    _check_tensor("v_cache", v_cache)
    _check_tensor("block_table", block_table, None)
    _check_tensor("cache_seqlens", cache_seqlens, None)

    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("q, k_cache and v_cache must be rank-4 tensors")
    if block_table.ndim != 2 or cache_seqlens.ndim != 1:
        raise ValueError("block_table must be rank-2 and cache_seqlens rank-1")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table must be int32 or int64")
    if cache_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache_seqlens must be int32 or int64")
    if not _same_device(q, k_cache, v_cache, block_table, cache_seqlens):
        raise ValueError("q, caches, block_table and cache_seqlens must be on the same CUDA device")

    bsz, q_len, n_q_heads, head_dim = q.shape
    _, page_size, n_kv_heads, cache_dim = k_cache.shape
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if cache_dim != head_dim:
        raise ValueError("q and cache head dimensions must match")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")
    if block_table.shape[0] != bsz or cache_seqlens.shape[0] != bsz:
        raise ValueError("batch dimensions do not match")
    if head_dim > 512 or not _is_power_of_2(head_dim):
        raise ValueError("paged_attn_triton currently supports power-of-two head_dim <= 512")

    kv_append_len = 0
    if k is not None or v is not None:
        if k is None or v is None:
            raise ValueError("k and v must be provided together")
        _check_tensor("k", k)
        _check_tensor("v", v)
        if not _same_device(q, k, v):
            raise ValueError("q, k and v must be on the same CUDA device")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if k.shape[:1] != (bsz,) or k.shape[2:] != (n_kv_heads, head_dim):
            raise ValueError("k/v shape must be [batch, seqlen_new, kv_heads, head_dim]")
        kv_append_len = k.shape[1]

    if out is None:
        out = torch.empty_like(q)
    else:
        _check_tensor("out", out)
        if out.shape != q.shape:
            raise ValueError("out must have the same shape as q")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    window_left, window_right = _normalize_window(window_size)
    sinks, has_sinks = _prep_sinks(sinks, n_q_heads, q)

    if block_m is None:
        block_m = 16
    if block_dv is None:
        block_dv = 64 if q_len <= 16 else min(128, head_dim)
    for name, value in (("block_m", block_m), ("block_n", block_n), ("block_dv", block_dv)):
        if not _is_power_of_2(value):
            raise ValueError(f"{name} must be a power of two")
    if block_dv > head_dim:
        block_dv = head_dim
    num_pages_per_seq = block_table.shape[1]

    with torch.cuda.device(q.device):
        if k is not None and kv_append_len:
            update_block_d = triton.next_power_of_2(head_dim)
            update_grid = (bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))
            _paged_kv_update_kernel[update_grid](
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                cache_seqlens,
                num_pages_per_seq,
                kv_append_len,
                n_kv_heads,
                page_size,
                head_dim,
                update_block_d,
                num_warps=2,
                num_stages=3,
            )

        q_blocks = triton.cdiv(q_len, block_m)
        attn_grid = (bsz * n_q_heads * q_blocks, triton.cdiv(head_dim, block_dv))
        _paged_attn_splitdv_kernel[attn_grid](
            q,
            k_cache,
            v_cache,
            block_table,
            cache_seqlens,
            out,
            sinks,
            q_len,
            kv_append_len,
            n_q_heads,
            n_kv_heads,
            num_pages_per_seq,
            page_size,
            head_dim,
            float(softmax_scale),
            bool(causal),
            int(window_left),
            int(window_right),
            float(softcap or 0.0),
            has_sinks,
            block_m,
            block_n,
            block_dv,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def paged_attn_triton_longq(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: int | tuple[int, int] | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_m: int = 8,
    block_h: int = 4,
    block_n: int = 32,
    block_dv: int | None = None,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """Long-query paged attention path that groups GQA sibling Q heads per program."""
    if not has_triton:
        raise RuntimeError("paged_attn_triton_longq requires Triton, but Triton is not available")

    _check_tensor("q", q)
    _check_tensor("k_cache", k_cache)
    _check_tensor("v_cache", v_cache)
    _check_tensor("block_table", block_table, None)
    _check_tensor("cache_seqlens", cache_seqlens, None)

    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("q, k_cache and v_cache must be rank-4 tensors")
    if block_table.ndim != 2 or cache_seqlens.ndim != 1:
        raise ValueError("block_table must be rank-2 and cache_seqlens rank-1")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table must be int32 or int64")
    if cache_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache_seqlens must be int32 or int64")
    if not _same_device(q, k_cache, v_cache, block_table, cache_seqlens):
        raise ValueError("q, caches, block_table and cache_seqlens must be on the same CUDA device")

    bsz, q_len, n_q_heads, head_dim = q.shape
    _, page_size, n_kv_heads, cache_dim = k_cache.shape
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if cache_dim != head_dim:
        raise ValueError("q and cache head dimensions must match")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")
    if block_table.shape[0] != bsz or cache_seqlens.shape[0] != bsz:
        raise ValueError("batch dimensions do not match")
    if head_dim > 512 or not _is_power_of_2(head_dim):
        raise ValueError("paged_attn_triton_longq currently supports power-of-two head_dim <= 512")

    kv_append_len = 0
    if k is not None or v is not None:
        if k is None or v is None:
            raise ValueError("k and v must be provided together")
        _check_tensor("k", k)
        _check_tensor("v", v)
        if not _same_device(q, k, v):
            raise ValueError("q, k and v must be on the same CUDA device")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if k.shape[:1] != (bsz,) or k.shape[2:] != (n_kv_heads, head_dim):
            raise ValueError("k/v shape must be [batch, seqlen_new, kv_heads, head_dim]")
        kv_append_len = k.shape[1]

    if out is None:
        out = torch.empty_like(q)
    else:
        _check_tensor("out", out)
        if out.shape != q.shape:
            raise ValueError("out must have the same shape as q")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    window_left, window_right = _normalize_window(window_size)
    sinks, has_sinks = _prep_sinks(sinks, n_q_heads, q)

    if block_dv is None:
        block_dv = min(512, head_dim)
    block_rows = block_m * block_h
    for name, value in (
        ("block_m", block_m),
        ("block_h", block_h),
        ("block_rows", block_rows),
        ("block_n", block_n),
        ("block_dv", block_dv),
    ):
        if not _is_power_of_2(value):
            raise ValueError(f"{name} must be a power of two")
    if block_dv > head_dim:
        block_dv = head_dim

    num_pages_per_seq = block_table.shape[1]
    with torch.cuda.device(q.device):
        if k is not None and kv_append_len:
            update_block_d = triton.next_power_of_2(head_dim)
            _paged_kv_update_kernel[(bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))](
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                cache_seqlens,
                num_pages_per_seq,
                kv_append_len,
                n_kv_heads,
                page_size,
                head_dim,
                update_block_d,
                num_warps=2,
                num_stages=3,
            )

        group_size = n_q_heads // n_kv_heads
        grid0 = bsz * n_kv_heads * triton.cdiv(q_len, block_m) * triton.cdiv(group_size, block_h)
        _paged_attn_longq_grouped_kernel[(grid0, triton.cdiv(head_dim, block_dv))](
            q,
            k_cache,
            v_cache,
            block_table,
            cache_seqlens,
            out,
            sinks,
            q_len,
            kv_append_len,
            n_q_heads,
            n_kv_heads,
            num_pages_per_seq,
            page_size,
            head_dim,
            float(softmax_scale),
            bool(causal),
            int(window_left),
            int(window_right),
            float(softcap or 0.0),
            has_sinks,
            block_m,
            block_rows,
            block_n,
            block_dv,
            block_h,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def fn_triton_paged_attn(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_triton or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.q_len > 256
    ):
        return None

    if args.non_causal_spans:
        arglist = get_non_causal_span_arglist(args)
        return torch.cat([paged_attn_triton(**a) for a in arglist], dim=1)

    return paged_attn_triton(
        q=args.q,
        k=args.k,
        v=args.v,
        k_cache=args.k_cache,
        v_cache=args.v_cache,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
        window_size=args.get_window_size(),
        softcap=args.softcap,
        sinks=args.sinks,
    )


def fn_triton_paged_attn_longq(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_triton or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.q_len <= 256 or
        args.dim > 512 or
        not _is_power_of_2(args.dim)
    ):
        return None

    if args.non_causal_spans:
        arglist = get_non_causal_span_arglist(args)
        return torch.cat([paged_attn_triton_longq(**a) for a in arglist], dim=1)

    return paged_attn_triton_longq(
        q=args.q,
        k=args.k,
        v=args.v,
        k_cache=args.k_cache,
        v_cache=args.v_cache,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
        window_size=args.get_window_size(),
        softcap=args.softcap,
        sinks=args.sinks,
    )


# Quantized-cache support: the cache stores t = quant(H32 x / sqrt(32) / s) per 32-value group
# (see exllamav3_ext/cache/q_cache_kernels.cuh). Because the rotation is orthonormal and block-
# diagonal (never crossing heads), it folds out of the attention inner loop entirely:
# q . k_hat = (H q / sqrt(32)) . (t s), and the output only needs one inverse rotation at the
# end: o = H o_lin / sqrt(32). So the in-loop dequant is a plain unpack-and-scale, and q/output
# are rotated once per program with a 32x32 tl.dot against Hs = H32 / sqrt(32) (involutory).

_h32_cache = {}

def _get_h32(device):
    if device not in _h32_cache:
        h = torch.ones(1, 1, dtype = torch.float32)
        while h.shape[0] < 32:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _h32_cache[device] = (h / 32 ** 0.5).to(device, torch.float16).contiguous()
    return _h32_cache[device]


@triton.jit
def _rot_h32(x, h32, ROWS: tl.constexpr, head_dim: tl.constexpr):
    h = tl.load(h32 + tl.arange(0, 32)[:, None] * 32 + tl.arange(0, 32)[None, :])
    x2 = tl.reshape(x.to(tl.float16), (ROWS * head_dim // 32, 32))
    y = tl.dot(x2, h)
    return tl.reshape(y.to(tl.float16), (ROWS, head_dim))


@triton.jit
def _qc_plane_kt(qwords_head, row_words, mask_n, pbase,
                 W: tl.constexpr, BITS: tl.constexpr, head_dim: tl.constexpr):
    """One power-of-two bit plane, (head_dim, BLOCK_N) int32: compact coalesced word tile
    expanded with broadcast shifts. Groups of 32 values own BITS words each; this plane's
    words sit at [pbase, pbase + W) within each group."""
    WPH: tl.constexpr = head_dim * W // 32   # words per head slice in this plane
    VPW: tl.constexpr = 32 // W              # values per word
    garr = tl.arange(0, WPH)
    cols = (garr // W) * BITS + pbase + (garr % W)
    w = tl.load(qwords_head + row_words[None, :] + cols[:, None], mask = mask_n[None, :], other = 0)
    nib = (w[:, None, :] >> (tl.arange(0, VPW) * W)[None, :, None]) & ((1 << W) - 1)
    return tl.reshape(nib, (head_dim, w.shape[1]))


@triton.jit
def _qc_plane_v(qwords_head, row_words, mask_n, pbase,
                W: tl.constexpr, BITS: tl.constexpr, head_dim: tl.constexpr):
    """Transposed orientation of _qc_plane_kt: (BLOCK_N, head_dim)."""
    WPH: tl.constexpr = head_dim * W // 32
    VPW: tl.constexpr = 32 // W
    garr = tl.arange(0, WPH)
    cols = (garr // W) * BITS + pbase + (garr % W)
    w = tl.load(qwords_head + row_words[:, None] + cols[None, :], mask = mask_n[:, None], other = 0)
    nib = (w[:, :, None] >> (tl.arange(0, VPW) * W)[None, None, :]) & ((1 << W) - 1)
    return tl.reshape(nib, (w.shape[0], head_dim))


@triton.jit
def _qc_load_kt(qwords, scales, tok_rows, kv_head, offs_d, mask_n,
                BITS: tl.constexpr, n_kv_heads: tl.constexpr, head_dim: tl.constexpr):
    """(head_dim, BLOCK_N) fp16 tile from the packed cache, linear midpoint grid, values stay
    in the rotated domain. The cache packs each group into power-of-two bit planes (BITS = sum
    of its set bits), so every width expands with vectorized shifts; no gathers, no straddling."""
    GPT: tl.constexpr = n_kv_heads * head_dim // 32
    base = kv_head * ((head_dim // 32) * BITS)
    row_words = tok_rows * (GPT * BITS)
    qh = qwords + base
    raw = tl.zeros((1, 1), tl.int32)  # replaced by first plane
    pbase = 0
    first = True
    if BITS & 8:
        raw = _qc_plane_kt(qh, row_words, mask_n, pbase, 8, BITS, head_dim)
        pbase += 8
        first = False
    if BITS & 4:
        p = _qc_plane_kt(qh, row_words, mask_n, pbase, 4, BITS, head_dim)
        raw = p if first else (raw << 4) | p
        pbase += 4
        first = False
    if BITS & 2:
        p = _qc_plane_kt(qh, row_words, mask_n, pbase, 2, BITS, head_dim)
        raw = p if first else (raw << 2) | p
        pbase += 2
        first = False
    if BITS & 1:
        p = _qc_plane_kt(qh, row_words, mask_n, pbase, 1, BITS, head_dim)
        raw = p if first else (raw << 1) | p
    sgb = kv_head * (head_dim // 32)
    sc = tl.load(scales + tok_rows[None, :] * GPT + (sgb + tl.arange(0, head_dim // 32))[:, None],
                 mask = mask_n[None, :], other = 0.0)
    scx = tl.reshape(tl.broadcast_to(sc[:, None, :], (head_dim // 32, 32, sc.shape[1])),
                     (head_dim, sc.shape[1]))
    mh = (1 << (BITS - 1)) - 0.5
    inv_m = 1.0 / (1 << (BITS - 1))
    return ((raw.to(tl.float32) - mh) * (scx.to(tl.float32) * inv_m)).to(tl.float16)


@triton.jit
def _qc_load_v(qwords, scales, tok_rows, kv_head, offs_d, mask_n,
               BITS: tl.constexpr, n_kv_heads: tl.constexpr, head_dim: tl.constexpr):
    """(BLOCK_N, head_dim) fp16 tile, transposed orientation of _qc_load_kt."""
    GPT: tl.constexpr = n_kv_heads * head_dim // 32
    base = kv_head * ((head_dim // 32) * BITS)
    row_words = tok_rows * (GPT * BITS)
    qh = qwords + base
    raw = tl.zeros((1, 1), tl.int32)
    pbase = 0
    first = True
    if BITS & 8:
        raw = _qc_plane_v(qh, row_words, mask_n, pbase, 8, BITS, head_dim)
        pbase += 8
        first = False
    if BITS & 4:
        p = _qc_plane_v(qh, row_words, mask_n, pbase, 4, BITS, head_dim)
        raw = p if first else (raw << 4) | p
        pbase += 4
        first = False
    if BITS & 2:
        p = _qc_plane_v(qh, row_words, mask_n, pbase, 2, BITS, head_dim)
        raw = p if first else (raw << 2) | p
        pbase += 2
        first = False
    if BITS & 1:
        p = _qc_plane_v(qh, row_words, mask_n, pbase, 1, BITS, head_dim)
        raw = p if first else (raw << 1) | p
    sgb = kv_head * (head_dim // 32)
    sc = tl.load(scales + tok_rows[:, None] * GPT + (sgb + tl.arange(0, head_dim // 32))[None, :],
                 mask = mask_n[:, None], other = 0.0)
    scx = tl.reshape(tl.broadcast_to(sc[:, :, None], (sc.shape[0], head_dim // 32, 32)),
                     (sc.shape[0], head_dim))
    mh = (1 << (BITS - 1)) - 0.5
    inv_m = 1.0 / (1 << (BITS - 1))
    return ((raw.to(tl.float32) - mh) * (scx.to(tl.float32) * inv_m)).to(tl.float16)

@triton.jit
def _paged_attn_decode_split_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    out,
    partial_o,
    partial_ml,
    k_scales,
    v_scales,
    h32,
    split_len,           # runtime: derived from the block-table bound, changes as it grows
    num_pages_per_seq,   # runtime: block-table width can grow without recompiling
    num_splits,          # runtime: the grid may be launched wider (graph path); extra splits idle
    sinks,               # last runtime arg: the BC launch appends it after the patched ints
    QCK: tl.constexpr,
    QCV: tl.constexpr,
    q_len: tl.constexpr,
    kv_append_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    SOFTCAP: tl.constexpr,
    FINAL: tl.constexpr,       # num_splits == 1: skip the combine pass, store directly to out
    HAS_SINKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash-decoding phase 1: one program per (batch, kv_head, h_block, kv split). GQA sibling
    q heads and query positions share the row axis so K/V tiles are read once per group."""
    pid = tl.program_id(0)
    split = tl.program_id(1)

    group_size = n_q_heads // n_kv_heads
    h_blocks = tl.cdiv(group_size, BLOCK_H)
    h_block = pid % h_blocks
    bh = pid // h_blocks
    batch = bh // n_kv_heads
    kv_head = bh - batch * n_kv_heads

    rows = tl.arange(0, BLOCK_ROWS)
    row_q = rows % BLOCK_M
    row_h_local = h_block * BLOCK_H + (rows // BLOCK_M)
    q_head = kv_head * group_size + row_h_local
    valid_row = (row_q < q_len) & (row_h_local < group_size)

    offs_d = tl.arange(0, head_dim)
    q_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
    q_tile = tl.load(q + q_base[:, None] + offs_d[None, :], mask=valid_row[:, None], other=0.0)
    if QCK > 0:
        q_tile = _rot_h32(q_tile, h32, BLOCK_ROWS, head_dim)

    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len
    q_abs = total_k_len - q_len + row_q

    n_start = split * split_len
    n_end = tl.minimum(n_start + split_len, total_k_len)

    m = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_ROWS,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_ROWS, head_dim), tl.float32)

    for n0 in range(n_start, n_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        page = offs_n // page_size
        page_off = offs_n - page * page_size
        phys = tl.load(
            block_table + batch * num_pages_per_seq + page,
            mask=offs_n < n_end,
            other=0,
        )

        if QCK > 0:
            tok_rows = phys * page_size + page_off
            k_tile = _qc_load_kt(k_cache, k_scales, tok_rows, kv_head, offs_d, offs_n < n_end, QCK, n_kv_heads, head_dim)
        else:
            k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
            k_tile = tl.load(k_ptrs, mask=offs_n[None, :] < n_end, other=0.0)
        scores = tl.dot(q_tile, k_tile) * scale
        if SOFTCAP > 0.0:
            scores_scaled = scores / SOFTCAP
            scores = (2.0 / (1.0 + tl.exp(-2.0 * scores_scaled)) - 1.0) * SOFTCAP

        valid = valid_row[:, None] & (offs_n[None, :] < n_end)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= q_abs[:, None])
        if WINDOW_LEFT >= 0:
            valid = valid & (offs_n[None, :] >= q_abs[:, None] - WINDOW_LEFT)
        if WINDOW_RIGHT >= 0:
            valid = valid & (offs_n[None, :] <= q_abs[:, None] + WINDOW_RIGHT)
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp(scores - m_exp[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_exp))
        l_new = l * alpha + tl.sum(p, axis=1)

        if QCV > 0:
            tok_rows_v = phys * page_size + page_off
            v_tile = _qc_load_v(v_cache, v_scales, tok_rows_v, kv_head, offs_d, offs_n < n_end, QCV, n_kv_heads, head_dim)
        else:
            v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
            v_tile = tl.load(v_ptrs, mask=offs_n[:, None] < n_end, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new
        l = l_new

    if FINAL:
        if HAS_SINKS:
            sink = tl.load(sinks + q_head, mask=valid_row, other=0.0).to(tl.float32)
            m_top = tl.maximum(m, sink)
            alpha_s = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_top))
            acc = acc * alpha_s[:, None]
            l = l * alpha_s + tl.exp(sink - m_top)
        out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
        if QCV > 0:
            out_tile = _rot_h32(out_tile, h32, BLOCK_ROWS, head_dim)
        out_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
        tl.store(out + out_base[:, None] + offs_d[None, :], out_tile, mask=valid_row[:, None])
    else:
        if split < num_splits:
            po_base = (pid * num_splits + split) * BLOCK_ROWS * head_dim
            tl.store(partial_o + po_base + rows[:, None] * head_dim + offs_d[None, :], acc)
            ml_base = (pid * num_splits + split) * BLOCK_ROWS * 2
            tl.store(partial_ml + ml_base + rows * 2, m)
            tl.store(partial_ml + ml_base + rows * 2 + 1, l)


@triton.jit
def _paged_attn_decode_combine_kernel(
    partial_o,
    partial_ml,
    out,
    h32,
    num_splits,          # runtime
    sinks,               # last runtime arg: the BC launch appends it after the patched int
    QCV: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    q_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """Flash-decoding phase 2: reduce the per-split partial accumulators."""
    pid = tl.program_id(0)

    group_size = n_q_heads // n_kv_heads
    h_blocks = tl.cdiv(group_size, BLOCK_H)
    h_block = pid % h_blocks
    bh = pid // h_blocks
    batch = bh // n_kv_heads
    kv_head = bh - batch * n_kv_heads

    rows = tl.arange(0, BLOCK_ROWS)
    row_q = rows % BLOCK_M
    row_h_local = h_block * BLOCK_H + (rows // BLOCK_M)
    q_head = kv_head * group_size + row_h_local
    valid_row = (row_q < q_len) & (row_h_local < group_size)

    offs_d = tl.arange(0, head_dim)

    m_max = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
    for s in range(num_splits):
        ml_base = (pid * num_splits + s) * BLOCK_ROWS * 2
        m_s = tl.load(partial_ml + ml_base + rows * 2)
        m_max = tl.maximum(m_max, m_s)

    if HAS_SINKS:
        # Learned per-head sink joins the softmax denominator at the final reduction
        sink = tl.load(sinks + q_head, mask=valid_row, other=0.0).to(tl.float32)
        m_max = tl.maximum(m_max, sink)

    l_sum = tl.zeros((BLOCK_ROWS,), tl.float32)
    acc = tl.zeros((BLOCK_ROWS, head_dim), tl.float32)
    m_safe = tl.where(m_max == -float("inf"), 0.0, m_max)
    for s in range(num_splits):
        ml_base = (pid * num_splits + s) * BLOCK_ROWS * 2
        m_s = tl.load(partial_ml + ml_base + rows * 2)
        l_s = tl.load(partial_ml + ml_base + rows * 2 + 1)
        w = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_safe))
        po_base = (pid * num_splits + s) * BLOCK_ROWS * head_dim
        o_s = tl.load(partial_o + po_base + rows[:, None] * head_dim + offs_d[None, :])
        acc += o_s * w[:, None]
        l_sum += l_s * w

    if HAS_SINKS:
        l_sum += tl.exp(sink - m_safe)
    out_tile = acc / tl.where(l_sum[:, None] == 0.0, 1.0, l_sum[:, None])
    if QCV > 0:
        out_tile = _rot_h32(out_tile, h32, BLOCK_ROWS, head_dim)
    out_base = ((batch * q_len + row_q) * n_q_heads + q_head) * head_dim
    tl.store(out + out_base[:, None] + offs_d[None, :], out_tile, mask=valid_row[:, None])


_decode_sm_count = {}

def paged_attn_triton_decode(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: int | tuple[int, int] | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_n: int | None = None,
    num_splits: int | None = None,
    max_kv_len: int | None = None,
    qc: tuple | None = None,            # (k_scales, v_scales, k_bits, v_bits): caches are packed int32
    pre_appended_len: int = 0,          # new tokens already written to the cache; count but don't append
    n_kv_heads_override: int | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Flash-decoding paged attention for short queries: the kv sequence is split across
    programs (sized from the block table, so no host sync on cache_seqlens) and reduced in a
    second pass. GQA sibling q heads share K/V tiles within a program."""
    if not has_triton:
        raise RuntimeError("paged_attn_triton_decode requires Triton, but Triton is not available")

    _check_tensor("q", q)
    _check_tensor("block_table", block_table, None)
    _check_tensor("cache_seqlens", cache_seqlens, None)

    bsz, q_len, n_q_heads, head_dim = q.shape
    if qc is None:
        _check_tensor("k_cache", k_cache)
        _check_tensor("v_cache", v_cache)
        if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
            raise ValueError("q, k_cache and v_cache must be rank-4 tensors")
        _, page_size, n_kv_heads, cache_dim = k_cache.shape
        if v_cache.shape != k_cache.shape:
            raise ValueError("v_cache must have the same shape as k_cache")
        if cache_dim != head_dim:
            raise ValueError("q and cache head dimensions must match")
    else:
        page_size = k_cache.shape[1]
        n_kv_heads = n_kv_heads_override
    if not _same_device(q, k_cache, v_cache, block_table, cache_seqlens):
        raise ValueError("q, caches, block_table and cache_seqlens must be on the same CUDA device")
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")
    if head_dim > 512 or not _is_power_of_2(head_dim):
        raise ValueError("paged_attn_triton_decode currently supports power-of-two head_dim <= 512")
    if q_len > 16:
        raise ValueError("paged_attn_triton_decode supports q_len <= 16")

    kv_append_len = pre_appended_len
    if k is not None or v is not None:
        if k is None or v is None:
            raise ValueError("k and v must be provided together")
        _check_tensor("k", k)
        _check_tensor("v", v)
        if k.shape[:1] != (bsz,) or k.shape[2:] != (n_kv_heads, head_dim):
            raise ValueError("k/v shape must be [batch, seqlen_new, kv_heads, head_dim]")
        kv_append_len = k.shape[1]

    if out is None:
        out = torch.empty_like(q)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    window_left, window_right = _normalize_window(window_size)
    sinks, has_sinks = _prep_sinks(sinks, n_q_heads, q)

    if qc is not None:
        k_scales, v_scales, qck, qcv = qc
        h32 = _get_h32(q.device)
    else:
        k_scales, v_scales, qck, qcv = q, q, 0, 0
        h32 = q

    if block_n is None:
        block_n = max(16, 8192 // head_dim)   # K + V tiles in smem across num_stages

    group_size = n_q_heads // n_kv_heads
    block_m = triton.next_power_of_2(q_len)
    block_h = max(16 // block_m, 1)
    block_rows = block_m * block_h
    h_blocks = triton.cdiv(group_size, block_h)
    num_pages_per_seq = block_table.shape[1]

    # Upper bound on kv length: caller-provided hint, else from the block table shape
    # (cache_seqlens stays on the device; no sync). A loose bound wastes split parallelism when
    # the table is allocated much larger than the live sequence
    max_k_len = num_pages_per_seq * page_size + kv_append_len
    if max_kv_len is not None:
        max_k_len = min(max_k_len, max_kv_len + kv_append_len)

    programs = bsz * n_kv_heads * h_blocks
    if num_splits is None:
        dev = q.device.index
        if dev not in _decode_sm_count:
            _decode_sm_count[dev] = torch.cuda.get_device_properties(q.device).multi_processor_count
        target = 2 * _decode_sm_count[dev]
        num_splits = max(1, min(target // programs, triton.cdiv(max_k_len, 4 * block_n), 128))
    split_len = triton.cdiv(triton.cdiv(max_k_len, num_splits), block_n) * block_n

    if num_splits > 1:
        partial_o = torch.empty(programs * num_splits * block_rows * head_dim, dtype = torch.float32, device = q.device)
        partial_ml = torch.empty(programs * num_splits * block_rows * 2, dtype = torch.float32, device = q.device)
    else:
        partial_o = q   # unused
        partial_ml = q  # unused

    with torch.cuda.device(q.device):
        if k is not None and kv_append_len:
            update_block_d = triton.next_power_of_2(head_dim)
            _paged_kv_update_kernel[(bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))](
                k, v, k_cache, v_cache, block_table, cache_seqlens,
                num_pages_per_seq, kv_append_len, n_kv_heads, page_size, head_dim, update_block_d,
                num_warps=2, num_stages=3,
            )

        _paged_attn_decode_split_kernel[(programs, num_splits)](
            q, k_cache, v_cache, block_table, cache_seqlens, out, partial_o, partial_ml,
            k_scales, v_scales, h32,
            split_len, num_pages_per_seq, num_splits, sinks,
            qck, qcv, q_len, kv_append_len, n_q_heads, n_kv_heads,
            page_size, head_dim, float(softmax_scale),
            bool(causal), int(window_left), int(window_right), float(softcap or 0.0),
            num_splits == 1, has_sinks, block_m, block_h, block_rows, block_n,
            num_warps=num_warps, num_stages=num_stages,
        )

        if num_splits > 1:
            _paged_attn_decode_combine_kernel[(programs,)](
                partial_o, partial_ml, out, h32,
                num_splits, sinks, qcv, has_sinks, q_len, n_q_heads, n_kv_heads, head_dim,
                block_m, block_h, block_rows,
                num_warps=4, num_stages=1,
            )
    return out


def fn_triton_paged_attn_decode(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_triton or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.q_len > 16 or
        args.dim > 512 or
        not _is_power_of_2(args.dim) or
        args.q.dtype != torch.float16 or
        args.k_cache.dtype != torch.float16
    ):
        return None

    if args.non_causal_spans:
        arglist = get_non_causal_span_arglist(args)
        return torch.cat([paged_attn_triton_decode(**a) for a in arglist], dim=1)

    return paged_attn_triton_decode(
        q=args.q,
        k=args.k,
        v=args.v,
        k_cache=args.k_cache,
        v_cache=args.v_cache,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
        window_size=args.get_window_size(),
        softcap=args.softcap,
        sinks=args.sinks,
    )


@triton.jit
def _paged_attn_prefill_inner(
    q_tile, acc, m, l,
    k_cache, v_cache, block_table_b,
    k_scales, v_scales,
    kv_head, offs_n_base, n_start, n_end,
    q_abs, valid_row,
    qk_scale_log2e, total_k_len,
    n_kv_heads: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    QCK: tl.constexpr,
    QCV: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    SOFTCAP: tl.constexpr,
    MASKED: tl.constexpr,
    SRC_NEW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One pass over kv tiles [n_start, n_end). With MASKED = False the tiles are known to be
    fully inside the causal/window region for every row and all bounds/mask logic is skipped.
    With SRC_NEW the k_cache/v_cache arguments are contiguous (rows, kv_heads, head_dim) base
    pointers pre-offset so the absolute kv index addresses them directly (block table unused)."""
    offs_d = tl.arange(0, head_dim)
    for n0 in range(n_start, n_end, BLOCK_N):
        offs_n = n0 + offs_n_base
        page = offs_n // page_size
        page_off = offs_n - page * page_size
        if SRC_NEW:
            phys = tl.zeros((BLOCK_N,), tl.int32)
        elif MASKED:
            phys = tl.load(block_table_b + page, mask = offs_n < n_end, other = 0)
        else:
            phys = tl.load(block_table_b + page)

        if SRC_NEW:
            k_ptrs = k_cache + ((offs_n[None, :] * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
            if MASKED:
                k_tile = tl.load(k_ptrs, mask = offs_n[None, :] < n_end, other = 0.0)
            else:
                k_tile = tl.load(k_ptrs)
        elif QCK > 0:
            tok_rows = phys * page_size + page_off
            if MASKED:
                k_tile = _qc_load_kt(k_cache, k_scales, tok_rows, kv_head, offs_d, offs_n < n_end, QCK, n_kv_heads, head_dim)
            else:
                k_tile = _qc_load_kt(k_cache, k_scales, tok_rows, kv_head, offs_d, offs_n >= 0, QCK, n_kv_heads, head_dim)
        else:
            k_ptrs = k_cache + (((phys[None, :] * page_size + page_off[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None])
            if MASKED:
                k_tile = tl.load(k_ptrs, mask = offs_n[None, :] < n_end, other = 0.0)
            else:
                k_tile = tl.load(k_ptrs)

        scores = tl.dot(q_tile, k_tile)
        if SOFTCAP > 0.0:
            s_nat = scores * (qk_scale_log2e * 0.6931471805599453)  # back to natural units
            s_nat = (2.0 / (1.0 + tl.exp(-2.0 * (s_nat / SOFTCAP))) - 1.0) * SOFTCAP
            scores = s_nat * 1.4426950408889634
        else:
            scores = scores * qk_scale_log2e

        if MASKED:
            valid = valid_row[:, None] & (offs_n[None, :] < n_end)
            if CAUSAL:
                valid = valid & (offs_n[None, :] <= q_abs[:, None])
            if WINDOW_LEFT >= 0:
                valid = valid & (offs_n[None, :] >= q_abs[:, None] - WINDOW_LEFT)
            if WINDOW_RIGHT >= 0:
                valid = valid & (offs_n[None, :] <= q_abs[:, None] + WINDOW_RIGHT)
            scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis = 1))
        if MASKED:
            m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        else:
            m_exp = m_new
        p = tl.exp2(scores - m_exp[:, None])
        if MASKED:
            p = tl.where(valid, p, 0.0)
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp2(m - m_exp))
        else:
            alpha = tl.exp2(m - m_exp)
        l = l * alpha + tl.sum(p, axis = 1)

        if SRC_NEW:
            v_ptrs = v_cache + ((offs_n[:, None] * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
            if MASKED:
                v_tile = tl.load(v_ptrs, mask = offs_n[:, None] < n_end, other = 0.0)
            else:
                v_tile = tl.load(v_ptrs)
        elif QCV > 0:
            tok_rows_v = phys * page_size + page_off
            if MASKED:
                v_tile = _qc_load_v(v_cache, v_scales, tok_rows_v, kv_head, offs_d, offs_n < n_end, QCV, n_kv_heads, head_dim)
            else:
                v_tile = _qc_load_v(v_cache, v_scales, tok_rows_v, kv_head, offs_d, offs_n >= 0, QCV, n_kv_heads, head_dim)
        else:
            v_ptrs = v_cache + (((phys[:, None] * page_size + page_off[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :])
            if MASKED:
                v_tile = tl.load(v_ptrs, mask = offs_n[:, None] < n_end, other = 0.0)
            else:
                v_tile = tl.load(v_ptrs)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new
    return acc, m, l


@triton.jit
def _paged_attn_prefill_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    out,
    partial_o,
    partial_ml,
    k_scales,
    v_scales,
    h32,
    k_new,
    v_new,
    sinks,
    num_splits: tl.constexpr,
    NEW_KV: tl.constexpr,     # 0: all kv in cache; 1: kv >= cache_seqlens[b] read from k_new/v_new;
                              # 2: like 1 but the cache is known empty (keeps the maskless split)
    QCK: tl.constexpr,
    QCV: tl.constexpr,
    q_len: tl.constexpr,
    kv_append_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    num_pages_per_seq: tl.constexpr,
    page_size: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FA2-style prefill over the paged cache: BLOCK_M query rows of one head per program;
    unmasked interior kv tiles take a maskless fast path, only the causal boundary and the
    sequence tail run with masking."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    split = tl.program_id(2)
    batch = bh // n_q_heads
    q_head = bh - batch * n_q_heads
    group_size = n_q_heads // n_kv_heads
    kv_head = q_head // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    valid_row = offs_m < q_len

    q_ptrs = q + (((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :])
    q_tile = tl.load(q_ptrs, mask = valid_row[:, None], other = 0.0)
    if QCK > 0:
        q_tile = _rot_h32(q_tile, h32, BLOCK_M, head_dim)

    total_k_len = tl.load(cache_seqlens + batch) + kv_append_len
    q_abs = total_k_len - q_len + offs_m
    qk_scale_log2e = scale * 1.4426950408889634

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), tl.float32)
    offs_n_base = tl.arange(0, BLOCK_N)
    block_table_b = block_table + batch * num_pages_per_seq

    # Bounds: rows of this block attend to kv [n_lo, n_hi); tiles strictly below the smallest
    # causal boundary in the block (and inside the window for every row) need no masking. With
    # num_splits > 1 the range is further divided across programs and reduced in a second pass
    # (grid-quantization fix: SM counts rarely divide the program count, and at 1 block/SM a
    # partial trailing wave of full-length programs costs up to a third of the runtime)
    q_abs_min = total_k_len - q_len + pid_m * BLOCK_M
    if CAUSAL:
        n_hi = tl.minimum(q_abs_min + BLOCK_M, total_k_len)
    else:
        n_hi = total_k_len
    n_lo = 0
    if WINDOW_LEFT >= 0:
        n_lo = tl.maximum(0, q_abs_min - WINDOW_LEFT)
        n_lo = (n_lo // BLOCK_N) * BLOCK_N
    if num_splits > 1:
        span = tl.cdiv(tl.cdiv(n_hi - n_lo, num_splits), BLOCK_N) * BLOCK_N
        s_lo = n_lo + split * span
        s_hi = tl.minimum(s_lo + span, n_hi)
    else:
        s_lo = n_lo
        s_hi = n_hi

    if NEW_KV > 0:
        # Base pointers for the direct-kv source: absolute kv index n >= past addresses row
        # (n - past) of the (bsz, kv_append_len, kvh, hd) tensors
        past = total_k_len - kv_append_len
        k_new_b = k_new + ((batch * kv_append_len - past) * n_kv_heads) * head_dim
        v_new_b = v_new + ((batch * kv_append_len - past) * n_kv_heads) * head_dim
    else:
        past = total_k_len
        k_new_b = k_new
        v_new_b = v_new

    if NEW_KV == 1:
        # Dual source, everything masked: the boundary at `past` breaks tile alignment, and the
        # only current user (sliding-window attention) runs fully masked regardless
        acc, m, l = _paged_attn_prefill_inner(
            q_tile, acc, m, l, k_cache, v_cache, block_table_b, k_scales, v_scales, kv_head,
            offs_n_base, s_lo, tl.minimum(s_hi, past), q_abs, valid_row, qk_scale_log2e, total_k_len,
            n_kv_heads, page_size, head_dim, QCK, QCV, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
            True, False, BLOCK_N,
        )
        acc, m, l = _paged_attn_prefill_inner(
            q_tile, acc, m, l, k_new_b, v_new_b, block_table_b, k_scales, v_scales, kv_head,
            offs_n_base, tl.maximum(s_lo, past), s_hi, q_abs, valid_row, qk_scale_log2e, total_k_len,
            n_kv_heads, page_size, head_dim, 0, 0, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
            True, True, BLOCK_N,
        )
    elif NEW_KV == 2:
        # Cache known empty: same structure as the cache path, reading the contiguous source
        if CAUSAL and WINDOW_LEFT < 0 and WINDOW_RIGHT < 0:
            n_full = tl.maximum(((q_abs_min + 1) // BLOCK_N) * BLOCK_N, 0)
            acc, m, l = _paged_attn_prefill_inner(
                q_tile, acc, m, l, k_new_b, v_new_b, block_table_b, k_scales, v_scales, kv_head,
                offs_n_base, s_lo, tl.minimum(n_full, s_hi), q_abs, valid_row, qk_scale_log2e, total_k_len,
                n_kv_heads, page_size, head_dim, 0, 0, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
                False, True, BLOCK_N,
            )
            acc, m, l = _paged_attn_prefill_inner(
                q_tile, acc, m, l, k_new_b, v_new_b, block_table_b, k_scales, v_scales, kv_head,
                offs_n_base, tl.maximum(n_full, s_lo), s_hi, q_abs, valid_row, qk_scale_log2e, total_k_len,
                n_kv_heads, page_size, head_dim, 0, 0, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
                True, True, BLOCK_N,
            )
        else:
            acc, m, l = _paged_attn_prefill_inner(
                q_tile, acc, m, l, k_new_b, v_new_b, block_table_b, k_scales, v_scales, kv_head,
                offs_n_base, s_lo, s_hi, q_abs, valid_row, qk_scale_log2e, total_k_len,
                n_kv_heads, page_size, head_dim, 0, 0, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
                True, True, BLOCK_N,
            )
    elif CAUSAL and WINDOW_LEFT < 0 and WINDOW_RIGHT < 0:
        n_full = tl.maximum(((q_abs_min + 1) // BLOCK_N) * BLOCK_N, 0)
        acc, m, l = _paged_attn_prefill_inner(
            q_tile, acc, m, l, k_cache, v_cache, block_table_b, k_scales, v_scales, kv_head,
            offs_n_base, s_lo, tl.minimum(n_full, s_hi), q_abs, valid_row, qk_scale_log2e, total_k_len,
            n_kv_heads, page_size, head_dim, QCK, QCV, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
            False, False, BLOCK_N,
        )
        acc, m, l = _paged_attn_prefill_inner(
            q_tile, acc, m, l, k_cache, v_cache, block_table_b, k_scales, v_scales, kv_head,
            offs_n_base, tl.maximum(n_full, s_lo), s_hi, q_abs, valid_row, qk_scale_log2e, total_k_len,
            n_kv_heads, page_size, head_dim, QCK, QCV, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
            True, False, BLOCK_N,
        )
    else:
        acc, m, l = _paged_attn_prefill_inner(
            q_tile, acc, m, l, k_cache, v_cache, block_table_b, k_scales, v_scales, kv_head,
            offs_n_base, s_lo, s_hi, q_abs, valid_row, qk_scale_log2e, total_k_len,
            n_kv_heads, page_size, head_dim, QCK, QCV, CAUSAL, WINDOW_LEFT, WINDOW_RIGHT, SOFTCAP,
            True, False, BLOCK_N,
        )

    if num_splits > 1:
        pid_lin = (pid_m * tl.num_programs(1) + bh) * num_splits + split
        po_base = pid_lin * BLOCK_M * head_dim
        tl.store(partial_o + po_base + tl.arange(0, BLOCK_M)[:, None] * head_dim + offs_d[None, :], acc)
        ml_base = pid_lin * BLOCK_M * 2
        tl.store(partial_ml + ml_base + tl.arange(0, BLOCK_M) * 2, m)
        tl.store(partial_ml + ml_base + tl.arange(0, BLOCK_M) * 2 + 1, l)
    else:
        if HAS_SINKS:
            # m/l live in the log2 domain here, so the sink logit is scaled by log2(e)
            sink = tl.load(sinks + q_head).to(tl.float32) * 1.4426950408889634
            m_top = tl.maximum(m, sink)
            alpha_s = tl.where(m == -float("inf"), 0.0, tl.exp2(m - m_top))
            acc = acc * alpha_s[:, None]
            l = l * alpha_s + tl.exp2(sink - m_top)
        out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
        if QCV > 0:
            out_tile = _rot_h32(out_tile, h32, BLOCK_M, head_dim)
        out_ptrs = out + (((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :])
        tl.store(out_ptrs, out_tile, mask = valid_row[:, None])


@triton.jit
def _paged_attn_prefill_combine_kernel(
    partial_o,
    partial_ml,
    out,
    h32,
    sinks,
    num_splits: tl.constexpr,
    QCV: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    q_len: tl.constexpr,
    n_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // n_q_heads
    q_head = bh - batch * n_q_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    rows = tl.arange(0, BLOCK_M)
    pid_lin = (pid_m * tl.num_programs(1) + bh) * num_splits

    m_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for sp in range(num_splits):
        m_s = tl.load(partial_ml + (pid_lin + sp) * BLOCK_M * 2 + rows * 2)
        m_max = tl.maximum(m_max, m_s)
    if HAS_SINKS:
        # partials are in the log2 domain; scale the sink logit by log2(e)
        sink = tl.load(sinks + q_head).to(tl.float32) * 1.4426950408889634
        m_max = tl.maximum(m_max, sink)
    m_safe = tl.where(m_max == -float("inf"), 0.0, m_max)

    l_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, head_dim), tl.float32)
    for sp in range(num_splits):
        ml_base = (pid_lin + sp) * BLOCK_M * 2
        m_s = tl.load(partial_ml + ml_base + rows * 2)
        l_s = tl.load(partial_ml + ml_base + rows * 2 + 1)
        w = tl.where(m_s == -float("inf"), 0.0, tl.exp2(m_s - m_safe))
        o_s = tl.load(partial_o + (pid_lin + sp) * BLOCK_M * head_dim + rows[:, None] * head_dim + offs_d[None, :])
        acc += o_s * w[:, None]
        l_sum += l_s * w

    if HAS_SINKS:
        l_sum += tl.exp2(sink - m_safe)
    out_tile = acc / tl.where(l_sum[:, None] == 0.0, 1.0, l_sum[:, None])
    if QCV > 0:
        out_tile = _rot_h32(out_tile, h32, BLOCK_M, head_dim)
    out_ptrs = out + (((batch * q_len + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :])
    tl.store(out_ptrs, out_tile, mask = (offs_m[:, None] < q_len))


def paged_attn_triton_prefill(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: int | tuple[int, int] | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    num_splits: int | None = None,
    max_kv_len: int | None = None,
    qc: tuple | None = None,
    pre_appended_len: int = 0,
    n_kv_heads_override: int | None = None,
    k_new: torch.Tensor | None = None,
    v_new: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prefill (large q_len) attention over the paged cache.

    Two ways to supply new K/V: `k`/`v` are appended to the cache before attention (the cache
    must have room); `k_new`/`v_new` are attended directly from the contiguous tensors without
    touching the cache -- kv positions at or above cache_seqlens[b] read from them. The direct
    form also works without a cache at all (k_cache = None), covering plain non-cached
    attention."""
    if not has_triton:
        raise RuntimeError("paged_attn_triton_prefill requires Triton, but Triton is not available")

    _check_tensor("q", q)

    bsz, q_len, n_q_heads, head_dim = q.shape
    new_kv_mode = 0
    if k_new is not None or v_new is not None:
        if k_new is None or v_new is None:
            raise ValueError("k_new and v_new must be provided together")
        if k is not None or qc is not None or pre_appended_len:
            raise ValueError("k_new/v_new cannot be combined with k/v, qc or pre_appended_len")
        _check_tensor("k_new", k_new)
        _check_tensor("v_new", v_new)
        new_kv_mode = 1 if k_cache is not None else 2

    if k_cache is None:
        if new_kv_mode != 2:
            raise ValueError("k_cache required unless attending directly over k_new/v_new")
        n_kv_heads = k_new.shape[2]
        page_size = 256
        block_table = torch.zeros((bsz, 1), dtype = torch.int32, device = q.device)
        cache_seqlens = torch.zeros((bsz,), dtype = torch.int32, device = q.device)
        k_cache = q  # never dereferenced: the page ranges are empty
        v_cache = q
    elif qc is None:
        _check_tensor("k_cache", k_cache)
        _check_tensor("v_cache", v_cache)
        _, page_size, n_kv_heads, cache_dim = k_cache.shape
        if v_cache.shape != k_cache.shape:
            raise ValueError("v_cache must have the same shape as k_cache")
        if cache_dim != head_dim:
            raise ValueError("q and cache head dimensions must match")
    else:
        page_size = k_cache.shape[1]
        n_kv_heads = n_kv_heads_override
    _check_tensor("block_table", block_table, None)
    _check_tensor("cache_seqlens", cache_seqlens, None)
    if n_q_heads % n_kv_heads != 0:
        raise ValueError("n_q_heads must be divisible by n_kv_heads")
    if head_dim > 512 or not _is_power_of_2(head_dim):
        raise ValueError("paged_attn_triton_prefill currently supports power-of-two head_dim <= 512")

    kv_append_len = pre_appended_len
    if k is not None or v is not None:
        if k is None or v is None:
            raise ValueError("k and v must be provided together")
        _check_tensor("k", k)
        _check_tensor("v", v)
        if k.shape[:1] != (bsz,) or k.shape[2:] != (n_kv_heads, head_dim):
            raise ValueError("k/v shape must be [batch, seqlen_new, kv_heads, head_dim]")
        kv_append_len = k.shape[1]
    if new_kv_mode:
        if k_new.shape[:1] != (bsz,) or k_new.shape[2:] != (n_kv_heads, head_dim):
            raise ValueError("k_new/v_new shape must be [batch, seqlen_new, kv_heads, head_dim]")
        if v_new.shape != k_new.shape:
            raise ValueError("v_new must have the same shape as k_new")
        kv_append_len = k_new.shape[1]

    if out is None:
        out = torch.empty_like(q)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    window_left, window_right = _normalize_window(window_size)
    sinks, has_sinks = _prep_sinks(sinks, n_q_heads, q)

    if qc is not None:
        k_scales, v_scales, qck, qcv = qc
        h32 = _get_h32(q.device)
    else:
        k_scales, v_scales, qck, qcv = q, q, 0, 0
        h32 = q

    # Tile configs by head_dim, sized for ~100 KB of smem with two pipeline stages. Blackwell
    # prefers narrower kv tiles (measured: 167 vs 153 TFLOPS on RTX 5090 at BN 32 vs 64)
    blackwell = torch.cuda.get_device_capability(q.device)[0] >= 10
    if head_dim <= 128:
        cfg = (128, 32, 8, 2) if blackwell else (128, 64, 8, 2)
    elif head_dim <= 256:
        cfg = (64, 32, 8, 2)
    else:
        cfg = (32, 16, 4, 2)
    block_m = block_m or cfg[0]
    block_n = block_n or cfg[1]
    num_warps = num_warps or cfg[2]
    num_stages = num_stages or cfg[3]
    if qc is not None:
        # compact plane tiles stage fewer smem bytes than fp16: wider kv tiles pay off
        block_n = max(16, min(128, 16384 // head_dim))

    num_pages_per_seq = block_table.shape[1]
    q_blocks = triton.cdiv(q_len, block_m)
    programs = q_blocks * bsz * n_q_heads

    # Split the kv range when the grid would quantize badly against the SM count (uniform
    # full-length programs at 1 block/SM leave a mostly idle trailing wave). Pick the split
    # count minimizing ceil-waves per unit of work, with a small penalty per extra split
    if num_splits is None:
        bound_kv = num_pages_per_seq * page_size + kv_append_len
        if max_kv_len is not None:
            bound_kv = min(bound_kv, max_kv_len + kv_append_len)
        dev = q.device.index
        if dev not in _decode_sm_count:
            _decode_sm_count[dev] = torch.cuda.get_device_properties(q.device).multi_processor_count
        sms = _decode_sm_count[dev]
        num_splits = 1
        if bound_kv >= 8192:
            best = None
            for cand in (1, 2, 3, 4, 5, 6, 8):
                cost = math.ceil(programs * cand / sms) / cand + 0.03 * (cand - 1)
                if best is None or cost < best[0]:
                    best = (cost, cand)
            num_splits = best[1]

    if num_splits > 1:
        partial_o = torch.empty(programs * num_splits * block_m * head_dim, dtype = torch.float32, device = q.device)
        partial_ml = torch.empty(programs * num_splits * block_m * 2, dtype = torch.float32, device = q.device)
    else:
        partial_o = q   # unused
        partial_ml = q  # unused

    with torch.cuda.device(q.device):
        if k is not None and kv_append_len:
            update_block_d = triton.next_power_of_2(head_dim)
            _paged_kv_update_kernel[(bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))](
                k, v, k_cache, v_cache, block_table, cache_seqlens,
                num_pages_per_seq, kv_append_len, n_kv_heads, page_size, head_dim, update_block_d,
                num_warps=2, num_stages=3,
            )

        grid = (q_blocks, bsz * n_q_heads, num_splits)
        _paged_attn_prefill_kernel[grid](
            q, k_cache, v_cache, block_table, cache_seqlens, out,
            partial_o, partial_ml, k_scales, v_scales, h32,
            k_new if new_kv_mode else q, v_new if new_kv_mode else q, sinks,
            num_splits, new_kv_mode, qck, qcv,
            q_len, kv_append_len, n_q_heads, n_kv_heads,
            num_pages_per_seq, page_size, head_dim, float(softmax_scale),
            bool(causal), int(window_left), int(window_right), float(softcap or 0.0),
            has_sinks, block_m, block_n,
            num_warps=num_warps, num_stages=num_stages,
        )
        if num_splits > 1:
            _paged_attn_prefill_combine_kernel[(q_blocks, bsz * n_q_heads)](
                partial_o, partial_ml, out, h32, sinks,
                num_splits, qcv, has_sinks, q_len, n_q_heads, head_dim, block_m,
                num_warps=8, num_stages=1,
            )
    return out


def fn_triton_attn_nocache(args: AttnArgs) -> torch.Tensor | None:
    """Non-cached attention through the prefill kernel's direct-kv source (NEW_KV=2)."""
    if (
        not has_triton or
        args.is_varlen() or
        args.has_kv_cache() or
        args.dim > 512 or
        not _is_power_of_2(args.dim) or
        args.q.dtype != torch.float16 or
        args.non_causal_spans or
        not args.q.is_contiguous() or
        not args.k.is_contiguous() or
        not args.v.is_contiguous()
    ):
        return None

    return paged_attn_triton_prefill(
        args.q, None, None, None, None, None, None,
        causal = args.causal,
        softmax_scale = args.sm_scale,
        window_size = args.get_window_size(),
        softcap = args.softcap,
        sinks = args.sinks,
        k_new = args.k,
        v_new = args.v,
    )


def fn_triton_paged_attn_prefill(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_triton or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.q_len <= 16 or
        args.dim > 512 or
        not _is_power_of_2(args.dim) or
        args.q.dtype != torch.float16 or
        args.k_cache.dtype != torch.float16
    ):
        return None

    if args.non_causal_spans:
        arglist = get_non_causal_span_arglist(args)
        return torch.cat([paged_attn_triton_prefill(**a) for a in arglist], dim=1)

    return paged_attn_triton_prefill(
        q=args.q,
        k=args.k,
        v=args.v,
        k_cache=args.k_cache,
        v_cache=args.v_cache,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
        window_size=args.get_window_size(),
        softcap=args.softcap,
        sinks=args.sinks,
    )


@triton.jit
def _varlen_attn_kernel(
    q,
    k,
    v,
    cu_seqlens,
    out,
    sinks,
    n_q_heads: tl.constexpr,
    n_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    HEAD_DIM_P2: tl.constexpr,
    scale: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Packed varlen self-attention (no cache): each segment of the packed sequence attends
    within itself. Grid: (q blocks of the longest segment, segments, q heads); programs beyond
    a segment's length exit early. head_dim need not be a power of two (padded loads)."""
    pid_m = tl.program_id(0)
    seg = tl.program_id(1)
    q_head = tl.program_id(2)
    group_size = n_q_heads // n_kv_heads
    kv_head = q_head // group_size

    q0 = tl.load(cu_seqlens + seg)
    q1 = tl.load(cu_seqlens + seg + 1)
    seg_len = q1 - q0
    if pid_m * BLOCK_M >= seg_len:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_P2)
    valid_row = offs_m < seg_len
    d_mask = offs_d < head_dim

    q_ptrs = q + ((q0 + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
    q_tile = tl.load(q_ptrs, mask = valid_row[:, None] & d_mask[None, :], other = 0.0)
    qk_scale_log2e = scale * 1.4426950408889634

    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM_P2), tl.float32)

    if CAUSAL:
        n_hi = tl.minimum(pid_m * BLOCK_M + BLOCK_M, seg_len)
    else:
        n_hi = seg_len
    n_lo = 0
    if WINDOW_LEFT >= 0:
        n_lo = tl.maximum(0, pid_m * BLOCK_M - WINDOW_LEFT)
        n_lo = (n_lo // BLOCK_N) * BLOCK_N

    # Interior tiles need no masking (invalid q rows produce harmless finite garbage that the
    # store mask drops); only the causal boundary / segment tail tiles run the masked path
    if WINDOW_LEFT >= 0 or WINDOW_RIGHT >= 0:
        n_full = n_lo
    elif CAUSAL:
        n_full = (tl.minimum(pid_m * BLOCK_M, n_hi) // BLOCK_N) * BLOCK_N
    else:
        n_full = (n_hi // BLOCK_N) * BLOCK_N

    for n0 in range(n_lo, n_full, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_ptrs = k + ((q0 + offs_n[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None]
        k_tile = tl.load(k_ptrs, mask = d_mask[:, None], other = 0.0)
        scores = tl.dot(q_tile, k_tile)
        if SOFTCAP > 0.0:
            s_nat = scores * scale
            s_nat = (2.0 / (1.0 + tl.exp(-2.0 * (s_nat / SOFTCAP))) - 1.0) * SOFTCAP
            scores = s_nat * 1.4426950408889634
        else:
            scores = scores * qk_scale_log2e

        m_new = tl.maximum(m, tl.max(scores, axis = 1))
        p = tl.exp2(scores - m_new[:, None])
        alpha = tl.exp2(m - m_new)
        l = l * alpha + tl.sum(p, axis = 1)

        v_ptrs = v + ((q0 + offs_n[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask = d_mask[None, :], other = 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new

    for n0 in range(tl.maximum(n_full, n_lo), n_hi, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        k_ptrs = k + ((q0 + offs_n[None, :]) * n_kv_heads + kv_head) * head_dim + offs_d[:, None]
        k_tile = tl.load(k_ptrs, mask = (offs_n[None, :] < n_hi) & d_mask[:, None], other = 0.0)
        scores = tl.dot(q_tile, k_tile)
        if SOFTCAP > 0.0:
            s_nat = scores * scale
            s_nat = (2.0 / (1.0 + tl.exp(-2.0 * (s_nat / SOFTCAP))) - 1.0) * SOFTCAP
            scores = s_nat * 1.4426950408889634
        else:
            scores = scores * qk_scale_log2e

        valid = valid_row[:, None] & (offs_n[None, :] < n_hi)
        if CAUSAL:
            valid = valid & (offs_n[None, :] <= offs_m[:, None])
        if WINDOW_LEFT >= 0:
            valid = valid & (offs_n[None, :] >= offs_m[:, None] - WINDOW_LEFT)
        if WINDOW_RIGHT >= 0:
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + WINDOW_RIGHT)
        scores = tl.where(valid, scores, -float("inf"))

        m_new = tl.maximum(m, tl.max(scores, axis = 1))
        m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp2(scores - m_exp[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m == -float("inf"), 0.0, tl.exp2(m - m_exp))
        l = l * alpha + tl.sum(p, axis = 1)

        v_ptrs = v + ((q0 + offs_n[:, None]) * n_kv_heads + kv_head) * head_dim + offs_d[None, :]
        v_tile = tl.load(v_ptrs, mask = (offs_n[:, None] < n_hi) & d_mask[None, :], other = 0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)
        m = m_new

    if HAS_SINKS:
        # m/l live in the log2 domain here, so the sink logit is scaled by log2(e)
        sink = tl.load(sinks + q_head).to(tl.float32) * 1.4426950408889634
        m_top = tl.maximum(m, sink)
        alpha_s = tl.where(m == -float("inf"), 0.0, tl.exp2(m - m_top))
        acc = acc * alpha_s[:, None]
        l = l * alpha_s + tl.exp2(sink - m_top)
    out_tile = acc / tl.where(l[:, None] == 0.0, 1.0, l[:, None])
    out_ptrs = out + ((q0 + offs_m[:, None]) * n_q_heads + q_head) * head_dim + offs_d[None, :]
    tl.store(out_ptrs, out_tile, mask = valid_row[:, None] & d_mask[None, :])


def varlen_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: int | tuple[int, int] | None = None,
    softcap: float = 0.0,
    sinks: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Packed varlen self-attention over (total, heads, head_dim) tensors, segments given by
    cu_seqlens (as flash_attn_varlen_func with cu_seqlens_q == cu_seqlens_k)."""
    if not has_triton:
        raise RuntimeError("varlen_attn_triton requires Triton, but Triton is not available")

    squeeze = q.ndim == 4
    if squeeze:
        q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    total_q, n_q_heads, head_dim = q.shape
    n_kv_heads = k.shape[1]

    if out is None:
        out = torch.empty_like(q)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    window_left, window_right = _normalize_window(window_size)
    sinks, has_sinks = _prep_sinks(sinks, n_q_heads, q)

    num_segs = cu_seqlens.shape[0] - 1
    hd_p2 = triton.next_power_of_2(head_dim)
    if hd_p2 <= 128:
        block_m, block_n, num_warps, num_stages = 128, 64, 8, 2
    elif hd_p2 <= 256:
        block_m, block_n, num_warps, num_stages = 64, 32, 8, 2
    else:
        block_m, block_n, num_warps, num_stages = 32, 16, 4, 2
    # Don't tile wider than the longest segment (vision windows are often 64 tokens)
    while block_m >= 32 and block_m >= 2 * max_seqlen:
        block_m //= 2
        if num_warps > 4:
            num_warps //= 2

    with torch.cuda.device(q.device):
        grid = (triton.cdiv(max_seqlen, block_m), num_segs, n_q_heads)
        _varlen_attn_kernel[grid](
            q, k, v, cu_seqlens, out, sinks,
            n_q_heads, n_kv_heads, head_dim, hd_p2, float(softmax_scale),
            bool(causal), int(window_left), int(window_right), float(softcap or 0.0),
            has_sinks, block_m, block_n,
            num_warps=num_warps, num_stages=num_stages,
        )
    return out.unsqueeze(0) if squeeze else out


def fn_triton_varlen_attn(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_triton or
        not args.is_varlen() or
        args.bsz > 1 or
        args.has_kv_cache() or
        args.non_causal_spans or
        args.dim > 512 or
        args.q.dtype != torch.float16
    ):
        return None

    return varlen_attn_triton(
        q=args.q,
        k=args.k,
        v=args.v,
        cu_seqlens=args.cu_seqlens,
        max_seqlen=args.max_seqlen,
        causal=args.causal,
        softmax_scale=args.sm_scale,
        window_size=args.get_window_size(),
        softcap=args.softcap,
        sinks=args.sinks,
    )


def fn_triton_paged_attn_decode_qc(args: AttnArgs) -> torch.Tensor | None:
    if (
        args.q_cache is None or
        not has_triton or
        args.q_len > 16 or
        args.dim > 512 or
        not _is_power_of_2(args.dim) or
        args.q.dtype != torch.float16 or
        args.non_causal_spans
    ):
        return None
    qk, sk, qv, sv, k_bits, v_bits = args.q_cache
    return paged_attn_triton_decode(
        q=args.q, k=None, v=None,
        k_cache=qk, v_cache=qv,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
        window_size=args.get_window_size(),
        softcap=args.softcap,
        sinks=args.sinks,
        qc=(sk, sv, k_bits, v_bits),
        pre_appended_len=args.q_len,
        n_kv_heads_override=args.num_kv_heads,
    )


def fn_triton_paged_attn_prefill_qc(args: AttnArgs) -> torch.Tensor | None:
    if (
        args.q_cache is None or
        not has_triton or
        args.q_len <= 16 or
        args.dim > 512 or
        not _is_power_of_2(args.dim) or
        args.q.dtype != torch.float16 or
        args.non_causal_spans
    ):
        return None
    qk, sk, qv, sv, k_bits, v_bits = args.q_cache
    return paged_attn_triton_prefill(
        q=args.q, k=None, v=None,
        k_cache=qk, v_cache=qv,
        block_table=args.block_table,
        cache_seqlens=args.cache_seqlens,
        causal=args.causal,
        softmax_scale=args.sm_scale,
        window_size=args.get_window_size(),
        softcap=args.softcap,
        sinks=args.sinks,
        qc=(sk, sv, k_bits, v_bits),
        pre_appended_len=args.q_len,
        n_kv_heads_override=args.num_kv_heads,
    )
