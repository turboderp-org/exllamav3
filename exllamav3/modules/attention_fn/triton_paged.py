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
    kv_append_len: tl.constexpr,
    n_kv_heads: tl.constexpr,
    num_pages_per_seq: tl.constexpr,
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
        if kv_append_len:
            update_block_d = triton.next_power_of_2(head_dim)
            update_grid = (bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))
            _paged_kv_update_kernel[update_grid](
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                cache_seqlens,
                kv_append_len,
                n_kv_heads,
                num_pages_per_seq,
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
        if kv_append_len:
            update_block_d = triton.next_power_of_2(head_dim)
            _paged_kv_update_kernel[(bsz * kv_append_len, n_kv_heads, triton.cdiv(head_dim, update_block_d))](
                k,
                v,
                k_cache,
                v_cache,
                block_table,
                cache_seqlens,
                kv_append_len,
                n_kv_heads,
                num_pages_per_seq,
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
    )
