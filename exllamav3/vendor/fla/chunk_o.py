# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import torch
import triton
import triton.language as tl
from .index import prepare_chunk_indices
from .op import exp2
from .utils import IS_NVIDIA_HOPPER
from .utils import autotune_cache_kwargs
from .utils import check_shared_mem

BKV_LIST = [64, 128] if check_shared_mem() else ([32, 64] if check_shared_mem('ada') else [32])
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_G_GAMMA': lambda args: args['g_gamma'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
        triton.Config({'BK': 64, 'BV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3),
    ],
    key=['H', 'HV', 'K', 'V', 'BT', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    g_gamma,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_G_GAMMA: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h // (HV // H)) * K
    k += (bos * H + i_h // (HV // H)) * K
    v += (bos * HV + i_h) * V
    o += (bos * HV + i_h) * V
    h += (i_tg * HV + i_h).to(tl.int64) * K*V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_k = k + o_k[:, None] + o_t[None, :] * (H*K)
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = (o_v[:, None] < V) & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & (o_v[None, :] < V)
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        if STATE_V_FIRST:
            b_o += tl.dot(b_q, tl.trans(b_h))
        else:
            b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * HV + i_h
        p_g = g + o_t * HV
        b_g = tl.load(p_g, mask=m_t, other=0.0)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    if USE_G_GAMMA:
        b_gamma = tl.load(g_gamma + i_h)
        b_g = b_gamma * (tl.arange(0, BT) + 1)
        b_o = b_o * exp2(b_g)[:, None]
        b_A = b_A * exp2(b_g[:, None] - b_g[None, :])
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV*V) + o_v[None, :]

    b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    g_gamma: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HV)
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        g_gamma=g_gamma,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o

