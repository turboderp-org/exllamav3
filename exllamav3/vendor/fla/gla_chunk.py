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
from .utils import autotune_cache_kwargs
from .utils import check_shared_mem

BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'HV', 'STATE_V_FIRST'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
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
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qk = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV*K) + o_k[None, :]
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]

        # [BT, BK]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        # [BT, BK]
        b_g = tl.load(p_g, mask=m_qk, other=0.0).to(tl.float32)
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        if i_k >= 0:
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = v + o_t[:, None] * (HV*V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV*V) + o_v[None, :]
    p_A = A + o_t[:, None] * (HV*BT) + o_i[None, :]
    # [BT, BV]
    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    # [BT, BT]
    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Please ensure zeros, since vllm will use padding v
    o = torch.zeros_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * HV)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
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

