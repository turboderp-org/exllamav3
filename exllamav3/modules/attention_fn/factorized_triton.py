"""Decode preparation for factorized differential attention."""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention_decode(Q, AK, BK, AV, BV, ShiftK, ShiftV, KNorm, QNorm, SoftmaxScale,
                      Slots, Positions, PrevK, PrevV, OutQ, OutK, OutV,
                      QHeads: tl.constexpr, KVHeads: tl.constexpr,
                      Dim: tl.constexpr, Rank: tl.constexpr, Window: tl.constexpr, Eps: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, Dim)
    pos = tl.load(Positions + b)
    q_ptr = (b * QHeads + h) * Dim + d
    q = tl.load(Q + q_ptr)
    q = (q * tl.rsqrt(tl.sum(q * q, 0) / Dim + Eps)) * (1. + tl.load(QNorm + d).to(tl.float32))
    q = (q * tl.log(tl.minimum(pos + 1, Window).to(tl.float32))) * tl.load(SoftmaxScale + h)
    tl.store(OutQ + q_ptr, q)

    if h < KVHeads:
        slot = tl.load(Slots + b).to(tl.int32)
        a = (b * KVHeads + h) * Rank
        cache = (slot * KVHeads + h) * Dim + d
        k = tl.full((Dim,), 0, tl.float32)
        v = tl.full((Dim,), 0, tl.float32)
        for r in tl.static_range(Rank):
            bd = (b * Rank + r) * Dim + d
            k = tl.fma(tl.load(AK + a + r), tl.load(BK + bd), k)
            v = tl.fma(tl.load(AV + a + r), tl.load(BV + bd), v)
        k = k / Rank
        v = v / Rank
        old_k = tl.load(PrevK + cache)
        old_v = tl.load(PrevV + cache)
        tl.store(PrevK + cache, k)
        tl.store(PrevV + cache, v)
        idx = b * KVHeads + h
        wk = tl.where(pos != 0, tl.sigmoid(tl.load(ShiftK + idx)), 0.)
        wv = tl.where(pos != 0, tl.sigmoid(tl.load(ShiftV + idx)), 0.)
        k = wk * old_k + (1. - wk) * k
        v = wv * old_v + (1. - wv) * v
        k = (k * tl.rsqrt(tl.sum(k * k, 0) / Dim + Eps)) * (1. + tl.load(KNorm + d).to(tl.float32))
        tl.store(OutK + idx * Dim + d, k)
        tl.store(OutV + idx * Dim + d, v)


def factorized_attention_decode(q, ak, bk, av, bv, shift_k, shift_v, weights, slots, positions, state,
                     q_heads, kv_heads, head_dim, rank, window, eps):
    """Prepare FP16 Q/K/V for paged attention and update the FP32 raw K/V state."""
    batch = q.shape[0]
    q_out = q.new_empty((batch, 1, q_heads, head_dim), dtype=torch.float16)
    k_out = q.new_empty((batch, 1, kv_heads, head_dim), dtype=torch.float16)
    v_out = torch.empty_like(k_out)
    _attention_decode[(batch, q_heads)](
        q, ak, bk, av, bv, shift_k, shift_v, weights["k_norm.norm.weight"], weights["q_norm.norm.weight"],
        weights["softmax_scaler"], slots, positions, *state, q_out, k_out, v_out,
        q_heads, kv_heads, head_dim, rank, window, eps, enable_fp_fusion=False,
    )
    return q_out, k_out, v_out


@triton.jit
def _subtract_noise(Attention, Weight, Output, GROUP: tl.constexpr, DIM: tl.constexpr, TILE: tl.constexpr):
    group = tl.program_id(0)
    i = tl.arange(0, TILE)
    signal_width = (GROUP - 1) * DIM
    src = Attention + group * GROUP * DIM
    signal = tl.load(src + i, i < signal_width, 0).to(tl.float32)
    noise = tl.load(src + signal_width + i % DIM).to(tl.float32)
    tl.store(Output + group * signal_width + i, signal - noise * tl.load(Weight + group), i < signal_width)


def subtract_noise(attn, weight, out_dtype):
    kv_heads = weight.shape[-1]
    group = attn.shape[-2] // kv_heads
    dim = attn.shape[-1]
    width = (group - 1) * dim
    if attn.is_cuda and attn.is_contiguous() and weight.is_contiguous() and width <= 1024:
        out = attn.new_empty((*attn.shape[:-2], kv_heads * width), dtype = out_dtype)
        _subtract_noise[(weight.numel(),)](attn, weight, out, group, dim, triton.next_power_of_2(width),
                                           enable_fp_fusion = False)
        return out
    groups = attn.float().unflatten(-2, (kv_heads, group))
    return (groups[..., :-1, :] - weight[..., None, None] * groups[..., -1:, :]).flatten(-3).to(out_dtype)
