"""Fused SenseNova Loop mHC routing, one program per token row."""
import torch
import triton
import triton.language as tl


@triton.jit
def _route(X, W, B, Branch, Post, Mix, H: tl.constexpr, S: tl.constexpr,
           ITER: tl.constexpr, FIXED: tl.constexpr, SCALE: tl.constexpr,
           N: tl.constexpr, BN: tl.constexpr, O: tl.constexpr, BO: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BN)
    x = tl.load(X + row * N + i, i < N, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / N + 1.e-6)
    o = tl.arange(0, BO)
    k = tl.arange(0, 512)
    acc = tl.full((BO, 512), 0, tl.float32)
    for start in range(tl.cdiv(N, 512)):
        idx = start * 512 + k
        v = tl.load(X + row * N + idx, idx < N, 0).to(tl.float32) * inv
        w = tl.load(W + o[:, None] * N + idx[None, :],
                    (o[:, None] < O) & (idx[None, :] < N), 0)
        acc += w * v[None, :]
    logits = tl.sum(acc, 1) + tl.load(B + o, o < O, 0)
    s = tl.arange(0, S)
    pre = 1. / (1. + tl.exp(-tl.gather(logits, s, 0)))
    post = 2. / (1. + tl.exp(-tl.gather(logits, s + S, 0)))
    tl.store(Post + row * S + s, post)
    if FIXED:
        m = tl.where(s[:, None] == s[None, :], 1., 0.)
    else:
        mi = 2 * S + s[:, None] * S + s[None, :]
        m = tl.reshape(tl.gather(logits, tl.reshape(mi, (S * S,)), 0), (S, S))
        for _ in range(ITER):
            maximum = tl.max(m, 1)
            m = m - (maximum + tl.log(tl.sum(tl.exp(m - maximum[:, None]), 1)))[:, None]
            maximum = tl.max(m, 0)
            m = m - (maximum + tl.log(tl.sum(tl.exp(m - maximum[None, :]), 0)))[None, :]
        m = tl.exp(m)
    tl.store(Mix + row * S * S + s[:, None] * S + s[None, :], m)
    h = tl.arange(0, triton.next_power_of_2(H))
    result = tl.full((triton.next_power_of_2(H),), 0, tl.float32)
    for j in tl.static_range(S):
        gate = tl.sum(tl.where(s == j, pre, 0), 0).to(X.dtype.element_ty, fp_downcast_rounding = "rtne").to(tl.float32)
        value = tl.load(X + row * N + j * H + h, h < H, 0).to(tl.float32)
        result += gate * value
    result = result.to(X.dtype.element_ty, fp_downcast_rounding = "rtne").to(tl.float32) * SCALE
    tl.store(Branch + row * H + h, result.to(X.dtype.element_ty, fp_downcast_rounding = "rtne"), h < H)


def route_triton(residual, weight, bias, streams, iterations, fixed_identity, pre_scale):
    hidden = residual.shape[-1]
    rows = residual.numel() // (streams * hidden)
    shape = residual.shape[:-2]
    branch = torch.empty((*shape, hidden), dtype = residual.dtype, device = residual.device)
    post = torch.empty((*shape, streams), dtype = torch.float32, device = residual.device)
    mix = torch.empty((*shape, streams, streams), dtype = torch.float32, device = residual.device)
    if rows:
        n = streams * hidden
        outputs = 2 * streams + (0 if fixed_identity else streams * streams)
        _route[(rows,)](residual, weight, bias, branch, post, mix, hidden, streams,
                        iterations, fixed_identity, pre_scale, n, triton.next_power_of_2(n),
                        outputs, triton.next_power_of_2(outputs), enable_fp_fusion = False)
    return branch, post, mix
