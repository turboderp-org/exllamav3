# CPU MoE offload kernels, model-free: random mul1 trellis experts through
# exl3_moe_cpu_forward. Checks that the result does not depend on how the work is partitioned
# (worker count), that the band-swizzled arena layout computes the same values as the native
# layout, and that moe_unswizzle_trellis restores native tile order from a swizzled expert block
# exactly (the layout _SharedArena.rehome produces and the GPU dequant path consumes).

import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.manual_seed(0)

H, I, K, E, TOPK = 512, 256, 4, 12, 4


def trellis(k, n):
    return torch.randint(0, 65536, (k // 16, n // 16, 16 * K), dtype = torch.int32).to(torch.int16)


def band_swizzle(t):
    tk, tn, ps = t.shape
    return t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps)


def make_experts():
    experts = []
    for _ in range(E):
        mats = []
        for k, n in ((H, I), (H, I), (I, H)):
            mats.append((trellis(k, n), torch.randn(k).half(), torch.randn(n).half()))
        experts.append(mats)
    return experts


def make_layer(experts, swizzled):
    def col(j, i, xf):
        return [xf(e[j][i]) for e in experts]
    tr = (lambda t: band_swizzle(t)) if swizzled else (lambda t: t.contiguous())
    ident = lambda t: t
    return ext.exl3_moe_cpu_make_layer(
        col(0, 0, tr), col(0, 1, ident), col(0, 2, ident),
        col(1, 0, tr), col(1, 1, ident), col(1, 2, ident),
        col(2, 0, tr), col(2, 1, ident), col(2, 2, ident),
        [], [], [], 0, 0.0, 1 if swizzled else 0)


def forward(handle, x, sel, w, threads):
    out = torch.empty(x.shape[0], H, dtype = torch.float)
    ext.exl3_moe_cpu_forward(handle, x, sel, w, out, threads)
    return out


@pytest.fixture(scope = "module")
def experts():
    return make_experts()


@pytest.fixture(scope = "module")
def inputs():
    cases = []
    for rows in (1, 3, 64):
        x = (torch.randn(rows, H) * 0.5).half()
        sel = torch.stack([torch.randperm(E)[:TOPK] for _ in range(rows)]).to(torch.int32)
        w = torch.rand(rows, TOPK).half()
        cases.append((x, sel, w))
    return cases


def test_partition_invariance(experts, inputs):
    handle = make_layer(experts, False)
    for x, sel, w in inputs:
        ref = forward(handle, x, sel, w, 1)
        for threads in (2, 3, 7, 12):
            assert torch.equal(forward(handle, x, sel, w, threads), ref)
    ext.exl3_moe_cpu_free_layer(handle)


@pytest.mark.skipif(not ext.exl3_moe_cpu_has_avx512_vbmi(), reason = "swizzled layout needs the VBMI tier")
def test_swizzled_matches_native(experts, inputs):
    native = make_layer(experts, False)
    swz = make_layer(experts, True)
    for x, sel, w in inputs:
        a = forward(native, x, sel, w, 12)
        b = forward(swz, x, sel, w, 12)
        assert torch.allclose(a, b, rtol = 1e-4, atol = 1e-3), (a - b).abs().max()
    ext.exl3_moe_cpu_free_layer(native)
    ext.exl3_moe_cpu_free_layer(swz)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_unswizzle_restores_native(experts):
    n_exp = 5
    native_blocks, swz_blocks = [], []
    for e in experts[:n_exp]:
        for (t, _, _) in e:
            native_blocks.append(t.contiguous().view(torch.uint8).reshape(-1))
            swz_blocks.append(band_swizzle(t).view(torch.uint8).reshape(-1))
    native = torch.cat(native_blocks)
    swz = torch.cat(swz_blocks)
    exp_b = native.numel() // n_exp
    dst = torch.zeros(native.numel() + 64, dtype = torch.uint8, device = "cuda")

    def unswizzle(src, swizzled):
        # As the streamed-prefill path issues it: one launch per projection over the batch
        off = 0
        for k, n in ((H, I), (H, I), (I, H)):
            ext.moe_unswizzle_trellis(src, dst, n_exp, exp_b, off, k // 16, n // 16, K, swizzled)
            off += (k // 16) * (n // 16) * 16 * K * 2
        torch.cuda.synchronize()

    unswizzle(swz.to("cuda"), True)
    assert torch.equal(dst[:native.numel()].cpu(), native)
    assert int(dst[native.numel():].sum()) == 0
    # swizzled = False (K8 matrices, or the whole arena when swizzling is off) is a plain copy
    dst.zero_()
    unswizzle(native.to("cuda"), False)
    assert torch.equal(dst[:native.numel()].cpu(), native)
