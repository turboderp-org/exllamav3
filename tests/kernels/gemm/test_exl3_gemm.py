"""
ext.exl3_gemm (EXL3-quantized B, fp16 A) against a dequantize-then-matmul reference (testlib.exl3.linear_ref,
which dequantizes through reconstruct_had_slice, so only the tensor-core matmul is under test), plus run-to-run
bit-identity and the dynamic shared memory budget of the shape selector.

The kernels are meant to be exact up to fp16 accumulation order. A wrong fragment mapping (e.g. the sm_75 port's
two m16n8k8 per m16n8k16, where half the k dimension dropped or double-counted is the realistic failure mode)
shows up as an O(1) relative error, not a small one.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import CODEBOOKS, codebook_flags, generator, linear_ref, rand_linear


def _gemm(A, w, mcg = False, mul1 = False):
    m, n = A.shape[0], w["trellis"].shape[1] * 16
    C = torch.empty((m, n), dtype = torch.float16, device = A.device)
    A_had = torch.empty_like(A)
    ext.exl3_gemm(A, w["trellis"], C, w["suh"], A_had, w["svh"], 0, mcg, mul1, 0)
    return C


@pytest.mark.parametrize("codebook", CODEBOOKS)
@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("m", [1, 8, 16, 32])
@torch.inference_mode()
def test_gemm_matches_reconstruct(device, codebook, K, m):
    """
    m sweeps the kernel's dispatch tiers deliberately: m=1 takes the GEMV path, m<=8 the small-m reduction, m>16
    the multi-pass loop. Each tier drives the mma differently, so a broken k split would not necessarily fail
    all of them.
    """
    k, n = 512, 512
    gen = generator(K * 100 + m)
    w = rand_linear(k, n, K, gen, device)
    A = (torch.randn((m, k), generator = gen) * 0.5).half().to(device)

    C = _gemm(A, w, *codebook_flags(codebook))
    ref = linear_ref(A, w["trellis"], w["suh"], w["svh"], K, codebook)

    # fp16 accumulation over k=512 with values ~O(1); compare on relative RMS rather than elementwise
    # tolerance, which fp16 rounding alone would breach on the tail
    err = (C.float() - ref.float()).square().mean().sqrt()
    scale = ref.float().square().mean().sqrt()
    rel = (err / scale).item()
    assert rel < 0.02, f"K={K} m={m} {codebook}: relative RMS error {rel:.4f}"


@pytest.mark.parametrize("K", [2, 4])
@torch.inference_mode()
def test_gemm_deterministic(device, K):
    """
    Repeat launches must be bit-identical. The sm_75 cp.async fallback relies on the pipeline's existing
    __syncthreads() for ordering; if that assumption were wrong, the result would be a race and would show up
    here as run-to-run variation rather than as a wrong-but-stable answer.
    """
    k, n, m = 512, 512, 16
    gen = generator(7)
    w = rand_linear(k, n, K, gen, device)
    A = (torch.randn((m, k), generator = gen) * 0.5).half().to(device)

    outs = []
    for _ in range(8):
        C = _gemm(A, w)
        torch.cuda.synchronize(device)
        outs.append(C.clone())

    for i, o in enumerate(outs[1:], 1):
        assert torch.equal(outs[0], o), f"K={K}: launch {i} differs from launch 0"


@torch.inference_mode()
def test_smem_budget_respected(device):
    """
    Every shape the kernel selector accepts must fit the device's dynamic shared memory limit. Turing caps
    dynamic smem at 64 KB while the kernels are written against 90 KB, so shape 4 at 8 bpw (66 KB) has to be
    filtered out rather than attempted; a shape that exceeds the limit fails the launch here.
    """
    limit = ext.g_get_smem_max(torch.cuda.current_device())

    k, n = 2048, 2048
    for K in range(1, 9):
        gen = generator(K)
        w = rand_linear(k, n, K, gen, device)
        A = (torch.randn((16, k), generator = gen) * 0.5).half().to(device)
        C = _gemm(A, w)
        torch.cuda.synchronize(device)
        assert torch.isfinite(C.float()).all(), f"K={K}: non-finite output (limit {limit})"
