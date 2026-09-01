"""
Numerical validation for the sm_75 (Turing) port.

The Turing path substitutes two mma.m16n8k8 for each mma.m16n8k16 and replaces cp.async with
synchronous shared-memory stores. Both are meant to be exact, not approximate, so these tests
compare the quantized GEMM against a dequantize-then-matmul reference computed by the same
extension. A correct k-split reproduces the reference to fp16 rounding; a wrong fragment
mapping (the realistic failure mode - half the k dimension dropped or double-counted) shows up
as a large relative error, not a small one.

Run on any device; on sm_80+ this is a regression test that the refactor changed nothing.

    python -m pytest tests/test_sm75_gemm.py -v
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda:0"


def _make_trellis(k, n, K, seed):
    """Random EXL3 trellis plus the scale/flip vectors the kernel expects."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    trellis = torch.randint(
        0, 65536, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32
    ).to(torch.int16).to(DEV)
    suh = (torch.randn(k, generator=g) * 0.1).to(torch.float16).to(DEV)
    svh = (torch.randn(n, generator=g) * 0.1).to(torch.float16).to(DEV)
    return trellis, suh, svh


def _reference(A, trellis, suh, svh, K, mcg, mul1):
    """Dequantize B to fp16 (same codebook path, no mma) and matmul, as ground truth.

    reconstruct_had_slice applies the same input/output Hadamard and scale/flip vectors the
    fused kernel folds in, so this isolates exactly the tensor-core matmul under test.
    """
    k = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    B = torch.empty((k, n), dtype=torch.float16, device=DEV)
    ext.reconstruct_had_slice(B, trellis, suh, svh, K, mcg, mul1, 0)
    return (A.float() @ B.float()).to(torch.float16)


@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("m", [1, 8, 16, 32])
def test_gemm_matches_reconstruct(K, m):
    """exl3_gemm must agree with reconstruct-then-matmul across the m regimes.

    m sweeps the kernel's dispatch tiers deliberately: m=1 takes the GEMV path, m<=8 the
    small-m reduction, m>16 the multi-pass loop. Each tier drives the mma differently, so a
    broken k=8 split would not necessarily fail all of them.
    """
    k, n = 512, 512
    trellis, suh, svh = _make_trellis(k, n, K, seed=K * 100 + m)
    A = (torch.randn((m, k), device=DEV) * 0.5).to(torch.float16)

    C = torch.empty((m, n), dtype=torch.float16, device=DEV)
    A_had = torch.empty_like(A)
    ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, False, 0)

    ref = _reference(A, trellis, suh, svh, K, False, False)

    # fp16 accumulation over k=512 with values ~O(1); compare on relative RMS rather than
    # elementwise tolerance, which fp16 rounding alone would breach on the tail.
    err = (C.float() - ref.float()).square().mean().sqrt()
    scale = ref.float().square().mean().sqrt()
    rel = (err / scale).item()
    assert rel < 0.02, f"K={K} m={m}: relative RMS error {rel:.4f}"


@pytest.mark.parametrize("K", [2, 4])
def test_gemm_deterministic(K):
    """Repeat launches must be bit-identical.

    The Turing cp.async fallback relies on the pipeline's existing __syncthreads() for
    ordering. If that assumption were wrong, the result would be a race and would show up
    here as run-to-run variation rather than as a wrong-but-stable answer.
    """
    k, n, m = 512, 512, 16
    trellis, suh, svh = _make_trellis(k, n, K, seed=7)
    A = (torch.randn((m, k), device=DEV) * 0.5).to(torch.float16)

    outs = []
    for _ in range(8):
        C = torch.empty((m, n), dtype=torch.float16, device=DEV)
        A_had = torch.empty_like(A)
        ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, False, 0)
        torch.cuda.synchronize()
        outs.append(C.clone())

    for i, o in enumerate(outs[1:], 1):
        assert torch.equal(outs[0], o), f"K={K}: launch {i} differs from launch 0"


def test_smem_budget_respected():
    """Selected shapes must fit the device's dynamic shared memory limit.

    Turing caps dynamic smem at 64 KB while the kernels are written against 90 KB, so shape 4
    at 8 bpw (66 KB) has to be filtered out rather than attempted. This asserts the filter is
    actually consulted: every shape the kernel selector accepts must be launchable.
    """
    props = torch.cuda.get_device_properties(0)
    limit = props.shared_memory_per_block_optin

    k, n = 2048, 2048
    for K in range(1, 9):
        trellis, suh, svh = _make_trellis(k, n, K, seed=K)
        A = (torch.randn((16, k), device=DEV) * 0.5).to(torch.float16)
        C = torch.empty((16, n), dtype=torch.float16, device=DEV)
        A_had = torch.empty_like(A)
        # A shape that exceeds the limit would fail the launch here rather than being skipped
        ext.exl3_gemm(A, trellis, C, suh, A_had, svh, 0, False, False, 0)
        torch.cuda.synchronize()
        assert torch.isfinite(C.float()).all(), f"K={K}: non-finite output (limit {limit})"
