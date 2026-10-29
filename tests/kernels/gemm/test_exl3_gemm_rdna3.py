"""
ext.exl3_gemm on gfx11 through the RDNA3 split-k matmul (rocm/quant/exl3_rdna3.cu), which exl3_gemm takes ahead of
every other path (return code 94) for any m: every bitrate and codebook, every rows-per-pass instance (m = 1..16 and
the multi-chunk tail above), k-splits from none to many, fp16 and fp32 output, and a k large enough that the rows
go through the transformed-input workspace in several passes.

Reference: testlib.exl3.linear_ref (fp32 product with the reconstructed weight). The kernel rounds the rotated input
to fp16 and accumulates in fp32, so the error is a few fp16 roundoffs of the output scale; a wrong tile, window or
split shows up as an O(1) error.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import CODEBOOKS, codebook_flags, generator, linear_ref, rand_linear

pytestmark = [pytest.mark.rocm_only]

RDNA3_PATH = 94


@pytest.fixture(autouse = True)
def _gfx11(device):
    if not torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx11"):
        pytest.skip("RDNA3 (gfx11) only")


def _gemm(A, w, codebook, c_dtype = torch.half):
    mcg, mul1 = codebook_flags(codebook)
    m, n = A.shape[0], w["trellis"].shape[1] * 16
    C = torch.full((m, n), float("nan"), dtype = c_dtype, device = A.device)
    A_had = torch.empty_like(A)
    path = ext.exl3_gemm(A, w["trellis"], C, w["suh"], A_had, w["svh"], 0, mcg, mul1, 0)
    return C, path


def _check(device, K, codebook, m, k, n, c_dtype = torch.half, seed = 0):
    gen = generator(seed + int(K * 10) * 1000 + m)
    w = rand_linear(k, n, K, gen, device)
    A = (torch.randn((m, k), generator = gen) * 0.5).half().to(device)
    C, path = _gemm(A, w, codebook, c_dtype)
    assert path == RDNA3_PATH, f"exl3_gemm took path {path}"
    ref = linear_ref(A, w["trellis"], w["suh"], w["svh"], K, codebook).float()
    rel = ((C.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()).item()
    assert rel < 2e-3, f"K={K} {codebook} m={m} k={k} n={n} {c_dtype}: relative RMS error {rel:.2e}"


@pytest.mark.parametrize("codebook", CODEBOOKS)
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@torch.inference_mode()
def test_bitrates(device, codebook, K):
    _check(device, K, codebook, 4, 1024, 768)


@pytest.mark.parametrize("K", [1.5, 2.5, 3.5])
@torch.inference_mode()
def test_half_bitrates(device, K):
    _check(device, K, "mul1", 3, 1024, 768)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 40, 144])
@pytest.mark.parametrize("c_dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_rows(device, m, c_dtype):
    _check(device, 4, "mul1", m, 2048, 1024, c_dtype)
    _check(device, 3, "mcg", m, 512, 256, c_dtype)


@pytest.mark.parametrize("k,n", [(128, 128), (5120, 128), (17408, 5120), (128, 17408)])
@torch.inference_mode()
def test_splits(device, k, n):
    # one Hadamard block (no split possible), few output groups (many splits), and wide shapes of a 27B model
    for K, codebook in ((4, "mul1"), (5, "3inst")):
        _check(device, K, codebook, 8, k, n)


@torch.inference_mode()
def test_input_workspace_passes(device):
    # 16 MiB transformed-input workspace: k = 65536 holds 128 rows per pass, so 144 rows take two
    _check(device, 2, "mul1", 144, 65536, 128)


@pytest.mark.parametrize("K", [2, 4])
@torch.inference_mode()
def test_deterministic(device, K):
    # Split partials are summed by the last block in a fixed order, so repeat launches are bit-identical
    gen = generator(7)
    w = rand_linear(4096, 1024, K, gen, device)
    A = (torch.randn((6, 4096), generator = gen) * 0.5).half().to(device)
    ref, _ = _gemm(A, w, "mul1")
    for _ in range(8):
        C, _ = _gemm(A, w, "mul1")
        assert torch.equal(C, ref)


def _path_taken(device: str) -> int:
    d = torch.device(device)
    w = rand_linear(512, 256, 4, generator(1), d)
    A = torch.randn((4, 512), device = d).half()
    return _gemm(A, w, "mul1")[1]


def test_disabled_falls_through(device):
    # EXL3_RDNA3_GEMM is read once per process, so check it in a fresh one
    from testlib.isolated import run_isolated
    assert run_isolated(_path_taken, str(device), env = {"EXL3_RDNA3_GEMM": "0"}) != RDNA3_PATH
