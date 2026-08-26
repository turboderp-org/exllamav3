# The EXL3 GEMM's threadblock k-reduction, at every TILEBLOCKS_K the kernel can be built at.
#
# The shipping shape table tops out at TILEBLOCKS_K == 2, which runs a single store/add pair, so
# nothing in the table exercises the staging regions the reduction interleaves at higher k-tile
# counts. These tests instantiate the kernel directly at TILESIZE_K 16 / 32 / 64 (TILEBLOCKS_K
# 1 / 2 / 4) so a shape added to the table cannot arrive without coverage.
#
# TILEBLOCKS_K == 1 runs no threadblock reduction at all and is the reference. The failure mode
# under test is non-deterministic, so every cell checks bitwise launch-to-launch stability as well
# as error against the reference, and a perturbed-input control confirms the comparison
# discriminates at all.

import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(ROOT, "exllamav3", "exllamav3_ext")
KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
NVCC = os.path.join(CUDA_HOME, "bin", "nvcc")

# TILESIZE_K -> TILEBLOCKS_K. 16 is the reference; 32 is what the shipping table reaches; 64 is
# the first count that interleaves a second store into a live staging region.
TILESIZE_K_REF = 16
TILESIZE_K_UNDER_TEST = [32, 64]

BITS = [2, 3, 4, 5, 6, 7, 8]
# Straddles the size_m <= 8 store_small/add_small path and the full-width path
SIZE_M = [1, 4, 8, 9, 20, 33]

SIZE_K = 2048
SIZE_N = 512
REPS = 8

# Quantization noise against the reference sits at ~1e-5 of output RMS; the reduction defect
# presents at 0.4-4.8, so any threshold in between separates them.
REL_TOL = 0.01
# The perturbed-input control must clear this, or the cell is not comparing anything
REL_CONTROL_MIN = 0.02


def _nvcc_available():
    return os.path.isfile(NVCC)


def _guard_arch():
    # The guard is a static_assert on template parameters, so any arch the kernel body itself
    # compiles for will do. cp.async in the pipeline needs sm_80 or later.
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            return f"sm_{major * 10 + minor}"
    return "sm_80"


@pytest.fixture(scope = "module")
def probe():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if not _nvcc_available():
        pytest.skip(f"no nvcc at {NVCC}")
    from torch.utils.cpp_extension import load
    return load(
        name = "exl3_k_reduction_probe",
        sources = [os.path.join(KERNEL_DIR, "exl3_k_reduction_probe.cu")],
        extra_include_paths = [EXT_DIR],
        extra_cflags = ["-O2"],
        extra_cuda_cflags = [
            "-O3", "--use_fast_math",
            "-Xcudafe", "--diag_suppress=177",
            "-Xcudafe", "--diag_suppress=20012",
        ],
        verbose = False,
    )


@pytest.fixture(scope = "module")
def device():
    return "cuda:0"


def _operands(bits, size_m, device):
    g = torch.Generator(device = "cpu").manual_seed(1234 + bits)
    B = torch.randint(-32768, 32767, (SIZE_K // 16, SIZE_N // 16, 16 * bits),
                      dtype = torch.int16, generator = g).to(device)
    suh = (torch.randn(SIZE_K, generator = g) * 0.1 + 1.0).half().to(device)
    svh = (torch.randn(SIZE_N, generator = g) * 0.1 + 1.0).half().to(device)
    A = (torch.randn((size_m, SIZE_K), generator = g) * 0.05).half().to(device)
    locks = torch.zeros(SIZE_N // 16 + 1024, dtype = torch.int32, device = device)
    return A, B, suh, svh, locks


def _run(probe, tilesize_k, A, B, suh, svh, locks, device):
    C = torch.zeros((A.size(0), SIZE_N), dtype = torch.float32, device = device)
    A_had = torch.empty_like(A)
    probe.probe_gemm(A, B, C, suh, A_had, svh, locks, tilesize_k)
    torch.cuda.synchronize()
    return C


@pytest.mark.parametrize("tilesize_k", TILESIZE_K_UNDER_TEST)
@pytest.mark.parametrize("size_m", SIZE_M)
@pytest.mark.parametrize("bits", BITS)
@torch.inference_mode()
def test_k_reduction_matches_single_k_tile(probe, device, bits, size_m, tilesize_k):
    A, B, suh, svh, locks = _operands(bits, size_m, device)

    ref = _run(probe, TILESIZE_K_REF, A, B, suh, svh, locks, device)
    rms = ref.double().pow(2).mean().sqrt().item()
    assert rms > 0.0

    # Control: the reference kernel on perturbed input, so a cell that cannot tell the two apart
    # fails loudly instead of passing on a comparison that measures nothing
    A_perturbed = (A.float() * 1.02).half().contiguous()
    control = _run(probe, TILESIZE_K_REF, A_perturbed, B, suh, svh, locks, device)
    rel_control = (control - ref).abs().max().item() / rms
    assert rel_control > REL_CONTROL_MIN, \
        f"perturbed-input control only reached {rel_control:.6f}; the comparison does not discriminate"

    outs = []
    for i in range(REPS):
        # Interleave the reference so threadblock scheduling and L2 state vary between reps
        if i:
            _run(probe, TILESIZE_K_REF, A, B, suh, svh, locks, device)
        outs.append(_run(probe, tilesize_k, A, B, suh, svh, locks, device))

    assert not any(torch.isnan(o).any().item() for o in outs), \
        f"TILESIZE_K {tilesize_k} produced NaN at bits {bits}, size_m {size_m}"

    for i, o in enumerate(outs[1:], 1):
        assert torch.equal(outs[0], o), \
            f"TILESIZE_K {tilesize_k} is not deterministic across launches on identical input " \
            f"(rep {i} differs) at bits {bits}, size_m {size_m}"

    rel = max((o - ref).abs().max().item() for o in outs) / rms
    assert rel < REL_TOL, \
        f"TILESIZE_K {tilesize_k} differs from the TILESIZE_K {TILESIZE_K_REF} reference by " \
        f"{rel:.6f} of output RMS at bits {bits}, size_m {size_m}"


# A_COLS = TILESIZE_K / 8 that is not a power of two makes the A-fragment XOR swizzle address
# past the end of a row. The kernel must refuse to compile rather than silently alias rows.
@pytest.mark.parametrize("tilesize_k, buildable", [
    (16, True), (32, True), (64, True),
    (48, False), (80, False), (112, False),
])
def test_tilesize_k_over_eight_must_be_power_of_two(tmp_path, tilesize_k, buildable):
    if not _nvcc_available():
        pytest.skip(f"no nvcc at {NVCC}")
    result = subprocess.run(
        [
            NVCC, "-std=c++20", f"-arch={_guard_arch()}", "-c",
            "-I", EXT_DIR, f"-DPROBE_TILESIZE_K={tilesize_k}",
            os.path.join(KERNEL_DIR, "exl3_tilesize_k_guard.cu"),
            "-o", str(tmp_path / "guard.o"),
        ],
        capture_output = True, text = True,
    )
    guard = "TILESIZE_K / 8 must be a power of two"
    if buildable:
        assert result.returncode == 0, \
            f"TILESIZE_K {tilesize_k} should compile but did not:\n{result.stderr}"
    else:
        assert result.returncode != 0, f"TILESIZE_K {tilesize_k} compiled, but its swizzle aliases rows"
        assert guard in result.stderr, \
            f"TILESIZE_K {tilesize_k} was rejected, but not by the swizzle guard:\n{result.stderr}"
