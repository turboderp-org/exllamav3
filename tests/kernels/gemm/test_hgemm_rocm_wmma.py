"""
ext.hgemm on ROCm through the WMMA GEMM (rocm/wmma_gemm.cu): the shapes the per-arch table routes there by default,
and every config pinned with EXL3_ROCM_WMMA_GEMM_CFG, match an fp32 reference for fp32 and fp16 output, on and off
the table's tuning grid (row counts and widths between grid points, odd K, N not a multiple of the tile).
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = [pytest.mark.rocm_only]

SHAPES = [
    (16, 512, 256), (100, 5120, 48), (145, 6144, 5120), (300, 17408, 5120), (300, 5120, 6144),
    (511, 2048, 1000), (513, 4096, 4096), (1000, 1024, 2056), (2048, 5120, 5120),
]


def check(device, M, K, N, dtype):
    torch.manual_seed(12345)
    a = torch.randn(M, K, device = device, dtype = torch.half)
    b = (torch.randn(K, N, device = device) / K ** 0.5).half()
    c = torch.full((M, N), float("nan"), device = device, dtype = dtype)
    ext.hgemm(a, b, c)
    ref = a.float() @ b.float()
    relative_rms = ((c.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert float(relative_rms) < (1e-4 if dtype == torch.float32 else 1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("M,K,N", SHAPES)
@torch.inference_mode()
def test_wmma_default_routing(device, monkeypatch, dtype, M, K, N):
    monkeypatch.delenv("EXL3_ROCM_WMMA_GEMM", raising = False)
    monkeypatch.delenv("EXL3_ROCM_WMMA_GEMM_CFG", raising = False)
    check(device, M, K, N, dtype)


@pytest.mark.parametrize("cfg", range(21))
@pytest.mark.parametrize("M,K,N", [(16, 512, 256), (300, 6144, 5120), (513, 4096, 1000)])
@torch.inference_mode()
def test_wmma_every_config(device, monkeypatch, cfg, M, K, N):
    # Mode 2 routes every eligible call; a config not compiled for this arch, or one whose K tile does not divide K,
    # falls back to hipBLAS, which must be correct as well
    monkeypatch.setenv("EXL3_ROCM_WMMA_GEMM", "2")
    monkeypatch.setenv("EXL3_ROCM_WMMA_GEMM_CFG", str(cfg))
    for dtype in (torch.float32, torch.float16):
        check(device, M, K, N, dtype)
