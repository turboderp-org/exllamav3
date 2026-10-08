"""
Small-m EXL3 GEMM with the rotation workspace (A_had) ending exactly at the end of its allocation.

The RDNA multi-row GEMV processes rows in tiles of 2/4/8 and must not read input rows past
size_m (gpt-oss-20b's dense per-expert path hands it an m-row workspace from the caching
allocator; m = 3, 5, 6, 7 used to fault when that buffer sat at a segment boundary). On CUDA
the same shapes go through the regular GEMM. Results are checked against testlib.exl3.linear_ref (mul1 codebook,
dequantize then fp32 matmul).
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, linear_ref, rand_linear


def _weights(k, n, bits, gen, device):
    w = rand_linear(k, n, bits, gen, device)
    return w["trellis"], w["suh"], w["svh"]


def _ref(x, tr, suh, svh, bits):
    return linear_ref(x, tr, suh, svh, bits, "mul1").float()


def _rel_rms(y, ref):
    return ((y.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def _tail(numel, device, dtype = torch.half):
    """A view of the last numel elements of a fresh, separately mapped allocation"""
    big = torch.empty(32 << 20, dtype = dtype, device = device)
    return big, big[-numel:]


def _ptrs(ts, device):
    return torch.tensor([t.data_ptr() for t in ts], dtype = torch.int64, device = device)


@pytest.mark.parametrize("k,n", [(2944, 2944), (2048, 3072)])
@pytest.mark.parametrize("fp32", [False, True])
@torch.inference_mode()
def test_gemm_tail_workspace(device, k, n, fp32):
    gen = generator(k + n)
    tr, suh, svh = _weights(k, n, 3, gen, device)
    for m in range(1, 9):
        x = (torch.randn(m, k, generator = gen) * 0.3).half().to(device)
        keep, xh = _tail(m * k, device)
        xh = xh.view(m, k)
        y = torch.empty(m, n, dtype = torch.float if fp32 else torch.half, device = device)
        ext.exl3_gemm(x, tr, y, suh, xh, svh, 0, False, True, 0)
        torch.cuda.synchronize(device)
        err = _rel_rms(y, _ref(x, tr, suh, svh, 3))
        # (loose on purpose: CUDA's default int8-activation GEMV for small-m mul1 shapes deviates
        # ~1% RMS; a row read from the wrong place is an O(1) error)
        assert err < 2e-2, (m, err)
        del keep


@torch.inference_mode()
def test_mgemm_tail_workspace(device):
    k, n, bits = 2944, 2944, 3
    gen = generator(1)
    mats = [_weights(k, n, bits, gen, device) for _ in range(3)]
    for m in range(1, 9):
        x = (torch.randn(m, k, generator = gen) * 0.3).half().to(device)
        keep, a_had = _tail(3 * m * k, device)
        a_had = a_had.view(3, m, k)
        out = torch.empty((3, m, n), dtype = torch.half, device = device)
        ext.exl3_mgemm(
            x.view(1, m, k), _ptrs([t[0] for t in mats], device), out, _ptrs([t[1] for t in mats], device), a_had,
            _ptrs([t[2] for t in mats], device), None, None, bits, -1, 0, 1, -1, -1, 0, 1, None, None, None, None, 0
        )
        torch.cuda.synchronize(device)
        for j, (tr, suh, svh) in enumerate(mats):
            err = _rel_rms(out[j], _ref(x, tr, suh, svh, bits))
            assert err < 2e-2, (m, j, err)
        del keep
