"""
Small-m EXL3 GEMM with the rotation workspace (A_had) ending exactly at the end of its allocation.

The RDNA multi-row GEMV processes rows in tiles of 2/4/8 and must not read input rows past
size_m (gpt-oss-20b's dense per-expert path hands it an m-row workspace from the caching
allocator; m = 3, 5, 6, 7 used to fault when that buffer sat at a segment boundary). On CUDA
the same shapes go through the regular GEMM; the test just checks the results there.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

DEV = torch.device("cuda:0")


def _weights(k, n, bits):
    tr = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype = torch.int16, device = DEV)
    suh = (torch.randint(0, 2, (k,), device = DEV) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), device = DEV) * 2 - 1).half()
    w = torch.empty(k, n, dtype = torch.half, device = DEV)
    ext.reconstruct(w, tr, bits, False, True)
    return tr, suh, svh, w


def _ref(x, suh, w, svh):
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y = (xh.float() @ w.float()).half()
    ext.had_r_128(y, y, None, svh, 1.0)
    return y.float()


def _tail(numel, dtype = torch.half):
    """A view of the last numel elements of a fresh, separately mapped allocation"""
    big = torch.empty(32 << 20, dtype = dtype, device = DEV)
    return big, big[-numel:]


def _ptrs(ts):
    return torch.tensor([t.data_ptr() for t in ts], dtype = torch.int64, device = DEV)


@pytest.mark.parametrize("k,n", [(2944, 2944), (2048, 3072)])
@pytest.mark.parametrize("fp32", [False, True])
def test_gemm_tail_workspace(k, n, fp32):
    torch.manual_seed(k + n)
    tr, suh, svh, w = _weights(k, n, 3)
    for m in range(1, 9):
        x = (torch.randn(m, k, device = DEV) * 0.3).half()
        keep, xh = _tail(m * k)
        xh = xh.view(m, k)
        y = torch.empty(m, n, dtype = torch.float if fp32 else torch.half, device = DEV)
        ext.exl3_gemm(x, tr, y, suh, xh, svh, 0, False, True, 0)
        torch.cuda.synchronize()
        ref = _ref(x, suh, w, svh)
        err = ((y.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        # (loose on purpose: CUDA's default int8-activation GEMV for small-m mul1 shapes deviates
        # ~1% RMS; a row read from the wrong place is an O(1) error)
        assert err < 2e-2, (m, err)
        del keep


def test_mgemm_tail_workspace():
    k, n, bits = 2944, 2944, 3
    torch.manual_seed(1)
    mats = [_weights(k, n, bits) for _ in range(3)]
    for m in range(1, 9):
        x = (torch.randn(m, k, device = DEV) * 0.3).half()
        keep, a_had = _tail(3 * m * k)
        a_had = a_had.view(3, m, k)
        out = torch.empty((3, m, n), dtype = torch.half, device = DEV)
        ext.exl3_mgemm(
            x.view(1, m, k), _ptrs([t[0] for t in mats]), out, _ptrs([t[1] for t in mats]), a_had,
            _ptrs([t[2] for t in mats]), None, None, bits, -1, 0, 1, -1, -1, 0, 1, None, None, None, None, 0
        )
        torch.cuda.synchronize()
        for j, (_, suh, svh, w) in enumerate(mats):
            ref = _ref(x, suh, w, svh)
            err = ((out[j].float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            assert err < 2e-2, (m, j, err)
        del keep
