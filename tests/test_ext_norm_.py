import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

device = "cuda:0"

def reference_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, out_dtype: torch.dtype):
    assert x.dtype in [torch.half, torch.float]
    assert w.dtype in [torch.half]
    x = x.float()
    w = w.float()
    var = (x * x).mean(dim = -1, keepdim = True) + eps
    x = x * w * torch.rsqrt(var)
    x = x.to(out_dtype)
    return x

@pytest.mark.parametrize("batch_size", [1, 4, 16, 384, 1024, 4096])
@pytest.mark.parametrize("dim", [8, 256, 384, 1024, 1536, 8192, 12288])
@pytest.mark.parametrize("in_dtype", [torch.half, torch.float])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
@pytest.mark.parametrize("epsilon", [1e-5, 1e-6])
@torch.inference_mode()
def test_rms_norm(batch_size, dim, in_dtype, out_dtype, epsilon):

    x = torch.randn(batch_size, dim, dtype = in_dtype, device = device)
    w = torch.randn(dim, dtype = torch.half, device = device)
    y = torch.empty_like(x, dtype = out_dtype)

    ref_y = reference_rms_norm(x, w, epsilon, y.dtype)

    ext.rms_norm(x, w, y, epsilon)
    torch.testing.assert_close(y, ref_y, rtol = 1e-3, atol = 1e-3)

    if in_dtype == out_dtype:
        ext.rms_norm(x, w, x, epsilon)
        torch.testing.assert_close(x, y, rtol = 1e-3, atol = 1e-3)

bm_batch = 8192
bm_batch_size = [1, 4, 1024]
bm_dim = [4096, 12288]

@pytest.mark.parametrize("batch_size", bm_batch_size)
@pytest.mark.parametrize("dim", bm_dim)
# @pytest.mark.parametrize("in_dtype", [torch.half, torch.float])
# @pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
@pytest.mark.parametrize("in_dtype", [torch.half])
@pytest.mark.parametrize("out_dtype", [torch.half])
@pytest.mark.benchmark(disable_gc = True, warmup = True)
@torch.inference_mode()
def test_rms_norm_benchmark(benchmark, batch_size, dim, in_dtype, out_dtype):

    x = torch.randn(batch_size, dim, dtype = in_dtype, device = device)
    w = torch.randn(dim, dtype = torch.half, device = device)
    y = torch.empty_like(x, dtype = out_dtype)
    epsilon = 1e-5
    torch.cuda.synchronize()

    def run():
        torch.cuda.synchronize()
        for _ in range(bm_batch // batch_size):
            ext.rms_norm(x, w, y, epsilon)
        torch.cuda.synchronize()

    benchmark(run)

