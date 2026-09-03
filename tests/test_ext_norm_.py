import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)
device = "cuda:0"


def reference_rms_norm(x, w, eps, out_dtype, constant_bias = 0.0, constant_scale = 1.0, w_groups = 1, residual = None):
    assert x.dtype in [torch.half, torch.float]
    x = x.float()
    var = (x * x).mean(dim = -1, keepdim = True) + eps
    x = x * torch.rsqrt(var) * constant_scale
    if w is not None:
        w = w.float() + constant_bias
        if w_groups > 1:
            w = w.view(w_groups, -1)[torch.arange(x.shape[0], device = x.device) % w_groups]
        x = x * w
    if residual is not None:
        x = x + residual.float()
    return x.to(out_dtype)


def rms_norm(x, w, y, eps, constant_bias = 0.0, constant_scale = 1.0, span_heads = False, add_residual = False, w_groups = 1):
    ext.rms_norm(x, w, y, eps, constant_bias, constant_scale, span_heads, add_residual, w_groups)


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
    rms_norm(x, w, y, epsilon)
    torch.testing.assert_close(y, ref_y, rtol = 1e-3, atol = 1e-3)
    if in_dtype == out_dtype:
        rms_norm(x, w, x, epsilon)
        torch.testing.assert_close(x, y, rtol = 1e-3, atol = 1e-3)


@pytest.mark.parametrize("dim", [256, 4096])
@pytest.mark.parametrize("constant_bias, constant_scale", [(0.0, 1.0), (1.0, 1.0), (0.0, 0.5)])
@torch.inference_mode()
def test_rms_norm_bias_scale_no_weight(dim, constant_bias, constant_scale):
    x = torch.randn(64, dim, dtype = torch.half, device = device)
    w = torch.randn(dim, dtype = torch.half, device = device)
    y = torch.empty_like(x)
    rms_norm(x, w, y, 1e-6, constant_bias, constant_scale)
    torch.testing.assert_close(y, reference_rms_norm(x, w, 1e-6, torch.half, constant_bias, constant_scale), rtol = 1e-3, atol = 1e-3)
    if constant_bias == 0.0:
        rms_norm(x, None, y, 1e-6, 0.0, constant_scale)
        torch.testing.assert_close(y, reference_rms_norm(x, None, 1e-6, torch.half, 0.0, constant_scale), rtol = 1e-3, atol = 1e-3)


@pytest.mark.parametrize("dim", [256, 4096])
@pytest.mark.parametrize("w_groups", [2, 4])
@torch.inference_mode()
def test_rms_norm_w_groups(dim, w_groups):
    # Row r uses weight group r % w_groups (interleaved streams, e.g. hyperconnection stacks)
    x = torch.randn(128, dim, dtype = torch.half, device = device)
    w = torch.randn(w_groups * dim, dtype = torch.half, device = device)
    y = torch.empty_like(x)
    rms_norm(x, w, y, 1e-6, w_groups = w_groups)
    torch.testing.assert_close(y, reference_rms_norm(x, w, 1e-6, torch.half, w_groups = w_groups), rtol = 1e-3, atol = 1e-3)


@pytest.mark.parametrize("dim", [256, 4096])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_rms_norm_add_residual(dim, out_dtype):
    # add_residual: y receives norm(x) + the previous contents of y
    x = torch.randn(64, dim, dtype = torch.half, device = device)
    w = torch.randn(dim, dtype = torch.half, device = device)
    y = torch.randn(64, dim, dtype = out_dtype, device = device)
    prev = y.clone()
    rms_norm(x, w, y, 1e-6, add_residual = True)
    torch.testing.assert_close(y, reference_rms_norm(x, w, 1e-6, out_dtype, residual = prev), rtol = 2e-3, atol = 2e-3)


@pytest.mark.skipif(torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3, reason = "needs ~10 GB of VRAM")
@torch.inference_mode()
def test_rms_norm_rows_times_dim_beyond_int32():
    # rows * dim exceeds 2^31 elements: row offsets must be computed in 64-bit. Only the tail rows are
    # checked against the reference (the rest would need a second 4 GB buffer).
    dim = 8192
    rows = 2**31 // dim + 64
    x = torch.randn(rows, dim, dtype = torch.half, device = device)
    w = torch.randn(dim, dtype = torch.half, device = device)
    y = torch.empty_like(x)
    rms_norm(x, w, y, 1e-6)
    torch.cuda.synchronize()
    tail = slice(rows - 128, rows)
    torch.testing.assert_close(y[tail], reference_rms_norm(x[tail], w, 1e-6, torch.half), rtol = 1e-3, atol = 1e-3)
    del x, y
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_rms_norm_benchmark_smoke():
    pytest.importorskip("pytest_benchmark")   # the timing variant lives in benchmarks/; this only checks a large launch runs
    x = torch.randn(1024, 12288, dtype = torch.half, device = device)
    w = torch.randn(12288, dtype = torch.half, device = device)
    y = torch.empty_like(x)
    rms_norm(x, w, y, 1e-5)
    torch.cuda.synchronize()
