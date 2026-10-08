"""
ext.rms_norm: y = x * rsqrt(mean(x^2) + eps) * constant_scale * (w + constant_bias), optionally with per-row
weight groups (row r uses group r % w_groups) and add_residual (y += norm(x)), fp16/fp32 in and out, in place
or not. Checked against an fp32 torch reference.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


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
def test_rms_norm(device, batch_size, dim, in_dtype, out_dtype, epsilon):
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
def test_rms_norm_bias_scale_no_weight(device, dim, constant_bias, constant_scale):
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
def test_rms_norm_w_groups(device, dim, w_groups):
    # Row r uses weight group r % w_groups (interleaved streams, e.g. hyperconnection stacks)
    x = torch.randn(128, dim, dtype = torch.half, device = device)
    w = torch.randn(w_groups * dim, dtype = torch.half, device = device)
    y = torch.empty_like(x)
    rms_norm(x, w, y, 1e-6, w_groups = w_groups)
    torch.testing.assert_close(y, reference_rms_norm(x, w, 1e-6, torch.half, w_groups = w_groups), rtol = 1e-3, atol = 1e-3)


@pytest.mark.parametrize("dim", [256, 4096])
@pytest.mark.parametrize("out_dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_rms_norm_add_residual(device, dim, out_dtype):
    # add_residual: y receives norm(x) + the previous contents of y
    x = torch.randn(64, dim, dtype = torch.half, device = device)
    w = torch.randn(dim, dtype = torch.half, device = device)
    y = torch.randn(64, dim, dtype = out_dtype, device = device)
    prev = y.clone()
    rms_norm(x, w, y, 1e-6, add_residual = True)
    torch.testing.assert_close(y, reference_rms_norm(x, w, 1e-6, out_dtype, residual = prev), rtol = 2e-3, atol = 2e-3)


@pytest.mark.slow
@torch.inference_mode()
def test_rms_norm_rows_times_dim_beyond_int32(device):
    # rows * dim exceeds 2^31 elements: row offsets must be computed in 64-bit. Only the tail rows are
    # checked against the reference (the rest would need a second 4 GB buffer).
    if torch.cuda.get_device_properties(device).total_memory < 14 * 1024**3:
        pytest.skip("needs ~10 GB of free VRAM")
    dim = 8192
    rows = 2**31 // dim + 64
    x = torch.randn(rows, dim, dtype = torch.half, device = device)
    w = torch.randn(dim, dtype = torch.half, device = device)
    y = torch.empty_like(x)
    rms_norm(x, w, y, 1e-6)
    torch.cuda.synchronize(device)
    tail = slice(rows - 128, rows)
    torch.testing.assert_close(y[tail], reference_rms_norm(x[tail], w, 1e-6, torch.half), rtol = 1e-3, atol = 1e-3)
    del x, y
    torch.cuda.empty_cache()


def assert_device_ok(device):
    # A failed launch would leave an error for the next op on the device
    torch.cuda.synchronize(device)
    assert torch.ones(8, device = device).sum().item() == 8


@pytest.mark.parametrize("case", ["rows", "rows_span_heads", "rows_res", "dim", "dim_span_heads", "w_groups"])
@torch.inference_mode()
def test_rms_norm_empty(device, case):
    # Empty row axis: nothing to do (no launch, nothing written). Empty normalized axis: the RMS of an empty vector
    # is undefined, so it raises
    buf = torch.full((64,), 7.0, dtype = torch.half, device = device)
    w = torch.randn(128, dtype = torch.half, device = device)
    if case.startswith("rows"):
        span = case == "rows_span_heads"
        shape = (0, 4, 32) if span else (0, 128)
        x = torch.randn(shape, dtype = torch.half, device = device)
        y = buf[8:8].view(shape)
        ext.rms_norm(x, w, y, 1e-6, 0.0, 1.0, span, case == "rows_res", 1)
        assert (buf == 7.0).all()
    elif case.startswith("dim"):
        span = case == "dim_span_heads"
        shape = (4, 0, 32) if span else (4, 0)
        x = torch.randn(shape, dtype = torch.half, device = device)
        with pytest.raises(RuntimeError, match = "rms_norm: norm over an empty dimension"):
            ext.rms_norm(x, torch.empty(0, dtype = torch.half, device = device), torch.empty_like(x), 1e-6, 0.0, 1.0,
                         span, False, 1)
    else:
        x = torch.randn(4, 128, dtype = torch.half, device = device)
        with pytest.raises(RuntimeError, match = "w_groups must be positive"):
            ext.rms_norm(x, torch.empty(0, dtype = torch.half, device = device), torch.empty_like(x), 1e-6, 0.0, 1.0,
                         False, False, 0)
    assert_device_ok(device)


@torch.inference_mode()
def test_rms_norm_empty_weight_rejected(device):
    # An empty weight tensor is a weight of the wrong size, not "no weight"
    x = torch.randn(4, 128, dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.rms_norm(x, torch.empty(0, dtype = torch.half, device = device), torch.empty_like(x), 1e-6, 0.0, 1.0,
                     False, False, 1)
    # Dtype validation still applies to empty input
    with pytest.raises(RuntimeError, match = "Invalid datatypes"):
        ext.rms_norm(torch.empty(0, 128, dtype = torch.bfloat16, device = device), None,
                     torch.empty(0, 128, dtype = torch.half, device = device), 1e-6, 0.0, 1.0, False, False, 1)
