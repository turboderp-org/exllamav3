"""
ext.rms_norm_res_in(x, w, y, r, eps, constant_bias, constant_scale), the fused pre-norm residual add of
TransformerBlock (RMSNorm.forward(residual_in = ...)). Contract, per row of the flattened (rows, dim) view:

    r <- round_to(r.dtype)(r + clamp_fp16(x))          (in place; clamp only for fp16 x, to +-65504)
    y  = r * rsqrt(mean(r^2) + eps) * constant_scale * (w + constant_bias)     (w None: no weight term)

- x float16 or float32, r float16 or float32 (x's shape), w float16 / bfloat16 / None, y float16 only
- the residual update is exact: one fp32 add rounded once to r's dtype, identical to an unfused torch add; y is
  computed from the rounded r (so it equals rms_norm of the updated residual)
- dim % 4 == 0; dims up to 4096 keep the row in registers, larger dims re-read r in a second pass
- x and w are read only; memory around r and y is untouched
- rejects (TORCH_CHECK): dim % 4 != 0, y or r shape != x shape, w size != dim, y float32 or other dtypes

Reference: float64 torch (the residual add in fp32 as an unfused torch op would do it).
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

GUARD = 64


def guarded(t):
    """Copy of t inside a buffer with GUARD NaN sentinels on each side"""
    buf = torch.full((t.numel() + 2 * GUARD,), float("nan"), dtype = t.dtype, device = t.device)
    v = buf[GUARD:GUARD + t.numel()].view(t.shape)
    v.copy_(t)
    return buf, v


def assert_guards_intact(buf, n):
    for part in (buf[:GUARD], buf[GUARD + n:]):
        assert torch.isnan(part).all(), "write outside the output tensor"


def reference(x, w, r, eps, constant_bias, constant_scale):
    xf = x.float()
    if x.dtype == torch.half:
        xf = xf.clamp(-65504.0, 65504.0)
    r_new = (r.float() + xf).to(r.dtype)
    rd = r_new.double()
    y = rd * torch.rsqrt(rd.pow(2).mean(-1, keepdim = True) + eps) * constant_scale
    if w is not None:
        y = y * (w.double() + constant_bias)
    return r_new, y


# y: fp32 sum of squares and rsqrtf (relative error ~1e-6), products of the normalized value with w, then one fp16
# rounding (2^-11 relative); all terms are products, so the bound is relative. atol covers fp16 subnormals.
Y_TOL = dict(rtol = 6e-4, atol = 1e-7)


def run_case(device, rows, dim, x_dtype, r_dtype, w_dtype = torch.half, constant_bias = 0.0, constant_scale = 1.0,
             eps = 1e-6):
    torch.manual_seed(rows * 31 + dim)
    x = torch.randn(rows, dim, device = device).to(x_dtype)
    r0 = (torch.randn(rows, dim, device = device) * 4).to(r_dtype)
    w = None if w_dtype is None else (torch.randn(dim, device = device) * 0.5 + 1.0).to(w_dtype)
    x0 = x.clone()
    w0 = None if w is None else w.clone()
    rbuf, r = guarded(r0)
    ybuf, y = guarded(torch.empty(rows, dim, dtype = torch.half, device = device))
    ext.rms_norm_res_in(x, w, y, r, eps, constant_bias, constant_scale)
    r_ref, y_ref = reference(x, w, r0, eps, constant_bias, constant_scale)
    assert torch.equal(r, r_ref), "residual update is not the exact rounded sum"
    torch.testing.assert_close(y.double(), y_ref, **Y_TOL)
    assert_guards_intact(rbuf, r.numel())
    assert_guards_intact(ybuf, y.numel())
    assert torch.equal(x, x0), "x modified"
    if w is not None:
        assert torch.equal(w, w0), "w modified"


@pytest.mark.parametrize("rows", [1, 5, 128, 1024])
@pytest.mark.parametrize("dim", [4, 8, 128, 2048, 4096, 4100, 8192, 12288])
@pytest.mark.parametrize("x_dtype, r_dtype", [(torch.half, torch.half), (torch.half, torch.float),
                                              (torch.float, torch.half), (torch.float, torch.float)])
@torch.inference_mode()
def test_rms_norm_res_in(device, rows, dim, x_dtype, r_dtype):
    # 4096 / 4100 straddle the register (one float4 per thread) / two-pass switch
    run_case(device, rows, dim, x_dtype, r_dtype)


@pytest.mark.parametrize("w_dtype", [torch.half, torch.bfloat16, None])
@pytest.mark.parametrize("constant_bias, constant_scale", [(0.0, 1.0), (1.0, 1.0), (0.0, 0.5), (1.0, 2.0)])
@pytest.mark.parametrize("dim", [256, 8192])
@torch.inference_mode()
def test_rms_norm_res_in_weight_variants(device, w_dtype, constant_bias, constant_scale, dim):
    if w_dtype is None and constant_bias != 0.0:
        pytest.skip("constant_bias applies to the weight")
    run_case(device, 33, dim, torch.half, torch.float, w_dtype, constant_bias, constant_scale)


@pytest.mark.parametrize("dim", [1024, 8192])
@torch.inference_mode()
def test_rms_norm_res_in_fp16_clamp_and_overflow(device, dim):
    # An inf in an fp16 sublayer output is clamped to 65504 before the add; an fp16 residual whose sum overflows
    # becomes inf exactly as the unfused add would
    x = torch.randn(4, dim, dtype = torch.half, device = device)
    x[0, 3] = float("inf")
    x[1, 5] = float("-inf")
    r0 = torch.randn(4, dim, dtype = torch.float, device = device)
    r = r0.clone()
    y = torch.empty(4, dim, dtype = torch.half, device = device)
    ext.rms_norm_res_in(x, None, y, r, 1e-6, 0.0, 1.0)
    r_ref, y_ref = reference(x, None, r0, 1e-6, 0.0, 1.0)
    assert torch.equal(r, r_ref)
    assert torch.isfinite(r).all()
    torch.testing.assert_close(y.double(), y_ref, **Y_TOL)

    xh = torch.full((2, dim), 60000.0, dtype = torch.half, device = device)
    rh0 = torch.full((2, dim), 60000.0, dtype = torch.half, device = device)
    rh = rh0.clone()
    ext.rms_norm_res_in(xh, None, y[:2], rh, 1e-6, 0.0, 1.0)
    assert torch.equal(rh, (rh0.float() + xh.float()).half())
    assert torch.isinf(rh).all()


@torch.inference_mode()
def test_rms_norm_res_in_rejects(device):
    x = torch.randn(4, 128, dtype = torch.half, device = device)
    r = torch.randn(4, 128, dtype = torch.half, device = device)
    w = torch.ones(128, dtype = torch.half, device = device)
    y = torch.empty(4, 128, dtype = torch.half, device = device)
    x6 = torch.randn(4, 130, dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "divisible"):
        ext.rms_norm_res_in(x6, None, x6.clone(), x6.clone(), 1e-6, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.rms_norm_res_in(x, w, y[:2], r, 1e-6, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.rms_norm_res_in(x, w, y, r[:2], 1e-6, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.rms_norm_res_in(x, w[:64], y, r, 1e-6, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "Invalid datatypes"):
        ext.rms_norm_res_in(x, w, y.float(), r, 1e-6, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "Invalid datatypes"):
        ext.rms_norm_res_in(x, w, y, r.bfloat16(), 1e-6, 0.0, 1.0)
