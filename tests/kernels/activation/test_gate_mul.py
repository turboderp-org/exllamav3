"""
In-place gating kernels of the attention output gates and the shared-expert gate:

- ext.mul_sigmoid_(x, y): x *= sigmoid(y), elementwise. x, y float16, same shape, both contiguous, numel even
  (half2 lanes). The gate is evaluated in fp16 arithmetic (h2exp / h2rcp), so sigmoid(y) flushes to 0 for
  y < ~-11.1 where exp(-y) overflows fp16 (true value < 1.5e-5)
- ext.mul_sigmoid_broadcast_(x, y): x[b, s, h, :] *= sigmoid(y[b, s, h]); x (B, S, H, D) and y (B, S, H) float16,
  contiguous, D even; same fp16 gate arithmetic
- ext.mul_softplus_broadcast_(x, y): x[b, s, h, :] *= softplus(y[b, s, h]); same shapes; softplus in fp32
  (max(y, 0) + log1p(exp(-|y|))), product in fp32 with one rounding to fp16
- ext.add_sigmoid_gate(x, y, z): z += x * sigmoid(y) with y holding one gate logit per row of x (y.size(-1) == 1);
  all float32, fp32 arithmetic

All modify only their output tensor (x, or z) and leave the gate input untouched. Rejections (TORCH_CHECK): wrong
dtypes, mismatched / non-broadcastable shapes, non-contiguous inputs, odd numel / D, and gate size(-1) != 1 for
add_sigmoid_gate.

Reference: float64 torch on the same inputs.
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


# fp16 gate: exp, add, reciprocal and the product each round to fp16 (2^-11 relative each), and h2exp / h2rcp are
# ~1 ulp approximations; the exp error enters sigmoid scaled by e / (1 + e) <= 1. Together <= ~6 fp16 ulps
# relative, 4e-3. Where exp(-y) overflows, the gate flushes to 0 with an absolute error below |x| * 1.5e-5
# (|x| < 6 here), covered by atol.
FP16_GATE_TOL = dict(rtol = 4e-3, atol = 1e-4)


@pytest.mark.parametrize("shape", [(2,), (1, 128), (7, 4096), (3, 5, 2048), (1, 33, 32 * 128), (1023, 6)])
@torch.inference_mode()
def test_mul_sigmoid_(device, shape):
    torch.manual_seed(len(shape) * 100 + shape[-1])
    x0 = torch.randn(shape, device = device).half()
    y = (torch.randn(shape, device = device) * 6).half()
    y0 = y.clone()
    buf, x = guarded(x0)
    ext.mul_sigmoid_(x, y)
    ref = x0.double() * torch.sigmoid(y.double())
    torch.testing.assert_close(x.double(), ref, **FP16_GATE_TOL)
    assert torch.equal(y, y0), "gate modified"
    assert_guards_intact(buf, x.numel())


@torch.inference_mode()
def test_mul_sigmoid_saturation(device):
    # Large positive gate -> exactly x, large negative -> 0 (fp16 exp overflow); no NaN anywhere in fp16 range
    x0 = torch.randn(8, 64, device = device).half()
    y = torch.empty(8, 64, device = device).half()
    y[0::2] = 60000.0
    y[1::2] = -60000.0
    x = x0.clone()
    ext.mul_sigmoid_(x, y)
    assert torch.equal(x[0::2], x0[0::2])
    assert torch.equal(x[1::2].abs(), torch.zeros_like(x[1::2]))


@pytest.mark.parametrize("B, S, H, D", [(1, 1, 1, 2), (1, 1, 32, 128), (2, 7, 8, 64), (1, 33, 16, 256),
                                        (3, 5, 3, 6), (1, 1, 1, 1024), (1, 300, 4, 96)])
@pytest.mark.parametrize("fn", ["sigmoid", "softplus"])
@torch.inference_mode()
def test_mul_gate_broadcast_(device, B, S, H, D, fn):
    torch.manual_seed(B * S * H + D)
    x0 = torch.randn(B, S, H, D, device = device).half()
    y = (torch.randn(B, S, H, device = device) * 6).half()
    y0 = y.clone()
    buf, x = guarded(x0)
    if fn == "sigmoid":
        ext.mul_sigmoid_broadcast_(x, y)
        ref = x0.double() * torch.sigmoid(y.double()).unsqueeze(-1)
        torch.testing.assert_close(x.double(), ref, **FP16_GATE_TOL)
    else:
        ext.mul_softplus_broadcast_(x, y)
        # fp32 softplus via __expf / log1pf (a few fp32 ulps) and an fp32 product, then one fp16 rounding: the
        # result is the correctly rounded fp16 of the float64 product, or one fp16 ulp off at a rounding tie
        ref = (x0.double() * torch.nn.functional.softplus(y.double()).unsqueeze(-1)).half()
        torch.testing.assert_close(x, ref, rtol = 2 ** -10, atol = 2 ** -24)
    assert torch.equal(y, y0), "gate modified"
    assert_guards_intact(buf, x.numel())


@torch.inference_mode()
def test_mul_softplus_broadcast_extremes(device):
    # softplus(y) == y for large y (fp16 exp would overflow at ~11), -> 0 for very negative y; product overflow to
    # inf matches the rounding of the exact product
    x0 = torch.full((1, 1, 6, 4), 2.0, device = device).half()
    y = torch.tensor([[[12.0, 30.0, 60000.0, -20.0, -60000.0, 0.0]]], device = device).half()
    x = x0.clone()
    ext.mul_softplus_broadcast_(x, y)
    ref = (x0.double() * torch.nn.functional.softplus(y.double()).unsqueeze(-1)).half()
    torch.testing.assert_close(x, ref, rtol = 2 ** -10, atol = 2 ** -24)
    assert torch.isinf(x[0, 0, 2]).all()


@pytest.mark.parametrize("fn", [ext.mul_sigmoid_broadcast_, ext.mul_softplus_broadcast_])
@torch.inference_mode()
def test_mul_gate_broadcast_rejects(device, fn):
    x = torch.randn(1, 2, 4, 8, device = device).half()
    y = torch.randn(1, 2, 4, device = device).half()
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        fn(x.float(), y)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        fn(x, y.float())
    with pytest.raises(RuntimeError, match = r"\[B, S, H, D\]"):
        fn(x.view(2, 4, 8), y)
    with pytest.raises(RuntimeError, match = r"\[B, S, H\]"):
        fn(x, y.view(8))
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        fn(x, torch.randn(1, 2, 2, device = device).half())
    with pytest.raises(RuntimeError, match = "contiguous"):
        fn(torch.randn(1, 4, 2, 8, device = device).half().transpose(1, 2), y)
    with pytest.raises(RuntimeError, match = "contiguous"):
        fn(x, torch.randn(1, 4, 2, device = device).half().transpose(1, 2))
    with pytest.raises(RuntimeError, match = "even"):
        fn(torch.randn(1, 2, 4, 7, device = device).half(), y)


@torch.inference_mode()
def test_mul_sigmoid_rejects(device):
    x = torch.randn(4, 64, device = device).half()
    y = torch.randn(4, 64, device = device).half()
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.mul_sigmoid_(x.float(), y)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.mul_sigmoid_(x, y.float())
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.mul_sigmoid_(x, y.view(64, 4))
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.mul_sigmoid_(torch.randn(64, 4, device = device).half().t(), y)
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.mul_sigmoid_(x, torch.randn(64, 4, device = device).half().t())
    with pytest.raises(RuntimeError, match = "even"):
        ext.mul_sigmoid_(torch.randn(3, 3, device = device).half(), torch.randn(3, 3, device = device).half())


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (33, 2048), (64, 4096), (2, 17, 2049), (100, 1)])
@torch.inference_mode()
def test_add_sigmoid_gate(device, shape):
    # Shared-expert gate for bsz > 32: z (rows, dim) += x * sigmoid(y), y (rows, 1)
    torch.manual_seed(shape[0] * 13 + shape[-1])
    x = torch.randn(shape, device = device)
    y = torch.randn(shape[:-1] + (1,), device = device) * 6
    z0 = torch.randn(shape, device = device)
    x0, y0 = x.clone(), y.clone()
    buf, z = guarded(z0)
    ext.add_sigmoid_gate(x, y, z)
    ref = z0.double() + x.double() * torch.sigmoid(y.double())
    # fp32: __expf (a few ulp), one division, one fma-able multiply-add: ~1e-6 relative to max(|z|, |x|)
    torch.testing.assert_close(z.double(), ref, rtol = 1e-5, atol = 1e-6)
    assert torch.equal(x, x0) and torch.equal(y, y0), "inputs modified"
    assert_guards_intact(buf, z.numel())


@torch.inference_mode()
def test_add_sigmoid_gate_rejects(device):
    x = torch.randn(4, 64, device = device)
    y = torch.randn(4, 1, device = device)
    z = torch.randn(4, 64, device = device)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.add_sigmoid_gate(x.half(), y, z)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.add_sigmoid_gate(x, y.half(), z)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.add_sigmoid_gate(x, y, z.half())
    with pytest.raises(RuntimeError, match = "size"):
        ext.add_sigmoid_gate(x, torch.randn(4, 2, device = device), z)
