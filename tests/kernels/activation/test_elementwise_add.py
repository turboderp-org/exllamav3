"""
ext.add(x, y, z): z = x + y with y repeated over x in flat (row-major) order, i.e.

    z.view(-1)[i] = x.view(-1)[i] + y.view(-1)[i % y.numel()]

(the bias add of the BC_* linear / MLP / attention / GDN graphs via add_gr; y.numel() == x.numel() or a proper
divisor of it, e.g. a (dim,) bias over (rows, dim)). x, y, z each float16 or float32 (all 8 combinations). If x and
y are both fp16 the sum is an fp16 add (__hadd) converted to z's dtype, otherwise an fp32 add rounded once to z's
dtype: either way bit-identical to the unfused torch expression. Works in place (z is x, or z is y when the sizes
are equal), writes exactly z.numel() == x.numel() elements, 64-bit indexing. Rejects (TORCH_CHECK) y.numel() >
x.numel() and y.numel() not dividing x.numel().

Reference: torch elementwise ops (exact).
"""

import itertools

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

GUARD = 64
DTYPES = [torch.half, torch.float]


def guarded_empty(shape, dtype, device):
    n = 1
    for s in shape:
        n *= s
    buf = torch.full((n + 2 * GUARD,), float("nan"), dtype = dtype, device = device)
    return buf, buf[GUARD:GUARD + n].view(shape)


def assert_guards_intact(buf, n):
    for part in (buf[:GUARD], buf[GUARD + n:]):
        assert torch.isnan(part).all(), "write outside the output tensor"


def reference(x, y, z_dtype):
    yr = y.reshape(-1).repeat(x.numel() // y.numel()).view(x.shape)
    if x.dtype == torch.half and yr.dtype == torch.half:
        s = x + yr
    else:
        s = x.float() + yr.float()
    return s.to(z_dtype)


def bits_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    it = torch.int16 if a.dtype == torch.half else torch.int32
    return torch.equal(a.view(it), b.view(it))


@pytest.mark.parametrize("xt, yt, zt", list(itertools.product(DTYPES, DTYPES, DTYPES)))
@pytest.mark.parametrize("shape, y_shape", [
    ((1,), (1,)),
    ((7, 33), (7, 33)),
    ((4, 4096), (4096,)),
    ((3, 5, 1000), (1000,)),
    ((6, 128), (2, 128)),       # period of two rows: flat modulo, not a trailing-dim broadcast
    ((1025, 3), (1,)),          # scalar over a grid that is not a multiple of the block size
])
@torch.inference_mode()
def test_add(device, xt, yt, zt, shape, y_shape):
    torch.manual_seed(len(shape) * 1000 + shape[-1])
    # Mixed magnitudes (finite inputs, so no NaN payloads to compare) so the fp16 sums round
    x = (torch.randn(shape, device = device) * torch.logspace(-3, 4, shape[-1], device = device)).to(xt)
    y = (torch.randn(y_shape, device = device) * 1000).to(yt)
    x0, y0 = x.clone(), y.clone()
    buf, z = guarded_empty(shape, zt, device)
    ext.add(x, y, z)
    assert bits_equal(z, reference(x, y, zt))
    assert bits_equal(x, x0) and bits_equal(y, y0), "inputs modified"
    assert_guards_intact(buf, z.numel())


@pytest.mark.parametrize("xt, yt", list(itertools.product(DTYPES, DTYPES)))
@torch.inference_mode()
def test_add_in_place(device, xt, yt):
    # z is x (bias add in place, the add_gr usage) and z is y
    x = torch.randn(64, 512, device = device).to(xt)
    b = torch.randn(512, device = device).to(yt)
    expected = reference(x, b, xt)
    ext.add(x, b, x)
    assert bits_equal(x, expected)

    x = torch.randn(64, 512, device = device).to(xt)
    y = torch.randn(64, 512, device = device).to(yt)
    expected = reference(x, y, yt)
    ext.add(x, y, y)
    assert bits_equal(y, expected)


@torch.inference_mode()
def test_add_rejects(device):
    x = torch.randn(4, 64, device = device).half()
    with pytest.raises(RuntimeError, match = r"y > x"):
        ext.add(x, torch.randn(5, 64, device = device).half(), torch.empty_like(x))
    with pytest.raises(RuntimeError, match = "y must divide x"):
        ext.add(x, torch.randn(3, 64, device = device).half(), torch.empty_like(x))


@pytest.mark.slow
@pytest.mark.vram(16)
@torch.inference_mode()
def test_add_numel_beyond_int32(device):
    # x.numel() > 2^31: index and modulo are 64-bit. In-place bias add, tail rows checked
    dim = 4096
    rows = 2**31 // dim + 64
    x = torch.zeros(rows, dim, dtype = torch.half, device = device)
    tail = slice(rows - 128, rows)
    x[tail] = torch.randn(128, dim, device = device).half()
    x_tail = x[tail].clone()
    b = torch.randn(dim, device = device).half()
    ext.add(x, b, x)
    assert bits_equal(x[tail], reference(x_tail, b, torch.half))
    assert bits_equal(x[:4], b.expand(4, dim).contiguous())
    del x
    torch.cuda.empty_cache()
