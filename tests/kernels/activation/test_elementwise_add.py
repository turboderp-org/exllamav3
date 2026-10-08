"""
ext.add(x, y, z): z = x + y with y repeated over x in flat (row-major) order, i.e.

    z.view(-1)[i] = x.view(-1)[i] + y.view(-1)[i % y.numel()]

(the bias add of the BC_* linear / MLP / attention / GDN graphs via add_gr; y.numel() == x.numel() or a proper
divisor of it, e.g. a (dim,) bias over (rows, dim)). x, y, z each float16 or float32 (all 8 combinations). If x and
y are both fp16 the sum is an fp16 add (__hadd) converted to z's dtype, otherwise an fp32 add rounded once to z's
dtype: either way bit-identical to the unfused torch expression. Works in place (z is x, or z is y when the sizes
are equal), writes exactly z.numel() == x.numel() elements, 64-bit indexing. Rejects (TORCH_CHECK) y.numel() >
x.numel(), y.numel() not dividing x.numel(), z.numel() != x.numel() and any dtype other than fp16/fp32. An empty x
is a no-op (any y repeats over it zero times); an empty y with a non-empty x is rejected.

Also here: the empty-input contract of the other flat elementwise ops (softcap, xielu, the act_mul family): no-ops;
and the argument checks of softcap and xielu (output dtype/size, flat contiguous buffers, xielu's aligned pairs).

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
    with pytest.raises(RuntimeError, match = "z must match x"):
        ext.add(x, x, torch.empty(3, 64, dtype = torch.half, device = device))
    # Unsupported dtypes raise instead of silently launching nothing
    for args in ((x.bfloat16(), x, torch.empty_like(x)), (x, x.bfloat16(), torch.empty_like(x)),
                 (x, x, torch.empty_like(x, dtype = torch.bfloat16))):
        with pytest.raises(RuntimeError, match = "must be kHalf or kFloat"):
            ext.add(*args)
    assert_device_ok(device)


@torch.inference_mode()
def test_softcap_xielu_rejects(device):
    x = torch.randn(4, 64, device = device).half()
    with pytest.raises(RuntimeError, match = "y must match x's dtype and size"):
        ext.softcap(x, torch.empty(3, 64, dtype = torch.half, device = device), 30.0)
    with pytest.raises(RuntimeError, match = "y must match x's dtype and size"):
        ext.softcap(x, torch.empty_like(x, dtype = torch.float), 30.0)
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.softcap(x.t(), x.t(), 30.0)
    xf = torch.randn(4, 64, device = device)
    alpha = torch.tensor([0.8])
    yh = lambda *s: torch.empty(*s, dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.xielu(xf, yh(3, 64), alpha, alpha)
    with pytest.raises(RuntimeError, match = "must be even"):
        ext.xielu(xf.view(-1)[:255], yh(255), alpha, alpha)
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.xielu(xf.t(), yh(64, 4), alpha, alpha)
    with pytest.raises(RuntimeError, match = "aligned to element pairs"):
        ext.xielu(xf.view(-1)[1:129], yh(128), alpha, alpha)
    assert_device_ok(device)


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


def assert_device_ok(device):
    # A failed launch would leave an error for the next op on the device
    torch.cuda.synchronize(device)
    assert torch.ones(8, device = device).sum().item() == 8


def empty_in_sentinel(shape, dtype, device):
    """(buffer, empty view of shape inside it): a write through the view would land in the sentinel buffer"""
    buf = torch.full((64,), 7.0, dtype = dtype, device = device)
    return buf, buf[8:8].view(shape)


ACT_MUL = ["silu_mul", "silu_oai_mul", "gelu_mul", "relu2_mul", "relu_mul"]


@pytest.mark.parametrize("op", ["add", "add_bias", "softcap_h", "softcap_f", "xielu"] + ACT_MUL)
@pytest.mark.parametrize("shape", [(0, 128), (4, 0)])
@torch.inference_mode()
def test_elementwise_empty(device, op, shape):
    # Elementwise ops on empty input are no-ops: no launch, nothing written. Dtype validation still applies
    h, f = torch.half, torch.float
    if op in ("add", "add_bias"):
        buf, z = empty_in_sentinel(shape, h, device)
        y = torch.empty(shape if op == "add" else shape[-1:], dtype = h, device = device)
        if op == "add_bias" and shape[-1] > 0:
            y = torch.randn(shape[-1], dtype = h, device = device)
        ext.add(torch.empty(shape, dtype = h, device = device), y, z)
    elif op.startswith("softcap"):
        dt = h if op == "softcap_h" else f
        buf, y = empty_in_sentinel(shape, dt, device)
        ext.softcap(torch.empty(shape, dtype = dt, device = device), y, 30.0)
        with pytest.raises(RuntimeError, match = "softcap wrong dtype"):
            ext.softcap(torch.empty(shape, dtype = torch.bfloat16, device = device), y, 30.0)
    elif op == "xielu":
        buf, y = empty_in_sentinel(shape, h, device)
        alpha = torch.tensor([0.8])
        ext.xielu(torch.empty(shape, dtype = f, device = device), y, alpha, alpha)
        with pytest.raises(RuntimeError, match = "xielu not implemented for float16"):
            ext.xielu(torch.empty(shape, dtype = h, device = device), y, alpha, alpha)
    else:
        fn = getattr(ext, op)
        buf, z = empty_in_sentinel(shape, h, device)
        for dt in (h, f):
            x = torch.empty(shape, dtype = dt, device = device)
            fn(x, torch.empty_like(x), z, 7.0)
        with pytest.raises(RuntimeError, match = "incorrect datatype"):
            fn(torch.empty(shape, dtype = h, device = device), torch.empty(shape, dtype = f, device = device), z, 7.0)
    assert (buf == 7.0).all()
    assert_device_ok(device)


@torch.inference_mode()
def test_add_empty_y_rejected(device):
    # An empty y cannot be repeated over a non-empty x
    x = torch.randn(4, 64, device = device).half()
    with pytest.raises(RuntimeError, match = "empty y cannot broadcast"):
        ext.add(x, torch.empty(0, device = device).half(), torch.empty_like(x))
    assert_device_ok(device)
