"""
ext.gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act), the gated RMSNorm of GDN, Mamba2
and KDA (modules/gated_rmsnorm.py). Contract, per row r of the flattened (rows, dim) view:

    w_r  = w.view(w_groups, dim)[r % w_groups] + constant_bias
    act  = silu (gate_act 0) | sigmoid (gate_act 1)
    gate_first = False:  y = x * rsqrt(mean(x^2) + eps) * w_r * act(g)
    gate_first = True:   h = x * act(g);  y = h * rsqrt(mean(h^2) + eps) * w_r

- x bfloat16 only; w float32 or bfloat16; g bfloat16 or float32 with x's shape; y float16 or float32. All arithmetic
  in fp32, one rounding to y's dtype. dim % 4 == 0; dim <= 256 runs a one-warp block, larger dims a 1024-thread
  block with a strided column loop (dim > 4096 gives several float4 columns per thread)
- writes exactly y (memory around y untouched), x / w / g are read only
- rejects (TORCH_CHECK): dim % 4 != 0, w size != dim (w_groups 1) or != w_groups * dim, g shape != x shape, and
  dtype combinations outside the list above

Reference: float64 torch on the same bf16/fp32 inputs.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

GUARD = 64


def guarded_empty(shape, dtype, device):
    """A tensor of `shape` inside a buffer with GUARD sentinel elements on each side (16-byte aligned)"""
    n = 1
    for s in shape:
        n *= s
    buf = torch.full((n + 2 * GUARD,), float("nan"), dtype = dtype, device = device)
    return buf, buf[GUARD:GUARD + n].view(shape)


def assert_guards_intact(buf, n):
    for part in (buf[:GUARD], buf[GUARD + n:]):
        assert torch.isnan(part).all(), "write outside the output tensor"


def reference(x, w, g, eps, constant_bias, w_groups, gate_first, gate_act):
    dim = x.shape[-1]
    xd = x.double().reshape(-1, dim)
    gd = g.double().reshape(-1, dim)
    rows = xd.shape[0]
    wd = w.double().view(w_groups, dim)[torch.arange(rows, device = x.device) % w_groups] + constant_bias
    act = torch.sigmoid(gd) if gate_act == 1 else torch.nn.functional.silu(gd)
    if gate_first:
        h = xd * act
        y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim = True) + eps) * wd
    else:
        y = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim = True) + eps) * wd * act
    return y.view(x.shape)


# fp32 path: fp32 sum of squares (relative error ~dim * 2^-24), rsqrtf / __expf / __fdividef (a few ulp each,
# __expf's error grows with |g| ~ 4 here), then a handful of fp32 products: well under 1e-5 relative. fp16 output
# adds one round-to-nearest (2^-11 relative). Every term is a product, so the error is relative per element;
# atol only covers fp16 subnormals near zero.
TOL = {
    torch.float: dict(rtol = 1e-5, atol = 1e-7),
    torch.half: dict(rtol = 6e-4, atol = 1e-7),
}


def run_case(device, rows, dim, w_dtype, g_dtype, y_dtype, w_groups = 1, gate_first = False, gate_act = 0,
             constant_bias = 0.0, eps = 1e-6, lead_shape = None):
    torch.manual_seed((rows or 0) * 7919 + dim)
    shape = lead_shape + (dim,) if lead_shape else (rows, dim)
    x = (torch.randn(shape, device = device) * 3).to(torch.bfloat16)
    g = (torch.randn(shape, device = device) * 2).to(g_dtype)
    w = (torch.randn(w_groups * dim, device = device) * 0.5 + 1.0).to(w_dtype)
    x0, g0, w0 = x.clone(), g.clone(), w.clone()
    buf, y = guarded_empty(shape, y_dtype, device)
    ext.gated_rms_norm(x, w, y, g, eps, constant_bias, w_groups, gate_first, gate_act)
    ref = reference(x, w, g, eps, constant_bias, w_groups, gate_first, gate_act)
    torch.testing.assert_close(y.double(), ref, **TOL[y_dtype])
    assert_guards_intact(buf, y.numel())
    assert torch.equal(x, x0) and torch.equal(g, g0) and torch.equal(w, w0), "inputs modified"


@pytest.mark.parametrize("rows", [1, 3, 64, 1000])
@pytest.mark.parametrize("dim", [4, 64, 128, 252, 256, 260, 512, 1024, 4096, 8192, 12292])
@pytest.mark.parametrize("y_dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_gated_rms_norm_shapes(device, rows, dim, y_dtype):
    # 256 / 260 straddle the one-warp / 1024-thread switch; 8192 and 12292 loop several columns per thread
    run_case(device, rows, dim, torch.float, torch.bfloat16, y_dtype)


@pytest.mark.parametrize("w_dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("g_dtype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("y_dtype", [torch.half, torch.float])
@pytest.mark.parametrize("dim", [128, 1024])
@torch.inference_mode()
def test_gated_rms_norm_dtypes(device, w_dtype, g_dtype, y_dtype, dim):
    run_case(device, 37, dim, w_dtype, g_dtype, y_dtype)


@pytest.mark.parametrize("gate_first", [False, True])
@pytest.mark.parametrize("gate_act", [0, 1])
@pytest.mark.parametrize("dim", [64, 128, 2048])
@torch.inference_mode()
def test_gated_rms_norm_gate_modes(device, gate_first, gate_act, dim):
    run_case(device, 50, dim, torch.float, torch.bfloat16, torch.half, gate_first = gate_first, gate_act = gate_act)


@pytest.mark.parametrize("w_groups", [2, 8])
@pytest.mark.parametrize("dim", [64, 512])
@pytest.mark.parametrize("gate_first", [False, True])
@torch.inference_mode()
def test_gated_rms_norm_w_groups(device, w_groups, dim, gate_first):
    # Mamba2 group norm: input (tokens, groups, dim), flattened row r uses weight row r % groups
    run_case(device, None, dim, torch.float, torch.bfloat16, torch.half, w_groups = w_groups,
             gate_first = gate_first, lead_shape = (3, 5, w_groups))
    # Rows not a multiple of w_groups: the cycle simply continues
    run_case(device, 7 * w_groups + 3, dim, torch.float, torch.bfloat16, torch.half, w_groups = w_groups,
             gate_first = gate_first)


@pytest.mark.parametrize("constant_bias", [0.0, 1.0, -0.25])
@pytest.mark.parametrize("dim", [128, 4096])
@torch.inference_mode()
def test_gated_rms_norm_constant_bias(device, constant_bias, dim):
    run_case(device, 16, dim, torch.bfloat16, torch.bfloat16, torch.float, constant_bias = constant_bias)


@pytest.mark.parametrize("eps", [1e-6, 1e-2])
@torch.inference_mode()
def test_gated_rms_norm_eps_and_zero_rows(device, eps):
    # A zero row gives exactly 0 (rsqrt(eps) is finite), a near-zero row is dominated by eps
    dim = 256
    x = torch.randn(4, dim, device = device).to(torch.bfloat16)
    x[0] = 0
    x[1] *= 1e-4
    g = torch.randn(4, dim, device = device).to(torch.bfloat16)
    w = torch.rand(dim, device = device) + 0.5
    y = torch.empty(4, dim, dtype = torch.float, device = device)
    ext.gated_rms_norm(x, w, y, g, eps, 0.0, 1, False, 0)
    assert torch.equal(y[0], torch.zeros_like(y[0]))
    torch.testing.assert_close(y.double(), reference(x, w, g, eps, 0.0, 1, False, 0), **TOL[torch.float])


@torch.inference_mode()
def test_gated_rms_norm_deterministic(device):
    x = torch.randn(300, 4096, device = device).to(torch.bfloat16)
    g = torch.randn(300, 4096, device = device).to(torch.bfloat16)
    w = torch.randn(4096, device = device)
    y1 = torch.empty(300, 4096, dtype = torch.half, device = device)
    y2 = torch.empty_like(y1)
    ext.gated_rms_norm(x, w, y1, g, 1e-6, 0.0, 1, False, 0)
    ext.gated_rms_norm(x, w, y2, g, 1e-6, 0.0, 1, False, 0)
    assert torch.equal(y1, y2)


@torch.inference_mode()
def test_gated_rms_norm_rejects(device):
    bf = torch.bfloat16
    x = torch.randn(4, 128, device = device).to(bf)
    g = torch.randn(4, 128, device = device).to(bf)
    w = torch.ones(128, device = device)
    y = torch.empty(4, 128, dtype = torch.half, device = device)
    x6 = torch.randn(4, 130, device = device).to(bf)
    with pytest.raises(RuntimeError, match = "divisible"):
        ext.gated_rms_norm(x6, torch.ones(130, device = device), torch.empty(4, 130, dtype = torch.half, device = device), x6.clone(), 1e-6, 0.0, 1, False, 0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.gated_rms_norm(x, torch.ones(64, device = device), y, g, 1e-6, 0.0, 1, False, 0)
    with pytest.raises(RuntimeError, match = "w_groups"):
        ext.gated_rms_norm(x, torch.ones(3 * 128, device = device), y, g, 1e-6, 0.0, 2, False, 0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.gated_rms_norm(x, w, y, g[:2], 1e-6, 0.0, 1, False, 0)
    with pytest.raises(RuntimeError, match = "Invalid datatypes"):
        ext.gated_rms_norm(x.half(), w, y, g, 1e-6, 0.0, 1, False, 0)
    with pytest.raises(RuntimeError, match = "Invalid datatypes"):
        ext.gated_rms_norm(x, w.half(), y, g, 1e-6, 0.0, 1, False, 0)
    with pytest.raises(RuntimeError, match = "Invalid datatypes"):
        ext.gated_rms_norm(x, w, y.bfloat16(), g, 1e-6, 0.0, 1, False, 0)
    with pytest.raises(RuntimeError, match = "Invalid datatypes"):
        ext.gated_rms_norm(x, w, y, g.half(), 1e-6, 0.0, 1, False, 0)


@pytest.mark.slow
@pytest.mark.vram(24)
@torch.inference_mode()
def test_gated_rms_norm_rows_times_dim_beyond_int32(device):
    # rows * dim > 2^31 elements: row offsets must be 64-bit. Only the tail rows are checked
    dim = 4096
    rows = 2**31 // dim + 64
    x = torch.ones(rows, dim, dtype = torch.bfloat16, device = device)
    g = torch.zeros(rows, dim, dtype = torch.bfloat16, device = device)
    tail = slice(rows - 128, rows)
    x[tail] = torch.randn(128, dim, device = device).to(torch.bfloat16)
    g[tail] = torch.randn(128, dim, device = device).to(torch.bfloat16)
    w = torch.randn(dim, device = device)
    y = torch.zeros(rows, dim, dtype = torch.half, device = device)
    ext.gated_rms_norm(x, w, y, g, 1e-6, 0.0, 1, False, 0)
    torch.testing.assert_close(y[tail].double(), reference(x[tail], w, g[tail], 1e-6, 0.0, 1, False, 0), **TOL[torch.half])
    del x, g, y
    torch.cuda.empty_cache()
