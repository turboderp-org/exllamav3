"""
Small GEMVs of the GDN/KDA graph paths (BC_GatedDeltaNetSplit), against a float64 torch matmul.

ext.gdn_ba_gemv(x, w_t, bias, y): y[r, j] = sum_i x[r, i] * w_t[j, i] (+ bias[j]), x [.., k] fp16, w_t [n, k] fp16,
    bias [n] fp16 or None, y [.., n] fp32 (rows = x.numel() / k). k must be even, all tensors contiguous; dtypes,
    the w_t shape and y.numel() are validated. Used for the merged b/a projection and the KDA b / f_a / g_a
    projections (x = the layer input, n = 2 * num_v_heads, num_v_heads or a head dim).
ext.gdn_lowrank_gemv_f(x, w_t, y): the same without bias for fp32 x and any k (KDA f_b / g_b second stages,
    k = head dim, n = num_v_heads * head dim).

Both reduce in a fixed order (strided per-lane fp32 FMA chains, then a warp butterfly), so results are
bit-reproducible. Every element of y is written, nothing beyond it.

Tolerance: fp16 x fp16 products are exact in fp32 and fp32 x fp16 products are rounded once per FMA, so the error
is that of fp32 summation: each lane accumulates ceil(k / 64) (fp16 pairs) or ceil(k / 32) (fp32 x) terms and
the warp reduction adds 5 levels, giving |err| <= (terms per lane + 5 + 2) * 2^-24 * sum_i |x_i w_i| (+ bias),
the usual first-order bound on recursive summation.
"""

import math

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

GUARD = 256
SENTINEL = -1232.0
U = 2.0 ** -24


def guarded_out(shape, device):
    n = math.prod(shape)
    buf = torch.full((n + 2 * GUARD,), SENTINEL, dtype = torch.float, device = device)
    return buf, buf[GUARD : GUARD + n].view(shape)


def assert_guards(buf):
    expect = torch.full((GUARD,), SENTINEL, dtype = buf.dtype, device = buf.device)
    assert torch.equal(buf[:GUARD], expect) and torch.equal(buf[-GUARD:], expect), "write outside y"


def gemv_reference(x, w_t, bias, terms_per_lane):
    k = x.shape[-1]
    x2 = x.reshape(-1, k).double()
    w = w_t.double()
    ref = x2 @ w.T
    mag = x2.abs() @ w.abs().T
    if bias is not None:
        ref = ref + bias.double()
        mag = mag + bias.double().abs()
    bound = (terms_per_lane + 7) * U * mag
    return ref, bound


def assert_within(y, ref, bound):
    y2 = y.reshape(ref.shape).double()
    err = (y2 - ref).abs()
    bad = err > bound
    assert not bad.any(), f"{int(bad.sum())} outputs outside the summation bound, max err/bound " \
                          f"{(err / bound.clamp_min(1e-300)).max().item():.2f}"


# (lead shape, k, n): merged b/a (Qwen3.5: n = 2 * 32 or 2 * 64), KDA b (n = 32) / f_a, g_a (n = 128) over a few
# hidden sizes incl. a non-multiple of 64, and n not a multiple of the 8 warps per block
BA_CASES = [
    ((1, 1), 2048, 64),
    ((1, 1), 5120, 128),
    ((4, 3), 4096, 32),
    ((8, 16), 7168, 128),
    ((2, 5), 2050, 13),
    ((1, 1), 2, 1),
    ((33,), 384, 96),
]


@pytest.mark.parametrize("lead, k, n", BA_CASES)
@pytest.mark.parametrize("with_bias", [False, True])
@torch.inference_mode()
def test_gdn_ba_gemv(device, lead, k, n, with_bias):
    torch.manual_seed(0)
    x = torch.randn(*lead, k, dtype = torch.half, device = device)
    w_t = (torch.randn(n, k, device = device) / k ** 0.5).half()
    bias = torch.randn(n, dtype = torch.half, device = device) if with_bias else None
    buf, y = guarded_out((*lead, n), device)
    x_in, w_in = x.clone(), w_t.clone()

    ext.gdn_ba_gemv(x, w_t, bias, y)

    ref, bound = gemv_reference(x, w_t, bias, math.ceil(k / 64))
    assert_within(y, ref, bound)
    assert_guards(buf)
    assert torch.equal(x, x_in) and torch.equal(w_t, w_in), "inputs modified"

    y2 = torch.empty_like(y)
    ext.gdn_ba_gemv(x, w_t, bias, y2)
    assert torch.equal(y, y2), "not bit-reproducible"


@torch.inference_mode()
def test_gdn_ba_gemv_rejects(device):
    x = torch.randn(2, 64, dtype = torch.half, device = device)
    w_t = torch.randn(8, 64, dtype = torch.half, device = device)
    y = torch.empty(2, 8, device = device)
    with pytest.raises(RuntimeError, match = "even"):
        ext.gdn_ba_gemv(x[:, :63].contiguous(), w_t[:, :63].contiguous(), None, y)
    with pytest.raises(RuntimeError, match = "w_t must be"):
        ext.gdn_ba_gemv(x, w_t[:, :32].contiguous(), None, y)
    with pytest.raises(RuntimeError, match = "y must be"):
        ext.gdn_ba_gemv(x, w_t, None, y[:, :7].contiguous())
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.gdn_ba_gemv(torch.randn(64, 2, dtype = torch.half, device = device).T, w_t, None, y)
    with pytest.raises(RuntimeError):
        ext.gdn_ba_gemv(x.float(), w_t, None, y)
    with pytest.raises(RuntimeError):
        ext.gdn_ba_gemv(x, w_t, torch.zeros(8, device = device), y)
    with pytest.raises(RuntimeError):
        ext.gdn_ba_gemv(x, w_t, None, y.half())


# KDA f_b / g_b: k = head dim (128), n = num_v_heads * head dim (TP shards included); odd k is allowed here
LR_CASES = [
    ((1, 1), 128, 4096),
    ((8, 16), 128, 4096),
    ((3, 2), 128, 1024),
    ((2, 1), 77, 50),
    ((5,), 1, 9),
]


@pytest.mark.parametrize("lead, k, n", LR_CASES)
@torch.inference_mode()
def test_gdn_lowrank_gemv_f(device, lead, k, n):
    torch.manual_seed(1)
    x = torch.randn(*lead, k, dtype = torch.float, device = device)
    w_t = (torch.randn(n, k, device = device) / k ** 0.5).half()
    buf, y = guarded_out((*lead, n), device)
    x_in = x.clone()

    ext.gdn_lowrank_gemv_f(x, w_t, y)

    # fp32 x fp16 products are rounded in the FMA, so each term counts once more than in the fp16 GEMV
    ref, bound = gemv_reference(x, w_t, None, 2 * math.ceil(k / 32))
    assert_within(y, ref, bound)
    assert_guards(buf)
    assert torch.equal(x, x_in), "input modified"

    y2 = torch.empty_like(y)
    ext.gdn_lowrank_gemv_f(x, w_t, y2)
    assert torch.equal(y, y2), "not bit-reproducible"


@torch.inference_mode()
def test_gdn_lowrank_gemv_f_rejects(device):
    x = torch.randn(2, 64, device = device)
    w_t = torch.randn(8, 64, dtype = torch.half, device = device)
    y = torch.empty(2, 8, device = device)
    with pytest.raises(RuntimeError, match = "w_t must be"):
        ext.gdn_lowrank_gemv_f(x, w_t[:, :32].contiguous(), y)
    with pytest.raises(RuntimeError, match = "y must be"):
        ext.gdn_lowrank_gemv_f(x, w_t, y[:, :7].contiguous())
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.gdn_lowrank_gemv_f(torch.randn(64, 2, device = device).T, w_t, y)
    with pytest.raises(RuntimeError):
        ext.gdn_lowrank_gemv_f(x.half(), w_t, y)
    with pytest.raises(RuntimeError):
        ext.gdn_lowrank_gemv_f(x, w_t.float(), y)
