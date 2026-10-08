"""
ext.test_distribution (quant/quantize.cu): a normalized histogram of a float tensor over [min_value, max_value), and
optionally of the selected codebook's 65536 decoded values, for comparing a weight distribution to the codebook.

Contract: with B = dist_output.numel() bins, value v falls in bin clamp(trunc((v - min) / (max - min) * B), 0, B - 1)
(so values below min, above max, exactly max and +-inf land in the edge bins), and dist_output[b] = count[b] / numel;
ref_output (when given) is the same histogram of decode(i) for i in 0..65535 under the mcg / mul1 / 3INST codebook,
divided by 65536. B <= 1024; ref_output must have B elements; input must be fp32.

Reference: the binning in float64 on the CPU over testlib.trellis.decode for the codebook values. The extension
builds with --use_fast_math, so its fp32 divisions are approximate (a few ulp): values within 1e-4 of a bin edge may
land on either side, and the normalized counts are compared at a few ulp. Bin counts are otherwise exact.
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref


def _check_hist(dist: torch.Tensor, v: torch.Tensor, bins: int, lo: float, hi: float):
    """dist (the kernel's normalized histogram of v) against the float64 binning, edge-ambiguous values allowed in
    either neighbouring bin"""
    dist = dist.cpu().double()
    v = v.cpu().double().flatten()
    n = v.numel()
    counts = dist * n
    # count / numel with an approximate fp32 division: a few ulp of relative error (6 ulp allowed, < 0.5 counts here)
    assert ((counts - counts.round()).abs() <= 6 * 2.0 ** -24 * counts.clamp(min = 1)).all(), "not count / numel"
    counts = counts.round().long()
    assert counts.sum().item() == n
    t = ((v - lo) / (hi - lo) * bins).clamp(-1.0, float(bins))
    b_lo = (t - 1e-4).floor().long().clamp(0, bins - 1)
    b_hi = (t + 1e-4).floor().long().clamp(0, bins - 1)
    certain = torch.bincount(b_lo[b_lo == b_hi], minlength = bins)
    amb = b_lo != b_hi
    touching = torch.bincount(b_lo[amb], minlength = bins) + torch.bincount(b_hi[amb], minlength = bins)
    assert ((counts >= certain) & (counts <= certain + touching)).all(), "bin counts differ from the reference"


@pytest.mark.parametrize("numel", [1, 1000, (1 << 20) + 3])
@pytest.mark.parametrize("bins", [1, 7, 64, 1024])
@torch.inference_mode()
def test_input_histogram(device, numel, bins):
    torch.manual_seed(numel + bins)
    x = torch.randn(numel) * 1.5
    if numel >= 8:
        x[:6] = torch.tensor([-100.0, 100.0, -3.8, 3.8, float("inf"), -float("inf")])
    dist = torch.full((bins,), -1.0, device = device)
    ext.test_distribution(x.to(device), dist, None, -3.8, 3.8, False, True)
    _check_hist(dist, x, bins, -3.8, 3.8)


@pytest.mark.parametrize("codebook", tref.CODEBOOKS)
@pytest.mark.parametrize("bins", [16, 100, 1024])
@torch.inference_mode()
def test_codebook_histogram(device, codebook, bins):
    mcg, mul1 = codebook == "mcg", codebook == "mul1"
    x = torch.randn(4096, device = device)
    dist = torch.empty((bins,), device = device)
    ref = torch.full((bins,), -1.0, device = device)
    ext.test_distribution(x, dist, ref, -3.8, 3.8, mcg, mul1)
    values = torch.from_numpy(tref.decode(np.arange(65536), codebook).astype(np.float64))
    _check_hist(ref, values, bins, -3.8, 3.8)
    _check_hist(dist, x, bins, -3.8, 3.8)


@torch.inference_mode()
def test_rejections(device):
    x = torch.randn(100, device = device)
    with pytest.raises(RuntimeError, match = "Too many bins"):
        ext.test_distribution(x, torch.empty(1025, device = device), None, -1.0, 1.0, False, False)
    with pytest.raises(RuntimeError):
        ext.test_distribution(x, torch.empty(64, device = device), torch.empty(63, device = device), -1.0, 1.0, False, False)
    with pytest.raises(RuntimeError):
        ext.test_distribution(x.half(), torch.empty(64, device = device), None, -1.0, 1.0, False, False)


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("numel, bins, match", [
    (0, 64, "empty input"),         # normalized histogram of nothing: 0 / 0
    (100, 0, "no bins"),
    (0, 0, "no bins"),
])
@torch.inference_mode()
def test_empty_distribution(device, numel, bins, match):
    x = torch.randn(numel, device = device)
    dist = torch.full((bins,), -1.0, device = device)
    ref = torch.full((bins,), -1.0, device = device)
    with pytest.raises(RuntimeError, match = f"test_distribution: .*{match}"):
        ext.test_distribution(x, dist, ref, -1.0, 1.0, False, True)
    assert (dist == -1.0).all() and (ref == -1.0).all()
    _device_still_works(device)


@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@torch.inference_mode()
def test_empty_histogram(device, dtype):
    """ext.histogram of an empty tensor counts nothing (all bins zero); a histogram without bins is rejected"""
    out = torch.full((16,), 7, dtype = torch.long, device = device)
    ext.histogram(torch.empty((0, 5), dtype = dtype, device = device), out, -1.0, 1.0, False)
    assert (out == 0).all()
    for numel in (0, 10):
        with pytest.raises(RuntimeError, match = "histogram: empty output"):
            ext.histogram(torch.zeros(numel, dtype = dtype, device = device),
                          torch.empty(0, dtype = torch.long, device = device), -1.0, 1.0, True)
    _device_still_works(device)
