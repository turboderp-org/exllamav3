"""
ext.reconstruct_slice (quant/reconstruct.cu): decode the column slice [n_offset, n_offset + w.shape[1]) of a packed
EXL3 trellis into the rotated-basis fp16 weight W_hat (no Hadamards, no scales), as LinearEXL3 does in
MAX_RECONSTRUCT_SLICE_N-wide pieces. Bit-exact against testlib.trellis.dequant_rotated (NumPy bitstream unpack,
codebook decode and tile element order, no extension code), for every integer K and codebook and the half-integer
rates (mul1).

Also: the output is written exactly in place (sentinel rows around it untouched), an empty slice is a no-op, and the
binding rejects slices not aligned to / not a multiple of 128 columns, slices past the packed width, negative
offsets, unsupported bitrates (half-integer rates without mul1, K outside 1..8), a packed tile width that does not
match K, and non-fp16 outputs.
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref

CASES = [(K, cb) for K in range(1, 9) for cb in tref.CODEBOOKS] + [(K, "mul1") for K in (1.5, 2.5, 3.5)]


def _trellis(k, n, K, seed):
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    return p


def _slice(packed_t, k, n_offset, ns, K, cb, device):
    big = torch.full((k + 2, ns), 1234.0, dtype = torch.half, device = device)
    w = big[1:-1]
    ext.reconstruct_slice(w, packed_t, K, cb == "mcg", cb == "mul1", n_offset)
    assert (big[0] == 1234.0).all() and (big[-1] == 1234.0).all(), "wrote outside the output slice"
    return w


@pytest.mark.parametrize("K, cb", CASES)
@torch.inference_mode()
def test_slices_match_reference(device, K, cb):
    k, n = 256, 640
    p = _trellis(k, n, K, int(K * 2) * 7 + len(cb))
    ref = tref.dequant_rotated(p, K, cb).view(np.uint16)
    pt = torch.from_numpy(p.copy()).to(device)
    for n_offset, ns in ((0, n), (0, 128), (128, 256), (512, 128), (256, 384)):
        w = _slice(pt, k, n_offset, ns, K, cb, device)
        assert np.array_equal(w.cpu().numpy().view(np.uint16), ref[:, n_offset : n_offset + ns]), (n_offset, ns)


@pytest.mark.parametrize("k, n", [(16, 128), (48, 384), (4096, 256)])
@torch.inference_mode()
def test_shapes(device, k, n):
    K, cb = 3, "mul1"
    p = _trellis(k, n, K, k + n)
    ref = tref.dequant_rotated(p, K, cb).view(np.uint16)
    pt = torch.from_numpy(p.copy()).to(device)
    w = _slice(pt, k, n - 128, 128, K, cb, device)
    assert np.array_equal(w.cpu().numpy().view(np.uint16), ref[:, n - 128 :])


@torch.inference_mode()
def test_empty_slice(device):
    pt = torch.zeros((4, 16, 48), dtype = torch.int16, device = device)
    ext.reconstruct_slice(torch.empty((64, 0), dtype = torch.half, device = device), pt, 3, False, True, 0)
    torch.cuda.synchronize(device)


@torch.inference_mode()
def test_rejections(device):
    k, n, K = 64, 512, 3
    pt = torch.zeros((k // 16, n // 16, 16 * K), dtype = torch.int16, device = device)
    w128 = torch.empty((k, 128), dtype = torch.half, device = device)

    def rejects(w, packed, K_, mcg, mul1, off, match = None):
        with pytest.raises(RuntimeError, match = match):
            ext.reconstruct_slice(w, packed, K_, mcg, mul1, off)

    rejects(w128, pt, K, False, True, 64, "divisible by 128")
    rejects(torch.empty((k, 192), dtype = torch.half, device = device), pt, K, False, True, 0, "divisible by 128")
    rejects(w128, pt, K, False, True, 512, "exceeds packed tensor bounds")
    rejects(torch.empty((k, 256), dtype = torch.half, device = device), pt, K, False, True, 384, "exceeds")
    rejects(w128, pt, K, False, True, -128, "non-negative")
    rejects(torch.empty((k + 16, 128), dtype = torch.half, device = device), pt, K, False, True, 0)
    rejects(w128, pt, 4, False, True, 0)                            # tile width does not match K
    rejects(torch.empty((k, 128), dtype = torch.float, device = device), pt, K, False, True, 0)
    p25 = torch.zeros((k // 16, n // 16, 40), dtype = torch.int16, device = device)
    rejects(w128, p25, 2.5, False, False, 0, "require the mul1 codebook")
    rejects(w128, p25, 2.5, True, False, 0, "require the mul1 codebook")
    for bad_K in (0, 9, 4.5, 2.25):
        rejects(w128, pt, bad_K, False, True, 0, "Unsupported EXL3 bitrate")
