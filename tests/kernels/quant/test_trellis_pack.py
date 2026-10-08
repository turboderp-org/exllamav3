"""
Trellis packing (quant/pack.cu, quant/frac.cu) and sign packing, bit-exact against the format definition in
testlib.trellis (NumPy bitstream reference, no extension code):

- unpack_trellis / unpack_trellis_frac: every 16 K-word packed tile expands to its 256 ring windows, window i being
  ring bits [S(i) - 16, S(i)) MSB first (wrapping); the result is a tail-biting sequence for any input bits
- pack_trellis / pack_trellis_frac: the inverse, storing only the low D(i) bits of each window; so
  pack(unpack(p)) == p for every packed tile and unpack(pack(w)) == w for every tail-biting w
- the half-integer layout with MASK = 0 is the integer layout ("same stream convention", frac.cu)
- outputs are written exactly in place: memory around the output tensor is untouched
- pack_trellis_frac / unpack_trellis_frac reject KA outside 1..7, MASK beyond 16 bits, an odd bit count per 16
  positions and mismatched shapes; the integer forms reject mismatched shapes
- pack_signs: bit b of packed[j] is the fp16 sign bit of x[16 j + b] (so -0.0 and negative NaN count as negative),
  for any column count (the kernel's 32-column blocks have a partial tail); dtypes are validated
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref

INT_K = list(range(1, 9))
HALF_K = [1.5, 2.5, 3.5]
SHAPES = [(1, 1), (3, 5), (17, 33)]


def _rng(*seed):
    return np.random.default_rng(list(seed))


def _guarded(shape, dtype, device, fill = 0x5A5A):
    """A contiguous view of `shape` inside a sentinel-filled buffer with one extra leading and trailing slab"""
    big = torch.full((shape[0] + 2,) + tuple(shape[1:]), fill, dtype = torch.int32).to(dtype).to(device)
    return big, big[1:-1]


def _assert_guard(big, fill = 0x5A5A):
    s = torch.full_like(big[0], fill, dtype = torch.int32).to(big.dtype)
    assert torch.equal(big[0], s) and torch.equal(big[-1], s), "wrote outside the output tensor"


def _unpack(packed, K, device):
    tk, tn, _ = packed.shape
    big, out = _guarded((tk, tn, 256), torch.int16, device)
    if float(K).is_integer():
        ext.unpack_trellis(out, packed, int(K))
    else:
        ext.unpack_trellis_frac(out, packed, *tref.frac(K))
    _assert_guard(big)
    return out


def _pack(words, K, device):
    tk, tn, _ = words.shape
    big, out = _guarded((tk, tn, int(16 * K)), torch.int16, device)
    if float(K).is_integer():
        ext.pack_trellis(out, words, int(K))
    else:
        ext.pack_trellis_frac(out, words, *tref.frac(K))
    _assert_guard(big)
    return out


def _np16(t):
    return t.cpu().numpy().view(np.uint16)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("K", INT_K + HALF_K)
@torch.inference_mode()
def test_unpack_matches_layout(device, K, shape):
    p = tref.random_packed(shape[0] * shape[1], K, _rng(int(K * 2), *shape)).reshape(*shape, -1)
    out = _unpack(torch.from_numpy(p.copy()).to(device), K, device)
    ref = tref.unpack(p, K)
    assert np.array_equal(_np16(out), ref)
    assert tref.is_tail_biting(ref, K)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("K", INT_K + HALF_K)
@torch.inference_mode()
def test_pack_matches_layout(device, K, shape):
    """Valid (tail-biting) windows pack to the reference stream and unpack back to themselves"""
    p = tref.random_packed(shape[0] * shape[1], K, _rng(int(K * 2), *shape, 1)).reshape(*shape, -1)
    words = tref.unpack(p, K)
    packed = _pack(torch.from_numpy(words.view(np.int16).copy()).to(device), K, device)
    assert np.array_equal(packed.cpu().numpy(), p)
    assert np.array_equal(_np16(_unpack(packed, K, device)), words)


@pytest.mark.parametrize("K", INT_K + HALF_K)
@torch.inference_mode()
def test_pack_uses_low_bits_only(device, K):
    """Arbitrary words (not tail-biting): each window contributes exactly its low D(i) bits"""
    words = _rng(int(K * 2), 9).integers(0, 1 << 16, size = (4, 6, 256), dtype = np.uint16)
    packed = _pack(torch.from_numpy(words.view(np.int16).copy()).to(device), K, device)
    assert np.array_equal(packed.cpu().numpy(), tref.pack(words, K))


@pytest.mark.parametrize("KA", range(1, 8))
@torch.inference_mode()
def test_frac_mask0_is_integer_layout(device, KA):
    words = torch.from_numpy(_rng(KA, 3).integers(0, 1 << 16, size = (3, 4, 256), dtype = np.uint16)
                             .view(np.int16).copy()).to(device)
    a = torch.zeros((3, 4, 16 * KA), dtype = torch.int16, device = device)
    b = torch.zeros_like(a)
    ext.pack_trellis(a, words, KA)
    ext.pack_trellis_frac(b, words, KA, 0)
    assert torch.equal(a, b)
    ua = torch.zeros_like(words)
    ub = torch.zeros_like(words)
    ext.unpack_trellis(ua, a, KA)
    ext.unpack_trellis_frac(ub, a, KA, 0)
    assert torch.equal(ua, ub)


@pytest.mark.parametrize("KA, MASK, msg", [
    (0, 0xAAAA, "KA must be 1..7"),
    (8, 0, "KA must be 1..7"),
    (2, 0x10000, "MASK 16 bits"),
    (2, -1, "MASK 16 bits"),
    (2, 0x0001, "must be even"),
])
@torch.inference_mode()
def test_frac_rejects_bad_rate(device, KA, MASK, msg):
    words = torch.zeros((1, 1, 256), dtype = torch.int16, device = device)
    packed = torch.zeros((1, 1, 64), dtype = torch.int16, device = device)
    with pytest.raises(RuntimeError, match = msg):
        ext.pack_trellis_frac(packed, words, KA, MASK)
    with pytest.raises(RuntimeError, match = msg):
        ext.unpack_trellis_frac(words, packed, KA, MASK)


@torch.inference_mode()
def test_rejects_shape_mismatch(device):
    words = torch.zeros((2, 3, 256), dtype = torch.int16, device = device)
    for K in (3, 2.5):
        w16 = int(16 * K)
        bad = [torch.zeros((2, 3, w16 + 2), dtype = torch.int16, device = device),
               torch.zeros((2, 4, w16), dtype = torch.int16, device = device),
               torch.zeros((1, 3, w16), dtype = torch.int16, device = device)]
        for packed in bad:
            with pytest.raises(RuntimeError):
                if float(K).is_integer():
                    ext.pack_trellis(packed, words, int(K))
                else:
                    ext.pack_trellis_frac(packed, words, *tref.frac(K))
            with pytest.raises(RuntimeError):
                if float(K).is_integer():
                    ext.unpack_trellis(words, packed, int(K))
                else:
                    ext.unpack_trellis_frac(words, packed, *tref.frac(K))


@pytest.mark.parametrize("K", HALF_K)
@torch.inference_mode()
def test_frac_rejects_noncontiguous(device, K):
    w16 = int(16 * K)
    words = torch.zeros((3, 2, 256), dtype = torch.int16, device = device).transpose(0, 1)
    packed = torch.zeros((2, 3, w16), dtype = torch.int16, device = device)
    with pytest.raises(RuntimeError, match = "contiguous"):
        ext.pack_trellis_frac(packed, words, *tref.frac(K))


@torch.inference_mode()
def test_frac_empty_is_noop(device):
    words = torch.zeros((0, 4, 256), dtype = torch.int16, device = device)
    packed = torch.zeros((0, 4, 40), dtype = torch.int16, device = device)
    ext.pack_trellis_frac(packed, words, 2, 0xAAAA)
    ext.unpack_trellis_frac(words, packed, 2, 0xAAAA)
    torch.cuda.synchronize(device)


def _sign_ref(x: torch.Tensor) -> np.ndarray:
    bits = (x.cpu().view(torch.int16).numpy().view(np.uint16) >> 15).astype(np.uint32).reshape(-1, 16)
    return (bits << np.arange(16, dtype = np.uint32)).sum(axis = 1).astype(np.uint16)


@pytest.mark.parametrize("cols", [1, 31, 32, 33, 1000, 4096])
@torch.inference_mode()
def test_pack_signs(device, cols):
    torch.manual_seed(cols)
    x = torch.randn(cols * 16).half()
    # Signed zeros, infinities and NaNs: the contract is the raw sign bit
    special = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), -float("nan")]).half()
    x[: min(len(special), x.numel())] = special[: x.numel()]
    x = x.to(device)
    buf = torch.full((cols + 2,), 0x1234, dtype = torch.int16, device = device)
    ext.pack_signs(buf[1:-1], x)
    assert buf[0].item() == 0x1234 and buf[-1].item() == 0x1234
    assert np.array_equal(buf[1:-1].cpu().numpy().view(np.uint16), _sign_ref(x))


@torch.inference_mode()
def test_pack_signs_rejects_dtypes(device):
    with pytest.raises(RuntimeError):
        ext.pack_signs(torch.zeros(2, dtype = torch.int16, device = device), torch.zeros(32, device = device))
    with pytest.raises(RuntimeError):
        ext.pack_signs(torch.zeros(2, dtype = torch.int32, device = device),
                       torch.zeros(32, dtype = torch.half, device = device))
