"""
quantize_tiles (the Viterbi trellis quantizer) and ext.decode, for every K and codebook:

- the encoding is a valid tail-biting trellis path (every word's top 16 - K bits are the previous word's low bits,
  wrapping from the last column to the first), and decode(encoding) reproduces the returned tile exactly
- the MSE against the input stays within per-K bounds for Gaussian tiles at the codebook's scale
- a tile that is itself a valid decoded codeword is recovered with zero error
- results are independent of batch composition, launch waves, stream and scratch reuse; nonfinite inputs stay
  memory-safe and an empty scratch is rejected
- the codebook selection is one contract (quant_codebook): {"mcg": True} / {"mul1": True} select, False or absent
  does not, both is rejected; the encoder and the stored marker follow the same selection
"""

import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import codebook_markers, get_temp_buffers, quant_codebook, quantize_tiles

# (codebook id, mcg, mul1, typical scale of the codebook's values)
CODEBOOKS = [(0, False, False, 1.24371088), (1, True, False, 1.24371088), (2, False, True, 1.0)]
CB_IDS = ["3inst", "mcg", "mul1"]

max_mse_per_K = {
    1: 0.3,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    5: 0.1,
    6: 0.07,
    7: 0.05,
    8: 0.04,
}


def args(K, cb):
    return {"K": K, **({"mcg": True} if cb == 1 else {"mul1": True} if cb == 2 else {})}


def assert_tail_biting(idx, K):
    """Every word's top 16 - K bits equal the previous word's low 16 - K bits, the first word wrapping to the last"""
    encoded = idx.int() & 65535
    assert torch.equal(encoded >> K, encoded.roll(1, 1) & ((65536 >> K) - 1))


def ideal_codeword(rows, L, K, device):
    """Random valid tail-biting encoding: word i packs the K-bit branches i, i - 1, ... (circularly), low first"""
    branch = torch.randint(1 << K, (rows, L), device = device)
    encoded = torch.zeros_like(branch)
    for j in range((16 + K - 1) // K):
        encoded |= branch.roll(j, 1) << (K * j)
    return (encoded & 65535).short()


@pytest.mark.parametrize("cb", CODEBOOKS, ids = CB_IDS)
@pytest.mark.parametrize("batch_size", [1, 16, 17, 128])
@pytest.mark.parametrize("K", range(1, 9))
@torch.inference_mode()
def test_encode(device, batch_size, K, cb):
    torch.manual_seed(0)
    cbi, _, _, scale = cb
    in_tile = torch.randn((batch_size, 256), device = device) * scale
    out_tile, out_idx = quantize_tiles(in_tile, args(K, cbi))
    assert_tail_biting(out_idx, K)
    mse = F.mse_loss(in_tile / scale, out_tile / scale).item()
    assert mse < max_mse_per_K[K], f"mse {mse:.4f}"


@pytest.mark.parametrize("cb", CODEBOOKS, ids = CB_IDS)
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("K", range(1, 9))
@torch.inference_mode()
def test_encode_ideal(device, batch_size, K, cb):
    """A decoded valid codeword quantizes back to itself with zero loss"""
    torch.manual_seed(0)
    cbi, mcg, mul1, _ = cb
    encoded = ideal_codeword(batch_size, 256, K, device)
    decoded = torch.empty_like(encoded, dtype = torch.float)
    ext.decode(encoded, decoded, mcg, mul1)
    out_tile, _ = quantize_tiles(decoded, args(K, cbi))
    torch.testing.assert_close(out_tile, decoded, rtol = 1e-6, atol = 1e-6)


@pytest.mark.parametrize("K", range(1, 9))
@pytest.mark.parametrize("cb,L", [(0, 256), (1, 256), (2, 256), (2, 160)])
@torch.inference_mode()
def test_quantize_dispatch(device, K, cb, L):
    torch.manual_seed(12345)
    # Cross a launch wave, including its final partial block batch.
    batch = torch.cuda.get_device_properties(device).multi_processor_count + 3
    x = torch.randn(batch, L, device = device) * (1 if cb == 2 else 1.24371088)
    x[0] = 0
    x[1] = 2
    x[2] = torch.linspace(-4, 4, L, device = device)
    y, idx = quantize_tiles(x, args(K, cb))
    assert_tail_biting(idx, K)
    decoded = torch.empty_like(y)
    ext.decode(idx, decoded, cb == 1, cb == 2)
    assert torch.equal(y, decoded)

    # Scratch reuse and launch-wave boundaries must not affect the solution.
    split_y, split_idx = quantize_tiles(x[-3:], args(K, cb))
    assert torch.equal(idx[-3:], split_idx)
    assert torch.equal(y[-3:], split_y)

    # Recover a known valid circular codeword with zero error.
    ideal = torch.empty((3, L), device = device)
    ext.decode(ideal_codeword(3, L, K, device), ideal, cb == 1, cb == 2)
    ideal_y, _ = quantize_tiles(ideal, args(K, cb))
    assert torch.equal(ideal_y, ideal)


@torch.inference_mode()
def test_quantize_lut_stream_reuse(device):
    if not ext.quantize_tiles_scratch(torch.cuda.current_device(), 6, False, True, 256)[0]:
        pytest.skip("the K = 6 LUT specialization is not dispatched on this device")
    x = torch.randn(3, 256, device = device)
    # Warm scratch on the default stream, then initialize/reuse the LUT on other streams.
    get_temp_buffers(device, 6)
    origin = torch.cuda.current_stream()
    for cb in range(3):
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        first.wait_stream(origin)
        with torch.cuda.stream(first):
            y, idx = quantize_tiles(x, args(6, cb))
        second.wait_stream(first)  # The quantizer's cached scratch is shared across calls.
        with torch.cuda.stream(second):
            other_y, other_idx = quantize_tiles(x, args(6, cb))
        origin.wait_stream(second)
        assert torch.equal(y, other_y)
        assert torch.equal(idx, other_idx)


@torch.inference_mode()
def test_quantize_rejects_empty_history(device):
    x = torch.randn(1, 256, device = device)
    costs, edges = get_temp_buffers(device, 4)
    with pytest.raises(RuntimeError, match = "scratch must hold at least one tile"):
        ext.quantize_tiles(x, torch.empty_like(x), torch.empty_like(x, dtype = torch.short),
                           costs, edges[:0], 4, False, True)


@pytest.mark.parametrize("K", range(1, 9))
@torch.inference_mode()
def test_quantize_nonfinite_stays_in_bounds(device, K):
    x = torch.empty(3, 256, device = device)
    x[0] = float("nan")
    x[1] = float("inf")
    x[2] = -float("inf")
    # No quality guarantee for nonfinite weights, but traceback must remain memory-safe.
    quantize_tiles(x, args(K, 2))
    torch.cuda.synchronize(device)


@pytest.mark.nogpu
@pytest.mark.parametrize("quant_args, expected", [
    ({}, (False, False)),
    ({"mcg": False, "mul1": False}, (False, False)),
    ({"mcg": True}, (True, False)),
    ({"mcg": True, "mul1": False}, (True, False)),
    ({"mul1": True}, (False, True)),
    ({"mcg": False, "mul1": True}, (False, True)),
])
def test_codebook_selection(quant_args, expected):
    assert quant_codebook(quant_args) == expected
    markers = codebook_markers(quant_args)
    assert set(markers) == {name for name, sel in zip(("mcg", "mul1"), expected) if sel}


@pytest.mark.nogpu
def test_codebook_selection_rejects_both():
    with pytest.raises(AssertionError):
        quant_codebook({"mcg": True, "mul1": True})


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("cb", CODEBOOKS, ids = CB_IDS)
@torch.inference_mode()
def test_encoder_follows_selection(device, K, cb):
    """Explicit False flags select exactly what leaving them out does (they used to switch the encoder to the
    flagged codebook while the stored marker said otherwise)"""
    cbi, mcg, mul1, scale = cb
    torch.manual_seed(0)
    tiles = torch.randn((64, 256), device = device) * scale
    _, idx = quantize_tiles(tiles, args(K, cbi))
    _, idx_explicit = quantize_tiles(tiles, {"K": K, "mcg": mcg, "mul1": mul1})
    assert torch.equal(idx, idx_explicit)


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("K", [1, 4, 8])
@torch.inference_mode()
def test_empty_quantize_tiles(device, K):
    """Zero tiles: a no-op that needs no scratch; the dtype/shape checks still apply"""
    x = torch.empty((0, 256), device = device)
    y = torch.empty_like(x)
    idx = torch.empty_like(x, dtype = torch.short)
    costs, edges = get_temp_buffers(device, K, cb = 2)
    ext.quantize_tiles(x, y, idx, costs, edges, K, False, True)
    ext.quantize_tiles(x, y, idx, costs[:0], edges[:0], K, False, True)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.quantize_tiles(x, y, idx.int(), costs, edges, K, False, True)
    with pytest.raises(RuntimeError, match = "length 160 requires the mul1 codebook"):
        x160 = torch.empty((0, 160), device = device)
        ext.quantize_tiles(x160, torch.empty_like(x160), torch.empty_like(x160, dtype = torch.short),
                           costs, edges, K, True, False)
    _device_still_works(device)


@pytest.mark.parametrize("shape", [(0, 256), (3, 0), (0, 0)])
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@torch.inference_mode()
def test_empty_decode(device, shape, dtype):
    """Elementwise: an empty index tensor decodes to nothing"""
    idx = torch.empty(shape, dtype = torch.short, device = device)
    out = torch.empty(shape, dtype = dtype, device = device)
    ext.decode(idx, out, False, True)
    with pytest.raises(RuntimeError, match = "decode: output_tiles must be float or half"):
        ext.decode(idx, out.bfloat16(), False, True)
    _device_still_works(device)
