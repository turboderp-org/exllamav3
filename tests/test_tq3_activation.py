"""
Test TQ3 activation compression round-trip accuracy.

Tests:
1. 3D tensor (batch, seq, hidden) round-trip
2. Shape preservation after compress/decompress
3. Padding correctness for non-multiple-of-32 sizes
4. memory_savings calculation
5. SQNR sanity check (>= 8 dB for Gaussian)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import math

devices = ["cuda:0"]


def sqnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = (original.float() ** 2).mean()
    noise_power = ((original.float() - reconstructed.float()) ** 2).mean()
    if noise_power < 1e-20:
        return float('inf')
    return 10 * math.log10(signal_power.item() / noise_power.item())


@pytest.fixture
def compressor():
    try:
        from exllamav3.modules.tq3_activation import TQ3ActivationCompressor
    except ImportError:
        pytest.skip("exllamav3 or CUDA extension not available")
    return TQ3ActivationCompressor


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_3d_roundtrip_shape(compressor, device):
    """Compress/decompress a 3D tensor and verify shape is preserved."""
    x = torch.randn(2, 128, 4096, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    y = compressor.decompress(compressed)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    assert y.dtype == torch.float16


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_3d_roundtrip_sqnr(compressor, device):
    """3D Gaussian round-trip should achieve SQNR >= 8 dB."""
    torch.manual_seed(42)
    x = torch.randn(4, 256, 4096, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    y = compressor.decompress(compressed)
    ratio = sqnr(x, y)
    assert ratio >= 8.0, f"SQNR too low: {ratio:.2f} dB"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_2d_roundtrip(compressor, device):
    """2D tensor round-trip should work and preserve shape."""
    torch.manual_seed(7)
    x = torch.randn(64, 4096, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    y = compressor.decompress(compressed)
    assert y.shape == x.shape
    ratio = sqnr(x, y)
    assert ratio >= 8.0, f"SQNR too low: {ratio:.2f} dB"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_non_multiple_of_32(compressor, device):
    """Tensor with numel not a multiple of 32 should pad correctly."""
    torch.manual_seed(99)
    # 3 * 10 * 100 = 3000 elements, 3000 % 32 = 8, so pad = 24
    x = torch.randn(3, 10, 100, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    assert compressed["pad"] == 24, f"Expected pad=24, got {compressed['pad']}"
    y = compressor.decompress(compressed)
    assert y.shape == x.shape


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_exact_multiple_of_32(compressor, device):
    """Tensor with numel exactly a multiple of 32 should have pad=0."""
    torch.manual_seed(11)
    # 2 * 16 * 32 = 1024, 1024 % 32 = 0
    x = torch.randn(2, 16, 32, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    assert compressed["pad"] == 0
    y = compressor.decompress(compressed)
    assert y.shape == x.shape


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_1d_roundtrip(compressor, device):
    """1D tensor round-trip (the simplest case)."""
    torch.manual_seed(0)
    x = torch.randn(1024, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    y = compressor.decompress(compressed)
    assert y.shape == x.shape
    ratio = sqnr(x, y)
    assert ratio >= 8.0


def test_memory_savings(compressor):
    """memory_savings returns correct ratio for known shape."""
    stats = compressor.memory_savings((2, 128, 4096))
    # 2*128*4096 = 1048576 elements
    # fp16: 1048576 * 2 = 2097152 bytes
    assert stats["fp16_bytes"] == 2097152
    # num_blocks = 1048576 / 32 = 32768
    # tq3_bytes = 32768 * 8 + 32768 * 2 = 327680
    assert stats["tq3_bytes"] == 327680
    assert stats["ratio"] == pytest.approx(6.4, abs=0.01)
    assert stats["savings_pct"] == pytest.approx(84.375, abs=0.01)


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_zeros_roundtrip(compressor, device):
    """Zero tensor should decompress to near-zero."""
    x = torch.zeros(2, 32, 128, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    y = compressor.decompress(compressed)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-3)


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_compressed_dict_keys(compressor, device):
    """Compressed dict should contain exactly the expected keys."""
    x = torch.randn(1, 32, 128, dtype=torch.float16, device=device)
    compressed = compressor.compress(x)
    assert set(compressed.keys()) == {"packed", "scales", "shape", "pad"}
    assert compressed["packed"].dtype == torch.int32
    assert compressed["scales"].dtype == torch.float16
    assert compressed["shape"] == (1, 32, 128)
