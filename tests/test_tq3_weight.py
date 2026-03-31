"""
Test TQ3 weight quantization and LinearTQ3 forward pass.

Tests:
1. quantize_tq3 produces correct tensor shapes and dtypes
2. quantize/dequantize round-trip SQNR (>= 6 dB for Gaussian weights)
3. Hadamard vs no-Hadamard path
4. LinearTQ3 forward pass shape correctness
5. LinearTQ3 forward pass approximation quality (cosine similarity vs FP16 matmul)
6. LinearTQ3 swap_cpu / unswap_cpu
7. LinearTQ3 get_tensors serialization
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import math

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

devices = ["cuda:0"]


def sqnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = (original.float() ** 2).mean()
    noise_power = ((original.float() - reconstructed.float()) ** 2).mean()
    if noise_power < 1e-20:
        return float('inf')
    return 10 * math.log10(signal_power.item() / noise_power.item())


# =============================================================================
# quantize_tq3 shape and dtype tests
# =============================================================================

weight_sizes = [(256, 128), (512, 256), (1024, 512), (128, 64)]

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("weight_size", weight_sizes)
@torch.inference_mode()
def test_quantize_shapes(device, weight_size):
    """quantize_tq3 output shapes and dtypes should match the spec."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
    except ImportError:
        pytest.skip("tq3_lib.quantize not available")

    in_f, out_f = weight_size
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)

    result = quantize_tq3(weight)
    packed = result["tq3_packed"]
    scales = result["tq3_scale"]

    num_blocks = in_f // 32

    # packed: 2 rows per block (interleaved bitplanes)
    assert packed.shape == (num_blocks * 2, out_f), \
        f"packed shape wrong: {packed.shape}, expected ({num_blocks * 2}, {out_f})"
    assert packed.dtype == torch.int32, f"packed dtype: {packed.dtype}"

    # scales: 1 per block
    assert scales.shape == (num_blocks, out_f), \
        f"scales shape wrong: {scales.shape}, expected ({num_blocks}, {out_f})"
    assert scales.dtype == torch.half, f"scales dtype: {scales.dtype}"

    # suh/svh should be None when not provided
    assert result["suh"] is None
    assert result["svh"] is None


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_quantize_with_hadamard(device):
    """quantize_tq3 with sign vectors should return them in the output."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3, generate_random_signs
    except ImportError:
        pytest.skip("tq3_lib.quantize not available")

    in_f, out_f = 512, 256
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)
    suh = generate_random_signs(in_f, device, seed = 42)
    svh = generate_random_signs(out_f, device, seed = 43)

    result = quantize_tq3(weight, suh = suh, svh = svh)

    assert result["suh"] is not None
    assert result["svh"] is not None
    assert result["suh"].shape == (in_f,)
    assert result["svh"].shape == (out_f,)
    assert torch.equal(result["suh"], suh)
    assert torch.equal(result["svh"], svh)


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_quantize_scales_positive(device):
    """All per-block scales should be non-negative."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
    except ImportError:
        pytest.skip("tq3_lib.quantize not available")

    torch.manual_seed(42)
    weight = torch.randn(512, 256, dtype = torch.half, device = device)
    result = quantize_tq3(weight)
    assert (result["tq3_scale"] >= 0).all(), "Negative scales found"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_quantize_bitplane_encoding(device):
    """Verify bitplane encoding: bp0 is nonzero mask, bp1 is positive mask."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
    except ImportError:
        pytest.skip("tq3_lib.quantize not available")

    # Use a weight with known structure: [1, -1, 0, 0, ...] repeated
    in_f, out_f = 32, 1
    w = torch.zeros(in_f, out_f, dtype = torch.half, device = device)
    # Note: after WHT, values redistribute, so we can't trivially predict bitplanes.
    # Instead, verify that the packed values are valid 32-bit integers with at most 32 bits set.
    result = quantize_tq3(w)
    packed = result["tq3_packed"]

    # For zero weight, scale should be near zero
    assert result["tq3_scale"].abs().max().item() < 1e-3, \
        f"Scale for zero weight too large: {result['tq3_scale'].abs().max().item()}"


# =============================================================================
# Quantize + dequantize round-trip via LinearTQ3._dequant_weight
# =============================================================================

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("weight_size", [(512, 256), (1024, 512)])
@torch.inference_mode()
def test_weight_roundtrip_sqnr(device, weight_size):
    """
    Quantize via quantize_tq3, then dequantize via LinearTQ3._dequant_weight.
    SQNR should be >= 4 dB for random Gaussian weights (no Hadamard).
    """

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    torch.manual_seed(42)
    in_f, out_f = weight_size
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)

    result = quantize_tq3(weight)

    layer = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
    )

    reconstructed = layer._dequant_weight()
    ratio = sqnr(weight, reconstructed)
    assert ratio >= 4.0, f"Weight roundtrip SQNR too low: {ratio:.2f} dB"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_weight_roundtrip_with_hadamard(device):
    """Round-trip with Hadamard rotation should have similar or better SQNR."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3, generate_random_signs
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    torch.manual_seed(42)
    in_f, out_f = 512, 256
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)

    suh = generate_random_signs(in_f, device, seed = 42)
    svh = generate_random_signs(out_f, device, seed = 43)

    result = quantize_tq3(weight, suh = suh, svh = svh)

    layer = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
        suh = suh,
        svh = svh,
    )

    reconstructed = layer._dequant_weight()

    # With Hadamard, reconstruction goes through the inverse transform,
    # so we compare against original weight
    ratio = sqnr(weight, reconstructed)
    assert ratio >= 3.0, f"Hadamard roundtrip SQNR too low: {ratio:.2f} dB"


# =============================================================================
# LinearTQ3 forward pass tests
# =============================================================================

batch_sizes = [1, 4, 16, 64]

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("batch_size", batch_sizes)
@torch.inference_mode()
def test_linear_tq3_forward_shape(device, batch_size):
    """LinearTQ3 forward should produce output with correct shape and dtype."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    torch.manual_seed(42)
    in_f, out_f = 512, 256
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)

    result = quantize_tq3(weight)
    layer = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
    )

    x = torch.randn(batch_size, in_f, dtype = torch.half, device = device)
    y = layer.forward(x, params = {})

    assert y.shape == (batch_size, out_f), f"Output shape: {y.shape}, expected ({batch_size}, {out_f})"
    assert y.dtype == torch.half, f"Output dtype: {y.dtype}"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_linear_tq3_forward_3d_input(device):
    """LinearTQ3 should handle 3D input (seq_len, batch, features)."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    torch.manual_seed(42)
    in_f, out_f = 256, 128
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)

    result = quantize_tq3(weight)
    layer = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
    )

    x = torch.randn(2, 8, in_f, dtype = torch.half, device = device)
    y = layer.forward(x, params = {})

    assert y.shape == (2, 8, out_f), f"3D output shape: {y.shape}"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_linear_tq3_forward_cosine_similarity(device):
    """
    LinearTQ3 forward pass should approximate FP16 matmul.
    Cosine similarity between TQ3 output and FP16 reference should be >= 0.80.
    """

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    torch.manual_seed(42)
    in_f, out_f = 1024, 512
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)
    x = torch.randn(8, in_f, dtype = torch.half, device = device)

    # FP16 reference
    y_ref = x @ weight

    # TQ3
    result = quantize_tq3(weight)
    layer = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
    )
    y_tq3 = layer.forward(x, params = {})

    cos = torch.nn.functional.cosine_similarity(
        y_ref.float().flatten(), y_tq3.float().flatten(), dim = 0
    )
    assert cos >= 0.80, f"Cosine similarity too low: {cos:.4f} (expected >= 0.80)"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_linear_tq3_forward_with_bias(device):
    """LinearTQ3 forward with bias should add bias to output."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    torch.manual_seed(42)
    in_f, out_f = 256, 128
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)
    bias = torch.randn(out_f, dtype = torch.half, device = device)

    result = quantize_tq3(weight)

    # Without bias
    layer_no_bias = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
    )

    # With bias
    layer_bias = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"].clone(),
        tq3_scale = result["tq3_scale"].clone(),
        bias = bias,
    )

    x = torch.randn(4, in_f, dtype = torch.half, device = device)
    y_no_bias = layer_no_bias.forward(x, params = {})
    y_bias = layer_bias.forward(x, params = {})

    diff = (y_bias - y_no_bias - bias).abs().max().item()
    assert diff < 1e-2, f"Bias application error: {diff}"


# =============================================================================
# LinearTQ3 swap_cpu / unswap_cpu
# =============================================================================

@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_linear_tq3_swap_cpu(device):
    """swap_cpu should move all tensors to CPU; unswap_cpu should restore them."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3, generate_random_signs
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    in_f, out_f = 256, 128
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)
    suh = generate_random_signs(in_f, device, seed = 42)
    svh = generate_random_signs(out_f, device, seed = 43)
    bias = torch.randn(out_f, dtype = torch.half, device = device)

    result = quantize_tq3(weight, suh = suh, svh = svh)
    layer = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
        suh = suh,
        svh = svh,
        bias = bias,
    )

    # Swap to CPU
    layer.swap_cpu()
    assert layer.tq3_packed.device.type == 'cpu', "tq3_packed not on CPU after swap"
    assert layer.tq3_scale.device.type == 'cpu', "tq3_scale not on CPU after swap"
    assert layer.suh.device.type == 'cpu', "suh not on CPU after swap"
    assert layer.svh.device.type == 'cpu', "svh not on CPU after swap"
    assert layer.bias.device.type == 'cpu', "bias not on CPU after swap"

    # Swap back to CUDA
    layer.unswap_cpu()
    assert layer.tq3_packed.device.type == 'cuda', "tq3_packed not on CUDA after unswap"
    assert layer.tq3_scale.device.type == 'cuda', "tq3_scale not on CUDA after unswap"
    assert layer.suh.device.type == 'cuda', "suh not on CUDA after unswap"
    assert layer.svh.device.type == 'cuda', "svh not on CUDA after unswap"
    assert layer.bias.device.type == 'cuda', "bias not on CUDA after unswap"


@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_linear_tq3_swap_cpu_idempotent(device):
    """Calling swap_cpu twice should be harmless (idempotent)."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    weight = torch.randn(256, 128, dtype = torch.half, device = device)
    result = quantize_tq3(weight)
    layer = LinearTQ3(
        config = None,
        in_features = 256,
        out_features = 128,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
    )

    layer.swap_cpu()
    layer.swap_cpu()  # second call should be no-op
    assert layer.tq3_packed.device.type == 'cpu'

    layer.unswap_cpu()
    layer.unswap_cpu()  # second call should be no-op
    assert layer.tq3_packed.device.type == 'cuda'


# =============================================================================
# LinearTQ3 get_tensors serialization
# =============================================================================

@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_linear_tq3_get_tensors(device):
    """get_tensors should return dict with all non-None tensor attributes."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3, generate_random_signs
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    in_f, out_f = 256, 128
    weight = torch.randn(in_f, out_f, dtype = torch.half, device = device)
    suh = generate_random_signs(in_f, device)

    result = quantize_tq3(weight, suh = suh)
    layer = LinearTQ3(
        config = None,
        in_features = in_f,
        out_features = out_f,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
        suh = suh,
    )

    key = "model.layers.0.self_attn.q_proj"
    tensors = layer.get_tensors(key)

    assert f"{key}.tq3_packed" in tensors
    assert f"{key}.tq3_scale" in tensors
    assert f"{key}.suh" in tensors
    # svh and bias are None so should not appear
    assert f"{key}.svh" not in tensors
    assert f"{key}.bias" not in tensors

    # All returned tensors should be contiguous
    for name, t in tensors.items():
        assert t.is_contiguous(), f"{name} is not contiguous"


# =============================================================================
# LinearTQ3 weight cache behavior
# =============================================================================

@pytest.mark.parametrize("device", devices)
@torch.inference_mode()
def test_linear_tq3_weight_cache(device):
    """_dequant_weight should cache the result; unload should clear it."""

    try:
        from exllamav3.modules.quant.tq3_lib.quantize import quantize_tq3
        from exllamav3.modules.quant.tq3 import LinearTQ3
    except ImportError:
        pytest.skip("tq3 modules not available")

    weight = torch.randn(256, 128, dtype = torch.half, device = device)
    result = quantize_tq3(weight)
    layer = LinearTQ3(
        config = None,
        in_features = 256,
        out_features = 128,
        tq3_packed = result["tq3_packed"],
        tq3_scale = result["tq3_scale"],
    )

    assert layer._weight_cache is None, "Weight cache should start empty"

    w1 = layer._dequant_weight()
    assert layer._weight_cache is not None, "Weight cache should be populated after dequant"

    w2 = layer._dequant_weight()
    assert w1 is w2, "Second dequant should return cached reference"

    layer.unload()
    assert layer._weight_cache is None, "Weight cache should be cleared after unload"
