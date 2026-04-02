"""
Tests for LinearTQInstant — instant WHT + Lloyd-Max weight quantization.

Run with:
    python -m pytest tests/test_tq_instant.py -v
or:
    python tests/test_tq_instant.py
"""
from __future__ import annotations
import sys
import os
import math

import torch
import pytest

# Allow running from repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from exllamav3.modules.quant.tq_instant import LinearTQInstant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fp16_weight(in_f: int, out_f: int, seed: int = 42) -> torch.Tensor:
    """Return a random (in_f, out_f) fp16 weight tensor on CPU."""
    torch.manual_seed(seed)
    return torch.randn(in_f, out_f, dtype=torch.float16)


def make_layer(in_f: int = 128, out_f: int = 256, bits: int = 4, seed: int = 42) -> LinearTQInstant:
    w = make_fp16_weight(in_f, out_f, seed=seed)
    return LinearTQInstant(
        config=None,
        in_features=in_f,
        out_features=out_f,
        weight_fp16=w,
        bits=bits,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_construction(self):
        layer = make_layer()
        assert layer.in_features == 128
        assert layer.out_features == 256
        assert layer.bits == 4

    def test_quant_type(self):
        layer = make_layer()
        assert layer.quant_type == "tq_instant"

    def test_packed_shape(self):
        in_f, out_f, bits = 128, 256, 4
        layer = make_layer(in_f, out_f, bits)
        num_blocks = in_f // 32
        # packed has shape (num_blocks * bits, out_features)
        assert layer.packed.shape == (num_blocks * bits, out_f)
        assert layer.packed.dtype == torch.int32

    def test_scales_shape(self):
        in_f, out_f, sub_size = 128, 256, 8
        layer = make_layer(in_f, out_f)
        num_blocks = in_f // 32
        num_subs = 32 // sub_size
        assert layer.scales.shape == (num_blocks, num_subs, out_f)
        assert layer.scales.dtype == torch.float16

    def test_suh_shape(self):
        in_f = 128
        layer = make_layer(in_f)
        assert layer.suh.shape == (in_f,)
        assert layer.suh.dtype == torch.float16

    def test_suh_values_are_pm1(self):
        layer = make_layer()
        vals = layer.suh.float().abs()
        assert torch.allclose(vals, torch.ones_like(vals))

    def test_weight_cache_starts_none(self):
        layer = make_layer()
        assert layer._weight_cache is None

    def test_construction_with_bias(self):
        in_f, out_f = 128, 256
        w = make_fp16_weight(in_f, out_f)
        bias = torch.randn(out_f, dtype=torch.float16)
        layer = LinearTQInstant(None, in_f, out_f, w, bias=bias)
        assert layer.bias is not None
        assert layer.bias.shape == (out_f,)

    def test_construction_without_bias(self):
        layer = make_layer()
        assert layer.bias is None

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_bits_2_3_4(self, bits):
        layer = make_layer(bits=bits)
        assert layer.bits == bits
        assert layer.packed.shape[0] == (128 // 32) * bits

    def test_key_stored(self):
        w = make_fp16_weight(128, 256)
        layer = LinearTQInstant(None, 128, 256, w, key="model.layer0.weight")
        assert layer.key == "model.layer0.weight"

    def test_deterministic_suh(self):
        """Same shape always produces the same sign vector."""
        w1 = make_fp16_weight(128, 256, seed=1)
        w2 = make_fp16_weight(128, 256, seed=99)
        l1 = LinearTQInstant(None, 128, 256, w1)
        l2 = LinearTQInstant(None, 128, 256, w2)
        assert torch.equal(l1.suh, l2.suh)


# ---------------------------------------------------------------------------
# forward()
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape_2d(self):
        in_f, out_f = 128, 256
        layer = make_layer(in_f, out_f)
        x = torch.randn(4, in_f, dtype=torch.float16)
        y = layer.forward(x, {})
        assert y.shape == (4, out_f)

    def test_output_shape_3d(self):
        in_f, out_f = 128, 256
        layer = make_layer(in_f, out_f)
        x = torch.randn(2, 5, in_f, dtype=torch.float16)
        y = layer.forward(x, {})
        assert y.shape == (2, 5, out_f)

    def test_output_dtype_default_half(self):
        layer = make_layer()
        x = torch.randn(3, 128, dtype=torch.float16)
        y = layer.forward(x, {})
        assert y.dtype == torch.float16

    def test_output_dtype_override(self):
        layer = make_layer()
        x = torch.randn(3, 128, dtype=torch.float16)
        y = layer.forward(x, {}, out_dtype=torch.float32)
        assert y.dtype == torch.float32

    def test_bias_applied(self):
        in_f, out_f = 128, 64
        w = make_fp16_weight(in_f, out_f)
        bias = torch.ones(out_f, dtype=torch.float16)
        layer = LinearTQInstant(None, in_f, out_f, w, bias=bias)
        x = torch.zeros(1, in_f, dtype=torch.float16)
        y = layer.forward(x, {})
        # zero input -> output should be ~bias (dequant(0 codes) * scales + bias)
        # Just check bias contribution is present (output != 0 when bias=1)
        assert not torch.all(y == 0)

    def test_cosine_similarity_4bit(self):
        """Dequantized matmul should be close to FP16 matmul (cosine sim > 0.85)."""
        in_f, out_f = 256, 512
        torch.manual_seed(7)
        w = torch.randn(in_f, out_f, dtype=torch.float16)
        layer = LinearTQInstant(None, in_f, out_f, w, bits=4)

        torch.manual_seed(13)
        x = torch.randn(8, in_f, dtype=torch.float16)

        y_tq = layer.forward(x, {}).float()
        y_fp16 = torch.matmul(x.float(), w.float())

        # Per-row cosine similarity
        cos = torch.nn.functional.cosine_similarity(y_tq, y_fp16, dim=1)
        mean_cos = cos.mean().item()
        assert mean_cos > 0.85, f"Cosine similarity {mean_cos:.4f} is below threshold 0.85"

    def test_cosine_similarity_3bit(self):
        """3-bit should still achieve reasonable similarity."""
        in_f, out_f = 256, 512
        torch.manual_seed(7)
        w = torch.randn(in_f, out_f, dtype=torch.float16)
        layer = LinearTQInstant(None, in_f, out_f, w, bits=3)

        torch.manual_seed(13)
        x = torch.randn(8, in_f, dtype=torch.float16)

        y_tq = layer.forward(x, {}).float()
        y_fp16 = torch.matmul(x.float(), w.float())

        cos = torch.nn.functional.cosine_similarity(y_tq, y_fp16, dim=1)
        mean_cos = cos.mean().item()
        assert mean_cos > 0.70, f"3-bit cosine similarity {mean_cos:.4f} is below threshold 0.70"

    def test_weight_cache_populated_after_forward(self):
        layer = make_layer()
        assert layer._weight_cache is None
        x = torch.randn(2, 128, dtype=torch.float16)
        layer.forward(x, {})
        assert layer._weight_cache is not None

    def test_forward_uses_cache(self):
        """Second forward should reuse cached weight (cache object identity)."""
        layer = make_layer()
        x = torch.randn(2, 128, dtype=torch.float16)
        layer.forward(x, {})
        cache_ref = layer._weight_cache
        layer.forward(x, {})
        assert layer._weight_cache is cache_ref


# ---------------------------------------------------------------------------
# CPU swap
# ---------------------------------------------------------------------------

class TestSwapCpu:
    def test_swap_moves_tensors_to_cpu(self):
        layer = make_layer()
        # Already on CPU in test environment — just verify swap_device is set
        layer.swap_device = torch.device("cpu")  # simulate being on a device
        layer.packed = layer.packed.cpu()
        layer.scales = layer.scales.cpu()
        layer.suh = layer.suh.cpu()
        # Reset and test the method itself
        layer.swap_device = None
        layer.swap_cpu()
        assert layer.swap_device is not None
        assert layer.packed.device.type == "cpu"
        assert layer.scales.device.type == "cpu"
        assert layer.suh.device.type == "cpu"

    def test_swap_clears_weight_cache(self):
        layer = make_layer()
        x = torch.randn(2, 128, dtype=torch.float16)
        layer.forward(x, {})
        assert layer._weight_cache is not None
        layer.swap_cpu()
        assert layer._weight_cache is None

    def test_swap_is_idempotent(self):
        layer = make_layer()
        layer.swap_cpu()
        dev = layer.swap_device
        layer.swap_cpu()  # second call should be no-op
        assert layer.swap_device == dev

    def test_unswap_clears_swap_device(self):
        layer = make_layer()
        layer.swap_cpu()
        assert layer.swap_device is not None
        layer.unswap_cpu()
        assert layer.swap_device is None

    def test_unswap_is_idempotent(self):
        layer = make_layer()
        layer.unswap_cpu()  # no-op when not swapped
        assert layer.swap_device is None

    def test_forward_after_swap_unswap(self):
        layer = make_layer()
        x = torch.randn(2, 128, dtype=torch.float16)
        layer.swap_cpu()
        layer.unswap_cpu()
        y = layer.forward(x, {})
        assert y.shape == (2, 256)


# ---------------------------------------------------------------------------
# get_tensors()
# ---------------------------------------------------------------------------

class TestGetTensors:
    def test_keys_present(self):
        layer = make_layer()
        t = layer.get_tensors("model.test")
        assert "model.test.tq_packed" in t
        assert "model.test.tq_scales" in t
        assert "model.test.suh" in t

    def test_bias_key_when_bias_present(self):
        in_f, out_f = 128, 256
        w = make_fp16_weight(in_f, out_f)
        bias = torch.randn(out_f, dtype=torch.float16)
        layer = LinearTQInstant(None, in_f, out_f, w, bias=bias)
        t = layer.get_tensors("x")
        assert "x.bias" in t

    def test_bias_key_absent_when_no_bias(self):
        layer = make_layer()
        t = layer.get_tensors("x")
        assert "x.bias" not in t

    def test_tensors_are_contiguous(self):
        layer = make_layer()
        t = layer.get_tensors("x")
        for name, tensor in t.items():
            assert tensor.is_contiguous(), f"{name} is not contiguous"

    def test_packed_tensor_dtype(self):
        layer = make_layer()
        t = layer.get_tensors("x")
        assert t["x.tq_packed"].dtype == torch.int32

    def test_scales_tensor_dtype(self):
        layer = make_layer()
        t = layer.get_tensors("x")
        assert t["x.tq_scales"].dtype == torch.float16

    def test_suh_tensor_dtype(self):
        layer = make_layer()
        t = layer.get_tensors("x")
        assert t["x.suh"].dtype == torch.float16


# ---------------------------------------------------------------------------
# unload()
# ---------------------------------------------------------------------------

class TestUnload:
    def test_unload_clears_cache(self):
        layer = make_layer()
        x = torch.randn(2, 128, dtype=torch.float16)
        layer.forward(x, {})
        assert layer._weight_cache is not None
        layer.unload()
        assert layer._weight_cache is None

    def test_forward_still_works_after_unload(self):
        layer = make_layer()
        x = torch.randn(2, 128, dtype=torch.float16)
        layer.forward(x, {})
        layer.unload()
        y = layer.forward(x, {})
        assert y.shape == (2, 256)


# ---------------------------------------------------------------------------
# get_weight_tensor / get_bias_tensor
# ---------------------------------------------------------------------------

class TestAccessors:
    def test_get_weight_tensor_shape(self):
        layer = make_layer(128, 256)
        w = layer.get_weight_tensor()
        assert w.shape == (128, 256)
        assert w.dtype == torch.float16

    def test_get_bias_tensor_none(self):
        layer = make_layer()
        assert layer.get_bias_tensor() is None

    def test_get_bias_tensor_value(self):
        in_f, out_f = 128, 64
        w = make_fp16_weight(in_f, out_f)
        bias = torch.arange(out_f, dtype=torch.float16)
        layer = LinearTQInstant(None, in_f, out_f, w, bias=bias)
        b = layer.get_bias_tensor()
        assert b is not None
        assert b.shape == (out_f,)


# ---------------------------------------------------------------------------
# Codebook
# ---------------------------------------------------------------------------

class TestCodebook:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_codebook_lengths(self, bits):
        boundaries, centroids = LinearTQInstant._get_codebook(bits)
        n = 1 << bits
        assert len(centroids) == n
        assert len(boundaries) == n - 1

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_boundaries_sorted(self, bits):
        boundaries, _ = LinearTQInstant._get_codebook(bits)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i + 1]

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_centroids_sorted(self, bits):
        _, centroids = LinearTQInstant._get_codebook(bits)
        for i in range(len(centroids) - 1):
            assert centroids[i] < centroids[i + 1]

    def test_fallback_bits(self):
        boundaries, centroids = LinearTQInstant._get_codebook(5)
        assert len(centroids) == 32
        assert len(boundaries) == 31


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [
        TestConstruction,
        TestForward,
        TestSwapCpu,
        TestGetTensors,
        TestUnload,
        TestAccessors,
        TestCodebook,
    ]

    passed = 0
    failed = 0
    errors = []

    for suite_cls in suites:
        suite = suite_cls()
        for name in dir(suite_cls):
            if not name.startswith("test_"):
                continue
            method = getattr(suite, name)
            # Handle parametrize manually for __main__ path — skip, pytest handles it
            if hasattr(method, "pytestmark"):
                continue
            try:
                method()
                print(f"  PASS  {suite_cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {suite_cls.__name__}.{name}: {e}")
                errors.append((suite_cls.__name__, name, traceback.format_exc()))
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if errors:
        print("\n--- Failures ---")
        for cls_name, test_name, tb in errors:
            print(f"\n{cls_name}.{test_name}:\n{tb}")
        sys.exit(1)
