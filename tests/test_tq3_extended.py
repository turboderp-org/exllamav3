"""
Tests for extended TQ3 features: MoE expert offload, activation compression,
embedding compression, and TP communication compression.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import math


def sqnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio in dB."""
    signal_power = (original.float() ** 2).mean()
    noise_power = ((original.float() - reconstructed.float()) ** 2).mean()
    if noise_power < 1e-20:
        return float('inf')
    return 10 * math.log10(signal_power.item() / noise_power.item())


# =============================================================================
# TQ3 Expert Offloader tests
# =============================================================================

class TestTQ3ExpertOffloader:
    """Test MoE expert TQ3 compression/decompression."""

    def test_compress_decompress_roundtrip(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader

        torch.manual_seed(42)
        offloader = TQ3ExpertOffloader(store_device="cpu")

        weight = torch.randn(512, 256, dtype=torch.float16, device="cuda")
        offloader.compress_expert(0, weight)

        recovered = offloader.get_expert(0, device="cuda")
        assert recovered.shape == weight.shape
        assert recovered.dtype == torch.float16

        ratio = sqnr(weight, recovered)
        assert ratio >= 4.0, f"Expert SQNR too low: {ratio:.2f} dB"
        print(f"Expert offload SQNR: {ratio:.2f} dB")

    def test_multiple_experts(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader

        torch.manual_seed(42)
        offloader = TQ3ExpertOffloader(store_device="cpu")

        num_experts = 8
        weights = []
        for i in range(num_experts):
            w = torch.randn(256, 128, dtype=torch.float16, device="cuda")
            weights.append(w)
            offloader.compress_expert(i, w)

        assert offloader.num_experts() == num_experts
        assert offloader.compressed_size() < offloader.uncompressed_size()

        # Decompress specific experts (simulating top-k selection)
        for i in [2, 5, 7]:
            recovered = offloader.get_expert(i, device="cuda")
            ratio = sqnr(weights[i], recovered)
            assert ratio >= 3.0, f"Expert {i} SQNR too low: {ratio:.2f} dB"

        offloader.release_all()
        assert len(offloader.active_cache) == 0

    def test_compression_ratio(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader

        offloader = TQ3ExpertOffloader()
        w = torch.randn(1024, 512, dtype=torch.float16, device="cuda")
        offloader.compress_expert(0, w)

        ratio = offloader.uncompressed_size() / offloader.compressed_size()
        assert ratio >= 3.0, f"Compression ratio too low: {ratio:.2f}x"
        print(f"Expert compression ratio: {ratio:.2f}x")

    def test_cache_returns_same_reference(self):
        """get_expert called twice should return the cached tensor (same object)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader

        torch.manual_seed(42)
        offloader = TQ3ExpertOffloader(store_device="cpu")
        w = torch.randn(256, 128, dtype=torch.float16, device="cuda")
        offloader.compress_expert(0, w)

        w1 = offloader.get_expert(0, device="cuda")
        w2 = offloader.get_expert(0, device="cuda")
        assert w1 is w2, "Second get_expert should return cached reference"

    def test_release_clears_cache(self):
        """release_expert should remove the expert from active_cache."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader

        torch.manual_seed(42)
        offloader = TQ3ExpertOffloader(store_device="cpu")
        w = torch.randn(256, 128, dtype=torch.float16, device="cuda")
        offloader.compress_expert(0, w)

        offloader.get_expert(0, device="cuda")
        assert 0 in offloader.active_cache
        offloader.release_expert(0)
        assert 0 not in offloader.active_cache

    def test_release_nonexistent_is_noop(self):
        """release_expert on an index not in cache should not raise."""
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader
        offloader = TQ3ExpertOffloader()
        offloader.release_expert(999)  # should not raise

    def test_in_features_not_multiple_of_32_raises(self):
        """compress_expert should assert if in_features is not multiple of 32."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader

        offloader = TQ3ExpertOffloader()
        w = torch.randn(100, 64, dtype=torch.float16, device="cuda")  # 100 % 32 != 0
        with pytest.raises(AssertionError):
            offloader.compress_expert(0, w)

    def test_storage_on_cpu(self):
        """Compressed data should be stored on CPU when store_device='cpu'."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_expert_offload import TQ3ExpertOffloader

        offloader = TQ3ExpertOffloader(store_device="cpu")
        w = torch.randn(256, 128, dtype=torch.float16, device="cuda")
        offloader.compress_expert(0, w)

        assert offloader.experts[0]["packed"].device.type == "cpu"
        assert offloader.experts[0]["scales"].device.type == "cpu"


# =============================================================================
# TQ3 Activation Compressor tests
# =============================================================================

class TestTQ3ActivationCompressor:
    """Test activation compression between layers."""

    def test_compress_decompress_3d(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from exllamav3.ext import exllamav3_ext as ext
        except ImportError:
            pytest.skip("exllamav3_ext not compiled")
        from exllamav3.modules.tq3_activation import TQ3ActivationCompressor

        torch.manual_seed(42)
        # Typical hidden state: (batch=4, seq=128, hidden=4096)
        x = torch.randn(4, 128, 4096, dtype=torch.float16, device="cuda")

        compressed = TQ3ActivationCompressor.compress(x)
        recovered = TQ3ActivationCompressor.decompress(compressed)

        assert recovered.shape == x.shape
        assert recovered.dtype == torch.float16

        ratio = sqnr(x, recovered)
        assert ratio >= 6.0, f"Activation SQNR too low: {ratio:.2f} dB"
        print(f"Activation compression SQNR: {ratio:.2f} dB")

    def test_memory_savings(self):
        from exllamav3.modules.tq3_activation import TQ3ActivationCompressor

        savings = TQ3ActivationCompressor.memory_savings((4, 2048, 4096))
        assert savings["ratio"] >= 3.0
        assert savings["savings_pct"] >= 60.0
        print(f"Activation memory savings: {savings['ratio']:.2f}x ({savings['savings_pct']:.1f}%)")

    def test_non_multiple_of_32(self):
        """Test with tensor size not multiple of 32."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        try:
            from exllamav3.ext import exllamav3_ext as ext
        except ImportError:
            pytest.skip("exllamav3_ext not compiled")
        from exllamav3.modules.tq3_activation import TQ3ActivationCompressor

        torch.manual_seed(42)
        # 33 elements = not multiple of 32
        x = torch.randn(1, 33, dtype=torch.float16, device="cuda")
        compressed = TQ3ActivationCompressor.compress(x)
        recovered = TQ3ActivationCompressor.decompress(compressed)

        assert recovered.shape == x.shape


# =============================================================================
# TQ3 Embedding tests
# =============================================================================

class TestTQ3Embedding:
    """Test TQ3-compressed embedding table."""

    def test_from_weight_and_lookup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_embedding import TQ3Embedding

        torch.manual_seed(42)
        vocab_size, hidden_size = 1024, 256  # Multiple of 32
        weight = torch.randn(vocab_size, hidden_size, dtype=torch.float16, device="cuda")

        emb = TQ3Embedding.from_weight(weight)

        input_ids = torch.tensor([[1, 5, 100, 999]], device="cuda")
        output = emb.forward(input_ids)

        assert output.shape == (1, 4, hidden_size)
        assert output.dtype == torch.float16

        # Check each looked-up row has reasonable SQNR
        for i, idx in enumerate(input_ids[0]):
            original_row = weight[idx]
            recovered_row = output[0, i]
            ratio = sqnr(original_row, recovered_row)
            assert ratio >= 2.0, f"Embedding row {idx} SQNR too low: {ratio:.2f} dB"

    def test_storage_savings(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_embedding import TQ3Embedding

        vocab_size, hidden_size = 32000, 4096
        weight = torch.randn(vocab_size, hidden_size, dtype=torch.float16, device="cuda")

        emb = TQ3Embedding.from_weight(weight)

        fp16_size = vocab_size * hidden_size * 2
        tq3_size = emb.storage_size()
        ratio = fp16_size / tq3_size

        assert ratio >= 3.0, f"Embedding compression ratio too low: {ratio:.2f}x"
        print(f"Embedding compression: {fp16_size/1e6:.1f}MB -> {tq3_size/1e6:.1f}MB ({ratio:.2f}x)")

    def test_batch_lookup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from exllamav3.modules.tq3_embedding import TQ3Embedding

        torch.manual_seed(42)
        weight = torch.randn(256, 64, dtype=torch.float16, device="cuda")
        emb = TQ3Embedding.from_weight(weight)

        # Batch of 3 sequences, length 5
        input_ids = torch.randint(0, 256, (3, 5), device="cuda")
        output = emb.forward(input_ids)

        assert output.shape == (3, 5, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
