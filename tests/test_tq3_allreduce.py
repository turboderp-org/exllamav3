"""
Tests for TQ3 compressed all-reduce.

Since actual multi-GPU all-reduce requires multiple processes,
these tests simulate the compression quality and verify that
TQ3 compress->sum->decompress produces acceptable results.
"""
import torch
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sqnr(original, reconstructed):
    sig = (original.float() ** 2).mean()
    noise = ((original.float() - reconstructed.float()) ** 2).mean()
    if noise < 1e-20:
        return float('inf')
    return 10 * math.log10(sig / noise)


def tq3_compress_py(x):
    """Pure PyTorch TQ3 compression (no CUDA ext needed)."""
    assert x.numel() % 32 == 0
    blocks = x.float().view(-1, 32)
    scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    normalized = blocks / scales
    nonzero = (normalized.abs() >= 0.5).int()
    positive = ((normalized > 0) & (nonzero == 1)).int()
    bit_idx = torch.arange(32, device=x.device).unsqueeze(0)
    bp0 = (nonzero << bit_idx).sum(dim=1).to(torch.int32)
    bp1 = (positive << bit_idx).sum(dim=1).to(torch.int32)
    return bp0, bp1, scales.squeeze(1).half()


def tq3_decompress_py(bp0, bp1, scales, numel):
    """Pure PyTorch TQ3 decompression."""
    bit_idx = torch.arange(32, device=bp0.device).unsqueeze(0)
    nz = ((bp0.unsqueeze(1) >> bit_idx) & 1).float()
    pos = ((bp1.unsqueeze(1) >> bit_idx) & 1).float()
    ternary = nz * (2.0 * pos - 1.0)
    result = (ternary * scales.float().unsqueeze(1)).reshape(-1)
    return result[:numel].half()


class TestTQ3CompressedAllReduceSimulation:

    def test_single_rank_roundtrip(self):
        torch.manual_seed(42)
        x = torch.randn(4096, dtype=torch.float16, device='cuda' if torch.cuda.is_available() else 'cpu')
        bp0, bp1, scales = tq3_compress_py(x)
        recovered = tq3_decompress_py(bp0, bp1, scales, x.numel())
        ratio = sqnr(x, recovered)
        assert ratio >= 6.0, f"SQNR {ratio:.2f} dB < 6 dB"

    def test_simulated_4rank_allreduce(self):
        torch.manual_seed(42)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_ranks = 4
        tensors = [torch.randn(4096, dtype=torch.float16, device=dev) for _ in range(num_ranks)]
        exact_sum = sum(t.float() for t in tensors).half()

        compressed_sum = torch.zeros(4096, dtype=torch.float32, device=dev)
        for t in tensors:
            bp0, bp1, scales = tq3_compress_py(t)
            decompressed = tq3_decompress_py(bp0, bp1, scales, t.numel())
            compressed_sum += decompressed.float()
        compressed_sum = compressed_sum.half()

        ratio = sqnr(exact_sum, compressed_sum)
        assert ratio >= 4.0, f"4-rank compressed SQNR {ratio:.2f} dB"
        print(f"4-rank compressed all-reduce SQNR: {ratio:.2f} dB")

    def test_simulated_8rank_allreduce(self):
        torch.manual_seed(42)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_ranks = 8
        tensors = [torch.randn(8192, dtype=torch.float16, device=dev) for _ in range(num_ranks)]
        exact_sum = sum(t.float() for t in tensors).half()

        compressed_sum = torch.zeros(8192, dtype=torch.float32, device=dev)
        for t in tensors:
            bp0, bp1, scales = tq3_compress_py(t)
            decompressed = tq3_decompress_py(bp0, bp1, scales, t.numel())
            compressed_sum += decompressed.float()
        compressed_sum = compressed_sum.half()

        ratio = sqnr(exact_sum, compressed_sum)
        assert ratio >= 3.0, f"8-rank compressed SQNR {ratio:.2f} dB"
        print(f"8-rank compressed all-reduce SQNR: {ratio:.2f} dB")

    def test_bandwidth_ratio(self):
        """Verify TQ3 achieves expected compression ratio."""
        numel = 8192
        fp16_bytes = numel * 2
        num_blocks = numel // 32
        tq3_bytes = num_blocks * (4 + 4 + 2)  # 2 uint32 + 1 fp16 per block
        ratio = fp16_bytes / tq3_bytes
        assert ratio >= 6.0, f"Compression ratio {ratio:.2f}x < 6x"
        print(f"TQ3 compression ratio: {ratio:.2f}x ({fp16_bytes} -> {tq3_bytes} bytes)")

    def test_zeros(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.zeros(1024, dtype=torch.float16, device=dev)
        bp0, bp1, scales = tq3_compress_py(x)
        recovered = tq3_decompress_py(bp0, bp1, scales, x.numel())
        assert torch.allclose(recovered.float(), x.float(), atol=1e-3)

    def test_sign_preservation(self):
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.tensor([1.0, -1.0, 0.5, -0.5] * 8, dtype=torch.float16, device=dev)
        bp0, bp1, scales = tq3_compress_py(x)
        recovered = tq3_decompress_py(bp0, bp1, scales, x.numel())
        # Signs should be preserved for large values
        for i in [0, 1]:
            assert (recovered[i] > 0) == (x[i] > 0), f"Sign mismatch at index {i}"

    def test_large_tensor(self):
        torch.manual_seed(42)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.randn(131072, dtype=torch.float16, device=dev)  # 128K values
        bp0, bp1, scales = tq3_compress_py(x)
        recovered = tq3_decompress_py(bp0, bp1, scales, x.numel())
        ratio = sqnr(x, recovered)
        assert ratio >= 5.0, f"128K SQNR {ratio:.2f} dB < 5 dB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
