"""
Benchmark TQ3 compressed all-reduce overhead.

Break-even: TQ3 wins when compress+decompress time < bandwidth savings.
  PCIe Gen4 x16 (32 GB/s):    break-even at ~26 ns/byte overhead
  InfiniBand HDR (25 GB/s):    break-even at ~34 ns/byte overhead
  Ethernet 100G (12.5 GB/s):   break-even at ~67 ns/byte overhead
"""
import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def tq3_compress_py(x):
    """Pure PyTorch TQ3 compression fallback."""
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
    """Pure PyTorch TQ3 decompression fallback."""
    bit_idx = torch.arange(32, device=bp0.device).unsqueeze(0)
    nz = ((bp0.unsqueeze(1) >> bit_idx) & 1).float()
    pos = ((bp1.unsqueeze(1) >> bit_idx) & 1).float()
    ternary = nz * (2.0 * pos - 1.0)
    result = (ternary * scales.float().unsqueeze(1)).reshape(-1)
    return result[:numel].half()


def bench_tq3_compress_decompress():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    print("=" * 70)
    print("TQ3 Compressed All-Reduce Bandwidth Analysis")
    print("=" * 70)

    # Attempt to load the CUDA extension once and report which path is active.
    use_ext = False
    try:
        from exllamav3.ext import exllamav3_ext as ext
        # Probe for the expected symbols so we fail fast rather than at loop time.
        _ = ext.quant_tq3_cache_cont
        _ = ext.dequant_tq3_cache_cont
        use_ext = True
        print("Backend: CUDA extension (exllamav3_ext)")
    except (ImportError, AttributeError):
        print("Backend: pure PyTorch fallback")

    sizes = [4096, 8192, 16384, 32768, 65536, 131072]
    warmup = 20
    iters = 100

    print(f"\n{'Size':>8} | {'Compress':>10} | {'Decompress':>10} | {'Ratio':>6} | {'Break-even BW':>14}")
    print("-" * 70)

    for size in sizes:
        x = torch.randn(size, dtype=torch.float16, device='cuda')
        num_blocks = size // 32

        if use_ext:
            from exllamav3.ext import exllamav3_ext as ext
            packed = torch.empty(num_blocks * 2, dtype=torch.int32, device='cuda')
            scales = torch.empty(num_blocks, dtype=torch.float16, device='cuda')
            output = torch.empty_like(x)

            # Warmup
            for _ in range(warmup):
                ext.quant_tq3_cache_cont(x, packed, scales)
                ext.dequant_tq3_cache_cont(packed, scales, output)
            torch.cuda.synchronize()

            # Compress
            t0 = time.perf_counter()
            for _ in range(iters):
                ext.quant_tq3_cache_cont(x, packed, scales)
            torch.cuda.synchronize()
            compress_ms = (time.perf_counter() - t0) / iters * 1000

            # Decompress
            t0 = time.perf_counter()
            for _ in range(iters):
                ext.dequant_tq3_cache_cont(packed, scales, output)
            torch.cuda.synchronize()
            decompress_ms = (time.perf_counter() - t0) / iters * 1000

        else:
            # PyTorch fallback — benchmark the pure-Python path.
            # Warmup
            for _ in range(warmup):
                bp0, bp1, sc = tq3_compress_py(x)
                _ = tq3_decompress_py(bp0, bp1, sc, size)
            torch.cuda.synchronize()

            # Compress
            t0 = time.perf_counter()
            for _ in range(iters):
                bp0, bp1, sc = tq3_compress_py(x)
            torch.cuda.synchronize()
            compress_ms = (time.perf_counter() - t0) / iters * 1000

            # Decompress (reuse last compressed result)
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = tq3_decompress_py(bp0, bp1, sc, size)
            torch.cuda.synchronize()
            decompress_ms = (time.perf_counter() - t0) / iters * 1000

        fp16_bytes = size * 2
        tq3_bytes = num_blocks * 10  # 2 x int32 (4B each) + 1 x fp16 (2B) per block
        ratio = fp16_bytes / tq3_bytes

        total_ms = compress_ms + decompress_ms
        saved_bytes = fp16_bytes - tq3_bytes
        if total_ms > 0:
            breakeven_bw_gbs = saved_bytes / (total_ms / 1000) / 1e9
        else:
            breakeven_bw_gbs = float('inf')

        print(
            f"{size:>8} | {compress_ms:>8.3f}ms | {decompress_ms:>8.3f}ms"
            f" | {ratio:>5.1f}x | {breakeven_bw_gbs:>10.1f} GB/s"
        )

    print(f"\nInterpretation:")
    print(f"  TQ3 wins when your interconnect is SLOWER than the break-even BW.")
    print(f"  PCIe Gen4 x16: 32 GB/s  -> TQ3 wins if break-even > 32 GB/s")
    print(f"  InfiniBand HDR: 25 GB/s  -> TQ3 wins if break-even > 25 GB/s")
    print(f"  NVLink (A100):  600 GB/s -> TQ3 unlikely to win")


if __name__ == "__main__":
    bench_tq3_compress_decompress()
