# Step 2 perf: GEMV decode throughput vs reconstruct+hgemm baseline.
# Measures the sm70 GEMV kernel (EXL3_GEMV=2 forces the gemv path)
# against the reconstruct-forced fallback on V100.
import torch, sys, time
sys.path.insert(0, "/home/nvidia/Dev/exllamav3-sm70")
import exllamav3_ext as ext

DEV = "cuda"
SHAPES = [(4096, 4096), (4096, 11008), (5120, 8192)]
BITS = [2, 3, 4]
MS = [1, 2, 4, 8]
ITERS = 50
WARMUP = 10


def bench(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    import os
    force_gemv = os.environ.get("EXL3_GEMV") == "2"
    print(f"{'K':>2} {'shape':>14} {'M':>2} {'gemv us':>9} {'rec+hgemm us':>13} {'speedup':>8}")
    for K in BITS:
        for (k, n) in SHAPES:
            torch.manual_seed(1)
            trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K),
                                    dtype=torch.int16, device=DEV)
            w = torch.empty((k, n), dtype=torch.half, device=DEV)
            ext.reconstruct(w, trellis, K, False, False)
            for M in MS:
                x = torch.randn(M, k, dtype=torch.half, device=DEV)
                y = torch.empty(M, n, dtype=torch.half, device=DEV)
                suh = torch.ones(k, dtype=torch.half, device=DEV)
                xh = torch.empty_like(x)
                svh = torch.ones(n, dtype=torch.half, device=DEV)

                # gemv path
                t_gemv = bench(lambda: ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, False, False, 0))

                # reconstruct+hgemm baseline (dequant amortized over the call —
                # realistic serving dequants once per weight set, so measure
                # hgemm alone as the floor and dequant separately)
                t_hgemm = bench(lambda: ext.hgemm(x, w, y))
                t_recon = bench(lambda: ext.reconstruct(w, trellis, K, False, False))

                # tok/s: M rows per call
                ts_gemv = M / t_gemv
                ts_floor = M / t_hgemm
                print(f"{K:>2} {f'{k}x{n}':>14} {M:>2} {t_gemv*1e6:>9.1f} {t_hgemm*1e6:>13.1f} {t_hgemm/t_gemv:>8.2f}x"
                      f"  (dequant {t_recon*1e3:.1f} ms, gemv {ts_gemv:.0f} tok/s vs floor {ts_floor:.0f})")


if __name__ == "__main__":
    main()