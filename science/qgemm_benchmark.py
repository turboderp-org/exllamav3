import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util import Timer
from exllamav3.util.memory import free_mem
from tabulate import tabulate

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

K = 8
runs = 50

shapes_m = [1, 4, 16]

shapes_kn = [
    (4096, 14336),
    (14336, 4096),
    (4096, 1024),
    (4096, 4096),
    (4096, 128000),
    (4096, 128256),
    (8192, 28672),
    (28672, 8192),
    (8192, 1024),
    (8192, 8192),
    (8192, 128000),
    (4096, 128256),
]

devices = [
    "cuda:1",
    "cuda:2",
    "cuda:3",
]

kernels = range(1, 1 + ext.exl3_gemm_num_kernel_shapes())

@torch.inference_mode()
def main():

    headers = ["(m, k, n)"]
    for idx in kernels:
        headers.append(f"[{idx}]")
    headers.append("D")

    results = []
    for device in devices:
        results.append([])
        for m in shapes_m:
            for (k, n) in shapes_kn:
                assert k % 16 == 0
                assert n % 16 == 0
                results[-1].append([])
                results[-1][-1].append(str((m, k, n)))

                free_mem()

                # Tensors for matmul kernel (not testing correctness)
                proto_a = torch.randn((m, k), dtype = torch.half, device = device)
                proto_b = torch.zeros((k // 16, n // 16, 16 * K), dtype = torch.short, device = device)
                proto_c = torch.zeros((m, n), dtype = torch.half, device = device)
                proto_suh = torch.randn((k,), dtype = torch.half, device = device)
                proto_svh = torch.randn((n,), dtype = torch.half, device = device)

                # Create enough clones to cycle through to prevent L2 cache from skewing results
                assume_cache = 512 * 1024**2
                proto_size = proto_a.numel() * 2 + proto_b.numel() * 2 + proto_c.numel() * 2
                num_buffers = max(assume_cache // proto_size + 1, 2)
                a = [proto_a.clone() for _ in range(num_buffers)]
                b = [proto_b.clone() for _ in range(num_buffers)]
                c = [proto_c.clone() for _ in range(num_buffers)]
                suh = [proto_suh.clone() for _ in range(num_buffers)]
                svh = [proto_svh.clone() for _ in range(num_buffers)]

                # Get preferred kernel for current shape
                pref = ext.exl3_gemm(a[0], b[0], c[0], suh[0], a[0], svh[0], -1)

                # Test all kernels
                kresults = []
                for kernel in kernels:
                    print(".", end = "", flush = True)

                    # Test if kernel is compatible
                    compat = ext.exl3_gemm_shape_compat(kernel, m, k, n, K)
                    if not compat:
                        results[-1][-1].append("N/A")
                        kresults.append(1e6)
                        continue

                    # Warmup passes for good measure
                    for i_ in range(10):
                        i = i_ % num_buffers
                        ext.exl3_gemm(a[i], b[i], c[i], suh[i], a[i], svh[i], kernel)

                    # Test
                    dummy = c[0][0, 0].item()
                    with Timer() as t:
                        for i_ in range(runs):
                            i = i_ % num_buffers
                            ext.exl3_gemm(a[i], b[i], c[i], suh[i], a[i], svh[i], kernel)
                        dummy = c[i][0, 0].item()

                    mean_time_ms = t.interval / runs * 1000
                    kresults.append(mean_time_ms)
                    results[-1][-1].append(f"{mean_time_ms:.5f}")

                # Highlight fastest and preferred kernel, mark shapes where preferred is within 1% of fastest
                b = min(kresults)
                p = 0
                for idx, v in enumerate(kresults):
                    if v == b:
                        results[-1][-1][idx + 1] += " *"
                    if kernels[idx] == pref:
                        results[-1][-1][idx + 1] += " P"
                        p = v
                d = (p - b) / b
                results[-1][-1].append(f"{d:.4f}" if d > 0.01 else "OK")

        print()

    for device, result in zip(devices, results):
        print()
        print(device)
        print()
        print(tabulate(result, headers = headers, tablefmt = "github", floatfmt=".5f"))

if __name__ == "__main__":
    main()
