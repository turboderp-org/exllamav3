# sm70 port — Step 0 golden-vector harness.
#
# Reference for every sm70 kernel gate: dequant (ext.reconstruct) + fp32 GEMM.
# The sm80 GEMV accumulates in fp16 with a fold cadence, so its output is
# compared against the fp32 reference with a documented tolerance; the sm70
# f32-acc path must hit the fp32 reference tightly.
#
# Trellis layout (exl3_gemm.cu:348): logically (k/16, n/16, 16*K) uint16.
# Plain reconstruct emits the rotated-basis W_hat (no suh/svh applied) —
# suh/svh are Hadamard sign vectors consumed by reconstruct_had_slice and
# had_r_128, not by the plain dequant, so the reference here is self-consistent.
#
# Run:  python tests/test_sm70_reference.py

import sys
import torch
sys.path.insert(0, ".")
from exllamav3_ext import reconstruct, hgemm

DEV = "cuda"
SHAPES = [(4096, 4096), (4096, 11008), (5120, 8192)]  # (K_dim, N_dim)
BITS = [2, 3, 4]
MS = [1, 2, 4, 8, 64, 144]


def make_pack(k_dim, n_dim, bits):
    K = bits
    trellis = torch.randint(-32768, 32767, (k_dim // 16, n_dim // 16, 16 * K),
                            dtype=torch.int16, device=DEV)
    return trellis


def reference_w(trellis, k_dim, n_dim, bits):
    w = torch.empty((k_dim, n_dim), dtype=torch.half, device=DEV)
    reconstruct(w, trellis, bits, False, False)
    return w


def main():
    torch.manual_seed(0)
    for bits in BITS:
        for (k_dim, n_dim) in SHAPES:
            trellis = make_pack(k_dim, n_dim, bits)
            w = reference_w(trellis, k_dim, n_dim, bits)
            for M in MS:
                x = torch.randn(M, k_dim, dtype=torch.half, device=DEV)
                y_ref = x.float() @ w.float()
                y_hg = torch.empty(M, n_dim, dtype=torch.half, device=DEV)
                hgemm(x, w, y_hg)
                # hgemm is fp16 cuBLAS — bound the reference gap
                err = (y_hg.float() - y_ref).abs().max().item()
                scale = y_ref.abs().max().item()
                assert err / scale < 2e-2, f"hgemm vs fp32 ref too far: {err} / {scale}"
            print(f"bits={bits} k={k_dim} n={n_dim}: reference+hgemm OK "
                  f"(w mean={w.float().mean().item():+.4f} std={w.float().std().item():.4f})")


if __name__ == "__main__":
    main()