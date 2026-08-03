# Fused reconstruct_had_slice vs the reference pipeline: W = diag(suh) H128 W_hat H128
# diag(svh) per 128-block, 1/sqrt(128) per side. Reference W_hat comes from the plain
# reconstruct kernel; the Hadamard reference is an explicit fp32 Sylvester matmul, so any
# sign/order/scale error in the fused kernel shows up directly. Also checks forward-path
# equivalence: had(x*su) @ W_hat -> had -> *sv (old pipeline) vs x @ W_fused.

import torch
from exllamav3.ext import exllamav3_ext as ext

torch.manual_seed(0)
device = "cuda:1"
torch.cuda.set_device(device)


def sylvester(n):
    h = torch.ones(1, 1, dtype = torch.float, device = device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


H = sylvester(128) / 128 ** 0.5


def ref_transform(w_hat, suh, svh):
    k, n = w_hat.shape
    w = w_hat.float().view(k // 128, 128, n)
    w = torch.einsum("ij,bjn->bin", H, w).reshape(k, n)
    w = w.view(k, n // 128, 128)
    w = torch.einsum("bki,ij->bkj", w.transpose(0, 1), H).transpose(0, 1).reshape(k, n)
    return (w * suh.float()[:, None] * svh.float()[None, :]).half()


def main():
    for (k, n, K, mcg, mul1) in [
        (256, 128, 3, False, False),
        (512, 384, 2, False, False),
        (1024, 512, 5, False, False),
        (384, 256, 4, True, False),
        (256, 512, 3, False, True),
        (4096, 1024, 3, False, False),
    ]:
        torch.manual_seed(k * 7 + n + K)
        trellis = torch.randint(0, 65536, (k // 16, n // 16, 256 * K // 16),
                                dtype = torch.int32, device = device).to(torch.short)
        suh = torch.sign(torch.randn(k, device = device)).half()
        svh = torch.sign(torch.randn(n, device = device)).half()

        w_hat = torch.empty(k, n, dtype = torch.half, device = device)
        ext.reconstruct(w_hat, trellis, K, mcg, mul1)
        ref = ref_transform(w_hat, suh, svh)

        w = torch.empty(k, n, dtype = torch.half, device = device)
        ext.reconstruct_had_slice(w, trellis, suh, svh, K, mcg, mul1, 0)

        err = (w.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        assert err / scale < 2e-3, f"({k},{n},K{K}): rel {err/scale:.2e}"
        print(f"  PASS fused vs ref ({k:5d},{n:5d}) K={K} mcg={mcg} mul1={mul1}: rel {err/scale:.2e}")

        # slice path: reconstruct columns [128, 128+n_sl) only
        if n >= 384:
            n_sl = 128
            ws = torch.empty(k, n_sl, dtype = torch.half, device = device)
            ext.reconstruct_had_slice(ws, trellis, suh, svh[128:], K, mcg, mul1, 128)
            errs = (ws.float() - ref[:, 128:256].float()).abs().max().item()
            assert errs / scale < 2e-3, f"slice: rel {errs/scale:.2e}"
            print(f"  PASS slice n_offset=128: rel {errs/scale:.2e}")

    # Forward equivalence: old had->gemm->had pipeline vs plain gemm on fused W
    k, n, K = 1024, 512, 3
    torch.manual_seed(99)
    trellis = torch.randint(0, 65536, (k // 16, n // 16, 256 * K // 16),
                            dtype = torch.int32, device = device).to(torch.short)
    suh = torch.sign(torch.randn(k, device = device)).half()
    svh = torch.sign(torch.randn(n, device = device)).half()
    x = torch.randn(64, k, dtype = torch.half, device = device) * 0.1

    w_hat = torch.empty(k, n, dtype = torch.half, device = device)
    ext.reconstruct(w_hat, trellis, K, False, False)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    y_old = torch.empty(64, n, dtype = torch.half, device = device)
    ext.hgemm(xh, w_hat, y_old)
    ext.had_r_128(y_old, y_old, None, svh, 1.0)

    w = torch.empty(k, n, dtype = torch.half, device = device)
    ext.reconstruct_had_slice(w, trellis, suh, svh, K, False, False, 0)
    y_new = torch.empty(64, n, dtype = torch.half, device = device)
    ext.hgemm(x, w, y_new)

    num = (y_old.float() - y_new.float()).abs().max().item()
    den = y_old.float().abs().max().item()
    assert num / den < 2e-2, f"forward: rel {num/den:.2e}"
    print(f"  PASS forward old-pipeline vs fused-W: rel {num/den:.2e}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
