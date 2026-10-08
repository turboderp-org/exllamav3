"""
ext.reconstruct_had_slice (fused dequantize + Hadamard + scales) against the reference pipeline:
W = diag(suh) H128 W_hat H128 diag(svh) per 128-block, 1/sqrt(128) per side. The reference W_hat comes from the
plain reconstruct kernel; the Hadamard reference is an explicit fp32 Sylvester matmul, so any sign/order/scale
error in the fused kernel shows up directly. Also checks column slices (n_offset) and forward-path equivalence:
had(x * su) @ W_hat -> had -> * sv (the unfused pipeline) vs x @ W_fused.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_trellis


def _sylvester(n, device):
    h = torch.ones(1, 1, dtype = torch.float, device = device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def _ref_transform(w_hat, suh, svh):
    H = _sylvester(128, w_hat.device) / 128 ** 0.5
    k, n = w_hat.shape
    w = w_hat.float().view(k // 128, 128, n)
    w = torch.einsum("ij,bjn->bin", H, w).reshape(k, n)
    w = w.view(k, n // 128, 128)
    w = torch.einsum("bki,ij->bkj", w.transpose(0, 1), H).transpose(0, 1).reshape(k, n)
    return (w * suh.float()[:, None] * svh.float()[None, :]).half()


def _sign_weights(k, n, K, seed, device):
    gen = generator(seed)
    trellis = rand_trellis(k, n, K, gen, device)
    suh = torch.sign(torch.randn(k, generator = gen)).half().to(device)
    svh = torch.sign(torch.randn(n, generator = gen)).half().to(device)
    return trellis, suh, svh


@pytest.mark.parametrize("k, n, K, mcg, mul1", [
    (256, 128, 3, False, False),
    (512, 384, 2, False, False),
    (1024, 512, 5, False, False),
    (384, 256, 4, True, False),
    (256, 512, 3, False, True),
    (4096, 1024, 3, False, False),
])
@torch.inference_mode()
def test_reconstruct_had_slice(device, k, n, K, mcg, mul1):
    trellis, suh, svh = _sign_weights(k, n, K, k * 7 + n + K, device)

    w_hat = torch.empty(k, n, dtype = torch.half, device = device)
    ext.reconstruct(w_hat, trellis, K, mcg, mul1)
    ref = _ref_transform(w_hat, suh, svh)

    w = torch.empty(k, n, dtype = torch.half, device = device)
    ext.reconstruct_had_slice(w, trellis, suh, svh, K, mcg, mul1, 0)

    scale = ref.float().abs().max().item()
    err = (w.float() - ref.float()).abs().max().item()
    assert err / scale < 2e-3, f"({k},{n}) K={K}: rel {err / scale:.2e}"

    # Slice path: reconstruct columns [128, 256) only
    if n >= 384:
        n_sl = 128
        ws = torch.empty(k, n_sl, dtype = torch.half, device = device)
        ext.reconstruct_had_slice(ws, trellis, suh, svh[128:], K, mcg, mul1, 128)
        errs = (ws.float() - ref[:, 128:256].float()).abs().max().item()
        assert errs / scale < 2e-3, f"slice n_offset=128: rel {errs / scale:.2e}"


@torch.inference_mode()
def test_fused_weight_forward_equivalence(device):
    """The unfused had -> hgemm -> had pipeline over W_hat vs a plain hgemm over the fused W"""
    k, n, K = 1024, 512, 3
    trellis, suh, svh = _sign_weights(k, n, K, 99, device)
    x = (torch.randn(64, k, generator = generator(100)) * 0.1).half().to(device)

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
    assert num / den < 2e-2, f"forward: rel {num / den:.2e}"
