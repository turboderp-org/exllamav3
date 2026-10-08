"""
exl3_mgemm sliced mode: a bundle of matrices with different widths (Q/K/V-like) cut into
equal-width column slices and run as one launch, each slice writing in place into its source's
full-width output. Checks against the per-matrix single GEMM (fp16 accumulation-order noise only)
and that the plain uniform-width MGEMM path still matches the single GEMM the same way.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def _weights(k, n, bits, device):
    B = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype = torch.int16, device = device)
    suh = torch.randn(k, device = device).sign().half()
    svh = torch.randn(n, device = device).sign().half()
    return B, suh, svh


def _ptrs(ts_or_ints, device):
    return torch.tensor([t.data_ptr() if isinstance(t, torch.Tensor) else t for t in ts_or_ints], dtype = torch.long, device = device)


def _single(x, B, suh, svh, mcg):
    m = x.shape[0]
    xh = torch.empty_like(x)
    c = torch.empty((m, B.shape[1] * 16), dtype = torch.half, device = x.device)
    ext.exl3_gemm(x, B, c, suh, xh, svh, 0, mcg, False, 0)
    return c


def _close(a, b, tol = 3e-3):
    rel = ((a.float() - b.float()).pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt()).item()
    assert rel < tol, f"relative rms {rel:.2e}"


@pytest.mark.parametrize("bits", [3, 4, 5])
@pytest.mark.parametrize("m", [1, 4, 16, 32])
@pytest.mark.parametrize("widths", [(4096, 1024, 1024), (2048, 512, 512), (3072, 1024, 1024, 3072)])
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@torch.inference_mode()
def test_sliced_bundle(device, bits, m, widths, codebook):
    torch.manual_seed(bits * 100 + m)
    k = 4096
    mcg = codebook == "mcg" and bits >= 5
    mul1 = codebook == "mul1"
    mats = [_weights(k, n, bits, device) for n in widths]
    W = min(widths)
    x = (torch.randn(m, k, device = device) * 0.5).half()
    # Reference: the plain uniform-width mgemm over the same slices (same kernel family and
    # numerics); the single GEMM's kernel shapes differ in accumulation order by up to ~1e-2 on
    # mul1 tensors, which would mask a slicing bug at that tolerance. The two launches tune
    # separately and may land on different row tiles, whose fp16-accumulated outputs (sm_86)
    # differ by a little over 2e-3 from each other while each stays as close to exact as before
    ref = [_uniform_slices(x, B, svh, suh, W, bits, mcg, mul1) for B, suh, svh in mats]

    outs = [torch.empty((m, n), dtype = torch.half, device = device) for n in widths]
    tr, sv, cp, src, stride = [], [], [], [], []
    for i, ((B, suh, svh), n) in enumerate(zip(mats, widths)):
        for n0 in range(0, n, W):
            tr.append(B.data_ptr() + (n0 // 16) * 16 * bits * 2)
            sv.append(svh.data_ptr() + n0 * 2)
            cp.append(outs[i].data_ptr() + n0 * 2)
            src.append(i)
            stride.append(n)
    S = len(tr)
    i32 = lambda v: torch.tensor(v, dtype = torch.int32, device = device)
    a_had = torch.empty((len(mats), m, k), dtype = torch.half, device = device)
    carrier = torch.empty((S, 1, W), dtype = torch.half, device = device).expand(S, m, W)
    ext.exl3_mgemm(
        x.view(1, m, k), _ptrs(tr, device), carrier, _ptrs([suh for _, suh, _ in mats], device), a_had, _ptrs(sv, device),
        None, None, bits, -1, mcg, mul1, -1, -1, 0,
        1, i32([W] * S), _ptrs(cp, device), i32(stride), i32(src), len(mats),
    )
    for o, r in zip(outs, ref):
        _close(o, r)


def _uniform_slices(x, B, svh, suh, W, bits, mcg, mul1):
    """One matrix as a plain uniform-width mgemm over contiguous copies of its W-wide column
    slices, reassembled to (m, n)"""
    m = x.shape[0]; n = B.shape[1] * 16; c = n // W
    device = x.device
    Bs = [B[:, i * W // 16:(i + 1) * W // 16, :].contiguous() for i in range(c)]
    svs = [svh[i * W:(i + 1) * W].contiguous() for i in range(c)]
    a_had = torch.empty((c, m, x.shape[1]), dtype = torch.half, device = device)
    out = torch.empty((c, m, W), dtype = torch.half, device = device)
    ext.exl3_mgemm(x.view(1, m, -1), _ptrs(Bs, device), out, _ptrs([suh] * c, device), a_had, _ptrs(svs, device),
                   None, None, bits, -1, mcg, mul1, -1, -1, 0, 1, None, None)
    return out.permute(1, 0, 2).reshape(m, n)


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("m", [1, 4, 32])
@torch.inference_mode()
def test_sliced_bundle_fp32_out(device, bits, m):
    """fp32 outputs (GDN qkv/z projections): an 8192 + 4096 bundle against the single GEMMs"""
    torch.manual_seed(m)
    k, widths, mcg = 2560, (8192, 4096), bits >= 5
    mats = [_weights(k, n, bits, device) for n in widths]
    W = min(widths)
    x = (torch.randn(m, k, device = device) * 0.5).half()
    ref = []
    for B, suh, svh in mats:
        xh = torch.empty_like(x); c = torch.empty((m, B.shape[1] * 16), dtype = torch.float, device = device)
        ext.exl3_gemm(x, B, c, suh, xh, svh, 0, mcg, False, 0); ref.append(c)
    outs = [torch.empty((m, n), dtype = torch.float, device = device) for n in widths]
    tr, sv, cp, src, stride = [], [], [], [], []
    for i, ((B, suh, svh), n) in enumerate(zip(mats, widths)):
        for n0 in range(0, n, W):
            tr.append(B.data_ptr() + (n0 // 16) * 16 * bits * 2); sv.append(svh.data_ptr() + n0 * 2)
            cp.append(outs[i].data_ptr() + n0 * 4); src.append(i); stride.append(n)
    S = len(tr)
    i32 = lambda v: torch.tensor(v, dtype = torch.int32, device = device)
    a_had = torch.empty((2, m, k), dtype = torch.half, device = device)
    carrier = torch.empty((S, 1, W), dtype = torch.float, device = device).expand(S, m, W)
    ext.exl3_mgemm(x.view(1, m, k), _ptrs(tr, device), carrier, _ptrs([suh for _, suh, _ in mats], device), a_had, _ptrs(sv, device),
                   None, None, bits, -1, mcg, False, -1, -1, 0, 1, i32([W] * S), _ptrs(cp, device), i32(stride), i32(src), 2)
    for o, r in zip(outs, ref):
        _close(o, r)


@pytest.mark.parametrize("bits", [4, 5])
@pytest.mark.parametrize("m", [1, 16])
@torch.inference_mode()
def test_uniform_bundle_unchanged(device, bits, m):
    """The plain (unsliced) K/V-style bundle: same result as the single GEMMs"""
    torch.manual_seed(m)
    k, n, mcg = 4096, 1024, bits >= 5
    mats = [_weights(k, n, bits, device) for _ in range(2)]
    x = (torch.randn(m, k, device = device) * 0.5).half()
    ref = [_single(x, B, suh, svh, mcg) for B, suh, svh in mats]
    a_had = torch.empty((2, m, k), dtype = torch.half, device = device)
    out = torch.empty((2, m, n), dtype = torch.half, device = device)
    ext.exl3_mgemm(
        x.view(1, m, k), _ptrs([B for B, _, _ in mats], device), out, _ptrs([s for _, s, _ in mats], device), a_had,
        _ptrs([v for _, _, v in mats], device), None, None, bits, -1, mcg, False, -1, -1, 0, 1, None, None,
    )
    for o, r in zip(out, ref):
        _close(o, r)
