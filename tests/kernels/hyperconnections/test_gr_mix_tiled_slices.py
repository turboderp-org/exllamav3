"""
ext.gr_mix_tiled_slices(R, D, Mpad) (hc_mix_tiled.cu): the number S of K slices the tiled GatedResidual mix splits
its dots GEMM into, which GatedResidual._mix uses to size the (S, Rpad, Mpad) / (S, Rpad) partial-sum workspaces of
ext.gr_mix_tiled.

Contract:
- S = 4 q (one set of q slices per stream, H = 4), q divides D and each slice D / q is a multiple of the 128-wide
  stage; S is a pure function of (R, D, Mpad) (no device state, so the slice split, and with it the fixed reduction
  order, is the same on every GPU) and does not grow with R (more row blocks need fewer slices to fill the grid)
- it is exactly what gr_mix_tiled uses: workspaces of S slices are accepted, S - 1 rejected, and with an oversized
  sentinel-filled workspace slice S onward stays untouched and the result is bit-identical to the exact-size one
- with those workspaces gr_mix_tiled matches the fp32 torch reference GatedResidual._mix_ref (the module test's
  tolerances: 3e-3 of max |mixed| for the half output, 1e-3 for post)
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.hyperconnections import GatedResidual

pytestmark = [
    pytest.mark.cc(8, 0),
    pytest.mark.skipif(not getattr(ext, "HAS_GR_MIX_TILED", False), reason = "build without the tiled mix kernels"),
]


def test_slices_properties(device):
    for D in (128, 256, 512, 640, 1024, 1280, 2048, 2560, 4096, 5120, 7168):
        for Mpad in (64, 128, 384, 512):
            prev = None
            for R in (1, 9, 33, 64, 65, 100, 257, 1000, 2048, 8192, 65536):
                S = ext.gr_mix_tiled_slices(R, D, Mpad)
                assert S == ext.gr_mix_tiled_slices(R, D, Mpad)
                assert S % 4 == 0 and S > 0
                q = S // 4
                assert D % q == 0 and (D // q) % 128 == 0, (R, D, Mpad, S)
                assert prev is None or S <= prev, (R, D, Mpad, S, prev)
                prev = S


def make_site(device, D, rank, use_combine, seed = 0):
    torch.manual_seed(seed)
    H = 4
    m = GatedResidual(config = None, key = "site", hc_mult = H, hidden_size = D, rms_norm_eps = 1e-6,
                      use_combine = use_combine)
    m.device = device
    m.norm_w_raw = torch.randn(H * D, device = device) * 0.1
    down = torch.randn(rank, H * D, device = device) * (1.0 / (H * D) ** 0.5)
    up = torch.randn(H * D, rank, device = device) * (1.0 / rank ** 0.5)
    inject = torch.randn(H, H * D, device = device) * (1.0 / (H * D) ** 0.5) if use_combine else None
    m._prepare(down, up, inject, keep_source_weights = True)
    assert m.tiled
    return m


def _run(m, s3, S_ws, fill = None):
    """gr_mix_tiled with S_ws-slice partial workspaces (sentinel-filled if fill is given)"""
    R, H, D = s3.shape
    dev = s3.device
    Mpad = m.proj_h.shape[0]
    Rpad = -(-R // 64) * 64
    ws = lambda shape, dtype: torch.empty(shape, dtype = dtype, device = dev)
    proj_i8, proj_sb, up_i8, up_sb = m._tiled_tables(ws)
    dm = torch.full((S_ws, Rpad, Mpad), fill if fill is not None else 0.0, device = dev)
    ss = torch.full((S_ws, Rpad), fill if fill is not None else 0.0, device = dev)
    post = torch.empty((R, H), device = dev) if m.use_combine else None
    mixed = torch.empty((R, D), dtype = torch.half, device = dev)
    ext.gr_mix_tiled(s3, m.w_h, proj_i8, proj_sb, up_i8, up_sb, m.rms_eps, m.proj_m, dm, ss, ws((R, H), torch.float),
                     ws((2, R, m.rank), torch.int8), ws((R, m.rank // 64), torch.float), post, mixed)
    return post, mixed, dm, ss


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.mark.parametrize("D, rank, use_combine", [(256, 64, True), (1024, 320, True), (512, 448, False)])
@pytest.mark.parametrize("R", [33, 257, 2048])
@torch.inference_mode()
def test_slices_size_the_workspace(device, D, rank, use_combine, R):
    m = make_site(device, D, rank, use_combine)
    torch.manual_seed(R)
    x = torch.randn(R, 4, D, device = device) * 3.0
    S = ext.gr_mix_tiled_slices(R, D, m.proj_h.shape[0])

    post, mixed, _, _ = _run(m, x, S)
    ref_post, ref_mixed = m._mix_ref(x.view(1, R, 4, D))
    assert _rel(mixed, ref_mixed.view(R, D)) < 3e-3
    if use_combine:
        assert _rel(post, ref_post.view(R, 4)) < 1e-3

    sentinel = 12345.0
    post_b, mixed_b, dm, ss = _run(m, x, S + 1, fill = sentinel)
    assert (dm[S:] == sentinel).all() and (ss[S:] == sentinel).all(), "kernel used more than S slices"
    assert torch.equal(mixed, mixed_b)
    if use_combine:
        assert torch.equal(post, post_b)

    with pytest.raises(RuntimeError, match = "dm_part workspace"):
        _run(m, x, S - 1)


def assert_device_ok(device):
    # A failed launch would leave an error for the next op on the device
    torch.cuda.synchronize(device)
    assert torch.ones(8, device = device).sum().item() == 8


@torch.inference_mode()
def test_tiled_empty_rows(device):
    # No rows: nothing to do, outputs untouched
    D, rank = 256, 64
    m = make_site(device, D, rank, True)
    s3 = torch.empty((0, 4, D), device = device)
    S = ext.gr_mix_tiled_slices(0, D, m.proj_h.shape[0])
    ws = lambda shape, dtype: torch.empty(shape, dtype = dtype, device = device)
    proj_i8, proj_sb, up_i8, up_sb = m._tiled_tables(ws)
    Mpad = m.proj_h.shape[0]
    pbuf = torch.full((64,), 7.0, device = device)
    mbuf = torch.full((64,), 7.0, dtype = torch.half, device = device)
    ext.gr_mix_tiled(s3, m.w_h, proj_i8, proj_sb, up_i8, up_sb, m.rms_eps, m.proj_m, ws((S, 0, Mpad), torch.float),
                     ws((S, 0), torch.float), ws((0, 4), torch.float), ws((2, 0, rank), torch.int8),
                     ws((0, rank // 64), torch.float), pbuf[8:8].view(0, 4), mbuf[8:8].view(0, D))
    assert (pbuf == 7.0).all() and (mbuf == 7.0).all()
    assert_device_ok(device)


@pytest.mark.parametrize("case", ["D", "LR"])
@torch.inference_mode()
def test_tiled_empty_dims_rejected(device, case):
    # D == 0: the per-stream RMS norm is over an empty vector. LR == 0 passes LR % 64 but the tiled kernels need a
    # positive rank (the generic gr_mix takes LR == 0)
    R, H = 4, 4
    D, LR = (0, 64) if case == "D" else (256, 0)
    HD, M_, Mpad = H * D, LR + H, 128 if LR else 64
    ws = lambda *shape, dtype = torch.float: torch.zeros(shape, dtype = dtype, device = device)
    S = ext.gr_mix_tiled_slices(R, max(D, 128), Mpad)
    match = "norm over an empty stream dimension" if case == "D" else "LR must be a positive multiple of 64"
    with pytest.raises(RuntimeError, match = match):
        ext.gr_mix_tiled(ws(R, H, D), ws(HD, dtype = torch.half), ws(2, Mpad, HD, dtype = torch.int8), ws(Mpad),
                         ws(2, HD, LR, dtype = torch.int8), ws(HD), 1e-6, M_, ws(S, 64, Mpad), ws(S, 64), ws(R, H),
                         ws(2, R, LR, dtype = torch.int8), ws(R, LR // 64), ws(R, H), ws(R, D, dtype = torch.half))
    assert_device_ok(device)
