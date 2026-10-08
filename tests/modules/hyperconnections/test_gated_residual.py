"""
GatedResidual prefill mix (ext.gr_mix_tiled, hc_mix_tiled.cu): the tiled deterministic kernels must match the fp32
torch reference (GatedResidual._mix_ref) to half-precision tolerance on every proj padding class, row count and both
module forms, agree with the cuBLAS GEMM path to the same tolerance, and be bit-reproducible run to run and across
GPU architectures (the property the TP replicated-routing design relies on). Also: the fused decode pair, the
shape fallback, source-weight release, the per-call int8 table derivation, and the weighted-copy handoff between
linked sites.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.hyperconnections import GatedResidual

pytestmark = [
    pytest.mark.cc(8, 0),
    pytest.mark.skipif(not getattr(ext, "HAS_GR_MIX_TILED", False), reason = "build without the tiled mix kernels"),
]

CASES = [
    # (D, rank, use_combine): padded proj rows 64 (NW 1), 128, 384 (Qwen3.8 shape), 512
    (256, 64, False),
    (256, 64, True),
    (1024, 320, True),
    (512, 512, False),
    (512, 448, True),
]
ROWS = [33, 64, 100, 257, 1000, 2048]


def make_site(device, D, rank, use_combine, seed = 0):
    torch.manual_seed(seed)
    H = 4
    m = GatedResidual(config = None, key = "site", hc_mult = H, hidden_size = D,
                      rms_norm_eps = 1e-6, use_combine = use_combine)
    m.device = device
    m.norm_w_raw = (torch.randn(H * D, device = device) * 0.1)
    down = torch.randn(rank, H * D, device = device) * (1.0 / (H * D) ** 0.5)
    up = torch.randn(H * D, rank, device = device) * (1.0 / rank ** 0.5)
    inject = torch.randn(H, H * D, device = device) * (1.0 / (H * D) ** 0.5) if use_combine else None
    m._prepare(down, up, inject, keep_source_weights = True)
    return m


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.mark.parametrize("D, rank, use_combine", CASES)
def test_tiled_matches_reference_and_gemm(device, D, rank, use_combine):
    m = make_site(device, D, rank, use_combine)
    assert m.tiled
    for R in ROWS:
        torch.manual_seed(R)
        x = torch.randn(1, R, 4, D, device = device) * 3.0
        ref_post, ref_mixed = m._mix_ref(x)
        ref_mixed = ref_mixed.view(R, D)
        post, mixed = m._mix(x)
        m.tiled = False
        post_g, mixed_g = m._mix(x)
        m.tiled = True
        assert rel(mixed, ref_mixed) < 3e-3, R
        assert rel(mixed, mixed_g) < 3e-3, R
        if use_combine:
            assert rel(post, ref_post.view(R, 4)) < 1e-3, R
            assert rel(post, post_g) < 1e-3, R
        else:
            assert post is None


@pytest.mark.parametrize("D, rank", [(2560, 320), (4096, 512), (1024, 320)])
def test_fused_decode_matches_reference(device, D, rank):
    # The fused decode pair (R <= FUSED_MAX_R) on the Qwen3.8 shape and a larger one
    m = make_site(device, D, rank, True)
    for R in (1, 2, 5, 8):
        torch.manual_seed(R)
        x = torch.randn(1, R, 4, D, device = device) * 3.0
        ref_post, ref_mixed = m._mix_ref(x)
        post, mixed = m._mix(x, cached = False)
        assert rel(mixed, ref_mixed.view(R, D)) < 3e-3, R
        assert rel(post, ref_post.view(R, 4)) < 1e-3, R
        post2, mixed2 = m._mix(x, cached = False)
        assert torch.equal(mixed, mixed2) and torch.equal(post, post2), R


def test_tiled_is_deterministic(device):
    m = make_site(device, 1024, 320, True)
    for R in (100, 2048):
        x = torch.randn(1, R, 4, 1024, device = device) * 3.0
        post_a, mixed_a = m._mix(x)
        post_a, mixed_a = post_a.clone(), mixed_a.clone()
        for _ in range(3):
            # different workspaces and other work in between must not change a bit
            torch.randn(2048, 2048, device = device) @ torch.randn(2048, 2048, device = device)
            post_b, mixed_b = m._mix(x)
            assert torch.equal(mixed_a, mixed_b)
            assert torch.equal(post_a, post_b)


def test_shape_fallback(device):
    # rank not a multiple of 64 -> cuBLAS path, still correct
    m = make_site(device, 256, 96, True)
    assert not m.tiled
    x = torch.randn(1, 200, 4, 256, device = device)
    ref_post, ref_mixed = m._mix_ref(x)
    post, mixed = m._mix(x)
    assert rel(mixed, ref_mixed.view(200, 256)) < 3e-3
    assert rel(post, ref_post.view(200, 4)) < 1e-3


@pytest.mark.multi_gpu(2)
def test_cross_device_identity(devices):
    # The int8 tensor-core path must agree bit for bit across GPU architectures
    torch.manual_seed(5)
    D, rank = 1024, 320
    H = 4
    norm = torch.randn(H * D) * 0.1
    down = torch.randn(rank, H * D) / (H * D) ** 0.5
    up = torch.randn(H * D, rank) / rank ** 0.5
    inject = torch.randn(H, H * D) / (H * D) ** 0.5
    devs = [d for d in devices if torch.cuda.get_device_capability(d) >= (8, 0)]
    if len(devs) < 2:
        pytest.skip("needs two devices of compute capability >= 8.0")
    for R in (100, 1000):
        x = torch.randn(1, R, H, D) * 3.0
        outs = []
        for dev in devs:
            with torch.cuda.device(dev):
                m = GatedResidual(config = None, key = "site", hc_mult = H, hidden_size = D,
                                  rms_norm_eps = 1e-6, use_combine = True)
                m.device = dev
                m.norm_w_raw = norm.to(dev)
                m._prepare(down.to(dev), up.to(dev), inject.to(dev), keep_source_weights = True)
                post, mixed = m._mix(x.to(dev))
                outs.append((post.cpu(), mixed.cpu()))
        for i in range(1, len(devs)):
            name = torch.cuda.get_device_name(devs[i])
            assert torch.equal(outs[0][0], outs[i][0]), (R, name)
            assert torch.equal(outs[0][1], outs[i][1]), (R, name)


def test_source_weights_released(device):
    # Inference loads drop the fp16 sources once the kernel tables exist; conversion keeps them
    m = make_site(device, 256, 64, True)
    assert m.down_h is not None
    torch.manual_seed(0)
    m2 = GatedResidual(config = None, key = "site", hc_mult = 4, hidden_size = 256, rms_norm_eps = 1e-6,
                       use_combine = True)
    m2.device = device
    m2.norm_w_raw = torch.randn(4 * 256, device = device) * 0.1
    m2._prepare(torch.randn(64, 1024, device = device), torch.randn(1024, 64, device = device),
                torch.randn(4, 1024, device = device))
    assert m2.tiled
    # One table set serves both kernel paths: the fp16 projection stays (the decode kernel reads it and the tiled
    # path derives its int8 tables per call), the checkpoint-layout up goes (the repacked copy remains)
    assert m2.up_h is None
    assert m2.proj_h is not None and m2.upx_h is not None
    assert not any(k for k in vars(m2) if k in ("fn_h", "proj_i8", "up_i8"))
    x = torch.randn(1, 100, 4, 256, device = device)
    m2._mix(x)                                   # tiled path still works
    m2._mix(x[:, :8])                            # fused decode path too
    with pytest.raises(AssertionError):
        m2.get_tensors()
    with pytest.raises(AssertionError):
        m2._mix_ref(x)


def test_tiled_tables_match_stored(device):
    # The per-call int8 derivation reproduces what a load-time copy would hold, bit for bit
    m = make_site(device, 1024, 320, True)

    def ws(shape, dtype):
        return torch.empty(shape, dtype = dtype, device = device)

    a = m._tiled_tables(ws)
    b = m._tiled_tables(ws)
    Mpad = m.proj_h.shape[0]
    ref_i8 = torch.empty((2, Mpad, 4 * 1024), dtype = torch.int8, device = device)
    ref_sb = torch.empty((Mpad,), dtype = torch.float, device = device)
    ext.det_quant_weight(m.proj_h, ref_i8, ref_sb)
    up_i8 = torch.empty((2, 4 * 1024, 320), dtype = torch.int8, device = device)
    up_sb = torch.empty((4 * 1024,), dtype = torch.float, device = device)
    ext.det_quant_weight(m.up_h, up_i8, up_sb)
    for x, y in zip(a, (ref_i8, ref_sb, up_i8, up_sb)):
        assert torch.equal(x, y)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.parametrize("D, rank", [(1024, 320), (2560, 320)])     # generic and shape-templated dots kernels
def test_weighted_copy_handoff(device, D, rank):
    # apply_ of one site emits the next site's weighted stream copy; the next mix must use it and agree with the
    # in-kernel weighting to fp32 rounding, and both with the reference
    a, b = make_site(device, D, rank, True, seed = 1), make_site(device, D, rank, True, seed = 2)
    GatedResidual.link_sites([a, b])
    assert a.next_site is b and b.next_site is None
    for R in (1, 3, 8):
        torch.manual_seed(R)
        x = torch.randn(1, R, 4, D, device = device) * 3.0
        y = torch.randn(1, R, D, device = device, dtype = torch.half)
        post = torch.rand(1, R, 4, device = device) * 2
        params = {}
        x_ref = x + post.unsqueeze(-1) * y.float().unsqueeze(-2)
        a.apply_(x, y, post, None, params)
        assert rel(x, x_ref) < 1e-6            # (the kernel's update contracts to an FMA)
        ent = params["gr_weighted"]
        assert ent[0] is b
        assert torch.equal(ent[2], x.view(R, 4, D) * b.w_h.float().view(1, 4, D))
        post_h, _, mixed_h = (t.clone() if t is not None else None for t in b.mix(x, params))   # consumes the copy
        assert "gr_weighted" not in params
        post_k, _, mixed_k = (t.clone() if t is not None else None for t in b.mix(x, {}))       # in-kernel weighting
        ref_post, ref_mixed = b._mix_ref(x)
        assert rel(mixed_h, ref_mixed.view(R, D)) < 3e-3, R
        assert rel(post_h.view(R, 4), ref_post.view(R, 4)) < 1e-3, R
        assert rel(mixed_h, mixed_k) < 1e-3, R
        assert rel(post_h, post_k) < 1e-4, R
        # and the kernel really reads the copy: a poisoned one changes the gates
        a.apply_(x, torch.zeros_like(y), post, None, params)
        params["gr_weighted"][2].mul_(0.0)
        post_p, _, _ = b.mix(x, params)
        assert rel(post_p, post_k) > 1e-2, R
    # Prefill row counts take the tiled path and emit no copy
    R = 100
    x = torch.randn(1, R, 4, D, device = device)
    y = torch.randn(1, R, D, device = device, dtype = torch.half)
    params = {}
    a.apply_(x, y, torch.rand(1, R, 4, device = device), None, params)
    assert "gr_weighted" not in params


def test_weighted_copy_rejected_when_stale(device):
    # A copy for another site, another stream tensor or another device is ignored (and consumed), and the mix
    # falls back to the in-kernel weighting
    D, rank = 1024, 320
    a, b = make_site(device, D, rank, True, seed = 1), make_site(device, D, rank, True, seed = 2)
    GatedResidual.link_sites([a, b])
    R = 4
    x = torch.randn(1, R, 4, D, device = device) * 3.0
    y = torch.randn(1, R, D, device = device, dtype = torch.half)
    params = {}
    a.apply_(x, y, torch.rand(1, R, 4, device = device), None, params)
    ref_post, ref_mixed = b._mix_ref(x)
    # same site, different stream tensor (a moved copy): must not use the entry
    x2 = x.clone()
    xw = params["gr_weighted"][2]
    xw.fill_(0)                                   # a used copy would give garbage
    post, _, mixed = b.mix(x2, params)
    assert "gr_weighted" not in params
    assert rel(mixed, ref_mixed.view(R, D)) < 3e-3
    # entry addressed to another site
    a.apply_(x, y, torch.zeros(1, R, 4, device = device), None, params)
    params["gr_weighted"][2].fill_(0)
    post, _, mixed = a.mix(x, params)
    assert "gr_weighted" not in params
    assert rel(mixed, a._mix_ref(x)[1].view(R, D)) < 3e-3


def test_link_sites_chain(device):
    class Block:
        def __init__(self, attn_hc, mlp_hc):
            self.attn_hc, self.mlp_hc = attn_hc, mlp_hc

    class Other:
        pass

    s = [make_site(device, 256, 64, True, seed = i) for i in range(6)]
    mixer = make_site(device, 256, 64, False, seed = 9)
    GatedResidual.link_sites([Other(), Block(s[0], s[1]), Block(s[2], s[3]), Other(), Block(s[4], s[5]), mixer])
    assert s[0].next_site is s[1] and s[1].next_site is s[2] and s[2].next_site is s[3]
    assert s[3].next_site is None              # the foreign module breaks the chain
    assert s[4].next_site is s[5] and s[5].next_site is mixer and mixer.next_site is None
