"""
mHC launch-count folds (hc_mix_fused, hc_fuse.cuh) against the unfused launches they replace: a pending
hc_apply run inside the next mix's partials kernel, and the block's RMSNorm run inside the mix's finalize.
Both reproduce the unfused kernels' arithmetic in the same order, so post / comb / collapsed / the normed
output and the updated streams must be bit-identical, at every decode row count the Python side folds
(R <= 32) and for every input/weight dtype combination the fold accepts.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

H = 4
M = 2 * H + H * H


def _mix_inputs(R, D, fn_half, gen, device):
    streams = (torch.randn((R, H, D), generator = gen) * 0.7).float().to(device)
    fn = (torch.randn((M, H * D), generator = gen) * 0.02).to(device)
    fn = fn.half() if fn_half else fn.float()
    base = (torch.randn((M,), generator = gen) * 0.5).float().to(device)
    scale = torch.tensor([0.9, 1.1, 0.8]).float().to(device)
    return streams, fn, base, scale


def _run_mix(streams, fn, base, scale, rms_eps, hc_eps, iters):
    R, _, D = streams.shape
    device = streams.device
    chunks = ext.hc_mix_num_chunks(R, H * D)
    partials = torch.empty((R, chunks, M + 1), dtype = torch.float, device = device)
    post = torch.empty((R, H), dtype = torch.float, device = device)
    comb = torch.empty((R, H, H), dtype = torch.float, device = device)
    collapsed = torch.empty((R, D), dtype = torch.half, device = device)
    ext.hc_mix(streams, fn, base, scale, rms_eps, hc_eps, iters, partials, post, comb, collapsed)
    return post, comb, collapsed


def _run_fused(streams, pend, fn, base, scale, rms_eps, hc_eps, iters, norm):
    R, _, D = streams.shape
    device = streams.device
    chunks = ext.hc_mix_num_chunks(R, H * D)
    partials = torch.empty((R, chunks, M + 1), dtype = torch.float, device = device)
    post = torch.empty((R, H), dtype = torch.float, device = device)
    comb = torch.empty((R, H, H), dtype = torch.float, device = device)
    collapsed = torch.empty((R, D), dtype = torch.half, device = device)
    normed = torch.empty((R, D), dtype = torch.half, device = device) if norm is not None else None
    y, ppost, pcomb = pend if pend else (None, None, None)
    w, eps, bias, cscale = norm if norm is not None else (None, 0.0, 0.0, 1.0)
    ext.hc_mix_fused(streams, y, ppost, pcomb, fn, base, scale, rms_eps, hc_eps, iters, partials, post, comb,
                     collapsed, w, normed, eps, bias, cscale)
    return post, comb, collapsed, normed


@pytest.mark.parametrize("R", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("D", [1024, 2048, 4096])
@pytest.mark.parametrize("fn_half", [True, False])
@pytest.mark.parametrize("y_half", [True, False])
def test_apply_fold(device, R, D, fn_half, y_half):
    """A pending apply folded into the partials kernel vs hc_apply followed by hc_mix"""
    gen = torch.Generator().manual_seed(R * 7919 + D)
    streams, fn, base, scale = _mix_inputs(R, D, fn_half, gen, device)
    y = (torch.randn((R, D), generator = gen) * 0.5).to(device)
    y = y.half() if y_half else y.float()
    ppost = (torch.rand((R, H), generator = gen) * 2).float().to(device)
    pcomb = torch.softmax(torch.randn((R, H, H), generator = gen), dim = -1).float().to(device)

    ref_streams = streams.clone()
    ext.hc_apply(ref_streams, y, ppost, pcomb, None, None)
    ref = _run_mix(ref_streams, fn, base, scale, 1e-6, 1e-4, 20)

    fused_streams = streams.clone()
    out = _run_fused(fused_streams, (y, ppost, pcomb), fn, base, scale, 1e-6, 1e-4, 20, None)
    torch.cuda.synchronize(device)
    assert torch.equal(fused_streams, ref_streams), "streams after the folded apply"
    for name, a, b in zip(("post", "comb", "collapsed"), ref, out):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("R", [1, 3, 32])
@pytest.mark.parametrize("D", [1024, 2048, 4096])
@pytest.mark.parametrize("w_kind", ["half", "bf16", "none"])
@pytest.mark.parametrize("bias_scale", [(0.0, 1.0), (1.0, 1.0), (0.0, 0.5)])
def test_norm_fold(device, R, D, w_kind, bias_scale):
    """The RMSNorm folded into the finalize vs hc_mix followed by rms_norm"""
    gen = torch.Generator().manual_seed(R * 104729 + D + len(w_kind))
    streams, fn, base, scale = _mix_inputs(R, D, True, gen, device)
    w = None
    if w_kind != "none":
        w = (torch.randn((D,), generator = gen) * 0.3 + 1.0).to(device)
        w = w.half() if w_kind == "half" else w.bfloat16()
    eps = 1e-6
    bias, cscale = bias_scale

    ref = _run_mix(streams, fn, base, scale, 1e-6, 1e-4, 20)
    ref_normed = torch.empty((R, D), dtype = torch.half, device = device)
    ext.rms_norm(ref[2], w, ref_normed, eps, bias, cscale, False, False, 1)

    out = _run_fused(streams.clone(), None, fn, base, scale, 1e-6, 1e-4, 20, (w, eps, bias, cscale))
    torch.cuda.synchronize(device)
    for name, a, b in zip(("post", "comb", "collapsed"), ref, out):
        assert torch.equal(a, b), name
    assert torch.equal(ref_normed, out[3]), "normed"


def test_both_folds(device):
    """Apply and norm folded in the same call"""
    R, D = 4, 2048
    gen = torch.Generator().manual_seed(5)
    streams, fn, base, scale = _mix_inputs(R, D, True, gen, device)
    y = (torch.randn((R, D), generator = gen) * 0.5).half().to(device)
    ppost = (torch.rand((R, H), generator = gen) * 2).float().to(device)
    pcomb = torch.softmax(torch.randn((R, H, H), generator = gen), dim = -1).float().to(device)
    w = (torch.randn((D,), generator = gen) * 0.3 + 1.0).half().to(device)

    ref_streams = streams.clone()
    ext.hc_apply(ref_streams, y, ppost, pcomb, None, None)
    ref = _run_mix(ref_streams, fn, base, scale, 1e-6, 1e-4, 20)
    ref_normed = torch.empty((R, D), dtype = torch.half, device = device)
    ext.rms_norm(ref[2], w, ref_normed, 1e-6, 0.0, 1.0, False, False, 1)

    fused_streams = streams.clone()
    out = _run_fused(fused_streams, (y, ppost, pcomb), fn, base, scale, 1e-6, 1e-4, 20, (w, 1e-6, 0.0, 1.0))
    torch.cuda.synchronize(device)
    assert torch.equal(fused_streams, ref_streams)
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    assert torch.equal(ref_normed, out[3])


# Empty inputs (all of hc_mix.cu's bindings): no rows is a no-op with nothing written; a zero stream width D makes
# the RMS norm of the (flattened) streams an empty-vector norm, undefined, so the mixes raise; hc_apply (elementwise
# over D) is a no-op at D == 0 too

def assert_device_ok(device):
    # A failed launch would leave an error for the next op on the device
    torch.cuda.synchronize(device)
    assert torch.ones(8, device = device).sum().item() == 8


def _sentinel(shape, device, dtype = torch.float):
    buf = torch.full((64,), 7.0, dtype = dtype, device = device)
    return buf, buf[8:8].view(shape)


@pytest.mark.parametrize("op", ["hc_mix", "hc_head", "hc_mix_fused"])
@torch.inference_mode()
def test_mix_empty_rows(device, op):
    gen = torch.Generator().manual_seed(0)
    R, D = 0, 1024
    streams, fn, base, scale = _mix_inputs(R, D, True, gen, device)
    chunks = ext.hc_mix_num_chunks(R, H * D)
    assert chunks > 0
    bufs = []
    def out(shape, dtype = torch.float):
        b, v = _sentinel(shape, device, dtype)
        bufs.append(b)
        return v
    if op == "hc_mix":
        ext.hc_mix(streams, fn, base, scale, 1e-6, 1e-6, 20, out((R, chunks, M + 1)), out((R, H)), out((R, H, H)),
                   out((R, D), torch.half))
    elif op == "hc_head":
        ext.hc_head(streams, fn[:H], base[:H], scale[:1], 1e-6, 1e-6, out((R, chunks, H + 1)), out((R, D)))
    else:
        y = torch.empty((R, D), dtype = torch.half, device = device)
        ext.hc_mix_fused(streams, y, out((R, H)), out((R, H, H)), fn, base, scale, 1e-6, 1e-6, 20,
                         out((R, chunks, M + 1)), out((R, H)), out((R, H, H)), out((R, D), torch.half),
                         torch.ones(D, dtype = torch.half, device = device), out((R, D), torch.half), 1e-6, 0.0, 1.0)
        # Fold validation still applies: folds need the half-output mix
        with pytest.raises(RuntimeError, match = "folds are for the half-output mix"):
            ext.hc_mix_fused(streams, y, out((R, H)), out((R, H, H)), fn, base, scale, 1e-6, 1e-6, 20,
                             out((R, chunks, M + 1)), out((R, H)), out((R, H, H)), out((R, D)),
                             None, out((R, D), torch.half), 1e-6, 0.0, 1.0)
    assert all((b == 7.0).all() for b in bufs)
    assert_device_ok(device)


@pytest.mark.parametrize("op", ["hc_mix", "hc_head", "hc_mix_fused"])
@torch.inference_mode()
def test_mix_empty_dim_rejected(device, op):
    gen = torch.Generator().manual_seed(0)
    R, D = 3, 0
    streams, fn, base, scale = _mix_inputs(R, D, False, gen, device)
    assert ext.hc_mix_num_chunks(R, 0) == 0
    partials = torch.empty((R, 1, M + 1), device = device)
    e = lambda *s: torch.empty(s, device = device)
    with pytest.raises(RuntimeError, match = "hc_mix: norm over an empty stream dimension"):
        if op == "hc_mix":
            ext.hc_mix(streams, fn, base, scale, 1e-6, 1e-6, 20, partials, e(R, H), e(R, H, H), e(R, D))
        elif op == "hc_head":
            ext.hc_head(streams, fn[:H], base[:H], scale[:1], 1e-6, 1e-6, partials[..., :H + 1], e(R, D))
        else:
            ext.hc_mix_fused(streams, None, None, None, fn, base, scale, 1e-6, 1e-6, 20, partials, e(R, H),
                             e(R, H, H), e(R, D).half(), None, None, 0.0, 0.0, 1.0)
    # A partials workspace with no chunks cannot hold the mix (and would be a host division by zero)
    streams, fn, base, scale = _mix_inputs(R, 256, False, gen, device)
    with pytest.raises(RuntimeError, match = "partials workspace too small"):
        ext.hc_mix(streams, fn, base, scale, 1e-6, 1e-6, 20, torch.empty((R, 0, M + 1), device = device), e(R, H),
                   e(R, H, H), e(R, 256))
    assert_device_ok(device)


@pytest.mark.parametrize("R, D", [(0, 1024), (3, 0)])
@pytest.mark.parametrize("form", ["comb", "post_only", "xw"])
@torch.inference_mode()
def test_apply_empty(device, R, D, form):
    xbuf, x = _sentinel((R, H, D), device)
    xwbuf, xw = _sentinel((R, H, D), device)
    y = torch.empty((R, D), device = device)
    post = torch.zeros((R, H), device = device)
    comb = torch.zeros((R, H, H), device = device) if form == "comb" else None
    wn = torch.ones(H * D, dtype = torch.half, device = device) if form == "xw" else None
    ext.hc_apply(x, y, post, comb, wn, xw if form == "xw" else None)
    assert (xbuf == 7.0).all() and (xwbuf == 7.0).all()
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.hc_apply(x, y, post.half(), comb, wn, xw if form == "xw" else None)
    assert_device_ok(device)


def _gr_inputs(R, D, LR, post, device, seed = 0):
    g = torch.Generator().manual_seed(seed)
    M_ = LR + (H if post else 0)
    streams = torch.randn((R, H, D), generator = g).to(device)
    fn = (torch.randn((M_, H * D), generator = g) * max(H * D, 1) ** -0.5).half().to(device)
    upt = (torch.randn((H, D // 4, LR, 4), generator = g) * max(LR, 1) ** -0.5).half().to(device)
    w = (torch.rand((H * D,), generator = g) + 0.5).half().to(device)
    return streams, fn, upt, w, M_


@pytest.mark.parametrize("D, LR", [(1024, 64), (4096, 320)])      # the generic and the fast decode path
@torch.inference_mode()
def test_gr_mix_empty_rows(device, D, LR):
    streams, fn, upt, w, M_ = _gr_inputs(0, D, LR, True, device)
    dbuf, dots = _sentinel((0, M_ + 1, H), device)
    pbuf, post = _sentinel((0, H), device)
    mbuf, mixed = _sentinel((0, D), device, torch.half)
    ext.gr_mix(streams, None, fn, upt, w, 1e-6, dots, post, mixed)
    assert (dbuf == 7.0).all() and (pbuf == 7.0).all() and (mbuf == 7.0).all()
    with pytest.raises(RuntimeError, match = "gr_mix: norm over an empty stream dimension"):
        s, f, u, ww, _ = _gr_inputs(2, 0, LR, True, device)
        ext.gr_mix(s, None, f, u, ww, 1e-6, torch.empty((2, M_ + 1, H), device = device),
                   torch.empty((2, H), device = device), torch.empty((2, 0), dtype = torch.half, device = device))
    assert_device_ok(device)


@torch.inference_mode()
def test_gr_mix_zero_rank(device):
    # LR == 0: the up-gate logits are empty dot products, i.e. 0, so g = sigmoid(0) = 1/2 and
    # mixed = mean_h normed[h] / 2 with normed[h] = streams[h] * rsqrt(mean(streams[h]^2) + eps) * w[h]
    R, D, eps = 3, 256, 1e-6
    streams, fn, upt, w, M_ = _gr_inputs(R, D, 0, False, device)
    mixed = torch.empty((R, D), dtype = torch.float, device = device)
    ext.gr_mix(streams, None, fn, upt, w, eps, torch.empty((R, M_ + 1, H), device = device), None, mixed)
    s = streams.double()
    normed = s * torch.rsqrt(s.pow(2).mean(-1, keepdim = True) + eps) * w.double().view(H, D)
    torch.testing.assert_close(mixed.double(), 0.5 * normed.mean(1), rtol = 1e-5, atol = 1e-6)
    assert_device_ok(device)
