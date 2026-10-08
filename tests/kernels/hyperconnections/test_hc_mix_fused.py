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
