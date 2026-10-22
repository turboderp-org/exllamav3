"""
gr_mix (hc_mix.cu): the GatedResidual decode mix against a plain torch reference of its
documented math. The Qwen4Exp residual block mixes H = 4 streams per site:

    rmr[h]      = rsqrt(mean(streams[h]^2) + eps)
    dots[r, i]  = sum_h (streams * w)[r, h, d] * fn[i, h, d]        (fn in half)
    dots[r, M]  = raw per-stream sums of squares of streams
    t[i]        = silu(sum_h rmr[h] * dots[r, i, h] / H)
    post[h]     = 2 * sigmoid(sum_h rmr[h] * dots[r, LR + h, h] / H)
    g[h, d]     = sigmoid(sum_i up[h, d, i] * t[i])                  (up = upt repacked)
    mixed[d]    = sum_h g[h, d] * streams[h, d] * rmr[h] * w[h * D + d] / H

With a pre-weighted stream copy (wstreams, emitted by the previous site's hc_apply) the dots
read the copy while the mixed output still scales the raw streams. mixed is fp16 or fp32.
The reference follows that arithmetic exactly; the kernel computes dots and the norm in fp16,
so a wrong index, stream or coefficient is an O(1) error while agreement sits at a few fp16
unit roundoffs. The same contract is covered at the tile boundary by
test_gr_mix_tiled_slices.py and at the module level by
modules/hyperconnections/test_gated_residual.py.
"""

import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext

H = 4


def _gr_mix_reference(streams, fn, upt, w, rms_eps, has_post, wstreams = None):
    rows, _, d = streams.shape
    lr = upt.shape[2]
    m = lr + (H if has_post else 0)
    rmr = torch.rsqrt(streams.square().mean(-1) + rms_eps)                      # (R, H)
    dot_src = wstreams if wstreams is not None else streams * w.float().view(1, H, d)
    dots = torch.einsum("rhd,jhd->rjh", dot_src, fn.view(m, H, d).float())       # (R, M, H)
    t = F.silu((rmr.unsqueeze(1) * dots[:, :lr, :]).sum(-1) / H)                 # (R, LR)
    up = upt.permute(0, 1, 3, 2).reshape(H, d, lr).float()                       # (H, D, LR)
    g = torch.sigmoid(torch.einsum("ri,hdi->rhd", t, up))                        # (R, H, D)
    post = None
    if has_post:
        post_dots = (rmr.unsqueeze(1) * dots[:, lr:, :]).sum(-1) / H             # (R, H)
        post = 2 * torch.sigmoid(post_dots)
    coef = rmr.unsqueeze(-1) * w.float().view(H, d)                              # (R, H, D)
    mixed = (g * streams * coef).sum(1) / H
    return post, mixed


def _gr_mix_args(rows, d, lr, device, *, has_post = True, mixed_dtype = torch.float16):
    m = lr + (H if has_post else 0)
    gen = torch.Generator().manual_seed(5201 + rows + int(has_post) + d)
    streams = (torch.randn((rows, H, d), generator = gen) * 0.25).to(device)
    fn = ((torch.randn((m, H * d), generator = gen) * 0.05).half()).to(device)
    upt = ((torch.randn((H, d // 4, lr, 4), generator = gen) * 0.08).half()).to(device)
    w = ((torch.randn((H * d,), generator = gen) * 0.1).half()).to(device)
    dots = torch.empty((rows, m + 1, H), device = device)
    post = torch.empty((rows, H), device = device) if has_post else None
    mixed = torch.empty((rows, d), dtype = mixed_dtype, device = device)
    return streams, fn, upt, w, dots, post, mixed


@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("has_post", [False, True], ids = ["final", "site"])
@torch.inference_mode()
def test_gr_mix_matches_reference(device, rows, has_post):
    d, lr = 2560, 320
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, device, has_post = has_post)

    ext.gr_mix(streams, None, fn, upt, w, 1e-5, dots, post, mixed)

    expected_post, expected_mixed = _gr_mix_reference(streams, fn, upt, w, 1e-5, has_post)
    if has_post:
        torch.testing.assert_close(post, expected_post, rtol = 2e-4, atol = 2e-4)
    assert mixed.dtype == torch.float16
    torch.testing.assert_close(mixed, expected_mixed.half(), rtol = 2e-3, atol = 2e-3)


# With a pre-weighted stream copy (emitted by the previous site's hc_apply) the dots read the
# copy while the mixed output still scales the raw streams
@torch.inference_mode()
def test_gr_mix_uses_weighted_stream_copy(device):
    rows, d, lr = 2, 512, 128
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, device, has_post = True)
    wstreams = streams * w.float().view(1, H, d)

    ext.gr_mix(streams, wstreams, fn, upt, w, 1e-5, dots, post, mixed)

    expected_post, expected_mixed = _gr_mix_reference(
        streams, fn, upt, w, 1e-5, True, wstreams = wstreams
    )
    torch.testing.assert_close(post, expected_post, rtol = 2e-4, atol = 2e-4)
    torch.testing.assert_close(mixed, expected_mixed.half(), rtol = 2e-3, atol = 2e-3)


@torch.inference_mode()
def test_gr_mix_float_output(device):
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, device, has_post = True,
                                                         mixed_dtype = torch.float32)
    ext.gr_mix(streams, None, fn, upt, w, 1e-5, dots, post, mixed)
    expected_post, expected_mixed = _gr_mix_reference(streams, fn, upt, w, 1e-5, True)
    assert mixed.dtype == torch.float32
    torch.testing.assert_close(post, expected_post, rtol = 2e-4, atol = 2e-4)
    torch.testing.assert_close(mixed, expected_mixed, rtol = 2e-3, atol = 2e-3)


# Host-side validation: the shape contracts gr_mix enforces before launch
@torch.inference_mode()
def test_gr_mix_rejects_bad_fn_rows(device):
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, device, has_post = True)
    fn_bad = fn[:-1].contiguous()
    with pytest.raises(RuntimeError, match = "fn rows must be LR"):
        ext.gr_mix(streams, None, fn_bad, upt, w, 1e-5, dots, post, mixed)


@torch.inference_mode()
def test_gr_mix_rejects_bad_upt_layout(device):
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, device, has_post = True)
    upt_bad = upt.reshape(H, d // 4, 4, lr).contiguous()
    with pytest.raises(RuntimeError, match = "repacked layout"):
        ext.gr_mix(streams, None, fn, upt_bad, w, 1e-5, dots, post, mixed)


@torch.inference_mode()
def test_gr_mix_rejects_bad_dots_shape(device):
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, device, has_post = True)
    dots_bad = dots[:, :-1].contiguous()
    with pytest.raises(RuntimeError, match = "dots shape"):
        ext.gr_mix(streams, None, fn, upt, w, 1e-5, dots_bad, post, mixed)


@torch.inference_mode()
def test_gr_mix_rejects_bad_wstreams_shape(device):
    rows, d, lr = 2, 256, 64
    streams, fn, upt, w, dots, post, mixed = _gr_mix_args(rows, d, lr, device, has_post = True)
    wstreams = torch.randn((rows + 1, H, d), device = device)
    with pytest.raises(RuntimeError, match = "wstreams shape"):
        ext.gr_mix(streams, wstreams, fn, upt, w, 1e-5, dots, post, mixed)
