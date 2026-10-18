"""
Auto output scales: the sensitivity captured from gated MLPs during calibration, and the rule in regularize()
that drops output scales when the sensitive channels are the high-energy ones.

Output scales from the output-side Hessian (--out_scales yaqa): the scales derived from its diagonal, their effect
on where quantization error lands, and the diagonal's way from util/yaqa_hessians.py into the converter.

References: central differences through the activation kernels, explicit perturbation of a float64 MLP, the closed
form of the scales and the loss model they minimize, and the full output Hessian for the diagonal-only collector.
"""

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.mlp import (
    gated_act_slopes,
    capture_out_sensitivity,
    merge_out_sensitivity,
    finalize_out_sensitivity,
)
from exllamav3.modules.quant.exl3_lib import quantize as Q
from exllamav3.modules.quant.exl3_lib.quantize import (
    out_scales_ratio, out_scales_max_ratio, regularize, sensitivity_out_scales, quantize_tiles,
    blockwise_preapply_had_l_, blockwise_preapply_had_r_,
)
from testlib.repo_scripts import load_repo_script

KERNELS = {
    "silu": ext.silu_mul,
    "gelu": ext.gelu_mul,
    "relu2": ext.relu2_mul,
    "swiglu_oai": ext.silu_oai_mul,
}


@pytest.fixture(scope = "module")
def yaqa_hessians():
    return load_repo_script("util/yaqa_hessians.py")


class StubInner:
    def __init__(self, weight):
        self.weight = weight

    def get_weight_tensor(self):
        return self.weight


class StubLinear:
    def __init__(self, key, weight = None, qmap = "block.mlp.input"):
        self.key = key
        self.qmap = qmap
        self.inner = StubInner(weight)


def kernel(name, g, u, act_limit):
    a = torch.empty_like(g, dtype = torch.half)
    KERNELS[name](g.float().contiguous(), u.float().contiguous(), a, act_limit)
    return a.double()


@pytest.mark.parametrize("name", ["silu", "gelu", "relu2", "swiglu_oai"])
@pytest.mark.parametrize("act_limit", [0.0, 2.5])
@torch.inference_mode()
def test_slopes_follow_the_activation_kernels(device, name, act_limit):
    torch.manual_seed(0)
    g = (torch.randn(64, 256, device = device) * 1.5).float()
    u = (torch.randn(64, 256, device = device) * 1.5).float()
    du, dg = gated_act_slopes(name, g, u, act_limit)

    # Central differences through the kernels themselves. Their output is fp16, so each difference
    # carries the rounding of the two values it is taken between (two units in the last place
    # allowed: a value rounded up to a power of two has the coarser spacing above it)
    h = 0.125
    def diff(lo, hi):
        ulp = torch.exp2(torch.floor(torch.log2(torch.maximum(lo.abs(), hi.abs()).clamp(min = 2.0 ** -14))) - 10)
        return (hi - lo) / (2 * h), 2 * ulp / (2 * h)
    fd_u, tol_u = diff(kernel(name, g, u - h, act_limit), kernel(name, g, u + h, act_limit))
    fd_g, tol_g = diff(kernel(name, g - h, u, act_limit), kernel(name, g + h, u, act_limit))

    # Away from the kinks of the clamps and of relu, where a central difference straddles two slopes
    ok = g.abs() > 2 * h
    if act_limit:
        ok &= ((u.abs() - act_limit).abs() > 2 * h) & ((g - act_limit).abs() > 2 * h)
        if name != "swiglu_oai":
            f = gated_act_slopes(name, g, torch.ones_like(u), 0.0)[0]
            f_lo = gated_act_slopes(name, g - h, torch.ones_like(u), 0.0)[0]
            f_hi = gated_act_slopes(name, g + h, torch.ones_like(u), 0.0)[0]
            ok &= ((f_lo < act_limit) == (f_hi < act_limit)) & ((f - act_limit).abs() > 0.05)
    assert ok.float().mean() > 0.5

    # Curvature of the activation over the step
    curve = 0.02 * (1.0 + u.abs().double())
    assert ((du.double() - fd_u).abs() <= tol_u + 1e-4)[ok].all()
    assert ((dg.double() - fd_g).abs() <= tol_g + curve)[ok].all()
    assert du[ok].abs().mean() > 0.2 and dg[ok].abs().mean() > 0.2


@torch.inference_mode()
def test_unknown_activation_is_skipped(device):
    g = torch.randn(4, 128, device = device)
    assert gated_act_slopes("xielu", g, g, 0.0) is None
    capture = {}
    capture_out_sensitivity(capture, StubLinear("g"), StubLinear("u"), StubLinear("d"), g, g, "xielu", 0.0)
    assert capture == {}


@pytest.mark.parametrize("weighted", [False, True])
@torch.inference_mode()
def test_sensitivity_matches_perturbation(device, weighted):
    """The captured sensitivity is the mean square response of the MLP output to a unit error in one
    intermediate channel of the gate or up projection's output"""
    torch.manual_seed(1)
    rows, n, hidden = 96, 128, 64
    g = torch.randn(rows, n, device = device, dtype = torch.double) * 3.0
    u = torch.randn(rows, n, device = device, dtype = torch.double) * 2.0
    # Down projection in (in, out) layout, padded past the intermediate width
    wd = torch.randn(n + 32, hidden, device = device, dtype = torch.double) * torch.rand(n + 32, 1, device = device, dtype = torch.double)
    w = torch.rand(rows, device = device, dtype = torch.double) if weighted else None
    gate, up, down = StubLinear("gate"), StubLinear("up"), StubLinear("down", wd.float(), "block.mlp.down")

    # Two calls, then a second shard merged in
    capture, other = {}, {}
    a, b = rows // 3, 2 * rows // 3
    ww = (lambda s: w[s].float()) if weighted else (lambda s: None)
    capture_out_sensitivity(capture, gate, up, down, g[:a].float(), u[:a].float(), "silu", 0.0, ww(slice(0, a)))
    capture_out_sensitivity(capture, gate, up, down, g[a:b].float(), u[a:b].float(), "silu", 0.0, ww(slice(a, b)))
    capture_out_sensitivity(other, gate, up, down, g[b:].float(), u[b:].float(), "silu", 0.0, ww(slice(b, rows)))
    merge_out_sensitivity(capture, other)
    sens = finalize_out_sensitivity(capture)
    assert set(sens) == {"gate", "up"}

    def mlp_out(g_, u_):
        o = (F.silu(g_) * u_) @ wd[:n]
        return o if w is None else o * w[:, None]

    ref = mlp_out(g, u)
    eps = 1e-5
    for j in (0, 17, n - 1):
        e = torch.zeros(n, device = device, dtype = torch.double)
        e[j] = eps
        want_g = ((mlp_out(g + e, u) - ref) / eps).square().sum(1).mean().item()
        want_u = ((mlp_out(g, u + e) - ref) / eps).square().sum(1).mean().item()
        assert sens["gate"][j].item() == pytest.approx(want_g, rel = 2e-3)
        assert sens["up"][j].item() == pytest.approx(want_u, rel = 2e-3)
        assert abs(want_g - want_u) > 0.05 * max(want_g, want_u)


@torch.inference_mode()
def test_projections_without_qmap_are_skipped(device):
    g = torch.randn(8, 128, device = device)
    capture = {}
    down = StubLinear("down", torch.randn(128, 64, device = device))
    capture_out_sensitivity(capture, StubLinear("gate", qmap = None), StubLinear("up"), down, g, g, "silu", 0.0)
    assert set(capture) == {"up"}


@torch.inference_mode()
def test_ratio(device):
    torch.manual_seed(2)
    n = 512
    sigma = torch.rand(n, device = device) + 0.5
    # No relation between sensitivity and energy
    assert out_scales_ratio(sigma, torch.ones(n)) == pytest.approx(1.0, abs = 1e-6)
    # Relative error matters equally in every channel
    assert out_scales_ratio(sigma, 1.0 / sigma.square()) < 0.95
    # The sensitive channels are the high-energy ones
    hot = sigma.topk(8).indices
    s = torch.full((n,), 1e-3, device = device)
    s[hot] = 1.0
    r = out_scales_ratio(sigma, s)
    assert r == pytest.approx((sigma[hot].double().square().mean() / sigma.double().square().mean()).item(), rel = 0.05)
    assert r > 1.5
    # Padded tensors, uninformative sensitivities
    assert out_scales_ratio(sigma, s[:n - 64].cpu()) is not None
    assert out_scales_ratio(sigma, torch.zeros(n)) is None
    nan = torch.ones(n)
    nan[3] = float("nan")
    assert out_scales_ratio(sigma, nan) is None


def _regularize(weight, out_scales, sens, hdiag = None, q_fallback = False, full = False):
    device = weight.device
    k, n = weight.shape
    su = (torch.randn(k, 1, device = device).sign() + 1e-5).sign()
    sv = (torch.randn(1, n, device = device).sign() + 1e-5).sign()
    quant_args = {"apply_out_scales": out_scales, "K": 4}
    if sens is not None:
        quant_args["out_sensitivity"] = sens
    if hdiag is not None:
        quant_args["out_hessian_diag"] = hdiag
    H_diag = torch.ones(k, device = device)
    applied, weight_r, _, su, sv = regularize(
        weight.clone(), su, sv, quant_args, False, H_diag, None, skip_g_scale = True, q_fallback = q_fallback)
    if full:
        return applied, weight_r, su, sv, quant_args
    return applied, sv


@torch.inference_mode()
def test_regularize_decision(device):
    torch.manual_seed(3)
    k, n = 256, 512
    weight = torch.randn(k, n, device = device)
    hot = torch.arange(0, n, 64, device = device)
    weight[:, hot] *= 4.0

    s_hot = torch.full((n,), 1e-3)
    s_hot[hot.cpu()] = 1.0
    s_cold = torch.ones(n)
    s_cold[hot.cpu()] = 1e-3
    assert out_scales_ratio(weight.square().mean(0).sqrt(), s_hot) > out_scales_max_ratio

    # Auto: on unless the sensitivity says otherwise
    assert _regularize(weight, None, None)[0] is True
    assert _regularize(weight, None, s_cold)[0] is True
    assert _regularize(weight, None, torch.zeros(n))[0] is True
    applied, sv = _regularize(weight, None, s_hot)
    assert applied is False
    assert torch.equal(sv.abs(), torch.ones_like(sv))
    applied, sv = _regularize(weight, None, s_cold)
    assert sv.abs()[0, hot].min() > 2.0 * sv.abs().median()

    # Forced settings ignore the sensitivity
    assert _regularize(weight, True, s_hot)[0] is True
    assert _regularize(weight, False, s_cold)[0] is False


def predicted_loss(t, sigma, s):
    t, sigma, s = t.flatten().double(), sigma.flatten().double(), s.flatten().double()
    live = t > 0
    return ((s * t.square())[live].sum() * (sigma.square() / t.square())[live].sum()).item()


@torch.inference_mode()
def test_sensitivity_out_scales(device):
    torch.manual_seed(5)
    n = 1024
    sigma = (torch.rand(1, n, device = device) * 3.0 + 0.2)
    s = torch.randn(n).exp()
    t = sensitivity_out_scales(sigma, s)
    assert t.shape == sigma.shape and t.dtype == sigma.dtype
    assert t.mean().item() == pytest.approx(1.0, abs = 1e-5)

    # The closed form, on the floored sensitivity
    sf = (s / s.mean() + Q.out_sensitivity_floor).double().to(device)
    ref = (sigma.flatten().double().square() / sf).pow(0.25)
    assert torch.allclose(t.flatten().double(), ref / ref.mean(), rtol = 1e-5)

    # It is the minimum of the predicted loss: no worse than plain scales, no scales, or anything near it
    best = predicted_loss(t, sigma, sf)
    assert best < 0.9 * predicted_loss(sigma, sigma, sf)
    assert best < 0.9 * predicted_loss(torch.ones_like(sigma), sigma, sf)
    for _ in range(50):
        near = t * (1.0 + 0.05 * torch.randn_like(t))
        assert predicted_loss(near, sigma, sf) >= best * (1.0 - 1e-6)

    # Plain output scales are the optimum when relative error matters equally in every channel
    floor = Q.out_sensitivity_floor
    try:
        Q.out_sensitivity_floor = 0.0
        t = sensitivity_out_scales(sigma, 1.0 / sigma.flatten().square())
    finally:
        Q.out_sensitivity_floor = floor
    assert torch.allclose(t, sigma / sigma.mean(), rtol = 1e-4)

    # The floor bounds the scale of a channel the estimate saw nothing of
    blind = s.clone()
    blind[7] = 0.0
    t = sensitivity_out_scales(sigma, blind).flatten()
    bound = (sigma.flatten()[7].double().square() / floor).pow(0.25) / ref.mean()
    assert t[7].item() == pytest.approx(bound.item(), rel = 0.02)

    # Zero channels stay zero and out of the normalization, padding takes the mean sensitivity
    z = sigma.clone()
    z[0, :16] = 0.0
    t = sensitivity_out_scales(z, s)
    assert (t[0, :16] == 0).all() and t[0, 16:].mean().item() == pytest.approx(1.0, abs = 1e-5)
    padded = sensitivity_out_scales(sigma, s[:n - 64]).flatten().double()
    sp = torch.cat((s[:n - 64] / s[:n - 64].mean(), torch.ones(64))).double().to(device) + floor
    refp = (sigma.flatten().double().square() / sp).pow(0.25)
    assert torch.allclose(padded, refp / refp.mean(), rtol = 1e-5)

    # Negative entries count as zero; uninformative sensitivities give nothing
    neg = s.clone()
    neg[7] = -3.0
    assert torch.equal(sensitivity_out_scales(sigma, neg), sensitivity_out_scales(sigma, blind))
    assert sensitivity_out_scales(sigma, torch.zeros(n)) is None
    assert sensitivity_out_scales(sigma, torch.ones(n + 16)) is None
    nan = torch.ones(n)
    nan[3] = float("nan")
    assert sensitivity_out_scales(sigma, nan) is None


@torch.inference_mode()
def test_regularize_hessian_scales(device):
    torch.manual_seed(6)
    k, n = 256, 512
    weight = torch.randn(k, n, device = device) * (torch.rand(1, n, device = device) * 3.0 + 0.2)
    hdiag = torch.randn(n).exp()
    sigma = weight.square().mean(0, keepdim = True).sqrt()
    sigma = sigma / sigma.mean()
    expect = sensitivity_out_scales(sigma, hdiag)

    applied, _, _, sv, qa = _regularize(weight, None, None, hdiag, full = True)
    assert applied is True and qa["out_scales_hessian"] is True
    assert torch.allclose(sv.abs(), expect, rtol = 1e-4)
    assert not torch.allclose(sv.abs(), sigma, rtol = 0.05)

    # It takes precedence over the binary rule, whichever way that would have gone
    s_hot = torch.zeros(n)
    s_hot[sigma.flatten().topk(8).indices.cpu()] = 1.0
    assert _regularize(weight, None, s_hot)[0] is False
    applied, _, _, sv, qa = _regularize(weight, None, s_hot, hdiag, full = True)
    assert applied is True and qa["out_scales_hessian"] is True and torch.allclose(sv.abs(), expect, rtol = 1e-4)

    # Without information in the diagonal the tensor is treated as in auto
    applied, _, _, sv, qa = _regularize(weight, None, s_hot, torch.zeros(n), full = True)
    assert applied is False and qa["out_scales_hessian"] is False

    # Forced settings and the fallback quantizer ignore it
    for forced in (True, False):
        applied, _, _, sv, qa = _regularize(weight, forced, None, hdiag, full = True)
        assert applied is forced and qa["out_scales_hessian"] is False
    applied, _, _, sv, qa = _regularize(weight, None, None, hdiag, q_fallback = True, full = True)
    assert qa["out_scales_hessian"] is False and torch.allclose(sv.abs(), sigma, rtol = 1e-4)


def quantized_channel_error(weight, out_scales, hdiag):
    """Error energy per output channel after quantizing the regularized weight (plain rounding, K = 3)"""
    applied, weight_r, su, sv, _ = _regularize(weight, out_scales, None, hdiag, full = True)
    def restore(w):
        w = w.clone()
        blockwise_preapply_had_l_(w, Q.had_k)
        w *= su
        blockwise_preapply_had_r_(w, Q.had_n)
        return w * sv
    assert torch.allclose(restore(weight_r), weight, rtol = 1e-3, atol = 1e-4)
    q, _ = quantize_tiles(weight_r.reshape(-1, 256).contiguous(), {"K": 3, "mul1": True})
    return (restore(q.view_as(weight_r)) - weight).double().square().sum(0), sv.abs().flatten().double()


@torch.inference_mode()
def test_hessian_scales_move_the_error(device):
    torch.manual_seed(7)
    k, n = 512, 1024
    weight = torch.randn(k, n, device = device) * (torch.rand(1, n, device = device) * 3.0 + 0.2)
    s = torch.randn(n).exp()
    sw = (s / s.mean() + Q.out_sensitivity_floor).double().to(device)

    err_h, t = quantized_channel_error(weight, None, s)
    err_p, _ = quantized_channel_error(weight, True, None)
    err_n, _ = quantized_channel_error(weight, False, None)

    # A channel's error energy follows the square of its scale
    x, y = t.square().log(), err_h.log()
    slope = ((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).square().sum()
    assert slope.item() == pytest.approx(1.0, abs = 0.05)

    # The loss the sensitivity stands for is lower than with plain scales or none, by about what the model predicts
    loss = lambda e: (sw * e).sum().item()
    sigma = weight.square().mean(0).sqrt()
    sigma = sigma / sigma.mean()
    assert loss(err_h) < 0.9 * loss(err_p) and loss(err_h) < 0.9 * loss(err_n)
    assert loss(err_h) / loss(err_p) == pytest.approx(
        predicted_loss(t, sigma, sw) / predicted_loss(sigma, sigma, sw), rel = 0.1)
    assert loss(err_h) / loss(err_n) == pytest.approx(
        predicted_loss(t, sigma, sw) / predicted_loss(torch.ones_like(sigma), sigma, sw), rel = 0.1)

    # Scales from the inverse of that sensitivity put the error where it costs the most
    err_w, _ = quantized_channel_error(weight, None, 1.0 / s)
    assert loss(err_w) > 1.2 * loss(err_p)


@pytest.mark.nogpu
def test_hessian_diagonal_from_files(tmp_path, yaqa_hessians):
    from exllamav3.conversion.convert_model import get_out_hessian_diags, unpack_sym
    pack_sym = yaqa_hessians.pack_sym

    torch.manual_seed(8)
    class L:
        def __init__(self, key, n): self.key, self.out_features = key, n
    linears = [L("a.packed", 48), L("a.square", 32), L("a.diag", 64), L("a.both", 16), L("a.hin_only", 16), L("a.absent", 16)]
    h = {l.key: (lambda m: m @ m.T)(torch.randn(l.out_features, l.out_features)) for l in linears}
    other = torch.rand(16) + 5.0
    save_file({"hout": pack_sym(h["a.packed"])}, str(tmp_path / "a.packed.safetensors"))
    save_file({"hout": h["a.square"]}, str(tmp_path / "a.square.safetensors"))
    save_file({"hout_diag": h["a.diag"].diagonal().contiguous()}, str(tmp_path / "a.diag.safetensors"))
    save_file({"hout": pack_sym(h["a.both"]), "hout_diag": other}, str(tmp_path / "a.both.safetensors"))
    save_file({"hin": pack_sym(h["a.hin_only"])}, str(tmp_path / "a.hin_only.safetensors"))

    args = {"hessians": str(tmp_path), "out_scales_hessian": True}
    diags = get_out_hessian_diags(args, linears)
    assert set(diags) == {"a.packed", "a.square", "a.diag", "a.both"}
    for key in ("a.packed", "a.square", "a.diag"):
        assert torch.equal(diags[key], h[key].diagonal())
    assert torch.equal(diags["a.packed"], unpack_sym(pack_sym(h["a.packed"]), 48).diagonal())
    assert torch.equal(diags["a.both"], other)
    assert get_out_hessian_diags({"hessians": str(tmp_path), "out_scales_hessian": False}, linears) == {}
    with pytest.raises(AssertionError):
        get_out_hessian_diags(args, [L("a.diag", 48)])


@pytest.mark.parametrize("weighted", (False, True))
@torch.inference_mode()
def test_collector_diagonal(tmp_path, device, yaqa_hessians, weighted):
    """util/yaqa_hessians.py --diag accumulates the diagonal of what the full collection accumulates"""
    Collector = yaqa_hessians.Collector
    from safetensors.torch import load_file
    from exllamav3.conversion.convert_model import unpack_sym

    torch.manual_seed(9)
    module = torch.nn.Linear(96, 160, bias = False, device = device)
    full = Collector("m", "full", module, None, weighted, True)
    diag = Collector("m", "diag", module, None, weighted, True, diag = True)
    assert full.hout.shape == (160, 160) and diag.hout.shape == (160,)
    # Tokens of very different input energy, so the sample weights differ
    rows = [(
        torch.randn(1, 40 + 10 * r, 96, device = device) * torch.randn(1, 40 + 10 * r, 1, device = device).exp(),
        torch.randn(1, 40 + 10 * r, 160, device = device) * torch.randn(1, 40 + 10 * r, 1, device = device).exp(),
    ) for r in range(4)]
    for c in (full, diag):
        for kind in (["fwd"] if weighted else []) + ["out"]:
            c.begin_pass(kind)
            for x, d in rows:
                c.forward(kind, x, None)
                c.backward(kind, d)
                c.end_row()
            c.end_pass(kind)
        c.save(str(tmp_path), "out")
    f, d = load_file(str(tmp_path / "full.safetensors")), load_file(str(tmp_path / "diag.safetensors"))
    assert set(f) == {"hout", "hout_diag"} and set(d) == {"hout_diag"}
    assert torch.equal(f["hout_diag"], unpack_sym(f["hout"], 160).diagonal())
    assert torch.allclose(d["hout_diag"], f["hout_diag"], rtol = 1e-4)

    # Against the definition: mean of the squared gradients, each token weighted by x^T Hin x / |Hin|^2
    xs = torch.cat([r[0].view(-1, 96) for r in rows]).double()
    ds = torch.cat([r[1].view(-1, 160) for r in rows]).double()
    plain = ds.square().mean(0).cpu()
    if weighted:
        hin = xs.T @ xs / xs.shape[0]
        w = ((xs @ hin) * xs).sum(-1, keepdim = True) / hin.norm().square()
        ref = (ds.square() * w).mean(0).cpu()
        assert not torch.allclose(ref, plain, rtol = 0.2)
    else:
        ref = plain
    assert torch.allclose(d["hout_diag"].double(), ref, rtol = 1e-3)
