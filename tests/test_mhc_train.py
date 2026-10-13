"""
CPU tests for the differentiable mHC hyper-connections (DeepSeek-V4 /
GLM-5.3-Flash) in ``exllamav3/training/hyperconnections.py`` (``mhc_mix`` /
``mhc_apply`` / ``mhc_head``), their wiring into the native block / trunk
forward (``native_llama``: mHC sites next to ordinary pre-norms, HyperHead
collapse before the final norm), and the clamped gated activation
(``swiglu_limit`` -> ``act_limit``).

No GPU, no compiled extension, no real model (same scaffolding as
``test_hyperconnections_train.py``, which this file imports). Checks:

  * ``mhc_mix`` / ``mhc_apply`` / ``mhc_head`` match verbatim transcriptions of
    the inference module's torch fallbacks (``HyperConnection._mix_unfused``,
    ``apply_``, ``HyperHead.forward``) and gradcheck in fp64;
  * an attention block on the mHC stream stack (``_block_forward`` with
    ``kind == "mhc"`` site specs AND pre-norm specs) matches an independent
    composition: reference mix -> RMSNorm -> attention -> reference inject ->
    reference mix -> RMSNorm -> SwiGLU -> reference inject;
  * backprop through an mHC block reaches the LoRA adapters and every stream,
    while the frozen base / hc tables stay untouched;
  * the trunk wiring: ExpandStreams -> mHC blocks -> HyperHead (mean and
    weighted) -> final RMSNorm equals the hand-composed chain;
  * ``_gated_mlp`` with ``act_limit`` equals ``min(silu(g), L) * clamp(u, -L, L)``
    (the inference ``act_mul`` kernel's order) and reduces to the plain path at 0.

Run:  python tests/test_mhc_train.py
"""

from __future__ import annotations
import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_hyperconnections_train as hct   # noqa: E402  (loads the training modules)

_hc = hct._hc
_nl = hct._nl
DiffLinear = hct.DiffLinear
MockLinear = hct.MockLinear


# ----------------------------------------------------------------------------
# Verbatim transcription of HyperConnection._mix_unfused / apply_ and
# HyperHead.forward (torch fallbacks), modules/hyperconnections.py.
# ----------------------------------------------------------------------------
def _ref_unweighted_norm(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _ref_mix(streams, fn, base, scale, hc, rms_eps, hc_eps, iters):
    flat = _ref_unweighted_norm(streams.flatten(2).float(), rms_eps)
    mix = F.linear(flat, fn)
    pre_w, post_w, comb_w = mix.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_s + pre_b) + hc_eps
    post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
    comb = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2)
    return post, comb, collapsed


def _ref_apply(x, y, post, comb):
    return post.unsqueeze(-1) * y.float().unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x)


def _ref_head(x, fn, base, scale, rms_eps, hc_eps):
    flat = _ref_unweighted_norm(x.flatten(2).float(), rms_eps)
    mixes = F.linear(flat, fn)
    pre = torch.sigmoid(mixes * scale + base) + hc_eps
    return (pre.unsqueeze(-1) * x).sum(dim=2)


def _make_mhc_spec(H, d, dtype=torch.float32, rms_eps=1e-5, hc_eps=1e-6, iters=20):
    fn = (torch.randn(2 * H + H * H, H * d) * 0.2).to(dtype)
    base = (torch.randn(2 * H + H * H) * 0.3).to(dtype)
    scale = (1.0 + 0.2 * torch.randn(3)).to(dtype)
    return {"kind": "mhc", "key": "hc", "hc_mult": H, "hidden_size": d, "eps": rms_eps,
            "hc_eps": hc_eps, "sinkhorn_iters": iters, "fn": fn, "base": base, "scale": scale}


def _make_head_spec(H, d, mean, dtype=torch.float32, rms_eps=1e-5, hc_eps=1e-6):
    spec = {"mean": mean, "hc_mult": H, "eps": rms_eps, "hc_eps": hc_eps}
    if not mean:
        spec.update({"fn": (torch.randn(H, H * d) * 0.2).to(dtype),
                     "base": (torch.randn(H) * 0.3).to(dtype),
                     "scale": (1.0 + 0.2 * torch.randn(1)).to(dtype)})
    return spec


def _ref_mix_spec(streams, spec):
    return _ref_mix(streams, spec["fn"].float(), spec["base"].float(), spec["scale"].float(),
                    spec["hc_mult"], spec["eps"], spec["hc_eps"], spec["sinkhorn_iters"])


def _ref_head_spec(streams, spec):
    if spec["mean"]:
        return streams.float().mean(dim=2)
    return _ref_head(streams, spec["fn"].float(), spec["base"].float(), spec["scale"].float(),
                     spec["eps"], spec["hc_eps"])


def test_mhc_mix_apply_head_match_inference_reference():
    torch.manual_seed(0)
    b, t, H, d = 2, 5, 4, 12
    spec = _make_mhc_spec(H, d)
    streams = torch.randn(b, t, H, d)
    y = torch.randn(b, t, d)
    post, comb, coll = _hc.mhc_mix(streams, spec)
    rpost, rcomb, rcoll = _ref_mix_spec(streams, spec)
    assert post.shape == (b, t, H) and comb.shape == (b, t, H, H) and coll.shape == (b, t, d)
    for a_, r_, n in ((post, rpost, "post"), (comb, rcomb, "comb"), (coll, rcoll, "collapsed")):
        err = (a_ - r_).abs().max().item()
        assert err < 1e-6, f"mhc_mix {n} mismatch: max|d|={err}"
    # The combine matrix is (approximately) doubly stochastic after Sinkhorn.
    assert (comb.sum(-1) - 1).abs().max() < 1e-3 and (comb.sum(-2) - 1).abs().max() < 1e-3
    out = _hc.mhc_apply(streams, y, post, comb)
    ref = _ref_apply(streams, y, rpost, rcomb)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, f"mhc_apply mismatch: max|d|={err}"
    for mean in (True, False):
        hs = _make_head_spec(H, d, mean)
        h = _hc.mhc_head(streams, hs)
        rh = _ref_head_spec(streams, hs)
        err = (h - rh).abs().max().item()
        assert h.shape == (b, t, d) and err < 1e-6, f"mhc_head(mean={mean}) mismatch: {err}"
    print("[mhc] mix / apply / head match the inference torch fallbacks PASSED")


def test_mhc_gradcheck():
    torch.manual_seed(1)
    b, t, H, d = 1, 3, 4, 6
    spec = _make_mhc_spec(H, d, dtype=torch.float64, iters=5)
    streams = torch.randn(b, t, H, d, dtype=torch.float64, requires_grad=True)
    y = torch.randn(b, t, d, dtype=torch.float64, requires_grad=True)

    def f(s, yy):
        post, comb, coll = _hc.mhc_mix(s, spec)
        return _hc.mhc_apply(s, yy, post, comb) + coll.unsqueeze(2)
    assert torch.autograd.gradcheck(f, (streams, y), eps=1e-6, atol=1e-5)
    hs = _make_head_spec(H, d, False, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda s: _hc.mhc_head(s, hs), (streams,), eps=1e-6, atol=1e-5)
    print("[mhc] mix + apply + head gradcheck (fp64) PASSED")


# ----------------------------------------------------------------------------
# Attention block on the mHC stream stack (pre-norms present).
# ----------------------------------------------------------------------------
def _ref_rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


def _build_mhc_attn_block(d, nq, nkv, hd, inter, H, r=0, dtype=torch.float32):
    entry, meta, ref_w, lins = hct._build_hc_attn_block(d, nq, nkv, hd, inter, H, 4, r=r, dtype=dtype)
    entry.attn_hc_spec = _make_mhc_spec(H, d, dtype=dtype)
    entry.mlp_hc_spec = _make_mhc_spec(H, d, dtype=dtype)
    norms = {"attn": 1.0 + 0.1 * torch.randn(d), "mlp": 1.0 + 0.1 * torch.randn(d)}
    entry.attn_norm_spec = hct._spec(norms["attn"], eps=1e-5)
    entry.mlp_norm_spec = hct._spec(norms["mlp"], eps=1e-5)
    return entry, meta, ref_w, lins, norms


def _ref_mhc_block(entry, norms, streams, sublayer_attn, sublayer_mlp):
    post, comb, c = _ref_mix_spec(streams, entry.attn_hc_spec)
    streams = _ref_apply(streams, sublayer_attn(_ref_rmsnorm(c, norms["attn"], 1e-5)), post, comb)
    post, comb, c = _ref_mix_spec(streams, entry.mlp_hc_spec)
    return _ref_apply(streams, sublayer_mlp(_ref_rmsnorm(c, norms["mlp"], 1e-5)), post, comb)


def test_mhc_attention_block_matches_reference():
    torch.manual_seed(2)
    d, nq, nkv, hd, inter, H = 16, 4, 2, 8, 32, 4
    entry, meta, w, _, norms = _build_mhc_attn_block(d, nq, nkv, hd, inter, H)
    net = hct._headless_net()
    b, t = 2, 6
    streams = torch.randn(b, t, H, d)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    bias = net._attn_bias(None, t, torch.device("cpu"), torch.float32)
    with torch.no_grad():
        out = net._block_forward(meta, entry, streams, positions, bias)
    ref = _ref_mhc_block(entry, norms, streams,
                         lambda y: hct._ref_attention(meta, w, y, positions),
                         lambda y: hct._ref_swiglu(w, y))
    assert out.shape == (b, t, H, d) and out.dtype == torch.float32
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"mHC attention block mismatch: max|d|={err}"
    print(f"[mhc] attention block on the stream stack matches reference (max|d|={err:.2e}) PASSED")


def test_mhc_block_backward_reaches_adapters_and_streams():
    torch.manual_seed(3)
    d, nq, nkv, hd, inter, H, r = 12, 2, 1, 6, 24, 4, 3
    entry, meta, _, lins, _ = _build_mhc_attn_block(d, nq, nkv, hd, inter, H, r=r)
    net = hct._headless_net()
    frozen_before = {k_: l.frozen_weight.clone() for k_, l in lins.items()}
    hc_before = {(s, k_): v.clone() for s in ("attn_hc_spec", "mlp_hc_spec")
                 for k_, v in getattr(entry, s).items() if torch.is_tensor(v)}
    wrappers = [entry.q_proj, entry.k_proj, entry.v_proj, entry.o_proj,
                entry.gates[0], entry.ups[0], entry.downs[0]]
    with torch.no_grad():
        for w in wrappers:
            w.lora_b.normal_(std=0.05)
    b, t = 2, 5
    streams = torch.randn(b, t, H, d, requires_grad=True)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    bias = net._attn_bias(None, t, torch.device("cpu"), torch.float32)
    out = net._block_forward(meta, entry, streams, positions, bias)
    out.pow(2).mean().backward()
    for w in wrappers:
        assert w.lora_a.grad is not None and w.lora_b.grad is not None, w.linear.key
        assert w.lora_b.grad.abs().sum() > 0, f"zero grad on {w.linear.key}"
        assert w.linear.frozen_weight.grad is None
    assert streams.grad is not None and (streams.grad.abs().amax(dim=(0, 1, 3)) > 0).all()
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen_before[k_])
    for s in ("attn_hc_spec", "mlp_hc_spec"):
        for k_, v in getattr(entry, s).items():
            if torch.is_tensor(v):
                assert torch.equal(v, hc_before[(s, k_)]) and v.grad is None
    print("[mhc] backward reaches every adapter and all four streams; base + hc tables frozen PASSED")


# ----------------------------------------------------------------------------
# Trunk wiring: ExpandStreams -> mHC blocks -> HyperHead -> final RMSNorm.
# ----------------------------------------------------------------------------
def test_trunk_expand_mhc_blocks_hyper_head():
    torch.manual_seed(4)
    d, nq, nkv, hd, inter, H, V = 16, 4, 2, 8, 32, 4, 40
    e0, m0, w0, _, n0 = _build_mhc_attn_block(d, nq, nkv, hd, inter, H)
    e1, m1, w1, _, n1 = _build_mhc_attn_block(d, nq, nkv, hd, inter, H)
    final_w = 1.0 + 0.1 * torch.randn(d)
    table = nn.Embedding(V, d)
    table.weight.requires_grad_(False)
    b, t = 2, 7
    ids = torch.randint(0, V, (b, t))
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    for mean in (True, False):
        net = hct._make_trunk_net([e0, e1], [m0, m1], None, table, H)
        net.final_norm_spec = hct._spec(final_w, eps=1e-5)
        net.hyper_head_spec = _make_head_spec(H, d, mean)
        with torch.no_grad():
            out, ctx = net._forward_trunk(ids)
        assert out.shape == (b, t, d) and out.dtype == torch.float32
        streams = table.weight[ids].float().unsqueeze(2).expand(-1, -1, H, -1)
        for e, m, w, n in ((e0, m0, w0, n0), (e1, m1, w1, n1)):
            streams = _ref_mhc_block(e, n, streams,
                                     lambda y, m=m, w=w: hct._ref_attention(m, w, y, positions),
                                     lambda y, w=w: hct._ref_swiglu(w, y))
        ref = _ref_rmsnorm(_ref_head_spec(streams, net.hyper_head_spec), final_w, 1e-5)
        err = (out - ref).abs().max().item()
        assert err < 1e-4, f"trunk (expand -> mhc blocks -> head mean={mean} -> norm) mismatch: {err}"
        # Taps read the stream mean.
        with net.collect_hidden([1]):
            with torch.no_grad():
                net._forward_trunk(ids)
        assert (net.collected[1] - streams.mean(dim=2)).abs().max().item() < 1e-4
    print("[mhc] trunk wiring expand -> mhc blocks -> hyper head (mean + weighted) -> final norm PASSED")


# ----------------------------------------------------------------------------
# Clamped gated activation (swiglu_limit).
# ----------------------------------------------------------------------------
def test_gated_mlp_act_limit():
    torch.manual_seed(5)
    d, inter, L = 8, 16, 1.5
    lins = {"gate": MockLinear(d, inter, "mlp.gate_proj", scale=1.0),
            "up": MockLinear(d, inter, "mlp.up_proj", scale=1.0),
            "down": MockLinear(inter, d, "mlp.down_proj", scale=0.2)}
    gates = [DiffLinear(lins["gate"], r=0, compute_dtype=torch.float32)]
    ups = [DiffLinear(lins["up"], r=0, compute_dtype=torch.float32)]
    downs = [DiffLinear(lins["down"], r=0, compute_dtype=torch.float32)]
    net = hct._headless_net()
    net.use_liger = False
    x = torch.randn(3, 5, d) * 2
    g = x @ lins["gate"].frozen_weight
    u = x @ lins["up"].frozen_weight
    assert (F.silu(g) > L).any() and (u.abs() > L).any(), "test inputs do not exercise the clamp"
    ref = (F.silu(g).clamp(max=L) * u.clamp(-L, L)) @ lins["down"].frozen_weight
    out = net._gated_mlp(gates, ups, downs, "silu", x, act_limit=L)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"act_limit gated MLP mismatch: {err}"
    plain = net._gated_mlp(gates, ups, downs, "silu", x)
    ref_plain = (F.silu(g) * u) @ lins["down"].frozen_weight
    assert (plain - ref_plain).abs().max().item() < 1e-5
    assert (plain - out).abs().max().item() > 1e-3, "clamp had no effect"
    print("[mhc] clamped gated activation min(silu(g), L) * clamp(u, -L, L) PASSED")


def main():
    from util import run_timed
    run_timed([
        test_mhc_mix_apply_head_match_inference_reference,
        test_mhc_gradcheck,
        test_mhc_attention_block_matches_reference,
        test_mhc_block_backward_reaches_adapters_and_streams,
        test_trunk_expand_mhc_blocks_hyper_head,
        test_gated_mlp_act_limit,
    ], label="mhc-train")
    print("\nAll mHC training checks passed.")


if __name__ == "__main__":
    main()
