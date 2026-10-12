"""
CPU tests for the differentiable PLE (per-layer embedding) layer of
Qwen3.8-Flash-Next (``exllamav3/training/ple.py``) and its wiring into the
native trunk forward (``native_llama._run_ple`` / ``backbone.ple_ngram_embed``).

No GPU, no compiled extension, no real model (and no 51B-row n-gram table):
the training modules are loaded under a synthetic package, the key/value
projections are mocked EXL3 linears, the n-gram embedding is a mock that
returns random rows, and the checks are:

  * ``ple_delta`` matches a verbatim transcription of the inference module's
    op-by-op reference (``PLELayer.forward_streams_reference`` +
    ``_short_conv`` from a zero conv state, with the ``ple_gate`` kernel's
    signed-sqrt sigmoid written out) on random weights;
  * the gate's signed sqrt is zero-safe and the whole delta gradchecks in
    fp64 w.r.t. the stream stack;
  * the dilated depthwise conv is causal with the right taps: perturbing one
    position moves only that position and the ``dilation``-spaced positions
    after it, never anything earlier;
  * the trunk wiring: ``_run_ple`` builds the eos-padded hashing history the
    inference layer builds for a fresh sequence, adds the delta into the
    streams in place of the inference ``x + delta``, and checkpointing
    reproduces the unchecked result.

Run:  python tests/test_ple_train.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_qll = _load("qlora_linear")
_fce = _load("fused_ce")
_gdn = _load("gdn")
_ple = _load("ple")
_bb = _load("backbone")
_nl = _load("native_llama")
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA


class _MockInner:
    def __init__(self, weight):
        self._w = weight
        self.trellis = weight
        self.bias = None

    def get_weight_tensor(self):
        return self._w

    def get_bias_tensor(self):
        return None


class MockLinear(nn.Module):
    def __init__(self, in_features, out_features, key, scale=0.05, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.key = key
        self.device = torch.device("cpu")
        w = torch.randn(in_features, out_features, dtype=dtype) * scale
        self.register_buffer("frozen_weight", w)
        self.inner = _MockInner(self.frozen_weight)
        self.lora_a_tensors = {}
        self.lora_b_tensors = {}


def _headless_net(dtype=torch.float32):
    n = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    n.compute_dtype = dtype
    n.attn_impl = "eager"
    n.use_liger = False
    return n


def _grouped_spec(H, d, dtype=torch.float32, eps=1e-6):
    """backbone.norm_spec of a grouped RMSNorm with constant_bias 1.0: weight
    viewed [H, d] (zero-init on the real model; random here so it matters)."""
    w = (0.1 * torch.randn(H, d)).to(dtype)
    return {"weight": w, "eps": eps, "bias": 1.0, "scale": 1.0}


def _make_ple(H, d, ple_dim, kernel=4, ngram=3, dtype=torch.float32, scale=0.3):
    """A PLE entry (spec + frozen DiffLinear projections) with random weights,
    plus the raw tensors for the reference."""
    lins = {
        "key": MockLinear(ple_dim, H * d, "ple.key_proj", scale=scale, dtype=dtype),
        "value": MockLinear(ple_dim, d, "ple.value_proj", scale=scale, dtype=dtype),
    }
    conv_w = (torch.randn(H * d, 1, kernel) * 0.4).to(dtype)
    spec = {
        "key": "ple", "hc_mult": H, "hidden_size": d,
        "gate_scale": 1.0 / (d ** 0.5), "dilation": ngram, "kernel": kernel,
        "state_len": (kernel - 1) * ngram, "conv_w": conv_w,
        "norm_key": _grouped_spec(H, d, dtype), "norm_query": _grouped_spec(H, d, dtype),
        "norm_conv": _grouped_spec(H, d, dtype),
    }
    pe = types.SimpleNamespace()
    pe.spec = spec
    pe.key_proj = DiffLinear(lins["key"], r=0, compute_dtype=dtype)
    pe.value_proj = DiffLinear(lins["value"], r=0, compute_dtype=dtype)
    pe.device = torch.device("cpu")
    return pe, lins


# ----------------------------------------------------------------------------
# Transcription of the inference reference: PLELayer.forward_streams_reference
# + _short_conv (conv_state None) + the ple_gate kernel, with RMSNorm.forward
# (grouped: rms per stream row, weight row per stream, applied as 1 + w).
# ----------------------------------------------------------------------------
def _ref_grouped_rmsnorm(x, spec):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True) + spec["eps"]
    return xf * torch.rsqrt(var) * (spec["weight"].float() + spec["bias"])


def _ref_ple_gate(gate, value, gate_scale):
    # ple_gate_kernel: g = gate * scale; a = sqrt(max(|g|, 1e-6));
    # ss = a if g > 0 else (-a if g < 0 else 0); s = sigmoid(ss); out = s * value
    g = gate.float() * gate_scale
    a = torch.sqrt(torch.clamp(g.abs(), min=1e-6))
    ss = torch.where(g > 0, a, torch.where(g < 0, -a, torch.zeros_like(a)))
    s = torch.sigmoid(ss)
    return s.unsqueeze(-1) * value.float().unsqueeze(-2)


def _ref_short_conv(x, conv_w, dilation, state_len):
    bsz, seq, ch = x.shape
    xt = x.transpose(1, 2)
    conv_state = xt.new_zeros((bsz, ch, state_len))
    xt = torch.cat((conv_state.to(xt.dtype), xt), dim=-1)
    y = F.conv1d(xt, conv_w.to(xt.dtype), groups=ch, dilation=dilation)
    return F.silu(y).transpose(1, 2)


def _ref_ple(streams, emb, spec, w_key, w_value):
    bsz, seq = streams.shape[:2]
    H, D = spec["hc_mult"], spec["hidden_size"]
    key = (emb @ w_key).view(bsz, seq, H, D)
    key = _ref_grouped_rmsnorm(key, spec["norm_key"])
    value = emb @ w_value
    query = _ref_grouped_rmsnorm(streams, spec["norm_query"])
    gate = torch.bmm(query.reshape(-1, 1, D), key.reshape(-1, D, 1)).view(bsz, seq, H)
    gated = _ref_ple_gate(gate, value, spec["gate_scale"])
    normed = _ref_grouped_rmsnorm(gated, spec["norm_conv"]).flatten(-2)
    conv_out = _ref_short_conv(normed, spec["conv_w"], spec["dilation"], spec["state_len"])
    return gated + conv_out.view(bsz, seq, H, D)


def test_ple_delta_matches_reference():
    torch.manual_seed(0)
    b, t, H, d, ple_dim = 2, 13, 4, 16, 24
    pe, lins = _make_ple(H, d, ple_dim)
    net = _headless_net()
    streams = torch.randn(b, t, H, d) * 2
    emb = torch.randn(b, t, ple_dim)
    out = _ple.ple_delta(streams, emb, pe.spec, pe.key_proj, pe.value_proj,
                         net._norm, torch.float32)
    ref = _ref_ple(streams, emb, pe.spec, lins["key"].frozen_weight,
                   lins["value"].frozen_weight)
    assert out.shape == streams.shape and out.dtype == torch.float32
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"ple_delta mismatch vs inference reference: max|d|={err}"
    assert out.abs().max() > 1e-2
    # The conv term must contribute: with a zero conv kernel (silu(0) = 0) the
    # delta is exactly the gated values, and that differs from the full delta.
    spec0 = dict(pe.spec, conv_w=torch.zeros_like(pe.spec["conv_w"]))
    gated_only = _ple.ple_delta(streams, emb, spec0, pe.key_proj, pe.value_proj,
                                net._norm, torch.float32)
    key = _ref_grouped_rmsnorm((emb @ lins["key"].frozen_weight).view(b, t, H, d),
                               pe.spec["norm_key"])
    query = _ref_grouped_rmsnorm(streams, pe.spec["norm_query"])
    gated = _ref_ple_gate((query * key).sum(-1), emb @ lins["value"].frozen_weight,
                          pe.spec["gate_scale"])
    assert (gated_only - gated).abs().max().item() < 1e-5
    assert not torch.allclose(out, gated_only, atol=1e-3)
    print(f"[ple] ple_delta matches forward_streams_reference transcription "
          f"(max|d|={err:.2e}) PASSED")


def test_ple_gate_zero_safe_and_gradcheck():
    # signed sqrt: zero at zero, odd, finite gradient everywhere.
    g = torch.tensor([-4.0, -1e-9, 0.0, 1e-9, 4.0])
    s = _ple.ple_signed_sqrt_gate(g, 1.0)
    assert s[2].item() == 0.5
    assert torch.allclose(s[0], torch.sigmoid(torch.tensor(-2.0)))
    assert torch.allclose(s[4], torch.sigmoid(torch.tensor(2.0)))
    assert torch.allclose(s[0] + s[4], torch.tensor(1.0))
    gg = g.clone().requires_grad_(True)
    _ple.ple_signed_sqrt_gate(gg, 1.0).sum().backward()
    assert torch.isfinite(gg.grad).all()

    torch.manual_seed(1)
    b, t, H, d, ple_dim = 1, 5, 2, 6, 8
    pe, _ = _make_ple(H, d, ple_dim, dtype=torch.float64)
    net = _headless_net(torch.float64)
    streams = torch.randn(b, t, H, d, dtype=torch.float64, requires_grad=True)
    emb = torch.randn(b, t, ple_dim, dtype=torch.float64)

    def fn(s):
        return _ple.ple_delta(s, emb, pe.spec, pe.key_proj, pe.value_proj,
                              net._norm, torch.float64)

    assert torch.autograd.gradcheck(fn, (streams,), eps=1e-6, atol=1e-5)
    print("[ple] signed-sqrt gate zero-safe; ple_delta gradcheck (fp64) PASSED")


def test_ple_conv_causal_dilated_taps():
    torch.manual_seed(2)
    b, t, ch, kernel, dil = 1, 16, 6, 4, 3
    conv_w = torch.randn(ch, 1, kernel) * 0.5
    x = torch.randn(b, t, ch)
    base = _ple.ple_dilated_causal_conv_silu(x, conv_w, dil)
    assert base.shape == x.shape
    p = 4
    x2 = x.clone()
    x2[:, p] += 1.0
    moved = ((_ple.ple_dilated_causal_conv_silu(x2, conv_w, dil) - base).abs()
             .amax(dim=(0, 2)) > 1e-6)
    expect = torch.zeros(t, dtype=torch.bool)
    for k in range(kernel):
        if p + k * dil < t:
            expect[p + k * dil] = True
    assert torch.equal(moved, expect), f"conv taps {moved.nonzero().flatten().tolist()}"
    # Zero left state: position 0 sees only itself through the last tap.
    y0 = F.silu(x[:, 0] * conv_w[:, 0, -1])
    assert torch.allclose(base[:, 0], y0, atol=1e-6)
    print("[ple] dilated depthwise conv is causal with taps at 0/3/6/9 and a zero state PASSED")


# ----------------------------------------------------------------------------
# Trunk wiring: _run_ple + backbone.ple_ngram_embed over a mock inference layer.
# ----------------------------------------------------------------------------
class _MockNGram:
    def __init__(self, ctx, eos, ple_dim, V):
        self.context_len = ctx
        self.eos_token_id = eos
        self.rows = torch.randn(V, ple_dim)
        self.calls = []

    def forward(self, history, params):
        self.calls.append(history.clone())
        # Row per position from the current token (a stand-in for the hashed
        # n-gram gather); returns the last seq positions, half like the module.
        ids = history[:, self.context_len:]
        return self.rows[ids].half()


class _MockPLEModule:
    """The slice of PLELayer that backbone.ple_ngram_embed touches."""

    def __init__(self, ngram, mm_token_id=None):
        self.ple_embedding = ngram
        self.mm_token_id = mm_token_id

    def _prepare_ids(self, ids):
        return ids.to("cpu", torch.int64)

    def _history(self, ids):
        pad = ids.new_full((ids.shape[0], self.ple_embedding.context_len),
                           self.ple_embedding.eos_token_id)
        return torch.cat((pad, ids), dim=1)


def test_run_ple_wiring_and_history():
    torch.manual_seed(3)
    b, t, H, d, ple_dim, V, eos, ngram = 2, 9, 4, 16, 24, 50, 7, 3
    pe, lins = _make_ple(H, d, ple_dim, ngram=ngram)
    ng = _MockNGram(ngram - 1, eos, ple_dim, V)
    mod = _MockPLEModule(ng)
    net = _headless_net()
    nn.Module.__init__(net)
    net.compute_dtype = torch.float32
    net._ple_modules = {1: mod}
    ids = torch.randint(0, V, (b, t))
    streams = torch.randn(b, t, H, d)

    out = net._run_ple(1, pe, streams, ids, ckpt=False)
    # The hashing history is the fresh-sequence one: ngram_size - 1 eos tokens
    # in front of the ids (PLELayer._history), built once per forward.
    assert len(ng.calls) == 1
    hist = ng.calls[0]
    assert hist.shape == (b, ngram - 1 + t) and hist.dtype == torch.int64
    assert (hist[:, :ngram - 1] == eos).all() and torch.equal(hist[:, ngram - 1:], ids)

    emb = ng.rows[ids].half()
    ref = streams + _ref_ple(streams, emb.float(), pe.spec, lins["key"].frozen_weight,
                             lins["value"].frozen_weight)
    err = (out - ref).abs().max().item()
    assert err < 1e-5, f"_run_ple mismatch: max|d|={err}"

    # Checkpointed path reproduces the unchecked one and still backprops into
    # the streams (the lookup runs once, outside the checkpoint).
    net.train()
    s = streams.clone().requires_grad_(True)
    out_c = net._run_ple(1, pe, s, ids, ckpt=True)
    assert torch.allclose(out_c, out, atol=1e-6)
    out_c.pow(2).mean().backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert len(ng.calls) == 2
    print(f"[ple] _run_ple: eos-padded history, delta added into the streams "
          f"(max|d|={err:.2e}), checkpoint parity + backward PASSED")


def main():
    from util import run_timed
    run_timed([
        test_ple_delta_matches_reference,
        test_ple_gate_zero_safe_and_gradcheck,
        test_ple_conv_causal_dilated_taps,
        test_run_ple_wiring_and_history,
    ], label="ple-train")
    print("\nAll PLE training checks passed.")


if __name__ == "__main__":
    main()
