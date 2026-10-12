"""
CPU tests for the differentiable gated-residual hyper-connections
(Qwen3.8-Flash-Next / ``qwen4_exp``) in ``exllamav3/training/hyperconnections.py``
and their wiring into the native block / trunk forward (``native_llama``), plus
the sigmoid GDN output gate (``training.gdn.gdn_gated_rmsnorm``).

No GPU, no compiled extension, no real model: the training modules are loaded
under a synthetic package (so their relative imports resolve without importing
the full exllamav3 package, which would build the CUDA ext), EXL3 linears are
mocked as frozen random weights, and the checks are:

  * ``gated_residual_mix`` matches a verbatim transcription of the inference
    module's own fp32 reference (``GatedResidual._mix_ref``) and, with
    ``gated_residual_apply``, gradchecks in fp64;
  * an attention block on the stream stack (``_block_forward`` with
    ``attn_hc_spec`` / ``mlp_hc_spec``) matches an independent composition:
    reference mix -> plain-torch GQA/RoPE attention -> reference inject ->
    reference mix -> SwiGLU -> reference inject;
  * a GatedDeltaNet block on the stream stack with the SIGMOID output gate
    (``output_gate_type = "sigmoid"``) matches the same composition over the
    transcribed inference GDN pieces, and the gate activation actually
    changes the result vs silu;
  * backprop through an hc block reaches the LoRA adapters and flows back
    into the stream stack, while the frozen base / hc tables stay untouched;
  * the trunk wiring: ExpandStreams -> blocks -> final combine-less mixer
    (``_forward_trunk`` on a headless net) equals the hand-composed chain,
    and the QSA length guard refuses sequences past the threshold.

Run:  python tests/test_hyperconnections_train.py
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
_hc = _load("hyperconnections")
_nl = _load("native_llama")
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA


# ----------------------------------------------------------------------------
# Mock EXL3 linear (same shape as test_native_llama's).
# ----------------------------------------------------------------------------
class _MockInner:
    def __init__(self, weight):
        self._w = weight                 # [in, out], frozen
        self.trellis = weight            # device inference
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


def _spec(weight, eps=1e-5, bias=0.0, scale=1.0):
    return {"weight": weight, "eps": eps, "bias": bias, "scale": scale}


def _headless_net(dtype=torch.float32):
    n = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    n.compute_dtype = dtype
    n.attn_impl = "eager"
    return n


# ----------------------------------------------------------------------------
# Verbatim transcription of GatedResidual._mix_ref / apply_ (torch fallback),
# modules/hyperconnections.py -- the inference parity tests' ground truth.
# ----------------------------------------------------------------------------
def _ref_mix(streams, norm_w, down_h, up_h, inject_h, hc_mult, hidden_size, rms_eps):
    x = streams.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_eps) * norm_w
    flat = normed.flatten(-2)
    t = F.silu(F.linear(flat, down_h.float()) / hc_mult)
    w = torch.sigmoid(F.linear(t, up_h.float()))
    mixed = (w.unflatten(-1, (hc_mult, hidden_size)) * normed).mean(dim=-2)
    post = 2.0 * torch.sigmoid(F.linear(flat, inject_h.float()) / hc_mult) \
        if inject_h is not None else None
    return post, mixed


def _ref_apply(x, y, post):
    return x + post.unsqueeze(-1) * y.float().unsqueeze(-2)


def _make_hc_spec(H, d, rank, site=True, dtype=torch.float32, eps=1e-6):
    """Random GatedResidual tables in the module's resident layout: norm_w
    [H, d] already 1 + w (fp32), down [rank, H*d] / up [H*d, rank] / inject
    [H, H*d] (half on the real model; here the test dtype)."""
    norm_w = (1.0 + 0.05 * torch.randn(H, d)).to(dtype)
    down = (torch.randn(rank, H * d) * 0.2).to(dtype)
    up = (torch.randn(H * d, rank) * 0.2).to(dtype)
    inject = (torch.randn(H, H * d) * 0.2).to(dtype) if site else None
    return {"key": "hc", "hc_mult": H, "hidden_size": d, "rank": rank, "eps": eps,
            "norm_w": norm_w, "down": down, "up": up, "inject": inject}


def _ref_mix_spec(streams, spec):
    return _ref_mix(streams, spec["norm_w"].float(), spec["down"], spec["up"],
                    spec["inject"], spec["hc_mult"], spec["hidden_size"], spec["eps"])


def test_mix_matches_inference_reference():
    torch.manual_seed(0)
    b, t, H, d, rank = 2, 5, 4, 16, 8
    for site in (True, False):
        spec = _make_hc_spec(H, d, rank, site=site)
        streams = torch.randn(b, t, H, d) * 3
        post, mixed = _hc.gated_residual_mix(streams, spec)
        rpost, rmixed = _ref_mix_spec(streams, spec)
        err = (mixed - rmixed).abs().max().item()
        assert err < 1e-6, f"mix mismatch (site={site}): max|d|={err}"
        assert mixed.shape == (b, t, d) and mixed.dtype == torch.float32
        if site:
            perr = (post - rpost).abs().max().item()
            assert perr < 1e-6, f"post mismatch: max|d|={perr}"
            assert post.shape == (b, t, H)
            y = torch.randn(b, t, d)
            out = _hc.gated_residual_apply(streams, y, post)
            aerr = (out - _ref_apply(streams, y, rpost)).abs().max().item()
            assert aerr < 1e-6, f"apply mismatch: max|d|={aerr}"
            assert out.dtype == torch.float32 and out.shape == streams.shape
        else:
            assert post is None
    # The mix is NOT a plain mean (the gate must matter) and the streams are
    # weighted per stream (post differs across streams).
    spec = _make_hc_spec(H, d, rank)
    streams = torch.randn(b, t, H, d)
    post, mixed = _hc.gated_residual_mix(streams, spec)
    assert not torch.allclose(mixed, streams.mean(dim=2), atol=1e-3)
    assert (post.std(dim=-1) > 1e-4).all()
    print("[hc] gated_residual_mix / apply match GatedResidual._mix_ref PASSED")


def test_mix_apply_gradcheck():
    torch.manual_seed(1)
    b, t, H, d, rank = 1, 3, 4, 8, 4
    spec = _make_hc_spec(H, d, rank, dtype=torch.float64)
    streams = torch.randn(b, t, H, d, dtype=torch.float64, requires_grad=True)
    y = torch.randn(b, t, d, dtype=torch.float64, requires_grad=True)

    def fn(s, yy):
        post, mixed = _hc.gated_residual_mix(s, spec)
        return _hc.gated_residual_apply(s, yy, post) + mixed.unsqueeze(2)

    assert torch.autograd.gradcheck(fn, (streams, y), eps=1e-6, atol=1e-6)
    print("[hc] mix + apply gradcheck (fp64) PASSED")


def test_expand_streams():
    h = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    s = _hc.expand_streams(h, 4)
    assert s.shape == (2, 3, 4, 8) and s.dtype == torch.float32 and s.is_contiguous()
    assert torch.equal(s[:, :, 2], h.float())
    assert torch.equal(_hc.streams_mean(s), h.float())
    print("[hc] expand_streams / streams_mean PASSED")


# ----------------------------------------------------------------------------
# Attention block on the stream stack.
# ----------------------------------------------------------------------------
def _ref_rope(x, inv_freq, positions):
    freqs = positions.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)
    emb = torch.cat((freqs, freqs), -1)
    cos = emb.cos().unsqueeze(2)
    sin = emb.sin().unsqueeze(2)
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def _ref_attention(meta, w, normed, positions):
    """Plain-torch GQA + NeoX RoPE + causal softmax attention + o_proj on an
    already-mixed [b, t, d] input (no norm: the hc mix replaces it)."""
    nq, nkv, hd = meta["num_q_heads"], meta["num_kv_heads"], meta["head_dim"]
    b, t, _ = normed.shape
    q = (normed @ w["q"]).view(b, t, nq, hd)
    k = (normed @ w["k"]).view(b, t, nkv, hd)
    v = (normed @ w["v"]).view(b, t, nkv, hd)
    q = _ref_rope(q, meta["inv_freq"], positions)
    k = _ref_rope(k, meta["inv_freq"], positions)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    rep = nq // nkv
    k = k.repeat_interleave(rep, 1)
    v = v.repeat_interleave(rep, 1)
    scores = (q @ k.transpose(-1, -2)) * meta["sm_scale"]
    mask = torch.triu(torch.full((t, t), float("-inf"), dtype=scores.dtype), 1)
    ctx = torch.softmax(scores + mask, -1) @ v
    ctx = ctx.transpose(1, 2).reshape(b, t, nq * hd)
    return ctx @ w["o"]


def _ref_swiglu(w, x):
    return (F.silu(x @ w["gate"]) * (x @ w["up"])) @ w["down"]


def _build_hc_attn_block(d, nq, nkv, hd, inter, H, rank, r=0, dtype=torch.float32):
    lins = {
        "q": MockLinear(d, nq * hd, "blk.self_attn.q_proj", dtype=dtype),
        "k": MockLinear(d, nkv * hd, "blk.self_attn.k_proj", dtype=dtype),
        "v": MockLinear(d, nkv * hd, "blk.self_attn.v_proj", dtype=dtype),
        "o": MockLinear(nq * hd, d, "blk.self_attn.o_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.mlp.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.mlp.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.mlp.down_proj", dtype=dtype),
    }
    entry = types.SimpleNamespace()
    # No pre-norms on an hc block (backbone.block_norms -> (None, None)).
    entry.attn_norm_spec = None
    entry.mlp_norm_spec = None
    entry.attn_post_spec = None
    entry.mlp_post_spec = None
    entry.attn_hc_spec = _make_hc_spec(H, d, rank, dtype=dtype)
    entry.mlp_hc_spec = _make_hc_spec(H, d, rank, dtype=dtype)
    entry.q_norm_spec = entry.k_norm_spec = entry.v_norm_spec = None
    entry.q_proj = DiffLinear(lins["q"], r=r, compute_dtype=dtype)
    entry.k_proj = DiffLinear(lins["k"], r=r, compute_dtype=dtype)
    entry.v_proj = DiffLinear(lins["v"], r=r, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=r, compute_dtype=dtype)
    entry.g_proj = None
    entry.gates = [DiffLinear(lins["gate"], r=r, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=r, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=r, compute_dtype=dtype)]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=dtype) / hd))
    meta = {
        "kind": "attn",
        "num_q_heads": nq, "num_kv_heads": nkv, "head_dim": hd,
        "sm_scale": hd ** -0.5, "inv_freq": inv_freq, "attn_factor": 1.0,
        "mrope_section": None, "sliding_window": -1, "softcap": 0.0,
        "activation": "silu", "mlp_kind": "dense", "use_k_as_v": False,
        "interleaved_gate": False, "full_gate": False, "headwise_gate": False,
        "layer_scalar": None, "qsa_threshold": None,
    }
    ref_w = {k_: v_.frozen_weight for k_, v_ in lins.items()}
    return entry, meta, ref_w, lins


def _ref_hc_block(entry, streams, sublayer_attn, sublayer_mlp):
    """Independent block composition on the stream stack: reference mix ->
    sublayer -> reference inject, twice."""
    post, y = _ref_mix_spec(streams, entry.attn_hc_spec)
    streams = _ref_apply(streams, sublayer_attn(y), post)
    post, y = _ref_mix_spec(streams, entry.mlp_hc_spec)
    return _ref_apply(streams, sublayer_mlp(y), post)


def test_hc_attention_block_matches_reference():
    torch.manual_seed(2)
    d, nq, nkv, hd, inter, H, rank = 16, 4, 2, 8, 32, 4, 8
    entry, meta, w, _ = _build_hc_attn_block(d, nq, nkv, hd, inter, H, rank)
    net = _headless_net()
    b, t = 2, 6
    streams = torch.randn(b, t, H, d)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    bias = net._attn_bias(None, t, torch.device("cpu"), torch.float32)
    out = net._block_forward(meta, entry, streams, positions, bias)
    ref = _ref_hc_block(entry, streams,
                        lambda y: _ref_attention(meta, w, y, positions),
                        lambda y: _ref_swiglu(w, y))
    assert out.shape == streams.shape and out.dtype == torch.float32
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"hc attention block mismatch vs reference: max|d|={err}"
    # Every stream moved, and differently (per-stream inject gates).
    delta = out - streams
    assert (delta.abs().amax(dim=(0, 1, 3)) > 1e-3).all()
    assert not torch.allclose(delta[:, :, 0], delta[:, :, 1], atol=1e-4)
    print(f"[hc] attention block on the stream stack matches reference "
          f"(max|d|={err:.2e}) PASSED")


# ----------------------------------------------------------------------------
# GDN block on the stream stack with the sigmoid output gate.
# Transcriptions of the inference pieces (as in test_gdn.py).
# ----------------------------------------------------------------------------
def _inference_reference_delta_rule(query, key, value, g, beta):
    def l2norm(x, dim=-1, eps=1e-6):
        inv_norm = 1 / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm
    query = l2norm(query)
    key = l2norm(key)
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    query, key, value = query.float(), key.float(), value.float()
    beta, g = beta.float(), g.float()
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, i, :], key[:, i, :], value[:, i, :]
        g_t = g[:, i, :].exp().unsqueeze(-1)
        beta_t = beta[:, i, :].unsqueeze(-1)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        v_t = v_t - kv_mem * g_t
        upd = k_t.unsqueeze(-1) * v_t.unsqueeze(-2) * beta_t.unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t.unsqueeze(-1) + upd
        core_attn_out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2) * scale
    return core_attn_out


def _inference_reference_conv(x, weight, bias):
    bsz, dim, seq_len = x.shape
    conv_kernel_size = weight.shape[-1]
    conv_state = torch.zeros(bsz, dim, conv_kernel_size, dtype=x.dtype)
    y = torch.cat([conv_state[:, :, :conv_kernel_size], x], dim=-1).to(weight.dtype)
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding=0, groups=dim)
    return F.silu(y[:, :, -seq_len:]).to(x.dtype)


def _inference_gated_rmsnorm_sigmoid(x, weight, gate, eps):
    """GatedRMSNorm.forward, gate_activation = "sigmoid" torch path: strict-fp32
    weighted norm, then * sigmoid(gate)."""
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight.to(torch.float32) * h
    return h * torch.sigmoid(gate.to(torch.float32))


def _build_hc_gdn_block(d, nk, dk, grp, dv, inter, H, rank, gate_act, r=0,
                        dtype=torch.float32):
    nv = nk * grp
    k_dim, v_dim = nk * dk, nv * dv
    conv_dim = 2 * k_dim + v_dim
    kernel = 4
    lins = {
        "qkv": MockLinear(d, conv_dim, "blk.linear_attn.in_proj_qkv", dtype=dtype),
        "z": MockLinear(d, v_dim, "blk.linear_attn.in_proj_z", dtype=dtype),
        "b": MockLinear(d, nv, "blk.linear_attn.in_proj_b", dtype=dtype),
        "a": MockLinear(d, nv, "blk.linear_attn.in_proj_a", dtype=dtype),
        "o": MockLinear(v_dim, d, "blk.linear_attn.out_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.mlp.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.mlp.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.mlp.down_proj", dtype=dtype),
    }
    gdn_norm_w = nn.Parameter(1.0 + 0.02 * torch.randn(dv, dtype=dtype), requires_grad=False)
    entry = types.SimpleNamespace()
    entry.attn_norm_spec = entry.mlp_norm_spec = None
    entry.attn_post_spec = entry.mlp_post_spec = None
    entry.attn_hc_spec = _make_hc_spec(H, d, rank, dtype=dtype)
    entry.mlp_hc_spec = _make_hc_spec(H, d, rank, dtype=dtype)
    entry.gdn_norm_spec = dict(_spec(gdn_norm_w, eps=1e-6), gate_activation=gate_act)
    entry.qkv_proj = DiffLinear(lins["qkv"], r=r, compute_dtype=dtype)
    entry.z_proj = DiffLinear(lins["z"], r=r, compute_dtype=dtype)
    entry.b_proj = DiffLinear(lins["b"], r=0, compute_dtype=dtype)
    entry.a_proj = DiffLinear(lins["a"], r=0, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=r, compute_dtype=dtype)
    entry.gates = [DiffLinear(lins["gate"], r=0, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=0, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=0, compute_dtype=dtype)]
    meta = {
        "kind": "gdn",
        "num_k_heads": nk, "num_v_heads": nv,
        "k_head_dim": dk, "v_head_dim": dv,
        "conv_kernel_size": kernel, "beta_scale": 1.0,
        "a_log": torch.randn(nv, dtype=dtype) * 0.3,
        "dt_bias": torch.randn(nv, dtype=dtype) * 0.3,
        "conv1d_weight": (torch.randn(conv_dim, kernel, dtype=dtype) * 0.4),
        "conv1d_bias": torch.randn(conv_dim, dtype=dtype) * 0.1,
        "activation": "silu", "mlp_kind": "dense", "layer_scalar": None,
    }
    ref_w = {k_: v_.frozen_weight for k_, v_ in lins.items()}
    ref_w["gdn_norm"] = gdn_norm_w
    return entry, meta, ref_w, lins


def _ref_gdn_sublayer(meta, w, normed, gate_act):
    b, t, d = normed.shape
    nk, nv = meta["num_k_heads"], meta["num_v_heads"]
    dk, dv = meta["k_head_dim"], meta["v_head_dim"]
    grp = nv // nk
    k_dim, v_dim = nk * dk, nv * dv
    qkv = normed @ w["qkv"]
    z = (normed @ w["z"]).view(b, t, nv, dv)
    beta = torch.sigmoid((normed @ w["b"]).float()) * meta["beta_scale"]
    g = -meta["a_log"].float().exp() * F.softplus((normed @ w["a"]).float() + meta["dt_bias"].float())
    x = _inference_reference_conv(qkv.transpose(1, 2), meta["conv1d_weight"],
                                  meta["conv1d_bias"]).transpose(1, 2)
    q, k, v = torch.split(x, [k_dim, k_dim, v_dim], dim=-1)
    q = q.view(b, t, nk, dk).repeat_interleave(grp, dim=2)
    k = k.view(b, t, nk, dk).repeat_interleave(grp, dim=2)
    v = v.view(b, t, nv, dv)
    core = _inference_reference_delta_rule(q, k, v, g, beta)
    if gate_act == "sigmoid":
        core = _inference_gated_rmsnorm_sigmoid(core, w["gdn_norm"], z, 1e-6)
    else:
        var = core.float().pow(2).mean(-1, keepdim=True) + 1e-6
        core = core.float() * torch.rsqrt(var) * w["gdn_norm"].float() * F.silu(z.float())
    return core.reshape(b, t, v_dim) @ w["o"]


def test_gdn_gated_rmsnorm_sigmoid_matches_inference():
    torch.manual_seed(3)
    b, t, nv, dv = 2, 5, 3, 8
    x = torch.randn(b, t, nv, dv)
    gate = torch.randn(b, t, nv, dv)
    w = 1.0 + 0.1 * torch.randn(dv)
    out = _gdn.gdn_gated_rmsnorm(x, dict(_spec(w, eps=1e-6), gate_activation="sigmoid"), gate)
    ref = _inference_gated_rmsnorm_sigmoid(x, w, gate, 1e-6)
    err = (out - ref).abs().max().item()
    assert err < 1e-6, f"sigmoid gated norm mismatch: max|d|={err}"
    silu = _gdn.gdn_gated_rmsnorm(x, dict(_spec(w, eps=1e-6), gate_activation="silu"), gate)
    assert not torch.allclose(out, silu, atol=1e-3), "sigmoid and silu gates coincide?"
    # Default (no key) is silu -- the Qwen3.5/3.6 behaviour, unchanged.
    assert torch.equal(_gdn.gdn_gated_rmsnorm(x, _spec(w, eps=1e-6), gate), silu)
    try:
        _gdn.gdn_gated_rmsnorm(x, dict(_spec(w, eps=1e-6), gate_activation="softplus"), gate)
        assert False, "unknown gate activation accepted"
    except AssertionError as e:
        assert "softplus" in str(e)
    print(f"[hc] GDN sigmoid output gate matches GatedRMSNorm (max|d|={err:.2e}) PASSED")


def test_hc_gdn_block_sigmoid_gate_matches_reference():
    torch.manual_seed(4)
    d, nk, dk, grp, dv, inter, H, rank = 16, 2, 8, 2, 6, 32, 4, 8
    net = _headless_net()
    b, t = 2, 6
    streams = torch.randn(b, t, H, d)
    outs = {}
    for act in ("sigmoid", "silu"):
        torch.manual_seed(4)
        entry, meta, w, _ = _build_hc_gdn_block(d, nk, dk, grp, dv, inter, H, rank, act)
        out = net._gdn_forward(meta, entry, streams)
        ref = _ref_hc_block(entry, streams,
                            lambda y: _ref_gdn_sublayer(meta, w, y, act),
                            lambda y: _ref_swiglu(w, y))
        err = (out - ref).abs().max().item()
        assert err < 1e-4, f"hc GDN block ({act}) mismatch vs reference: max|d|={err}"
        outs[act] = out
    assert not torch.allclose(outs["sigmoid"], outs["silu"], atol=1e-3), \
        "gate activation had no effect on the block output"
    print("[hc] GDN block on the stream stack (sigmoid + silu gates) matches reference PASSED")


def test_hc_block_backward_reaches_adapters_and_streams():
    torch.manual_seed(5)
    d, nq, nkv, hd, inter, H, rank, r = 12, 2, 1, 6, 24, 4, 4, 3
    entry, meta, _, lins = _build_hc_attn_block(d, nq, nkv, hd, inter, H, rank, r=r)
    net = _headless_net()
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
    assert streams.grad is not None and streams.grad.abs().sum() > 0
    # Gradient reaches EVERY stream (the mix reads all of them).
    assert (streams.grad.abs().amax(dim=(0, 1, 3)) > 0).all()
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen_before[k_])
    for s in ("attn_hc_spec", "mlp_hc_spec"):
        for k_, v in getattr(entry, s).items():
            if torch.is_tensor(v):
                assert torch.equal(v, hc_before[(s, k_)]) and v.grad is None
    print("[hc] backward reaches every adapter and all four streams; base + hc tables frozen PASSED")


# ----------------------------------------------------------------------------
# Trunk wiring: ExpandStreams -> hc blocks -> final mixer (headless net).
# ----------------------------------------------------------------------------
def _make_trunk_net(entries, metas, final_spec, table, H):
    net = _headless_net()
    nn.Module.__init__(net)
    net.compute_dtype = torch.float32
    net.attn_impl = "eager"
    net._flash_ok = False
    net.gradient_checkpointing = False
    net.use_liger = False
    net.offload_activations = False
    net.head_pre_scale = 1.0
    net._adapters_off = False
    net._qa_state = None
    net.embed_weight = None
    net.embed_lora_a = net.embed_lora_b = None
    net.embed = types.SimpleNamespace(embedding=table)
    net.embed_norm_spec = None
    net.blocks = entries
    net._block_meta = metas
    net._block_devices = [torch.device("cpu")] * len(entries)
    net._deepstack = {}
    net._bidir_mm = False
    net.has_gdn = net.has_shortconv = net.has_ple = False
    net.has_mrope = False
    net.hc_mult = H
    net.final_mixer_spec = final_spec
    net.final_norm_spec = None
    net._ple = {}
    net._ple_modules = {}
    net.qsa_threshold = None
    net.device = torch.device("cpu")
    net.eval()
    return net


def test_trunk_expand_blocks_final_mixer():
    torch.manual_seed(6)
    d, nq, nkv, hd, inter, H, rank, V = 16, 4, 2, 8, 32, 4, 8, 40
    e0, m0, w0, _ = _build_hc_attn_block(d, nq, nkv, hd, inter, H, rank)
    e1, m1, w1, _ = _build_hc_attn_block(d, nq, nkv, hd, inter, H, rank)
    final_spec = _make_hc_spec(H, d, rank, site=False)
    table = nn.Embedding(V, d)
    table.weight.requires_grad_(False)
    net = _make_trunk_net([e0, e1], [m0, m1], final_spec, table, H)
    b, t = 2, 7
    ids = torch.randint(0, V, (b, t))
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    with torch.no_grad():
        out, ctx = net._forward_trunk(ids)
    assert out.shape == (b, t, d) and out.dtype == torch.float32
    assert ctx["trunk_state"] is out

    # Hand-composed chain from the reference pieces.
    streams = table.weight[ids].float().unsqueeze(2).expand(-1, -1, H, -1)
    for e, m, w in ((e0, m0, w0), (e1, m1, w1)):
        streams = _ref_hc_block(e, streams,
                                lambda y, m=m, w=w: _ref_attention(m, w, y, positions),
                                lambda y, w=w: _ref_swiglu(w, y))
    _, ref = _ref_mix_spec(streams, final_spec)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"trunk (expand -> hc blocks -> final mixer) mismatch: max|d|={err}"

    # The EBFT taps read the stream MEAN after a block (the inference export).
    with net.collect_hidden([0]):
        with torch.no_grad():
            net._forward_trunk(ids)
    tap = net.collected[0]
    s0 = table.weight[ids].float().unsqueeze(2).expand(-1, -1, H, -1)
    s0 = _ref_hc_block(e0, s0, lambda y: _ref_attention(m0, w0, y, positions),
                       lambda y: _ref_swiglu(w0, y))
    assert tap.shape == (b, t, d)
    assert (tap - s0.mean(dim=2)).abs().max().item() < 1e-4

    # QSA length guard: refuse past the threshold, accept at it.
    net.qsa_threshold = t
    with torch.no_grad():
        net._forward_trunk(ids)
    net.qsa_threshold = t - 1
    try:
        with torch.no_grad():
            net._forward_trunk(ids)
        assert False, "sequence past the QSA threshold was accepted"
    except ValueError as e:
        assert "QSA" in str(e)
    print(f"[hc] trunk wiring expand -> blocks -> final mixer matches reference "
          f"(max|d|={err:.2e}); taps read the stream mean; QSA guard PASSED")


def main():
    from util import run_timed
    run_timed([
        test_mix_matches_inference_reference,
        test_mix_apply_gradcheck,
        test_expand_streams,
        test_hc_attention_block_matches_reference,
        test_gdn_gated_rmsnorm_sigmoid_matches_inference,
        test_hc_gdn_block_sigmoid_gate_matches_reference,
        test_hc_block_backward_reaches_adapters_and_streams,
        test_trunk_expand_blocks_final_mixer,
    ], label="hyperconnections-train")
    print("\nAll gated-residual hyper-connection training checks passed.")


if __name__ == "__main__":
    main()
