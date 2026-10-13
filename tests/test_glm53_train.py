"""
CPU tests for the GLM-5.3-Flash training pieces: KDA linear attention
(``training.gdn`` KDA section + ``native_llama._gdn_forward`` kda branch)
and NoPE MLA (``native_llama._mla_qkv`` + the shared attention core).

No GPU, no compiled extension, no real model (scaffolding from
``test_gdn.py`` / ``test_hyperconnections_train.py``). Checks:

  * ``kda_delta_rule_reference`` matches a verbatim transcription of the
    inference reference ``torch_recurrent_kda`` and gradchecks in fp64;
  * ``kda_beta_g`` reproduces the inference module's gate expressions (safe
    gate with a lower bound, and the softplus form without);
  * a KDA block forward (``_gdn_forward`` with ``meta["kda"]``) matches an
    independent composition: norm -> qkv / b / f_a->f_b / g_a->g_b -> conv+silu
    -> KDA rule -> gated norm * sigmoid(z) -> o_proj, then SwiGLU;
  * a NoPE MLA block (``_block_forward`` with ``kind == "mla"``) matches:
    q_b(norm(q_a(x))), c = norm(kv_a(x)), K = c W_UK, V = c W_UV per head,
    dense causal softmax attention at sm_scale, o_proj, then SwiGLU -- on the
    eager path and with v_head_dim != qk_head_dim;
  * backprop through both block kinds reaches the LoRA adapters (incl. the
    explicitly-named KDA gate projections) and leaves the frozen tables alone.

Run:  python tests/test_glm53_train.py
"""

from __future__ import annotations
import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_gdn as tg   # noqa: E402  (loads the training modules)

_gdn = tg._gdn
_nl = tg._nl
DiffLinear = tg.DiffLinear
MockLinear = tg.MockLinear
_spec = tg._spec


# ----------------------------------------------------------------------------
# Verbatim transcription of torch_recurrent_kda (modules/gated_delta_net_fn/
# gated_delta_rule.py), minus the bf16 output cast.
# ----------------------------------------------------------------------------
def _inference_reference_kda(q, k, v, g, beta):
    b, s, h, dk = k.shape
    dv = v.shape[-1]
    q, k, v, g, beta = [t.float() for t in (q, k, v, g, beta)]

    def l2norm(x):
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)
    q = l2norm(q) * (dk ** -0.5)
    k = l2norm(k)
    state = torch.zeros((b, h, dk, dv), dtype=torch.float, device=v.device)
    out = torch.empty((b, s, h, dv), dtype=torch.float, device=v.device)
    for i in range(s):
        g_i = g[:, i].exp().unsqueeze(-1)
        state = state * g_i
        kv_mem = (state * k[:, i].unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, i] - kv_mem) * beta[:, i].unsqueeze(-1)
        state = state + k[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q[:, i].unsqueeze(-1)).sum(dim=-2)
    return out


def test_kda_rule_matches_inference_reference():
    torch.manual_seed(0)
    b, t, h, dk, dv = 2, 7, 3, 8, 6
    q, k, v = torch.randn(b, t, h, dk), torch.randn(b, t, h, dk), torch.randn(b, t, h, dv)
    g = -torch.rand(b, t, h, dk) * 2.0
    beta = torch.sigmoid(torch.randn(b, t, h))
    out = _gdn.kda_delta_rule_reference(q, k, v, g, beta)
    ref = _inference_reference_kda(q, k, v, g, beta)
    err = (out - ref).abs().max().item()
    assert out.shape == (b, t, h, dv) and err < 1e-5, f"KDA rule mismatch: {err}"
    # fp64 gradcheck
    args = [x.double().requires_grad_(True) for x in (q[:1, :4], k[:1, :4], v[:1, :4], g[:1, :4], beta[:1, :4])]
    assert torch.autograd.gradcheck(_gdn.kda_delta_rule_reference, args, eps=1e-6, atol=1e-5)
    print(f"[glm53] KDA delta rule matches torch_recurrent_kda (max|d|={err:.2e}) + gradcheck PASSED")


def test_kda_beta_g_matches_inference():
    torch.manual_seed(1)
    b, t, nv, dk = 2, 5, 3, 4
    b_raw = torch.randn(b, t, nv)
    f_raw = torch.randn(b, t, nv * dk)
    a_log = torch.randn(nv) * 0.3
    dt_bias = torch.randn(nv * dk) * 0.3
    # Inference (GatedDeltaNet.forward, kda branch)
    gf = (f_raw.float() + dt_bias.float().view(1, 1, -1)).view(b, t, nv, dk)
    decay = torch.exp(a_log.float()).view(1, 1, nv, 1)
    for lb in (-5.0, None):
        beta, g = _gdn.kda_beta_g(b_raw, f_raw, a_log, dt_bias, lb, 1.0)
        ref_beta = torch.sigmoid(b_raw.float())
        ref_g = torch.sigmoid(gf * decay) * lb if lb is not None else F.softplus(gf, threshold=20.0) * (-decay)
        assert (beta - ref_beta).abs().max() < 1e-6
        assert g.shape == (b, t, nv, dk) and (g - ref_g).abs().max() < 1e-6, f"g mismatch (lb={lb})"
        assert (g <= 0).all()
    print("[glm53] KDA beta / per-channel decay (safe gate + softplus forms) PASSED")


# ----------------------------------------------------------------------------
# KDA block.
# ----------------------------------------------------------------------------
def _build_kda_block(d, nh, dh, inter, rank, r=0, r_gate=0, dtype=torch.float32):
    k_dim = v_dim = nh * dh
    conv_dim = 2 * k_dim + v_dim
    kernel = 4
    lins = {
        "qkv": MockLinear(d, conv_dim, "blk.self_attn.qkv_proj", dtype=dtype),
        "b": MockLinear(d, nh, "blk.self_attn.b_proj", dtype=dtype),
        "f_a": MockLinear(d, rank, "blk.self_attn.f_a_proj", dtype=dtype),
        "f_b": MockLinear(rank, k_dim, "blk.self_attn.f_b_proj", dtype=dtype, scale=0.3),
        "g_a": MockLinear(d, rank, "blk.self_attn.g_a_proj", dtype=dtype),
        "g_b": MockLinear(rank, v_dim, "blk.self_attn.g_b_proj", dtype=dtype, scale=0.3),
        "o": MockLinear(v_dim, d, "blk.self_attn.o_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.mlp.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.mlp.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.mlp.down_proj", dtype=dtype),
    }
    norm_a = 1.0 + 0.02 * torch.randn(d, dtype=dtype)
    norm_m = 1.0 + 0.02 * torch.randn(d, dtype=dtype)
    gdn_norm_w = 1.0 + 0.02 * torch.randn(dh, dtype=dtype)
    entry = types.SimpleNamespace()
    entry.attn_norm_spec = _spec(norm_a)
    entry.mlp_norm_spec = _spec(norm_m)
    entry.attn_post_spec = entry.mlp_post_spec = None
    entry.attn_hc_spec = entry.mlp_hc_spec = None
    entry.gdn_norm_spec = {**_spec(gdn_norm_w, eps=1e-6), "gate_activation": "sigmoid"}
    entry.qkv_proj = DiffLinear(lins["qkv"], r=r, compute_dtype=dtype)
    entry.b_proj = DiffLinear(lins["b"], r=0, compute_dtype=dtype)
    entry.f_a_proj = DiffLinear(lins["f_a"], r=r_gate, compute_dtype=dtype)
    entry.f_b_proj = DiffLinear(lins["f_b"], r=0, compute_dtype=dtype)
    entry.g_a_proj = DiffLinear(lins["g_a"], r=0, compute_dtype=dtype)
    entry.g_b_proj = DiffLinear(lins["g_b"], r=r_gate, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=r, compute_dtype=dtype)
    entry.z_proj = entry.a_proj = None
    entry.gates = [DiffLinear(lins["gate"], r=0, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=0, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=0, compute_dtype=dtype)]
    meta = {
        "kind": "gdn", "kda": True, "gate_lower_bound": -5.0,
        "num_k_heads": nh, "num_v_heads": nh, "k_head_dim": dh, "v_head_dim": dh,
        "conv_kernel_size": kernel, "beta_scale": 1.0,
        "a_log": torch.randn(nh, dtype=dtype) * 0.3,
        "dt_bias": torch.randn(nh * dh, dtype=dtype) * 0.3,
        "conv1d_weight": torch.randn(conv_dim, kernel, dtype=dtype) * 0.4,
        "conv1d_bias": None,
        "activation": "silu", "act_limit": 0.0, "mlp_kind": "dense", "layer_scalar": None,
    }
    ref_w = {k_: v_.frozen_weight for k_, v_ in lins.items()}
    ref_w.update({"attn_norm": norm_a, "mlp_norm": norm_m, "gdn_norm": gdn_norm_w})
    return entry, meta, ref_w, lins


def _ref_kda_block(meta, w, hidden):
    b, t, d = hidden.shape
    nh, dh = meta["num_v_heads"], meta["v_head_dim"]
    k_dim = v_dim = nh * dh
    normed = tg._ref_rmsnorm(hidden, w["attn_norm"])
    qkv = normed @ w["qkv"]
    z = ((normed @ w["g_a"]) @ w["g_b"]).view(b, t, nh, dh)
    beta = torch.sigmoid((normed @ w["b"]).float())
    f = ((normed @ w["f_a"]) @ w["f_b"]).float() + meta["dt_bias"].float()
    g = torch.sigmoid(f.view(b, t, nh, dh) * meta["a_log"].float().exp().view(1, 1, nh, 1)) * meta["gate_lower_bound"]
    x = tg._inference_reference_conv(qkv.transpose(1, 2), meta["conv1d_weight"], None).transpose(1, 2)
    q, k, v = torch.split(x, [k_dim, k_dim, v_dim], dim=-1)
    core = _inference_reference_kda(q.view(b, t, nh, dh), k.view(b, t, nh, dh), v.view(b, t, nh, dh), g, beta)
    var = core.pow(2).mean(-1, keepdim=True) + 1e-6
    core = core * torch.rsqrt(var) * w["gdn_norm"].float() * torch.sigmoid(z.float())
    hidden = hidden + core.reshape(b, t, v_dim) @ w["o"]
    normed2 = tg._ref_rmsnorm(hidden, w["mlp_norm"])
    return hidden + (F.silu(normed2 @ w["gate"]) * (normed2 @ w["up"])) @ w["down"]


def test_kda_block_matches_reference():
    torch.manual_seed(2)
    d, nh, dh, inter, rank = 16, 2, 8, 32, 4
    entry, meta, ref_w, _ = _build_kda_block(d, nh, dh, inter, rank)
    net = tg._headless_net()
    b, t = 2, 6
    hidden = torch.randn(b, t, d)
    with torch.no_grad():
        out = net._gdn_forward(meta, entry, hidden)
    ref = _ref_kda_block(meta, ref_w, hidden)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"KDA block forward mismatch: {err}"
    print(f"[glm53] KDA block forward matches reference (max|d|={err:.2e}) PASSED")


def test_kda_block_backward_reaches_adapters():
    torch.manual_seed(3)
    d, nh, dh, inter, rank = 12, 2, 4, 24, 4
    entry, meta, _, lins = _build_kda_block(d, nh, dh, inter, rank, r=3, r_gate=2)
    net = tg._headless_net()
    frozen_before = {k_: l.frozen_weight.clone() for k_, l in lins.items()}
    adapted = [entry.qkv_proj, entry.f_a_proj, entry.g_b_proj, entry.o_proj]
    with torch.no_grad():
        for w in adapted:
            w.lora_b.normal_(std=0.05)
    hidden = torch.randn(2, 5, d, requires_grad=True)
    out = net._gdn_forward(meta, entry, hidden)
    out.pow(2).mean().backward()
    for w in adapted:
        assert w.lora_a.grad is not None and w.lora_b.grad.abs().sum() > 0, w.linear.key
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen_before[k_]) and l.frozen_weight.grad is None
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    print("[glm53] KDA backward reaches qkv / f_a / g_b / o adapters; base frozen PASSED")


# ----------------------------------------------------------------------------
# NoPE MLA block.
# ----------------------------------------------------------------------------
def _build_mla_block(d, H, nope, hd_v, D_c, q_rank, inter, r=0, dtype=torch.float32):
    lins = {
        "q_a": MockLinear(d, q_rank, "blk.self_attn.q_a_proj", dtype=dtype),
        "q_b": MockLinear(q_rank, H * nope, "blk.self_attn.q_b_proj", dtype=dtype),
        "kv_a": MockLinear(d, D_c, "blk.self_attn.kv_a_proj_with_mqa", dtype=dtype),
        "o": MockLinear(H * hd_v, d, "blk.self_attn.o_proj", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.mlp.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.mlp.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.mlp.down_proj", dtype=dtype),
    }
    norm_a = 1.0 + 0.02 * torch.randn(d, dtype=dtype)
    norm_m = 1.0 + 0.02 * torch.randn(d, dtype=dtype)
    norm_qa = 1.0 + 0.05 * torch.randn(q_rank, dtype=dtype)
    norm_kva = 1.0 + 0.05 * torch.randn(D_c, dtype=dtype)
    entry = types.SimpleNamespace()
    entry.attn_norm_spec = _spec(norm_a)
    entry.mlp_norm_spec = _spec(norm_m)
    entry.attn_post_spec = entry.mlp_post_spec = None
    entry.attn_hc_spec = entry.mlp_hc_spec = None
    entry.q_a_norm_spec = _spec(norm_qa, eps=1e-6)
    entry.kv_a_norm_spec = _spec(norm_kva, eps=1e-6)
    entry.q_norm_spec = entry.k_norm_spec = entry.v_norm_spec = None
    entry.q_a_proj = DiffLinear(lins["q_a"], r=r, compute_dtype=dtype)
    entry.q_b_proj = DiffLinear(lins["q_b"], r=r, compute_dtype=dtype)
    entry.kv_a_proj = DiffLinear(lins["kv_a"], r=r, compute_dtype=dtype)
    entry.o_proj = DiffLinear(lins["o"], r=r, compute_dtype=dtype)
    entry.g_proj = None
    entry.gates = [DiffLinear(lins["gate"], r=0, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=0, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=0, compute_dtype=dtype)]
    w_uk = (torch.randn(D_c, H * nope, dtype=dtype) * 0.1)
    w_uv = (torch.randn(D_c, H * hd_v, dtype=dtype) * 0.1)
    meta = {
        "kind": "mla", "num_q_heads": H, "num_kv_heads": H, "head_dim": nope, "v_head_dim": hd_v,
        "kv_lora_rank": D_c, "qk_nope_head_dim": nope, "qk_rope_head_dim": 0,
        "sm_scale": nope ** -0.5, "w_uk": w_uk, "w_uv": w_uv,
        "inv_freq": None, "attn_factor": 1.0, "mrope_section": None, "sliding_window": -1,
        "softcap": 0.0, "activation": "silu", "act_limit": 0.0, "mlp_kind": "dense",
        "use_k_as_v": False, "interleaved_gate": False, "full_gate": False,
        "headwise_gate": False, "layer_scalar": None, "qsa_threshold": 2048,
    }
    ref_w = {k_: v_.frozen_weight for k_, v_ in lins.items()}
    ref_w.update({"attn_norm": norm_a, "mlp_norm": norm_m, "q_a_norm": norm_qa,
                  "kv_a_norm": norm_kva, "w_uk": w_uk, "w_uv": w_uv})
    return entry, meta, ref_w, lins


def _ref_mla_block(meta, w, hidden):
    b, t, d = hidden.shape
    H, nope, hd_v, D_c = meta["num_q_heads"], meta["qk_nope_head_dim"], meta["v_head_dim"], meta["kv_lora_rank"]
    normed = tg._ref_rmsnorm(hidden, w["attn_norm"])
    q = (tg._ref_rmsnorm(normed @ w["q_a"], w["q_a_norm"], eps=1e-6) @ w["q_b"]).view(b, t, H, nope)
    c = tg._ref_rmsnorm(normed @ w["kv_a"], w["kv_a_norm"], eps=1e-6)
    k = (c @ w["w_uk"]).view(b, t, H, nope)
    v = (c @ w["w_uv"]).view(b, t, H, hd_v)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) * meta["sm_scale"]
    mask = torch.triu(torch.full((t, t), float("-inf")), 1)
    ctx = (torch.softmax(scores + mask, -1) @ v).transpose(1, 2).reshape(b, t, H * hd_v)
    hidden = hidden + ctx @ w["o"]
    normed2 = tg._ref_rmsnorm(hidden, w["mlp_norm"])
    return hidden + (F.silu(normed2 @ w["gate"]) * (normed2 @ w["up"])) @ w["down"]


def test_mla_block_matches_reference():
    torch.manual_seed(4)
    net = tg._headless_net()
    for (nope, hd_v) in ((8, 8), (8, 6)):
        d, H, D_c, q_rank, inter = 16, 3, 10, 12, 32
        entry, meta, ref_w, _ = _build_mla_block(d, H, nope, hd_v, D_c, q_rank, inter)
        b, t = 2, 6
        hidden = torch.randn(b, t, d)
        positions = torch.arange(t).unsqueeze(0).expand(b, t)
        bias = net._attn_bias(None, t, torch.device("cpu"), torch.float32)
        with torch.no_grad():
            out = net._block_forward(meta, entry, hidden, positions, bias)
        ref = _ref_mla_block(meta, ref_w, hidden)
        err = (out - ref).abs().max().item()
        assert err < 1e-4, f"MLA block mismatch (nope={nope}, v={hd_v}): {err}"
        assert net._attn_mode_for(meta, True) == ("flash" if nope == hd_v else "eager")
    print("[glm53] NoPE MLA block (equal and unequal v head dim) matches reference PASSED")


def test_mla_block_backward_reaches_adapters():
    torch.manual_seed(5)
    d, H, nope, hd_v, D_c, q_rank, inter = 12, 2, 4, 4, 6, 8, 24
    entry, meta, _, lins = _build_mla_block(d, H, nope, hd_v, D_c, q_rank, inter, r=3)
    net = tg._headless_net()
    frozen_before = {k_: l.frozen_weight.clone() for k_, l in lins.items()}
    adapted = [entry.q_a_proj, entry.q_b_proj, entry.kv_a_proj, entry.o_proj]
    with torch.no_grad():
        for w in adapted:
            w.lora_b.normal_(std=0.05)
    b, t = 2, 5
    hidden = torch.randn(b, t, d, requires_grad=True)
    positions = torch.arange(t).unsqueeze(0).expand(b, t)
    bias = net._attn_bias(None, t, torch.device("cpu"), torch.float32)
    out = net._block_forward(meta, entry, hidden, positions, bias)
    out.pow(2).mean().backward()
    for w in adapted:
        assert w.lora_a.grad is not None and w.lora_b.grad.abs().sum() > 0, w.linear.key
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen_before[k_]) and l.frozen_weight.grad is None
    assert meta["w_uk"].grad is None and meta["w_uv"].grad is None
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    print("[glm53] MLA backward reaches q_a / q_b / kv_a / o adapters; base + kv_b tables frozen PASSED")


def main():
    from util import run_timed
    run_timed([
        test_kda_rule_matches_inference_reference,
        test_kda_beta_g_matches_inference,
        test_kda_block_matches_reference,
        test_kda_block_backward_reaches_adapters,
        test_mla_block_matches_reference,
        test_mla_block_backward_reaches_adapters,
    ], label="glm53-train")
    print("\nAll GLM-5.3-Flash (KDA + NoPE MLA) training checks passed.")


if __name__ == "__main__":
    main()
