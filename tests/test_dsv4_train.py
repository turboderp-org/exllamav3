"""
CPU tests for the DeepSeek-V4 training pieces: the DSV4 attention block
(``native_llama._dsv4_forward``: sliding / CSA / HCA layers with gated-pool
compressors, sink column, V == K, de-rotated output, grouped low-rank
o_proj) and the sqrt-softplus / hash-MoE routers (``_moe_out``).

No GPU, no compiled extension, no real model (scaffolding from
``test_moe.py`` / ``test_gdn.py``). The references are independent
transcriptions of the transformers ``modeling_deepseek_v4`` math (rotary with
interleaved pairs on the trailing slice, HCA / CSA gated pooling with the
position bias, the combined sliding + block-bias + sink softmax, TopK /
hash routing with ``sqrt(softplus)`` affinities). Checks:

  * ``_rope_gptj_trailing`` equals HF ``apply_rotary_pos_emb`` and the
    ``sign = -1`` call inverts it;
  * the HCA and CSA compressors equal the HF modules (incl. the -inf Ca
    half of window 0 and the remainder drop);
  * a sliding block past the window, an HCA block with several complete
    windows and a CSA block each match the HF-style composition end to end;
  * right-padding: pad keys are hidden from real queries;
  * backprop reaches the q_a / q_b / wkv / wo_a / wo_b adapters, the frozen
    compressor projections and tables stay untouched;
  * ``sqrtsp`` and ``sqrtsp_hash`` routing match the HF TopK / hash routers
    with the clamped experts and ungated shared expert.

Run:  python tests/test_dsv4_train.py
"""

from __future__ import annotations
import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_moe as tm   # noqa: E402  (loads the training modules)

_nl = tm._nl
DiffLinear = tm.DiffLinear
MockLinear = tm.MockLinear
_spec = tm._spec


# ----------------------------------------------------------------------------
# HF transcriptions (modeling_deepseek_v4).
# ----------------------------------------------------------------------------
def _hf_rope(x, inv_freq, pos, rd, sign=1.0):
    """apply_rotary_pos_emb on the trailing rd dims of x [b, t, h, D] (HF keeps
    heads at dim 1; the pairing math is position-wise identical)."""
    freqs = pos.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)      # [b,t,rd/2]
    cos = freqs.cos().repeat_interleave(2, -1).unsqueeze(2)
    sin = (sign * freqs.sin()).repeat_interleave(2, -1).unsqueeze(2)
    nope, rope = x[..., :-rd], x[..., -rd:].float()
    rot = torch.stack((-rope[..., 1::2], rope[..., 0::2]), -1).flatten(-2)
    return torch.cat((nope, (rope * cos + rot * sin).to(x.dtype)), -1)


def _hf_rmsnorm(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()) if w is not None else y


def _hf_hca(x, w_kv, w_gate, ape, norm_w, eps, m, hd, inv_freq, rope_dim):
    b, S, _ = x.shape
    kv, g = x @ w_kv, x @ w_gate
    nw = S // m
    if nw == 0:
        return None
    kv_w = kv[:, :nw * m].view(b, nw, m, hd)
    g_w = g[:, :nw * m].view(b, nw, m, hd) + ape
    C = (torch.softmax(g_w.float(), dim=2) * kv_w.float()).sum(2)
    C = _hf_rmsnorm(C, norm_w, eps)
    wpos = (torch.arange(nw) * m).unsqueeze(0).expand(b, nw)
    return _hf_rope(C.unsqueeze(2), inv_freq, wpos, rope_dim).squeeze(2)


def _hf_csa(x, w_kv, w_gate, ape, norm_w, eps, m, hd, inv_freq, rope_dim):
    b, S, _ = x.shape
    kv, g = x @ w_kv, x @ w_gate                                      # [b, S, 2hd]
    nw = S // m
    if nw == 0:
        return None
    kv = kv[:, :nw * m].view(b, nw, m, 2 * hd)
    g = g[:, :nw * m].view(b, nw, m, 2 * hd) + ape
    new_kv = kv.new_zeros(b, nw, 2 * m, hd)
    new_g = g.new_full((b, nw, 2 * m, hd), -float("inf"))
    new_kv[:, :, m:] = kv[..., hd:]
    new_g[:, :, m:] = g[..., hd:]
    if nw > 1:
        new_kv[:, 1:, :m] = kv[:, :-1, :, :hd]
        new_g[:, 1:, :m] = g[:, :-1, :, :hd]
    C = (torch.softmax(new_g.float(), dim=2) * new_kv.float()).sum(2)
    C = _hf_rmsnorm(C, norm_w, eps)
    wpos = (torch.arange(nw) * m).unsqueeze(0).expand(b, nw)
    return _hf_rope(C.unsqueeze(2), inv_freq, wpos, rope_dim).squeeze(2)


def _hf_dsv4_attention(w, meta, x, pos, pad_mask=None):
    """DeepseekV4Attention.forward + eager attention, from the normed input x."""
    b, S, _ = x.shape
    H, D, rd, G, m = meta["num_q_heads"], meta["head_dim"], meta["rope_head_dim"], meta["o_groups"], meta["compress_rate"]
    q_res = _hf_rmsnorm(x @ w["q_a"], w["q_a_norm"], 1e-6)
    q = (q_res @ w["q_b"]).view(b, S, H, D)
    q = _hf_rmsnorm(q, None, 1e-6)
    q = _hf_rope(q, meta["inv_freq"], pos, rd)
    kv = _hf_rmsnorm(x @ w["wkv"], w["kv_norm"], 1e-6).view(b, S, 1, D)
    kv = _hf_rope(kv, meta["inv_freq"], pos, rd).squeeze(2)
    i = torch.arange(S).unsqueeze(1)
    j = torch.arange(S).unsqueeze(0)
    allow = ((j <= i) & ((i - j) < meta["sliding_window"])).unsqueeze(0).expand(b, S, S)
    if pad_mask is not None:
        allow = allow & pad_mask.bool().unsqueeze(1)
    keys = kv
    if meta["layer_type"] != "sliding":
        comp = meta["compressor"]
        fn = _hf_csa if comp["overlapping"] else _hf_hca
        C = fn(x, w["c_kv"], w["c_gate"], comp["ape"], w["c_norm"], comp["norm"]["eps"], m, D,
               meta["inv_freq_compress"], comp["rope_dim"])
        if C is not None:
            T = C.shape[1]
            vis = torch.arange(T).view(1, 1, T) < ((pos + 1) // m).unsqueeze(-1)
            allow = torch.cat((allow, vis), -1)
            keys = torch.cat((kv, C), 1)
    mask = torch.zeros(allow.shape).masked_fill(~allow, float("-inf")).unsqueeze(1)
    logits = (q.transpose(1, 2).float() @ keys.float().unsqueeze(1).transpose(-1, -2)) * (D ** -0.5) + mask
    sinks = meta["sinks"].view(1, H, 1, 1).expand(b, H, S, 1)
    p = torch.softmax(torch.cat((logits, sinks), -1), -1)[..., :-1]
    o = (p @ keys.float().unsqueeze(1)).transpose(1, 2)                  # [b, S, H, D]
    o = _hf_rope(o, meta["inv_freq"], pos, rd, sign=-1.0)
    grouped = o.reshape(b, S, G, (H // G) * D)
    y = torch.cat([grouped[:, :, g] @ w["wo_a"][g] for g in range(G)], -1)
    return y @ w["wo_b"]


# ----------------------------------------------------------------------------
# Block builder.
# ----------------------------------------------------------------------------
def _build_dsv4_block(d, H, D, rd, G, q_rank, o_rank, layer_type, m, inter, window=4,
                      r=0, dtype=torch.float32):
    hpg = H // G
    lins = {
        "q_a": MockLinear(d, q_rank, "blk.attn.wq_a", dtype=dtype),
        "q_b": MockLinear(q_rank, H * D, "blk.attn.wq_b", dtype=dtype),
        "wkv": MockLinear(d, D, "blk.attn.wkv", dtype=dtype),
        "wo_b": MockLinear(G * o_rank, d, "blk.attn.wo_b", dtype=dtype),
        "gate": MockLinear(d, inter, "blk.ffn.gate_proj", dtype=dtype),
        "up": MockLinear(d, inter, "blk.ffn.up_proj", dtype=dtype),
        "down": MockLinear(inter, d, "blk.ffn.down_proj", dtype=dtype),
    }
    wo_a = [MockLinear(hpg * D, o_rank, f"blk.attn.wo_a.slice.{g}", dtype=dtype) for g in range(G)]
    entry = types.SimpleNamespace()
    norm_a, norm_m = 1.0 + 0.05 * torch.randn(d), 1.0 + 0.05 * torch.randn(d)
    entry.attn_norm_spec, entry.mlp_norm_spec = _spec(norm_a), _spec(norm_m)
    entry.attn_post_spec = entry.mlp_post_spec = None
    entry.attn_hc_spec = entry.mlp_hc_spec = None
    q_a_norm, kv_norm = 1.0 + 0.05 * torch.randn(q_rank), 1.0 + 0.05 * torch.randn(D)
    entry.q_a_norm_spec = _spec(q_a_norm, eps=1e-6)
    entry.kv_norm_spec = _spec(kv_norm, eps=1e-6)
    entry.q_a_proj = DiffLinear(lins["q_a"], r=r, compute_dtype=dtype)
    entry.q_b_proj = DiffLinear(lins["q_b"], r=r, compute_dtype=dtype)
    entry.wkv = DiffLinear(lins["wkv"], r=r, compute_dtype=dtype)
    entry.wo_a = nn.ModuleList([DiffLinear(l, r=r, compute_dtype=dtype) for l in wo_a])
    entry.wo_b = DiffLinear(lins["wo_b"], r=r, compute_dtype=dtype)
    entry.gates = [DiffLinear(lins["gate"], r=0, compute_dtype=dtype)]
    entry.ups = [DiffLinear(lins["up"], r=0, compute_dtype=dtype)]
    entry.downs = [DiffLinear(lins["down"], r=0, compute_dtype=dtype)]
    inv_main = 1.0 / (10000.0 ** (torch.arange(0, rd, 2).float() / rd))
    inv_comp = 1.0 / (160000.0 ** (torch.arange(0, rd, 2).float() / rd)) * 0.7
    comp = None
    if layer_type != "sliding":
        W = 2 * D if layer_type == "csa" else D
        lins["c_kv"] = MockLinear(d, W, "blk.attn.compressor.wkv", dtype=dtype)
        lins["c_gate"] = MockLinear(d, W, "blk.attn.compressor.wgate", dtype=dtype)
        entry.comp_wkv = DiffLinear(lins["c_kv"], r=0, compute_dtype=dtype)
        entry.comp_wgate = DiffLinear(lins["c_gate"], r=0, compute_dtype=dtype)
        c_norm = 1.0 + 0.05 * torch.randn(D)
        comp = {"overlapping": layer_type == "csa", "head_dim": D, "rope_dim": min(rd, D),
                "ape": torch.randn(m, W) * 0.5, "norm": _spec(c_norm, eps=1e-6)}
    else:
        entry.comp_wkv = entry.comp_wgate = None
    meta = {
        "kind": "dsv4", "layer_type": layer_type, "num_q_heads": H, "num_kv_heads": 1,
        "head_dim": D, "v_head_dim": D, "rope_head_dim": rd, "o_groups": G, "sm_scale": D ** -0.5,
        "sliding_window": window, "sinks": torch.randn(H) * 0.5,
        "inv_freq": inv_main if layer_type == "sliding" else inv_comp, "inv_freq_compress": inv_comp,
        "compress_rate": m if layer_type != "sliding" else 1, "compressor": comp, "index_topk": 512,
        "attn_factor": 1.0, "mrope_section": None, "softcap": 0.0,
        "activation": "silu", "act_limit": 0.0, "mlp_kind": "dense", "layer_scalar": None,
        "qsa_threshold": None,
    }
    w = {k_: v_.frozen_weight for k_, v_ in lins.items()}
    w["wo_a"] = [l.frozen_weight for l in wo_a]
    w.update({"attn_norm": norm_a, "mlp_norm": norm_m, "q_a_norm": q_a_norm, "kv_norm": kv_norm})
    if comp is not None:
        w["c_norm"] = comp["norm"]["weight"]
    return entry, meta, w, lins, wo_a


def _ref_block(meta, w, hidden, pos, pad_mask=None):
    x = _hf_rmsnorm(hidden, w["attn_norm"], 1e-5)
    hidden = hidden + _hf_dsv4_attention(w, meta, x, pos, pad_mask)
    x2 = _hf_rmsnorm(hidden, w["mlp_norm"], 1e-5)
    return hidden + (F.silu(x2 @ w["gate"]) * (x2 @ w["up"])) @ w["down"]


def test_rope_trailing_matches_hf_and_inverts():
    torch.manual_seed(0)
    net = tm._headless_net()
    b, t, h, D, rd = 2, 5, 3, 12, 8
    x = torch.randn(b, t, h, D)
    pos = torch.arange(t).unsqueeze(0).expand(b, t)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rd, 2).float() / rd))
    y = net._rope_gptj_trailing(x, inv, pos, rd)
    ref = _hf_rope(x, inv, pos, rd)
    assert (y - ref).abs().max() < 1e-6 and torch.equal(y[..., :D - rd], x[..., :D - rd])
    back = net._rope_gptj_trailing(y, inv, pos, rd, sign=-1.0)
    assert (back - x).abs().max() < 1e-5
    print("[dsv4] trailing-slice GPT-J rope matches HF; de-rotation inverts PASSED")


def test_compressors_match_hf():
    torch.manual_seed(1)
    net = tm._headless_net()
    net.use_liger = False
    d, D, rd, m = 8, 12, 8, 4
    for lt in ("hca", "csa"):
        entry, meta, w, _, _ = _build_dsv4_block(d, 2, D, rd, 1, 6, 5, lt, m, 16)
        for S in (3, 4, 11):   # no complete window / one / two + remainder
            x = torch.randn(2, S, d)
            with torch.no_grad():
                out = net._dsv4_compress(meta, entry, x, 2, S)
            fn = _hf_csa if lt == "csa" else _hf_hca
            ref = fn(x, w["c_kv"], w["c_gate"], meta["compressor"]["ape"], w["c_norm"], 1e-6, m, D,
                     meta["inv_freq_compress"], meta["compressor"]["rope_dim"])
            if S < m:
                assert out is None and ref is None
                continue
            assert out.shape == ref.shape == (2, S // m, D)
            err = (out - ref).abs().max().item()
            assert err < 1e-5, f"{lt} compressor mismatch at S={S}: {err}"
    print("[dsv4] HCA / CSA compressors match HF (incl. remainder drop, window-0 Ca) PASSED")


def test_dsv4_blocks_match_hf_composition():
    torch.manual_seed(2)
    net = tm._headless_net()
    net.use_liger = False
    d, H, D, rd, G, q_rank, o_rank, inter, m = 16, 4, 12, 8, 2, 10, 6, 32, 4
    for lt, S in (("sliding", 9), ("hca", 11), ("csa", 13)):
        entry, meta, w, _, _ = _build_dsv4_block(d, H, D, rd, G, q_rank, o_rank, lt, m, inter, window=4)
        b = 2
        hidden = torch.randn(b, S, d)
        pos = torch.arange(S).unsqueeze(0).expand(b, S)
        with torch.no_grad():
            out = net._dsv4_forward(meta, entry, hidden, pos, None, None)
        ref = _ref_block(meta, w, hidden, pos)
        err = (out - ref).abs().max().item()
        assert err < 1e-4, f"DSV4 {lt} block mismatch: {err}"
        # Sanity: the sink / window actually matter (not a vacuous equality).
        meta2 = dict(meta, sinks=meta["sinks"] + 3.0)
        with torch.no_grad():
            out2 = net._dsv4_forward(meta2, entry, hidden, pos, None, None)
        assert (out2 - out).abs().max() > 1e-3
    print("[dsv4] sliding / HCA / CSA blocks match the HF-style composition PASSED")


def test_dsv4_right_padding_hides_pad_keys():
    torch.manual_seed(3)
    net = tm._headless_net()
    net.use_liger = False
    d, H, D, rd, G = 8, 2, 12, 8, 1
    entry, meta, w, _, _ = _build_dsv4_block(d, H, D, rd, G, 6, 5, "hca", 4, 16, window=3)
    b, S = 2, 10
    hidden = torch.randn(b, S, d)
    pos = torch.arange(S).unsqueeze(0).expand(b, S)
    am = torch.ones(b, S, dtype=torch.long)
    am[1, 7:] = 0
    with torch.no_grad():
        out = net._dsv4_forward(meta, entry, hidden, pos, am, None)
    ref = _ref_block(meta, w, hidden, pos, pad_mask=am)
    assert (out - ref).abs().max() < 1e-4
    # Row 1's real positions equal the unpadded forward of its first 7 tokens
    # only where no compressed window reaches past them: positions < 4 see no
    # window at all (window 0 completes at token 3, visible from token 4).
    with torch.no_grad():
        out_short = net._dsv4_forward(meta, entry, hidden[1:, :7], pos[1:, :7], None, None)
    assert (out[1, :4] - out_short[0, :4]).abs().max() < 1e-4
    print("[dsv4] right-padding hides pad keys from real queries PASSED")


def test_dsv4_backward_reaches_adapters():
    torch.manual_seed(4)
    net = tm._headless_net()
    net.use_liger = False
    d, H, D, rd, G = 8, 2, 12, 8, 2
    entry, meta, _, lins, wo_a = _build_dsv4_block(d, H, D, rd, G, 6, 5, "csa", 4, 16, window=3, r=2)
    adapted = [entry.q_a_proj, entry.q_b_proj, entry.wkv, entry.wo_b] + list(entry.wo_a)
    with torch.no_grad():
        for wr in adapted:
            wr.lora_b.normal_(std=0.05)
    frozen = {k_: l.frozen_weight.clone() for k_, l in lins.items()}
    b, S = 2, 9
    hidden = torch.randn(b, S, d, requires_grad=True)
    pos = torch.arange(S).unsqueeze(0).expand(b, S)
    out = net._dsv4_forward(meta, entry, hidden, pos, None, None)
    out.pow(2).mean().backward()
    for wr in adapted:
        assert wr.lora_a.grad is not None and wr.lora_b.grad.abs().sum() > 0, wr.linear.key
    for k_, l in lins.items():
        assert torch.equal(l.frozen_weight, frozen[k_]) and l.frozen_weight.grad is None
    assert entry.comp_wkv.r == 0 and meta["sinks"].grad is None and meta["compressor"]["ape"].grad is None
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    print("[dsv4] backward reaches q_a / q_b / wkv / wo_a / wo_b adapters; base + tables frozen PASSED")


# ----------------------------------------------------------------------------
# sqrt-softplus / hash routers.
# ----------------------------------------------------------------------------
def _ref_sqrtsp_moe(meta, lins, x, ids=None, L=0.0):
    b, t, d = x.shape
    y = x.reshape(-1, d)
    E, k = meta["num_experts"], meta["num_experts_per_tok"]
    logits = y @ lins["router"].frozen_weight
    s = F.softplus(logits).sqrt()
    if ids is None:
        idx = torch.topk(s + meta["e_score_bias"], k, dim=-1).indices
    else:
        idx = meta["tid2eid"][ids.reshape(-1)]
    wts = s.gather(1, idx)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * meta["routed_scaling_factor"]

    def mlp(g, u, dn, z):
        a = F.silu(z @ g.frozen_weight)
        up = z @ u.frozen_weight
        if L:
            a, up = a.clamp(max=L), up.clamp(-L, L)
        return (a * up) @ dn.frozen_weight
    out = torch.zeros_like(y)
    for n in range(y.shape[0]):
        for j in range(k):
            e = int(idx[n, j])
            out[n] += wts[n, j] * mlp(lins[f"g{e}"], lins[f"u{e}"], lins[f"d{e}"], y[n:n + 1])[0]
    out = out + mlp(lins["sg"], lins["su"], lins["sd"], y)
    return out.view(b, t, d)


def test_sqrtsp_and_hash_routers_match_hf():
    torch.manual_seed(5)
    net = tm._headless_net()
    net.use_liger = False
    d, inter, E, k, L = 8, 12, 6, 2, 1.0
    for rtype in ("sqrtsp", "sqrtsp_hash"):
        entry, meta, lins = tm._build_moe(d, inter, E, k, shared_inter=12, shared_gate=False,
                                          router_type=rtype, e_score_bias=torch.randn(E) * 0.3,
                                          routed_scaling=1.5, router_scale=1.0)
        meta["act_limit"] = meta["shared_act_limit"] = L
        V = 20
        meta["tid2eid"] = torch.stack([torch.randperm(E)[:k] for _ in range(V)])
        b, t = 2, 5
        x = torch.randn(b, t, d) * 2
        ids = torch.randint(0, V, (b, t))
        with torch.no_grad():
            out = net._moe_out(meta, entry, x, None, input_ids=ids)
        ref = _ref_sqrtsp_moe(meta, lins, x, ids if rtype == "sqrtsp_hash" else None, L)
        err = (out - ref).abs().max().item()
        assert err < 1e-4, f"{rtype} MoE mismatch: {err}"
    print("[dsv4] sqrtsp + hash routers with clamped experts match HF PASSED")


def main():
    from util import run_timed
    run_timed([
        test_rope_trailing_matches_hf_and_inverts,
        test_compressors_match_hf,
        test_dsv4_blocks_match_hf_composition,
        test_dsv4_right_padding_hides_pad_keys,
        test_dsv4_backward_reaches_adapters,
        test_sqrtsp_and_hash_routers_match_hf,
    ], label="dsv4-train")
    print("\nAll DeepSeek-V4 training checks passed.")


if __name__ == "__main__":
    main()
