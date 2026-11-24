"""Pure-torch math of Inkling-2-Mini (inkling2_loop), shared by the reference script and the exllamav3 port.

Validated against vLLM's Inkling 2 implementation on the original checkpoint (top-1 agreement 2872/2873, mean
|delta logprob| 0.0037 on a 2,873-token prompt). Weight access goes through two hooks so the exllamav3 modules can route the
quantizable projections and the MoE through their own Linear / BlockSparseMLP modules:

    W[key]                       raw tensor, key relative to "model.llm." (".weight" may be omitted)
    proj(key, x)                 x @ W[key].T for the quantizable projections (wq_du, wk_dv, wv_dv, wr_du,
                                 gate_attn, wo_ud)
    moe(i, x)                    routed experts of layer i on the normalized MLP input

No torch.nn, no exllamav3 imports: this file must stay importable on its own.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

DT = torch.bfloat16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def rms(x, w, eps):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (w * x).to(dt) if w is not None else x.to(dt)


def lin(x, w):
    return F.linear(x, w)


def dequant_experts(w, scale, amax, denom, dtype=DT):
    """HF Inkling2LoopDequantizeExperts. NVFP4 when amax is given (uint8, two E2M1 codes per byte, even element
    in the low nibble, float8 scale per 16 values, per-expert amax / outer_scale_denominator); MXFP8 otherwise
    (float8 values, uint8 power-of-two scale 2 ** (s - 127) per 32 values)."""
    E, rows = w.shape[:2]
    if amax is not None:
        codes = (w.view(torch.uint8) if w.dtype != torch.uint8 else w).long()
        cols = w.shape[2] * 2
        e2m1 = torch.tensor(E2M1, device=w.device)
        byte = torch.arange(256, device=w.device)
        b2v = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
        bs = scale.float()
        vals = b2v[codes].view(*bs.shape, 16) * bs[..., None]
        vals = vals * (amax.float() / denom.float())[:, None, None, None]
    else:
        cols = w.shape[2]
        bs = scale.float()
        vals = w.float().view(*bs.shape, 32) * torch.exp2(bs - 127)[..., None]
    return vals.reshape(E, rows, cols).to(dtype)


def deinterleave_rows(t):
    """HF Interleave(dim=1): rows [g0,u0,g1,u1,...] -> [g...; u...]."""
    E, n, c = t.shape
    return t.reshape(E, n // 2, 2, c).transpose(1, 2).reshape(E, n, c).contiguous()


def softcapped_swiglu(g, u, gate_cap, up_cap):
    """Inkling expert activation: softcap(g) * sigmoid(g) * softcap(u), in fp32."""
    g, u = g.float(), u.float()
    return torch.tanh(g / gate_cap) * gate_cap * torch.sigmoid(g) * torch.tanh(u / up_cap) * up_cap


def message_queries(W, D):
    """(num_loop_layers, 4, D) cross-pass message queries in order q, k, v, mlp."""
    q = W["middle_cycle_cross_pass_queries"].float().unflatten(0, (-1, 2))
    kv = W["middle_cycle_cross_pass_kv_queries"].float().unflatten(0, (-1, 2))
    return torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)


class InklingMath:
    """Layer math. Subclasses provide W (mapping) and may override proj() and moe()."""

    def __init__(self, cfg: dict, W):
        self.c = cfg
        self.W = W
        self.D = cfg["hidden_size"]
        self.eps = cfg["rms_norm_eps"]
        self.nl = cfg["num_hidden_layers"]
        self.local = set(cfg["local_layer_ids"])
        self.lpb = cfg["n_layers_per_attn_res_block"]
        lc = cfg["loop_config"]
        self.ls, self.le, self.np = lc["core_start"], lc["core_end"], lc["num_passes"]
        self.K = cfg["sconv_kernel_size"]

    # ---- hooks ----
    def proj(self, key, x):
        return lin(x, self.W[key])

    def moe(self, i, x):
        """Routed experts on the dequantized original checkpoint (HF semantics)."""
        p = f"layers.{i}.mlp."
        W = self.W
        c = self.c
        b, s, D = x.shape
        flat = x.reshape(-1, D)
        logits = lin(flat.float(), W[p + "gate.weight"].float())
        scores = torch.sigmoid(logits)
        idx = torch.topk(scores + W[p + "gate.bias"].float(), c["num_experts_per_tok"], dim=-1, sorted=False).indices
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * c["route_scale"]

        e = p + "experts."
        w13 = deinterleave_rows(dequant_experts(W[e + "w13_weight"], W[e + "w13_weight_scale"],
                                                W.get(e + "w13_weight_amax"), W.get(e + "w13_weight_outer_scale_denominator")))
        w2 = dequant_experts(W[e + "w2_weight"], W[e + "w2_weight_scale"],
                             W.get(e + "w2_weight_amax"), W.get(e + "w2_weight_outer_scale_denominator"))
        out = torch.zeros_like(flat, dtype=torch.float32)
        gc, uc = c["mlp_gate_softcap"], c["mlp_up_softcap"]
        for ex in idx.unique().tolist():
            tok, kpos = torch.where(idx == ex)
            gu = F.linear(flat[tok], w13[ex])
            g, u = gu.float().chunk(2, dim=-1)
            a = softcapped_swiglu(g, u, gc, uc).to(DT)
            y = F.linear(a, w2[ex]) * wts[tok, kpos, None]
            out.index_add_(0, tok, y.float())
        return out.to(x.dtype).reshape(b, s, D)

    # ---- small pieces ----
    def attn_res(self, query, blocks, partial):
        """Inkling2LoopBlockAttentionResidual / MessageAttentionResidual. query (nq, D) -> list of (b,s,D) fp32."""
        st = torch.cat([blocks, partial.unsqueeze(-2)], dim=-2).float()
        st = rms(st, None, 1e-6)
        q = query.float().view(-1, self.D)
        logits = torch.einsum("bsnd,qd->bsqn", st, q) * self.D ** -0.5
        p = torch.softmax(logits, dim=-1)
        out = torch.einsum("bsqn,bsnd->bsqd", p, st)
        return out.unbind(-2)

    def msg_gate(self, p, h, m):
        h, m = h.float(), m.float()
        g = torch.cat([rms(h, None, self.eps), rms(m, None, self.eps)], dim=-1)
        g = lin(g.to(DT), self.W[p + "down_weight"]).float()
        f = torch.tanh(lin(g, self.W[p + "forget_weight"].float()))
        mg = torch.tanh(lin(g, self.W[p + "gain_weight"].float()))
        return (1 - f) * h + mg * m

    def conv(self, x, w):
        """Inkling2LoopShortConvolution (fp32 conv, output (conv + x) / sqrt(k+1))."""
        dt = x.dtype
        x = x.float()
        s = x.shape[1]
        y = F.conv1d(x.transpose(1, 2), w.float(), padding=self.K - 1, groups=x.shape[-1])[:, :, :s]
        y = y.transpose(1, 2)
        return ((y + x) / math.sqrt(self.K + 1)).to(dt)

    def dyn_conv(self, x, dyn_in, static_w, wproj, nheads):
        """Inkling2LoopDynamicShortConvolution."""
        dt = x.dtype
        b, s, C = x.shape
        K = self.K
        dw = lin(dyn_in, wproj).float().view(b, s, nheads, 1, K)
        weights = static_w.float().T.reshape(nheads, -1, K) + dw
        x = x.float()
        ctx = F.pad(x.transpose(1, 2), (K - 1, 0))
        win = ctx.unfold(-1, K, 1).flip(-1).transpose(1, 2).reshape(b, s, nheads, -1, K)
        y = (win * weights).sum(-1).reshape(b, s, C)
        return ((y + x) / math.sqrt(2 * K + 1)).to(dt)

    @staticmethod
    def l2clip(x, thr, scale=None):
        dt = x.dtype
        x = x.float()
        if scale is not None:
            x = x * scale.float()
        n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        return (x * (thr / (n + 1e-6)).clamp(max=1.0)).to(dt)

    def log_scale(self, x, pos):
        sc = 1.0 + self.c["log_scaling_alpha"] * torch.log(((pos + 1).float() / self.c["log_scaling_n_floor"]).clamp(min=1.0))
        return (x.float() * sc[..., None, None]).to(x.dtype)

    # ---- attention ----
    def attention(self, i, hq, hk, hv, pos):
        p = f"layers.{i}.attn."
        W = self.W
        glob = i not in self.local
        c = self.c
        nh = c["num_attention_heads"] if glob else c["swa_num_attention_heads"]
        nkv = c["num_key_value_heads"] if glob else c["swa_num_key_value_heads"]
        hd = c["head_dim"] if glob else c["swa_head_dim"]
        b, s, _ = hq.shape
        thr = c["qk_l2_clip_threshold"]

        q = self.proj(p + "wq_du", hq).view(b, s, nh, hd)
        q = self.l2clip(q, thr, W[p + "q_l2_clip.scale"])
        if glob:
            q = self.log_scale(q, pos)
        q = q.transpose(1, 2)

        def kv_proj(h, name, is_key):
            x = self.proj(p + f"w{name}_dv", h)
            if glob:
                x = self.dyn_conv(x, h, W[p + f"{name}_sconv.static_w"], W[p + f"{name}_sconv.weight_proj"], nkv)
            else:
                x = self.conv(x, W[p + f"{name}_sconv.weight"])
            x = x.view(b, s, nkv, hd)
            x = self.l2clip(x, thr) if is_key else rms(x, None, self.eps)
            return x.transpose(1, 2)

        k = kv_proj(hk, "k", True)
        v = kv_proj(hv, "v", False)

        # relative position bias (conditioned on the query input)
        rel = self.proj(p + "wr_du", hq).view(b, s, nh, -1)
        rel = self.l2clip(rel, c["rel_l2_clip_threshold"], W[p + "rel_l2_clip.scale"])
        if glob:
            rel = self.log_scale(rel, pos)
        extent = c["rel_extent"] if glob else c["sliding_window_size"]
        proj = W[p + "rel_logits_proj.proj"]
        rel_logits = (rel @ proj).transpose(1, 2)                      # (b, h, s, extent)
        dist = pos[0][:, None] - pos[0][None, :]                       # (q, k)
        gi = dist.clamp(0, extent - 1).expand(b, nh, -1, -1)
        bias = rel_logits.gather(-1, gi).masked_fill((dist < 0) | (dist >= extent), 0.0)

        allowed = dist >= 0
        if not glob:
            allowed = allowed & (dist < c["sliding_window_size"])
        mask = torch.zeros(s, s, dtype=q.dtype, device=q.device).masked_fill(~allowed, torch.finfo(q.dtype).min)

        sinks = lin(hq, W[p + "attn_sink_proj.weight"]).transpose(1, 2).unsqueeze(-1)
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        aw = torch.matmul(q, k.transpose(2, 3)) * hd ** -0.5 + bias + mask
        aw = torch.softmax(torch.cat([aw, sinks], dim=-1).float(), dim=-1)[..., :-1].to(q.dtype)
        o = torch.matmul(aw, v).transpose(1, 2).reshape(b, s, -1)
        gate = torch.sigmoid(self.proj(p + "gate_attn", hq).float())
        o = (o.float() * gate).to(o.dtype)
        return self.proj(p + "wo_ud", o)

    # ---- decoder layer ----
    def layer(self, i, h, blocks, pos, messages=None):
        W = self.W
        p = f"layers.{i}."
        glob = i not in self.local
        boundary = i % self.lpb == 0
        block_residual = h

        if glob:
            if i == 0:
                hq = hk = hv = h
            else:
                hq, hk, hv = self.attn_res(W[f"attn_res.attn_queries.{i - 1}"], blocks, h)
            if messages is not None:
                mq, mk, mv, _ = messages.unbind(-2)
                hq = self.msg_gate(p + "controllers.attn.", hq, mq)
                hk = self.msg_gate(p + "controllers.k.", hk, mk)
                hv = self.msg_gate(p + "controllers.v.", hv, mv)
            hq = rms(hq, W[p + "q_input_norm.weight"], self.eps).to(DT)
            hk = rms(hk, W[p + "k_input_norm.weight"], self.eps).to(DT)
            hv = rms(hv, W[p + "v_input_norm.weight"], self.eps).to(DT)
        else:
            x = h if i == 0 else self.attn_res(W[f"attn_res.attn_queries.{i - 1}"], blocks, h)[0]
            if messages is None:
                hq = hk = hv = rms(x, W[p + "attn_norm.weight"], self.eps).to(DT)
            else:
                mq, mk, mv, _ = messages.unbind(-2)
                hq = rms(self.msg_gate(p + "controllers.attn.", x, mq), W[p + "attn_norm.weight"], self.eps).to(DT)
                hk = rms(self.msg_gate(p + "controllers.k.", x, mk), W[p + "attn_norm.weight"], self.eps).to(DT)
                hv = rms(self.msg_gate(p + "controllers.v.", x, mv), W[p + "attn_norm.weight"], self.eps).to(DT)

        a = self.attention(i, hq, hk, hv, pos)
        a = self.conv(a, W[p + "attn_sconv.weight"])
        residual = a = a.float()
        if boundary:
            blocks = torch.cat([blocks, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = a = block_residual.float() + a
        m = self.attn_res(W[f"attn_res.mlp_queries.{i}"], blocks, a)[0]
        if messages is not None:
            m = self.msg_gate(p + "controllers.mlp.", m, messages.unbind(-2)[3])
        m = self.moe(i, rms(m, W[p + "mlp_norm.weight"], self.eps).to(DT))
        m = self.conv(m, W[p + "mlp_sconv.weight"])
        return (residual + m).to(DT)

    def output(self, blocks, h):
        """Final AttnRes over all blocks + final norm (before the unembedding)."""
        h = self.attn_res(self.W["attn_res.output_query"], blocks, h)[0]
        return rms(h, self.W["norm.weight"], self.eps).to(DT)
