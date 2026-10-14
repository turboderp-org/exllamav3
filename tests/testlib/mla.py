"""
Multi-head latent attention (MLAttention) test helpers: random-weight modules loaded through the real
safetensors loader, the explicit per-head reference forward, paged caches and a chunked driver.

    t = mla_weights(H = 8, q_lora = 256, seed = 0)              # checkpoint tensors, KEY namespace
    module = load_mla(t, tmp_path, device, H = 8, q_lora = 256)   # MLAttention(...).load(device)
    ref = ref_forward(module, t, KEY, x, positions)                # fp32 reference, per-head K/V built
    out = run_module(module, x, make_cache(module, n), bt, chunk = 128)

The reference builds per-head K and V from the latent (DeepSeek-V2/V3 form) and runs plain causal MHA,
optionally restricted to a per-query selection (DSA) and with the latent round-tripped through the cache
quantizer. RoPE is applied with the module's own RoPE object, so it isolates the MLA math (absorption,
cache layout, kernels) from rope conventions.
"""

import torch

from exllamav3.cache import CacheLayer_MLA_fp16, CacheLayer_MLA_quant
from exllamav3.util.rope import RopeSettings, RopeStyle

from testlib.checkpoint import module_config

KEY = "model.layers.0.self_attn"


def mla_weights(H = 8, hidden = 1024, kv_lora = 512, nope = 128, rope_dim = 64, v_head = 128, q_lora = None,
                seed = 0, wscale = 0.085) -> dict[str, torch.Tensor]:
    """fp16 MLA projections under KEY (q_proj, or q_a/q_b with q_lora).

    wscale is chosen so pre-softmax scores land around unit standard deviation. The q and k paths both pass
    through an RMSNorm, so the input scale is normalized away and the weight scale alone sets the score
    magnitude: at the obvious 0.02 the scores have std 0.12, softmax is nearly uniform and the comparison goes
    blind to score errors (a 1% W_UK perturbation moves the output by less than the fp16 noise floor)."""
    g = torch.Generator(device = "cpu").manual_seed(seed)

    def rnd(*shape):
        return (torch.randn(*shape, generator = g) * wscale).half()

    qk_head = nope + rope_dim
    t = {
        f"{KEY}.kv_a_proj_with_mqa.weight": rnd(kv_lora + rope_dim, hidden),
        f"{KEY}.kv_a_layernorm.weight": (torch.randn(kv_lora, generator = g) * 0.1 + 1).half(),
        f"{KEY}.kv_b_proj.weight": rnd(H * (nope + v_head), kv_lora),
        f"{KEY}.o_proj.weight": rnd(hidden, H * v_head),
    }
    if q_lora is None:
        t[f"{KEY}.q_proj.weight"] = rnd(H * qk_head, hidden)
    else:
        t[f"{KEY}.q_a_proj.weight"] = rnd(q_lora, hidden)
        t[f"{KEY}.q_a_layernorm.weight"] = (torch.randn(q_lora, generator = g) * 0.1 + 1).half()
        t[f"{KEY}.q_b_proj.weight"] = rnd(H * qk_head, q_lora)
    return t


def load_mla(tensors, directory, device, H = 8, hidden = 1024, kv_lora = 512, nope = 128, rope_dim = 64,
             v_head = 128, q_lora = None, rope_style = RopeStyle.NEOX, nope_only = False, **kwargs):
    """MLAttention over the tensors (written to directory, read back through SafetensorsCollection), loaded
    on device. nope_only: no rope instance at all (Kimi Linear). kwargs go to MLAttention (indexer etc.)"""
    from exllamav3.modules import MLAttention
    rope_settings = None if nope_only else RopeSettings(
        head_dim = rope_dim, rope_theta = 10000.0, rope_style = rope_style,
    )
    module = MLAttention(
        config = module_config(tensors, directory), key = KEY, layer_idx = 0, hidden_size = hidden,
        num_q_heads = H, kv_lora_rank = kv_lora, qk_nope_head_dim = nope, qk_rope_head_dim = rope_dim,
        v_head_dim = v_head, rope_settings = rope_settings, q_lora_rank = q_lora, rms_norm_eps = 1e-6,
        **kwargs,
    )
    module.load(torch.device(device))
    return module


def build_mla(directory, device, H = 8, hidden = 1024, kv_lora = 512, nope = 128, rope_dim = 64, v_head = 128,
              q_lora = None, nope_only = False, seed = 0, wscale = 0.085):
    """Random-weight MLAttention plus its raw weights on the device: (module, tensors, KEY)"""
    t = mla_weights(H, hidden, kv_lora, nope, rope_dim, v_head, q_lora, seed, wscale)
    module = load_mla(t, directory, device, H, hidden, kv_lora, nope, rope_dim, v_head, q_lora,
                      nope_only = nope_only)
    return module, {k: v.to(device) for k, v in t.items()}, KEY


def rms_norm(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps) * w.float()


def sim_cache_quant(v, bits):
    """Round-trip a (..., D) fp16 tensor through the cache quantizer at `bits` (the CUDA kernels that fill
    CacheLayer_MLA_quant), so a reference can carry the cache's exact latent values"""
    from exllamav3.ext import exllamav3_ext as ext
    D = v.shape[-1]
    rows = v.numel() // D
    pq = torch.empty((rows, D // 32 * bits), dtype = torch.int32, device = v.device)
    ps = torch.empty((rows, D // 32), dtype = torch.half, device = v.device)
    ext.quant_cache_cont(v.reshape(rows, D).half().contiguous(), pq, ps, 0.0)
    out = torch.empty((rows, D), dtype = torch.half, device = v.device)
    ext.dequant_cache_cont(pq, ps, out, 0.0)
    return out.view(v.shape)


def ref_forward(module, t, key, x, positions, indices = None, ckv_quant_bits = 0):
    """Reference MLA: per-head K and V built explicitly from the latent, then plain causal MHA in fp32.

    indices: optional (bsz, S, k) per-query selection (-1 padded); attention is restricted to it (DSA).
    ckv_quant_bits: round-trip the latent through the cache quantizer at that width (the values a
    CacheLayer_MLA_quant holds; the fp16 rope key is exact in both)."""
    m = module
    bsz, S, _ = x.shape
    H, nope, rope_dim, v_head = m.num_q_heads, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim

    xf = x.float()
    if m.q_lora_rank is None:
        q = xf @ t[f"{key}.q_proj.weight"].float().T
    else:
        q = xf @ t[f"{key}.q_a_proj.weight"].float().T
        q = rms_norm(q, t[f"{key}.q_a_layernorm.weight"], m.norm_eps)
        q = q @ t[f"{key}.q_b_proj.weight"].float().T
    q = q.view(bsz, S, H, m.qk_head_dim)
    q_nope, q_pe = q[..., :nope], q[..., nope:]

    ckv_kpe = xf @ t[f"{key}.kv_a_proj_with_mqa.weight"].float().T
    ckv = rms_norm(ckv_kpe[..., :m.kv_lora_rank], t[f"{key}.kv_a_layernorm.weight"], m.norm_eps)
    if ckv_quant_bits:
        ckv = sim_cache_quant(ckv.half(), ckv_quant_bits).float()
    k_pe = ckv_kpe[..., m.kv_lora_rank:].view(bsz, S, 1, rope_dim)

    # Same RoPE object as the module, so this isolates the MLA math
    if m.rope is not None:
        q_pe, k_pe = m.rope.apply(
            q_pe.half().contiguous(), k_pe.half().contiguous(),
            0, positions, None, False, None, None, m.norm_eps, 0.0, None,
        )
    q_pe, k_pe = q_pe.float(), k_pe.float()

    kv = (ckv.half().float() @ t[f"{key}.kv_b_proj.weight"].float().T).view(bsz, S, H, nope + v_head)
    k_nope, v = kv[..., :nope], kv[..., nope:]
    k = torch.cat([k_nope, k_pe.expand(bsz, S, H, rope_dim)], dim = -1)
    q_full = torch.cat([q_nope, q_pe], dim = -1)

    scores = torch.einsum("bqhd,bkhd->bhqk", q_full, k) * m.sm_scale
    pos = positions.view(bsz, 1) + torch.arange(S, device = x.device).view(1, S)
    allowed = pos.view(bsz, 1, S) <= pos.view(bsz, S, 1)
    if indices is not None:
        sel = torch.zeros((bsz, S, S), dtype = torch.bool, device = x.device)
        idx = indices.view(bsz, S, -1).long()
        bb, qq, kk = (idx >= 0).nonzero(as_tuple = True)
        sel[bb, qq, idx[bb, qq, kk]] = True
        allowed = allowed & sel
    scores = scores.masked_fill(~allowed.unsqueeze(1), -float("inf"))
    p = torch.softmax(scores, dim = -1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, v).reshape(bsz, S, H * v_head)
    return o @ t[f"{key}.o_proj.weight"].float().T


def make_cache(module, max_tokens):
    """fp16 latent cache layer for the module, allocated on its device"""
    layer = CacheLayer_MLA_fp16(None, module, 0, max_tokens)
    layer.alloc(module.device)
    return layer


def make_qcache(module, max_tokens, bits):
    """Packed-quantized latent cache layer (k_bits = bits) for the module, allocated on its device"""
    layer = CacheLayer_MLA_quant(None, module, 0, max_tokens, k_bits = bits)
    layer.alloc(module.device)
    return layer


def cached_params(layer, bt, seqlens) -> dict:
    return {
        "attn_mode": "flash_attn",
        "cache": layer,
        "block_table": bt,
        "cache_seqlens": seqlens,
        "positions": seqlens.clone(),
    }


def run_module(module, x, layer, bt, chunk = None, params_out: list | None = None):
    """Feed x through the module over the paged cache in chunks of `chunk` rows (default: one chunk), as the
    generator would. params_out, if given, collects each chunk's params dict (e.g. its DSA selection)"""
    bsz, S, _ = x.shape
    chunk = chunk or S
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = x.device)
    outs = []
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        params = cached_params(layer, bt, seqlens)
        outs.append(module.forward(x[:, a:b].contiguous(), params))
        if params_out is not None:
            params_out.append(params)
        seqlens = seqlens + (b - a)
    return torch.cat(outs, dim = 1)
