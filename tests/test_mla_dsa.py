import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pytest
import torch
import torch.nn.functional as F

from exllamav3.modules import MLAttention
from exllamav3.cache import CacheLayer_MLA_fp16
from exllamav3.constants import PAGE_SIZE
from exllamav3.util.rope import RopeSettings, RopeStyle
from exllamav3.modules.attention_fn.mla_triton import has_triton

from test_mla import FakeConfig, rms_norm

# DSA-on-MLA (GLM-5.2): the lightning indexer selects index_topk tokens per query and the
# attention core gathers only those latent rows. These tests transcribe the reference indexer
# (transformers glm_moe_dsa) in plain torch and check, on random weights:
#
#   - the module's top-k selection against the reference scores (allowing a small overlap
#     slack at the k-th-score boundary, where fp16 kernel scores and fp32 reference scores
#     can order ties differently),
#   - the sparse attention output against a masked dense reference driven by the MODULE's own
#     selection (tight: this isolates the gather/attention math from boundary selection noise),
#   - dense equivalence when index_topk >= T,
#   - cross-layer sharing ("shared" layers consume the published selection),
#   - the cached path (paged indexer-key plane) against the cache-less path, chunked.

device = "cuda:0"
pytestmark = pytest.mark.skipif(not has_triton, reason = "requires Triton")


def build_dsa(H = 8, hidden = 512, kv_lora = 512, nope = 128, rope_dim = 64, v_head = 128,
              q_lora = 256, idx_heads = 4, idx_dim = 128, topk = 64, mode = "full",
              seed = 0, wscale = 0.085):
    g = torch.Generator(device = "cpu").manual_seed(seed)

    def rnd(*shape, scale = wscale):
        return (torch.randn(*shape, generator = g) * scale).half()

    key = "model.layers.0.self_attn"
    qk_head = nope + rope_dim
    t = {
        f"{key}.q_a_proj.weight": rnd(q_lora, hidden),
        f"{key}.q_a_layernorm.weight": (torch.randn(q_lora, generator = g) * 0.1 + 1).half(),
        f"{key}.q_b_proj.weight": rnd(H * qk_head, q_lora),
        f"{key}.kv_a_proj_with_mqa.weight": rnd(kv_lora + rope_dim, hidden),
        f"{key}.kv_a_layernorm.weight": (torch.randn(kv_lora, generator = g) * 0.1 + 1).half(),
        f"{key}.kv_b_proj.weight": rnd(H * (nope + v_head), kv_lora),
        f"{key}.o_proj.weight": rnd(hidden, H * v_head),
    }
    if mode == "full":
        # Indexer weights run hotter than the attention ones so the relu keeps a healthy
        # mix of active and clamped scores and the top-k boundary is not pure noise
        t[f"{key}.indexer.wq_b.weight"] = rnd(idx_heads * idx_dim, q_lora, scale = 0.25)
        t[f"{key}.indexer.wk.weight"] = rnd(idx_dim, hidden, scale = 0.25)
        t[f"{key}.indexer.k_norm.weight"] = (torch.randn(idx_dim, generator = g) * 0.1 + 1).half()
        t[f"{key}.indexer.k_norm.bias"] = (torch.randn(idx_dim, generator = g) * 0.05).half()
        t[f"{key}.indexer.weights_proj.weight"] = rnd(idx_heads, hidden, scale = 0.25)

    rope_settings = RopeSettings(
        head_dim = rope_dim, rope_theta = 10000.0, rope_style = RopeStyle.GPTJ,
    )
    module = MLAttention(
        config = FakeConfig(t), key = key, layer_idx = 0, hidden_size = hidden,
        num_q_heads = H, kv_lora_rank = kv_lora, qk_nope_head_dim = nope,
        qk_rope_head_dim = rope_dim, v_head_dim = v_head, rope_settings = rope_settings,
        q_lora_rank = q_lora, rms_norm_eps = 1e-6,
        indexer_mode = mode, index_n_heads = idx_heads, index_head_dim = idx_dim,
        index_topk = topk,
    )
    module.load(torch.device(device))
    return module, {k: v.to(device) for k, v in t.items()}, key


def ref_index_scores(module, t, key, x, positions):
    """Reference lightning-indexer scores (B, S, S): relu(q . k) * D**-0.5, head-weighted,
    fp32, interleaved rope on the first rope_dim dims, -inf past the causal bound."""
    m = module
    bsz, S, _ = x.shape
    Hi, Di, rd = m.index_n_heads, m.index_head_dim, m.qk_rope_head_dim
    xf = x.float()

    q_resid = rms_norm(xf @ t[f"{key}.q_a_proj.weight"].float().T,
                       t[f"{key}.q_a_layernorm.weight"], m.norm_eps).half().float()
    q = (q_resid @ t[f"{key}.indexer.wq_b.weight"].float().T).view(bsz, S, Hi, Di)
    k = F.layer_norm(xf @ t[f"{key}.indexer.wk.weight"].float().T, (Di,),
                     t[f"{key}.indexer.k_norm.weight"].float(),
                     t[f"{key}.indexer.k_norm.bias"].float(), eps = 1e-6)
    k = k.view(bsz, S, 1, Di)

    q_rot, k_rot = m.rope.apply(
        q[..., :rd].half().contiguous(), k[..., :rd].half().contiguous(),
        0, positions, None, False, None, None, m.norm_eps, 0.0, None,
    )
    q = torch.cat([q_rot.float(), q[..., rd:]], dim = -1)
    k = torch.cat([k_rot.float(), k[..., rd:]], dim = -1).squeeze(2)

    scores = torch.einsum("bqhd,bkd->bqhk", q, k) * Di ** -0.5
    scores = F.relu(scores)
    w = (xf @ t[f"{key}.indexer.weights_proj.weight"].float().T) * Hi ** -0.5
    scores = torch.einsum("bqh,bqhk->bqk", w, scores)

    pos = positions.view(bsz, 1) + torch.arange(S, device = x.device).view(1, S)
    causal = pos.view(bsz, 1, S) > pos.view(bsz, S, 1)
    return scores.masked_fill(causal, -float("inf"))


def ref_forward_indices(module, t, key, x, positions, indices):
    """Dense reference MLA restricted to a given per-query selection: the module's gathered
    output must match this regardless of how the selection was made."""
    m = module
    bsz, S, _ = x.shape

    # Rebuild the dense score path (same transcription as test_mla.ref_forward) with an
    # additional selection mask
    H, nope, rope_dim, v_head = m.num_q_heads, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim
    xf = x.float()
    q = xf @ t[f"{key}.q_a_proj.weight"].float().T
    q = rms_norm(q, t[f"{key}.q_a_layernorm.weight"], m.norm_eps)
    q = q @ t[f"{key}.q_b_proj.weight"].float().T
    q = q.view(bsz, S, H, m.qk_head_dim)
    q_nope, q_pe = q[..., :nope], q[..., nope:]

    ckv_kpe = xf @ t[f"{key}.kv_a_proj_with_mqa.weight"].float().T
    ckv = rms_norm(ckv_kpe[..., :m.kv_lora_rank], t[f"{key}.kv_a_layernorm.weight"], m.norm_eps)
    k_pe = ckv_kpe[..., m.kv_lora_rank:].view(bsz, S, 1, rope_dim)

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
    sel = torch.zeros((bsz, S, S), dtype = torch.bool, device = x.device)
    idx = indices.view(bsz, S, -1).long()
    bb, qq, kk = (idx >= 0).nonzero(as_tuple = True)
    sel[bb, qq, idx[bb, qq, kk]] = True
    allowed = (pos.view(bsz, 1, S) <= pos.view(bsz, S, 1)) & sel
    scores = scores.masked_fill(~allowed.unsqueeze(1), -float("inf"))
    p = torch.softmax(scores, dim = -1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, v).reshape(bsz, S, H * v_head)
    return o @ t[f"{key}.o_proj.weight"].float().T


def rel_err(a, b):
    return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-6)


def nc_forward(module, x, positions = None, params = None):
    p = {"attn_mode": "flash_attn_nc"}
    if positions is not None:
        p["positions"] = positions
    if params is not None:
        p.update(params)
    out = module.forward(x, p)
    return out, p


@pytest.mark.parametrize("S", [96, 300])
def test_dsa_selection(S):
    """Module top-k membership against the reference scores. fp16 kernel scores can order the
    k-th boundary differently from the fp32 reference, so require near-total overlap rather
    than identity."""
    topk = 64
    module, t, key = build_dsa(topk = topk, seed = S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    indices = params["dsa_topk_indices"].view(bsz, S, -1)

    ref_scores = ref_index_scores(module, t, key, x, positions)
    for b in range(bsz):
        for q_row in range(0, S, 17):
            k_eff = min(topk, q_row + 1)
            ref_top = set(ref_scores[b, q_row].topk(k_eff).indices.tolist())
            got = set(i for i in indices[b, q_row].tolist() if i >= 0)
            assert len(got) == k_eff, f"row {q_row}: {len(got)} selected, expected {k_eff}"
            overlap = len(ref_top & got)
            assert overlap >= k_eff - max(2, k_eff // 16), \
                f"row {q_row}: only {overlap}/{k_eff} of the reference selection"


@pytest.mark.parametrize("S", [96, 300])
def test_dsa_sparse_output(S):
    """Gathered attention output against the masked dense reference, driven by the module's
    own selection (so a boundary tie cannot fail this test; only the attention math can)."""
    module, t, key = build_dsa(topk = 64, seed = 100 + S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    indices = params["dsa_topk_indices"]
    ref = ref_forward_indices(module, t, key, x, positions, indices)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"


def test_dsa_dense_equivalence():
    """T <= index_topk: the sparse machinery must stand down and reproduce dense MLA."""
    module, t, key = build_dsa(topk = 64, seed = 7)
    bsz, S = 2, 64
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    assert "dsa_topk_indices" not in params, "selection ran below the sparse threshold"

    module.index_topk = 1 << 30
    dense, _ = nc_forward(module, x, positions)
    module.index_topk = 64
    assert rel_err(out, dense) == 0.0, "dense-regime forward diverged from plain dense MLA"


def test_dsa_sharing():
    """A "shared" module must consume the published selection, and must refuse to run
    without one."""
    S = 200
    full, t, key = build_dsa(topk = 64, seed = 11)
    shared, t2, _ = build_dsa(topk = 64, mode = "shared", seed = 11)

    bsz = 1
    x = (torch.randn((bsz, S, full.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out_full, params = nc_forward(full, x, positions)
    indices = params["dsa_topk_indices"]

    out_shared, _ = nc_forward(shared, x, positions, {"dsa_topk_indices": indices})
    ref = ref_forward_indices(shared, t2, key, x, positions, indices)
    assert rel_err(out_shared, ref) < 5e-3, f"rel err {rel_err(out_shared, ref):.3e}"

    with pytest.raises(AssertionError, match = "shared-indexer"):
        nc_forward(shared, x, positions)


def test_dsa_cached_vs_nc():
    """Cached path with the paged indexer plane, fed in chunks. Chunk 2 scores over the paged
    plane (past + current). The output is checked against the masked dense reference driven by
    the cached path's own per-chunk selections (the paged and contiguous scoring kernels can
    order fp16 ties at the k-th score differently, so raw output comparison against the nc
    path is only held to selection-overlap standards)."""
    S, chunk = 384, 128
    topk = 64
    module, t, key = build_dsa(topk = topk, seed = 23)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    _, nc_params = nc_forward(module, x, positions)
    nc_indices = nc_params["dsa_topk_indices"].view(bsz, S, -1)

    layer = CacheLayer_MLA_fp16(None, module, 0, 4 * PAGE_SIZE * bsz)
    layer.alloc(torch.device(device))
    assert layer.k_idx is not None, "full-indexer layer allocated no indexer plane"
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    outs = []
    chunk_indices = []
    for a in range(0, S, chunk):
        b = min(a + chunk, S)
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, a:b].contiguous(), params))
        chunk_indices.append(params["dsa_topk_indices"].view(bsz, b - a, -1))
        seqlens = seqlens + (b - a)
    out = torch.cat(outs, dim = 1)

    # Attention math over the paged pool, given the selection actually made
    k_pad = max(ci.shape[-1] for ci in chunk_indices)
    indices = torch.cat(
        [F.pad(ci, (0, k_pad - ci.shape[-1]), value = -1) for ci in chunk_indices], dim = 1)
    ref = ref_forward_indices(module, t, key, x, positions, indices)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"

    # Selection agreement with the cache-less path, allowing boundary ties to differ
    for b in range(bsz):
        for q_row in range(topk, S, 37):
            a_set = set(i for i in indices[b, q_row].tolist() if i >= 0)
            b_set = set(i for i in nc_indices[b, q_row].tolist() if i >= 0)
            overlap = len(a_set & b_set)
            assert overlap >= topk - max(2, topk // 16), \
                f"row {q_row}: cached/nc selection overlap {overlap}/{topk}"


def test_dsa_cached_decode():
    """Single-token decode steps over a sparse context: cached selection + gather at
    seqlen 1, against the cache-less forward's last row."""
    S = 200
    module, t, key = build_dsa(topk = 64, seed = 31)
    bsz = 1
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    layer = CacheLayer_MLA_fp16(None, module, 0, 4 * PAGE_SIZE)
    layer.alloc(torch.device(device))
    bt = torch.arange(4, dtype = torch.int32, device = device).view(1, 4)
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)
    prefill = S - 8
    params = {
        "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
        "cache_seqlens": seqlens, "positions": seqlens.clone(),
    }
    module.forward(x[:, :prefill].contiguous(), params)
    seqlens += prefill
    outs = []
    step_indices = []
    for i in range(prefill, S):
        params = {
            "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
            "cache_seqlens": seqlens, "positions": seqlens.clone(),
        }
        outs.append(module.forward(x[:, i:i + 1].contiguous(), params))
        step_indices.append(params["dsa_topk_indices"].view(bsz, 1, -1))
        seqlens += 1
    out = torch.cat(outs, dim = 1)

    # Full-sequence reference driven by the decode steps' own selections. Attention rows are
    # independent, so rows outside the compared region just select everything (causality is
    # intersected inside the reference)
    k_pad = step_indices[0].shape[-1]
    indices = torch.arange(S, dtype = torch.int32, device = device) \
        .view(1, 1, S).expand(bsz, S, S).contiguous()
    indices[:, prefill:, :] = F.pad(
        torch.cat(step_indices, dim = 1), (0, S - k_pad), value = -1)
    ref = ref_forward_indices(module, t, key, x, positions, indices)
    assert rel_err(out, ref[:, prefill:]) < 5e-3, \
        f"rel err {rel_err(out, ref[:, prefill:]):.3e}"
