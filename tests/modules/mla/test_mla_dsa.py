"""
DSA-on-MLA (GLM-5.2): the lightning indexer selects index_topk tokens per query and the attention core gathers
only those latent rows. The reference indexer (transformers glm_moe_dsa) is transcribed here in plain torch and
the attention reference is testlib.mla.ref_forward restricted to a selection. On random weights:

  - the module's top-k selection against the reference scores (allowing a small overlap slack at the k-th-score
    boundary, where fp16 kernel scores and fp32 reference scores can order ties differently),
  - the sparse attention output against a masked dense reference driven by the MODULE's own selection (tight:
    this isolates the gather/attention math from boundary selection noise),
  - dense equivalence when index_topk >= T,
  - cross-layer sharing ("shared" layers consume the published selection),
  - the cached path (paged indexer-key plane) against the cache-less path, chunked, over fp16 and packed-
    quantized latent caches.
"""

import pytest
import torch
import torch.nn.functional as F

from exllamav3.constants import PAGE_SIZE
from exllamav3.util.rope import RopeStyle

from testlib.compare import rel_err
from testlib.mla import (
    KEY, load_mla, rms_norm, ref_forward, make_cache, make_qcache, cached_params, run_module,
)


def build_dsa(directory, device, H = 8, hidden = 512, kv_lora = 512, nope = 128, rope_dim = 64, v_head = 128,
              q_lora = 256, idx_heads = 4, idx_dim = 128, topk = 64, mode = "full", seed = 0, wscale = 0.085):
    g = torch.Generator(device = "cpu").manual_seed(seed)

    def rnd(*shape, scale = wscale):
        return (torch.randn(*shape, generator = g) * scale).half()

    qk_head = nope + rope_dim
    t = {
        f"{KEY}.q_a_proj.weight": rnd(q_lora, hidden),
        f"{KEY}.q_a_layernorm.weight": (torch.randn(q_lora, generator = g) * 0.1 + 1).half(),
        f"{KEY}.q_b_proj.weight": rnd(H * qk_head, q_lora),
        f"{KEY}.kv_a_proj_with_mqa.weight": rnd(kv_lora + rope_dim, hidden),
        f"{KEY}.kv_a_layernorm.weight": (torch.randn(kv_lora, generator = g) * 0.1 + 1).half(),
        f"{KEY}.kv_b_proj.weight": rnd(H * (nope + v_head), kv_lora),
        f"{KEY}.o_proj.weight": rnd(hidden, H * v_head),
    }
    if mode == "full":
        # Indexer weights run hotter than the attention ones so the relu keeps a healthy mix of active and
        # clamped scores and the top-k boundary is not pure noise
        t[f"{KEY}.indexer.wq_b.weight"] = rnd(idx_heads * idx_dim, q_lora, scale = 0.25)
        t[f"{KEY}.indexer.wk.weight"] = rnd(idx_dim, hidden, scale = 0.25)
        t[f"{KEY}.indexer.k_norm.weight"] = (torch.randn(idx_dim, generator = g) * 0.1 + 1).half()
        t[f"{KEY}.indexer.k_norm.bias"] = (torch.randn(idx_dim, generator = g) * 0.05).half()
        t[f"{KEY}.indexer.weights_proj.weight"] = rnd(idx_heads, hidden, scale = 0.25)

    module = load_mla(
        t, directory, device, H = H, hidden = hidden, kv_lora = kv_lora, nope = nope, rope_dim = rope_dim,
        v_head = v_head, q_lora = q_lora, rope_style = RopeStyle.GPTJ,
        indexer_mode = mode, index_n_heads = idx_heads, index_head_dim = idx_dim, index_topk = topk,
    )
    return module, {k: v.to(device) for k, v in t.items()}, KEY


def ref_index_scores(module, t, key, x, positions):
    """Reference lightning-indexer scores (B, S, S): relu(q . k) * D**-0.5, head-weighted, fp32, interleaved rope
    on the first rope_dim dims, -inf past the causal bound."""
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


def nc_forward(module, x, positions = None, params = None):
    p = {"attn_mode": "flash_attn_nc"}
    if positions is not None:
        p["positions"] = positions
    if params is not None:
        p.update(params)
    out = module.forward(x, p)
    return out, p


def chunked_with_selection(module, x, layer, bt, chunk):
    """Cached chunked prefill; returns the output and the per-chunk selections stacked to (bsz, S, k_max)"""
    bsz, S, _ = x.shape
    chunk_params = []
    out = run_module(module, x, layer, bt, chunk = chunk, params_out = chunk_params)
    rows = [min(chunk, S - a) for a in range(0, S, chunk)]
    chunk_indices = [p["dsa_topk_indices"].view(bsz, r, -1) for p, r in zip(chunk_params, rows)]
    k_pad = max(ci.shape[-1] for ci in chunk_indices)
    indices = torch.cat([F.pad(ci, (0, k_pad - ci.shape[-1]), value = -1) for ci in chunk_indices], dim = 1)
    return out, indices


def decode_with_selection(module, x, layer, bt, prefill):
    """Cached prefill of the first `prefill` rows, then single-token decode steps. Returns the decode outputs and
    a full (bsz, S, S) selection for the reference: the steps' own selections on the decoded rows, everything
    (causality is intersected inside the reference) elsewhere, since attention rows are independent"""
    bsz, S, _ = x.shape
    seqlens = torch.zeros((bsz,), dtype = torch.int32, device = x.device)
    module.forward(x[:, :prefill].contiguous(), cached_params(layer, bt, seqlens))
    seqlens += prefill
    outs, step_indices = [], []
    for i in range(prefill, S):
        params = cached_params(layer, bt, seqlens)
        outs.append(module.forward(x[:, i:i + 1].contiguous(), params))
        step_indices.append(params["dsa_topk_indices"].view(bsz, 1, -1))
        seqlens += 1
    k_pad = step_indices[0].shape[-1]
    indices = torch.arange(S, dtype = torch.int32, device = x.device).view(1, 1, S).expand(bsz, S, S).contiguous()
    indices[:, prefill:, :] = F.pad(torch.cat(step_indices, dim = 1), (0, S - k_pad), value = -1)
    return torch.cat(outs, dim = 1), indices


@pytest.mark.parametrize("S", [96, 300])
def test_dsa_selection(tmp_path, device, S):
    """Module top-k membership against the reference scores. fp16 kernel scores can order the k-th boundary
    differently from the fp32 reference, so require near-total overlap rather than identity."""
    topk = 64
    module, t, key = build_dsa(tmp_path, device, topk = topk, seed = S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    indices = params["dsa_topk_indices"].view(bsz, S, -1)

    ref_scores = ref_index_scores(module, t, key, x, positions)
    for b in range(bsz):
        for q_row in range(0, S, 17):
            k_eff = min(topk, q_row + 1)
            ref_row = ref_scores[b, q_row]
            ref_top = set(ref_row.topk(k_eff).indices.tolist())
            got = set(i for i in indices[b, q_row].tolist() if i >= 0)
            assert len(got) == k_eff, f"row {q_row}: {len(got)} selected, expected {k_eff}"
            # The ReLU in the index score makes exact ties (at 0) common at the k-th boundary, where any tie
            # member is as valid as the reference's pick
            kth = ref_row.topk(k_eff).values[-1].item()
            overlap = len(ref_top & got) + sum(1 for i in got - ref_top if ref_row[i].item() == kth)
            assert overlap >= k_eff - max(2, k_eff // 16), \
                f"row {q_row}: only {overlap}/{k_eff} of the reference selection"


@pytest.mark.parametrize("S", [96, 300])
def test_dsa_sparse_output(tmp_path, device, S):
    """Gathered attention output against the masked dense reference, driven by the module's own selection (so a
    boundary tie cannot fail this test; only the attention math can)."""
    module, t, key = build_dsa(tmp_path, device, topk = 64, seed = 100 + S)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    indices = params["dsa_topk_indices"]
    ref = ref_forward(module, t, key, x, positions, indices)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"


def test_dsa_dense_equivalence(tmp_path, device):
    """T <= index_topk: the sparse machinery must stand down and reproduce dense MLA."""
    module, t, key = build_dsa(tmp_path, device, topk = 64, seed = 7)
    bsz, S = 2, 64
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out, params = nc_forward(module, x, positions)
    assert "dsa_topk_indices" not in params, "selection ran below the sparse threshold"

    module.index_topk = 1 << 30
    dense, _ = nc_forward(module, x, positions)
    module.index_topk = 64
    assert rel_err(out, dense) == 0.0, "dense-regime forward diverged from plain dense MLA"


def test_dsa_sharing(tmp_path, device):
    """A "shared" module must consume the published selection, and must refuse to run without one."""
    S = 200
    full, t, key = build_dsa(tmp_path / "full", device, topk = 64, seed = 11)
    shared, t2, _ = build_dsa(tmp_path / "shared", device, topk = 64, mode = "shared", seed = 11)

    bsz = 1
    x = (torch.randn((bsz, S, full.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    out_full, params = nc_forward(full, x, positions)
    indices = params["dsa_topk_indices"]

    out_shared, _ = nc_forward(shared, x, positions, {"dsa_topk_indices": indices})
    ref = ref_forward(shared, t2, key, x, positions, indices)
    assert rel_err(out_shared, ref) < 5e-3, f"rel err {rel_err(out_shared, ref):.3e}"

    with pytest.raises(AssertionError, match = "shared-indexer"):
        nc_forward(shared, x, positions)


def test_dsa_cached_vs_nc(tmp_path, device):
    """Cached path with the paged indexer plane, fed in chunks. Chunk 2 scores over the paged plane (past +
    current). The output is checked against the masked dense reference driven by the cached path's own per-chunk
    selections (the paged and contiguous scoring kernels can order fp16 ties at the k-th score differently, so
    raw output comparison against the nc path is only held to selection-overlap standards)."""
    S, chunk = 384, 128
    topk = 64
    module, t, key = build_dsa(tmp_path, device, topk = topk, seed = 23)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    _, nc_params = nc_forward(module, x, positions)
    nc_indices = nc_params["dsa_topk_indices"].view(bsz, S, -1)

    layer = make_cache(module, 4 * PAGE_SIZE * bsz)
    assert layer.k_idx is not None, "full-indexer layer allocated no indexer plane"
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    out, indices = chunked_with_selection(module, x, layer, bt, chunk)

    # Attention math over the paged pool, given the selection actually made
    ref = ref_forward(module, t, key, x, positions, indices)
    assert rel_err(out, ref) < 5e-3, f"rel err {rel_err(out, ref):.3e}"

    # Selection agreement with the cache-less path, allowing boundary ties to differ
    for b in range(bsz):
        for q_row in range(topk, S, 37):
            a_set = set(i for i in indices[b, q_row].tolist() if i >= 0)
            b_set = set(i for i in nc_indices[b, q_row].tolist() if i >= 0)
            overlap = len(a_set & b_set)
            assert overlap >= topk - max(2, topk // 16), \
                f"row {q_row}: cached/nc selection overlap {overlap}/{topk}"


def test_dsa_cached_decode(tmp_path, device):
    """Single-token decode steps over a sparse context: cached selection + gather at seqlen 1, against the
    reference driven by the steps' own selections."""
    S = 200
    module, t, key = build_dsa(tmp_path, device, topk = 64, seed = 31)
    bsz = 1
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    layer = make_cache(module, 4 * PAGE_SIZE)
    bt = torch.arange(4, dtype = torch.int32, device = device).view(1, 4)
    prefill = S - 8
    out, indices = decode_with_selection(module, x, layer, bt, prefill)
    ref = ref_forward(module, t, key, x, positions, indices)
    assert rel_err(out, ref[:, prefill:]) < 5e-3, f"rel err {rel_err(out, ref[:, prefill:]):.3e}"


@pytest.mark.parametrize("bits", [8, 4])
def test_dsa_cached_quant_prefill(tmp_path, device, bits):
    """Sparse cached prefill over the packed-quantized latent (CacheLayer_MLA_quant): the gather kernel
    dequantizes online in the H32-rotated domain. Reference: the masked dense attention driven by the module's
    own selection, with the latent round-tripped through the same quantizer (the selection is identical to the
    fp16-cache case: indexer planes stay fp16). Residual error is the reference's fp16 ckv rounding flipping a
    few quantization levels, hence the width-dependent tolerance."""
    S, chunk = 384, 128
    topk = 64
    module, t, key = build_dsa(tmp_path, device, topk = topk, seed = 41)
    bsz = 2
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    layer = make_qcache(module, 4 * PAGE_SIZE * bsz, bits)
    bt = torch.arange(4 * bsz, dtype = torch.int32, device = device).view(bsz, 4)
    out, indices = chunked_with_selection(module, x, layer, bt, chunk)
    ref = ref_forward(module, t, key, x, positions, indices, ckv_quant_bits = bits)
    tol = {8: 6e-3, 4: 2.5e-2}[bits]
    assert rel_err(out, ref) < tol, f"rel err {rel_err(out, ref):.3e} (tol {tol})"


@pytest.mark.parametrize("bits", [8, 4])
def test_dsa_cached_quant_decode(tmp_path, device, bits):
    """Single-token decode steps over a sparse context held in the packed-quantized cache."""
    S = 200
    module, t, key = build_dsa(tmp_path, device, topk = 64, seed = 43)
    bsz = 1
    x = (torch.randn((bsz, S, module.hidden_size), device = device) * 0.5).half()
    positions = torch.zeros((bsz,), dtype = torch.int32, device = device)

    layer = make_qcache(module, 4 * PAGE_SIZE, bits)
    bt = torch.arange(4, dtype = torch.int32, device = device).view(1, 4)
    prefill = S - 8
    out, indices = decode_with_selection(module, x, layer, bt, prefill)
    ref = ref_forward(module, t, key, x, positions, indices, ckv_quant_bits = bits)
    tol = {8: 6e-3, 4: 2.5e-2}[bits]
    e = rel_err(out, ref[:, prefill:])
    assert e < tol, f"rel err {e:.3e} (tol {tol})"
