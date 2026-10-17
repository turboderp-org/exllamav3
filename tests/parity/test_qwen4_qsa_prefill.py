"""
Qwen3.8-Flash-Next QSA cached sparse path: the kernel-backed implementation (select_indices_paged +
qsa_sparse_attend_rows PAGED=1) against the chunked-einsum implementation it replaced (kept here as the fp32
reference), on the quantized 4-layer stub's QSA attention module (registry role qwen4-exp-stub). Covers a prefill
chunk straddling the sparse threshold, a long sparse prefill chunk, batched decode on the eager fallback (bsz 2) and
the MTP-verify shape (q_len 5); every attention path is state-idempotent, so the same forward runs through both.
Also the plane update and selection kernels against their torch references (*_ref): selection may differ by a pool
where fp16 kernel scores tie at the top-k boundary, so it is checked as exact-row fraction plus overlap.
"""

import pytest
import torch
import torch.nn.functional as F

from testlib.parity import Gate, mutated, rfn

pytestmark = pytest.mark.model("qwen4-exp-stub")

MAX_TOKENS = 8192
B = 2


def ref_sparse_attend(self, layer, a, q, q_idx, block_table, cache_seqlens_cpu, q_chunk = 128):
    """The replaced chunked grouped-einsum implementation, verbatim (fp32 math)"""
    bsz, seqlen = q.shape[:2]
    dev = q.device
    cr = self.compress_ratio
    dk = self.head_dim
    page_sz = layer.raw_k.shape[1]
    blocks_per_page = layer.pooled.shape[1]
    pooled_flat = layer.pooled.view(-1, dk)
    kvh, hd = a.num_kv_heads, a.head_dim
    k_flat = layer.k.view(-1, kvh, hd)
    v_flat = layer.v.view(-1, kvh, hd)
    out = torch.empty_like(q)
    for r in range(bsz):
        p = int(cache_seqlens_cpu[r].item())
        pages = block_table[r].long()
        nbf = (p + seqlen) // cr
        blocks = torch.arange(nbf, device = dev)
        pflat = pages[(blocks * cr) // page_sz] * blocks_per_page + blocks % blocks_per_page
        pk = pooled_flat[pflat]
        for c0 in range(0, seqlen, q_chunk):
            c1 = min(c0 + q_chunk, seqlen)
            C = c1 - c0
            qpos = p + torch.arange(c0, c1, device = dev)
            nbq = (qpos + 1) // cr
            scores = torch.einsum("shd,nd->shn", q_idx[r, c0:c1].float(), pk.float())
            scores = F.relu(scores).sum(dim = 1) * self.scale
            scores = scores.masked_fill(blocks.unsqueeze(0) >= nbq.unsqueeze(1), -torch.inf)
            ksel = min(self.block_topk, nbf)
            top = scores.topk(ksel, dim = -1)
            sel_valid = top.values > -torch.inf
            sel_tok = (top.indices * cr).unsqueeze(-1) + torch.arange(cr, device = dev)
            sel_tok = sel_tok.flatten(1)
            sel_ok = sel_valid.unsqueeze(-1).expand(-1, -1, cr).flatten(1)
            tail_tok = (nbq * cr).unsqueeze(1) + torch.arange(cr, device = dev)
            tok = torch.cat((sel_tok, tail_tok), dim = 1)
            ok = torch.cat((sel_ok, torch.ones_like(tail_tok, dtype = torch.bool)), dim = 1)
            ok &= tok <= qpos.unsqueeze(1)
            tok = tok.clamp(max = p + seqlen - 1)
            tflat = pages[tok // page_sz] * page_sz + tok % page_sz
            Ks = k_flat[tflat]
            Vs = v_flat[tflat]
            g = a.num_q_heads // kvh
            qs = q[r, c0:c1].view(C, kvh, g, hd)
            sc = torch.einsum("ckgd,clkd->ckgl", qs.float(), Ks.float()) * a.sm_scale
            sc = sc.masked_fill(~ok.view(C, 1, 1, -1), -torch.inf)
            probs = torch.softmax(sc, dim = -1)
            o = torch.einsum("ckgl,clkd->ckgd", probs, Vs.float())
            out[r, c0:c1] = o.view(C, a.num_q_heads, hd).half()
    return out


def _rowsets(t):
    return [set(r[r >= 0].tolist()) for r in t]


def _selection_stats(a, b):
    """(fraction of rows with identical index sets, minimum per-row overlap with the reference)"""
    A, R = _rowsets(a), _rowsets(b)
    exact = sum(x == y for x, y in zip(A, R)) / len(A)
    ov = min(len(x & y) / max(1, len(y)) for x, y in zip(A, R))
    return exact, ov


@torch.inference_mode()
def test_sparse_path_vs_reference(model_registry, device):
    import exllamav3.modules.attn as attn_mod
    from exllamav3 import Config, Model
    from exllamav3.cache.qsa import CacheLayer_qsa
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.modules import Attention
    from exllamav3.modules.qsa_indexer import QSAIndexer

    config = Config.from_directory(model_registry.get("qwen4-exp-stub").path)
    model = Model.from_config(config)
    attn = next(m for module in model.modules for m in module if isinstance(m, Attention) and m.qsa_indexer is not None)
    config.stc.begin_deferred_load()
    attn.load(device)
    config.stc.end_deferred_load()
    config.stc.close()
    idx = attn.qsa_indexer
    layer = CacheLayer_qsa(config, attn, cache_id = 0, max_num_tokens = MAX_TOKENS)
    layer.alloc(device)
    pages_per_row = MAX_TOKENS // PAGE_SIZE // B
    bt = torch.arange(B * pages_per_row, dtype = torch.int32, device = device).view(B, pages_per_row)
    hid = attn.hidden_size
    gen = torch.Generator().manual_seed(4)

    def x_rand(seqlen):
        return (torch.randn(B, seqlen, hid, generator = gen) * 0.5).half().to(device)

    def params(seqlens):
        return {
            "attn_mode": "flash_attn",
            "cache": layer,
            "block_table": bt,
            "cache_seqlens": torch.tensor(seqlens, dtype = torch.int32),
            "positions": torch.tensor(seqlens, dtype = torch.int32),
            "causal": True,
            "batch_shape": (B, MAX_TOKENS // B),
        }

    def run_both(x, seqlens):
        o_new = attn.forward(x, params(seqlens)).clone()
        with mutated([QSAIndexer], "sparse_attend", ref_sparse_attend):
            o_ref = attn.forward(x, params(seqlens))
        return rfn(o_new, o_ref)

    try:
        # dense prefill of both rows to 1500
        for t in (0, 500, 1000):
            attn.forward(x_rand(500), params([t, t]))
        gate = Gate("QSA cached sparse path vs the einsum reference")
        r = {"straddling chunk 1500 -> 2700": run_both(x_rand(1200), [1500, 1500]),
             "sparse chunk 2700 -> 3900": run_both(x_rand(1200), [2700, 2700])}
        # batched decode fallback (bsz 2, q_len 1) and the MTP-verify shape (q_len 5), eager even where BC could run
        with mutated([attn_mod], "_bc_attn_enable", False):
            r["decode fallback bsz 2 q_len 1"] = run_both(x_rand(1), [3900, 3900])
            r["MTP-verify bsz 2 q_len 5"] = run_both(x_rand(5), [3901, 3901])
        # fp16 kernel scores tie-break differently from the fp32 reference at the top-k boundary now and then; on
        # few-row shapes one flipped pool moves that row's output ~1e-2
        for name, v in r.items():
            gate.lt(name, v, 1e-2)

        # plane update and selection kernels vs the torch references: reference first, snapshot, then the kernel path
        # over the same chunk (idempotent)
        pos = [3906, 3906]      # rows must stay inside the cache (MAX_TOKENS // B per row)
        x = x_rand(150)
        csl = torch.tensor(pos, dtype = torch.int32)
        q_ref = idx.update_planes_ref(layer, x, attn.rope, bt, csl, {}).clone()
        raw_ref, pool_ref = layer.raw_k.clone(), layer.pooled.clone()
        q_k = idx.update_planes(layer, x, attn.rope, bt, csl, {}).clone()
        cr = idx.compress_ratio
        nbc = (pos[0] + 150) // cr
        bpp = layer.pooled.shape[1]
        blk = torch.arange(nbc, device = device)
        pflat = torch.cat([bt[b].long()[(blk * cr) // layer.raw_k.shape[1]] * bpp + blk % bpp for b in range(B)])
        gate.true("raw plane exact", torch.equal(layer.raw_k, raw_ref))
        gate.lt("pooled plane rfn", rfn(layer.pooled.view(-1, idx.head_dim)[pflat],
                                        pool_ref.view(-1, idx.head_dim)[pflat]), 5e-3)
        gate.lt("q rfn", rfn(q_k, q_ref), 5e-3)
        exact, ov = _selection_stats(idx.select_indices_paged(layer, q_k, bt, csl),
                                     idx.select_indices_paged_ref(layer, q_k, bt, csl))
        gate.gt("select_indices_paged exact rows", exact, 0.9)
        gate.gt("select_indices_paged min overlap", ov, 0.99)
        q2, rk2 = idx.project(x_rand(2600), attn.rope, {})
        pooled2 = idx.pool_keys(rk2, attn.rope, {})
        exact, ov = _selection_stats(idx.select_indices(q2, pooled2, 0, 2600),
                                     idx.select_indices_ref(q2, pooled2, 0, 2600))
        gate.gt("select_indices (nc) exact rows", exact, 0.9)
        gate.gt("select_indices (nc) min overlap", ov, 0.99)
        gate.assert_passes()
    finally:
        attn.unload()
