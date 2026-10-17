"""
Qwen3.8-Flash-Next QSA attention: the BC_Attention graph path (captured sparse decode) against the eager Python path,
on the quantized 4-layer stub's real QSA attention module (registry role qwen4-exp-stub), over an fp16 cache
(CacheLayer_qsa) and an 8-bit quantized one (CacheLayer_qsa_quant).

One attention module is driven through chunked prefill (eager dispatch), then single-token decode steps crossing the
sparse threshold. Every step runs twice on identical pre-step state, eager first (BC disabled) then BC, which is
sound because every plane / cache write is an idempotent overwrite at the same positions. Checked: outputs per step
in the dense and sparse regimes, the selected token sets in the sparse regime against an fp32 torch selection, the
pooled plane in absolute terms against a torch recompute (persistent plane corruption that path-vs-path comparison
cannot see, since a wrong pool written by BC is read by the eager reference too), and multi-token sparse steps
(spec-decode shapes, q_len 2..16) through both C++ passes (first call, then capture + replay).
"""

import pytest
import torch

from testlib.parity import Gate, mutated, rfn

pytestmark = pytest.mark.model("qwen4-exp-stub")

MAX_TOKENS = 4096
T0 = 2000           # prefill length, below the sparse threshold
STEPS = 160         # decode steps: dense until the threshold, then sparse
CHUNK = 500
MT_QLENS = (2, 3, 4, 8, 16)


@pytest.fixture(scope = "module")
def qsa_attn(model_registry, device):
    from exllamav3 import Config, Model
    from exllamav3.modules import Attention
    config = Config.from_directory(model_registry.get("qwen4-exp-stub").path)
    model = Model.from_config(config)
    attn = next((m for module in model.modules for m in module if isinstance(m, Attention) and m.qsa_indexer
                 is not None), None)
    assert attn is not None, "no QSA attention module found"
    config.stc.begin_deferred_load()
    attn.load(device)
    config.stc.end_deferred_load()
    config.stc.close()
    yield config, attn
    attn.unload()


def _cache_layer(config, attn, cache, device):
    from exllamav3.cache.qsa import CacheLayer_qsa, CacheLayer_qsa_quant
    if cache == "fp16":
        layer = CacheLayer_qsa(config, attn, cache_id = 0, max_num_tokens = MAX_TOKENS)
    else:
        bits = int(cache[1:])
        layer = CacheLayer_qsa_quant(config, attn, cache_id = 0, max_num_tokens = MAX_TOKENS, k_bits = bits,
                                     v_bits = bits)
    layer.alloc(device)
    return layer


@pytest.mark.parametrize("cache", ["fp16", "q8"])
@torch.inference_mode()
def test_bc_vs_eager(qsa_attn, cache, device):
    import exllamav3.modules.attn as attn_mod
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.modules.qsa_indexer import _rope as qsa_rope
    from exllamav3.util.tensor import g_tensor_cache
    config, attn = qsa_attn
    idx = attn.qsa_indexer
    layer = _cache_layer(config, attn, cache, device)
    bt = torch.arange(MAX_TOKENS // PAGE_SIZE, dtype = torch.int32, device = device).unsqueeze(0)
    x_all = (torch.randn(1, T0 + STEPS, attn.hidden_size, generator = torch.Generator().manual_seed(3))
             .half().to(device))

    def params(pos):
        return {
            "attn_mode": "flash_attn",
            "cache": layer,
            "block_table": bt,
            "cache_seqlens": torch.tensor([pos], dtype = torch.int32),
            "positions": torch.tensor([pos], dtype = torch.int32),
            "causal": True,
            "batch_shape": (1, MAX_TOKENS),
        }

    def eager(x, pos):
        with mutated([attn_mod], "_bc_attn_enable", False):
            return attn.forward(x, params(pos)).clone()

    def bc_indices():
        """The BC sparse slot's expanded index static (shared bucketed buffer)"""
        cr, sel = idx.compress_ratio, idx.block_topk
        k_pad = -(-(sel * cr + cr - 1) // 32) * 32
        return g_tensor_cache.get_bucketed(device, k_pad, torch.int32, "bca_qsa_indices").view(1, k_pad)

    def eager_selection(q_idx, pos):
        """Reference selected token set for the single query at pos (fp32 torch, independent of the kernel path)"""
        cr, dk = idx.compress_ratio, idx.head_dim
        bpp = layer.pooled.shape[1]
        nbq = (pos + 1) // cr
        blocks = torch.arange(nbq, device = device)
        pflat = bt[0].long()[(blocks * cr) // PAGE_SIZE] * bpp + blocks % bpp
        pk = layer.pooled.view(-1, dk)[pflat]
        scores = torch.einsum("hd,nd->hn", q_idx.view(idx.n_heads, dk).float(), pk.float())
        scores = torch.relu(scores).sum(dim = 0) * idx.scale
        top = scores.topk(min(idx.block_topk, nbq)).indices
        toks = ((top * cr).unsqueeze(1) + torch.arange(cr, device = device)).flatten()
        tail = nbq * cr + torch.arange(cr, device = device)
        return set(toks.tolist()) | set(tail[tail <= pos].tolist())

    # prefill (eager dispatch path: seqlen > BC max)
    for t in range(0, T0, CHUNK):
        attn.forward(x_all[:, t:t + CHUNK].contiguous(), params(t))

    # decode steps, eager vs BC on identical state
    dense, sparse, overlap_min = [], [], 1.0
    for step in range(STEPS):
        pos = T0 + step
        xt = x_all[:, pos:pos + 1].contiguous()
        is_sparse = pos + 1 > idx.sparse_threshold()
        o_eager = eager(xt, pos)
        o_bc = attn.forward(xt, params(pos))
        assert o_bc is not None
        (sparse if is_sparse else dense).append(rfn(o_bc, o_eager))
        if is_sparse and step % 20 == 0:
            q_idx, _ = idx.project(xt, attn.rope, {}, position = pos)
            got = bc_indices()[0]
            got = set(got[got >= 0].tolist())
            ref = eager_selection(q_idx, pos)
            overlap_min = min(overlap_min, len(got & ref) / max(1, len(ref)))

    # pooled plane audit: recompute every complete pool from the raw plane in torch, against the HF-validated eager
    # math (pool mean -> k_layernorm -> rope at the block start)
    t_end = T0 + STEPS
    cr, dk = idx.compress_ratio, idx.head_dim
    nb = t_end // cr
    tok = torch.arange(nb, device = device).unsqueeze(1) * cr + torch.arange(cr, device = device)
    tflat = bt[0].long()[tok // PAGE_SIZE] * PAGE_SIZE + tok % PAGE_SIZE
    ref = layer.raw_k.view(-1, dk)[tflat].float().mean(dim = 1).half()
    ref = idx.k_layernorm.forward(ref.unsqueeze(0), {}).squeeze(0)
    attn.rope.expand_cache(t_end + 8)
    starts = torch.arange(nb, device = device) * cr
    ref = qsa_rope(ref, attn.rope.cached_cos[starts].half(), attn.rope.cached_sin[starts].half())
    bpp = layer.pooled.shape[1]
    pflat = bt[0].long()[starts // PAGE_SIZE] * bpp + torch.arange(nb, device = device) % bpp
    pool_rfn = rfn(layer.pooled.view(-1, dk)[pflat], ref)

    # multi-token sparse steps (spec-decode shapes): per-row selection through the regime-1 slots, eager C++ first
    # call, then capture + replay
    mt = []
    mt_pos = t_end
    g = torch.Generator().manual_seed(11)
    for q_len in MT_QLENS:
        xt = (torch.randn(1, q_len, attn.hidden_size, generator = g) * 0.5).half().to(device)
        o_eager = eager(xt, mt_pos)
        for _ in range(2):
            mt.append(rfn(attn.forward(xt, params(mt_pos)).clone(), o_eager))
        mt_pos += q_len

    gate = Gate(f"QSA BC vs eager ({cache} cache)")
    gate.true("both regimes", dense and sparse, f"{len(dense)} dense, {len(sparse)} sparse steps")
    gate.lt("dense steps max rfn", max(dense), 0.02)
    gate.lt("sparse steps max rfn", max(sparse), 0.05)
    gate.gt("min selection overlap", overlap_min, 0.98)
    gate.lt("pooled plane rfn", pool_rfn, 1e-3)
    gate.lt("multi-token steps max rfn", max(mt), 0.05)
    gate.assert_passes()
