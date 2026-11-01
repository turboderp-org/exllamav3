"""The prefill pick is cached per key, so the key must cover everything that can change the probe.

A chunked prefill calls this kernel at several q_len under one module. q_len was not in the key, so
the pick made for the first chunk was reused for the rest, including chunks whose probe would have
walked a different ladder. Regression: OutOfResources, 131072 B required on a 65536 B LDS budget.
"""
import pytest
import torch

from exllamav3.modules.attention_fn import smem, triton_paged

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEVICE = torch.device("cuda:0")
# A tensor-parallel shard of Qwen3-27B: 18 q heads / 3, 3 kv heads / 3, head_dim 256
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 6, 1, 256
PAST_TOKENS = 22528


def _keys(monkeypatch, lengths, max_kv_len = None):
    from testlib.attention import rand_paged_cache

    keys = []
    original = triton_paged.pick_config

    def spy(device, name, key, candidates, probe):
        if name == "paged_attn_prefill":
            keys.append(key)
        return original(device, name, key, candidates, probe)

    monkeypatch.setattr(triton_paged, "pick_config", spy)
    smem._picks.clear()
    # One cache for every length: rebuilding it changes the block-table width, which moves the kv
    # bound and splits keys for a reason unrelated to q_len
    k_cache, v_cache, block_table, seqlens = rand_paged_cache(
        1, PAST_TOKENS + 2048, N_KV_HEADS, HEAD_DIM, PAST_TOKENS, DEVICE)
    for q_len in lengths:
        q = torch.randn((1, q_len, N_Q_HEADS, HEAD_DIM), dtype = torch.half, device = DEVICE)
        triton_paged.paged_attn_triton_prefill(
            q, None, None, k_cache, v_cache, block_table, seqlens,
            causal = True, max_kv_len = max_kv_len)
    return keys


def test_prefill_pick_key_covers_the_launch_shape(monkeypatch):
    # 4096 and 132 were the lengths in the report: they straddle no single predicate, but they do
    # differ in the %16 bucket and, at 127 vs 128, the stock-tile branch flips. Both must be
    # distinct picks; sharing one is what crashed.
    assert len(set(_keys(monkeypatch, [4096, 132]))) == 2

    # The stock-tile branch delegates on q_len >= 128 and hands the ladder a different stock tile,
    # so these must not share a pick even though both are long enough to be realistic chunks
    assert len(set(_keys(monkeypatch, [128, 127]))) == 2

    # Lengths that agree on every predicate share a pick, so a chunked prefill does not probe once
    # per chunk
    assert len(set(_keys(monkeypatch, [1024, 2048, 4096]))) == 1

    # max_kv_len clamps bound_kv in prepare(), so it reaches the key too
    assert _keys(monkeypatch, [4096], max_kv_len = 1024) != _keys(monkeypatch, [4096])
