"""
BC_GatedDeltaNetSplit / BC_Mamba2 capture one CUDA graph per (bsz, seqlen, history) slot, and each graph bakes in
the geometry of the recurrent state buffers it was captured against (conv-state width, history stride). A process
can hold several caches with different max_history (e.g. one with and one without speculative history).
Regression: an older slot must fall back to eager when it is replayed against a cache whose geometry differs from
the one it captured, even after another slot's eager run against the new cache.

Runs over every registry model tagged "recurrent". Each generation's decode logits are checked against the plain
no-cache forward over the same tokens (testlib.graph.assert_decode_matches_forward).
"""

import pytest
import torch

from testlib.graph import assert_decode_matches_forward, generate_with_logits, load_with_caches, unload_model

NEW_TOKENS = 12
TOL = 2e-2
PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "In 1969, the first humans landed on the Moon. The mission was called",
    "Water boils at 100 degrees Celsius at sea level, but on a mountain",
]


@pytest.fixture
def loaded(model_dir, device):
    torch.manual_seed(0)
    # cache A: decode without speculative history; cache B: e.g. a generator with 4 draft tokens
    _, model, caches, tokenizer = load_with_caches(model_dir, device, [{"max_history": 0}, {"max_history": 4}])
    yield model, caches, tokenizer
    unload_model(model)


@pytest.mark.models("recurrent")
def test_graph_slots_follow_cache_geometry(model_id, loaded):
    model, (cache_a, cache_b), tok = loaded

    def check(label, prompt, result):
        tokens, logits = result
        assert_decode_matches_forward(model, tok.encode(prompt, add_bos = True), tokens, logits, TOL,
                                      f"{model_id}, {label}")

    # 1) capture the bsz-1 decode slot against cache A's geometry
    (a0,) = generate_with_logits(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS)
    check("cache A bsz-1", PROMPTS[0], a0)
    # 2) first use of the bsz-2 slot against cache B (eager run) - with an instance-level guard this re-armed the
    #    bsz-1 slot for cache B's geometry
    b1, b2 = generate_with_logits(model, cache_b, tok, PROMPTS[1:3], NEW_TOKENS)
    check("cache B bsz-2", PROMPTS[1], b1)
    check("cache B bsz-2", PROMPTS[2], b2)
    # 3) bsz-1 decode against cache B must not replay the graph captured against cache A
    (c0,) = generate_with_logits(model, cache_b, tok, PROMPTS[:1], NEW_TOKENS)
    check("cache B bsz-1 after slot re-arm", PROMPTS[0], c0)
    # 4) and cache A still works
    (a1,) = generate_with_logits(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS)
    check("cache A bsz-1 after cache B", PROMPTS[0], a1)
