"""
BC_GatedDeltaNetSplit / BC_Mamba2 capture one CUDA graph per (bsz, seqlen, history) slot, and each graph bakes in
the geometry of the recurrent state buffers it was captured against (conv-state width, history stride). A process
can hold several caches with different max_history (e.g. one with and one without speculative history).
Regression: an older slot must fall back to eager when it is replayed against a cache whose geometry differs from
the one it captured, even after another slot's eager run against the new cache.

Runs over every registry model tagged "recurrent". The scenario runs twice, in child processes with graph capture
enabled and disabled; every generation's decode logits must match the eager run (testlib.graph.assert_matches_eager).
"""

import pytest
import torch

from testlib.env import get_test_device
from testlib.graph import (assert_matches_eager, generate_with_logits, load_with_caches, run_with_and_without_graphs,
                           unload_model)

NEW_TOKENS = 12
PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "In 1969, the first humans landed on the Moon. The mission was called",
    "Water boils at 100 degrees Celsius at sea level, but on a mountain",
]


def scenario(model_dir: str) -> list:
    """The cache-switching sequence (child process); [(label, tokens, step logits)] per generation"""
    torch.manual_seed(0)
    # cache A: decode without speculative history; cache B: e.g. a generator with 4 draft tokens
    _, model, (cache_a, cache_b), tok = load_with_caches(model_dir, get_test_device(),
                                                         [{"max_history": 0}, {"max_history": 4}])
    out = []

    def keep(label, result):
        tokens, logits = result
        out.append((label, tokens, [l.cpu() for l in logits]))

    # 1) capture the bsz-1 decode slot against cache A's geometry
    (a0,) = generate_with_logits(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS)
    keep("cache A bsz-1", a0)
    # 2) first use of the bsz-2 slot against cache B (eager run) - with an instance-level guard this re-armed the
    #    bsz-1 slot for cache B's geometry
    b1, b2 = generate_with_logits(model, cache_b, tok, PROMPTS[1:3], NEW_TOKENS)
    keep("cache B bsz-2 (1)", b1)
    keep("cache B bsz-2 (2)", b2)
    # 3) bsz-1 decode against cache B must not replay the graph captured against cache A
    (c0,) = generate_with_logits(model, cache_b, tok, PROMPTS[:1], NEW_TOKENS)
    keep("cache B bsz-1 after slot re-arm", c0)
    # 4) and cache A still works
    (a1,) = generate_with_logits(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS)
    keep("cache A bsz-1 after cache B", a1)
    unload_model(model)
    return out


@pytest.mark.models("recurrent")
def test_graph_slots_follow_cache_geometry(model_id, model_dir, device):
    graphed, eager = run_with_and_without_graphs(scenario, model_dir, device = device)
    assert_matches_eager([(f"{model_id}, {l}", t, g) for l, t, g in graphed],
                         [(f"{model_id}, {l}", t, g) for l, t, g in eager])
