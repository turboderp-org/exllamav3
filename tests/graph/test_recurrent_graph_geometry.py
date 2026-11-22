"""
BC_GatedDeltaNetSplit / BC_Mamba2 capture one CUDA graph per (bsz, seqlen, history) slot, and each graph bakes in
the geometry of the recurrent state buffers it was captured against (conv-state width, history stride, state
element type). A process can hold several caches with different max_history (e.g. one with and one without
speculative history) or with different recurrent state formats (fp16 and int8 pools).
Regression: an older slot must fall back to eager when it is replayed against a cache whose geometry differs from
the one it captured, even after another slot's eager run against the new cache. The state element type is part of
that snapshot: a cache whose pool format differs from the captured one falls back to the eager path (the snapshot
is written on the slot's first use, so it is not re-captured), and a same-format run keeps replaying the graph.

Runs over every registry model tagged "recurrent". The scenarios run twice, in child processes with graph capture
enabled and disabled; every generation's decode logits must match the eager run (testlib.graph.assert_matches_eager).
"""

import pytest
import torch

from exllamav3.modules.gated_delta_net import QuantPoolUnsupported
from testlib.env import get_test_device
from testlib.graph import (assert_matches_eager, generate_with_logits, has_int_pool, load_with_caches,
                           run_with_and_without_graphs, unload_model)

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


def scenario_state_dtype(model_dir: str) -> list:
    """An fp16 cache and two packed-int caches (W = 8 and W = 10) in one process, switching the pool
    format under the same (bsz, seqlen, history) slot; [(label, tokens, step logits)] per generation.
    [] if the model has no layer that carries an int pool"""
    torch.manual_seed(0)
    try:
        _, model, (cache_fp16, cache_int8, cache_int10), tok = load_with_caches(
            model_dir, get_test_device(),
            [{"max_history": 0},
             {"max_history": 0, "recurrent_dtype": torch.int8, "recurrent_bits": 8},
             {"max_history": 0, "recurrent_dtype": torch.int8, "recurrent_bits": 10}])
    except QuantPoolUnsupported:
        return []                   # no 128x128 GDN layer: KDA never goes int, so the cache is rejected
    if not has_int_pool(cache_int8):
        return []                   # the model has no layer that takes an int pool (Mamba2 keeps fp32)
    out = []

    def keep(label, result):
        tokens, logits = result
        out.append((label, tokens, [l.cpu() for l in logits]))

    # 1) the bsz-1 decode slot under the W = 8 pool: first use arms the slot (eager run; the
    #    snapshot takes int, bits 8)
    (i0,) = generate_with_logits(model, cache_int8, tok, PROMPTS[:1], NEW_TOKENS)
    keep("int8 bsz-1 (1)", i0)
    # 2) the slot's geometry matches the snapshot, so the slot captures and replays the int graph
    (i1,) = generate_with_logits(model, cache_int8, tok, PROMPTS[:1], NEW_TOKENS)
    keep("int8 bsz-1 (2)", i1)
    # 3) the W = 10 pool: recurrent_bits differs from the snapshot's bits, so the slot takes the
    #    eager path; the snapshot stays int W = 8 -- no re-arm, no re-capture
    (i2,) = generate_with_logits(model, cache_int10, tok, PROMPTS[:1], NEW_TOKENS)
    keep("int10 bsz-1 (1)", i2)
    # 4) back to W = 8: the snapshot matches again, so the slot replays the graph captured in step 2
    (i3,) = generate_with_logits(model, cache_int8, tok, PROMPTS[:1], NEW_TOKENS)
    keep("int8 bsz-1 (3)", i3)
    # 5) the fp16 pool: the pool format differs from the snapshot, so the slot takes the eager path
    (f0,) = generate_with_logits(model, cache_fp16, tok, PROMPTS[:1], NEW_TOKENS)
    keep("fp16 bsz-1", f0)
    unload_model(model)
    return out


@pytest.mark.models("recurrent")
def test_graph_state_dtype_snapshot(model_id, model_dir, device):
    """The state element type is part of the per-slot graph snapshot"""
    graphed, eager = run_with_and_without_graphs(scenario_state_dtype, model_dir, device = device)
    if not graphed and not eager:
        pytest.skip(f"{model_id}: no layer carries an int pool")
    assert_matches_eager([(f"{model_id}, {l}", t, g) for l, t, g in graphed],
                         [(f"{model_id}, {l}", t, g) for l, t, g in eager])
