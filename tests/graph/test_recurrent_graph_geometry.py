"""Recurrent CUDA graphs follow cache geometry, state format and packed width.

A geometry or format mismatch falls back to eager execution. Packed graphs patch both state and scale pointers
when replayed with a different cache of the same format and width. Graphed and eager scenarios must produce the
same tokens and decode logits.
"""

import pytest
import torch

from exllamav3.modules.gated_delta_net import QuantizedStateUnsupported
from testlib.env import get_test_device
from testlib.graph import (assert_matches_eager, generate_with_logits, has_quantized_state, load_with_caches,
                           run_with_and_without_graphs, unload_model)

NEW_TOKENS = 12
PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "In 1969, the first humans landed on the Moon. The mission was called",
    "Water boils at 100 degrees Celsius at sea level, but on a mountain",
]


def scenario(model_dir: str) -> list:
    """Cache A bsz1 captures a slot; cache B bsz2 uses different cache geometry. Cache B's later bsz1
    is another cache-geometry mismatch requiring eager fallback; cache A's final bsz1 reuses its graph."""
    torch.manual_seed(0)
    # cache A: decode without speculative history; cache B: e.g. a generator with 4 draft tokens
    _, model, (cache_a, cache_b), tok = load_with_caches(model_dir, get_test_device(),
                                                         [{"max_history": 0}, {"max_history": 4}])
    out = []

    def keep(label, result):
        tokens, logits = result
        out.append((label, tokens, [l.cpu() for l in logits]))

    (a0,) = generate_with_logits(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS, max_batch_size = 1)
    keep("cache A bsz-1", a0)
    b1, b2 = generate_with_logits(model, cache_b, tok, PROMPTS[1:3], NEW_TOKENS)
    keep("cache B bsz-2 (1)", b1)
    keep("cache B bsz-2 (2)", b2)
    (c0,) = generate_with_logits(model, cache_b, tok, PROMPTS[:1], NEW_TOKENS, max_batch_size = 1)
    keep("cache B bsz-1 geometry fallback", c0)
    (a1,) = generate_with_logits(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS, max_batch_size = 1)
    keep("cache A bsz-1 after cache B", a1)
    unload_model(model)
    return out


@pytest.mark.models("recurrent")
def test_graph_slots_follow_cache_geometry(model_id, model_dir, device):
    graphed, eager = run_with_and_without_graphs(scenario, model_dir, device = device)
    assert_matches_eager([(f"{model_id}, {l}", t, g) for l, t, g in graphed],
                         [(f"{model_id}, {l}", t, g) for l, t, g in eager])


def scenario_state_format(model_dir: str) -> list:
    torch.manual_seed(0)
    try:
        _, model, (cache_fp16, cache_int8_a, cache_int8_b, cache_int10), tok = load_with_caches(
            model_dir, get_test_device(),
            [{"max_history": 0},
             {"max_history": 0, "recurrent_dtype": torch.int8, "recurrent_bits": 8},
             {"max_history": 0, "recurrent_dtype": torch.int8, "recurrent_bits": 8},
             {"max_history": 0, "recurrent_dtype": torch.int8, "recurrent_bits": 10}])
    except QuantizedStateUnsupported:
        return []
    if not has_quantized_state(cache_int8_a):
        return []
    out = []

    def generate(label, cache):
        (result,) = generate_with_logits(model, cache, tok, PROMPTS[:1], NEW_TOKENS, max_batch_size = 1)
        tokens, logits = result
        out.append((label, tokens, [logit.cpu() for logit in logits]))

    generate("int8 A capture", cache_int8_a)
    generate("int8 A replay", cache_int8_a)
    generate("int8 B scale pointer patch", cache_int8_b)
    generate("int10 width fallback", cache_int10)
    generate("fp16 format fallback", cache_fp16)
    unload_model(model)
    return out


@pytest.mark.models("recurrent")
def test_graph_state_format_fallback_and_replay(model_id, model_dir, device):
    graphed, eager = run_with_and_without_graphs(scenario_state_format, model_dir, device = device)
    if not graphed and not eager:
        pytest.skip(f"{model_id}: packed recurrent state is not supported")
    assert_matches_eager([(f"{model_id}, {l}", t, g) for l, t, g in graphed],
                         [(f"{model_id}, {l}", t, g) for l, t, g in eager])
