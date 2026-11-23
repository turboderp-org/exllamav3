"""
Cached generation is the uncached forward pass, computed incrementally: for every registry model, teacher-forced
logits from the generator (chunked prefill, then single-token decode through the paged cache, recurrent state and
CUDA graphs) must match a cache-less forward over the same tokens.

The comparison is statistical and calibrated per model. Some models (MoE with many experts in particular) flip
expert choices on routing near-ties whenever the arithmetic differs at all, so even two cache-less forward passes
over different lengths disagree on the shared positions. That disagreement is measured first as the model's noise
floor, and the decode path may exceed it only by a fixed factor (with an absolute floor for models that are
nearly deterministic). A broken cache layout, stale recurrent state or misplaced graph buffer moves the median
KL by orders of magnitude, far past either bound.
"""

from contextlib import ExitStack

import pytest

from exllamav3.modules.gated_delta_net import QuantizedStateUnsupported
from testlib.e2e import assert_logits_agree, forward_logits, load_model, noise_floor, prefill_decode_logits
from testlib.graph import has_quantized_state

pytestmark = [pytest.mark.models(), pytest.mark.slow]

DECODE_STEPS = 32
PROMPT_TOKENS = 256

CACHE_VARIANTS = {
    "fp16_cache": ([], 2e-3),
    "q8_cache": (["-cq", "8"], 5e-3),
    "fp32_state": (["-rs32"], 2e-3),
    "int8_state": (["-rsf", "int8"], 4e-3),
}


@pytest.mark.parametrize("variant", list(CACHE_VARIANTS))
def test_decode_matches_forward(model_id, model_dir, variant, device):
    cache_args, abs_median_kl = CACHE_VARIANTS[variant]
    with ExitStack() as stack:
        try:
            lm = stack.enter_context(load_model(model_dir, device, *cache_args))
        except QuantizedStateUnsupported:
            pytest.skip(f"{model_id}: packed recurrent state is not supported")
        if variant == "fp32_state" and not lm.model.caps.get("recurrent_states"):
            pytest.skip("fp32 recurrent state: model has no recurrent layers")
        if variant == "int8_state" and not has_quantized_state(lm.cache):
            pytest.skip(f"{model_id}: packed recurrent state is not supported")
        ids = lm.encode(max_tokens = PROMPT_TOKENS)
        full = forward_logits(lm, ids)[0]
        floor = noise_floor(lm, ids, DECODE_STEPS, full)
        got = prefill_decode_logits(lm, ids, DECODE_STEPS)
    ref = full[-(DECODE_STEPS + 1):]

    assert_logits_agree(ref, got, floor, f"{model_id}", abs_median_kl = abs_median_kl)
