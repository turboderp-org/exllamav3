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

from exllamav3.modules.gated_delta_net import Int8PoolUnsupported
from testlib.e2e import assert_logits_agree, forward_logits, load_model, noise_floor, prefill_decode_logits
from testlib.graph import has_int8_pool

pytestmark = [pytest.mark.models(), pytest.mark.slow]

DECODE_STEPS = 32
PROMPT_TOKENS = 256

# The 8-bit cache adds its own (small) quantization error on top of the fp16-cache bound. The default fp16
# recurrent state (recurrent models only) rounds the state once per decode step, which stays within that
# bound; the fp32 state variant checks the other storage dtype. The int8 pool quantizes once per pass-end
# store, so its bound is 4e-3, twice the fp16 one
CACHE_VARIANTS = {
    "fp16_cache": ([], 2e-3),
    "q8_cache": (["-cq", "8"], 5e-3),
    "fp32_state": (["-rsf", "fp32"], 2e-3),
    "int8_state": (["-rsf", "int8"], 4e-3),
}


@pytest.mark.parametrize("variant", list(CACHE_VARIANTS))
def test_decode_matches_forward(model_id, model_dir, variant, device):
    cache_args, abs_median_kl = CACHE_VARIANTS[variant]
    with ExitStack() as stack:
        try:
            lm = stack.enter_context(load_model(model_dir, device, *cache_args))
        except Int8PoolUnsupported:
            # the int8 cache is rejected on a model with no 128x128 GDN layer (KDA never goes int8)
            pytest.skip(f"{model_id}: no layer carries an int8 pool")
        if variant == "fp32_state" and not lm.model.caps.get("recurrent_states"):
            pytest.skip("fp32 recurrent state: model has no recurrent layers")
        if variant == "int8_state" and not has_int8_pool(lm.cache):
            pytest.skip(f"{model_id}: no layer carries an int8 pool")
        ids = lm.encode(max_tokens = PROMPT_TOKENS)
        full = forward_logits(lm, ids)[0]
        floor = noise_floor(lm, ids, DECODE_STEPS, full)
        got = prefill_decode_logits(lm, ids, DECODE_STEPS)
    ref = full[-(DECODE_STEPS + 1):]

    assert_logits_agree(ref, got, floor, f"{model_id}", abs_median_kl = abs_median_kl)
