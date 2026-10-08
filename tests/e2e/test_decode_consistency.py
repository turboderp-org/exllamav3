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

import pytest

from testlib.e2e import assert_logits_agree, forward_logits, load_model, noise_floor, prefill_decode_logits

pytestmark = [pytest.mark.models(), pytest.mark.slow]

DECODE_STEPS = 32
PROMPT_TOKENS = 256

# The 8-bit cache adds its own (small) quantization error on top of the fp16-cache bound
ABS_MEDIAN_KL = {None: 2e-3, 8: 5e-3}


@pytest.mark.parametrize("cache_bits", [None, 8], ids = ["fp16_cache", "q8_cache"])
def test_decode_matches_forward(model_id, model_dir, cache_bits, device):
    cache_args = ["-cq", str(cache_bits)] if cache_bits else []
    with load_model(model_dir, device, *cache_args) as lm:
        ids = lm.encode(max_tokens = PROMPT_TOKENS)
        full = forward_logits(lm, ids)[0]
        floor = noise_floor(lm, ids, DECODE_STEPS, full)
        got = prefill_decode_logits(lm, ids, DECODE_STEPS)
    ref = full[-(DECODE_STEPS + 1):]

    assert_logits_agree(ref, got, floor, f"{model_id}", abs_median_kl = ABS_MEDIAN_KL[cache_bits])
