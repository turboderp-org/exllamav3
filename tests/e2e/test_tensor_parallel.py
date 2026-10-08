"""
Tensor parallelism: a model loaded with -tp over two devices must produce the logits of the same model on one
device, through the cache-less forward and the generator's cached prefill + decode.

Each rank computes a slice of every layer and the partial results are reduced, so the arithmetic (and on MoE
models the routing on near-ties) differs from the single-device load. As in test_decode_consistency, the bounds
are statistical and scaled by the model's own noise floor, so they hold for near-deterministic dense models and
for routing-sensitive MoE models alike, while a wrong split, a missing reduction or a rank reading another
rank's weights moves the median KL by orders of magnitude. Runs over every registry model whose architecture
supports TP.
"""

import pytest

from testlib.e2e import assert_logits_agree, forward_logits, gpu_split, load_model, noise_floor, prefill_decode_logits

pytestmark = [pytest.mark.tp, pytest.mark.models(), pytest.mark.multi_gpu(2), pytest.mark.slow]

DECODE_STEPS = 16
PROMPT_TOKENS = 256


def test_tp_matches_single_device(model_id, model_dir, devices):
    tp_devices = devices[:2]
    with load_model(model_dir, tp_devices[0]) as lm:
        if not lm.model.caps.get("supports_tp"):
            pytest.skip(f"{lm.config.architecture}: no tensor-parallel support")
        ids = lm.encode(max_tokens = PROMPT_TOKENS)
        ref_forward = forward_logits(lm, ids)[0]
        floor = noise_floor(lm, ids, DECODE_STEPS, ref_forward)
        ref_decode = prefill_decode_logits(lm, ids, DECODE_STEPS)

    with load_model(model_dir, None, "-tp", "-gs", gpu_split(tp_devices)) as lm:
        tp_forward = forward_logits(lm, ids)[0]
        tp_decode = prefill_decode_logits(lm, ids, DECODE_STEPS)

    assert_logits_agree(ref_forward, tp_forward, floor, "forward")
    assert_logits_agree(ref_decode, tp_decode, floor, "cached decode")
