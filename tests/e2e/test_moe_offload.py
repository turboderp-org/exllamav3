"""
CPU expert offload, end to end: a model whose MoE experts run (partly) on the CPU must produce the same logits as
the all-GPU load, through both the prefill path and single-token / batched decode.

Placements:
    layer   --moe_cpu_offload: the routed experts of the first half of the MoE layers run on the CPU
    split   --moe_cpu_split: the tail quarter of every MoE layer's routed experts runs on the CPU

Reference: the same model loaded entirely on the GPU, teacher-forced over the same tokens. The CPU kernels
quantize activations to int8 and accumulate differently, so agreement is statistical (testlib.e2e.
assert_logits_agree: median/mean KL scaled by the model's noise floor, top-1 at confident positions), at bounds
that a wrong expert mapping, stale weights or a broken handoff (KL of order 1) cannot meet. Runs over every registry model tagged "moe" + "mul1" (CPU offload needs mul1 experts).
"""

import json
import os

import pytest

from testlib.e2e import REFERENCE_TEXT, assert_logits_agree, forward_logits, load_model, noise_floor, prefill_decode_logits

pytestmark = [pytest.mark.moe, pytest.mark.moe_cpu, pytest.mark.models("moe", "mul1"), pytest.mark.slow]

DECODE_STEPS = 16
PROMPT_TOKENS = 256

# The CPU path's int8 activations add error on top of the routing noise floor
ABS_MEDIAN_KL = 1e-2


def placement_args(model_dir: str, placement: str) -> list[str]:
    """model_init arguments for a placement, sized from the model's config"""
    with open(os.path.join(model_dir, "config.json"), encoding = "utf8") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    if placement == "layer":
        return ["-mcl", str(max(1, tc["num_hidden_layers"] // 2))]
    if placement == "split":
        experts = tc.get("num_experts") or tc.get("n_routed_experts") or tc.get("num_local_experts")
        return ["-mcs", str(max(1, experts // 4))]
    raise ValueError(placement)


def _prompts(lm):
    ids = lm.encode(max_tokens = PROMPT_TOKENS)
    # Second, different sequence for the batched-decode run: the reference text from a later offset
    alt = lm.tokenizer.encode(REFERENCE_TEXT[len(REFERENCE_TEXT) // 3:], add_bos = True)[:, :PROMPT_TOKENS // 2]
    return ids, alt


def _run(lm, floor = False):
    ids, alt = _prompts(lm)
    forward = forward_logits(lm, ids)[0]
    return {
        "forward": forward,
        "floor": noise_floor(lm, ids, DECODE_STEPS, forward) if floor else None,
        "decode": prefill_decode_logits(lm, ids, DECODE_STEPS),
        "batched": prefill_decode_logits(lm, [ids, alt], DECODE_STEPS),
    }


@pytest.fixture(scope = "module")
def gpu_reference(model_registry, device):
    """All-GPU logits per model, computed once per model id and shared by the placements"""
    cache = {}

    def get(model_id):
        if model_id not in cache:
            with load_model(model_registry.get(model_id).path, device) as lm:
                cache[model_id] = _run(lm, floor = True)
        return cache[model_id]
    return get


@pytest.mark.parametrize("placement", ["layer", "split"])
def test_offload_matches_gpu(model_id, model_dir, placement, device, gpu_reference):
    ref = gpu_reference(model_id)
    with load_model(model_dir, device, *placement_args(model_dir, placement)) as lm:
        got = _run(lm)
    floor = ref["floor"]
    assert_logits_agree(ref["forward"], got["forward"], floor, f"{placement} prefill", abs_median_kl = ABS_MEDIAN_KL)
    assert_logits_agree(ref["decode"], got["decode"], floor, f"{placement} decode", abs_median_kl = ABS_MEDIAN_KL)
    for i, (r, g) in enumerate(zip(ref["batched"], got["batched"])):
        assert_logits_agree(r, g, floor, f"{placement} batched decode, job {i}", abs_median_kl = ABS_MEDIAN_KL)
