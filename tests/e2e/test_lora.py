"""
A LoRA adapter (model/lora.py) is applied in Linear.forward, which the fused and graph-captured decode paths skip:
mgemm bundles and BC classes read the layers' weights directly. Layers with an adapter attached decline those paths
(Module.has_lora), so cached generation must still be the uncached forward pass with the adapter in place:
teacher-forced logits from the generator (chunked prefill, then single-token decode) must match a cache-less forward
over the same tokens, within the model's noise floor, as test_decode_consistency.py requires without an adapter.
Detaching the adapter must restore both the base model's output and the fused paths.

The adapter is synthetic: a random rank-8 pair on every linear layer of the attention, MLP and gated delta net
blocks, written in PEFT format under the model's own module keys and scaled so each delta is a tenth of its layer's
weight norm. Layers inside block-sparse MoE modules are left out: the routed and shared experts run in the MoE
kernels, which have no per-layer path to fall back to.
"""

import json
import os

import pytest
import torch

from testlib.compare import kl_divergence
from testlib.e2e import assert_logits_agree, forward_logits, load_model, noise_floor, prefill_decode_logits

pytestmark = [pytest.mark.models("attention"), pytest.mark.slow]

DECODE_STEPS = 32
PROMPT_TOKENS = 256
RANK = 8
ALPHA = 16
REL_DELTA = 0.1


def write_adapter(model, directory: str, seed: int = 0) -> int:
    """Synthetic PEFT adapter on the covered layers of the loaded model; returns the number of layers it targets"""
    from safetensors.torch import save_file
    from exllamav3.modules import Attention, BlockSparseMLP, GatedDeltaNet, GatedMLP, Linear, MLP

    covered = (Attention, GatedMLP, MLP, GatedDeltaNet)
    gen = torch.Generator().manual_seed(seed)
    tensors = {}

    def add(linear):
        k, n = linear.in_features_unpadded, linear.out_features_unpadded
        a = torch.randn(RANK, k, generator = gen) / k ** 0.5
        b = torch.randn(n, RANK, generator = gen)
        # ||B A||_F through the rank x rank Gram matrices, without forming the n x k product
        delta_norm = ALPHA / RANK * torch.trace((b.T @ b) @ (a @ a.T)).sqrt()
        b *= REL_DELTA * linear.inner.get_weight_tensor().float().norm().item() / delta_norm
        tensors[f"base_model.model.{linear.key}.lora_A.weight"] = a
        tensors[f"base_model.model.{linear.key}.lora_B.weight"] = b

    def visit(module, parent_covered: bool):
        for child in module.modules:
            if isinstance(child, Linear):
                if parent_covered:
                    add(child)
            elif not isinstance(child, BlockSparseMLP):
                visit(child, isinstance(child, covered))

    for module in model.modules:
        visit(module, False)

    save_file(tensors, os.path.join(directory, "adapter_model.safetensors"))
    with open(os.path.join(directory, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA", "r": RANK, "lora_alpha": ALPHA}, f)
    return len(tensors) // 2


def test_decode_matches_forward_with_lora(model_id, model_dir, device, tmp_path):
    from exllamav3.model.lora import LoRA

    with load_model(model_dir, device) as lm:
        targets = write_adapter(lm.model, str(tmp_path))
        if not targets:
            pytest.skip("model has no layers the adapter covers")
        ids = lm.encode(max_tokens = PROMPT_TOKENS)
        base = forward_logits(lm, ids)[0]

        lora = LoRA.from_directory(lm.model, str(tmp_path))
        assert len(lora.target_modules) == targets
        full = forward_logits(lm, ids)[0]
        floor = noise_floor(lm, ids, DECODE_STEPS, full)
        got = prefill_decode_logits(lm, ids, DECODE_STEPS)

        lora.unload()
        assert not any(m.has_lora for m in lm.model)
        restored = prefill_decode_logits(lm, ids, DECODE_STEPS)

    ref = full[-(DECODE_STEPS + 1):]
    base = base[-(DECODE_STEPS + 1):]

    # The adapter has to move the output well past the noise floor, or agreement below would prove nothing
    effect = kl_divergence(base, ref).median().item()
    assert effect > 10 * max(floor.median().item(), 1e-4), f"{model_id}: adapter effect too small ({effect:.2e})"

    assert_logits_agree(ref, got, floor, f"{model_id} with LoRA")
    assert_logits_agree(base, restored, floor, f"{model_id} after LoRA unload")
