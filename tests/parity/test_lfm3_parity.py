"""
LFM3 (Lfm3ForCausalLM) parity against the reference implementation shipped with the checkpoint (transformers has no
native LFM3 class yet, so the release's own modeling_lfm3.py is loaded with trust_remote_code) on real weights
(registry role lfm3-hf, unquantized bf16).

Covers the parts of the architecture that are easy to get wrong: the sandwich norms (a norm after the attention/conv
output and after the MLP output, before the residual add), the per-layer-type RoPE (full attention rotates half the
head dim while keeping the frequency progression of the full dim, sliding attention rotates all of it with a smaller
theta), the sigmoid gate applied to the attention output per head, the depthwise short convolution with the b/c
gating, and the sliding window counted like the HF mask (the query included). The mutation test drops the sliding
window, which has to move the logits.
"""

import pytest
import torch

from testlib.parity import Gate, LIGHTHOUSE_TEXT, exl3_logits, hf_reference, logits_floor_gate

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.model("lfm3-hf")]

SEQ_LEN = 256


@pytest.fixture(scope = "module")
def input_ids(model_registry):
    from exllamav3 import Config, Tokenizer
    model_dir = model_registry.get("lfm3-hf").path
    return Tokenizer.from_config(Config.from_directory(model_dir)).encode(LIGHTHOUSE_TEXT)[:, :SEQ_LEN]


def _hf(model_dir, ids, device, attn_impl, dtype):
    from transformers import AutoModelForCausalLM
    return hf_reference(AutoModelForCausalLM, model_dir, ids, device, variant = "bundled", dtype = dtype,
                        attn_impl = attn_impl, trust_remote_code = True)


def _drop_sliding_window(model):
    """configure(): let the sliding-window layers attend to the whole sequence"""
    from exllamav3.modules import Attention
    for m in model.modules:
        if isinstance(m, Attention) and m.sliding_window > 0:
            m.sliding_window = -1


def test_logits(model_dir, input_ids, device):
    ref = _hf(model_dir, input_ids, device, "eager", torch.float32)
    floor = _hf(model_dir, input_ids, device, "sdpa", torch.bfloat16)
    got = exl3_logits(model_dir, input_ids, device)
    gate = Gate("LFM3 vs HF fp32 eager, floor HF bf16 sdpa")
    logits_floor_gate(gate, got, ref["logits"], floor["logits"])
    gate.assert_passes()


def test_mutation_sliding_window(model_dir, input_ids, device):
    ref = _hf(model_dir, input_ids, device, "eager", torch.float32)
    got = exl3_logits(model_dir, input_ids, device, configure = _drop_sliding_window)
    gate = Gate("LFM3 with the sliding window dropped")
    logits_floor_gate(gate, got, ref["logits"], ref["logits"])
    gate.assert_fails()
