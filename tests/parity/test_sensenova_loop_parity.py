"""SenseNova Loop text logits against the checkpoint's reference implementation."""
import pytest
import torch

from testlib.parity import Gate, LIGHTHOUSE_TEXT, exl3_logits, hf_reference, logits_floor_gate

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.model("sensenova-loop-hf")]


def test_logits(model_dir, device):
    from exllamav3 import Config, Tokenizer
    from transformers import AutoModelForCausalLM
    ids = Tokenizer.from_config(Config.from_directory(model_dir)).encode(LIGHTHOUSE_TEXT)[:, :256]
    ref = hf_reference(AutoModelForCausalLM, model_dir, ids, device, variant = "bundled",
                       dtype = torch.float32, attn_impl = "eager", trust_remote_code = True)
    floor = hf_reference(AutoModelForCausalLM, model_dir, ids, device, variant = "bundled",
                         dtype = torch.bfloat16, attn_impl = "sdpa", trust_remote_code = True)
    got = exl3_logits(model_dir, ids, device)
    gate = Gate("SenseNova Loop vs reference fp32 eager")
    logits_floor_gate(gate, got, ref["logits"], floor["logits"])
    gate.assert_passes()
