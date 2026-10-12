"""
GLM-5.2 (GlmMoeDsaForCausalLM) per-stage parity against HF transformers on a real-weight stub (registry role
glm52-stub-hf: the first layers of the release, covering dense + full-indexer, sparse + shared and sparse + full
blocks, unquantized).

exllamav3 states are captured after the embedding, after every block but the last and after the final norm, aligned
with HF's hidden_states. Judgement is against the HF eager-vs-sdpa noise floor (testlib.parity.FLOOR_K) for every
stage and for the logits. With index_topk below the sequence length the run exercises the real sparse selection path
and cross-layer top-k sharing; with the config's index_topk (>= T) the dense path must be selection-equivalent.
exllamav3 streams one module at a time; the HF references are disk-cached ($EXL3_HFREF_CACHE).
"""

import pytest
import torch

from testlib.parity import (Gate, LIGHTHOUSE_TEXT, exl3_stream, hf_aligned_capture, hf_reference,
                            hidden_state_gate, logits_floor_gate)

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.model("glm52-stub-hf")]

SEQ_LEN = 512
INDEX_TOPK = {"dense": None, "sparse": 64}


@pytest.fixture(scope = "module")
def input_ids(model_registry):
    from exllamav3 import Config, Tokenizer
    model_dir = model_registry.get("glm52-stub-hf").path
    return Tokenizer.from_config(Config.from_directory(model_dir)).encode(LIGHTHOUSE_TEXT)[:, :SEQ_LEN]


def _hf(model_dir, ids, device, attn_impl, index_topk):
    from transformers import AutoModelForCausalLM
    kwargs = {} if index_topk is None else {"index_topk": index_topk}
    return hf_reference(AutoModelForCausalLM, model_dir, ids, device, variant = f"topk{index_topk}",
                        dtype = torch.bfloat16, attn_impl = attn_impl, experts_impl = "eager", **kwargs)


@pytest.mark.parametrize("regime", list(INDEX_TOPK))
def test_hidden_states_and_logits(regime, model_dir, input_ids, device):
    from exllamav3.modules import MLAttention, TransformerBlock
    index_topk = INDEX_TOPK[regime]
    ref = _hf(model_dir, input_ids, device, "eager", index_topk)
    floor = _hf(model_dir, input_ids, device, "sdpa", index_topk)

    def configure(model):
        if index_topk is not None:
            for m in model.modules:
                if isinstance(m, TransformerBlock) and isinstance(m.attn, MLAttention):
                    m.attn.index_topk = index_topk

    states, logits, _ = exl3_stream(model_dir, input_ids, device, hf_aligned_capture(embed_after = 0), configure)
    gate = Gate(f"GLM-5.2 stub ({regime}) vs HF bf16 eager, floor HF sdpa")
    hidden_state_gate(gate, states, ref["hs"], floor["hs"])
    logits_floor_gate(gate, logits, ref["logits"], floor["logits"])
    gate.assert_passes()
