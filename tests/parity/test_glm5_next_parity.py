"""
GLM-5.3-Flash (Glm5NextForConditionalGeneration, text component) per-stage parity against HF transformers on a
real-weight stub (registry role glm53-stub-hf: the first layers of the release, covering dense + full-indexer,
sparse + shared and sparse + full blocks; bf16, or fp8 block-scaled which both sides dequantize).

HF's hidden_states are the mHC stream stacks (b, s, hc_mult, D), so exllamav3 states are captured after the stream
expansion, after every block but the last, and after the final norm. Judgement is against the HF eager-vs-sdpa noise
floor (testlib.parity.FLOOR_K), not against zero, for every stage and for the logits. With index_topk below the
sequence length the run exercises the real sparse selection path and cross-layer top-k sharing; with the config's
index_topk (>= T) the dense path must be selection-equivalent. exllamav3 streams one module at a time; the HF
references are disk-cached ($EXL3_HFREF_CACHE).
"""

import pytest
import torch

from testlib.parity import (Gate, LIGHTHOUSE_TEXT, exl3_stream, hf_aligned_capture, hf_reference,
                            hidden_state_gate, logits_floor_gate, trim_layer_lists, truncated_checkpoint)

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.model("glm53-stub-hf")]

SEQ_LEN = 512
INDEX_TOPK = {"dense": None, "sparse": 64}


@pytest.fixture(scope = "module")
def stub_dir(model_registry, tmp_path_factory):
    """The stub with its per-layer config lists trimmed to its layer count"""
    src = model_registry.get("glm53-stub-hf").path
    return truncated_checkpoint(src, str(tmp_path_factory.mktemp("glm53_stub")),
                                lambda cfg: trim_layer_lists(cfg, "text_config"))


@pytest.fixture(scope = "module")
def input_ids(stub_dir):
    from exllamav3 import Config, Tokenizer
    return Tokenizer.from_config(Config.from_directory(stub_dir)).encode(LIGHTHOUSE_TEXT)[:, :SEQ_LEN]


def _hf(model_dir, ids, device, attn_impl, index_topk):
    from transformers.models.glm5_next import Glm5NextConfig, Glm5NextForConditionalGeneration
    kwargs = {}
    if index_topk is not None:
        # index_topk lives in text_config: override through a preloaded config object
        cfg = Glm5NextConfig.from_pretrained(model_dir)
        cfg.text_config.index_topk = index_topk
        kwargs["config"] = cfg
    return hf_reference(Glm5NextForConditionalGeneration, model_dir, ids, device, variant = f"topk{index_topk}",
                        dtype = torch.bfloat16, attn_impl = attn_impl, experts_impl = "eager", **kwargs)


@pytest.mark.parametrize("regime", list(INDEX_TOPK))
def test_hidden_states_and_logits(regime, stub_dir, input_ids, device):
    from exllamav3.modules import MLAttention, TransformerBlock
    index_topk = INDEX_TOPK[regime]
    ref = _hf(stub_dir, input_ids, device, "eager", index_topk)
    floor = _hf(stub_dir, input_ids, device, "sdpa", index_topk)

    def configure(model):
        if index_topk is not None:
            for m in model.modules:
                if isinstance(m, TransformerBlock) and isinstance(m.attn, MLAttention):
                    m.attn.index_topk = index_topk

    states, logits, _ = exl3_stream(stub_dir, input_ids, device, hf_aligned_capture(embed_after = 1), configure)
    gate = Gate(f"GLM-5.3 stub ({regime}) vs HF bf16 eager, floor HF sdpa")
    hidden_state_gate(gate, states, ref["hs"], floor["hs"])
    logits_floor_gate(gate, logits, ref["logits"], floor["logits"])
    gate.assert_passes()
