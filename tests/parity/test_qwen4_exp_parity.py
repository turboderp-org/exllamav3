"""
Qwen3.8-Flash-Next (Qwen4ExpForConditionalGeneration, text component) end-to-end parity against HF transformers on
the unquantized 4-layer stub (registry role qwen4-exp-stub-hf: 3x GDN with the PLE layer before layer 1, then one QSA
full-attention layer).

HF's hidden_states are [pre-layer-0 stack, layer 0..n-2 stream stacks (flat hc * hidden), the final mixer's
hidden-wide output]; exllamav3 captures every block's stream stack and the final mixer. Reference: HF fp32 sdpa;
judgement against the HF bf16 run as the dtype noise floor (testlib.parity.FLOOR_K), for every stage and for the
logits. With an indexer budget below the sequence length the run exercises the real sparse selection end to end;
with the config's budget the selection is non-binding at test length and the mask must be causal-equivalent. The
random input carries EOS boundaries for the n-gram segmentation. exllamav3 streams one module at a time; the HF
references are disk-cached ($EXL3_HFREF_CACHE).
"""

import pytest
import torch

from testlib.parity import Gate, exl3_stream, hf_reference, hidden_state_gate, logits_floor_gate

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.model("qwen4-exp-stub-hf")]

SEQ_LEN = 384
EOS = 248044
INDEXER_BUDGET = {"dense": None, "sparse": 64}


@pytest.fixture(scope = "module")
def input_ids():
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, 200000, (1, SEQ_LEN), generator = g)
    ids[0, SEQ_LEN // 3] = EOS
    ids[0, 2 * SEQ_LEN // 3] = EOS
    return ids


def _hf(model_dir, ids, device, dtype, budget):
    from transformers.models.qwen4_exp import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
    kwargs = {}
    if budget is not None:
        cfg = Qwen4ExpConfig.from_pretrained(model_dir)
        cfg.text_config.indexer_budget = budget
        kwargs["config"] = cfg
    return hf_reference(Qwen4ExpForConditionalGeneration, model_dir, ids, device, variant = f"budget{budget}",
                        dtype = dtype, attn_impl = "sdpa", experts_impl = "eager", **kwargs)


def _capture(model, idx, module, state):
    from exllamav3.modules import GatedResidual, TransformerBlock
    if isinstance(module, TransformerBlock):
        return state.flatten(-2)
    if isinstance(module, GatedResidual) and not module.use_combine:
        return state
    return None


@pytest.mark.parametrize("regime", list(INDEXER_BUDGET))
def test_hidden_states_and_logits(regime, model_dir, input_ids, device):
    from exllamav3.modules import Attention, TransformerBlock
    budget = INDEXER_BUDGET[regime]
    ref = _hf(model_dir, input_ids, device, torch.float32, budget)
    floor = _hf(model_dir, input_ids, device, torch.bfloat16, budget)

    def configure(model):
        if budget is not None:
            for m in model.modules:
                if isinstance(m, TransformerBlock) and isinstance(m.attn, Attention) and m.attn.qsa_indexer is not None:
                    m.attn.qsa_indexer.token_budget = budget
                    m.attn.qsa_indexer.block_topk = budget // m.attn.qsa_indexer.compress_ratio

    states, logits, _ = exl3_stream(model_dir, input_ids, device, _capture, configure)
    # [blocks 0..n-1, mixer] -> HF's [pre-layer-0, layers 0..n-2, mixer]; the last block's stack has no HF state
    got = [None] + states[:-2] + states[-1:]
    gate = Gate(f"Qwen3.8 stub ({regime}) vs HF fp32 sdpa, floor HF bf16")
    gate.true("block count", len(states) == len(ref["hs"]), f"{len(states) - 1} blocks + mixer vs "
                                                             f"{len(ref['hs'])} HF states")
    names = ["pre-layer-0"] + [f"layer {i}" for i in range(len(got) - 2)] + ["post-mixer"]
    hidden_state_gate(gate, got, ref["hs"], floor["hs"], names = names)
    logits_floor_gate(gate, logits, ref["logits"], floor["logits"])
    gate.assert_passes()
