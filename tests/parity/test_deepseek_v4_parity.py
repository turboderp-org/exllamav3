"""
DeepSeek-V4 full-model parity against HF transformers (DeepseekV4ForCausalLM).

Tiny random-weight checkpoint in DeepSeek's native tensor namespace (testlib.tiny_models.make_dsv4_checkpoint: the
naming of the real V4-Flash release, so HF exercises its conversion mapping and exllamav3 its native keys against one
directory), all three attention layer types (sliding / CSA / HCA), hash + sqrtsoftplus MoE routing, mHC, sinks,
partial rope with the yarn compress table and the output de-rotation. Reference: HF fp32 eager; exllamav3 cache-less
full-sequence logits. The random tiny model is chaos-prone (near-tie flips at the fp16-attention rounding scale), so
the decisive argmax criterion is agreement where HF's top-1 margin is unambiguous, with KL as the primary metric.

Real checkpoint (slow): the first REAL_LAYERS layers of the V4-Flash release (registry role dsv4-flash-hf, as a
truncated-config symlink farm), HF fp32 eager with the fp8/fp4 weights dequantized (transformers' bf16 forward feeds
the fp32 hyper-connection output into bf16 projections) vs exllamav3 on the same weights. In fp32 a V4-Flash layer is
24 GiB of fused experts plus a 16 GiB conversion transient, so only the two sliding-window layers (hash routing,
mHC, sinks) fit on one device; the compressed layer types are covered by the tiny checkpoint.
"""

import pytest
import torch

from testlib.parity import (Gate, RUMEN_TEXT, exl3_logits, free_cuda, hf_forward, hf_tokenizer_ids, load_hf,
                            logit_gate, logit_stats, token_ids, truncated_checkpoint)
from testlib.tiny_models import DSV4_TINY, make_dsv4_checkpoint

pytestmark = pytest.mark.hf

SEED = 7
SEQ_LEN = 315           # deliberately not aligned to the window or the compress rates
REAL_SEQ_LEN = 512
REAL_LAYERS = 2


@pytest.fixture(scope = "module")
def tiny_dir(tmp_path_factory):
    return make_dsv4_checkpoint(str(tmp_path_factory.mktemp("dsv4_tiny")), seed = SEED)


@pytest.fixture(scope = "module")
def tiny_ids():
    return token_ids(SEQ_LEN, DSV4_TINY["vocab_size"], seed = SEED + 1)


@pytest.fixture(scope = "module")
def tiny_logits(tiny_dir, tiny_ids, device):
    from transformers import DeepseekV4ForCausalLM
    model = load_hf(DeepseekV4ForCausalLM, tiny_dir, device = device, dtype = torch.float32)
    ref = hf_forward(model, tiny_ids, hidden_states = False)["logits"]
    del model
    free_cuda()
    return ref, exl3_logits(tiny_dir, tiny_ids, device)


def test_tiny_logits(tiny_logits):
    ref, got = tiny_logits
    gate = Gate("DeepSeek-V4 tiny vs HF fp32 eager")
    logit_gate(gate, logit_stats(got, ref), kl_mean = 5e-4, argmax = 0.98, argmax_conf = 0.999)
    gate.assert_passes()


def _truncate(cfg):
    cfg["num_hidden_layers"] = REAL_LAYERS
    cfg["num_nextn_predict_layers"] = 0
    cfg["compress_ratios"] = cfg["compress_ratios"][:REAL_LAYERS]
    for k in [k for k in cfg if k.startswith("dspark_")]:
        del cfg[k]


@pytest.mark.slow
@pytest.mark.model("dsv4-flash-hf")
def test_real_checkpoint_logits(model_dir, device, tmp_path):
    from transformers import DeepseekV4ForCausalLM
    stub = truncated_checkpoint(model_dir, str(tmp_path / "dsv4_trunc"), _truncate)
    ids = hf_tokenizer_ids(stub, RUMEN_TEXT, REAL_SEQ_LEN)
    model = load_hf(DeepseekV4ForCausalLM, stub, device = device, dtype = torch.float32)
    ref = hf_forward(model, ids, hidden_states = False)["logits"]
    del model
    free_cuda()
    got = exl3_logits(stub, ids, device)
    gate = Gate("DeepSeek-V4 checkpoint vs HF fp32 eager")
    logit_gate(gate, logit_stats(got, ref), kl_mean = 5e-4, argmax = 0.98, argmax_conf = 0.999)
    gate.assert_passes()
