"""
Mistral-Small-4 (Mistral3ForConditionalGeneration with text_config.model_type "mistral4") full-model parity against
HF transformers.

Tiny random-weight checkpoint in the real release's tensor namespace (old-style language_model.* keys, stacked 3D
expert tensors in HF DeepseekV3Experts orientation): MLA with asymmetric qk_nope != v_head_dim like the real model,
the softmax top-k router with a shared expert, fused gate_up/down expert slicing, yarn rope with interleaved (GPT-J)
pairs and, with original_max_position_embeddings shrunk to 64, the Llama-4 position scale on the full query at
positions past the boundary (which a short test on the real model never reaches). Reference: HF fp32 eager;
exllamav3 cache-less full-sequence logits, gated separately before and past the boundary.

One deliberate departure from HF: transformers >= 5.17 folds DeepSeek's YaRN mscale_all_dim**2 into this model's
softmax scale (yarn_apply_mscale). Mistral's own params.json for the release sets yarn apply_scale = false, and the
release's perplexity agrees, so exllamav3 keeps the plain scale and the HF reference is built with the fold
disabled (hf_plain_softmax_scale). Everything else is compared against transformers as is. The mutation test disables
the l4 scale in exllamav3 and requires the same gate to fail.

Real checkpoint (slow): the first REAL_LAYERS layers of the release (registry role mistral4-hf, truncated-config
symlink farm), HF fp32 eager with the fp8 weights dequantized, vs exllamav3 on the same weights.
"""

import contextlib
import json
import os

import pytest
import torch
from safetensors.torch import save_file

from testlib.parity import (Gate, RUMEN_TEXT, exl3_logits, free_cuda, hf_forward, hf_tokenizer_ids, load_hf,
                            logit_gate, logit_stats, mutate, token_ids, truncated_checkpoint)

pytestmark = pytest.mark.hf

TINY = dict(
    hidden_size = 256,
    num_attention_heads = 8,
    q_lora_rank = 96,
    kv_lora_rank = 64,
    qk_nope_head_dim = 32,
    qk_rope_head_dim = 32,
    v_head_dim = 64,
    num_hidden_layers = 4,
    moe_intermediate_size = 64,
    intermediate_size = 128,
    n_routed_experts = 8,
    num_experts_per_tok = 2,
    n_shared_experts = 1,
    first_k_dense_replace = 0,
    rms_norm_eps = 1e-6,
    vocab_size = 512,
    original_max = 64,
)

SEED = 7
SEQ_LEN = 192           # crosses original_max twice
REAL_SEQ_LEN = 512
REAL_LAYERS = 2


def config_json() -> dict:
    c = TINY
    return {
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "image_token_index": 10,
        "multimodal_projector_bias": False,
        "projector_hidden_act": "gelu",
        "spatial_merge_size": 2,
        "vision_feature_layer": -1,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "mistral4",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "head_dim": c["qk_nope_head_dim"] + c["qk_rope_head_dim"],
            "hidden_act": "silu",
            "hidden_size": c["hidden_size"],
            "intermediate_size": c["intermediate_size"],
            "kv_lora_rank": c["kv_lora_rank"],
            "max_position_embeddings": c["original_max"] * 4,
            "moe_intermediate_size": c["moe_intermediate_size"],
            "n_group": 1,
            "n_routed_experts": c["n_routed_experts"],
            "n_shared_experts": c["n_shared_experts"],
            "norm_topk_prob": True,
            "num_attention_heads": c["num_attention_heads"],
            "num_experts_per_tok": c["num_experts_per_tok"],
            "num_hidden_layers": c["num_hidden_layers"],
            "num_key_value_heads": c["num_attention_heads"],
            "first_k_dense_replace": c["first_k_dense_replace"],
            "q_lora_rank": c["q_lora_rank"],
            "qk_head_dim": c["qk_nope_head_dim"] + c["qk_rope_head_dim"],
            "qk_nope_head_dim": c["qk_nope_head_dim"],
            "qk_rope_head_dim": c["qk_rope_head_dim"],
            "rms_norm_eps": c["rms_norm_eps"],
            "rope_interleave": True,
            "rope_parameters": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 4.0,
                "llama_4_scaling_beta": 0.1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": c["original_max"],
                "rope_theta": 10000.0,
                "rope_type": "yarn",
                "type": "yarn",
            },
            "routed_scaling_factor": 1.0,
            "sliding_window": None,
            "tie_word_embeddings": False,
            "topk_group": 1,
            "use_cache": True,
            "v_head_dim": c["v_head_dim"],
            "vocab_size": c["vocab_size"],
        },
        "vision_config": {
            "model_type": "pixtral",
            "attention_dropout": 0.0,
            "head_dim": 32,
            "hidden_act": "silu",
            "hidden_size": 64,
            "image_size": 154,
            "initializer_range": 0.02,
            "intermediate_size": 128,
            "num_attention_heads": 2,
            "num_channels": 3,
            "num_hidden_layers": 2,
            "patch_size": 14,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    }


def processor_json() -> dict:
    return {
        "image_processor": {
            "image_processor_type": "PixtralImageProcessorFast",
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "resample": 3,
            "rescale_factor": 0.00392156862745098,
            "size": {"longest_edge": 154},
            "patch_size": 14,
        }
    }


def make_checkpoint(out_dir: str, seed: int) -> str:
    torch.manual_seed(seed)
    c = TINY
    H = c["hidden_size"]
    nh = c["num_attention_heads"]
    qk = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
    inter = c["moe_intermediate_size"]
    t = {}

    def w(name, *shape, scale = 0.02):
        t[name] = (torch.randn(*shape, dtype = torch.float) * scale).to(torch.bfloat16)

    def normw(name, dim):
        t[name] = (1.0 + 0.1 * torch.randn(dim)).to(torch.bfloat16)

    lm = "language_model.model"
    w("language_model.lm_head.weight", c["vocab_size"], H)
    w(f"{lm}.embed_tokens.weight", c["vocab_size"], H)
    normw(f"{lm}.norm.weight", H)

    for i in range(c["num_hidden_layers"]):
        L = f"{lm}.layers.{i}"
        normw(f"{L}.input_layernorm.weight", H)
        normw(f"{L}.post_attention_layernorm.weight", H)
        w(f"{L}.self_attn.q_a_proj.weight", c["q_lora_rank"], H)
        normw(f"{L}.self_attn.q_a_layernorm.weight", c["q_lora_rank"])
        w(f"{L}.self_attn.q_b_proj.weight", nh * qk, c["q_lora_rank"])
        w(f"{L}.self_attn.kv_a_proj_with_mqa.weight", c["kv_lora_rank"] + c["qk_rope_head_dim"], H)
        normw(f"{L}.self_attn.kv_a_layernorm.weight", c["kv_lora_rank"])
        w(f"{L}.self_attn.kv_b_proj.weight", nh * (c["qk_nope_head_dim"] + c["v_head_dim"]), c["kv_lora_rank"])
        w(f"{L}.self_attn.o_proj.weight", H, nh * c["v_head_dim"])
        w(f"{L}.mlp.gate.weight", c["n_routed_experts"], H, scale = 0.05)
        # Stacked experts, HF DeepseekV3Experts orientation: (E, out, in), gate rows first
        w(f"{L}.mlp.experts.gate_up_proj", c["n_routed_experts"], 2 * inter, H, scale = 0.05)
        w(f"{L}.mlp.experts.down_proj", c["n_routed_experts"], H, inter, scale = 0.05)
        w(f"{L}.mlp.shared_experts.gate_proj.weight", inter * c["n_shared_experts"], H, scale = 0.05)
        w(f"{L}.mlp.shared_experts.up_proj.weight", inter * c["n_shared_experts"], H, scale = 0.05)
        w(f"{L}.mlp.shared_experts.down_proj.weight", H, inter * c["n_shared_experts"], scale = 0.05)

    os.makedirs(out_dir, exist_ok = True)
    save_file(t, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as fp:
        json.dump(config_json(), fp, indent = 2)
    with open(os.path.join(out_dir, "processor_config.json"), "w") as fp:
        json.dump(processor_json(), fp, indent = 2)
    return out_dir


# Median KL per segment: a single MoE routing flip on a near-tie dominates the mean (one position at ~1e-2), while
# a wrong or missing l4 query scale shifts every position past original_max (median ~2e-6 vs ~1e-7 when exact)
MAX_MEDIAN_KL = 5e-7


@contextlib.contextmanager
def hf_plain_softmax_scale():
    """Context in which transformers' Mistral4 attention keeps the plain qk_head_dim**-0.5 softmax scale, as
    Mistral's reference does. The attention modules compute their scale at construction, so the HF model must be
    built inside"""
    from transformers.models.mistral4 import modeling_mistral4
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(modeling_mistral4, "yarn_apply_mscale", lambda rope_parameters, scaling: scaling)
        yield


def boundary_gate(got: torch.Tensor, ref: torch.Tensor, tag: str) -> Gate:
    """KL mean < 5e-4, median KL < MAX_MEDIAN_KL and argmax > 0.98, separately before original_max (scale 1) and past
    it (l4 scale active)"""
    gate = Gate(tag)
    om = TINY["original_max"]
    for name, lo, hi in (("pos < original_max", 0, om), ("pos >= original_max", om, got.shape[0])):
        stats = logit_stats(got[lo:hi], ref[lo:hi])
        logit_gate(gate, stats, name, kl_mean = 5e-4, argmax = 0.98)
        gate.lt(f"{name} KL median", stats.kl.median().item(), MAX_MEDIAN_KL)
    return gate


@pytest.fixture(scope = "module")
def tiny(tmp_path_factory, device):
    from transformers import Mistral3ForConditionalGeneration
    model_dir = make_checkpoint(str(tmp_path_factory.mktemp("mistral4_tiny")), seed = SEED)
    ids = token_ids(SEQ_LEN, TINY["vocab_size"], seed = SEED + 1)
    with hf_plain_softmax_scale():
        model = load_hf(Mistral3ForConditionalGeneration, model_dir, device = device, dtype = torch.float32)
    ref = hf_forward(model, ids, hidden_states = False)["logits"]
    del model
    free_cuda()
    return model_dir, ids, ref


def _disable_l4_scale(model):
    mutate((getattr(m, "attn", None) for m in model.modules), "l4_beta", 0.0)


def test_tiny_logits(tiny, device):
    model_dir, ids, ref = tiny
    boundary_gate(exl3_logits(model_dir, ids, device), ref, "Mistral-Small-4 tiny vs HF fp32 eager").assert_passes()


def test_tiny_mutation_l4_scale(tiny, device):
    model_dir, ids, ref = tiny
    got = exl3_logits(model_dir, ids, device, configure = _disable_l4_scale)
    boundary_gate(got, ref, "Mistral-Small-4 tiny without the l4 query scale").assert_fails()


def _truncate(cfg):
    cfg["text_config"]["num_hidden_layers"] = REAL_LAYERS


@pytest.mark.slow
@pytest.mark.model("mistral4-hf")
def test_real_checkpoint_logits(model_dir, device, tmp_path):
    from transformers import Mistral3ForConditionalGeneration
    stub = truncated_checkpoint(model_dir, str(tmp_path / "mistral4_trunc"), _truncate)
    ids = hf_tokenizer_ids(stub, RUMEN_TEXT, REAL_SEQ_LEN)
    with hf_plain_softmax_scale():
        model = load_hf(Mistral3ForConditionalGeneration, stub, device = device, dtype = torch.float32)
    ref = hf_forward(model, ids, hidden_states = False)["logits"]
    del model
    free_cuda()
    got = exl3_logits(stub, ids, device)
    gate = Gate("Mistral-Small-4 checkpoint vs HF fp32 eager")
    logit_gate(gate, logit_stats(got, ref), kl_mean = 5e-4, argmax = 0.98)
    gate.assert_passes()
