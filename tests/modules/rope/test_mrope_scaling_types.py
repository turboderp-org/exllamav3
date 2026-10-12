"""
PR #332: a multimodal model's mrope_section must be picked up whatever rope_type it uses. Qwen3.5's model card
recommends adding YaRN (rope_type "yarn", factor, original_max_position_embeddings) to rope_parameters for long
context while mrope_section stays in place; RoPE used to read mrope_section only in the default branch, so the first
image request failed in get_mrope_freqs with "'NoneType' object is not subscriptable".

Builds RoPE through Config.from_directory from a synthetic Qwen3.5 config (the checkpoint's text_config
rope_parameters layout, no weights) edited to each scaling type, and runs the mrope frequency path on the CPU.
"""

import copy
import json
import os

import pytest
import torch

from exllamav3 import Config
from exllamav3.util.rope import RoPE

pytestmark = pytest.mark.nogpu

SECTION = [11, 11, 10]

QWEN3_5_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "image_token_id": 1000,
    "video_token_id": 1001,
    "vision_start_token_id": 1002,
    "vision_end_token_id": 1003,
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text",
        "head_dim": 256,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 8,
        "linear_num_value_heads": 8,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "eos_token_id": 1,
        "hidden_act": "silu",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": SECTION,
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "vision_config": {
        "model_type": "qwen3_5",
        "deepstack_visual_indexes": [],
        "depth": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 256,
        "in_channels": 3,
        "intermediate_size": 512,
        "num_heads": 4,
        "num_position_embeddings": 2304,
        "out_hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
}

PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}

VARIANTS = {
    "default": {},
    "yarn": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 65536},
    "linear": {"rope_type": "linear", "factor": 2.0},
    "proportional": {"rope_type": "proportional", "factor": 2.0},
}


def rope_for(directory, extra) -> RoPE:
    c = copy.deepcopy(QWEN3_5_CONFIG)
    c["text_config"]["rope_parameters"].update(extra)
    os.makedirs(directory, exist_ok = True)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(c, f)
    with open(os.path.join(directory, "preprocessor_config.json"), "w") as f:
        json.dump(PREPROCESSOR_CONFIG, f)
    return RoPE("cpu", Config.from_directory(str(directory)).rope_settings)


@pytest.fixture(scope = "module")
def ids():
    return torch.randint(0, 1000, (1, 40), generator = torch.Generator().manual_seed(0))


@pytest.fixture(scope = "module")
def default_freqs(tmp_path_factory, ids):
    return rope_for(tmp_path_factory.mktemp("default"), {}).get_mrope_freqs(ids, [], 40)[0]


@pytest.mark.parametrize("name", list(VARIANTS))
def test_mrope_section_kept(tmp_path, ids, default_freqs, name):
    r = rope_for(tmp_path, VARIANTS[name])
    assert r.mrope_section == SECTION
    assert r.mrope_interleaved is True
    freqs, nxt = r.get_mrope_freqs(ids, [], 40)       # text only: the sectioned path still runs
    assert freqs.shape == (1, 40, r.inv_freq.numel()) and nxt == 40
    if name == "yarn":
        assert r.attn_factor != 1.0 and not torch.equal(freqs, default_freqs)   # scaling still applied
