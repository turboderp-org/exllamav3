"""
Config.max_position_embeddings must come from the top-level key when present, from text_config when a
multimodal config keeps the text model's limit there (PR #328 reported 8192 for Qwen3.8-27B whose real limit
is 262144), and from the architecture default otherwise; an explicit top-level key wins over the nested one.

Synthetic config.json files (small dimensions, no weights) written to tmp_path and read through
Config.from_directory, so the real architecture config classes do the parsing.
"""

import copy
import json

import pytest

from exllamav3 import Config

pytestmark = pytest.mark.nogpu

ARCH_DEFAULT = 8192

# Text-only architecture: the limit is a top-level key
QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"],
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 256,
    "intermediate_size": 512,
    "max_position_embeddings": 40960,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
    "vocab_size": 1024,
}

# Multimodal architecture: the text model's limit is under text_config, with a vision tower beside it
QWEN3_5_VL = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "full_attention_interval": 2,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 256,
        "intermediate_size": 512,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_value_head_dim": 32,
        "max_position_embeddings": 262144,
        "num_attention_heads": 4,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 1024,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    },
    "tie_word_embeddings": False,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "model_type": "qwen3_5",
        "depth": 2,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 128,
        "in_channels": 3,
        "intermediate_size": 256,
        "num_heads": 2,
        "num_position_embeddings": 2304,
        "out_hidden_size": 256,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_start_token_id": 1000,
    "vision_end_token_id": 1001,
}

QWEN3_5_VL_PREPROCESSOR = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


def max_pos(tmp_path, config, edit = None):
    """Config.from_directory(...).max_position_embeddings for a copy of `config` after edit(copy)"""
    config = copy.deepcopy(config)
    if edit:
        edit(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    if "vision_config" in config:
        (tmp_path / "preprocessor_config.json").write_text(json.dumps(QWEN3_5_VL_PREPROCESSOR))
    return Config.from_directory(str(tmp_path)).max_position_embeddings


def test_top_level_key(tmp_path):
    assert max_pos(tmp_path, QWEN3) == 40960


def test_top_level_absent_uses_architecture_default(tmp_path):
    def drop(c): del c["max_position_embeddings"]
    assert max_pos(tmp_path, QWEN3, drop) == ARCH_DEFAULT


def test_nested_text_config(tmp_path):
    assert max_pos(tmp_path, QWEN3_5_VL) == 262144


def test_explicit_top_level_key_wins_over_nested(tmp_path):
    def top_and_nested(c): c["max_position_embeddings"] = 4096
    assert max_pos(tmp_path, QWEN3_5_VL, top_and_nested) == 4096


def test_neither_key_uses_architecture_default(tmp_path):
    def neither(c): del c["text_config"]["max_position_embeddings"]
    assert max_pos(tmp_path, QWEN3_5_VL, neither) == ARCH_DEFAULT
