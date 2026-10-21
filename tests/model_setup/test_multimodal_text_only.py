"""
Text-only multimodal checkpoints:

A fine-tune of a multimodal Gemma keeps the wrapper architecture and the vision_config for provenance but may ship
no processor file, because the derivative never trains or serves images. Loading used to fail while parsing the
config, before any weight was read:

- Gemma 3 opens preprocessor_config.json unconditionally
- Gemma 4 opens processor_config.json unconditionally, and asks for vision_config->model_patch_size first

These tests pin the behaviour: without a processor the config parses and carries no vision preprocessor, the vision
model refuses before allocating, and with a processor present nothing changes. A processor that exists but cannot be
read is still an error, not "no processor".

Synthetic config.json files (small dimensions, no weights) written to tmp_path and read through
Config.from_directory, so the real architecture config classes do the parsing.
"""

import copy
import json
import struct

import pytest

from exllamav3 import Config
from exllamav3.architecture.gemma3 import Gemma3VisionModel
from exllamav3.architecture.gemma4 import Gemma4UnifiedVisionModel

pytestmark = pytest.mark.nogpu

GEMMA3 = {
    "architectures": ["Gemma3ForConditionalGeneration"],
    "model_type": "gemma3",
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
    "image_token_index": 262144,
    "mm_tokens_per_image": 4,
    "text_config": {
        "head_dim": 32,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 131072,
        "model_type": "gemma3_text",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "query_pre_attn_scalar": 32,
        "rms_norm_eps": 1e-6,
        "rope_local_base_freq": 10000.0,
        "rope_theta": 1000000.0,
        "sliding_window": 8,
        "sliding_window_pattern": 2,
        "tie_word_embeddings": True,
        "final_logit_softcapping": 30.0,
        "vocab_size": 1024,
    },
    "vision_config": {
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "image_size": 64,
        "intermediate_size": 128,
        "layer_norm_eps": 1e-6,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 2,
        "num_channels": 3,
        "num_hidden_layers": 2,
        "patch_size": 14,
    },
}

GEMMA3_PREPROCESSOR = {
    "image_processor_type": "Gemma3ImageProcessor",
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 64, "width": 64},
}

GEMMA4 = {
    "architectures": ["Gemma4UnifiedForConditionalGeneration"],
    "model_type": "gemma4_unified",
    "boi_token_id": 101,
    "eoi_token_id": 102,
    "image_token_id": 100,
    "tie_word_embeddings": False,
    "text_config": {
        "attention_bias": False,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 32,
        "head_dim": 32,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 64,
        "hidden_size_per_layer_input": 0,
        "intermediate_size": 128,
        "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 4096,
        "model_type": "gemma4_unified_text",
        "num_attention_heads": 2,
        "num_global_key_value_heads": 1,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 0,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
        "sliding_window": 8,
        "tie_word_embeddings": True,
        "vocab_size": 1024,
    },
    "vision_config": {
        "hidden_size": 64,
        "intermediate_size": 128,
        "mm_embed_dim": 64,
        "mm_posemb_size": 64,
        "model_patch_size": 16,
        "model_type": "gemma4_unified_vision",
        "num_soft_tokens": 4,
        "output_proj_dims": 64,
        "patch_size": 16,
        "pooling_kernel_size": 1,
        "rms_norm_eps": 1e-6,
    },
}

GEMMA4_PROCESSOR = {
    "image_processor": {
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "max_soft_tokens": 4,
        "mm_posemb_size": 64,
        "patch_size": 16,
        "pooling_kernel_size": 1,
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"height": 64, "width": 64},
    },
}


def empty_safetensors(path):
    """Smallest file SafetensorsCollection will accept: an empty header, no tensors."""
    header = b"{}"
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def load_config(tmp_path, config, processor = None, processor_body = None) -> Config:
    """Config.from_directory over a copy of `config`, with an optional processor file beside it"""
    config = copy.deepcopy(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    empty_safetensors(tmp_path / "model.safetensors")
    if processor is not None:
        body = json.dumps(GEMMA3_PREPROCESSOR if processor_body is None else processor_body)
        (tmp_path / processor).write_text(body)
    return Config.from_directory(str(tmp_path))


def test_gemma3_text_only_parses_without_processor(tmp_path):
    config = load_config(tmp_path, GEMMA3)
    assert config.vision_pp is None
    assert config.vision is None


def test_gemma3_with_processor_parses_as_before(tmp_path):
    config = load_config(tmp_path, GEMMA3, "preprocessor_config.json")
    assert config.vision_pp is not None
    assert config.vision.patch_size == 14


def test_gemma3_vision_refuses_text_only_config(tmp_path):
    config = load_config(tmp_path, GEMMA3)
    with pytest.raises(FileNotFoundError):
        Gemma3VisionModel(config)


def test_gemma3_malformed_processor_still_fails(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text("{ not json")
    (tmp_path / "config.json").write_text(json.dumps(GEMMA3))
    empty_safetensors(tmp_path / "model.safetensors")
    with pytest.raises(json.JSONDecodeError):
        Config.from_directory(str(tmp_path))


def test_gemma4_text_only_parses_without_processor(tmp_path):
    config = load_config(tmp_path, GEMMA4)
    assert config.vision_pp is None
    assert config.vision is None


def test_gemma4_with_processor_parses_as_before(tmp_path):
    config = load_config(tmp_path, GEMMA4, "processor_config.json", GEMMA4_PROCESSOR)
    assert config.vision_pp is not None
    assert config.vision.patch_size == 16


def test_gemma4_vision_refuses_text_only_config(tmp_path):
    config = load_config(tmp_path, GEMMA4)
    with pytest.raises(FileNotFoundError):
        Gemma4UnifiedVisionModel(config)


def test_gemma4_malformed_processor_still_fails(tmp_path):
    (tmp_path / "processor_config.json").write_text("{ not json")
    (tmp_path / "config.json").write_text(json.dumps(GEMMA4))
    empty_safetensors(tmp_path / "model.safetensors")
    with pytest.raises(json.JSONDecodeError):
        Config.from_directory(str(tmp_path))
