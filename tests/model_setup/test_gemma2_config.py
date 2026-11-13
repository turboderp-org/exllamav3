"""
Gemma2 sliding window from config.json. transformers takes the window from "sliding_window" (Gemma2Config, default
4096) and applies it to the even layers. gemma-2-9b and gemma-2-27b ship that key together with a legacy
"sliding_window_size" of the same value, while gemma-2-2b ships "sliding_window" alone, so reading only the legacy key
leaves every layer of the 2b models on full attention.

Synthetic config.json files (small dimensions, no weights) written to tmp_path and read through Config.from_directory
and Model.from_config, so the real architecture classes build the attention layers.
"""

import copy
import json

import pytest

from exllamav3 import Config, Model

pytestmark = pytest.mark.nogpu

# Key set of gemma-2-2b(-it), dimensions shrunk
GEMMA2 = {
    "architectures": ["Gemma2ForCausalLM"],
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "head_dim": 64,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 256,
    "intermediate_size": 512,
    "max_position_embeddings": 8192,
    "model_type": "gemma2",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "query_pre_attn_scalar": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "vocab_size": 1024,
}


def attn_windows(tmp_path, edit = None) -> list[int]:
    """Window of each layer's attention, as built from a copy of GEMMA2 after edit(copy)"""
    config = copy.deepcopy(GEMMA2)
    if edit:
        edit(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = Model.from_config(Config.from_directory(str(tmp_path)))
    return [m.attn.sliding_window for m in model.modules[model.first_block_idx:] if hasattr(m, "attn")]


def hf_windows(window: int, num_layers: int = 4) -> list[int]:
    """Sliding attention on the even layers; the HF mask counts the query in the window"""
    return [window - 1 if i % 2 == 0 else -1 for i in range(num_layers)]


def test_sliding_window_key(tmp_path):
    # gemma-2-2b
    assert attn_windows(tmp_path) == hf_windows(4096)


def test_legacy_sliding_window_size_key(tmp_path):
    # gemma-2-9b, gemma-2-27b
    def legacy(c): c["sliding_window_size"] = 4096
    assert attn_windows(tmp_path, legacy) == hf_windows(4096)


def test_custom_sliding_window(tmp_path):
    def small(c): c["sliding_window"] = 512
    assert attn_windows(tmp_path, small) == hf_windows(512)


def test_no_window_key_uses_hf_default(tmp_path):
    def drop(c): del c["sliding_window"]
    assert attn_windows(tmp_path, drop) == hf_windows(4096)
