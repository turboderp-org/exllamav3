"""
MemT-1.0-Nano config parsing through the real architecture registry
(``MemTQwen3_5MoeForCausalLM``): the release wraps the backbone's HF config
under ``backbone_configs`` and carries the memory geometry in a top-level
``memt`` object. A synthetic small-dimension config.json (no weights) checks
that the Qwen3.5-MoE backbone fields are read through the wrapper, that the
memory layout lands on the full-attention layers, and that a config flipping
a pinned algorithm switch is refused.

Needs the package importable (the CUDA extension builds on import), hence the
model_setup location; no GPU is used.
"""

import copy
import json

import pytest

from exllamav3 import Config

pytestmark = pytest.mark.nogpu

MEMT = {
    "memory_update_mode": "token",
    "gdn_version": "stable_gdn2",
    "memory_use_short_conv": False,
    "memory_head_mode": "mha",
    "memory_read_scale": "k_inv_sqrt",
    "linear_attention_state_policy": "reset_per_chunk",
    "tail_policy": "commit",
    "output_gate_type": "silu",
    "output_gate_bias": False,
    "output_gate_softcap_enabled": True,
    "stable_gdn2_sqrt_erase_gate": True,
    "backbone_gdn_override": False,
    "num_memory_heads": 4,
    "num_kv_heads": 4,
    "chunk_tokens": 2048,
    "output_gate_softcap_limit": 2.0,
}

MEMT_NANO_LIKE = {
    "architectures": ["MemTQwen3_5MoeForCausalLM"],
    "model_type": "memt_qwen3_5_moe",
    "memt": MEMT,
    "backbone_configs": {
        "model_type": "qwen3_5_moe",
        "tie_word_embeddings": False,
        "text_config": {
            "full_attention_interval": 4,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 256,
            "intermediate_size": 512,
            "moe_intermediate_size": 128,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "shared_expert_intermediate_size": 128,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_value_head_dim": 32,
            "max_position_embeddings": 262144,
            "num_attention_heads": 4,
            "num_hidden_layers": 8,
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
    },
}


def load_config(tmp_path, config, edit = None) -> Config:
    config = copy.deepcopy(config)
    if edit:
        edit(config)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return Config.from_directory(str(tmp_path))


def test_backbone_read_through_wrapper(tmp_path):
    cfg = load_config(tmp_path, MEMT_NANO_LIKE)
    assert cfg.architecture == "MemTQwen3_5MoeForCausalLM"
    assert cfg.hidden_size == 256 and cfg.num_q_heads == 4 and cfg.head_dim == 64
    assert cfg.num_hidden_layers == 8 and cfg.num_experts == 8
    assert cfg.max_position_embeddings == 262144
    assert cfg.layer_types == ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2


def test_memory_layout_on_full_attention_layers(tmp_path):
    cfg = load_config(tmp_path, MEMT_NANO_LIKE)
    lay = cfg.memt
    assert lay.layers == (3, 7)
    assert lay.spec.num_heads == 4 and lay.spec.head_dim == 64 and lay.spec.hidden_size == 256
    assert lay.spec.output_gate_softcap == 2.0 and lay.chunk_tokens == 2048
    assert lay.spec.state_shape == (4, 64, 64)


def test_model_attaches_memory_blocks(tmp_path):
    from exllamav3.modules import TransformerBlock
    cfg = load_config(tmp_path, MEMT_NANO_LIKE)
    model = cfg.model_classes["text"](cfg)
    blocks = {m.layer_idx: m for m in model.modules if isinstance(m, TransformerBlock)}
    assert set(model.memt_modules) == {3, 7}
    for i, blk in blocks.items():
        mem = getattr(blk.attn, "memt", None)
        if i in (3, 7):
            assert mem is not None and mem.key == f"model.language_model.layers.{i}.self_attn.memt"
            assert mem in list(blk.attn)          # converts / loads / unloads with the layer
            assert mem.weights_numel() == (5 * 4 * 64 + 2 * 64) * 256 + 2 * 4 * 64 * 64 + 4 + 4 * 64 + 64
        else:
            assert mem is None
    assert model.caps.get("memt") is True
    z = model.memt_zero_state()
    assert set(z) == {3, 7} and tuple(z[3].shape) == (1, 4, 64, 64)


def test_pinned_switch_refused(tmp_path):
    def flip(c):
        c["memt"]["gdn_version"] = "gdn2"
    with pytest.raises(ValueError):
        load_config(tmp_path, MEMT_NANO_LIKE, flip)


def test_head_geometry_refused(tmp_path):
    def bad(c):
        c["memt"]["num_kv_heads"] = 2
    with pytest.raises(ValueError):
        load_config(tmp_path, MEMT_NANO_LIKE, bad)
