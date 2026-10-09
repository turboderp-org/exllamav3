"""Ling assembly and grouped routing tests; synthetic fixtures, not model-quality evidence."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from exllamav3 import Config, Model
from exllamav3.architecture.bailing_moe_v3 import BailingMoeV3Config
from exllamav3.modules import GatedDeltaNet, MLAttention, GatedMLP, BlockSparseMLP, Linear
from exllamav3.modules.block_sparse_mlp_routing import routing_ds3_fp32
from exllamav3.util.rope import RopeStyle


# Reduced dimensions, real Flash scheduling and tensor key conventions. Not pretrained weights.
TINY = {
    "architectures": ["BailingMoeV3ForCausalLM"], "vocab_size": 256,
    "hidden_size": 128, "num_hidden_layers": 6, "num_attention_heads": 2,
    "head_dim": 64, "kv_lora_rank": 64, "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 32, "qk_head_dim": 64, "rotary_dim": 32,
    "v_head_dim": 64, "intermediate_size": 256, "moe_intermediate_size": 128,
    "moe_shared_expert_intermediate_size": 128, "num_shared_experts": 1,
    "num_experts": 8, "num_experts_per_tok": 2, "n_group": 2, "topk_group": 1,
    "layer_group_size": 6, "short_conv_kernel_size": 4,
    "first_k_dense_replace": 2, "num_nextn_predict_layers": 1,
    "rms_norm_eps": 1e-6, "kda_lower_bound": -5.0, "routed_scaling_factor": 2.5,
    "rope_theta": 6000000.0, "max_position_embeddings": 262144,
    "partial_rotary_factor": 0.5, "q_lora_rank": None,
    "expert_swiglu_limit_list": [0, 0, 0, 0, 4, 4],
    "share_expert_swiglu_limit_list": [0, 0, 0, 5, 5, 7],
}


def make_config(path, **overrides):
    data = copy.deepcopy(TINY)
    data.update(overrides)
    path.mkdir(parents = True, exist_ok = True)
    (path / "config.json").write_text(json.dumps(data))
    # Assembly needs a tensor collection, but never loads this sentinel as model weights.
    save_file({"sentinel": torch.zeros(1)}, path / "model.safetensors")
    return Config.from_directory(str(path))


def test_registered_assembly_and_mtp(tmp_path):
    config = make_config(tmp_path)
    assert isinstance(config, BailingMoeV3Config)
    model = Model.from_config(config)
    blocks = model.modules[model.first_block_idx:model.last_kv_module_idx + 1]
    assert [isinstance(b.attn, GatedDeltaNet) for b in blocks] == [True] * 5 + [False]
    assert [isinstance(b.mlp, GatedMLP) for b in blocks] == [True, True, False, False, False, False]
    assert blocks[0].attn.kda_direct
    assert blocks[0].attn.f_proj.key == "model.layers.0.attention.f_proj"
    assert isinstance(blocks[-1].attn, MLAttention)
    assert blocks[-1].attn.g_proj.key == "model.layers.5.attention.g_proj"
    assert blocks[-1].attn.o_proj.key == "model.layers.5.attention.dense"
    assert blocks[-1].mlp.router_type == "ds3_fp32"
    assert blocks[-1].mlp.act_limit == 4
    assert blocks[-1].mlp.shared_experts.act_limit == 7
    assert model.modules[0].key == "model.word_embeddings"
    assert model.modules[-1].out_features_unpadded == TINY["vocab_size"]
    assert config.rope_settings.rope_style == RopeStyle.GPTJ
    assert config.rope_settings.partial_rotary_factor == 1
    assert config.rope_settings.rotary_dim == 32
    assert model.caps["supports_tp"] is False
    mtp = Model.from_config(config, component = "mtp")
    mtp.attach_to(model)
    assert mtp.target_lm_head() is model.modules[-1]
    assert mtp.target_embed() is model.modules[0]
    assert mtp.draft_verifier_params == {"export_state_norm_keys": {"model.norm"}}
    assert mtp.input_layer.pre_fc_norm_embedding.key == "model.layers.6.enorm"
    assert mtp.input_layer.pre_fc_norm_hidden.key == "model.layers.6.hnorm"
    assert mtp.input_layer.fc.qbits_key == "mtp_bits"
    block = mtp.modules[mtp.first_block_idx]
    assert block.attn.layer_idx == 0
    assert block.attn.o_proj.qbits_key == "mtp_bits"
    assert block.mlp.act_limit == block.mlp.shared_experts.act_limit == 0
    assert block.mlp.downs[0].qbits_key == "mtp_bits"
    assert mtp.final_norm.key == "model.layers.6.final_layernorm"


def test_production_schedule(tmp_path):
    expert = [0] * 35 + [4] * 7
    shared = [0] * 34 + [5] * 6 + [7] * 2
    config = make_config(tmp_path, num_hidden_layers = 42,
                         expert_swiglu_limit_list = expert, share_expert_swiglu_limit_list = shared)
    assert [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"] == [5, 11, 17, 23, 29, 35, 41]
    assert config.layer_types.count("linear_attention") == 35
    assert config.expert_swiglu_limits == expert
    assert config.shared_swiglu_limits == shared


@pytest.mark.parametrize("overrides", [
    {"expert_swiglu_limit_list": [0]}, {"share_expert_swiglu_limit_list": [0]},
    {"expert_swiglu_limit_list": [0, 0, 0, 0, -1, 4]},
    {"expert_swiglu_limit_list": [0, 0, 0, 0, float("nan"), 4]},
    {"share_expert_swiglu_limit_list": [True] * 6},
    {"num_experts": 7}, {"topk_group": 3}, {"num_experts_per_tok": 5},
    {"kda_lower_bound": 0.0}, {"kda_lower_bound": float("nan")},
    {"num_nextn_predict_layers": 2}, {"q_lora_rank": 128},
    {"rope_scaling": {"type": "linear", "factor": 2.0}},
    {"rope_interleave": False}, {"no_kda_lora": False}, {"router_dtype": "bf16"},
    {"use_qkv_bias": True}, {"use_qk_norm": False}, {"linear_silu": False},
    {"num_kv_heads_for_linear_attn": 2}, {"group_norm_size": 2},
])
def test_invalid_config_is_rejected(tmp_path, overrides):
    with pytest.raises((ValueError, AssertionError)):
        make_config(tmp_path, **overrides)


def router_cfg(weight, bias, topk = 2):
    return SimpleNamespace(
        gate_tensor = weight, gate_tensor_f32 = None,
        e_score_correction_bias = bias, num_experts = weight.shape[1],
        num_experts_per_tok = topk, n_group = 2, topk_group = 1, routed_scaling_factor = 2.5,
    )


def independent_route(y, cfg):
    # A scalar/list selection oracle, deliberately not the engine's reshape/topk/masking code.
    logits = y.double() @ cfg.gate_tensor.double()
    scores = logits.sigmoid()
    output = []
    for row in scores:
        corrected = (row + cfg.e_score_correction_bias.double()).tolist()
        width = cfg.num_experts // cfg.n_group
        groups = [list(range(i * width, (i + 1) * width)) for i in range(cfg.n_group)]
        sums = [sum(sorted((corrected[j] for j in g), reverse = True)[:2]) for g in groups]
        eligible = sorted(range(cfg.n_group), key = lambda g: sums[g], reverse = True)[:cfg.topk_group]
        chosen = sorted([j for g in eligible for j in groups[g]], key = lambda j: corrected[j], reverse = True)[:cfg.num_experts_per_tok]
        weights = row[chosen]
        if len(chosen) > 1:
            weights = weights / (weights.sum() + 1e-20)
        output.append(dict(zip(chosen, (weights * cfg.routed_scaling_factor).tolist())))
    return output


@pytest.mark.parametrize("rows", [1, 2, 7, 33, 129])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_grouped_fp32_routing_against_scalar_oracle(rows, topk):
    gen = torch.Generator().manual_seed(19)
    w = torch.randn(12, 8, generator = gen).half()
    y = torch.randn(rows, 12, generator = gen).half()
    # All correction scores are negative: zero masking would leak excluded groups.
    bias = torch.tensor([-3.2, -3.0, -2.9, -3.4, -2.3, -2.8, -2.5, -2.4])
    cfg = router_cfg(w, bias, topk)
    ids, weights = routing_ds3_fp32(rows, cfg, y, {})
    expected = independent_route(y, cfg)
    for row in range(rows):
        actual = dict(zip(ids[row].tolist(), weights[row].float().tolist()))
        assert actual.keys() == expected[row].keys()
        for key in actual:
            assert actual[key] == pytest.approx(expected[row][key], rel = 6e-4, abs = 6e-4)
    assert cfg.gate_tensor_f32.dtype == torch.float
    assert cfg.gate_tensor.data_ptr() == w.data_ptr()
    cached = cfg.gate_tensor_f32
    routing_ds3_fp32(rows, cfg, y, {})
    assert cfg.gate_tensor_f32 is cached


def test_all_expert_calibration_does_not_mix_selection_bias():
    w = torch.arange(24).reshape(3, 8).half() / 32
    y = torch.tensor([[1, -1, 0.5]], dtype = torch.half)
    cfg = router_cfg(w, torch.arange(8).float() * 20)
    ids, weights = routing_ds3_fp32(1, cfg, y, {"activate_all_experts": True})
    raw = (y.float() @ w.float()).sigmoid()
    expected = (2.5 * raw / raw.sum(dim = -1, keepdim = True)).half()
    torch.testing.assert_close(ids, torch.arange(8).view(1, 8))
    torch.testing.assert_close(weights, expected, rtol = 0, atol = 0)
