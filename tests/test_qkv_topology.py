import pytest
import torch

from exllamav3.model.config import NullConfig
from exllamav3.modules.attn import Attention
from exllamav3.conversion.qkv_topology import (
    QKVTopologyError,
    SCHEMA,
    concatenate_qkv_bf16,
    resolve_topology,
    split_qkv,
    validate_payload_index,
)


def descriptor(layer, q_K = 6, k_K = 5, v_K = 4):
    return {
        "layer": layer,
        "qmap": layer + ".input",
        "projections": [
            {
                "name": name,
                "K": K,
                "codebook": "mul1",
                "scale": "always",
                "out_features": width,
            }
            for name, K, width in zip(
                ("q_proj", "k_proj", "v_proj"),
                (q_K, k_K, v_K),
                (12288, 1024, 1024),
            )
        ],
    }


def mixed_plan():
    return {
        "schema": SCHEMA,
        "layers": [
            {"layer": "model.layers.0.self_attn", "variant": "split"},
            {
                "layer": "model.layers.1.self_attn",
                "variant": "fused_uniform",
                "K": 6,
                "codebook": "mul1",
                "scale": "always",
            },
        ],
    }


def test_mixed_layer_map_is_sorted_complete_and_defaults_to_split():
    topology = resolve_topology(
        [
            descriptor("model.layers.2.self_attn"),
            descriptor("model.layers.0.self_attn"),
            descriptor("model.layers.1.self_attn"),
        ],
        mixed_plan(),
    )

    assert topology["schema"] == SCHEMA
    assert [row["layer"] for row in topology["layers"]] == [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
        "model.layers.2.self_attn",
    ]
    assert [row["variant"] for row in topology["layers"]] == [
        "split",
        "fused_uniform",
        "split",
    ]
    assert topology["layers"][0]["projections"] == [
        {"name": "q_proj", "K": 6, "codebook": "mul1", "scale": "always"},
        {"name": "k_proj", "K": 5, "codebook": "mul1", "scale": "always"},
        {"name": "v_proj", "K": 4, "codebook": "mul1", "scale": "always"},
    ]
    assert topology["layers"][1]["projection"] == {
        "name": "qkv_proj",
        "K": 6,
        "codebook": "mul1",
        "scale": "always",
    }
    assert topology["layers"][1]["output_splits"] == [12288, 1024, 1024]

def test_attention_construction_rebuilds_mixed_logical_payloads_from_metadata():
    topology = resolve_topology(
        [descriptor("model.layers.0.self_attn"), descriptor("model.layers.1.self_attn")],
        mixed_plan(),
    )
    config = NullConfig()
    config.qkv_topology = {row["layer"]: row for row in topology["layers"]}

    def make_attention(layer, layer_idx):
        return Attention(
            config,
            layer,
            layer_idx,
            hidden_size = 5120,
            head_dim = 128,
            num_q_heads = 96,
            num_kv_heads = 8,
            rope_settings = None,
            key_q = "q_proj",
            key_k = "k_proj",
            key_v = "v_proj",
            key_o = "o_proj",
            qmap = layer,
        )

    split = make_attention("model.layers.0.self_attn", 0)
    fused = make_attention("model.layers.1.self_attn", 1)

    assert split.qkv_proj is None
    assert [split.q_proj.key, split.k_proj.key, split.v_proj.key] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert fused.q_proj is fused.k_proj is fused.v_proj is None
    assert fused.qkv_proj.key == "model.layers.1.self_attn.qkv_proj"
    assert fused.qkv_proj.out_features_unpadded == 12288 + 1024 + 1024

    with pytest.raises(ValueError, match = "tensor parallelism is not supported"):
        fused.make_tp_allocation({})


def test_bf16_concatenation_and_split_reconstruct_exact_bits():
    q = torch.arange(2 * 12, dtype = torch.float32).reshape(2, 12).to(torch.bfloat16)
    k = (100 + torch.arange(2 * 3, dtype = torch.float32)).reshape(2, 3).to(torch.bfloat16)
    v = (200 + torch.arange(2 * 3, dtype = torch.float32)).reshape(2, 3).to(torch.bfloat16)

    fused = concatenate_qkv_bf16((q, k, v))
    reconstructed = split_qkv(fused, (12, 3, 3))

    assert fused.dtype == torch.bfloat16
    assert fused.is_contiguous()
    for actual, expected in zip(reconstructed, (q, k, v)):
        assert torch.equal(actual.view(torch.uint16), expected.view(torch.uint16))


def test_fused_uniform_requires_one_common_K_codebook_and_scale():
    invalid_K = mixed_plan()
    invalid_K["layers"][1]["K"] = {"q_proj": 6, "k_proj": 5, "v_proj": 5}
    with pytest.raises(QKVTopologyError, match = "one integer value"):
        resolve_topology([descriptor("model.layers.0.self_attn"), descriptor("model.layers.1.self_attn")], invalid_K)

    mismatched_codebook = mixed_plan()
    mismatched_codebook["layers"][1]["codebook"] = "mcg"
    with pytest.raises(QKVTopologyError, match = "converter's one codebook"):
        resolve_topology(
            [descriptor("model.layers.0.self_attn"), descriptor("model.layers.1.self_attn")],
            mismatched_codebook,
        )


def test_index_reconstruction_rejects_missing_or_duplicate_payloads():
    topology = resolve_topology(
        [descriptor("model.layers.0.self_attn"), descriptor("model.layers.1.self_attn")],
        mixed_plan(),
    )
    keys = {
        "model.layers.0.self_attn.q_proj.trellis",
        "model.layers.0.self_attn.q_proj.mul1",
        "model.layers.0.self_attn.k_proj.trellis",
        "model.layers.0.self_attn.k_proj.mul1",
        "model.layers.0.self_attn.v_proj.trellis",
        "model.layers.0.self_attn.v_proj.mul1",
        "model.layers.1.self_attn.qkv_proj.trellis",
        "model.layers.1.self_attn.qkv_proj.mul1",
    }
    validate_payload_index(keys, topology)

    with pytest.raises(QKVTopologyError, match = "duplicate"):
        validate_payload_index(keys | {"model.layers.1.self_attn.q_proj.trellis"}, topology)
    with pytest.raises(QKVTopologyError, match = "missing"):
        validate_payload_index(keys - {"model.layers.0.self_attn.k_proj.trellis"}, topology)


def test_plan_rejects_unknown_layers_and_duplicate_declarations():
    plan = mixed_plan()
    plan["layers"].append({"layer": "model.layers.99.self_attn", "variant": "split"})
    with pytest.raises(QKVTopologyError, match = "unknown or unsupported"):
        resolve_topology(
            [descriptor("model.layers.0.self_attn"), descriptor("model.layers.1.self_attn")],
            plan,
        )

    plan = mixed_plan()
    plan["layers"].append({"layer": "model.layers.1.self_attn", "variant": "split"})
    with pytest.raises(QKVTopologyError, match = "Duplicate"):
        resolve_topology(
            [descriptor("model.layers.0.self_attn"), descriptor("model.layers.1.self_attn")],
            plan,
        )
