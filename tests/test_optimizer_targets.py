from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TargetNode = str | list["TargetNode"]
QWEN38_NUM_LAYERS = 64
QWEN38_FULL_ATTENTION_INTERVAL = 4


def _source_node(relative_path: str, node_name: str, node_type: type) -> tuple[Path, Any]:
    source = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    original = next(
        node
        for node in tree.body
        if isinstance(node, node_type) and node.name == node_name
    )
    return source, copy.deepcopy(original)


def _optimizer_probe_class(relative_path: str, class_name: str) -> type:
    source, probe = _source_node(relative_path, class_name, ast.ClassDef)
    probe.name = f"{class_name}Probe"
    probe.bases = []
    probe.keywords = []
    probe.decorator_list = []
    probe.body = [
        node
        for node in probe.body
        if isinstance(node, ast.FunctionDef) and node.name == "optimizer_targets"
    ]
    assert len(probe.body) == 1
    probe.body[0].decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    namespace: dict[str, Any] = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[probe.name]


def _qwen38_layer_types() -> list[str]:
    source, function = _source_node(
        "exllamav3/architecture/qwen3_5.py",
        "read_qwen3_5_layer_types",
        ast.FunctionDef,
    )
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace: dict[str, Any] = {"Config": Any}
    exec(compile(module, str(source), "exec"), namespace)
    config = SimpleNamespace(read_cfg=lambda _type, _path, default: default)
    return namespace["read_qwen3_5_layer_types"](
        config,
        "",
        QWEN38_NUM_LAYERS,
        QWEN38_FULL_ATTENTION_INTERVAL,
    )


def _gdn_probe_class() -> type:
    return _optimizer_probe_class(
        "exllamav3/modules/gated_delta_net.py",
        "GatedDeltaNet",
    )


def _linear(key: str) -> SimpleNamespace:
    return SimpleNamespace(optimizer_targets=lambda: [key])


def _flatten_measure_targets(
    node: list[TargetNode], max_depth: int
) -> list[list[str]]:
    """Mirror conversion/measure_model.py's target grouping traversal."""
    groups: list[list[str]] = []

    def flatten(current: list[TargetNode], depth: int = 0) -> list[str]:
        leaves: list[str] = []
        for child in current:
            if isinstance(child, str):
                leaves.append(child)
            else:
                leaves.extend(flatten(child, depth + 1))
        if depth == max_depth:
            groups.append(leaves)
        return leaves

    flatten(node)
    return groups


def _split_gdn() -> Any:
    module = _gdn_probe_class()()
    module.qkvz_proj = None
    module.qkv_proj = _linear("gdn.qkv")
    module.z_proj = _linear("gdn.z")
    module.o_proj = _linear("gdn.o")
    return module


def _fused_gdn() -> Any:
    module = _gdn_probe_class()()
    module.qkvz_proj = _linear("gdn.qkvz")
    module.qkv_proj = None
    module.z_proj = None
    module.o_proj = _linear("gdn.o")
    return module


def test_split_gdn_level3_measures_inputs_and_output_once() -> None:
    transformer_targets = [_split_gdn().optimizer_targets(), []]
    assert _flatten_measure_targets(transformer_targets, 3) == [
        ["gdn.qkv"],
        ["gdn.z"],
        ["gdn.o"],
    ]


def test_split_gdn_level2_keeps_output_in_combined_group() -> None:
    transformer_targets = [_split_gdn().optimizer_targets(), []]
    assert _flatten_measure_targets(transformer_targets, 2) == [
        ["gdn.qkv", "gdn.z", "gdn.o"],
    ]


def test_fused_gdn_level3_measures_input_and_output_once() -> None:
    transformer_targets = [_fused_gdn().optimizer_targets(), []]
    assert _flatten_measure_targets(transformer_targets, 3) == [
        ["gdn.qkvz"],
        ["gdn.o"],
    ]


def test_qwen38_optimizer_topology_has_expected_unique_groups_and_linears() -> None:
    gdn_class = _gdn_probe_class()
    attention_class = _optimizer_probe_class(
        "exllamav3/modules/attn.py",
        "Attention",
    )
    mlp_class = _optimizer_probe_class(
        "exllamav3/modules/mlp.py",
        "GatedMLP",
    )
    block_class = _optimizer_probe_class(
        "exllamav3/modules/transformer.py",
        "TransformerBlock",
    )
    linears: list[SimpleNamespace] = []

    def linear(key: str) -> SimpleNamespace:
        module = _linear(key)
        linears.append(module)
        return module

    blocks = []
    for layer_idx, layer_type in enumerate(_qwen38_layer_types()):
        prefix = f"layers.{layer_idx}"
        if layer_type == "linear_attention":
            attn = gdn_class()
            attn.qkvz_proj = None
            attn.qkv_proj = linear(f"{prefix}.gdn.qkv")
            attn.z_proj = linear(f"{prefix}.gdn.z")
            attn.o_proj = linear(f"{prefix}.gdn.o")
        else:
            assert layer_type == "full_attention"
            attn = attention_class()
            attn.qkv_proj = None
            attn.q_proj = linear(f"{prefix}.attention.q")
            attn.k_proj = linear(f"{prefix}.attention.k")
            attn.v_proj = linear(f"{prefix}.attention.v")
            attn.o_proj = linear(f"{prefix}.attention.o")

        mlp = mlp_class()
        mlp.gates = [linear(f"{prefix}.mlp.gate")]
        mlp.ups = [linear(f"{prefix}.mlp.up")]
        mlp.downs = [linear(f"{prefix}.mlp.down")]

        block = block_class()
        block.attn = attn
        block.mlp = mlp
        blocks.append(block)

    groups = [
        group
        for block in blocks
        for group in _flatten_measure_targets(block.optimizer_targets(), 3)
    ]
    group_kinds = [
        "gdn" if ".gdn." in group[0]
        else "attention" if ".attention." in group[0]
        else "mlp"
        for group in groups
    ]
    targets = [target for group in groups for target in group]

    assert group_kinds.count("gdn") == 144
    assert group_kinds.count("attention") == 48
    assert group_kinds.count("mlp") == 128
    assert len(groups) == 320
    assert len({tuple(group) for group in groups}) == 320
    assert len(linears) == 400
    assert len({id(linear) for linear in linears}) == 400
    assert len(targets) == len(set(targets)) == 400
