from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TargetNode = str | list["TargetNode"]


def _gdn_probe_class() -> type:
    source = (
        Path(__file__).resolve().parents[1]
        / "exllamav3"
        / "modules"
        / "gated_delta_net.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GatedDeltaNet"
    )
    probe = copy.deepcopy(original)
    probe.name = "GatedDeltaNetProbe"
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
    return namespace["GatedDeltaNetProbe"]


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
