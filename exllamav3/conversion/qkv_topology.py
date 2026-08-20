from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable

import torch
if TYPE_CHECKING:
    from ..modules import Attention, Linear



SCHEMA = "exl3_qkv_topology/1"
COMPONENTS = ("q_proj", "k_proj", "v_proj")
VARIANTS = ("split", "fused_uniform")
SCALE_VALUES = ("always", "never", "auto")
CODEBOOK_VALUES = ("mcg", "mul1")


class QKVTopologyError(ValueError):
    pass


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys = True, separators = (",", ":"))


def load_topology_plan(path: str) -> dict:
    with open(path, "r", encoding = "utf8") as f:
        plan = json.load(f)
    if not isinstance(plan, dict):
        raise QKVTopologyError("QKV topology plan must be a JSON object")
    if plan.get("schema") != SCHEMA:
        raise QKVTopologyError(f"QKV topology plan schema must be {SCHEMA!r}")
    layers = plan.get("layers")
    if not isinstance(layers, list):
        raise QKVTopologyError("QKV topology plan layers must be a list")
    return plan


def _validate_K(K, where: str) -> int:
    if isinstance(K, bool) or not isinstance(K, int) or not 3 <= K <= 8:
        raise QKVTopologyError(f"{where} K must be one integer value between 3 and 8")
    return K


def _validate_scale(scale, where: str) -> str:
    if scale not in SCALE_VALUES:
        raise QKVTopologyError(f"{where} scale must be one of {SCALE_VALUES}")
    return scale


def _plan_rows(plan: dict | None) -> dict[str, dict]:
    if plan is None:
        return {}
    if plan.get("schema") != SCHEMA or not isinstance(plan.get("layers"), list):
        raise QKVTopologyError(f"QKV topology plan must use schema {SCHEMA!r} and a layers list")
    result = {}
    for row in plan["layers"]:
        if not isinstance(row, dict):
            raise QKVTopologyError("Each QKV topology layer entry must be an object")
        allowed = {"layer", "variant", "K", "codebook", "scale"}
        extra = set(row) - allowed
        if extra:
            raise QKVTopologyError(f"Unknown QKV topology fields for layer entry: {sorted(extra)}")
        layer = row.get("layer")
        if not isinstance(layer, str) or not layer:
            raise QKVTopologyError("Each QKV topology layer entry requires a nonempty layer string")
        if layer in result:
            raise QKVTopologyError(f"Duplicate QKV topology declaration for {layer}")
        variant = row.get("variant")
        if variant not in VARIANTS:
            raise QKVTopologyError(f"Unknown QKV topology variant for {layer}: {variant!r}")
        if variant == "split":
            forbidden = set(row) & {"K", "codebook", "scale"}
            if forbidden:
                raise QKVTopologyError(
                    f"Split layer {layer} inherits independent projection settings; "
                    f"do not provide {sorted(forbidden)}"
                )
        else:
            _validate_K(row.get("K"), layer)
            if row.get("codebook") not in CODEBOOK_VALUES:
                raise QKVTopologyError(f"Fused layer {layer} requires one supported codebook")
            _validate_scale(row.get("scale"), layer)
        result[layer] = row
    return result


def attention_descriptor(attn: Attention, strategy: dict, codebook: str, scale: str) -> dict | None:
    from ..modules import Linear
    if attn.use_k_as_v:
        return None
    projections = []
    for name in COMPONENTS:
        linear = getattr(attn, name, None)
        if not isinstance(linear, Linear):
            return None
        if linear.key != attn.key + "." + name:
            return None
        if linear.key not in strategy:
            raise QKVTopologyError(f"No quantization strategy for {linear.key}")
        projections.append({
            "name": name,
            "K": strategy[linear.key],
            "codebook": codebook,
            "scale": scale,
            "out_features": linear.out_features_unpadded,
        })
    qmaps = {getattr(attn, name).qmap for name in COMPONENTS}
    if len(qmaps) != 1 or None in qmaps:
        raise QKVTopologyError(f"Attention layer {attn.key} does not have one shared QKV calibration identity")
    return {
        "layer": attn.key,
        "qmap": next(iter(qmaps)),
        "projections": projections,
    }


def resolve_topology(
    descriptors: Iterable[dict],
    plan: dict | None,
) -> dict:
    requested = _plan_rows(plan)
    by_layer = {}
    for descriptor in descriptors:
        layer = descriptor["layer"]
        if layer in by_layer:
            raise QKVTopologyError(f"Duplicate attention descriptor for {layer}")
        projections = descriptor.get("projections")
        if not isinstance(projections, list) or [p.get("name") for p in projections] != list(COMPONENTS):
            raise QKVTopologyError(f"Attention descriptor {layer} must declare q, k, v in frozen order")
        splits = [p.get("out_features") for p in projections]
        if any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in splits):
            raise QKVTopologyError(f"Attention descriptor {layer} has invalid output splits")
        by_layer[layer] = descriptor

    unknown = set(requested) - set(by_layer)
    if unknown:
        raise QKVTopologyError(f"QKV topology plan names unknown or unsupported attention layers: {sorted(unknown)}")

    rows = []
    for layer in sorted(by_layer):
        descriptor = by_layer[layer]
        projection_rows = descriptor["projections"]
        request = requested.get(layer, {"variant": "split"})
        variant = request["variant"]
        common = {
            "layer": layer,
            "variant": variant,
            "components": list(COMPONENTS),
            "output_splits": [p["out_features"] for p in projection_rows],
        }
        if variant == "split":
            for projection in projection_rows:
                _validate_K(projection["K"], layer)
                if projection["codebook"] not in CODEBOOK_VALUES:
                    raise QKVTopologyError(f"Projection codebook is invalid for {layer}")
                _validate_scale(projection["scale"], layer)
            common["projections"] = [
                {k: p[k] for k in ("name", "K", "codebook", "scale")}
                for p in projection_rows
            ]
        else:
            codebooks = {p["codebook"] for p in projection_rows}
            scales = {p["scale"] for p in projection_rows}
            if codebooks != {request["codebook"]} or scales != {request["scale"]}:
                raise QKVTopologyError(
                    f"Fused layer {layer} must use the converter's one codebook and scale policy"
                )
            common["projection"] = {
                "name": "qkv_proj",
                "K": request["K"],
                "codebook": request["codebook"],
                "scale": request["scale"],
            }
        rows.append(common)
    return {"schema": SCHEMA, "layers": rows}


def resolve_target_topology(
    target_model: Iterable,
    strategy: dict,
    codebook: str,
    scale: str,
    plan: dict | None,
) -> tuple[dict | None, tuple[Attention, ...]]:
    if plan is None:
        return None, ()

    from ..modules import Attention
    attentions = tuple(module for module in target_model if isinstance(module, Attention))
    descriptors = tuple(attention_descriptor(attn, strategy, codebook, scale) for attn in attentions)
    unsupported = [attn.key for attn, descriptor in zip(attentions, descriptors) if descriptor is None]
    if unsupported:
        raise QKVTopologyError(
            f"Topology metadata cannot describe full-attention layers with nonstandard QKV: {unsupported}"
        )
    return resolve_topology(descriptors, plan), attentions


def concatenate_qkv_bf16(weights: Iterable[torch.Tensor]) -> torch.Tensor:
    weights = tuple(weights)
    if len(weights) != 3:
        raise QKVTopologyError("QKV concatenation requires exactly q, k, v weights")
    if any(w.dtype != torch.bfloat16 for w in weights):
        raise QKVTopologyError("Fused-uniform QKV must be derived directly from BF16 weights")
    if any(w.ndim != 2 for w in weights):
        raise QKVTopologyError("QKV weights must be rank-2 tensors")
    if len({w.shape[0] for w in weights}) != 1:
        raise QKVTopologyError("QKV weights must have one common input width")
    return torch.cat(weights, dim = 1).contiguous()


def split_qkv(weight: torch.Tensor, output_splits: Iterable[int]) -> tuple[torch.Tensor, ...]:
    splits = tuple(output_splits)
    if len(splits) != 3 or any(isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in splits):
        raise QKVTopologyError("QKV output_splits must contain three positive integers")
    if weight.ndim != 2 or sum(splits) != weight.shape[1]:
        raise QKVTopologyError("QKV output_splits do not reconstruct the fused output width")
    return torch.split(weight, splits, dim = 1)


def _load_source_bf16(linear: Linear) -> tuple[torch.Tensor, torch.Tensor | None]:
    stc = linear.config.stc
    if linear.alt_key or linear.fkey or linear.is_sliced:
        raise QKVTopologyError(f"Fused-uniform PoC requires a direct, unsliced source tensor for {linear.key}")
    weight_key = linear.key + ".weight"
    if not stc.has_tensor(weight_key):
        raise QKVTopologyError(f"Missing direct source tensor {weight_key}")
    weight = stc.get_tensor(
        weight_key,
        linear.device,
        allow_bf16 = True,
        transpose = linear.transposed_load,
        no_defer = True,
    )
    expected = (linear.in_features_unpadded, linear.out_features_unpadded)
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != expected:
        raise QKVTopologyError(
            f"{weight_key} must be BF16 with logical shape {expected}, got {weight.dtype} {tuple(weight.shape)}"
        )
    bias_key = linear.key + ".bias"
    bias = stc.get_tensor(bias_key, linear.device, optional = True, allow_bf16 = True, no_defer = True)
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (linear.out_features_unpadded,)):
        raise QKVTopologyError(f"{bias_key} must be BF16 with logical output width")
    return weight, bias


def install_fused_qkv(attn: Attention) -> Linear:
    from ..modules import Linear
    from ..modules.quant import LinearFP16
    components = [getattr(attn, name, None) for name in COMPONENTS]
    if not all(isinstance(linear, Linear) for linear in components):
        raise QKVTopologyError(f"Attention layer {attn.key} is not a split full-attention QKV block")
    if attn.use_k_as_v:
        raise QKVTopologyError(f"Attention layer {attn.key} cannot preserve independent q,k,v outputs")
    if any(not isinstance(linear.inner, LinearFP16) for linear in components):
        raise QKVTopologyError(f"Attention layer {attn.key} must be loaded from source before QKV fusion")
    if len({linear.qmap for linear in components}) != 1 or components[0].qmap is None:
        raise QKVTopologyError(f"Attention layer {attn.key} does not share one QKV calibration identity")
    if len({linear.in_features_unpadded for linear in components}) != 1:
        raise QKVTopologyError(f"Attention layer {attn.key} QKV input widths differ")

    source = [_load_source_bf16(linear) for linear in components]
    weight = concatenate_qkv_bf16(item[0] for item in source)
    biases = [item[1] for item in source]
    if any(b is None for b in biases) and not all(b is None for b in biases):
        raise QKVTopologyError(f"Attention layer {attn.key} has only a partial QKV bias set")
    bias = torch.cat(biases).contiguous() if biases[0] is not None else None

    q = components[0]
    qkv = Linear(
        q.config,
        attn.key + ".qkv_proj",
        q.in_features_unpadded,
        sum(linear.out_features_unpadded for linear in components),
        qmap = q.qmap,
        qbits_key = q.qbits_key,
        trim_padded_out = True,
    )
    weight = qkv.pad_out(weight)
    bias = qkv.pad_out(bias)
    qkv.device = attn.device
    qkv.inner = LinearFP16(
        qkv.in_features,
        qkv.out_features,
        weight,
        bias,
        qkv.full_in_features,
        qkv.full_out_features,
        qkv.first_in_feature,
        qkv.first_out_feature,
        qkv.out_dtype,
        key = qkv.key,
    )
    qkv.quant_type = "fp16"

    component_ids = {id(linear) for linear in components}
    first = min(i for i, module in enumerate(attn.modules) if id(module) in component_ids)
    attn.modules = [module for module in attn.modules if id(module) not in component_ids]
    attn.modules.insert(first, qkv)
    for linear in components:
        linear.unload()
    attn.q_proj = None
    attn.k_proj = None
    attn.v_proj = None
    attn.qkv_proj = qkv
    return qkv


def apply_fused_strategy(attn: Attention, row: dict, strategy: dict) -> None:
    if row["variant"] != "fused_uniform":
        return
    component_keys = [getattr(attn, name).key for name in COMPONENTS]
    for key in component_keys:
        strategy.pop(key, None)
    strategy[attn.key + ".qkv_proj"] = row["projection"]["K"]


def topology_layer_map(topology: dict | None) -> dict[str, dict]:
    if topology is None:
        return {}
    if not isinstance(topology, dict) or topology.get("schema") != SCHEMA or not isinstance(topology.get("layers"), list):
        raise QKVTopologyError(f"Topology metadata must use schema {SCHEMA!r} and a layers list")
    result = {}
    previous = None
    common_keys = {"layer", "variant", "components", "output_splits"}
    projection_keys = {"name", "K", "codebook", "scale"}
    for row in topology["layers"]:
        if not isinstance(row, dict):
            raise QKVTopologyError("Topology layer declarations must be objects")
        layer = row.get("layer")
        if not isinstance(layer, str) or not layer or layer in result:
            raise QKVTopologyError("Topology layers must have unique nonempty layer names")
        if previous is not None and layer <= previous:
            raise QKVTopologyError("Topology layers must be in deterministic lexical order")
        previous = layer
        if row.get("components") != list(COMPONENTS):
            raise QKVTopologyError(f"Topology component order changed for {layer}")
        splits = row.get("output_splits")
        if not isinstance(splits, list) or len(splits) != 3 or any(
            isinstance(s, bool) or not isinstance(s, int) or s <= 0 for s in splits
        ):
            raise QKVTopologyError(f"Topology output_splits are invalid for {layer}")
        if row.get("variant") == "split":
            if set(row) != common_keys | {"projections"}:
                raise QKVTopologyError(f"Split topology fields are invalid for {layer}")
            declarations = row["projections"]
            if not isinstance(declarations, list) or [p.get("name") for p in declarations] != list(COMPONENTS):
                raise QKVTopologyError(f"Split topology for {layer} must declare q, k, v")
        elif row.get("variant") == "fused_uniform":
            if set(row) != common_keys | {"projection"}:
                raise QKVTopologyError(f"Fused topology fields are invalid for {layer}")
            declarations = [row["projection"]]
            if not isinstance(declarations[0], dict) or declarations[0].get("name") != "qkv_proj":
                raise QKVTopologyError(f"Fused topology for {layer} must declare qkv_proj")
        else:
            raise QKVTopologyError(f"Unknown topology variant for {layer}: {row.get('variant')!r}")
        for declaration in declarations:
            if not isinstance(declaration, dict) or set(declaration) != projection_keys:
                raise QKVTopologyError(f"Projection fields are invalid for {layer}")
            _validate_K(declaration["K"], layer)
            if declaration["codebook"] not in CODEBOOK_VALUES:
                raise QKVTopologyError(f"Projection codebook is invalid for {layer}")
            _validate_scale(declaration["scale"], layer)
        result[layer] = row
    return result


def validate_payload_index(tensor_keys: Iterable[str], topology: dict) -> None:
    layer_map = topology_layer_map(topology)
    keys = set(tensor_keys)
    for layer, row in layer_map.items():
        variant = row.get("variant")
        if variant == "split":
            declarations = row.get("projections")
            if not isinstance(declarations, list) or [p.get("name") for p in declarations] != list(COMPONENTS):
                raise QKVTopologyError(f"Compiled split topology for {layer} must declare q, k, v")
            logical = [(layer + "." + declaration["name"], declaration) for declaration in declarations]
            forbidden = layer + ".qkv_proj."
        elif variant == "fused_uniform":
            declaration = row.get("projection")
            if not isinstance(declaration, dict) or declaration.get("name") != "qkv_proj":
                raise QKVTopologyError(f"Compiled fused topology for {layer} must declare qkv_proj")
            logical = [(layer + ".qkv_proj", declaration)]
            forbidden = tuple(layer + "." + name + "." for name in COMPONENTS)
        else:
            raise QKVTopologyError(f"Compiled topology for {layer} has unknown variant {variant!r}")
        for prefix, declaration in logical:
            if prefix + ".trellis" not in keys:
                raise QKVTopologyError(f"Compiled payload is missing {prefix}.trellis")
            marker = {"mul1": ".mul1", "mcg": ".mcg"}.get(declaration.get("codebook"))
            if declaration.get("codebook") not in CODEBOOK_VALUES:
                raise QKVTopologyError(f"Compiled payload declares an unknown codebook for {prefix}")
            if marker is not None and prefix + marker not in keys:
                raise QKVTopologyError(f"Compiled payload is missing codebook marker {prefix + marker}")
            other_markers = {prefix + ".mul1", prefix + ".mcg"}
            expected_marker = {prefix + marker} if marker is not None else set()
            if keys & (other_markers - expected_marker):
                raise QKVTopologyError(f"Compiled payload has a conflicting codebook marker for {prefix}")
        forbidden_prefixes = (forbidden,) if isinstance(forbidden, str) else forbidden
        duplicate = sorted(key for key in keys if key.startswith(forbidden_prefixes))
        if duplicate:
            raise QKVTopologyError(f"Compiled payload contains duplicate QKV topology tensors: {duplicate[:3]}")
