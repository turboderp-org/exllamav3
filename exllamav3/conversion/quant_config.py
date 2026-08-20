from ..model import Config, Model
import os, json
import torch
from .qkv_topology import COMPONENTS, SCHEMA, QKVTopologyError

def update_config(
    config_dict: dict
):
    """
    Make necessary updates to config.json
    """
    if "tied_word_embeddings" in config_dict:
        config_dict["tied_word_embeddings"] = True


def create_quantization_config_json(
    model_dir: str
):
    # Create model instance without loading
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)

    # Create tensor map
    storage_dict = {}
    for module in model:
        # Only list leaf nodes
        if len(module.modules) > 0:
            continue

        module_dict = {}
        stored_tensors = config.stc.list_tensors(module.key, only_serializable = True)
        module_dict["stored_tensors"] = stored_tensors

        qformat = module.quant_format_id()
        if qformat == "exl3":
            shape = stored_tensors[f"{module.key}.trellis"]["shape"]

            mul1 = config.stc.get_tensor(f"{module.key}.mul1", optional = True, no_defer = True)
            mul1_mult = mul1.view(torch.uint32).item() if mul1 is not None else 0
            mcg = config.stc.get_tensor(f"{module.key}.mcg", optional = True, no_defer = True)
            mcg_mult = mcg.view(torch.uint32).item() if mcg is not None else 0

            module_dict["quant_format"] = "exl3"
            module_dict["bits_per_weight"] = shape[-1] // 16
            if mul1_mult:
                module_dict["mul1_multiplier"] = mul1_mult
            if mcg_mult:
                module_dict["mcg_multiplier"] = mcg_mult

        storage_dict[module.key] = module_dict

    # Grab quantization_config from config.json
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config_dict = json.load(f)
        assert "quantization_config" in config_dict, f"{model_dir} does not appear to be a quantized model"
        quantization_config = config_dict["quantization_config"]
    topology = quantization_config.get("exl3_qkv_topology")
    if topology is not None:
        if topology.get("schema") != SCHEMA or not isinstance(topology.get("layers"), list):
            raise QKVTopologyError(f"quantization_config.exl3_qkv_topology must use {SCHEMA!r}")

        def storage_entry(logical_key):
            stored_tensors = config.stc.list_tensors(logical_key, only_serializable = True)
            trellis_key = logical_key + ".trellis"
            if trellis_key not in stored_tensors:
                raise QKVTopologyError(f"Missing indexed logical payload {trellis_key}")
            shape = stored_tensors[trellis_key]["shape"]
            entry = {
                "stored_tensors": stored_tensors,
                "quant_format": "exl3",
                "bits_per_weight": shape[-1] // 16,
            }
            mul1 = config.stc.get_tensor(logical_key + ".mul1", optional = True, no_defer = True)
            mcg = config.stc.get_tensor(logical_key + ".mcg", optional = True, no_defer = True)
            if mul1 is not None and mcg is not None:
                raise QKVTopologyError(f"Logical payload {logical_key} has multiple codebook markers")
            entry["codebook"] = "mul1" if mul1 is not None else "mcg" if mcg is not None else "3inst"
            if mul1 is not None:
                entry["mul1_multiplier"] = mul1.view(torch.uint32).item()
            if mcg is not None:
                entry["mcg_multiplier"] = mcg.view(torch.uint32).item()
            return entry

        seen_layers = set()
        for row in topology["layers"]:
            layer = row.get("layer")
            if not isinstance(layer, str) or layer in seen_layers:
                raise QKVTopologyError("Topology layers must have unique nonempty layer names")
            seen_layers.add(layer)
            if row.get("components") != list(COMPONENTS):
                raise QKVTopologyError(f"Topology component order changed for {layer}")
            if row.get("variant") == "split":
                declarations = row.get("projections")
                if not isinstance(declarations, list) or [p.get("name") for p in declarations] != list(COMPONENTS):
                    raise QKVTopologyError(f"Split topology for {layer} must declare q, k, v projections")
                storage_dict.pop(layer + ".qkv_proj", None)
            elif row.get("variant") == "fused_uniform":
                declaration = row.get("projection")
                if not isinstance(declaration, dict) or declaration.get("name") != "qkv_proj":
                    raise QKVTopologyError(f"Fused topology for {layer} must declare qkv_proj")
                declarations = [declaration]
                for name in COMPONENTS:
                    storage_dict.pop(layer + "." + name, None)
            else:
                raise QKVTopologyError(f"Unknown topology variant for {layer}: {row.get('variant')!r}")

            for declaration in declarations:
                logical_key = layer + "." + declaration["name"]
                entry = storage_entry(logical_key)
                if entry["bits_per_weight"] != declaration.get("K"):
                    raise QKVTopologyError(
                        f"Indexed K for {logical_key} is {entry['bits_per_weight']}, "
                        f"metadata declares {declaration.get('K')}"
                    )
                if entry["codebook"] != declaration.get("codebook"):
                    raise QKVTopologyError(
                        f"Indexed codebook for {logical_key} is {entry['codebook']}, "
                        f"metadata declares {declaration.get('codebook')}"
                    )
                entry["scale"] = declaration.get("scale")
                storage_dict[logical_key] = entry


    # Update config with storage data
    quantization_config["tensor_storage"] = storage_dict

    # Write
    with open(os.path.join(model_dir, "quantization_config.json"), "w") as f:
        f.write(json.dumps(quantization_config, indent = 4))
