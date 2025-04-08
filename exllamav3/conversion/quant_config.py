from ..models import Config, Model
import os, json

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
        stored_tensors = config.stc.list_tensors(module.key)
        module_dict["stored_tensors"] = stored_tensors

        qformat = module.quant_format_id()
        if qformat == "exl3":
            shape = stored_tensors[f"{module.key}.trellis"]["shape"]
            module_dict["quant_format"] = "exl3"
            module_dict["bits_per_weight"] = shape[-1] // 16

        storage_dict[module.key] = module_dict

    # Grab quantization_config from config.json
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config_dict = json.load(f)
        assert "quantization_config" in config_dict, f"{model_dir} does not appear to be a quantized model"
        quantization_config = config_dict["quantization_config"]

    # Update config with storage data
    quantization_config["tensor_storage"] = storage_dict

    # Write
    with open(os.path.join(model_dir, "quantization_config.json"), "w") as f:
        f.write(json.dumps(quantization_config, indent = 4))
