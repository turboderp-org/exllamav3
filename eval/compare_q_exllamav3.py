import torch
from exllamav3 import Config, Model, Tokenizer, Cache
from exllamav3.modules import Linear

def get_tensor_size(tensors):
    return 8 * sum(t.element_size() * t.numel() for t in tensors.values())

def get_storage_info(model):
    sum_bits = 0
    sum_numel = 0
    head_bpw = 0
    head_numel = 0
    for module in model:
        if module.key == "lm_head":
            head_bpw = get_tensor_size(module.get_tensors()) / module.weights_numel()
            head_numel = module.weights_numel()
        elif isinstance(module, Linear):
            sum_bits += get_tensor_size(module.get_tensors())
            sum_numel += module.weights_numel()
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

def load_exllamav3(model_dir: str | list):
    if isinstance(model_dir, list):
        model_dir, override_tensors = model_dir
        config = Config.from_directory(model_dir)
        config.stc.add_tensor_files(override_tensors)
    else:
        config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(max_output_size = 2048, max_output_factor = 3)
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

def fwd_exllamav3(model_instance, input_ids: torch.Tensor):
    input_ids = input_ids.cpu()
    output = model_instance.forward(input_ids, {"attn_mode": "flash_attn_nc"})
    return output