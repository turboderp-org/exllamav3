import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from exllamav2.model import ExLlamaV2Linear

def get_tensor_size(tensors):
    return 8 * sum(t.element_size() * t.numel() for t in tensors.values())

def get_storage_info(model):
    sum_bits = 0
    sum_numel = 0
    head_bpw = 0
    head_numel = 0
    for key, module in model.modules_dict.items():
        if module.key == "lm_head":
            head_bpw = get_tensor_size(module.q_tensors) / module.numel()
            head_numel = module.numel()
        elif isinstance(module, ExLlamaV2Linear):
            sum_bits += get_tensor_size(module.q_tensors)
            sum_numel += module.numel()
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

def load_exllamav2(model_dir: str | list):
    config = ExLlamaV2Config(model_dir)
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, batch_size = 1, max_seq_len = 2048)  # Cache isn't used but reqd by autosplit
    model.load_autosplit(cache, reserve_vram = 1024**3)
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

def fwd_exllamav2(model_instance, input_ids: torch.Tensor):
    input_ids = input_ids.cpu()
    output = model_instance.forward(input_ids)
    return output