import torch
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from transformers import AutoTokenizer, AutoModelForCausalLM
from aqlm import QuantizedLinear
from awq.modules.linear import WQLinear_GEMM
from vptq import VQuantLinear
from bitsandbytes.nn import Linear4bit

def get_tensors_size(tensors):
    return 8 * sum(t.element_size() * t.numel() for t in tensors.values() if t is not None)

def get_tensor_size(tensor):
    return 8 * tensor.element_size() * tensor.numel()

def scan_gpu_tensors(obj, seen = None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    total_size = 0
    # If it's a GPU tensor, add its memory usage.
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        total_size += obj.element_size() * obj.nelement()
    else:
        if isinstance(obj, dict):
            for key, value in obj.items():
                total_size += scan_gpu_tensors(key, seen)
                total_size += scan_gpu_tensors(value, seen)
            return total_size
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                total_size += scan_gpu_tensors(item, seen)
            return total_size
        if hasattr(obj, '__dict__'):
            total_size += scan_gpu_tensors(vars(obj), seen)
        if hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                try:
                    attr = getattr(obj, slot)
                    total_size += scan_gpu_tensors(attr, seen)
                except AttributeError:
                    continue
    return total_size

def get_storage_info(model):
    sum_bits = 0
    sum_numel = 0
    head_bpw = 0
    head_numel = 0
    for name, module in model.named_modules():
        if any(isinstance(module, x) for x in [Linear4bit]):
            if module.out_features >= model.vocab_size * 0.9:  # this is foolproof
                head_numel = module.in_features * module.out_features
                head_bpw = module.weight.numel() * 8
                head_bpw = (head_bpw + scan_gpu_tensors(module.quant_state) * 8) / head_numel
            else:
                sum_bits += module.weight.numel() * 8
                sum_bits += scan_gpu_tensors(module.quant_state) * 8
                sum_numel += module.in_features * module.out_features
        elif any(isinstance(module, x) for x in [torch.nn.Linear]):
            if module.out_features >= model.vocab_size * 0.9:
                head_bpw = module.weight.element_size() * 8
                head_numel = module.weight.numel()
            else:
                sum_bits += get_tensor_size(module.weight)
                sum_numel +=  module.weight.numel()
        elif any(isinstance(module, x) for x in [QuantizedLinear, VQuantLinear]):
            sum_bits += get_tensors_size(dict(module.named_parameters()))
            sum_numel += module.in_features * module.out_features
        elif any(isinstance(module, x) for x in [WQLinear_GEMM]):
            sum_bits += get_tensors_size({
                "qweight": module.qweight,
                "qzeros": module.qzeros,
                "scales": module.scales,
            })
            sum_numel += module.in_features * module.out_features
        elif any(isinstance(module, x) for x in [MarlinQuantLinear]):
            sum_bits += get_tensors_size({
                "g_idx": module.g_idx,
                "g_idx_sort_indices": module.g_idx_sort_indices,
                "qweight": module.qweight,
                "qzeros": module.qzeros,
                "scales": module.scales,
            })
            sum_numel += module.in_features * module.out_features
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

@torch.inference_mode
def load_transformers(model_dir: str, auto = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map = "auto" if auto else "cuda:0",
        torch_dtype = torch.half
    )
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

@torch.inference_mode
def load_transformers_auto(model_dir: str):
    return load_transformers(model_dir, auto = True)

@torch.inference_mode
def fwd_transformers(model_instance, input_ids: torch.Tensor):
    input_ids = input_ids.to("cuda:0")
    output = model_instance(input_ids)
    return output.logits

@torch.inference_mode
def tokenize_transformers(tokenizer_dir: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    output = tokenizer(text, return_tensors="pt")
    return output.input_ids
