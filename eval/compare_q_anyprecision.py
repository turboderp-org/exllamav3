import torch
try:
    from any_precision.modules.AnyPrecisionForCausalLM import AnyPrecisionForCausalLM
    from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
except ModuleNotFoundError:
    pass
except ImportError:
    pass

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
        if isinstance(module, AnyPrecisionLinear):
            mods = {"qweight": module.qweight}
            match module.precision:
                case 1:
                    mods.update({"g_idx": module.lut1})
                case 2:
                    mods.update({"g_idx": module.lut2})
                case 3:
                    mods.update({"g_idx": module.lut3})
                case 4:
                    mods.update({"g_idx": module.lut4})
                case 5:
                    mods.update({"g_idx": module.lut5})
                case 6:
                    mods.update({"g_idx": module.lut6})
                case 7:
                    mods.update({"g_idx": module.lut7})
                case 8:
                    mods.update({"g_idx": module.lut8})
            sum_bits += get_tensors_size(mods)
            sum_numel += module.in_features * module.out_features
        elif any(isinstance(module, x) for x in [torch.nn.Linear]):
            if module.out_features >= model.config.vocab_size * 0.9:
                head_bpw = module.weight.element_size() * 8
                head_numel = module.weight.numel()
            else:
                sum_bits += get_tensor_size(module.weight)
                sum_numel +=  module.weight.numel()
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

@torch.inference_mode
def load_anyprecision(model_dir: str, auto = False, bf16 = False):
    model = AnyPrecisionForCausalLM.from_quantized(
        model_dir,
    )
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

@torch.inference_mode
def fwd_anyprecision(model_instance, input_ids: torch.Tensor):
    input_ids = input_ids.to("cuda:0")
    output = model_instance(input_ids)
    return output.logits
