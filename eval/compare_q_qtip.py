import torch

"""
Very kludgy wrapper for a custom, hacky install of the QTIP repo as a package. Don't expect this to work generally. 
Uses the QTIP package for inference so only supports Llama.
"""

try:
    import qtip
    from qtip.lib.linear.quantized_linear import QuantizedLinear
    from qtip.lib.utils.unsafe_import import model_from_hf_path
except ModuleNotFoundError:
    pass

def get_tensors_size(tensors):
    return 8 * sum(t.element_size() * t.numel() for t in tensors.values() if t is not None)

def get_tensor_size(tensor):
    return 8 * tensor.element_size() * tensor.numel()

def get_storage_info(model):
    sum_bits = 0
    sum_numel = 0
    head_bpw = 0
    head_numel = 0
    for name, module in model.named_modules():
        if any(isinstance(module, x) for x in [torch.nn.Linear]):
            if module.out_features >= model.vocab_size * 0.9:
                head_bpw = module.weight.element_size() * 8
                head_numel = module.weight.numel()
            else:
                sum_bits += get_tensor_size(module.weight)
                sum_numel +=  module.weight.numel()
        elif any(isinstance(module, x) for x in [QuantizedLinear]):
            sum_bits += get_tensors_size({
                "SU": module.SU,
                "SV": module.SV,
                "tlut": module.tlut,
                "trellis": module.trellis,
            })
            sum_numel += module.in_features * module.out_features
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

@torch.inference_mode
@torch.compiler.disable
def load_qtip(model_dir: str, auto = False, bf16 = False):
    model, model_str = model_from_hf_path(model_dir, max_mem_ratio = 0.7)
    bpw_layer, bpw_head, vram_bits = get_storage_info(model)
    return model, bpw_layer, bpw_head, vram_bits

@torch.inference_mode
@torch.compiler.disable
def fwd_qtip(model_instance, input_ids: torch.Tensor):
    input_ids = input_ids.to("cuda:0")
    output = model_instance(input_ids)
    return output.logits
