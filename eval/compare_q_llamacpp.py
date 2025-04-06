try:
    import llama_cpp
    import gguf
    from gguf import GGUFReader
    from llama_cpp import Llama
except:
    pass
import torch
from functools import lru_cache
from exllamav3.util.file import disk_lru_cache

@lru_cache  # run once
def init_backend():
    llama_cpp.llama_backend_init(False)

@disk_lru_cache("lcpp_get_storage_info")
def get_storage_info(model_dir):
    reader = GGUFReader(model_dir)
    tensors = reader.tensors
    sum_bits = 0
    sum_numel = 0
    head_bpw = 0
    head_numel = 0
    for tensor_info in tensors:
        name = tensor_info.name
        if any(name.endswith(k) for k in [
            ".ffn_down.weight",
            ".ffn_gate.weight",
            ".ffn_up.weight",
            ".attn_q.weight",
            ".attn_k.weight",
            ".attn_v.weight",
            ".attn_output.weight",
        ]):
            sum_bits += tensor_info.n_bytes * 8
            sum_numel += tensor_info.n_elements
        if (name == "token_embd.weight" and head_bpw == 0) or \
            name == "output.weight":
            head_bpw = tensor_info.n_bytes * 8 / tensor_info.n_elements
            head_numel = tensor_info.n_elements
    vram_bits = head_numel * head_bpw + sum_bits
    return sum_bits / sum_numel, head_bpw, vram_bits

def load_llamacpp(model_dir: str):
    init_backend()
    model = Llama(
        model_path = model_dir,
        logits_all = True,
        verbose = False,
        n_ctx = 2048,
        n_gpu_layers = 999
    )
    bpw_layer, bpw_head, vram_bits = get_storage_info(model_dir)
    return model, bpw_layer, bpw_head, vram_bits

def fwd_llamacpp(model_instance, input_ids: torch.Tensor):
    input_ids_list = input_ids[0].tolist()
    model_instance.reset()
    model_instance.eval(input_ids_list)
    logits = torch.from_numpy(model_instance.scores).unsqueeze(0).cuda()
    return logits