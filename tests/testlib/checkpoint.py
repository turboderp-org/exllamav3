"""
Synthetic checkpoints served through the real loader.

Module tests build a module from a dict of tensors. Rather than a dict-backed stand-in for the safetensors
collection (which drifts from the real loader's signature and semantics), the tensors are written to a
safetensors file and read back through SafetensorsCollection, so codebook markers, arena allocation, padding
and dtype handling are the production code paths.

    stc = make_stc(tensors, tmp_path)
    config = module_config(tensors, tmp_path)       # has .stc and .infer_params, enough for Module(config, ...)
"""

import os
from types import SimpleNamespace

import torch
from safetensors.torch import save_file


def write_tensors(tensors: dict[str, torch.Tensor], directory: str, filename: str = "model.safetensors") -> str:
    os.makedirs(directory, exist_ok = True)
    path = os.path.join(directory, filename)
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path)
    return path


def make_stc(tensors: dict[str, torch.Tensor], directory) -> "SafetensorsCollection":
    from exllamav3.loader.safetensors import SafetensorsCollection
    directory = str(directory)
    write_tensors(tensors, directory)
    return SafetensorsCollection(directory)


def module_config(tensors: dict[str, torch.Tensor], directory, **infer_params):
    """Minimal config for constructing and loading standalone modules: .stc and .infer_params"""
    from exllamav3.model.config import InferParams
    ip = InferParams()
    for k, v in infer_params.items():
        assert hasattr(ip, k), f"InferParams has no field {k}"
        setattr(ip, k, v)
    return SimpleNamespace(stc = make_stc(tensors, directory), infer_params = ip)
