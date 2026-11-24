"""Synthetic control vectors; no model downloads or local deployment files."""
import torch
from safetensors.torch import save_file


def write_control_vector(config, path, layers = None, seed = 41):
    path.parent.mkdir(parents = True, exist_ok = True)
    generator = torch.Generator().manual_seed(seed)
    tensors = {
        f"direction.{i}": torch.randn(config.hidden_size, generator = generator)
        for i in (range(config.num_hidden_layers) if layers is None else layers)
    }
    metadata = {
        "source_architecture": "qwen4_exp", "operation": "project",
        "hidden_size": str(config.hidden_size),
        "num_hidden_layers": str(config.num_hidden_layers), "hc_count": str(config.hc_mult),
    }
    save_file(tensors, str(path), metadata = metadata)
    return tensors, metadata
