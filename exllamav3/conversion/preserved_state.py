"""Checkpoint tensor-valued side inputs used during layerwise calibration."""
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def save_preserved_state(rows, filename):
    tensors = {}
    for i, row in enumerate(rows):
        for key, value in row.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Calibration side input {key!r} is not a tensor")
            tensors[f"{i}/{key}"] = value.detach().cpu().contiguous()
    save_file(tensors, str(filename), metadata={"rows": str(len(rows))})


def load_preserved_state(filename, num_rows):
    rows = [{} for _ in range(num_rows)]
    with safe_open(str(filename), framework="pt", device="cpu") as f:
        if int(f.metadata()["rows"]) != num_rows:
            raise ValueError("Calibration side-input row count does not match checkpoint")
        for name in f.keys():
            row, key = name.split("/", 1)
            rows[int(row)][key] = f.get_tensor(name)
    return rows
