import os
import sys

import torch
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.loader.safetensors import SafetensorsCollection


def test_cast_dtype_promotes_bf16_tensor(tmp_path):
    tensor = torch.tensor([1.0, -2.0, 3.5], dtype = torch.bfloat16)
    save_file({"state.A_log": tensor}, tmp_path / "model.safetensors")

    stc = SafetensorsCollection(str(tmp_path), load_method = "python")
    loaded = stc.get_tensor(
        "state.A_log",
        device = "cpu",
        allow_bf16 = True,
        cast_dtype = torch.float,
    )

    assert loaded.dtype == torch.float
    torch.testing.assert_close(loaded, tensor.float())


def test_cast_dtype_leaves_bf16_tensor_unchanged_when_not_requested(tmp_path):
    tensor = torch.tensor([1.0, -2.0, 3.5], dtype = torch.bfloat16)
    save_file({"state.dt_bias": tensor}, tmp_path / "model.safetensors")

    stc = SafetensorsCollection(str(tmp_path), load_method = "python")
    loaded = stc.get_tensor(
        "state.dt_bias",
        device = "cpu",
        allow_bf16 = True,
    )

    assert loaded.dtype == torch.bfloat16
    torch.testing.assert_close(loaded, tensor)
