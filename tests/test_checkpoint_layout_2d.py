import os
import sys

import torch
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.loader.safetensors import SafetensorsCollection


def test_auto_transpose_to_pad_flips_only_when_needed(tmp_path):
    weight = torch.arange(6, dtype = torch.float16).reshape(2, 3)
    save_file({"layer.weight": weight}, tmp_path / "model.safetensors")

    stc = SafetensorsCollection(str(tmp_path), load_method = "python")
    loaded = stc.get_tensor(
        "layer.weight",
        device = "cpu",
        float2half = True,
        transpose = False,
        pad_to = (3, 2),
        auto_transpose_to_pad = True,
    )

    assert loaded.shape == (3, 2)
    torch.testing.assert_close(loaded, weight.T.contiguous())


def test_auto_transpose_to_pad_keeps_current_orientation_when_it_already_fits(tmp_path):
    weight = torch.arange(6, dtype = torch.float16).reshape(2, 3)
    save_file({"layer.weight": weight}, tmp_path / "model.safetensors")

    stc = SafetensorsCollection(str(tmp_path), load_method = "python")
    loaded = stc.get_tensor(
        "layer.weight",
        device = "cpu",
        float2half = True,
        transpose = False,
        pad_to = (3, 4),
        auto_transpose_to_pad = True,
    )

    expected = torch.zeros((3, 4), dtype = torch.float16)
    expected[:2, :3] = weight

    assert loaded.shape == (3, 4)
    torch.testing.assert_close(loaded, expected)
