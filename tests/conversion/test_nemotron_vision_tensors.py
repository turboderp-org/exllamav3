"""Unquantized Nemotron Omni vision weights must survive the compile extras path."""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.architecture.nemotronh import NemotronHOmniVisionModel
from exllamav3.loader.safetensors import SafetensorsCollection
from testlib.checkpoint import write_tensors

pytestmark = pytest.mark.nogpu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("include_final_norm", [False, True])
def test_unquantized_vision_compile_tensors(tmp_path, dtype, include_final_norm):
    # The tower is not walked by compile_model when left unquantized. Its nested
    # weights and the separate projector must both be collected as extra tensors.
    vision = {
        "vision_model.radio_model.model.patch_generator.embedder.weight": torch.ones((4, 8), dtype = dtype),
        "vision_model.radio_model.model.blocks.0.norm1.bias": torch.zeros(4, dtype = dtype),
        "mlp1.0.weight": torch.ones(4, dtype = dtype),
        "mlp1.1.weight": torch.ones((8, 4), dtype = dtype),
    }
    if include_final_norm:
        vision.update({
            "vision_projector.vision_final_layernorm.weight": torch.ones(4, dtype = dtype),
            "vision_projector.vision_final_layernorm.bias": torch.zeros(4, dtype = dtype),
        })
    unrelated = {
        "language_model.embedding.weight": torch.ones((8, 4), dtype = dtype),
        "vision_model_extra.weight": torch.ones(4, dtype = dtype),
        "vision_projector_extra.weight": torch.ones(4, dtype = dtype),
        "mlp10.weight": torch.ones(4, dtype = dtype),
    }
    write_tensors(vision, tmp_path, "model-00001-of-00002.safetensors")
    write_tensors(unrelated, tmp_path, "model-00002-of-00002.safetensors")
    stc = SafetensorsCollection(str(tmp_path))

    extra = NemotronHOmniVisionModel.get_additional_compiled_tensors(SimpleNamespace(stc = stc))

    assert set(extra) == set(vision)
    for key, tensor in vision.items():
        assert extra[key]["shape"] == list(tensor.shape)
        assert extra[key]["n_bytes"] == tensor.numel() * tensor.element_size()
        assert extra[key]["torch_dtype"] == tensor.dtype
