"""
vision_pinned loading (vision-tower weights moved to pinned host memory behind zero-copy device aliases), loading
only the vision component of a multimodal model through the autosplit loader on the test device:

- the VRAM held after the pinned load drops by the bytes moved to host (a native handle built before pinning would
  otherwise keep the original trellis tensors alive)
- the pinned tower produces the same embeddings as the resident one (native pointer tables rebuilt after the move)
"""

import numpy as np
import pytest
import torch
from PIL import Image

from exllamav3 import Config, Model, Tokenizer
from exllamav3.modules import Linear
from exllamav3.util.memory import free_mem

pytestmark = pytest.mark.models("vision")

RESERVE_GB = 96 / 1024
SLACK = 16 << 20


@pytest.fixture(autouse = True)
def inference_mode():
    with torch.inference_mode():
        yield


def load_vision(model_dir, device, pinned: bool):
    config = Config.from_directory(model_dir)
    config.infer_params.vision_pinned = pinned
    model = Model.from_config(config, component = "vision")
    # Autosplit path restricted to the test device (negative reserve excludes a device)
    reserve = [-1.0] * torch.cuda.device_count()
    reserve[device.index] = RESERVE_GB
    model.load(reserve_per_device = reserve, progressbar = False)
    return config, model


def pinned_bytes(model):
    stores = [
        m.inner._pinned_store for m in model
        if isinstance(m, Linear) and getattr(m.inner, "_pinned_store", None) is not None
    ]
    assert stores, "no linear was pinned; vision_pinned had no effect"
    return sum(s.numel() * s.element_size() for s in stores)


def test_pinned_vision_load_releases_moved_weights(model_id, model_dir, device):
    def allocated_after_load(pinned):
        _, model = load_vision(model_dir, device, pinned)
        try:
            return torch.cuda.memory_allocated(device), pinned_bytes(model) if pinned else 0
        finally:
            model.unload()
            free_mem()

    resident, _ = allocated_after_load(False)
    pinned, moved = allocated_after_load(True)
    # Everything moved to pinned host memory must leave VRAM
    assert pinned <= resident - moved + SLACK, (
        f"pinned load holds {pinned >> 20} MiB of VRAM, resident load {resident >> 20} MiB, "
        f"{moved >> 20} MiB moved to host: {(pinned - (resident - moved)) >> 20} MiB of moved "
        f"weights are still resident"
    )


def test_pinned_vision_embeddings_match_resident(model_id, model_dir, device):
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (448, 448, 3), dtype = np.uint8))

    def embed(pinned):
        config, model = load_vision(model_dir, device, pinned)
        try:
            tokenizer = Tokenizer.from_config(config)
            return model.get_image_embeddings(tokenizer, image).embeddings.float()
        finally:
            model.unload()
            free_mem()

    resident = embed(False)
    pinned = embed(True)
    assert resident.shape == pinned.shape
    torch.testing.assert_close(pinned, resident, atol = 1e-3, rtol = 1e-3)
