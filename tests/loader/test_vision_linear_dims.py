"""
Vision-tower Linear declarations against the checkpoint: every quantizable Linear's unpadded dims must
match the stored weight (the loader pads a smaller tensor up to the declared shape, which silently
inflates an fp16 tower and breaks quantization), and its padded dims must sit on the 128-wide EXL3 tile
grid. GLM-4.6V (issue #446) had both faults: block MLPs declared at vision_config.intermediate_size
(10944) over 4096-wide weights, and an unpadded 10944-wide merger.

Runs over whichever of these VLM checkpoints exist under the registry's model root; skips the rest. Only builds
the module tree and reads safetensors headers.
"""

import os

import pytest

from exllamav3 import Config, Model
from exllamav3.modules import Linear

pytestmark = pytest.mark.nogpu

# Relative to the model root
MODELS = [
    "glm4.6v/exl3/4.07bpw",
    "glm4.6v-flash/hf",
    "glm5.3-flash/exl3/2.05bpw",
    "qwen3.5-2b/hf",
    "qwen3.8-27b/hf",
    "gemma4-12b-it/hf",
    "mimo-v2.6-pro-rl/hf",
    "ministral-3-3b-instruct-2512/hf",
    "gemma3-4b-it/hf",
]


@pytest.mark.parametrize("rel_dir", MODELS)
def test_vision_linear_dims(model_registry, rel_dir):
    if not model_registry.root:
        pytest.skip("no model root configured")
    model_dir = os.path.join(model_registry.root, rel_dir)
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        pytest.skip(f"{model_dir} not present")
    cfg = Config.from_directory(model_dir)
    if cfg.vision is None:
        pytest.skip("no vision tower")
    vm = Model.from_config(cfg, component = "vision")
    # Tile alignment matters for the towers the converter quantizes (architectures declaring a default
    # vision bitrate); the others are stored fp16 and may keep odd widths
    quantizable = bool(vm.caps.get("default_vision_bits"))
    stc = cfg.stc
    checked = 0
    for module in vm.modules:
        for lin in module:
            if not isinstance(lin, Linear) or not lin.qmap:
                continue
            if quantizable:
                assert lin.in_features % 128 == 0 and lin.out_features % 128 == 0, \
                    f"{lin.key}: padded dims {lin.in_features} x {lin.out_features} off the 128 tile grid"
            key = lin.key + ".weight"
            if lin.fkey is not None or not stc.has_tensor(key):
                continue   # fused-source or alt-key loads: the split is checked by the load itself
            filename = stc.tensor_file_map[key]
            h = stc.file_headers[filename][key]
            if len(h["shape"]) != 2 or h["dtype"] == "I16":   # EXL3 trellis tensors are tiles, not (out, in)
                continue
            out_f, in_f = h["shape"]
            assert (in_f, out_f) == (lin.in_features_unpadded, lin.out_features_unpadded), \
                f"{lin.key}: declared {lin.in_features_unpadded} x {lin.out_features_unpadded}, stored {in_f} x {out_f}"
            checked += 1
    if not checked:
        pytest.skip("tower stored quantized: no (out, in) weights to compare")
