"""
GLM-5.3-Flash vision tower parity against HF transformers (Glm5NextImageProcessor + Glm5NextVisionModel) on the
unquantized stub with the full vision tensors (registry role glm53-stub-hf):

  1. preprocessing: exllamav3's pixel patches and grid vs the HF image processor
  2. tower on the SAME pixels against HF fp32: exllamav3 (fp16) within 3x the distance of HF's own bf16 run (the
     tower amplifies precision noise, so an fp32 control is the only meaningful bar)
  3. end to end: exllamav3 embeddings vs the full HF bf16 pipeline (reported)
"""

import pytest
import torch

from testlib.parity import (Gate, control_check, hf_vision_tower, make_test_image, min_cos, rfn, trim_layer_lists,
                            truncated_checkpoint)

pytestmark = [pytest.mark.hf, pytest.mark.model("glm53-stub-hf")]


@pytest.fixture(scope = "module")
def stub_dir(model_registry, tmp_path_factory):
    """The stub with its per-layer config lists trimmed to its layer count (transformers validates them)"""
    src = model_registry.get("glm53-stub-hf").path
    return truncated_checkpoint(src, str(tmp_path_factory.mktemp("glm53_stub")),
                                lambda cfg: trim_layer_lists(cfg, "text_config"))


@torch.inference_mode()
def test_vision_tower(stub_dir, device):
    from exllamav3 import Config, Model, Tokenizer
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
    from transformers.models.glm5_next.image_processing_glm5_next import Glm5NextImageProcessor
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    image = make_test_image()
    config = Config.from_directory(stub_dir)
    assert "vision" in config.model_classes, "vision component not registered"
    vmodel = Model.from_config(config, component = "vision")
    vmodel.load(device)
    try:
        pix_e, _, grid = vmodel.preprocess(image)
        emb_e = vmodel.get_image_embeddings(Tokenizer.from_config(config), image).embeddings.float()
    finally:
        vmodel.unload()

    hf_in = Glm5NextImageProcessor.from_pretrained(stub_dir)(images = [image], return_tensors = "pt")
    pix_h, grid_h = hf_in["pixel_values"], hf_in["image_grid_thw"]
    gate = Gate("GLM-5.3 vision vs HF")
    gate.true("grid", list(grid_h[0]) == list(grid), f"HF {grid_h.tolist()} vs exl3 {list(grid)}")
    gate.lt("preprocess rfn", rfn(pix_e, pix_h), 5e-3)

    hf_vis = hf_vision_tower(Glm5NextVisionModel, Glm5NextConfig.from_pretrained(stub_dir), stub_dir,
                             "model.visual.", device)

    def run(pix, dtype):
        return hf_vis(pix.to(device, dtype), grid_thw = grid_h.to(device)).pooler_output.float().cpu()

    emb_h = run(pix_h, torch.bfloat16)
    emb_h_same = run(pix_e, torch.bfloat16)
    hf_vis.to(torch.float32)
    emb_32 = run(pix_e, torch.float32)
    gate.info("tower vs HF bf16, same pixels", f"rfn {rfn(emb_e, emb_h_same):.6f} min cos {min_cos(emb_e, emb_h_same):.6f}")
    gate.info("end to end vs HF bf16", f"rfn {rfn(emb_e, emb_h):.6f} min cos {min_cos(emb_e, emb_h):.6f}")
    control_check(gate, "tower rfn vs HF fp32", rfn(emb_e, emb_32), rfn(emb_h_same, emb_32))
    gate.assert_passes()
