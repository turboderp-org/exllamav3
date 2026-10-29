"""
LFM3-VL (Lfm3VlForConditionalGeneration) parity against the reference implementation shipped with the checkpoint
(transformers has no native LFM3-VL class, so the release's own modeling_lfm3_vl.py is loaded with trust_remote_code)
on real weights (registry role lfm3-vl-hf, unquantized bf16).

Covers the parts of the vision tower that are easy to get wrong: the gated depthwise 2D convolution over the patch
grid (the patches arrive grouped by merge block and are reordered onto the grid for the convolution), the axial 2D
rope (the same frequencies for the row and the column axis, laid out as [h, w, h, w]), the bidirectional
per-frame attention with RMSNorm on the query and key heads, and the projector that merges each 2x2 block into one
language model token.
"""

import pytest
import torch

from testlib.parity import Gate, make_test_image, logits_floor_gate

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.model("lfm3-vl-hf")]

PROMPT = "user\n<|vision_start|><|image|><|vision_end|>\nDescribe the image.\nassistant\n<think>"


def _hf_class(model_dir):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    return get_class_from_dynamic_module(
        "modeling_lfm3_vl.Lfm3VlForConditionalGeneration", model_dir)


def _hf_model(model_dir, dtype, attn_impl, device):
    cls = _hf_class(model_dir)
    model = cls.from_pretrained(
        model_dir, dtype = dtype, attn_implementation = attn_impl, trust_remote_code = True)
    model = model.to(device)
    model.eval()
    return model


def _hf_image_features(model_dir, image, device, dtype = torch.float32, attn_impl = "eager"):
    """Image embeddings (vision tower + projector) from the reference implementation."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code = True)
    inputs = processor(images = [image], return_tensors = "pt")
    model = _hf_model(model_dir, dtype, attn_impl, device)
    with torch.inference_mode():
        out = model.get_image_features(
            pixel_values = inputs["pixel_values"].to(device),
            image_grid_thw = inputs["image_grid_thw"].to(device))
    features = out.pooler_output[0] if hasattr(out, "pooler_output") else out[0]
    del model
    torch.cuda.empty_cache()
    return features.float().cpu()


def _exl3_image_features(model_dir, image, device):
    from exllamav3 import Config, Model, Tokenizer
    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)
    vision = Model.from_config(config, component = "vision")
    vision.load(device)
    try:
        mme = vision.get_image_embeddings(tokenizer, image)
        return mme.embeddings.float().cpu()
    finally:
        vision.unload()
        torch.cuda.empty_cache()


def _exl3_logits(model_dir, image, device):
    """Full text + image forward through the language model."""
    from exllamav3 import Config, Model, Tokenizer
    config = Config.from_directory(model_dir)
    tokenizer = Tokenizer.from_config(config)

    vision = Model.from_config(config, component = "vision")
    vision.load(device)
    text = Model.from_config(config, component = "text")
    text.load(device)
    try:
        mme = vision.get_image_embeddings(tokenizer, image, text_alias = "<|image|>")
        ids = tokenizer.encode(PROMPT, embeddings = [mme], encode_special_tokens = True)
        params = {"indexed_embeddings": [mme], "attn_mode": "flash_attn_nc"}
        with torch.inference_mode():
            logits = text.forward(ids, params)[0]
        return logits[0].float().cpu() if logits.dim() == 3 else logits.float().cpu()
    finally:
        vision.unload()
        text.unload()
        torch.cuda.empty_cache()


def _hf_logits(model_dir, image, device, dtype = torch.float32, attn_impl = "eager"):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code = True)
    inputs = processor(text = [PROMPT], images = [image], return_tensors = "pt")
    model = _hf_model(model_dir, dtype, attn_impl, device)
    with torch.inference_mode():
        out = model(input_ids = inputs["input_ids"].to(device),
                    pixel_values = inputs["pixel_values"].to(device),
                    image_grid_thw = inputs["image_grid_thw"].to(device),
                    use_cache = False)
    return out.logits[0].float().cpu()


def test_image_embeddings(model_dir, device):
    image = make_test_image()
    ref = _hf_image_features(model_dir, image, device)
    floor = _hf_image_features(model_dir, image, device, torch.bfloat16, "sdpa")
    got = _exl3_image_features(model_dir, image, device)
    gate = Gate("LFM3-VL image embeddings vs HF fp32 eager, floor HF bf16 sdpa")
    logits_floor_gate(gate, got, ref, floor)
    gate.assert_passes()


def test_logits(model_dir, device):
    image = make_test_image()
    ref = _hf_logits(model_dir, image, device)
    floor = _hf_logits(model_dir, image, device, torch.bfloat16, "sdpa")
    got = _exl3_logits(model_dir, image, device)
    gate = Gate("LFM3-VL text + image vs HF fp32 eager, floor HF bf16 sdpa")
    logits_floor_gate(gate, got, ref, floor)
    gate.assert_passes()
