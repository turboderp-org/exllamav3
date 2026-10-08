"""
Architecture integration smoke checks over registry models: the config resolves to the expected architecture, the
text (and vision) graphs construct, one block of every attention layer type loads and runs a short forward with the
right output shape and finite values, and architecture-specific structure holds (Gemma4 full attention omits v_proj,
sliding attention has it; Qwen3.5 linear-attention layers are Gated DeltaNet). With a vision tower, an image yields
embeddings, and (slow, Gemma4) greedy multimodal generation names the animal in examples/media/cat.png.
"""

import os

import pytest
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAT_IMAGE = os.path.join(REPO_DIR, "examples", "media", "cat.png")

ARCHS = {
    "swa": {"Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"},
    "recurrent": {"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"},
}


def _check_gemma4(cfg, layer_type, block):
    if layer_type == "sliding_attention":
        assert block.attn.v_proj is not None, "Gemma4 sliding attention should have a v_proj"
    if layer_type == "full_attention":
        assert block.attn.v_proj is None, "Gemma4 full attention should omit v_proj"


def _check_qwen3_5(cfg, layer_type, block):
    from exllamav3.modules import GatedDeltaNet
    if layer_type == "linear_attention":
        assert isinstance(block.attn, GatedDeltaNet), f"linear attention is {type(block.attn).__name__}"


CHECKS = {"swa": _check_gemma4, "recurrent": _check_qwen3_5}
MULTIMODAL_GENERATION = {"swa"}


@pytest.fixture(scope = "module", params = [pytest.param(r, marks = pytest.mark.model(r)) for r in ARCHS])
def role(request):
    return request.param


@pytest.fixture(scope = "module")
def setup(role, model_registry):
    from exllamav3 import Config, Model, Tokenizer
    cfg = Config.from_directory(model_registry.get(role).path)
    model = Model.from_config(cfg)
    vision = Model.from_config(cfg, component = "vision") if "vision" in cfg.model_classes else None
    return cfg, model, vision, Tokenizer.from_config(cfg)


def test_architecture(setup, role):
    cfg = setup[0]
    assert cfg.architecture in ARCHS[role], f"unexpected architecture {cfg.architecture}"


@torch.inference_mode()
def test_block_forward(setup, role, device):
    cfg, model, _, _ = setup
    layer_types = sorted(set(cfg.layer_types))
    if cfg.architecture.startswith("Gemma4"):
        assert {"sliding_attention", "full_attention"} <= set(layer_types)
    else:
        assert "full_attention" in layer_types, "no full_attention layer in layer_types"
    if getattr(cfg, "enable_moe_block", False):
        layer_types.append("moe")
    for lt in layer_types:
        idx = model.first_block_idx + (0 if lt == "moe" else cfg.layer_types.index(lt))
        block = model.modules[idx]
        block.load(device)
        try:
            CHECKS[role](cfg, lt, block)
            x = torch.randn((1, 8, cfg.hidden_size), device = device, dtype = torch.half)
            y = block.forward(x, params = {})
            assert y.shape[:2] == (1, 8) and y.shape[-1] == cfg.hidden_size, f"{lt} block output {tuple(y.shape)}"
            assert torch.isfinite(y).all(), f"{lt} block output not finite"
        finally:
            block.unload()


@torch.inference_mode()
def test_image_embeddings(setup, device):
    from PIL import Image
    _, _, vision, tokenizer = setup
    if vision is None:
        pytest.skip("no vision component")
    vision.load(device)
    try:
        emb = vision.get_image_embeddings(tokenizer, Image.new("RGB", (640, 480), (255, 0, 0)))
        assert emb.mm_length > 0, "vision tower produced no image embeddings"
    finally:
        vision.unload()


@pytest.mark.slow
@torch.inference_mode()
def test_multimodal_generation(setup, role, device):
    from PIL import Image
    from exllamav3 import Cache, Generator, Job
    from exllamav3.generator.sampler import GreedySampler
    if role not in MULTIMODAL_GENERATION:
        pytest.skip("multimodal generation is checked on Gemma4 only")
    cfg, model, vision, tokenizer = setup
    vision.load(device)
    try:
        emb = vision.get_image_embeddings(tokenizer, Image.open(CAT_IMAGE).convert("RGB"))
    finally:
        vision.unload()
    prompt_ids = tokenizer.hf_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                      {"type": "text", "text": "What animal is shown? Answer with one word."}]}],
        add_generation_prompt = True,
        enable_thinking = False,
        embeddings = [emb],
    )
    cache = Cache(model, max_num_tokens = 4096)
    model.load(device)
    try:
        generator = Generator(model, cache, tokenizer)
        job = Job(input_ids = prompt_ids.cpu(), max_new_tokens = 8, embeddings = [emb], sampler = GreedySampler(),
                  stop_conditions = [tokenizer.eos_token_id, "<turn|>"], decode_special_tokens = True)
        generator.enqueue(job)
        text = ""
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("stage") == "streaming":
                    text += r.get("text", "")
    finally:
        model.unload()
        del cache
        torch.cuda.empty_cache()
    assert "cat" in text.lower(), f"unexpected multimodal generation output: {text!r}"
