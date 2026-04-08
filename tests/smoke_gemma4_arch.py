#!/usr/bin/env python3
"""
Lightweight smoke checks for Gemma4 text architecture integration.

Checks:
1) Config/architecture resolution
2) Text and vision model graph construction
3) One sliding-attn block forward
4) One full-attn block forward
5) One MoE block forward when enabled
6) One image embedding extraction through the vision tower

Optional:
7) Attempt full text model load
"""

import argparse
import os
import sys
import torch
from PIL import Image

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full_load", action="store_true")
    ap.add_argument("--reserve_per_device", default=None, help="e.g. 0.25,0.25")
    ap.add_argument("--multimodal_generation", action="store_true")
    ap.add_argument("--multimodal_image", default=None)
    args = ap.parse_args()

    cfg = Config.from_directory(args.model_dir)
    print("[INFO] architecture:", cfg.architecture)
    if cfg.architecture != "Gemma4ForConditionalGeneration":
        fail(f"Unexpected architecture: {cfg.architecture}")

    model = Model.from_config(cfg)
    vision_model = Model.from_config(cfg, component = "vision")
    tokenizer = Tokenizer.from_config(cfg)
    print("[INFO] model graph ok")
    print("[INFO] vision graph ok")

    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")

    device = torch.device(args.device)
    hidden_size = cfg.hidden_size

    idx_sliding = model.first_block_idx + cfg.layer_types.index("sliding_attention")
    blk_sliding = model.modules[idx_sliding]
    blk_sliding.load(device)
    if blk_sliding.attn.v_proj is None:
        fail("Expected sliding attention to have a v_proj")
    x = torch.randn((1, 8, hidden_size), device = device, dtype = torch.half)
    with torch.inference_mode():
        y = blk_sliding.forward(x, params = {})
    print("[INFO] sliding block forward:", blk_sliding.key, tuple(y.shape))
    blk_sliding.unload()

    idx_full = model.first_block_idx + cfg.layer_types.index("full_attention")
    blk_full = model.modules[idx_full]
    blk_full.load(device)
    if blk_full.attn.v_proj is not None:
        fail("Expected Gemma4 full attention to omit v_proj")
    x = torch.randn((1, 8, hidden_size), device = device, dtype = torch.half)
    with torch.inference_mode():
        y = blk_full.forward(x, params = {})
    print("[INFO] full block forward:", blk_full.key, tuple(y.shape))
    blk_full.unload()

    if cfg.enable_moe_block:
        idx_moe = model.first_block_idx
        blk_moe = model.modules[idx_moe]
        blk_moe.load(device)
        x = torch.randn((1, 8, hidden_size), device = device, dtype = torch.half)
        with torch.inference_mode():
            y = blk_moe.forward(x, params = {})
        print("[INFO] moe block forward:", blk_moe.key, tuple(y.shape))
        blk_moe.unload()

    vision_model.load(device)
    image = Image.new("RGB", (640, 480), (255, 0, 0))
    image_embedding = vision_model.get_image_embeddings(tokenizer, image)
    print("[INFO] image embeddings:", image_embedding.mm_length)
    if image_embedding.mm_length <= 0:
        fail("Vision tower produced no image embeddings")
    vision_model.unload()

    if args.full_load:
        reserve = None
        if args.reserve_per_device:
            reserve = [float(x) for x in args.reserve_per_device.split(",")]
        print("[INFO] attempting full model load")
        try:
            model.load(reserve_per_device = reserve)
            print("[INFO] full model load ok")
            model.unload()
        except Exception as e:
            print("[WARN] full model load failed:", repr(e))

    if args.multimodal_generation:
        image_path = args.multimodal_image
        if image_path is None:
            image_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "examples",
                "media",
                "cat.png",
            )
        if not os.path.exists(image_path):
            fail(f"Multimodal test image not found: {image_path}")

        reserve = None
        if args.reserve_per_device:
            reserve = [float(x) for x in args.reserve_per_device.split(",")]

        cache = Cache(model, max_num_tokens = 4096)
        vision_model.load(device)
        image_embedding = vision_model.get_image_embeddings(tokenizer, Image.open(image_path).convert("RGB"))
        prompt_ids = tokenizer.hf_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What animal is shown? Answer with one word."},
                    ],
                }
            ],
            add_generation_prompt = True,
            enable_thinking = False,
            embeddings = [image_embedding],
        )
        vision_model.unload()

        model.load(reserve_per_device = reserve)
        generator = Generator(model, cache, tokenizer)
        job = Job(
            input_ids = prompt_ids.cpu(),
            max_new_tokens = 8,
            stop_conditions = [tokenizer.eos_token_id, "<turn|>"],
            decode_special_tokens = True,
            embeddings = [image_embedding],
            sampler = GreedySampler(),
        )
        generator.enqueue(job)
        output_text = ""
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if result.get("stage") == "streaming":
                    output_text += result.get("text", "")
        print("[INFO] multimodal generation:", repr(output_text))
        if "cat" not in output_text.lower():
            fail(f"Unexpected multimodal generation output: {output_text!r}")
        model.unload()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PASS] gemma4 smoke checks complete")


if __name__ == "__main__":
    main()
