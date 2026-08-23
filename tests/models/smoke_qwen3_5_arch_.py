#!/usr/bin/env python3
"""
Lightweight smoke checks for Qwen3.5 architecture integration.

Checks:
1) Config/architecture resolution
2) Model graph construction
3) One linear-attn block forward
4) One full-attn block forward

Optional:
5) Attempt full model load (can fail on low VRAM for FP16 checkpoints)
6) --vision_cpu: load the vision component on CPU and extract image embeddings there
7) --vision_cpu --full_load: keep the CUDA text model resident while the vision tower runs on
   host RAM, assert zero transient/permanent vision VRAM allocation, and run greedy multimodal
   generation against the test image
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from PIL import Image

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def default_image() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "examples",
        "media",
        "cat.png",
    )


def run_vision_cpu(model, cfg, tokenizer, image_path, full_load, reserve):
    vision_model = Model.from_config(cfg, component = "vision")

    if full_load:
        # Text model stays resident; watch every device's live and peak allocation across the
        # CPU vision load to prove the vision tower never touches VRAM
        cache = Cache(model, max_num_tokens = 4096)
        print("[INFO] loading text model")
        try:
            model.load(reserve_per_device = reserve)
        except Exception as e:
            fail(f"text model load failed: {e!r}")

        num_devices = torch.cuda.device_count()
        baseline = {}
        for d in range(num_devices):
            torch.cuda.synchronize(d)
            baseline[d] = torch.cuda.memory_allocated(d)
            torch.cuda.reset_peak_memory_stats(d)
        print("[INFO] VRAM baseline:", {f"cuda:{k}": v for k, v in baseline.items()})
    else:
        cache = None

    print("[INFO] loading vision component on CPU")
    vision_model.load(device = "cpu")

    if not os.path.exists(image_path):
        fail(f"Test image not found: {image_path}")
    image_embedding = vision_model.get_image_embeddings(
        tokenizer, Image.open(image_path).convert("RGB")
    )
    if image_embedding.mm_length is None or image_embedding.mm_length <= 0:
        fail("Vision tower produced no image embeddings")
    ds_expected = len(cfg.vision.deepstack_visual_indexes)
    ds_actual = len(image_embedding.deepstack_embeddings or [])
    if ds_actual != ds_expected:
        fail(f"Deepstack count mismatch: {ds_actual} != {ds_expected}")
    print(f"[INFO] cpu image embeddings ok: mm_length {image_embedding.mm_length}, "
          f"deepstack {ds_actual}")

    if full_load:
        for d in range(num_devices):
            torch.cuda.synchronize(d)
            peak = torch.cuda.max_memory_allocated(d)
            live = torch.cuda.memory_allocated(d)
            if peak != baseline[d]:
                fail(f"cuda:{d} peak allocation changed during CPU vision pass: "
                     f"{peak} != baseline {baseline[d]}")
            if live != baseline[d]:
                fail(f"cuda:{d} live allocation leaked during CPU vision pass: "
                     f"{live} != baseline {baseline[d]}")
        print("[INFO] no transient or permanent vision VRAM allocation on any device")

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

    output_text = None
    if full_load:
        generator = Generator(model, cache, tokenizer)
        job = Job(
            input_ids = prompt_ids.cpu(),
            max_new_tokens = 8,
            stop_conditions = [tokenizer.eos_token_id],
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
        if not any(w in output_text.lower() for w in ("cat", "kitten", "feline", "tabby")):
            fail(f"Unexpected multimodal generation output: {output_text!r}")

    vision_model.unload()

    if full_load:
        model.unload()
        del cache


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev = False)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full_load", action="store_true")
    ap.add_argument("--reserve_per_device", default=None, help="e.g. 0.25,0.25")
    ap.add_argument("--vision_cpu", action="store_true",
                    help="run the vision component on CPU (Qwen 3.5 offload)")
    ap.add_argument("--multimodal_image", default=None)
    args = ap.parse_args()

    cfg = Config.from_directory(args.model_dir)
    print("[INFO] architecture:", cfg.architecture)
    if cfg.architecture not in ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"):
        fail(f"Unexpected architecture: {cfg.architecture}")

    model = Model.from_config(cfg)
    print("[INFO] model graph ok")

    # Verify split projection path exists for linear attention blocks
    first_block = model.modules[model.first_block_idx]
    if not hasattr(first_block.attn, "qkv_proj") and not hasattr(first_block.attn, "qkvz_proj"):
        fail("Expected Qwen3.5 linear attention projection attributes not found")

    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")

    device = torch.device(args.device)
    hidden_size = cfg.hidden_size

    # First linear-attn block
    idx_linear = model.first_block_idx
    blk_linear = model.modules[idx_linear]
    blk_linear.load(device)
    x = torch.randn((1, 8, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_linear.forward(x, params={})
    print("[INFO] linear block forward:", blk_linear.key, tuple(y.shape))
    blk_linear.unload()

    # First full-attn block
    if "full_attention" not in cfg.layer_types:
        fail("No full_attention layer found in layer_types")
    idx_full = model.first_block_idx + cfg.layer_types.index("full_attention")
    blk_full = model.modules[idx_full]
    blk_full.load(device)
    x = torch.randn((1, 8, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_full.forward(x, params={})
    print("[INFO] full block forward:", blk_full.key, tuple(y.shape))
    blk_full.unload()

    if args.vision_cpu:
        tokenizer = Tokenizer.from_config(cfg)
        image_path = args.multimodal_image or default_image()
        reserve = None
        if args.reserve_per_device:
            reserve = [float(v) for v in args.reserve_per_device.split(",")]
        run_vision_cpu(model, cfg, tokenizer, image_path, args.full_load, reserve)

    elif args.multimodal_image:
        fail("--multimodal_image requires --vision_cpu")

    elif args.full_load:
        reserve = None
        if args.reserve_per_device:
            reserve = [float(x) for x in args.reserve_per_device.split(",")]
        print("[INFO] attempting full model load")
        try:
            model.load(reserve_per_device=reserve)
            print("[INFO] full model load ok")
            model.unload()
        except Exception as e:
            print("[WARN] full model load failed:", repr(e))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PASS] qwen3.5 smoke checks complete")


if __name__ == "__main__":
    main()
