#!/usr/bin/env python3
"""
Objective multimodal smoke checks for Gemma4 image understanding.

Checks:
1) Single-image recognition
2) Ordered two-image reasoning
3) Duplicate image consistency
4) Optional quantized KV-cache path using the model-specific cache layer
"""

import argparse
import os
import sys

import torch
from PIL import Image

from exllamav3 import Cache, Config, Job, Model, Tokenizer
from exllamav3.generator import Generator
from exllamav3.generator.sampler import GreedySampler


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def run_prompt(
    generator: Generator,
    tokenizer: Tokenizer,
    embeddings: list,
    question: str,
) -> str:
    prompt_ids = tokenizer.hf_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"} for _ in embeddings
                ] + [{"type": "text", "text": question}],
            }
        ],
        add_generation_prompt = True,
        enable_thinking = False,
        embeddings = embeddings,
    )
    job = Job(
        input_ids = prompt_ids.cpu(),
        max_new_tokens = 16,
        stop_conditions = [tokenizer.eos_token_id, "<turn|>"],
        decode_special_tokens = True,
        embeddings = embeddings,
        sampler = GreedySampler(),
    )
    generator.enqueue(job)
    output = ""
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result.get("stage") == "streaming":
                output += result.get("text", "")
    return output


def expect_contains(label: str, output: str, expected: list[str]) -> None:
    n = normalize(output)
    if not any(e in n for e in expected):
        fail(f"{label}: expected one of {expected!r}, got {output!r}")
    print(f"[INFO] {label}: {output!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reserve_per_device", default=None, help="e.g. 0.5,0.5")
    ap.add_argument("--cache_bits", default=None, help="Either single value or k_bits,v_bits")
    ap.add_argument("--swa_cache_size", type=int, default=None)
    ap.add_argument("--cat_image", default=None)
    ap.add_argument("--fruit_image", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")

    cat_image = args.cat_image or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "doc",
        "cat.png",
    )
    fruit_image = args.fruit_image or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples",
        "media",
        "strawberry.png",
    )
    for path in [cat_image, fruit_image]:
        if not os.path.exists(path):
            fail(f"Required image not found: {path}")

    reserve = None
    if args.reserve_per_device:
        reserve = [float(x) for x in args.reserve_per_device.split(",")]

    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg)
    vision_model = Model.from_config(cfg, component="vision")
    tokenizer = Tokenizer.from_config(cfg)

    cache_kwargs = {}
    if args.cache_bits is not None:
        split = [int(bits) for bits in args.cache_bits.split(",")]
        if len(split) == 1:
            k_bits = v_bits = split[0]
        elif len(split) == 2:
            k_bits, v_bits = split
        else:
            fail("--cache_bits must be 'bits' or 'k_bits,v_bits'")
        cache_kwargs = {
            "layer_type": model.caps.get("quantized_kv_cache_layer"),
            "k_bits": k_bits,
            "v_bits": v_bits,
        }
    if args.swa_cache_size is not None:
        cache_kwargs["swa_max_num_tokens"] = args.swa_cache_size
    cache = Cache(model, max_num_tokens=1024, **cache_kwargs)

    device = torch.device(args.device)
    vision_model.load(device)
    cat = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image).convert("RGB"))
    cat_dup = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image).convert("RGB"))
    fruit = vision_model.get_image_embeddings(tokenizer, Image.open(fruit_image).convert("RGB"))
    vision_model.unload()

    model.load(reserve_per_device=reserve)
    generator = Generator(model, cache, tokenizer)

    expect_contains(
        "single cat",
        run_prompt(generator, tokenizer, [cat], "What animal is shown? Answer with one word."),
        ["cat"],
    )
    expect_contains(
        "single fruit",
        run_prompt(generator, tokenizer, [fruit], "What fruit is shown? Answer with one word."),
        ["strawberry"],
    )

    pair = [cat, fruit]
    expect_contains(
        "animal position",
        run_prompt(generator, tokenizer, pair, "Which image shows an animal? Answer first or second."),
        ["first"],
    )
    expect_contains(
        "fruit position",
        run_prompt(generator, tokenizer, pair, "Which image shows a fruit? Answer first or second."),
        ["second"],
    )
    expect_contains(
        "first animal yesno",
        run_prompt(generator, tokenizer, pair, "Does the first image show an animal? Answer yes or no."),
        ["yes"],
    )
    expect_contains(
        "second fruit yesno",
        run_prompt(generator, tokenizer, pair, "Does the second image show a fruit? Answer yes or no."),
        ["yes"],
    )
    expect_contains(
        "same subject no",
        run_prompt(generator, tokenizer, pair, "Do the two images show the same subject? Answer yes or no."),
        ["no"],
    )
    expect_contains(
        "duplicate subject yes",
        run_prompt(
            generator,
            tokenizer,
            [cat, cat_dup],
            "Do the two images show the same subject? Answer yes or no.",
        ),
        ["yes"],
    )

    model.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PASS] gemma4 multimodal smoke checks complete")


if __name__ == "__main__":
    main()
