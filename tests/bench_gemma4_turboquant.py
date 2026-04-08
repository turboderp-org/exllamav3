#!/usr/bin/env python3
"""
Short benchmark harness for Gemma4 KV turboquant configurations.

Measures:
1) Cache footprint from cache layer metadata
2) First-run / second-run latency for repeated prompts
3) Optional multimodal short prompt latency sweep across SWA sizes

This is intentionally lightweight and reproducible, aimed at PR validation rather
than exhaustive serving benchmarks.
"""

import argparse
import json
import statistics
import os
import sys
import time

import torch
from PIL import Image

from exllamav3 import Cache, Config, Job, Model, Tokenizer
from exllamav3.generator import Generator
from exllamav3.generator.sampler import GreedySampler


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def mib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 2)


def cache_footprint_bytes(cache: Cache) -> int:
    return sum(layer.storage_size() + layer.overhead_size() for layer in cache.layers.values())


def count_shadow_pages(cache: Cache) -> int:
    total = 0
    for layer in cache.layers.values():
        shadow_pages = getattr(layer, "shadow_k_pages", None)
        if shadow_pages:
            total += len(shadow_pages)
    return total


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def first_or_none(values: list):
    return values[0] if values else None


def second_or_none(values: list):
    return values[1] if len(values) > 1 else None


def get_role_max_tokens(cache: Cache, role: str) -> int | None:
    role_tokens = [
        layer.max_num_tokens
        for layer in cache.layers.values()
        if getattr(layer, "cache_role", "default") == role
    ]
    if not role_tokens:
        return None
    return min(role_tokens)


def run_job(generator: Generator, job: Job, timeout_s: float) -> tuple[str, float]:
    generator.enqueue(job)
    output = ""
    t0 = time.perf_counter()
    while generator.num_remaining_jobs():
        if time.perf_counter() - t0 > timeout_s:
            fail(f"Benchmark job timed out after {timeout_s:.1f}s")
        for result in generator.iterate():
            if result.get("stage") == "streaming":
                output += result.get("text", "")
    return output, time.perf_counter() - t0


def run_repeated_jobs(generator: Generator, job_factory, repeats: int, timeout_s: float):
    jobs = []
    outputs = []
    times = []
    for _ in range(repeats):
        job = job_factory()
        out, dt = run_job(generator, job, timeout_s)
        jobs.append(job)
        outputs.append(out.strip())
        times.append(round(dt, 4))
    return jobs, outputs, times


def make_repeated_text_prompt() -> str:
    repeated_clause = "The black cat sits on the warm window sill and watches the garden birds. "
    return (
        "Read the passage and answer with exactly one word: cat.\n\n"
        + repeated_clause * 48
    )


def make_text_job(tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 8) -> Job:
    prompt_ids = tokenizer.hf_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt = True,
        enable_thinking = False,
    )
    return Job(
        input_ids = prompt_ids.cpu(),
        max_new_tokens = max_new_tokens,
        stop_conditions = [tokenizer.eos_token_id, "<turn|>"],
        decode_special_tokens = True,
        sampler = GreedySampler(),
    )


def make_mm_job(tokenizer: Tokenizer, embeddings: list, question: str, max_new_tokens: int = 16) -> Job:
    prompt_ids = tokenizer.hf_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in embeddings] + [{"type": "text", "text": question}],
            }
        ],
        add_generation_prompt = True,
        enable_thinking = False,
        embeddings = embeddings,
    )
    return Job(
        input_ids = prompt_ids.cpu(),
        max_new_tokens = max_new_tokens,
        stop_conditions = [tokenizer.eos_token_id, "<turn|>"],
        decode_special_tokens = True,
        embeddings = embeddings,
        sampler = GreedySampler(),
    )


def load_demo_embeddings(cfg: Config, tokenizer: Tokenizer, cat_image_path: str, fruit_image_path: str):
    vision_model = Model.from_config(cfg, component = "vision")
    vision_model.load("cuda:0")
    cat = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image_path).convert("RGB"))
    cat_dup = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image_path).convert("RGB"))
    fruit = vision_model.get_image_embeddings(tokenizer, Image.open(fruit_image_path).convert("RGB"))
    vision_model.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cat, cat_dup, fruit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required = True)
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--reserve_per_device", default = None, help = "e.g. 0.5,0.5")
    ap.add_argument("--cache_bits", default = "4", help = "Either single value or k_bits,v_bits")
    ap.add_argument("--cache_size", type = int, default = 1024)
    ap.add_argument(
        "--swa_sizes",
        default = "1024,768",
        help = "Comma separated requested SWA cache sizes in tokens. Values are page-aligned internally.",
    )
    ap.add_argument("--cat_image", default = None)
    ap.add_argument("--fruit_image", default = None)
    ap.add_argument("--job_timeout_s", type = float, default = 120.0)
    ap.add_argument("--text_repeats", type = int, default = 2)
    ap.add_argument("--mm_repeats", type = int, default = 2)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        fail("CUDA is required for this benchmark")

    cat_image = args.cat_image or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "doc",
        "cat.png",
    )
    if not os.path.exists(cat_image):
        fail(f"Required image not found: {cat_image}")
    fruit_image = args.fruit_image or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "examples",
        "media",
        "strawberry.png",
    )
    if not os.path.exists(fruit_image):
        fail(f"Required image not found: {fruit_image}")

    reserve = None
    if args.reserve_per_device:
        reserve = [float(x) for x in args.reserve_per_device.split(",")]

    split = [int(bits) for bits in args.cache_bits.split(",")]
    if len(split) == 1:
        k_bits = v_bits = split[0]
    elif len(split) == 2:
        k_bits, v_bits = split
    else:
        fail("--cache_bits must be 'bits' or 'k_bits,v_bits'")

    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg)
    tokenizer = Tokenizer.from_config(cfg)
    print(f"[INFO] loading vision embeddings for {args.model_dir}", flush = True)
    cat, cat_dup, fruit = load_demo_embeddings(cfg, tokenizer, cat_image, fruit_image)

    results = []
    for swa_size in [int(x.strip()) for x in args.swa_sizes.split(",") if x.strip()]:
        print(f"[INFO] bench start swa_cache_size={swa_size}", flush = True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        cache = Cache(
            model,
            max_num_tokens = args.cache_size,
            layer_type = model.caps.get("quantized_kv_cache_layer"),
            k_bits = k_bits,
            v_bits = v_bits,
            swa_max_num_tokens = swa_size,
        )

        print(f"[INFO] loading text model swa_cache_size={swa_size}", flush = True)
        model.load(reserve_per_device = reserve)
        generator = Generator(model, cache, tokenizer)
        effective_full_cache_size = get_role_max_tokens(cache, "full")
        effective_swa_cache_size = get_role_max_tokens(cache, "swa")
        atomic_mm_prefill = bool(model.caps.get("atomic_mm_prefill", False))

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        repeated_text_prompt = make_repeated_text_prompt()
        text_jobs = []
        text_outputs = []
        text_times = []
        if args.text_repeats > 0:
            text_jobs, text_outputs, text_times = run_repeated_jobs(
                generator,
                lambda: make_text_job(tokenizer, repeated_text_prompt, max_new_tokens = 4),
                args.text_repeats,
                args.job_timeout_s,
            )
        for idx, dt in enumerate(text_times, start = 1):
            print(f"[INFO] swa={swa_size} text-repeat #{idx} done in {dt:.4f}s", flush = True)
        shadow_pages_after_text = count_shadow_pages(cache)

        repeated_jobs = []
        repeated_outputs = []
        repeated_times = []
        if args.mm_repeats > 0:
            repeated_jobs, repeated_outputs, repeated_times = run_repeated_jobs(
                generator,
                lambda: make_mm_job(
                    tokenizer,
                    [cat, cat_dup],
                    "Do the two images show the same subject? Answer yes or no.",
                ),
                args.mm_repeats,
                args.job_timeout_s,
            )
        for idx, dt in enumerate(repeated_times, start = 1):
            print(f"[INFO] swa={swa_size} duplicate #{idx} done in {dt:.4f}s", flush = True)

        pair_job = make_mm_job(
            tokenizer,
            [cat, fruit],
            "Which image shows an animal? Answer first or second.",
        )
        pair_out, pair_t = run_job(generator, pair_job, args.job_timeout_s)
        print(f"[INFO] swa={swa_size} pair-animal done in {pair_t:.4f}s", flush = True)
        shadow_pages_after_mm = count_shadow_pages(cache)

        peak_alloc = 0
        peak_reserved = 0
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()

        result = {
            "swa_cache_size": swa_size,
            "effective_full_cache_size": effective_full_cache_size,
            "effective_swa_cache_size": effective_swa_cache_size,
            "cache_footprint_mib": round(mib(cache_footprint_bytes(cache)), 2),
            "atomic_mm_prefill": atomic_mm_prefill,
            "text_repeat_times_s": text_times,
            "text_repeat_avg_s": safe_mean(text_times),
            "text_repeat_outputs": text_outputs,
            "text_repeat_cached_pages": [job.cached_pages for job in text_jobs],
            "text_repeat_cached_tokens": [job.cached_tokens for job in text_jobs],
            "text_repeat_first_s": first_or_none(text_times),
            "text_repeat_second_s": second_or_none(text_times),
            "text_repeat_first_out": first_or_none(text_outputs),
            "text_repeat_second_out": second_or_none(text_outputs),
            "dup_same_times_s": repeated_times,
            "dup_same_avg_s": safe_mean(repeated_times),
            "dup_same_outputs": repeated_outputs,
            "dup_same_cached_pages": [job.cached_pages for job in repeated_jobs],
            "dup_same_cached_tokens": [job.cached_tokens for job in repeated_jobs],
            "dup_same_first_s": first_or_none(repeated_times),
            "dup_same_second_s": second_or_none(repeated_times),
            "pair_animal_s": round(pair_t, 4),
            "dup_same_first_out": first_or_none(repeated_outputs),
            "dup_same_second_out": second_or_none(repeated_outputs),
            "pair_animal_out": pair_out.strip(),
            "dup_same_first_cached_pages": first_or_none([job.cached_pages for job in repeated_jobs]),
            "dup_same_first_cached_tokens": first_or_none([job.cached_tokens for job in repeated_jobs]),
            "dup_same_second_cached_pages": second_or_none([job.cached_pages for job in repeated_jobs]),
            "dup_same_second_cached_tokens": second_or_none([job.cached_tokens for job in repeated_jobs]),
            "cached_swa_snapshots": len(getattr(generator.pagetable, "cached_full_to_swa_pages", {})),
            "shadow_pages_after_text": shadow_pages_after_text,
            "shadow_pages_after_mm": shadow_pages_after_mm,
            "shadow_pages": shadow_pages_after_mm,
            "peak_alloc_mib": round(mib(peak_alloc), 2),
            "peak_reserved_mib": round(mib(peak_reserved), 2),
        }
        print(json.dumps(result, ensure_ascii = False), flush = True)
        results.append(result)

        model.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[SUMMARY]")
    print(json.dumps(results, indent = 2, ensure_ascii = False), flush = True)


if __name__ == "__main__":
    main()
