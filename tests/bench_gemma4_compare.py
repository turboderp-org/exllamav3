#!/usr/bin/env python3
"""
Compare Gemma4 support-only baseline vs current turboquant path.

This benchmark intentionally follows the lightweight PR-validation style of
`tests/bench_gemma4_turboquant.py`, but runs each comparison target in an
isolated subprocess so different exllamav3 trees can be compared safely.

Modes:
1) compare mode (default): spawn one subprocess per repo/mode and summarize
2) single mode: run one benchmark target and print one JSON result
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file = sys.stderr)
    sys.exit(1)


def parse_json_line(stdout: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("No JSON result line found")


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator in (None, 0) or denominator in (None, 0):
        return None
    return round(numerator / denominator, 4)


def run_subprocess(args, repo_root: str, mode: str) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single",
        "--repo_root", repo_root,
        "--mode", mode,
        "--model_dir", args.model_dir,
        "--cache_bits", args.cache_bits,
        "--cache_size", str(args.cache_size),
        "--job_timeout_s", str(args.job_timeout_s),
        "--text_repeats", str(args.text_repeats),
        "--mm_repeats", str(args.mm_repeats),
    ]
    if args.reserve_per_device:
        cmd += ["--reserve_per_device", args.reserve_per_device]
    if args.swa_cache_size is not None:
        cmd += ["--swa_cache_size", str(args.swa_cache_size)]
    if args.cat_image:
        cmd += ["--cat_image", args.cat_image]
    if args.fruit_image:
        cmd += ["--fruit_image", args.fruit_image]

    proc = subprocess.run(
        cmd,
        text = True,
        capture_output = True,
        env = os.environ.copy(),
    )

    result = {
        "mode": mode,
        "repo_root": repo_root,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
    }
    if proc.returncode == 0:
        try:
            payload = parse_json_line(proc.stdout)
            result.update(payload)
            result["status"] = payload.get("status", "ok")
        except Exception as exc:  # pragma: no cover - benchmark plumbing
            result["status"] = "error"
            result["parse_error"] = str(exc)
    else:
        result["status"] = "error"

    return result


def run_single(args) -> dict:
    sys.path.insert(0, args.repo_root)

    from PIL import Image
    import torch
    from exllamav3 import Cache, Config, Job, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator import Generator
    from exllamav3.generator.sampler import GreedySampler

    def mib(num_bytes: int) -> float:
        return num_bytes / (1024 ** 2)

    def cache_footprint_bytes(cache: Cache) -> int:
        return sum(layer.storage_size() + layer.overhead_size() for layer in cache.layers.values())

    def run_job(generator: Generator, job: Job) -> tuple[str, float]:
        generator.enqueue(job)
        output = ""
        t0 = time.perf_counter()
        while generator.num_remaining_jobs():
            if time.perf_counter() - t0 > args.job_timeout_s:
                fail(f"Benchmark job timed out after {args.job_timeout_s:.1f}s")
            for result in generator.iterate():
                if result.get("stage") == "streaming":
                    output += result.get("text", "")
        return output.strip(), round(time.perf_counter() - t0, 4)

    def run_repeated_jobs(generator: Generator, job_factory, repeats: int):
        outputs = []
        times = []
        for _ in range(repeats):
            out, dt = run_job(generator, job_factory())
            outputs.append(out)
            times.append(dt)
        return outputs, times

    def make_repeated_text_prompt() -> str:
        repeated_clause = "The black cat sits on the warm window sill and watches the garden birds. "
        return (
            "Read the passage and answer with exactly one word: cat.\n\n"
            + repeated_clause * 48
        )

    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg)
    tokenizer = Tokenizer.from_config(cfg)

    cat_image = args.cat_image or os.path.join(args.repo_root, "doc", "cat.png")
    fruit_image = args.fruit_image or os.path.join(args.repo_root, "examples", "media", "strawberry.png")
    if not os.path.exists(cat_image):
        fail(f"Required image not found: {cat_image}")
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

    vision_model = Model.from_config(cfg, component = "vision")
    vision_model.load("cuda:0")
    cat = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image).convert("RGB"))
    cat_dup = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image).convert("RGB"))
    fruit = vision_model.get_image_embeddings(tokenizer, Image.open(fruit_image).convert("RGB"))
    vision_model.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.mode == "support_default":
        cache = Cache(
            model,
            max_num_tokens = args.cache_size,
        )
    elif args.mode == "support_generic_quant":
        cache = Cache(
            model,
            max_num_tokens = args.cache_size,
            layer_type = CacheLayer_quant,
            k_bits = k_bits,
            v_bits = v_bits,
        )
    elif args.mode == "current_turboquant":
        cache = Cache(
            model,
            max_num_tokens = args.cache_size,
            layer_type = model.caps.get("quantized_kv_cache_layer"),
            k_bits = k_bits,
            v_bits = v_bits,
            swa_max_num_tokens = args.swa_cache_size,
        )
    else:
        fail(f"Unknown mode: {args.mode}")

    model.load(reserve_per_device = reserve)
    generator = Generator(model, cache, tokenizer)

    def make_text_job() -> Job:
        prompt_ids = tokenizer.hf_chat_template(
            [{"role": "user", "content": make_repeated_text_prompt()}],
            add_generation_prompt = True,
            enable_thinking = False,
        )
        return Job(
            input_ids = prompt_ids.cpu(),
            max_new_tokens = 4,
            stop_conditions = [tokenizer.eos_token_id, "<turn|>"],
            decode_special_tokens = True,
            sampler = GreedySampler(),
        )

    def make_mm_job(embeddings: list, question: str) -> Job:
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
            max_new_tokens = 16,
            stop_conditions = [tokenizer.eos_token_id, "<turn|>"],
            decode_special_tokens = True,
            embeddings = embeddings,
            sampler = GreedySampler(),
        )

    text_outputs, text_times = run_repeated_jobs(generator, make_text_job, args.text_repeats)

    dup_outputs = []
    dup_times = []
    pair_output = None
    pair_time = None
    if args.mm_repeats > 0:
        dup_outputs, dup_times = run_repeated_jobs(
            generator,
            lambda: make_mm_job([cat, cat_dup], "Do the two images show the same subject? Answer yes or no."),
            args.mm_repeats,
        )
        pair_output, pair_time = run_job(
            generator,
            make_mm_job([cat, fruit], "Which image shows an animal? Answer first or second."),
        )

    result = {
        "status": "ok",
        "mode": args.mode,
        "repo_root": args.repo_root,
        "cache_footprint_mib": round(mib(cache_footprint_bytes(cache)), 2),
        "text_repeat_times_s": text_times,
        "dup_same_times_s": dup_times,
        "pair_animal_s": pair_time,
        "text_repeat_outputs": text_outputs,
        "dup_same_outputs": dup_outputs,
        "pair_animal_out": pair_output,
    }
    print(json.dumps(result), flush = True)

    model.unload()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action = "store_true")
    ap.add_argument("--repo_root", default = str(REPO_ROOT))
    ap.add_argument("--baseline_repo_root", default = None)
    ap.add_argument("--current_repo_root", default = str(REPO_ROOT))
    ap.add_argument("--model_dir", required = True)
    ap.add_argument("--reserve_per_device", default = None)
    ap.add_argument("--cache_bits", default = "4")
    ap.add_argument("--cache_size", type = int, default = 4096)
    ap.add_argument("--swa_cache_size", type = int, default = 1024)
    ap.add_argument("--job_timeout_s", type = float, default = 120.0)
    ap.add_argument("--text_repeats", type = int, default = 2)
    ap.add_argument("--mm_repeats", type = int, default = 2)
    ap.add_argument("--cat_image", default = None)
    ap.add_argument("--fruit_image", default = None)
    ap.add_argument("--mode", choices = ["support_default", "support_generic_quant", "current_turboquant"])
    args = ap.parse_args()

    if args.single:
        try:
            run_single(args)
        except Exception as exc:  # pragma: no cover - benchmark plumbing
            payload = {
                "status": "error",
                "mode": args.mode,
                "repo_root": args.repo_root,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc().splitlines()[-20:],
            }
            print(json.dumps(payload), flush = True)
            raise
        return

    if not args.baseline_repo_root:
        fail("--baseline_repo_root is required in compare mode")

    baseline = run_subprocess(args, args.baseline_repo_root, "support_default")
    current = run_subprocess(args, args.current_repo_root, "current_turboquant")
    summary = {
        "baseline": baseline,
        "current": current,
    }
    if baseline.get("status") == "ok" and current.get("status") == "ok":
        baseline_text = baseline.get("text_repeat_times_s", [])
        current_text = current.get("text_repeat_times_s", [])
        baseline_dup = baseline.get("dup_same_times_s", [])
        current_dup = current.get("dup_same_times_s", [])
        baseline_cache = baseline.get("cache_footprint_mib")
        current_cache = current.get("cache_footprint_mib")
        comparison = {
            "cache_ratio_baseline_over_current": ratio(baseline_cache, current_cache),
            "cache_reduction_pct": round((1.0 - current_cache / baseline_cache) * 100.0, 2) if baseline_cache and current_cache else None,
            "text_first_speedup": ratio(baseline_text[0] if baseline_text else None, current_text[0] if current_text else None),
            "text_second_speedup": ratio(baseline_text[1] if len(baseline_text) > 1 else None, current_text[1] if len(current_text) > 1 else None),
            "dup_same_first_speedup": ratio(baseline_dup[0] if baseline_dup else None, current_dup[0] if current_dup else None),
            "dup_same_second_speedup": ratio(baseline_dup[1] if len(baseline_dup) > 1 else None, current_dup[1] if len(current_dup) > 1 else None),
            "pair_animal_speedup": ratio(baseline.get("pair_animal_s"), current.get("pair_animal_s")),
        }
        summary["comparison"] = comparison
    print(json.dumps(summary, indent = 2))


if __name__ == "__main__":
    main()
