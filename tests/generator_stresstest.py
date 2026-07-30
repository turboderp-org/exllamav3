"""
Exercise the Generator's paged-cache allocation and defragmentation logic.

The test continuously varies the queue depth. Prompts contain increasing
sequences of integers, and each completed sequence is checked for correctness.
"""

import argparse
import os
import random
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3 import ArgmaxSampler, Generator, Job, model_init


col_default = "\u001b[0m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"

prefixes = ["All the numbers: ", "It never ends: ", "Counting forever: "]


def is_consecutive_integers(text: str) -> bool:
    numbers = [int(value.strip()) for value in text.split(",")]
    return all(b == a + 1 for a, b in zip(numbers, numbers[1:]))


def start_new_job(args, generator, tokenizer, suffix, rng):
    prefix = rng.choice(prefixes)
    prompt_length = rng.randint(*args.prompt_len)
    prompt = (prefix + suffix)[:prompt_length]
    prompt = prompt[:prompt.rfind(",") + 1]
    input_ids = tokenizer.encode(prompt, add_bos = True)
    job = Job(
        input_ids = input_ids,
        max_new_tokens = rng.randint(*args.completion_len),
        sampler = ArgmaxSampler(),
        identifier = prompt,
        max_rq_tokens = args.max_rq_tokens,
    )
    generator.enqueue(job)


def check_result(result, num_pending, num_active, total_tps):
    cached_tokens = result["cached_tokens"]
    cached_pages = result["cached_pages"]
    print(
        f"{str(result['job']):20}  pending: {num_pending:3}  active: {num_active:3}  "
        f"cached_p: {cached_pages:3}  cached_t: {cached_tokens:5}  "
        f"tps: {total_tps:8.2f}  -  ",
        end = "",
    )

    prompt = result["identifier"]
    full = prompt + result["full_completion"]
    prompt = prompt.partition(": ")[2]
    full = full.partition(": ")[2]
    full = full[:full.rfind(",")]
    try:
        ok = is_consecutive_integers(full)
    except ValueError:
        ok = False

    if ok:
        print("OK!")
    else:
        print("Sus!")
        print("--------")
        print(col_green + prompt + col_red + full[len(prompt):] + col_default)
        print("--------")

    return ok


def iterate(
    generator,
    iteration,
    validate_interval,
    throughput_tokens,
    throughput_time,
    passed_results,
    suspicious_results,
    completed_serials,
):
    if validate_interval and iteration % validate_interval == 0:
        generator.pagetable.validate_pagetable(generator.active_jobs)

    num_active = generator.num_active_jobs()
    num_pending = generator.num_pending_jobs()
    start_time = time.perf_counter()
    results = generator.iterate()
    throughput_time += time.perf_counter() - start_time
    throughput_tokens += sum(
        result["token_ids"].shape[-1]
        for result in results
        if result.get("token_ids") is not None
    )
    total_tps = throughput_tokens / throughput_time if throughput_time else 0.0

    reset_throughput = False
    for result in results:
        requeued = result.get("requeue", False)
        if requeued:
            print(f"{str(result['job'])}  requeued")
        if result["eos"]:
            reset_throughput = True
            serial = result["serial"]
            if not requeued and serial not in completed_serials:
                completed_serials.add(serial)
                if check_result(result, num_pending, num_active, total_tps):
                    passed_results += 1
                else:
                    suspicious_results += 1

    if reset_throughput:
        throughput_tokens = 0
        throughput_time = 0.0
    return throughput_tokens, throughput_time, passed_results, suspicious_results


def run_stress_test(args, generator, tokenizer):
    rng = random.Random(args.seed)
    suffix = ", ".join(str(i) for i in range(args.prompt_len[1]))
    next_target_q_depth = 0
    depth_0_interval = args.force_depth_0_interval
    last_defrag_serial = generator.pagetable.last_defrag_serial
    iteration = 0
    throughput_tokens = 0
    throughput_time = 0.0
    jobs_started = 0
    passed_results = 0
    suspicious_results = 0
    completed_serials = set()

    while True:
        if generator.num_remaining_jobs() > next_target_q_depth:
            print(f" - Generating, target depth {next_target_q_depth}")
        while generator.num_remaining_jobs() > next_target_q_depth:
            iteration += 1
            (
                throughput_tokens,
                throughput_time,
                passed_results,
                suspicious_results,
            ) = iterate(
                generator,
                iteration,
                args.validate_interval,
                throughput_tokens,
                throughput_time,
                passed_results,
                suspicious_results,
                completed_serials,
            )
            if last_defrag_serial != generator.pagetable.last_defrag_serial:
                print(" !! DEFRAG")
                last_defrag_serial = generator.pagetable.last_defrag_serial

        if args.num_jobs is not None and jobs_started >= args.num_jobs:
            if generator.num_remaining_jobs() == 0:
                break
            next_target_q_depth = 0
            continue

        next_target_q_depth = rng.randint(
            args.target_q_depth[0] + 1,
            args.target_q_depth[1],
        )
        if args.num_jobs is not None:
            remaining_job_budget = args.num_jobs - jobs_started
            next_target_q_depth = min(
                next_target_q_depth,
                generator.num_remaining_jobs() + remaining_job_budget,
            )

        if generator.num_remaining_jobs() < next_target_q_depth:
            print(f" - Creating jobs, target depth {next_target_q_depth}")
        while generator.num_remaining_jobs() < next_target_q_depth:
            start_new_job(args, generator, tokenizer, suffix, rng)
            jobs_started += 1

        if args.num_jobs is not None and jobs_started >= args.num_jobs:
            next_target_q_depth = 0
            continue

        depth_0_interval -= 1
        if depth_0_interval == 0:
            next_target_q_depth = 0
            depth_0_interval = args.force_depth_0_interval
        else:
            next_target_q_depth = rng.randint(
                args.target_q_depth[0],
                generator.num_remaining_jobs() - 1,
            )

    print()
    print(f" -- Completed jobs: {len(completed_serials)}")
    print(f" -- Passed results: {passed_results}")
    print(f" -- Suspicious results: {suspicious_results}")


@torch.inference_mode()
def main(args):
    (
        model,
        _,
        cache,
        tokenizer,
        draft_model,
        _,
        draft_cache,
    ) = model_init.init(args)
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        max_batch_size = args.autosplit_max_batch_size,
        max_chunk_size = args.max_chunk_size,
        draft_model = draft_model,
        draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens,
        show_visualizer = args.visualize_cache,
    )

    bpw_layer, bpw_head, _ = model.get_storage_info()
    print(f" -- Model: {args.model_dir}")
    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")
    print(f" -- Cache size: {args.cache_size:,} tokens")
    print()

    run_stress_test(args, generator, tokenizer)


def validate_args(parser, args):
    for name in ("prompt_len", "completion_len"):
        minimum, maximum = getattr(args, name)
        if minimum <= 0 or maximum < minimum:
            parser.error(f"--{name} must be two positive, ascending values")
    minimum_prompt_len = max(len(prefix) for prefix in prefixes) + 2
    if args.prompt_len[0] < minimum_prompt_len:
        parser.error(f"--prompt_len MIN must be at least {minimum_prompt_len}")
    if args.target_q_depth[0] < 0 or args.target_q_depth[1] <= args.target_q_depth[0]:
        parser.error("--target_q_depth must be two nonnegative, ascending values")
    if args.force_depth_0_interval <= 0:
        parser.error("--force_depth_0_interval must be positive")
    if args.validate_interval < 0:
        parser.error("--validate_interval cannot be negative")
    if args.max_rq_tokens <= 0:
        parser.error("--max_rq_tokens must be positive")
    if args.max_chunk_size <= 0:
        parser.error("--max_chunk_size must be positive")
    if args.num_jobs is not None and args.num_jobs <= 0:
        parser.error("--num_jobs must be positive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__, allow_abbrev = False)
    model_init.add_args(
        parser,
        cache = True,
        add_draft_model_args = True,
        default_cache_size = 16384,
        default_autosplit_max_batch_size = 32,
    )
    parser.add_argument("-pl", "--prompt_len", type = int, nargs = 2, metavar = ("MIN", "MAX"), default = (50, 4096), help = "Prompt length range in characters (default: 50 4096)")
    parser.add_argument("-cl", "--completion_len", type = int, nargs = 2, metavar = ("MIN", "MAX"), default = (50, 768), help = "Completion length range in tokens (default: 50 768)")
    parser.add_argument("-qd", "--target_q_depth", type = int, nargs = 2, metavar = ("MIN", "MAX"), default = (0, 25), help = "Queue-depth range (default: 0 25)")
    parser.add_argument("-f0", "--force_depth_0_interval", type = int, default = 3, help = "Drain the queue after this many depth changes (default: 3)")
    parser.add_argument("-vi", "--validate_interval", type = int, default = 3000, help = "Validate the page table every N iterations; 0 disables (default: 3000)")
    parser.add_argument("-mrt", "--max_rq_tokens", type = int, default = 512, help = "Maximum result-queue tokens per job (default: 512)")
    parser.add_argument("-chunk", "--max_chunk_size", type = int, default = 2048, help = "Maximum prefill chunk size (default: 2048)")
    parser.add_argument("-seed", "--seed", type = int, default = 0, help = "Random seed (default: 0)")
    parser.add_argument("-num", "--num_jobs", type = int, help = "Stop after this many jobs; default is unlimited")
    parser.add_argument("-vis", "--visualize_cache", action = "store_true", help = "Show cache visualizer (slow)")
    args = parser.parse_args()
    validate_args(parser, args)
    main(args)
