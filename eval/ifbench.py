from __future__ import annotations
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.util.file import disk_lru_cache
from exllamav3 import model_init, Generator, Job, ComboSampler
from exllamav3.util.progress import ProgressBar
import argparse
from pathlib import Path
import torch
import time, random, json
from pathlib import Path
# from datasets import load_dataset
from urllib import request
from collections import deque

# ! Doesn't work, removes trailing whitespace from questions, which breaks the eval script
# @disk_lru_cache("fetch_ifbench_test_data")
# def fetch_ifbench_test_data() -> list[dict]:
#     ds = load_dataset("allenai/IFBench_test", split = "train")
#     data = [row for row in ds]
#     print(f"Loaded {len(data)} IFBench test prompts from allenai/IFBench_test")
#     return data

@disk_lru_cache("fetch_ifbench_test_data_git")
def fetch_ifbench_test_data_git() -> (list[dict], list):
    url = "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl"
    with request.urlopen(url) as response:
        raw = response.read().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()], raw


def strip_reasoning(s: str) -> str:
    # TODO: Respect other think tags
    l = s.find("<think>")
    r = s.rfind("</think>")
    if l >= 0 and r >= 0:
        s = s[:l] + s[r+8:]
    elif r >= 0:
        s = s[r + 8:]
    return s.strip()


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.inference_mode()
def main(args):

    # Get the stuff
    ifbench, ifbench_raw = fetch_ifbench_test_data_git()
    if args.limit:
        ifbench = ifbench[:args.limit]

    # Initialize
    model, config, cache, tokenizer = model_init.init(args)
    generator = Generator(
        model = model,
        cache = cache,
        max_batch_size = args.max_batch_size,
        tokenizer = tokenizer,
        show_visualizer = args.visualize_cache,
    )
    sampler = model_init.get_arg_sampler(args)

    # Create prompts
    template_args = {}
    if args.think: template_args["enable_thinking"] = True
    if args.nothink: template_args["enable_thinking"] = False

    all_results: list[dict] = []
    with ProgressBar("Prompts", len(ifbench), transient = False) as progress:
        for idx, bp in enumerate(ifbench):
            prompt = bp["prompt"]
            all_results.append({ "prompt": prompt })
            input_ids = tokenizer.hf_chat_template(
                [{ "role": "user", "content": prompt }],
                add_generation_prompt = True,
                **template_args,
            )
            job = Job(
                input_ids = input_ids,
                max_new_tokens = args.max_tokens,
                stop_conditions = config.eos_token_id_list,
                sampler = sampler,
                identifier = idx,
                # max_rq_tokens = 512,
            )
            generator.enqueue(job)
            progress.update(idx + 1)

    # Generate
    tps_hist = deque()
    sampled_tokens = 0
    last_update = time.time()
    total_jobs = generator.num_remaining_jobs()
    done_jobs = 0
    with ProgressBar("Generating samples", total_jobs, transient = False) as progress:

        while generator.num_remaining_jobs():
            results = generator.iterate()

            # Some feedback
            now = time.time()
            if now > last_update + 1:
                tps = sampled_tokens / (now - last_update)
                sampled_tokens = 0
                tps_hist.append(tps)
                if len(tps_hist) > 3:
                    tps_hist.popleft()
                tps = round(sum(tps_hist) / len(tps_hist))
                num_pend = generator.num_pending_jobs()
                num_act = generator.num_active_jobs()
                print(f" -- pending: {num_pend:4}    active {num_act:4}   {tps:6} tokens/s", end = "")
                if num_act:
                    sjob = random.choice(generator.active_jobs)
                    snum = "#" + str(sjob.identifier)
                    ssamp = repr(sjob.full_completion)[-64:-1]
                    print(f"    sample from {snum:>4}: {ssamp}")
                else:
                    print()
                last_update = now

            # Collect results
            for result in results:
                if "token_ids" in result:
                    sampled_tokens += result["token_ids"].shape[-1]
                if result.get("eos"):
                    idx = result["identifier"]
                    completion = result["full_completion"]
                    if result["eos_reason"] == "max_new_tokens":
                        print(f" !! Job #{idx} exceeded token limit, ends in: {repr(completion)[-100:-1]}")
                    all_results[idx]["response"] = strip_reasoning(completion)
                    done_jobs += 1
                    progress.update(done_jobs)

    # Save results
    write_jsonl(all_results, args.output)
    print (f" -- Responses written to {args.output}")

    # Evaluate
    if args.eval:
        absfile = Path(args.output).resolve()
        print(" !! Sorry, eval is not implemented since IFBench is not available as a Python package. "
              "To evaluate, please clone the IFBench repo and run:")
        print(f"python run_eval.py --input_data=data/IFBench_test.jsonl --input_response_data={absfile} --output_dir=./eval/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run HumanEval evaluation")
    model_init.add_args(
        parser,
        add_sampling_args = True,
        default_cache_size = 65536,
        default_sampling_args = {
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_range": 1024,
            "min_p": 0.0,
            "top_k": 0,
            "top_p": 0.0,
            "adaptive_target": 1.0,
            "adaptive_decay": 0.9,
        }
    )
    parser.add_argument("-vis", "--visualize_cache", action = "store_true", help = "Show cache visualizer (slow)")
    parser.add_argument("-o", "--output", type = str, help = "Output .jsonl filename", required = True)
    parser.add_argument("-e", "--eval", action = "store_true", help = "Run evaluation script on output file after sampling")
    parser.add_argument("-mt", "--max_tokens", type = int, default = 16384, help = "Max number of tokens for each completion")
    parser.add_argument("-mbs", "--max_batch_size", type = int, default = 64, help = "Max batch size")
    parser.add_argument("-think", "--think", action = "store_true", help = "set template_arg enable_thinking=true")
    parser.add_argument("-nothink", "--nothink", action = "store_true", help = "set template_arg enable_thinking=false")
    parser.add_argument("-limit", "--limit", type = int, help = "Limit number of questions (creates incomplete results file)", default = 0)
    args = parser.parse_args()

    # Validate args
    directory = os.path.dirname(args.output)
    if os.path.exists(args.output):
        print(f" !! Warning: Output file exists and will be overwritten.")

    assert not (args.think and args.nothink), "Be nice"
    main(args)
