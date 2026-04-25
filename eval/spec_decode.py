import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Generator, Job, model_init, GreedySampler, TopPSampler
import argparse
import torch
import json
from tabulate import tabulate

# ANSI codes
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_magenta = "\u001b[35;1m"
col_red = "\u001b[31;1m"
col_green = "\u001b[32;1m"  # Green

prompt_files = [
    ("Agentic, code", "agentic_code_01.json", True),
    ("Agentic, code", "agentic_code_05.json", True),
    ("Agentic, code", "agentic_code_10.json", True),
    ("Agentic, code", "agentic_code_20.json", True),
    ("Agentic, code", "agentic_code_29.json", True),
    ("Agentic, curl", "agentic_curl_05.json", True),
    ("Agentic, curl", "agentic_curl_10.json", True),
    ("Agentic, curl", "agentic_curl_15.json", True),
    ("Agentic, curl", "agentic_curl_16.json", True),
    ("Creative", "creative_01.json", False),
    ("Creative", "creative_02.json", False),
    ("Creative (reasoning)", "creative_01.json", True),
    ("Creative (reasoning)", "creative_02.json", True),
    ("Creative (reasoning)", "creative_03.json", True),
    ("Translation", "translate_01.json", False),
    ("Translation", "translate_02.json", False),
    ("Translation (reasoning)", "translate_01.json", True),
    ("Translation (reasoning)", "translate_02.json", True),
    ("Coding", "coding_01.json", False),
    ("Coding", "coding_02.json", False),
    ("Coding", "coding_03.json", False),
]


def measure(generator, tokenizer, sampler, max_new_tokens):
    path = os.path.dirname(os.path.abspath(__file__))
    all_results = {}

    for category, filename, think in prompt_files:
        with open(os.path.join(path, "prompts", filename), "r") as f:
            data = json.load(f)

        for msg in data["messages"]:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        func["arguments"] = json.loads(args)

        prompt_ids = tokenizer.hf_chat_template(
            messages = data["messages"],
            add_generation_prompt = True,
            functions = data.get("functions"),
            tools = data.get("tools"),
            enable_thinking = think
        )

        job = Job(
            input_ids = prompt_ids,
            max_new_tokens = max_new_tokens,
            stop_conditions = generator.model.config.eos_token_id_list,
            sampler = sampler,
        )
        generator.enqueue(job)

        while generator.num_remaining_jobs():
            results = generator.iterate()
            for result in results:
                if result["stage"] == "prefill":
                    curr_progress = result["curr_progress"]
                    max_progress = result["max_progress"]
                    print(f"Prefill: {curr_progress:,} / {max_progress:,}...")
                text = result.get("text", "")
                print(text, end = "", flush = True)
                if result.get("eos"):
                    print("\n--------")
                    break

        new_tokens = result["new_tokens"]
        gen_tps = new_tokens / result["time_generate"]
        if "accepted_draft_tokens" in result:
            dacc = result["accepted_draft_tokens"]
            drej = result["rejected_draft_tokens"]
            dtot = dacc + drej
            acc_rate = dacc / dtot if dtot else 0
        else:
            acc_rate = None

        if category not in all_results:
            all_results[category] = []
        all_results[category].append({
            "new_tokens": new_tokens,
            "gen_tps": gen_tps,
            "acc_rate": acc_rate,
        })

    aggregated = {
        category: (
            sum(m["gen_tps"] * m["new_tokens"] for m in cat_results) /
            sum(m["new_tokens"] for m in cat_results)
        )
        for category, cat_results in all_results.items()
    }
    return aggregated


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)

    # Baseline
    result_baseline = None
    if not args.no_baseline:
        generator = Generator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
            max_chunk_size = 4096,
        )
        result_baseline = measure(generator, tokenizer, GreedySampler(), args.max_new_tokens)

    # N-gram draft
    result_ngram = None
    result_ngram_temp = None
    if args.ngram_match_min:
        generator = Generator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
            ngram_match_min = args.ngram_match_min,
            num_draft_tokens = args.ngram_draft_length,
            max_chunk_size = 4096,
        )
        result_ngram = measure(generator, tokenizer, GreedySampler(), args.max_new_tokens)
        if args.temperature:
            result_ngram_temp = measure(generator, tokenizer, TopPSampler(0.9, 1), args.max_new_tokens)

    # SD with draft model
    result_draft = None
    result_draft_temp = None
    if args.draft_model_dir:
        generator = Generator(
            model = model,
            cache = cache,
            draft_model = draft_model,
            draft_cache = draft_cache,
            tokenizer = tokenizer,
            max_chunk_size = 4096,
        )
        result_draft = measure(generator, tokenizer, GreedySampler(), args.max_new_tokens)
        if args.temperature:
            result_draft_temp = measure(generator, tokenizer, TopPSampler(0.9, 1), args.max_new_tokens)

    # Print results
    draft_mode = "Draft model"
    if args.draft_model_dir and draft_model.caps.get("dflash_draft"):
        draft_mode = "DFlash"
    r = {
        "Baseline": result_baseline,
        "N-gram (greedy)": result_ngram,
        "N-gram (temp 1)": result_ngram_temp,
        f"{draft_mode} (greedy)": result_draft,
        f"{draft_mode} (temp 1)": result_draft_temp,
    }
    r = {k: v for k, v in r.items() if v is not None}
    if not r:
        print("No results to display")
        return

    categories = sorted({ category for result in r.values() for category in result.keys() })
    headers = [
        f"{col_yellow}Category{col_default}",
        *[f"{col_yellow}{name}{col_default}" for name in r.keys()],
    ]
    rows = []
    for category in categories:
        row = [f"{category}"]
        baseline = None
        for name, result in r.items():
            res_cat = result.get(category)
            if res_cat is not None:
                speedup = None
                if name == "Baseline":
                    baseline = res_cat
                elif baseline is not None:
                    speedup = res_cat / baseline
                s = f"{col_magenta}{res_cat:.2f}{col_default} t/s"
                if speedup is not None:
                    s += f", {col_green}{speedup:.2f}{col_default}x"
            else:
                s = "-"
            row.append(s)
        rows.append(row)

    print()
    print(tabulate(rows, headers = headers, tablefmt = "pipe", colalign=("left", *["right"] * (len(headers) - 1)),))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, default_cache_size = 65536, cache = True, add_draft_model_args = True, )
    parser.add_argument("-nbl", "--no_baseline", action = "store_true", help = "Skip baseline measurement")
    parser.add_argument("-ngram_min", "--ngram_match_min", type = int, help = "N-gram minimum match length, default = 0 (disabled)", default = 0)
    parser.add_argument("-ngram_len", "--ngram_draft_length", type = int, help = "N-gram draft length, default = 4", default = 4)
    parser.add_argument("-tokens", "--max_new_tokens", type = int, help = "Max sampled tokens per round", default = 1024)
    parser.add_argument("-temp", "--temperature", action = "store_true", help = "Also sample with temperature")
    _args = parser.parse_args()
    main(_args)
