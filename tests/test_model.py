import sys, os
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
import time
from exllamav3 import Generator, Job, model_init, Sampler, Tokenizer
from pprint import pprint
from xml.etree import ElementTree as ET
import random

col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_magenta = "\u001b[35m"
col_red = "\u001b[31;1m"
col_green = "\u001b[32;1m"
col_white = "\u001b[37;1m"

generator: Generator
stop_conditions: list
sampler: Sampler
tokenizer: Tokenizer

def generate_single(args, ids):
    global generator, stop_conditions, sampler, tokenizer

    job = Job(
        input_ids = ids,
        max_new_tokens = args.max_response_tokens,
        stop_conditions = stop_conditions,
        sampler = sampler,
    )
    generator.enqueue(job)

    print(f"{col_blue}Prompt:{col_green}")
    prompt = tokenizer.decode(ids, decode_special_tokens = True)[0]
    if len(prompt) > 2000:
        prompt = prompt[:1000] + f"{col_magenta} \n\n...{col_green} \n\n" + prompt[-1000:]
    print(prompt)
    print()
    print(f"{col_blue}Response:{col_default}")

    completion = ""
    last_result = {}
    stop_reason = "error"
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            chunk = r.get("text", "")
            completion += chunk
            print(chunk, end = "", flush = True)
            if r["eos"]:
                stop_reason = r["eos_reason"]
                last_result = r
                last_result["tokens_second"] = r["new_tokens"] / r["time_generate"]

    print(f"{col_yellow}[{stop_reason}]{col_default}")
    print()
    return completion, last_result


def extract_svg(s: str, begin: str = "<svg", end: str = "</svg>"):
    # Find all tag occurrences in order
    pattern = re.compile(rf"{re.escape(begin)}|{re.escape(end)}")
    tags = list(pattern.finditer(s))

    best = None
    for i in range(len(tags) - 1):
        t1, t2 = tags[i], tags[i+1]
        if t1.group() == begin and t2.group() == end:
            start = t1.start()
            stop  = t2.end()
            length = stop - start
            if best is None or length > best[0]:
                best = (length, start, stop)

    if not best:
        return None

    _, start, stop = best
    return s[start:stop]


def is_valid_svg(text: str) -> bool:
    SVG_NS = "http://www.w3.org/2000/svg"
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return False
    return root.tag == f"{{{SVG_NS}}}svg" or root.tag == "svg"


def task_valid_svg_nothink(args):
    return task_valid_svg(args, think = False)

def task_valid_svg(args, think = True):
    global tokenizer
    ids = tokenizer.hf_chat_template(
        [
            {"role": "user", "content": "Create a detailed SVG image of a cute kitten."}
        ],
        add_generation_prompt = True,
        enable_thinking = think,
    )
    response, _ = generate_single(args, ids)
    svg = extract_svg(response) or ""
    result = {
        "pass": is_valid_svg(svg),
        "response_length": len(response),
        "extracted_length": len(svg),
    }
    return result


def task_kv_cache_reuse_longrec(args):
    return task_kv_cache_reuse(args, [10000, 70000, 250000, 200000, 100000, 50000], tolerance = 32768)

def task_kv_cache_reuse(args, prompt_lengths = None, tolerance = None):
    tolerance = tolerance or generator.recurrent_checkpoint_interval
    prompt_lengths = prompt_lengths or [40000, 38000, 37000, 35000]
    def make_key(index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"exllamav3-kv-cache-reuse:{index}"))

    def make_entries(min_length: int) -> tuple[list[str], str, str]:
        r = random.Random(0)
        lines = ["Entries (key, value):\n"]
        length = len(lines[0])
        target_key = ""
        target_value = ""
        num = 0
        while length < min_length:
            key = make_key(num)
            value = str(r.randint(0, 90000) + 10000)
            line = f"{key}: {value}\n"
            lines.append(line)
            length += len(line)
            if num == 42:
                target_key = key
                target_value = value
            num += 1
        return lines, target_key, target_value

    checks = []
    prefixes = []
    lengths = []
    tps = []
    last_length = 0
    all_lines, target_key, target_value = make_entries(max(prompt_lengths))
    for prompt_length in prompt_lengths:
        lines = []
        length = 0
        for line in all_lines:
            lines.append(line)
            length += len(line)
            if length >= prompt_length:
                break
        prompt = "".join(lines)
        prompt += "\n---\n"
        prompt += f"What is the value for the key, {target_key}?"
        ids = tokenizer.hf_chat_template(
            [
                {"role": "user", "content": prompt}
            ],
            add_generation_prompt = True,
            enable_thinking = False,
        )
        response, last_result = generate_single(args, ids)
        checks.append(target_value in response)
        prefix = last_result.get("cached_tokens", 0)
        length = ids.shape[-1]
        min_prefix = max(0, (min(length, last_length) - tolerance) // 256 * 256)
        checks.append(prefix >= min_prefix)
        lengths.append(length)
        prefixes.append(prefix)
        tps.append(last_result["tokens_second"])
        last_length = length
    result = {
        "pass": all(checks),
        "lengths": lengths,
        "prefixes": prefixes,
        "tps": tps,
    }
    if generator.recurrent_cache:
        result["recurrent_checkpoints"] = sorted([
            cp["position"] for _, cp in generator.recurrent_cache.items()
        ])

    return result


all_tasks = {
    "valid_svg": task_valid_svg,
    "valid_svg_nothink": task_valid_svg_nothink,
    "kv_cache_reuse": task_kv_cache_reuse,
    "kv_cache_reuse_longrec": task_kv_cache_reuse_longrec,
}


def main(args):
    global generator, stop_conditions, sampler, tokenizer
    started = time.time()

    # Load model
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)

    # Generator
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        draft_model = draft_model,
        draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens,
        ngram_match_min = args.ngram_match_min,
        recurrent_cache_size = args.sysmem_recurrent_cache * 1024**2,
        recurrent_checkpoint_interval_pp = args.recurrent_checkpoint_interval_pp,
    )
    stop_conditions = config.eos_token_id_list
    sampler = model_init.get_arg_sampler(args)

    # Run tasks
    if args.tasks == "all":
        tasks = list(all_tasks.keys())
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {}
    for task in tasks:
        if task not in all_tasks:
            raise ValueError(f"Unknown task: {task}")
        result = all_tasks[task](args)
        results[task] = result

    # Results
    output = {
        "run": args.run_name,
        "model_dir": args.model_dir,
        "tasks": results,
        "elapsed_sec": time.time() - started,
    }
    print(col_white, end = "")
    pprint(output)
    print(col_default)

    if args.result_json:
        with open(args.result_json, "w") as f:
            json.dump(output, f, indent = 2)
            f.write("\n")

    if args.result_jsonl:
        with open(args.result_jsonl, "a") as f:
            json.dump(output, f)
            f.write("\n")

    if args.print_result_json:
        print("RESULT_JSON: " + json.dumps(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, cache = True, add_sampling_args = True, add_draft_model_args = True, default_cache_size = 16384)
    parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")
    parser.add_argument("-maxr", "--max_response_tokens", type = int, default = 8192, help = "Max tokens per response, default = 4096")
    parser.add_argument("-tasks", "--tasks", type = str, default = "all", help = "Comma-separated list of task names, default = all")
    parser.add_argument("-ngram_min", "--ngram_match_min", type = int, help = "N-gram draft minimum match length, default = 0 (disabled)", default = 0)
    parser.add_argument("-run", "--run_name", type = str, default = None, help = "Name to include in structured test results")
    parser.add_argument("-result", "--result_json", type = str, default = None, help = "Write structured result to a JSON file")
    parser.add_argument("-resultl", "--result_jsonl", type = str, default = None, help = "Append structured result to a JSONL file")
    parser.add_argument("-print_result", "--print_result_json", action = "store_true", help = "Print structured result as one RESULT_JSON line")
    parser.add_argument("-smc", "--sysmem_recurrent_cache", type = int, default = 4096, help = "Max size of recurrent cache (sysmem) in MB")
    parser.add_argument("-rcpip", "--recurrent_checkpoint_interval_pp", type = int, default = 32768, help = "Recurrent checkpoint interval on long prompts")
    main(parser.parse_args())
