import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.progress import ProgressBar
from exllamav3 import model_init, Generator, Job, FormatronFilter
from formatron.formatter import FormatterBuilder
from formatron.schemas.dict_inference import infer_mapping
import torch
import json
from collections import Counter
import math
import re

"""
Sampling diversity test, highly scientific.
"""

system_prompt = \
"""You are a creative writing assistant."""

prompts = [
    {
        "prompt": (
            """Write the opening paragraph to a short story about a cat and its owner. The story should at a minimum mention """
            """the owner's name and the cat's name and color. The ownwer is wearing a colorful dress. Make sure to also """
            """mention the color of the dress."""
        ),
        "questions": [
            ("cat_name", "What is the name of the cat in the paragraph above?", "x"),
            ("cat_color", "What is the color of the cat in the paragraph above?", "x"),
            ("owner_name", "What is the name of the cat's owner in the paragraph above?", "x"),
            ("dress_color", "What is the color of the owner's dress in the paragraph above?", "x"),
        ]
    },
    {
        "prompt": (
            """I'm writing a story. Give me the first paragraph, which should describe the main character. Make sure to """
            """include their name and occupation, and also make it clear which city the story takes place in."""
        ),
        "questions": [
            ("character_name", "What is the name of the main character in the paragraph above?", "x"),
            ("occupation", "What is the occupation of the main character in the paragraph above?", "x"),
            ("location", "In which town or city does the story take place?", "x"),
        ]
    }
]

post_prompt = \
""" Answer in JSON format."""

prefix_response = \
"""Here is the answer, in JSON format:\n\n"""


# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_gray = "\u001b[37;1m"


def diversity_score(samples):
    """
    Compute score as (1 - P(X1 = X2)) ^ 2 where X1 and X2 are two random samples:

    0.0 means all samples are the same
    1.0 means all samples are unique
    """
    n = len(samples)
    if n < 2: return 0.0

    counts = Counter(samples)

    # number of matching unordered pairs
    same_pairs = sum(c * (c - 1) for c in counts.values())

    # total unordered pairs
    total_pairs = n * (n - 1)

    theta = same_pairs / total_pairs
    return (1 - theta) ** 2


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<seed:think>.*?</seed:think>", "", text, flags=re.DOTALL)
    text = text.strip()
    return text

@torch.inference_mode()
def main(args):

    # Load model
    model, config, cache, tokenizer = model_init.init(args)
    generator = Generator(
        model,
        cache,
        tokenizer,
        show_visualizer = args.visualize_cache,
        max_batch_size = args.max_batch_size,
    )
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()

    print(f" -- Model: {args.model_dir}")
    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")

    def make_job(messages, ex_dict = None, prefix = ""):
        nonlocal args
        input_ids = tokenizer.hf_chat_template(messages, add_generation_prompt = True)
        if ex_dict is not None:
            f = FormatterBuilder()
            schema = infer_mapping(ex_dict)
            f.append_line(f"{prefix}{f.json(schema, capture_name = 'json')}")
            filters = [FormatronFilter(tokenizer, eos_after_completed = True, formatter_builder = f)]
        else:
            filters = None
        job = Job(
            input_ids = input_ids,
            max_new_tokens = args.max_tokens,
            stop_conditions = config.eos_token_id_list,
            sampler = model_init.get_arg_sampler(args),
            filters = filters,
            max_rq_tokens = 512,
        )
        return job

    all_sets = {}
    for p in prompts:

        # Generate samples
        jobs = []
        for i in range(args.num_samples):
            jobs.append(make_job([
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": p["prompt"]
                }
            ]))
        generator.enqueue(jobs)
        with ProgressBar("Inference", len(jobs)) as pb:
            while j := generator.num_remaining_jobs():
                generator.iterate()
                pb.update(len(jobs) - j)
        samples = [clean(job.full_completion) for job in jobs]

        # Some feedback
        print(f"{col_yellow}\nSample:{col_default}")
        print(f"{col_gray}{samples[0]}{col_default}")

        # Extract information
        jobs = []
        for i in range(args.num_samples):
            for j in range(len(p["questions"])):
                var, prompt, ex = p["questions"][j]
                prompt += " Anwser in JSON format."
                jobs.append(make_job([
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": samples[i] + "\n\n" + prompt
                    }
                ], {var: ex}, prefix_response))
        generator.enqueue(jobs)
        with ProgressBar("Inference", len(jobs)) as pb:
            while j := generator.num_remaining_jobs():
                generator.iterate()
                pb.update(len(jobs) - j)
        results = [job.full_completion.strip() for job in jobs]

        # Parse results
        sets = {v: [] for v, _, _ in p["questions"]}
        for result in results:
            r = result[len(prefix_response):]
            try:
                j = json.loads(r)
            except json.JSONDecodeError:
                continue
            for k, v in j.items():
                sets[k].append(v.strip().lower())

        # Even more feedback
        print(f"\n{col_yellow}Extracted:{col_default}")
        for k, v in sets.items():
            print(f"{k:20}", end = "")
            c = Counter(v)
            for s, t in c.most_common():
                print(f"{col_blue}{s}{col_gray}: {t}, ", end = "")
            print(f"{col_default}")

        all_sets.update(sets)

    # Compute and print scores
    print(f"\n{col_yellow}Scores:{col_default}")
    score_sum = 0.0
    for k, v in all_sets.items():
        score = diversity_score(v)
        print(f"{k:20} {col_green}{score:8.6f}{col_default}")
        score_sum += score
    mean = score_sum / len(all_sets)
    print("-" * 29)
    print(f"{'mean':20} {col_green}{mean:8.6f}{col_default}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(
        parser,
        default_cache_size = 32768,
        add_sampling_args = True,
        default_sampling_args = {
            "temperature": 0.8,
            "min_p": 0.08,
        }
    )
    parser.add_argument("-samples", "--num_samples", type = int, help = "Number of samples (default: 50)", default = 50)
    parser.add_argument("-vis", "--visualize_cache", action = "store_true", help = "Show cache visualizer (slow)")
    parser.add_argument("-max_tokens", "--max_tokens", type = int, help = "Max number of tokens per sample (default: 2048)", default = 2048)
    parser.add_argument("-mbs", "--max_batch_size", type = int, help = "Max batch size (default: 16)", default = 16)
    _args = parser.parse_args()
    main(_args)
