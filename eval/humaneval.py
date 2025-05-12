from __future__ import annotations
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import model_init, Generator, Job, ComboSampler
from exllamav3.util.progress import ProgressBar
import argparse, contextlib, subprocess
from human_eval.data import write_jsonl, read_problems
from pathlib import Path

# Prompt formats
prompt_formats = {
    "raw": (
        "```python\n{{problem}}    ",
        "    "
    ),
    "granite": (
        "<|endoftext|>Question:\nComplete the following Python function:\n\n{{problem}}\n\nAnswer:\n"
        "Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "llama": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful AI coding assistant.\n"
        "<</SYS>>\n\n"
        "Complete the following Python function:\n\n"
        "{{problem}} [/INST] "
        "Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI coding assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Complete the following Python function:\n\n{{problem}}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "mistral": (
        "<s>[INST] You are a helpful AI coding assistant.\n\n"
        "Complete the following Python function:\n\n"
        "{{problem}}[/INST]"
        " Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "gemma": (
        "<bos><start_of_turn>user\n"
        "Complete the following Python function:\n\n{{problem}}<|eot_id|>"
        "<start_of_turn>model\n"
        "```python\n{{problem}}",
        "    "
    ),
    "reka": (
        "<|endoftext|>human: Complete the following Python function."
        " Provide your reasoning in comments, but be concise and don't second-guess."
        "\n\n{{problem}}"
        " <sep> assistant: ```python\n{{problem}}",
        "    "
    ),
    "chatml": (
        "<|im_start|>system\n"
        "You are a helpful AI coding assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Complete the following Python function:\n\n{{problem}}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "qwen3": (
        "<|im_start|>system\n"
        "You are a helpful AI coding assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Complete the following Python function:\n\n{{problem}}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\nSure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "deepseek": (
        "<｜begin▁of▁sentence｜>You are a helpful AI coding assistant.\n"
        "<｜User｜>Complete the following Python function:\n\n{{problem}}"
        "<｜Assistant｜>Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    )
}

def main(args):

    # Validate args
    directory = os.path.dirname(args.output)
    if os.path.exists(args.output):
        print(f" !! Warning: Output file exists and will be overwritten.")

    if args.prompt_format is None:
        prompt_format, prefix = "{{problem}}", "    "
    elif args.prompt_format in prompt_formats:
        prompt_format, prefix = prompt_formats[args.prompt_format]
    else:
        print("Prompt format is not supported. Available formats:")
        print("\n".join(prompt_formats.keys()))
        sys.exit()

    # Initialize
    model, config, cache, tokenizer = model_init.init(args)
    generator = Generator(
        model = model,
        cache = cache,
        max_batch_size = 256,
        tokenizer = tokenizer
    )
    sampler = ComboSampler(
        temperature = args.temperature,
        min_p = args.min_p,
        top_k = args.top_k,
        top_p = args.top_p,
        temp_last = args.temp_last
    )

    # Get problems
    problems = read_problems()
    num_samples_per_task = args.samples_per_task

    # Create jobs
    with ProgressBar("Creating sample jobs", len(problems), transient = False) as progress:
        for idx, (problem_id, problem) in enumerate(problems.items()):
            b_problem = problem["prompt"]
            f_problem = prompt_format.replace("{{problem}}", b_problem)
            input_ids = tokenizer.encode(
                f_problem,
                encode_special_tokens = True,
                add_bos = (args.prompt_format == "raw")
            )
            for s in range(num_samples_per_task):
                job = Job(
                    input_ids = input_ids,
                    sampler = sampler,
                    max_new_tokens = args.max_tokens,
                    stop_conditions = [tokenizer.eos_token_id],
                    token_healing = True,
                    identifier = (problem_id, s),
                    min_new_tokens = 6
                )
                generator.enqueue(job)
            progress.update(idx)

    # Collect samples here
    samples = []

    # Work
    total_jobs = generator.num_remaining_jobs()
    with ProgressBar("Generating samples" if not args.verbose else None, total_jobs, transient = False) as progress:

        while generator.num_remaining_jobs():
            results = generator.iterate()
            for result in results:

                # End sample if generator says EOS or if there is a non-indented line at the end of the output
                job = result["job"]
                eos = False
                completion = job.full_completion
                last_newline_index = completion.rfind("\n")
                if last_newline_index >= 0:
                    last_line = completion[last_newline_index + 1:]
                    if last_line != "" and not last_line[0].isspace():
                        completion = completion[:last_newline_index]
                        eos = True
                eos = eos or result["eos"]

                # Collect completed sample
                if eos:
                    identifier = result["identifier"]
                    sample = problems[identifier[0]]["prompt"] + prefix + completion.strip()
                    if not result["eos"]:
                        generator.cancel(job)

                    if args.verbose:
                        print("----------------------------------------------------------------------")
                        print(f" ** Problem {identifier[0]}, sample {identifier[1] + 1} / {num_samples_per_task}")
                        print("----------------------------------------------------------------------")
                        print(sample)
                        print()
                    progress.update(total_jobs - generator.num_remaining_jobs())
                    samples.append(dict(task_id = identifier[0], completion = prefix + completion.strip()))

    # Save output
    print(f" -- Saving: {args.output}")
    Path(directory).mkdir(parents = True, exist_ok = True)
    write_jsonl(args.output, samples)

    # Optionally launch eval script
    if args.eval:
        subprocess.run(["evaluate_functional_correctness", args.output])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run HumanEval evaluation")
    model_init.add_args(parser)
    parser.add_argument("-o", "--output", type = str, help = "Output .jsonl filename", required = True)
    parser.add_argument("-spt", "--samples_per_task", type = int, default = 200)
    parser.add_argument("-pf", "--prompt_format", type = str, help = "Instruct format to apply. Default is raw completion (for base models) ")
    parser.add_argument("-v", "--verbose", action = "store_true", help = "Spam completions to console while generating")
    parser.add_argument("-e", "--eval", action = "store_true", help = "Run evaluation script on output file after sampling")
    parser.add_argument("-temp", "--temperature", type = float, help = "Sampling temperature (0 for greedy), default: 0.6", default = 0.6)
    parser.add_argument("-minp", "--min_p", type = float, help = "Min-p sampling, default: 0.0 (disabled)", default = 0.0)
    parser.add_argument("-topk", "--top_k", type = int, help = "Top-k sampling, default: 0 (disabled)", default = 0)
    parser.add_argument("-topp", "--top_p", type = float, help = "Top-p sampling, default: 0.6", default = 0.6)
    parser.add_argument("-templast", "--temp_last", action = "store_true", help = "Use temperature last")
    parser.add_argument("--max_tokens", type = int, default = 768, help = "Max number of tokens for each completion")
    _args = parser.parse_args()
    main(_args)
