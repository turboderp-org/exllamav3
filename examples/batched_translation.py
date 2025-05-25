import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from transformers import AutoTokenizer
from exllamav3.util.progress import ProgressBar
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from exllamav3 import Config, Model, Cache, Tokenizer, model_init, Generator, Job
from datasets import load_dataset
import torch
import time

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_gray = "\u001b[37;1m"

@disk_lru_cache("get_dataset")
def get_dataset(path: str, name: str, split: str, key: str, min_length: int):
    data = load_dataset(path, name, split = split)
    r = []
    for d in data:
        ds = d[key].strip()
        if len(ds) >= min_length:
            r.append(ds)
    return r

def format_request(hf_tokenizer, eos_tokens, text, max_new_tokens, idx):
    instruction = (
        f"Translate the the text between the #BEGIN# and #END# tags into zoomer slang. "
        f"Reply only with the translation enclosed in the same tags:\n\n"
        f"#BEGIN#\n"
        f"{text}\n"
        f"#END#\n"
        f" /no_think"
    )
    chat = [{
        "role": "user",
        "content": instruction
    }]
    input_ids = hf_tokenizer.apply_chat_template(chat, add_generation_prompt = True)
    input_ids = torch.tensor(input_ids, dtype = torch.long).unsqueeze(0)
    job = Job(
        input_ids = input_ids,
        max_new_tokens = max_new_tokens,
        stop_conditions = eos_tokens,
        identifier = idx
    )
    return job

@torch.inference_mode()
def main(args):

    # Load dataset as list
    print(f" -- Loading dataset...")
    in_data = get_dataset(args.dataset_path, args.dataset_name, args.dataset_split, args.dataset_key, 125)
    avg_len = sum([len(d) for d in in_data]) / len(in_data)
    print(f" -- Loaded {len(in_data)} items, avg. item length {avg_len:.2f} chars")

    # Load model
    model, config, cache, tokenizer = model_init.init(args)
    generator = Generator(model, cache, tokenizer, show_visualizer = args.visualize_cache)
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()
    print(f" -- Model: {args.model_dir}")
    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")

    # Use HF tokenizer for prompt formatting
    print(f" -- Loading HF tokenizer...")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Create jobs
    completions = []
    with ProgressBar(" -- Creating jobs", len(in_data)) as pb:
        for idx, text in enumerate(in_data):
            job = format_request(hf_tokenizer, config.eos_token_id_list, text, args.max_reply, idx)
            generator.enqueue(job)
            completions.append("")
            pb.update(idx)

    # Inference
    num_completions = 0
    num_tokens = 0
    time_begin = time.time()
    while generator.num_remaining_jobs():
        results = generator.iterate()

        # We'll always get at least one result for each active job, even if the result contains no output text
        bsz = len(set([r["identifier"] for r in results]))
        num_tokens += bsz

        for result in results:
            if not result["eos"]: continue

            # EOS signal is always accompanied by the full completion, so we don't need to collect text chunks
            idx = result["identifier"]
            response = result["full_completion"]
            completions[idx] += response

            # Measure performance
            num_completions += 1
            elapsed_time = time.time() - time_begin
            rpm = num_completions / (elapsed_time / 60)
            tps = num_tokens / elapsed_time
            print()
            print(f"{col_blue}---------------------------------------------------------------------------{col_default}")
            print(f"{col_blue}Current batch size: {col_yellow}{bsz}{col_default}")
            print(f"{col_blue}Avg. completions/minute: {col_yellow}{rpm:.2f}{col_default}")
            print(f"{col_blue}Avg. output tokens/second: {col_yellow}{tps:.2f}{col_default}")
            print(f"{col_blue}---------------------------------------------------------------------------{col_default}")
            print()

            # Spam completions to the console
            print(f"{col_green}Input {idx}:{col_default}")
            print()
            print(in_data[idx])
            print()
            print(f"{col_green}Completion {idx}:{col_default}")
            print()
            print(completions[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, default_cache_size = 65536)
    parser.add_argument("-vis", "--visualize_cache", action = "store_true", help = "Show cache visualizer (slow)")
    parser.add_argument("-dsp", "--dataset_path", type = str, default = "wikitext", help = "Dataset path")
    parser.add_argument("-dsn", "--dataset_name", type = str, default = "wikitext-2-raw-v1", help = "Dataset name")
    parser.add_argument("-dss", "--dataset_split", type = str, default = "test", help = "Dataset split")
    parser.add_argument("-dsk", "--dataset_key", type = str, default = "text", help = "Dataset key to extract")
    parser.add_argument("-maxr", "--max_reply", type = int, default = 1024, help = "Max length of each reply")

    _args = parser.parse_args()
    main(_args)
