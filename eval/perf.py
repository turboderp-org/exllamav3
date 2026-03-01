import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.util.progress import ProgressBar
from exllamav3.util.misc import Timer, cuda_sync_active
from exllamav3 import model_init, Generator, Job, GreedySampler
import torch
import argparse
from functools import lru_cache

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_gray = "\u001b[37;1m"


@lru_cache
def cached_ids(length, value = 1):
    return torch.full((1, length), value, dtype = torch.long)


def get_lengths(max_length):
    length = 256
    lengths = [length]
    while length < max_length:
        length = min(length * 2, max_length)
        lengths.append(length)
    return lengths


def get_generate_lengths(max_length, max_new_tokens, cache_size):
    return [
        length for length in [0] + get_lengths(max_length)
        if length + max_new_tokens <= cache_size
    ]


def get_bench_token_id(tokenizer):
    for text in (" hello", "a"):
        ids = tokenizer.encode(text, add_bos = False)
        if ids.numel():
            return int(ids[0, 0].item())
    return 1


faux_recurrent_states = None


def measure_prefill(args, model, cache, warmup = False):
    global faux_recurrent_states
    chunk_size = args.chunk_size
    lengths = get_lengths(chunk_size if warmup else args.max_length)

    progress = 0
    results = {}
    max_progress = sum(lengths)
    with (ProgressBar("Warmup" if warmup else "Prefill", max_progress) as pb):
        for length in lengths:
            cuda_sync_active()
            with Timer() as t:
                start, end = 0, length
                pre_time = 0
                if length >= chunk_size * 2:
                    pre_time = (length // 2) / results[length // 2]
                    start = length // 2
                chunks = [(i, min(i + chunk_size, end)) for i in range(start, end, chunk_size)]
                for start, end in chunks:
                    params = {
                        "attn_mode": "flashinfer",
                        "cache": cache,
                        "past_len": start,
                        "batch_shape": (1, max(length, 256)),
                    }
                    if "recurrent_states" in model.caps and start > 0:
                        for v in faux_recurrent_states.values():
                            v.position = start
                        params.update({
                            "recurrent_states": faux_recurrent_states
                        })
                    model.prefill(cached_ids(end - start), params)
                    if "recurrent_states" in params and faux_recurrent_states is None:
                        faux_recurrent_states = params["recurrent_states"]
                cuda_sync_active()

            results[length] = length / (pre_time + t.interval)
            if not warmup:
                print(f"Length  {length: 6}: {col_green}{results[length]:10.2f}{col_default} tokens/s")
            progress += length
            pb.update(progress)

    return results


def measure_generate(args, model, cache, tokenizer, warmup = False):
    chunk_size = args.chunk_size
    bench_new_tokens = 100
    lengths = get_generate_lengths(
        chunk_size if warmup else args.max_length,
        bench_new_tokens,
        cache.max_num_tokens
    )
    bench_token_id = get_bench_token_id(tokenizer)
    generator = Generator(
        model,
        cache,
        tokenizer,
        max_batch_size = 1,
        max_chunk_size = chunk_size,
    )
    progress = 0
    results = {}
    max_progress = len(lengths)
    with (ProgressBar("Warmup" if warmup else "Generate", max_progress) as pb):
        for length in lengths:
            input_ids = cached_ids(length + 1, bench_token_id)
            job = Job(
                input_ids = input_ids,
                max_new_tokens = bench_new_tokens + 1,
                sampler = GreedySampler(),
                stop_conditions = [],
            )
            generator.enqueue(job)

            last_result = None
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    if result.get("eos"):
                        last_result = result

            assert last_result is not None, "Generator produced no terminal benchmark result."
            time_generate = max(last_result["time_generate"], 1e-9)
            results[length] = last_result["new_tokens"] / time_generate
            if not warmup:
                print(f"Context {length: 6}: {col_green}{results[length]:10.2f}{col_default} tokens/s")
            progress += 1
            pb.update(progress)

    return results


@torch.inference_mode()
def main(args):

    model, config, cache, tokenizer = model_init.init(args, max_chunk_size = args.chunk_size)
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()

    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")
    print(f" -- Chunk size: {args.chunk_size}")
    print()

    # Test prefill
    measure_prefill(args, model, cache, warmup = True)
    print(f"{col_yellow}Prefill:{col_default}")
    prefill_results = measure_prefill(args, model, cache)
    print()

    # Test generation
    measure_generate(args, model, cache, tokenizer, warmup = True)
    print(f"{col_yellow}Generation{col_default}")
    generate_results = measure_generate(args, model, cache, tokenizer)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(
        parser,
        default_cache_size = 32768,
    )
    parser.add_argument("-max_length", "--max_length", type = int, help = "Max context length to measure (default: 32768)", default = 32768)
    parser.add_argument("-chunk_size", "--chunk_size", type = int, help = "Max chunk size (default: 4096)", default = 4096)
    _args = parser.parse_args()
    main(_args)
