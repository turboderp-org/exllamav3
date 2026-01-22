import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.util.progress import ProgressBar
from exllamav3.util.misc import Timer, cuda_sync_active
from exllamav3 import model_init
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
def cached_ids(length):
    return torch.ones((1, length), dtype = torch.long)


def get_lengths(max_length):
    length = 256
    lengths = [length]
    while length < max_length:
        length = min(length * 2, max_length)
        lengths.append(length)
    return lengths


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
                        "attn_mode": "flash_attn",
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


def measure_generate(args, model, cache, warmup = False):
    global faux_recurrent_states
    chunk_size = args.chunk_size
    lengths = [0] + get_lengths(chunk_size if warmup else args.max_length)
    progress = 0
    results = {}
    max_progress = len(lengths)
    with (ProgressBar("Warmup" if warmup else "Generate", max_progress) as pb):
        for length in lengths:
            torch.cuda.synchronize()
            with Timer() as t:
                for _ in range(100):
                    params = {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "past_len": length,
                        "batch_shape": (1, max(length, 256)),
                    }
                    if "recurrent_states" in model.caps and length > 0:
                        for v in faux_recurrent_states.values():
                            v.position = length
                        params.update({
                            "recurrent_states": faux_recurrent_states
                        })
                    logits = model.forward(cached_ids(1), params)
                    sample = torch.argmax(logits)
                    sample = sample.cpu()  # force sync
                    del logits
            results[length] = 100 / t.interval
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
    measure_generate(args, model, cache, warmup = True)
    print(f"{col_yellow}Generation{col_default}")
    generate_results = measure_generate(args, model, cache)
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
