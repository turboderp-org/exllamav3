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

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

@lru_cache
def cached_ids(length):
    return torch.arange(length, dtype = torch.long).unsqueeze(0)


def get_lengths(max_length):
    length = 256
    lengths = [length]
    while length < max_length:
        length = min(length * 2, max_length)
        lengths.append(length)
    return lengths


def measure_prefill(args, model, cache, warmup = False):
    chunk_size = args.chunk_size
    lengths = get_lengths(chunk_size if warmup else args.max_length)
    if args.short_prefill:
        lengths = list(range(lengths[0])) + lengths

    is_recurrent = model.caps.get("recurrent_states", False)
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
                recurrent = [cache.get_test_state(start)] if is_recurrent else None
                for start, end in chunks:
                    params = {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "past_len": start,
                        "batch_shape": (1, max(length, 256)),
                        "recurrent_states": recurrent,
                    }
                    model.prefill(cached_ids(end - start), params)
                cuda_sync_active()
                if is_recurrent:
                    recurrent[0].free()

            results[length] = length / (pre_time + t.interval)
            if not warmup:
                print(f"Length  {length: 6}: {col_green}{results[length]:10.2f}{col_default} tokens/s")
            progress += length
            pb.update(progress)

    return results


def measure_generate(args, model, cache, warmup = False):
    chunk_size = args.chunk_size
    lengths = [0] + get_lengths(chunk_size if warmup else args.max_length - 256)
    is_recurrent = model.caps.get("recurrent_states", False)
    progress = 0
    results = {}
    max_progress = len(lengths)
    with (ProgressBar("Warmup" if warmup else "Generate", max_progress) as pb):
        for length in lengths:
            recurrent = [cache.get_test_state(length)] if is_recurrent else None
            torch.cuda.synchronize()
            with Timer() as t:
                for i in range(100):
                    params = {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "past_len": length + i,
                        "batch_shape": (1, max(length + 256, 256)),
                        "recurrent_states": recurrent
                    }
                    logits = model.forward(cached_ids(1), params)
                    sample = torch.argmax(logits)
                    sample = sample.cpu()  # force sync
                    del logits
            if is_recurrent:
                recurrent[0].free()
            results[length] = 100 / t.interval
            if not warmup:
                print(f"Context {length: 6}: {col_green}{results[length]:10.2f}{col_default} tokens/s")
            progress += 1
            pb.update(progress)

    return results


@torch.inference_mode()
def main(args):

    assert args.max_length <= args.cache_size, \
        "max_length cannot exceed cache size"

    model, config, cache, tokenizer = model_init.init(args, max_chunk_size = args.chunk_size)
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()

    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")
    print(f" -- Chunk size: {args.chunk_size}")
    print()

    if not args.skip_prefill:
        # Test prefill
        if not args.skip_warmup:
            for _ in range(1):
                measure_prefill(args, model, cache, warmup = True)
        print(f"{col_yellow}Prefill:{col_default}")
        prefill_results = measure_prefill(args, model, cache)
        print()

    # Test generation
    if not args.skip_warmup:
        for _ in range(1):
            measure_generate(args, model, cache, warmup = True)
    print(f"{col_yellow}Generation{col_default}")
    generate_results = measure_generate(args, model, cache)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(
        parser,
        default_cache_size = 32768,
        default_autosplit_max_batch_size = 1,
    )
    parser.add_argument("-max_length", "--max_length", type = int, help = "Max context length to measure (default: 32768)", default = 32768)
    parser.add_argument("-chunk_size", "--chunk_size", type = int, help = "Max chunk size (default: 4096)", default = 4096)
    parser.add_argument("-spf", "--skip_prefill", action = "store_true", help = "Skip measuring prefill speed")
    parser.add_argument("-swu", "--skip_warmup", action = "store_true", help = "Skip warmup passes")
    parser.add_argument("-short", "--short_prefill", action = "store_true", help = "Test short-prefill/batch throughput")
    _args = parser.parse_args()
    main(_args)
