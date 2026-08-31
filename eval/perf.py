import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.util.progress import ProgressBar
from exllamav3.util.misc import Timer, cuda_sync_active
from exllamav3.util.file import disk_lru_cache
from exllamav3 import model_init
import torch
import argparse

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_gray = "\u001b[37;1m"

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

@disk_lru_cache("_load_wikitext2_raw")
def _load_wikitext2_raw() -> str:
    """
    Get raw wikitext2 test split exactly as used by perplexity.c (datasets version differs slightly on whitespace)
    """

    import tempfile, zipfile, pathlib, urllib.request

    _WIKITEXT2_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

    cache_dir = pathlib.Path(tempfile.gettempdir()) / "llama_cpp_ppl_wikitext2"
    cache_dir.mkdir(parents = True, exist_ok = True)

    raw_path = cache_dir / "wikitext-2-raw" / "wiki.test.raw"
    if not raw_path.exists():

        zip_path = cache_dir / "wikitext-2-raw-v1.zip"
        if not zip_path.exists():
            print(f"Downloading WikiText-2 raw to {zip_path} ...")
            urllib.request.urlretrieve(_WIKITEXT2_URL, str(zip_path))

        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(cache_dir))

        zip_path.unlink(missing_ok = True)
        if not raw_path.exists():
            raise FileNotFoundError(f"Failed to extract to {raw_path}.")

    with open(raw_path, "r", encoding = "utf-8") as f:
        return f.read()


# Measurement token stream: raw tokenized wikitext2 (same cached text as ppl.py) rather than
# a synthetic pattern, so data-dependent work sees realistic activations (MoE expert routing
# especially)
_workload_ids = None

def load_workload_ids(tokenizer, needed):
    global _workload_ids
    ids = tokenizer.encode(_load_wikitext2_raw())
    if ids.shape[-1] < needed:
        ids = ids.repeat(1, -(-needed // ids.shape[-1]))
    _workload_ids = ids


def workload_ids(pos, length):
    a = pos % max(_workload_ids.shape[-1] - length, 1)
    return _workload_ids[:, a : a + length]


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

    ids_offset = 0 if warmup else args.max_length
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
                    model.prefill(workload_ids(start + ids_offset, end - start), params)
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

    ids_offset = args.max_length * 2 if warmup else args.max_length * 3
    is_recurrent = model.caps.get("recurrent_states", False)
    progress = 0
    results = {}
    seqlens = [1, 2, 3, 4] if args.spec_dec else [1]
    unit = "it" if args.spec_dec else "tokens"
    max_progress = len(lengths)
    with (ProgressBar("Warmup" if warmup else "Generate", max_progress) as pb):
        for length in lengths:
            for seqlen in seqlens:
                recurrent = [cache.get_test_state(length)] if is_recurrent else None
                torch.cuda.synchronize()
                with Timer() as t:
                    for i in range(100 // seqlen):
                        params = {
                            "attn_mode": "flash_attn",
                            "cache": cache,
                            "past_len": length + i * seqlen,
                            "batch_shape": (1, max(length + 256, 256)),
                            "recurrent_states": recurrent
                        }
                        logits = model.forward(workload_ids(ids_offset + length + i, seqlen), params)
                        sample = torch.argmax(logits)
                        sample = sample.cpu()  # force sync
                        del logits
                if is_recurrent:
                    recurrent[0].free()
                results[seqlen, length] = (100 // seqlen) / t.interval

            if not warmup:
                print(
                    f"Context {length: 6}: " +
                    ",   ".join([
                        f"S={col_gray}{seqlen} {col_green}{results[seqlen, length]:10.2f}{col_default} {unit}/s"
                        for seqlen in seqlens
                    ])
                )

            progress += 1
            pb.update(progress)

    return results


@torch.inference_mode()
def main(args):

    if args.max_length % 256 != 0:
        args.max_length = args.max_length // 256 * 256

    if args.max_length > args.cache_size:
        args.max_length = args.cache_size
        print(f" !! max_length cannot exceed cache size, limiting to {args.max_length}")

    model, config, cache, tokenizer = model_init.init(args, max_chunk_size = args.chunk_size)
    load_workload_ids(tokenizer, args.max_length + 512)
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

    if not args.skip_gen:
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
    parser.add_argument("-sg", "--skip_gen", action = "store_true", help = "Skip measuring generaition speed")
    parser.add_argument("-swu", "--skip_warmup", action = "store_true", help = "Skip warmup passes")
    parser.add_argument("-short", "--short_prefill", action = "store_true", help = "Test short-prefill/batch throughput")
    parser.add_argument("-sd", "--spec_dec", action = "store_true", help = "Test spec-decode seqlens 1..4")
    _args = parser.parse_args()
    main(_args)
