"""
DeepSeek-V4-Flash 2.77bpw on Turing, with MoE experts offloaded to system RAM.

This is the case the port was built for: 93 GiB of weights against 4x15 GiB of T4. The routed
experts of the leading MoE layers run on the CPU (--moe_cpu_offload), which also exercises the
CPU handoff path and DeepSeek's DSA sparse attention - neither of which the smaller models
touch.

The offload path spawns worker processes, so this must run under a __main__ guard.

    python tests/sm75_dsv4.py <model_dir> [cpu_layers]
"""

import sys
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


def main():
    model_dir = sys.argv[1]
    cpu_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 40

    props = torch.cuda.get_device_properties(0)
    from exllamav3.ext import exllamav3_ext as ext
    print(f"device: {props.name} sm_{props.major}{props.minor} "
          f"x{torch.cuda.device_count()}  smem_optin={ext.g_get_smem_max(0)}",
          flush=True)

    config = Config.from_directory(model_dir)
    config.infer_params.moe_cpu_offload = cpu_layers
    print(f"moe_cpu_offload: {cpu_layers} layers", flush=True)

    model = Model.from_config(config)
    # 4096 rather than a token-count minimum: DeepSeek-V4's load-time autosplit probe runs a
    # full chunk through attention and asserts the cache can hold it.
    cache = Cache(model, max_num_tokens=4096)

    t0 = time.time()
    model.load(progressbar=True)
    print(f"load: {time.time() - t0:.0f}s", flush=True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    prompt = (
        "<|begin_of_sentence|><|User|>Name three primary colors."
        "<|Assistant|></think>"
    )
    job = Job(
        input_ids=tokenizer.encode(prompt, add_bos=False),
        max_new_tokens=32,
        sampler=GreedySampler(),
    )
    generator.enqueue(job)

    out = ""
    first = None
    n = 0
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("stage") == "streaming":
                if first is None:
                    first = time.time()
                n += 1
            out += r.get("text", "")

    if first and n > 1:
        print(f"generate: {(n - 1) / (time.time() - first):.2f} tok/s over {n} tokens")

    text = out.strip()
    print("-" * 60)
    print(text[:400])
    print("-" * 60)

    if len(text) < 10:
        print(f"FAIL: output too short ({len(text)} chars)")
        return 1
    printable = sum(c.isascii() and (c.isprintable() or c.isspace()) for c in text) / len(text)
    if printable < 0.95:
        print(f"FAIL: {(1 - printable) * 100:.0f}% non-ASCII, likely garbage")
        return 1
    print(f"PASS: {len(text.split())} words, {printable * 100:.0f}% ASCII")
    return 0


if __name__ == "__main__":
    sys.exit(main())
