"""
End-to-end smoke test for the sm_75 port: load a real EXL3 model and generate greedily.

The unit tests cover the GEMM in isolation. This exercises the paths they cannot: attention
(which on Turing has no flash-attn and must fall through to the Triton/SDPA backends), the
cache, the sampler, and the full stack of fused kernels a real forward pass touches. Greedy
decoding makes the output reproducible, so a garbled result is a real failure rather than
sampling noise.

    python tests/sm75_e2e.py <model_dir>
"""

import sys
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


def main(model_dir):
    props = torch.cuda.get_device_properties(0)
    from exllamav3.ext import exllamav3_ext as ext
    print(f"device: {props.name}  sm_{props.major}{props.minor}  "
          f"smem_optin={ext.g_get_smem_max(0)}")

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)

    t0 = time.time()
    model.load()
    print(f"load: {time.time() - t0:.1f}s")

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Name three primary colors, comma separated, nothing else."
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    job = Job(
        input_ids=tokenizer.encode(prompt, add_bos=False),
        max_new_tokens=48,
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
        print(f"generate: {(n - 1) / (time.time() - first):.1f} tok/s over {n} tokens")

    print("-" * 60)
    print(out.strip())
    print("-" * 60)

    # Judge coherence, not task completion: reasoning models spend the 48-token budget
    # thinking and never reach the answer, which says nothing about kernel correctness.
    # A broken matmul produces repetition or non-ASCII garbage, and both are caught here.
    text = out.strip()
    if len(text) < 20:
        print(f"FAIL: output too short ({len(text)} chars)")
        return 1

    printable = sum(c.isascii() and (c.isprintable() or c.isspace()) for c in text) / len(text)
    if printable < 0.95:
        print(f"FAIL: {(1 - printable) * 100:.0f}% non-ASCII, likely garbage")
        return 1

    words = text.lower().split()
    if len(words) >= 12 and len(set(words)) < len(words) * 0.35:
        print(f"FAIL: degenerate repetition ({len(set(words))}/{len(words)} unique)")
        return 1

    print(f"PASS: {len(words)} words, {len(set(words))} unique, {printable * 100:.0f}% ASCII")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
