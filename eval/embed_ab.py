"""Embedding placement A/B: CPU-resident (upstream default, prefer_cpu) vs CUDA.

The fp32 hidden slab (1,T,5120) is born on CPU and hauled H2D once per forward.
This moves the table device-side and times identical generations both ways.
Outputs must be bit-identical (lookup is exact); only wall may move.

Usage: python eval/embed_ab.py -m <target> -dm <draft> -cs 8192
"""
import argparse
import sys
import time

import torch

from exllamav3 import model_init
from exllamav3.generator import Generator

PROMPT = "Explain why the sky is blue in one paragraph."


def find_embed(model):
    for m in model.modules:
        if type(m).__name__ == "Embedding" and "embed_tokens" in (m.key or ""):
            return m
    for m in model.modules:
        if type(m).__name__ == "Embedding":
            return m
    return None


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, add_sampling_args=True, add_draft_model_args=True)
    args = parser.parse_args()
    sampler = model_init.get_arg_sampler(args)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          draft_model=draft_model, draft_cache=draft_cache)

    emb = find_embed(model)
    print(f"embed module: key={emb.key} device={emb.device} "
          f"numel={emb._numel} (~{emb._numel * 2 / 1e9:.2f} GB fp16)", flush=True)

    outs = {}
    order = ("cpu", "cuda", "cpu2")
    for tag in order:
        if tag == "cuda":
            emb.load(torch.device("cuda:0"))
            print(f"embed moved: device={emb.device}", flush=True)
            _orig_fwd = emb.forward
            def _fwd_cuda(x, params, out_dtype=None, _f=_orig_fwd, _dev=emb.device):
                # indices H2D (tiny: 8 int64); slab is then born on device and
                # every downstream prepare_for_device becomes a no-op
                if torch.is_tensor(x) and str(x.device) != str(_dev):
                    x = x.to(_dev, non_blocking=True)
                return _f(x, params, out_dtype)
            emb.forward = _fwd_cuda
        elif tag == "cpu2":
            emb.forward = _orig_fwd
            emb.load(torch.device("cpu"))
            print(f"embed moved back: device={emb.device}", flush=True)
        t0 = time.time()
        out = generator.generate(PROMPT, max_new_tokens=64, seed=1234, sampler=sampler)
        wall = time.time() - t0
        outs[tag] = (out, wall)
        print(f"EMBED-{tag}: wall={wall:.2f}s text={out[:60]!r}", flush=True)

    same = outs["cpu"][0] == outs["cuda"][0] == outs["cpu2"][0]
    print(f"BIT-IDENTICAL: {same}", flush=True)
    print(f"WALL cpu={outs['cpu'][1]:.2f}s cuda={outs['cuda'][1]:.2f}s cpu2={outs['cpu2'][1]:.2f}s", flush=True)
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(main())
