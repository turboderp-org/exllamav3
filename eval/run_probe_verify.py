"""Run the verify-forward capture probe against a live target+draft load.

Serve must be DOWN. Loads target + DFlash2 draft (small cache is fine:
capture feasibility is shape/launch behavior, not capacity), runs a short
DFlash2 generation, snapshots one width-8 VERIFY forward (input_ids + tensor
params), then captures model.forward and replays 20x with argmax parity.

Usage: python eval/run_probe_verify.py -m <target> -dm <draft> -cs 8192
"""
import argparse
import sys

import torch

from exllamav3 import model_init
from exllamav3.generator import Generator
from eval.probe_capture_verify import capture_verify


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, add_sampling_args=True, add_draft_model_args=True)
    args = parser.parse_args()
    sampler = model_init.get_arg_sampler(args)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          draft_model=draft_model, draft_cache=draft_cache)

    snap = {}
    orig = model.forward

    def spy(input_ids, params):
        if input_ids.shape[1] == 8 and "ids" not in snap:
            snap["ids"] = input_ids.clone()
            for k in ("cache_seqlens", "block_table", "positions", "position_ids"):
                v = params.get(k)
                if torch.is_tensor(v):
                    snap[k] = v.clone()
            snap["static_keys"] = [k for k in snap if k != "ids"]
        return orig(input_ids, params)

    model.forward = spy
    generator.generate("Explain why the sky is blue in one paragraph.",
                       max_new_tokens=16, seed=1234, sampler=sampler)
    model.forward = orig

    if "ids" not in snap:
        print("no width-8 verify call observed; draft may not have engaged", flush=True)
        return 1
    print("verify shapes:", {k: (tuple(snap[k].shape), str(snap[k].dtype)) for k in snap["static_keys"]}, flush=True)
    ok = capture_verify(
        model, cache, snap["ids"],
        snap.get("cache_seqlens"), snap.get("block_table"),
        snap.get("positions", snap.get("position_ids")),
    )
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
