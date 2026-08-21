import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch

from exllamav3 import Config, Model
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant.exl3 import LinearEXL3

"""
Measure the actual weight-space relative error of an EXL3 quantized model against its
unquantized source, per tensor: rfn = ||W_q - W_ref||_F / ||W_ref||_F. Streams one top-level
module at a time from each model.
"""

@torch.inference_mode()
def main(args):
    device = torch.device("cuda", args.device)

    config_q = Config.from_directory(args.model_q)
    config_r = Config.from_directory(args.model_r)
    model_q = Model.from_config(config_q)
    model_r = Model.from_config(config_r)
    assert len(model_q.modules) == len(model_r.modules), "top-level module count mismatch"

    def walk(mod, out):
        out.append(mod)
        for ch in getattr(mod, "modules", []):
            walk(ch, out)

    def load_module(config, module):
        config.stc.begin_deferred_load()
        module.load("cpu" if module.caps.get("prefer_cpu") else device)
        config.stc.end_deferred_load()
        config.stc.close()

    results = []
    for i in range(len(model_q.modules)):
        mod_q, mod_r = model_q.modules[i], model_r.modules[i]
        tree_q, tree_r = [], []
        walk(mod_q, tree_q)
        walk(mod_r, tree_r)
        assert len(tree_q) == len(tree_r), f"module tree mismatch at index {i}"
        if not any(isinstance(mq, Linear) and mq.qmap for mq in tree_q):
            continue

        load_module(config_q, mod_q)
        load_module(config_r, mod_r)

        for mq, mr in zip(tree_q, tree_r):
            assert mq.key == mr.key, f"{mq.key} != {mr.key}"
            if not isinstance(mq, Linear) or not mq.qmap:
                continue
            if not isinstance(mq.inner, LinearEXL3):
                print(f"    {mq.key:60} (not EXL3, skipped)")
                continue
            wq = mq.inner.get_weight_tensor()          # (in, out), half
            wr = mr.inner.get_weight_tensor()          # (in, out), half
            k_in = min(wq.shape[0], wr.shape[0])
            k_out = min(wq.shape[1], wr.shape[1])
            err_sq, ref_sq = 0.0, 0.0
            for a in range(0, k_out, 8192):
                b = min(a + 8192, k_out)
                dq = wq[:k_in, a:b].float()
                dr = wr[:k_in, a:b].float().to(dq.device)
                err_sq += (dq - dr).square().sum().item()
                ref_sq += dr.square().sum().item()
                del dq, dr
            del wq, wr
            rfn = (err_sq / ref_sq) ** 0.5
            res = dict(
                key = mq.key,
                K = mq.inner.K,
                qbits_key = mq.qbits_key,
                numel = mq.weights_numel(),
                rfn = rfn,
                ref_norm = ref_sq ** 0.5,
            )
            results.append(res)
            print(f"    {mq.key:60} K={res['K']}  rfn {rfn:.5f}")

        mod_q.unload()
        mod_r.unload()
        torch.cuda.empty_cache()

    by_k = {}
    for r in results:
        by_k.setdefault(r["K"], []).append(r["rfn"])
    print("\n -- rfn by K:")
    for k, v in sorted(by_k.items()):
        v = sorted(v)
        print(f"    K={k}: n={len(v)}  min {v[0]:.5f}  median {v[len(v)//2]:.5f}  max {v[-1]:.5f}")

    with open(args.out, "w") as f:
        json.dump(dict(model_q = args.model_q, model_r = args.model_r, results = results), f, indent = 2)
    print(f" -- Saved: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mq", "--model_q", type = str, required = True)
    parser.add_argument("-mr", "--model_r", type = str, required = True)
    parser.add_argument("-d", "--device", type = int, default = 0)
    parser.add_argument("-o", "--out", type = str, required = True)
    main(parser.parse_args())
