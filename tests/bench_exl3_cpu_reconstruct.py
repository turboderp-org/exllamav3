# Benchmark for the dense EXL3 CPU reconstructor used by the Qwen 3.5 vision offload path.
#
# Discovers representative EXL3 linears (first attention-block projection, first MLP projection,
# final merger projection) in a vision component, then times ext.reconstruct_had_slice_cpu on
# each weight. Timings are evidence only -- correctness is owned by tests/test_cpu_vision.py and
# the scalar reference (EXL3_CPU_RECON_SCALAR=1).
#
# Usage:
#   python tests/bench_exl3_cpu_reconstruct.py --model_dir <exl3 vision checkpoint> \
#       [--threads N] [--runs R]
#
# Not collected by pytest.

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext


def find_exl3_linears(module, prefix = ""):
    """Yield (path, Linear wrapper) for every loaded EXL3 linear under module."""
    from exllamav3.modules.linear import Linear
    for name, value in vars(module).items():
        if isinstance(value, Linear) and value.quant_type == "exl3":
            yield (f"{prefix}{name}", value)
    # Recurse into composite children that hold further modules
    for name, value in vars(module).items():
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if hasattr(item, "modules") or hasattr(item, "attn"):
                    yield from find_exl3_linears(item, f"{prefix}{name}[{i}].")


def discover_targets(vision_model):
    """Pick the first attention-block linear, first MLP linear and the final merger linear."""
    blocks = [m for m in vision_model.modules if type(m).__name__ == "TransformerBlock"]
    mergers = [m for m in vision_model.modules if "Merger" in type(m).__name__]

    targets = []

    def first_linear(container, label):
        for path, lin in find_exl3_linears(container):
            targets.append((f"{label}.{path}", lin))
            return True
        return False

    if blocks:
        first_linear(blocks[0].__dict__.get("attn"), "attn_block0")
        first_linear(blocks[0].__dict__.get("mlp"), "mlp_block0")
    if mergers:
        first_linear(mergers[-1], "merger_final")
    return targets


def bench_one(lin, runs):
    k, n = lin.inner.in_features, lin.inner.out_features
    w = torch.empty((k, n), dtype = torch.half)
    args = (w, lin.inner.trellis, lin.inner.suh, lin.inner.svh,
            lin.inner.K, lin.inner.mcg, lin.inner.mul1, 0)

    times = []
    scratch = (128 * 128 * 4 + 256 * 4) * torch.get_num_threads()  # per-worker tile + subtile
    for _ in range(runs):
        t0 = time.perf_counter()
        ext.reconstruct_had_slice_cpu(*args)
        times.append(time.perf_counter() - t0)

    median = statistics.median(times)
    gbps = trellis_bytes(lin.inner) / median / 1e9
    return {
        "key": lin.key if hasattr(lin, "key") else "?",
        "shape": (k, n),
        "K": lin.inner.K,
        "median_s": median,
        "packed_gbps": gbps,
        "scratch_bytes": scratch,
    }


def trellis_bytes(inner):
    return inner.trellis.numel() * 2


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model_dir", type = str, required = True,
                        help = "EXL3 checkpoint directory with a vision component")
    parser.add_argument("--threads", type = int, default = None,
                        help = "worker threads (default: current torch setting)")
    parser.add_argument("--runs", type = int, default = 10)
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"[FAIL] model_dir not found: {args.model_dir}", file = sys.stderr)
        return 1

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    from exllamav3 import Config, Model

    config = Config.from_directory(args.model_dir)
    vision_model = Model.from_config(config, component = "vision")
    vision_model.load(device = "cpu")

    targets = discover_targets(vision_model)
    if not targets:
        print("[FAIL] no EXL3 linears found under the vision component "
              "(is this a --vision_bits 1..8 conversion?)", file = sys.stderr)
        return 1

    print(f"threads: {torch.get_num_threads()}, runs: {args.runs}")
    print(f"{'key':40s} {'shape':>14s} {'K':>2s} {'median':>10s} {'packed GB/s':>12s} {'scratch':>10s}")
    for key, lin in targets:
        r = bench_one(lin, args.runs)
        print(f"{key:40s} {str(r['shape']):>14s} {r['K']:2d} {r['median_s']*1e3:8.2f}ms "
              f"{r['packed_gbps']:12.2f} {r['scratch_bytes']/1024:8.0f}KB")

    vision_model.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
