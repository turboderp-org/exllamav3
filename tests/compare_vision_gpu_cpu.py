# Tensor-level comparison: GPU vs CPU vision encoder.
#
# Feeds bit-identical preprocessed inputs (pixel tensor, grid_thw, inv_freq) to the same vision
# component loaded once on CUDA and once on CPU, captures every intermediate hidden state
# (patch embedding, all transformer blocks, deepstack mergers if present, final merger) via
# forward hooks, and reports elementwise statistics per tap point.
#
# Usage:
#   python tests/compare_vision_gpu_cpu.py --model_dir <dir> [--image cat.png] [--device cuda:0]

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
from PIL import Image

from exllamav3 import Config, Model
from exllamav3.architecture.mm_processing.qwen2 import qwen2_position_embedding_grid_2d

# Pass/fail thresholds applied to the final merger output (informational for intermediate taps).
PASS_COSINE = 0.999
PASS_REL_L2 = 0.02


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.float().flatten()
    b = b.float().flatten()
    assert a.shape == b.shape, f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}"
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)

    # Relative error restricted to significant elements (|ref| >= 1% of peak), so near-zero
    # elements don't dominate; plus a global relative L2 norm which is robust to them.
    significant = b.abs() >= 0.01 * b.abs().max()
    rel_sig = (diff[significant] / denom[significant]) if significant.any() else diff.new_tensor([0.0])

    norm_b = b.norm().clamp_min(1e-12)
    return {
        "shape": tuple(a.shape),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_l2": ((a - b).norm() / norm_b).item(),
        "max_rel_sig": rel_sig.max().item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim = 0).item(),
    }


def attach_captures(vision_model: Model, captures: dict):
    # exllamav3 modules are not torch.nn.Module, so wrap each target's forward to snapshot
    # its output hidden state.
    for m in vision_model.modules:
        if not (m.key.endswith(".patch_embed.proj")
                or ".blocks." in m.key
                or ".merger" in m.key):
            continue
        orig_forward = m.forward

        def wrapped(x, params = None, _f = orig_forward, _key = m.key):
            out = _f(x, params)
            t = out[0] if isinstance(out, tuple) else out
            # Blocks mutate the residual buffer in place (x += y) and return it, so a
            # no-op .float().cpu() on CPU tensors would alias the live buffer. Always copy.
            captures[_key] = t.detach().to(device = "cpu", dtype = torch.float32, copy = True)
            return out

        m.forward = wrapped


def run_pass(vision_model: Model, device: str, image_tensor, grid_thw, inv_freq):
    captures = {}
    attach_captures(vision_model, captures)
    vision_model.load(device = device)
    try:
        params = {
            "causal": False,
            "grid_thw": torch.tensor([grid_thw], dtype = torch.int),
            "inv_freq": inv_freq,
        }
        with torch.no_grad():
            out = vision_model.forward(image_tensor.to(device), params = params)
    finally:
        vision_model.unload()
    return out.float().cpu(), captures


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model_dir", type = str, required = True)
    parser.add_argument("--image", type = str, default = "examples/media/cat.png")
    parser.add_argument("--device", type = str, default = "cuda:0")
    args = parser.parse_args()

    print(f"torch {torch.__version__} | comparing vision encoder: CUDA ({args.device}) vs CPU")

    config = Config.from_directory(args.model_dir)
    image = Image.open(args.image)

    # Preprocess exactly once; both passes consume the identical tensors.
    prep_model = Model.from_config(config, component = "vision")
    image_tensor, prep_image_size, grid_thw = prep_model.preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    inv_freq = qwen2_position_embedding_grid_2d(
        grid_thw, config.vision.head_dim, config.vision.spatial_merge_size, config.vision.rope_theta
    )
    del prep_model
    print(f"input: {args.image} | pixel tensor {tuple(image_tensor.shape)} "
          f"{image_tensor.dtype} | grid {grid_thw}")

    model_gpu = Model.from_config(config, component = "vision")
    model_cpu = Model.from_config(config, component = "vision")
    out_gpu, cap_gpu = run_pass(model_gpu, args.device, image_tensor, grid_thw, inv_freq)
    out_cpu, cap_cpu = run_pass(model_cpu, "cpu", image_tensor, grid_thw, inv_freq)

    keys = sorted(set(cap_gpu) | set(cap_cpu), key = lambda k: (len(k), k))
    if not keys:
        sys.exit("no captures recorded — hook placement failed")

    print(f"\n{'tap point':<40} {'max_abs':>10} {'mean_abs':>10} {'rel_l2':>10} "
          f"{'max_rel_sig':>11} {'cosine':>12}")
    all_pass = True
    for key in keys:
        if key not in cap_gpu or key not in cap_cpu:
            print(f"{key:<40} MISSING on {'GPU' if key not in cap_gpu else 'CPU'}")
            all_pass = False
            continue
        m = tensor_metrics(cap_gpu[key], cap_cpu[key])
        print(f"{key:<40} {m['max_abs']:>10.3e} {m['mean_abs']:>10.3e} {m['rel_l2']:>10.3e} "
              f"{m['max_rel_sig']:>11.3e} {m['cosine']:>12.8f}")

        if key.endswith(".merger") and "deepstack" not in key:
            final_ok = m["cosine"] > PASS_COSINE and m["rel_l2"] < PASS_REL_L2
            if not final_ok:
                all_pass = False

    print(f"\nfinal output shape: GPU {tuple(out_gpu.shape)} vs CPU {tuple(out_cpu.shape)}")
    m = tensor_metrics(out_gpu, out_cpu)
    print(f"forward return  : max_abs {m['max_abs']:.3e} | rel_l2 {m['rel_l2']:.3e} | "
          f"cosine {m['cosine']:.8f}")

    print(f"\n[{'PASS' if all_pass else 'FAIL'}] GPU vs CPU vision encoder comparison "
          f"(thresholds: final cosine > {PASS_COSINE}, rel_l2 < {PASS_REL_L2})")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
