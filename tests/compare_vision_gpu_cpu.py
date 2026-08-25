# Tensor-level comparison: GPU vs CPU vision encoder.
#
# Feeds bit-identical preprocessed inputs (pixel tensor, grid_thw, inv_freq, cu_seqlens) to the
# same vision component loaded once on CUDA and once on CPU, captures every intermediate hidden
# state (patch embedding, all transformer blocks, deepstack mergers if present, final merger) via
# forward hooks, and reports elementwise statistics per tap point.
#
# Modes:
#   single image (default) : dense attention request, no cu_seqlens
#   --images f1 f2 ...     : different-size images packed into one varlen sequence,
#                            one attention segment per image
#   --video_frames K       : K deterministic pseudo-frames derived from --image, packed as one
#                            clip with one attention segment per frame
#   --bench                : timing-only leg over synthetic multi-image/video shapes
#   --profile              : CPU op-attribution leg over the same synthetic shapes
#
# Usage:
#   python tests/compare_vision_gpu_cpu.py --model_dir <dir> [--image cat.png] [--device cuda:0]
#   python tests/compare_vision_gpu_cpu.py --model_dir <dir> --images a.png b.png c.png
#   python tests/compare_vision_gpu_cpu.py --model_dir <dir> --video_frames 8
#   python tests/compare_vision_gpu_cpu.py --model_dir <dir> --bench
#   python tests/compare_vision_gpu_cpu.py --model_dir <dir> --profile

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
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
    diff = (a - b).abs()
    l2a = a.pow(2).sum().sqrt().clamp_min(1e-12)
    l2b = b.pow(2).sum().sqrt().clamp_min(1e-12)
    cos = (a @ b / (l2a * l2b)).item()
    rel_l2 = (diff.pow(2).sum().sqrt() / l2b.clamp_min(1e-12)).item()
    max_rel_sig = (diff / b.abs().clamp_min(1e-6)).max().item()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_l2": rel_l2,
        "max_rel_sig": max_rel_sig,
        "cosine": cos,
    }


def attach_captures(vision_model: Model, captures: dict):
    # exllamav3 modules are not torch.nn.Module, so wrap each target's forward to snapshot
    # its output hidden state.
    for m in vision_model.modules:
        name = m.key
        fwd = m.forward

        def wrapped(*args, _key=name, _fwd=fwd, **kwargs):
            out = _fwd(*args, **kwargs)
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                captures.setdefault(_key, t.detach().float().cpu())
            return out

        m.forward = wrapped


def build_params(config, grids, inv_freq, cu_seqlens = None, max_seqlen = None):
    params = {
        "causal": False,
        "grid_thw": torch.tensor(grids, dtype = torch.int),
        "inv_freq": inv_freq,
    }
    if cu_seqlens is not None:
        params["cu_seqlens"] = cu_seqlens
        params["max_seqlen"] = max_seqlen
    return params


def run_pass(vision_model: Model, device: str, pixel_tensor: torch.Tensor, params: dict):
    captures = {}
    attach_captures(vision_model, captures)
    vision_model.load(device = device)
    try:
        with torch.no_grad():
            out = vision_model.forward(pixel_tensor.to(device), params = params)
    finally:
        vision_model.unload()
    return out.float().cpu(), captures


def pack_inputs(v, pixel_tensors, grids):
    """Pack separately preprocessed inputs into one varlen batch: concatenated tokens,
    stacked RoPE tables, one segment per input grid."""
    packed = torch.cat([t.unsqueeze(0) for t in pixel_tensors], dim = 1)
    inv_freq = torch.cat([
        qwen2_position_embedding_grid_2d(g, v.head_dim, v.spatial_merge_size, v.rope_theta)
        for g in grids
    ], dim = 0)
    lens = [g[0] * g[1] * g[2] for g in grids]
    bounds = [0]
    for length in lens:
        bounds.append(bounds[-1] + length)
    cu_seqlens = torch.tensor(bounds, dtype = torch.int32)
    return packed, inv_freq, cu_seqlens, max(lens)
def frame_boundaries(grid):
    """cu_seqlens for a single clip grid: one attention segment per frame."""
    tt, hh, ww = grid
    return torch.tensor([i * hh * ww for i in range(tt)] + [tt * hh * ww], dtype = torch.int32)




def load_image_inputs(config, prep_model, paths):
    tensors, grids, labels = [], [], []
    for p in paths:
        img = Image.open(p)
        t, _, g = prep_model.preprocess(img)
        tensors.append(t)
        grids.append(g)
        labels.append(f"{os.path.basename(p)} {img.size}-> {g}")
    packed, inv_freq, cu, max_len = pack_inputs(prep_model.config.vision, tensors, grids)
    print("input: multi-image")
    for lb in labels:
        print(f"  {lb}")
    print(f"  packed pixel tensor {tuple(packed.shape)} | cu_seqlens {cu.tolist()}")
    params = build_params(config, grids, inv_freq, cu, max_len)
    return packed, params


def load_video_inputs(config, prep_model, image_path, num_frames):
    # Deterministic pseudo-frames: constant brightness ramp over copies of the source image.
    src = np.asarray(Image.open(image_path).convert("RGB")).astype(np.int16)
    frames = [
        Image.fromarray(np.clip(src + (i - num_frames // 2) * 6, 0, 255).astype(np.uint8))
        for i in range(num_frames)
    ]
    t, _, grid = prep_model.preprocess(frames)
    t = t.unsqueeze(0)
    v = prep_model.config.vision
    tt, hh, ww = grid
    inv_freq = qwen2_position_embedding_grid_2d(grid, v.head_dim, v.spatial_merge_size, v.rope_theta)
    # HF convention (merge_temporal=False): one attention segment per frame.
    cu = frame_boundaries(grid)
    print(f"input: video | {num_frames} frames | grid {grid} | "
          f"pixel tensor {tuple(t.shape)} | cu_seqlens {cu.tolist()}")
    params = build_params(config, [grid], inv_freq, cu, hh * ww)
    return t, params

def compare(config, pixel_tensor, params):
    model_gpu = Model.from_config(config, component = "vision")
    model_cpu = Model.from_config(config, component = "vision")
    out_gpu, cap_gpu = run_pass(model_gpu, ARGS.device, pixel_tensor, params)
    out_cpu, cap_cpu = run_pass(model_cpu, "cpu", pixel_tensor, params)

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
    return all_pass


def timed_forward(model, device, pixel_tensor, params, repeats = 3):
    model.load(device = device)
    try:
        pt = pixel_tensor.to(device)
        with torch.no_grad():
            for _ in range(repeats):
                model.forward(pt, params = params)
        # Wall time of a single additional pass, reported separately from warmup.
        torch.cuda.synchronize(device) if device.startswith("cuda") else None
        t0 = time.perf_counter()
        with torch.no_grad():
            model.forward(pt, params = params)
        torch.cuda.synchronize(device) if device.startswith("cuda") else None
        return (time.perf_counter() - t0) * 1000.0
    finally:
        model.unload()


def bench(config, prep_model):
    """Timing-only sweep over synthetic packed shapes. Pixel content is random; shapes are
    what matter: total tokens scale, attention cost stays bounded by per-segment length."""
    v = config.vision
    # Reference patch width from a real preprocess call.
    ref_t, _, _ = prep_model.preprocess(Image.open(ARGS.image))
    d_in = ref_t.shape[1]
    del prep_model

    gen = torch.Generator().manual_seed(0)

    shapes = (
        [("images", f"{n} x (1,50,40)", [(1, 50, 40)] * n) for n in (1, 4, 16)] +
        [("video", f"t={t_} x (1,50,40)", [(t_, 50, 40)]) for t_ in (8, 32)]
    )

    def synth(mode, grids):
        tensors = [torch.randn(g[0] * g[1] * g[2], d_in, generator = gen).half() for g in grids]
        packed, inv_freq, cu, max_len = pack_inputs(v, tensors, grids)
        if mode == "video":
            # One segment per frame inside the single clip grid, matching load_video_inputs.
            cu = frame_boundaries(grids[0])
            max_len = grids[0][1] * grids[0][2]
        return packed, inv_freq, cu, max_len

    print(f"\n{'mode':<6} {'shape':<18} {'tokens':>8} {'segments':>9} "
          f"{'cpu ms':>10} {'gpu ms':>10}")
    for mode, label, grids in shapes:
        packed, inv_freq, cu, max_len = synth(mode, grids)
        params = build_params(config, grids, inv_freq, cu, max_len)
        tokens = packed.shape[1]
        segments = len(cu) - 1
        cpu_model = Model.from_config(config, component = "vision")
        cpu_ms = timed_forward(cpu_model, "cpu", packed, params)
        gpu_ms = float("nan")
        if torch.cuda.is_available():
            gpu_model = Model.from_config(config, component = "vision")
            gpu_ms = timed_forward(gpu_model, ARGS.device, packed, params)
        print(f"{mode:<6} {label:<18} {tokens:>8} {segments:>9} {cpu_ms:>10.1f} {gpu_ms:>10.1f}")



def profile_leg(config, prep_model):
    """Op-attribution sweep over the same synthetic packed shapes as --bench: one warmup
    pass, then a single CPU forward under torch.profiler with self CPU time aggregated into
    coarse categories (GEMM / attention / cast / other). Also measures serial preprocess
    cost for a 16-image list, which sizes the threaded-preprocessing lever."""
    from torch.profiler import ProfilerActivity, profile

    v = config.vision
    ref_t, _, _ = prep_model.preprocess(Image.open(ARGS.image))
    d_in = ref_t.shape[1]
    del ref_t

    gen = torch.Generator().manual_seed(0)
    shapes = (
        [("images", f"{n} x (1,50,40)", [(1, 50, 40)] * n) for n in (1, 4, 16)] +
        [("video", f"t={t_} x (1,50,40)", [(t_, 50, 40)]) for t_ in (8, 32)]
    )

    def synth(mode, grids):
        tensors = [torch.randn(g[0] * g[1] * g[2], d_in, generator = gen).half() for g in grids]
        packed, inv_freq, cu, max_len = pack_inputs(v, tensors, grids)
        if mode == "video":
            # One segment per frame inside the single clip grid, matching load_video_inputs.
            cu = frame_boundaries(grids[0])
            max_len = grids[0][1] * grids[0][2]
        return packed, inv_freq, cu, max_len

    def categorize(prof):
        cats = {"gemm": 0.0, "attn": 0.0, "cast": 0.0, "other": 0.0}
        for evt in prof.key_averages():
            us = evt.self_cpu_time_total
            if evt.key in ("aten::mm", "aten::addmm"):
                cats["gemm"] += us
            elif "_scaled_dot_product" in evt.key:
                cats["attn"] += us
            elif evt.key in ("aten::to", "aten::_to_copy", "aten::copy_"):
                cats["cast"] += us
            else:
                cats["other"] += us
        return {k: us / 1000.0 for k, us in cats.items()}

    print(f"\n{'mode':<6} {'shape':<18} {'tokens':>8} {'gemm ms':>9} {'attn ms':>9} "
          f"{'cast ms':>9} {'other ms':>10} {'wall ms':>9}")
    walls = {}
    for mode, label, grids in shapes:
        packed, inv_freq, cu, max_len = synth(mode, grids)
        params = build_params(config, grids, inv_freq, cu, max_len)
        model = Model.from_config(config, component = "vision")
        model.load("cpu")
        try:
            pt = packed.cpu()
            with torch.no_grad():
                model.forward(pt, params = params)  # warmup pass, timed_forward convention
            with profile(activities = [ProfilerActivity.CPU]) as prof:
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.forward(pt, params = params)
                wall_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            model.unload()
        cats = categorize(prof)
        walls[label] = wall_ms
        print(f"{mode:<6} {label:<18} {packed.shape[1]:>8} {cats['gemm']:>9.1f} "
              f"{cats['attn']:>9.1f} {cats['cast']:>9.1f} {cats['other']:>10.1f} {wall_ms:>9.1f}")

    # Serial preprocess cost for a 16-image list: decides whether threaded preprocessing
    # is worth its complexity (gate: < 10% of the batch tower wall -> skip).
    n_prep = 16
    t0 = time.perf_counter()
    for _ in range(n_prep):
        prep_model.preprocess(Image.open(ARGS.image))
    prep_ms = (time.perf_counter() - t0) * 1000.0
    tower_ms = walls.get("16 x (1,50,40)", float("nan"))
    print(f"\npreprocess: {n_prep} images serial wall {prep_ms:.1f} ms "
          f"(16-image tower wall {tower_ms:.1f} ms)")


def main():
    global ARGS
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model_dir", type = str, required = True)
    parser.add_argument("--image", type = str, default = "examples/media/cat.png")
    parser.add_argument("--images", type = str, nargs = "+", default = None,
                        help = "multi-image mode: pack these images into one varlen batch")
    parser.add_argument("--video_frames", type = int, default = None,
                        help = "video mode: build this many pseudo-frames from --image")
    leg = parser.add_mutually_exclusive_group()
    leg.add_argument("--bench", action = "store_true",
                     help = "timing-only sweep over synthetic packed shapes")
    leg.add_argument("--profile", action = "store_true",
                     help = "CPU op-attribution sweep over the same synthetic shapes")
    parser.add_argument("--device", type = str, default = "cuda:0")
    ARGS = parser.parse_args()

    print(f"torch {torch.__version__} | comparing vision encoder: CUDA ({ARGS.device}) vs CPU")

    config = Config.from_directory(ARGS.model_dir)

    if ARGS.bench:
        prep_model = Model.from_config(config, component = "vision")
        bench(config, prep_model)
        return

    if ARGS.profile:
        prep_model = Model.from_config(config, component = "vision")
        profile_leg(config, prep_model)
        return

    # Preprocess exactly once per distinct input; both passes consume identical tensors.
    prep_model = Model.from_config(config, component = "vision")

    if ARGS.images:
        pixel_tensor, params = load_image_inputs(config, prep_model, ARGS.images)
    elif ARGS.video_frames:
        pixel_tensor, params = load_video_inputs(config, prep_model, ARGS.image, ARGS.video_frames)
    else:
        image = Image.open(ARGS.image)
        t, _, grid = prep_model.preprocess(image)
        pixel_tensor = t.unsqueeze(0)
        inv_freq = qwen2_position_embedding_grid_2d(
            grid, config.vision.head_dim, config.vision.spatial_merge_size, config.vision.rope_theta
        )
        params = build_params(config, [grid], inv_freq)
        print(f"input: {ARGS.image} | pixel tensor {tuple(pixel_tensor.shape)} "
              f"{pixel_tensor.dtype} | grid {grid}")

    ok = compare(config, pixel_tensor, params)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
