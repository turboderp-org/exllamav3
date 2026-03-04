#!/usr/bin/env python3
"""
Lightweight smoke checks for Qwen3.5 architecture integration.

Checks:
1) Config/architecture resolution
2) Model graph construction
3) One linear-attn block forward
4) One full-attn block forward

Optional:
5) Attempt full model load (can fail on low VRAM for FP16 checkpoints)
"""

import argparse
import sys
import torch

from exllamav3 import Config, Model


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--full_load", action="store_true")
    ap.add_argument("--reserve_per_device", default=None, help="e.g. 0.25,0.25")
    args = ap.parse_args()

    cfg = Config.from_directory(args.model_dir)
    print("[INFO] architecture:", cfg.architecture)
    if cfg.architecture not in ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"):
        fail(f"Unexpected architecture: {cfg.architecture}")

    model = Model.from_config(cfg)
    print("[INFO] model graph ok")

    # Verify split projection path exists for linear attention blocks
    first_block = model.modules[model.first_block_idx]
    if not hasattr(first_block.attn, "qkv_proj") and not hasattr(first_block.attn, "qkvz_proj"):
        fail("Expected Qwen3.5 linear attention projection attributes not found")

    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")

    device = torch.device(args.device)
    hidden_size = cfg.hidden_size

    # First linear-attn block
    idx_linear = model.first_block_idx
    blk_linear = model.modules[idx_linear]
    blk_linear.load(device)
    x = torch.randn((1, 8, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_linear.forward(x, params={})
    print("[INFO] linear block forward:", blk_linear.key, tuple(y.shape))
    blk_linear.unload()

    # First full-attn block
    if "full_attention" not in cfg.layer_types:
        fail("No full_attention layer found in layer_types")
    idx_full = model.first_block_idx + cfg.layer_types.index("full_attention")
    blk_full = model.modules[idx_full]
    blk_full.load(device)
    x = torch.randn((1, 8, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_full.forward(x, params={})
    print("[INFO] full block forward:", blk_full.key, tuple(y.shape))
    blk_full.unload()

    if args.full_load:
        reserve = None
        if args.reserve_per_device:
            reserve = [float(x) for x in args.reserve_per_device.split(",")]
        print("[INFO] attempting full model load")
        try:
            model.load(reserve_per_device=reserve)
            print("[INFO] full model load ok")
            model.unload()
        except Exception as e:
            print("[WARN] full model load failed:", repr(e))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PASS] qwen3.5 smoke checks complete")


if __name__ == "__main__":
    main()
