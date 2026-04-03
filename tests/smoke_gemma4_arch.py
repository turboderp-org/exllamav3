#!/usr/bin/env python3
"""
Lightweight smoke checks for Gemma4 text architecture integration.

Checks:
1) Config/architecture resolution
2) Model graph construction
3) One sliding-attn block forward
4) One full-attn block forward
5) One MoE block forward when enabled

Optional:
6) Attempt full model load
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
    if cfg.architecture != "Gemma4ForConditionalGeneration":
        fail(f"Unexpected architecture: {cfg.architecture}")

    model = Model.from_config(cfg)
    print("[INFO] model graph ok")

    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")

    device = torch.device(args.device)
    hidden_size = cfg.hidden_size

    idx_sliding = model.first_block_idx + cfg.layer_types.index("sliding_attention")
    blk_sliding = model.modules[idx_sliding]
    blk_sliding.load(device)
    if blk_sliding.attn.v_proj is None:
        fail("Expected sliding attention to have a v_proj")
    x = torch.randn((1, 8, hidden_size), device = device, dtype = torch.half)
    with torch.inference_mode():
        y = blk_sliding.forward(x, params = {})
    print("[INFO] sliding block forward:", blk_sliding.key, tuple(y.shape))
    blk_sliding.unload()

    idx_full = model.first_block_idx + cfg.layer_types.index("full_attention")
    blk_full = model.modules[idx_full]
    blk_full.load(device)
    if blk_full.attn.v_proj is not None:
        fail("Expected Gemma4 full attention to omit v_proj")
    x = torch.randn((1, 8, hidden_size), device = device, dtype = torch.half)
    with torch.inference_mode():
        y = blk_full.forward(x, params = {})
    print("[INFO] full block forward:", blk_full.key, tuple(y.shape))
    blk_full.unload()

    if cfg.enable_moe_block:
        idx_moe = model.first_block_idx
        blk_moe = model.modules[idx_moe]
        blk_moe.load(device)
        x = torch.randn((1, 8, hidden_size), device = device, dtype = torch.half)
        with torch.inference_mode():
            y = blk_moe.forward(x, params = {})
        print("[INFO] moe block forward:", blk_moe.key, tuple(y.shape))
        blk_moe.unload()

    if args.full_load:
        reserve = None
        if args.reserve_per_device:
            reserve = [float(x) for x in args.reserve_per_device.split(",")]
        print("[INFO] attempting full model load")
        try:
            model.load(reserve_per_device = reserve)
            print("[INFO] full model load ok")
            model.unload()
        except Exception as e:
            print("[WARN] full model load failed:", repr(e))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PASS] gemma4 smoke checks complete")


if __name__ == "__main__":
    main()
