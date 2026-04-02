#!/usr/bin/env python3
"""
Lightweight smoke checks for NemotronH architecture integration.

Checks:
1) Config/architecture resolution
2) Model graph construction
3) One Mamba block forward
4) Mamba recurrent state consistency (full sequence vs token-by-token)
5) One MoE block forward
6) One attention block forward

Optional:
7) Attempt full model load
"""

import argparse
import sys
import torch

from exllamav3 import Config, Model


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def first_layer_idx(layer_types: list[str], name: str) -> int:
    try:
        return layer_types.index(name)
    except ValueError as exc:
        raise RuntimeError(f"No {name} layer found") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--full_load", action="store_true")
    ap.add_argument("--reserve_per_device", default=None, help="e.g. 0.25,0.25")
    args = ap.parse_args()

    cfg = Config.from_directory(args.model_dir)
    print("[INFO] architecture:", cfg.architecture)
    if cfg.architecture != "NemotronHForCausalLM":
        fail(f"Unexpected architecture: {cfg.architecture}")

    model = Model.from_config(cfg)
    print("[INFO] model graph ok")

    if not torch.cuda.is_available():
        fail("CUDA is required for this smoke check")

    device = torch.device(args.device)
    hidden_size = cfg.hidden_size
    tol = 5e-2

    idx_mamba = model.first_block_idx + first_layer_idx(cfg.layer_types, "mamba")
    blk_mamba = model.modules[idx_mamba]
    blk_mamba.load(device)
    x = torch.randn((1, args.seq_len, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y_full = blk_mamba.forward(x, params={})
    print("[INFO] mamba block forward:", blk_mamba.key, tuple(y_full.shape))

    rs = blk_mamba.attn.new_recurrent_state()
    params = {"recurrent_states": {blk_mamba.layer_idx: rs}}
    y_steps = []
    with torch.inference_mode():
        for i in range(args.seq_len):
            y_steps.append(blk_mamba.forward(x[:, i:i+1, :], params=params))
    y_step = torch.cat(y_steps, dim=1)
    diff = (y_full - y_step).abs().max().item()
    print("[INFO] mamba recurrent max diff:", diff)
    if diff > tol:
        fail(f"Mamba recurrent consistency check failed: diff={diff}")
    blk_mamba.unload()

    idx_moe = model.first_block_idx + first_layer_idx(cfg.layer_types, "moe")
    blk_moe = model.modules[idx_moe]
    blk_moe.load(device)
    x = torch.randn((1, args.seq_len, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_moe.forward(x, params={})
    print("[INFO] moe block forward:", blk_moe.key, tuple(y.shape))
    blk_moe.unload()

    idx_attn = model.first_block_idx + first_layer_idx(cfg.layer_types, "attention")
    blk_attn = model.modules[idx_attn]
    blk_attn.load(device)
    x = torch.randn((1, args.seq_len, hidden_size), device=device, dtype=torch.half)
    with torch.inference_mode():
        y = blk_attn.forward(x, params={"attn_mode": "flash_attn_nc", "position": 0})
    print("[INFO] attention block forward:", blk_attn.key, tuple(y.shape))
    blk_attn.unload()

    if args.full_load:
        reserve = None
        if args.reserve_per_device:
            reserve = [float(x) for x in args.reserve_per_device.split(",")]
        print("[INFO] attempting full model load")
        try:
            if reserve is not None:
                model.load(reserve_per_device=reserve)
            else:
                model.load(device=device)
            print("[INFO] full model load ok")
            model.unload()
        except Exception as e:
            print("[WARN] full model load failed:", repr(e))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[PASS] nemotron-h smoke checks complete")


if __name__ == "__main__":
    main()
