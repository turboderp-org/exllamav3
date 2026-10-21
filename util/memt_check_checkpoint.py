"""
Check a MemT-1.0-Nano HF checkpoint against what exl3-qlora's MemT support assumes
(doc/memt_plan.md section 6), and optionally against a Qwen3.6-35B-A3B checkpoint to see
whether the backbone tensors are byte-identical (which enables the sidecar route: an
existing EXL3 quant of Qwen3.6-35B-A3B + the memory tensors, no requantization).

    python util/memt_check_checkpoint.py /models/MemT-1.0-Nano
    python util/memt_check_checkpoint.py /models/MemT-1.0-Nano --backbone /models/Qwen3.6-35B-A3B
    python util/memt_check_checkpoint.py /models/MemT-1.0-Nano --backbone ... --sidecar memt_memory.safetensors

Reads safetensors headers only for the shape/dtype checks; --backbone hashes the tensor bytes
(reads both checkpoints once). --sidecar writes the memory tensors into one safetensors file
to drop next to an EXL3 quant of the backbone (see the plan doc for the config.json rewrite).
No torch needed except for --sidecar.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import struct
import sys

MEMT_RE = ".self_attn.memt."


def read_headers(directory: str) -> dict[str, tuple[str, dict]]:
    """key -> (filename, header entry) over every *.safetensors in directory."""
    out = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(directory, fn)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            out[k] = (fn, v)
    return out


def tensor_hash(directory: str, fn: str, entry: dict) -> str:
    path = os.path.join(directory, fn)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        beg, end = entry["data_offsets"]
        f.seek(8 + n + beg)
        h = hashlib.sha256()
        remaining = end - beg
        while remaining:
            chunk = f.read(min(remaining, 1 << 24))
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def expected_memory_shapes(cfg: dict) -> dict[str, list[int]]:
    tc = cfg["backbone_configs"]["text_config"]
    H, D, hid = tc["num_attention_heads"], tc["head_dim"], tc["hidden_size"]
    return {
        "in_proj.weight": [5 * H * D + 2 * D, hid],
        "f_down.weight": [H * D, D],
        "g_down.weight": [H * D, D],
        "decay_scale_log": [H],
        "decay_bias": [H * D],
        "o_norm.weight": [D],
    }


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("memt_dir")
    ap.add_argument("--backbone", default = None, help = "Qwen3.6-35B-A3B HF checkpoint to compare the backbone tensors against")
    ap.add_argument("--sidecar", default = None, help = "write the memory tensors to this safetensors file")
    args = ap.parse_args()

    with open(os.path.join(args.memt_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg.get("model_type") == "memt_qwen3_5_moe", cfg.get("model_type")
    tc = cfg["backbone_configs"]["text_config"]
    layer_types = tc.get("layer_types") or [
        "full_attention" if (i + 1) % tc.get("full_attention_interval", 4) == 0 else "linear_attention"
        for i in range(tc["num_hidden_layers"])]
    mem_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    print(f"memory layers: {mem_layers}")

    keys = read_headers(args.memt_dir)
    mem_keys = {k: v for k, v in keys.items() if MEMT_RE in k}
    bb_keys = {k: v for k, v in keys.items() if MEMT_RE not in k}
    print(f"{len(keys)} tensors: {len(mem_keys)} memory, {len(bb_keys)} backbone/other")

    ok = True
    exp = expected_memory_shapes(cfg)
    total = 0
    for i in mem_layers:
        for suffix, shape in exp.items():
            k = f"model.language_model.layers.{i}.self_attn.memt.{suffix}"
            if k not in mem_keys:
                print(f"  MISSING {k}")
                ok = False
                continue
            fn, e = mem_keys[k]
            n = 1
            for s in e["shape"]:
                n *= s
            total += n
            if list(e["shape"]) != shape or e["dtype"] != "BF16":
                print(f"  MISMATCH {k}: {e['dtype']} {e['shape']} (expected BF16 {shape})")
                ok = False
    extra = [k for k in mem_keys if int(k.split(".layers.")[1].split(".")[0]) not in mem_layers
             or k.split(".memt.")[1] not in exp]
    for k in extra:
        print(f"  UNEXPECTED memory tensor {k}")
        ok = False
    print(f"memory tensors {'OK' if ok else 'PROBLEMS'}: {total:,} parameters (paper Table 6: 450,931,360)")

    others = sorted(k for k in bb_keys if not k.startswith("model.language_model.") and k != "lm_head.weight")
    if others:
        print(f"other tensors (ignored by the loader unless an MTP/vision model is registered): {others[:5]}{' ...' if len(others) > 5 else ''}")

    if args.backbone:
        bkeys = read_headers(args.backbone)
        shared = [k for k in bb_keys if k in bkeys]
        only_m = [k for k in bb_keys if k not in bkeys]
        only_b = [k for k in bkeys if k not in bb_keys]
        print(f"backbone compare: {len(shared)} shared, {len(only_m)} only in MemT, {len(only_b)} only in backbone")
        if only_m: print(f"  only in MemT: {only_m[:5]}{' ...' if len(only_m) > 5 else ''}")
        if only_b: print(f"  only in backbone: {only_b[:5]}{' ...' if len(only_b) > 5 else ''}")
        diff = 0
        for n_, k in enumerate(shared):
            fm, em = bb_keys[k]
            fb, eb = bkeys[k]
            if em["shape"] != eb["shape"] or em["dtype"] != eb["dtype"]:
                print(f"  SHAPE/DTYPE {k}: {em['dtype']} {em['shape']} vs {eb['dtype']} {eb['shape']}")
                diff += 1
                continue
            if tensor_hash(args.memt_dir, fm, em) != tensor_hash(args.backbone, fb, eb):
                print(f"  DIFFERS {k}")
                diff += 1
            if n_ % 200 == 0:
                print(f"  ... {n_}/{len(shared)}", file = sys.stderr)
        print("backbone tensors BYTE-IDENTICAL -> sidecar route is valid" if diff == 0 and not only_b
              else f"backbone differs in {diff} tensors -> convert the MemT checkpoint directly")

    if args.sidecar:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        tensors = {}
        by_file: dict[str, list[str]] = {}
        for k, (fn, _) in mem_keys.items():
            by_file.setdefault(fn, []).append(k)
        for fn, ks in by_file.items():
            with safe_open(os.path.join(args.memt_dir, fn), framework = "pt", device = "cpu") as f:
                for k in ks:
                    tensors[k] = f.get_tensor(k).contiguous()
        save_file(tensors, args.sidecar, metadata = {"format": "pt", "source": "MemT-1.0-Nano memory tensors"})
        print(f"wrote {len(tensors)} memory tensors to {args.sidecar}")


if __name__ == "__main__":
    main()
