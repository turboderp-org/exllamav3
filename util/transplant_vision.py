"""
Transplant the vision component from one converted model into another, without requantizing
either. Useful when text and vision towers were quantized in separate jobs (e.g. an older text
quant plus a newer conversion made with --vision_bits), or to try different vision bitrates
against the same text weights.

    python util/transplant_vision.py -m <target_dir> -d <donor_dir> -o <output_dir>

The output directory receives every non-vision tensor from the target model, the vision tensors
from the donor, a rebuilt safetensors index, and the target's config.json with the donor's
quantization_config->vision_bits carried over (or removed, if the donor's vision is
unquantized). Non-tensor files (tokenizer, chat template, ...) are copied from the target.

Vision tensors are identified architecture-generically: each model's vision component class
lists its tensors via get_additional_compiled_tensors() on that model's own tensor collection,
so both fp16 and EXL3-quantized vision layouts are matched. Both models must share the same
architecture, and the donor must contain a vision component. The target and donor are only
read; the output directory must not already contain a model.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from exllamav3 import Config


def vision_tensor_names(model_dir: str) -> (Config, set):
    config = Config.from_directory(model_dir)
    if "vision" not in config.model_classes:
        return config, set()
    cls = config.model_classes["vision"]
    return config, set(cls.get_additional_compiled_tensors(config).keys())


def shard_map(model_dir: str) -> dict:
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return json.load(open(index_path))["weight_map"]
    single = os.path.join(model_dir, "model.safetensors")
    assert os.path.exists(single), f"No safetensors found in {model_dir}"
    with safe_open(single, framework = "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def main():
    parser = argparse.ArgumentParser(description = "Transplant a vision component between converted models")
    parser.add_argument("-m", "--model_dir", type = str, required = True, help = "Target model directory (text weights kept)")
    parser.add_argument("-d", "--donor_dir", type = str, required = True, help = "Donor model directory (vision weights taken)")
    parser.add_argument("-o", "--out_dir", type = str, required = True, help = "Output directory")
    parser.add_argument("-ss", "--shard_size", type = int, default = 8192, help = "Max output shard size in MB, default: 8192")
    args = parser.parse_args()

    max_shard = args.shard_size * 1024 ** 2

    target_config, target_vision = vision_tensor_names(args.model_dir)
    donor_config, donor_vision = vision_tensor_names(args.donor_dir)
    assert target_config.architecture == donor_config.architecture, \
        f"Architecture mismatch: {target_config.architecture} vs {donor_config.architecture}"
    assert donor_vision, "Donor model has no vision component"

    target_map = shard_map(args.model_dir)
    donor_map = shard_map(args.donor_dir)
    assert target_vision <= set(target_map), "Target vision tensors missing from its shards"
    assert donor_vision <= set(donor_map), "Donor vision tensors missing from its shards"
    expected = (set(target_map) - target_vision) | donor_vision

    print(f" -- Target: {len(target_map)} tensors ({len(target_vision)} vision, dropped)")
    print(f" -- Donor:  {len(donor_vision)} vision tensors")
    print(f" -- Output: {len(expected)} tensors")

    os.makedirs(args.out_dir, exist_ok = True)
    assert not any(f.endswith(".safetensors") for f in os.listdir(args.out_dir)), \
        f"Output directory already contains a model: {args.out_dir}"

    # Stream-repack: target non-vision tensors in original shard order, then donor vision tensors
    out_files = []
    accum = {}
    accum_size = 0
    total_size = 0
    map_dict = {}

    def flush():
        nonlocal accum, accum_size
        if not accum:
            return
        fn = f"model-tmp-{len(out_files):05}.safetensors"
        print(f" -- Writing {fn} ({accum_size / 1024**2:.0f} MB, {len(accum)} tensors)")
        save_file(accum, os.path.join(args.out_dir, fn))
        for k in accum:
            map_dict[k] = fn
        out_files.append(fn)
        accum = {}
        accum_size = 0

    def add(k, t):
        nonlocal accum_size, total_size
        size = t.nelement() * t.element_size()
        if accum_size + size > max_shard and accum_size > 0:
            flush()
        accum[k] = t.contiguous()
        accum_size += size
        total_size += size

    jobs = (
        (args.model_dir, target_map, lambda k: k not in target_vision),
        (args.donor_dir, donor_map, lambda k: k in donor_vision),
    )
    for d, wmap, keep in jobs:
        for shard in sorted(set(wmap.values())):
            with safe_open(os.path.join(d, shard), framework = "pt") as f:
                for k in f.keys():
                    if keep(k):
                        add(k, f.get_tensor(k))
    flush()

    assert set(map_dict) == expected, "Output tensor set mismatch"

    # Rename shards now that the count is known
    num_files = len(out_files)
    final_names = {}
    for i, fn in enumerate(out_files):
        new_fn = (
            "model.safetensors" if num_files == 1 else
            f"model-{i + 1:05}-of-{num_files:05}.safetensors"
        )
        os.rename(os.path.join(args.out_dir, fn), os.path.join(args.out_dir, new_fn))
        final_names[fn] = new_fn
    map_dict = {k: final_names[v] for k, v in map_dict.items()}

    if num_files > 1:
        with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps({
                "metadata": {"total_size": total_size},
                "weight_map": map_dict,
            }, indent = 4))

    # Target config with the donor's vision_bits carried over
    config_dict = json.load(open(os.path.join(args.model_dir, "config.json")))
    donor_config_dict = json.load(open(os.path.join(args.donor_dir, "config.json")))
    donor_vb = donor_config_dict.get("quantization_config", {}).get("vision_bits")
    qcfg = config_dict.get("quantization_config", {})
    if donor_vb is not None:
        qcfg["vision_bits"] = donor_vb
    else:
        qcfg.pop("vision_bits", None)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        f.write(json.dumps(config_dict, indent = 4))

    # Non-tensor files from the target
    for fn in os.listdir(args.model_dir):
        p = os.path.join(args.model_dir, fn)
        if (
            os.path.isfile(p) and not fn.endswith(".safetensors")
            and fn not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy(p, os.path.join(args.out_dir, fn))

    print(f" -- Done: {num_files} shard(s), {total_size / 1024**3:.2f} GB, "
          f"vision_bits: {donor_vb if donor_vb is not None else 'unquantized'}")


if __name__ == "__main__":
    main()
