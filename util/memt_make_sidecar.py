"""
Assemble a MemT-1.0-Nano EXL3 serving directory from a stock EXL3 quant of Qwen3.6-35B-A3B and the MemT
files of an existing sidecar (or the bf16 release plus the fork's chat template).

    python util/memt_make_sidecar.py STOCK_QUANT_DIR OUT_DIR --memt-from EXISTING_SIDECAR_DIR

Everything from the stock quant is hardlinked unchanged (its README.md is left out); the MemT pieces
(configuration.json, generation_config.json, tokenizer_config.json, chat_template.jinja,
memt_block.safetensors) come from --memt-from, and config.json is the MemT config with the stock quant's
quantization_config. The backbone weights are byte-identical to the stock quant, no requant.
"""

import argparse, json, os, shutil, sys

MEMT_FILES = ["configuration.json", "generation_config.json", "tokenizer_config.json",
              "chat_template.jinja", "memt_block.safetensors"]
SKIP_STOCK = {"README.md", ".gitattributes", ".cache"}


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stock")
    ap.add_argument("out")
    ap.add_argument("--memt-from", required = True, help = "an existing MemT sidecar directory")
    a = ap.parse_args()
    for f in MEMT_FILES:
        assert os.path.exists(os.path.join(a.memt_from, f)), f"{a.memt_from} lacks {f}"
    os.makedirs(a.out, exist_ok = False)
    n = 0
    for f in sorted(os.listdir(a.stock)):
        if f in SKIP_STOCK or f in MEMT_FILES or f.endswith(".log"):
            continue
        link_or_copy(os.path.join(a.stock, f), os.path.join(a.out, f))
        n += 1
    for f in MEMT_FILES:
        link_or_copy(os.path.join(a.memt_from, f), os.path.join(a.out, f))
    # config.json: the MemT config (model_type memt_qwen3_5_moe, no transformers auto_map) carrying THIS
    # quant's quantization_config (bits / head_bits differ per bpw)
    cfg = json.load(open(os.path.join(a.memt_from, "config.json")))
    cfg.pop("auto_map", None)
    stock_cfg = json.load(open(os.path.join(a.stock, "config.json")))
    assert "quantization_config" in stock_cfg, "stock dir is not an EXL3 quant (no quantization_config)"
    cfg["quantization_config"] = stock_cfg["quantization_config"]
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent = 2)
    print(f"{a.out}: {n} stock files + {len(MEMT_FILES) + 1} MemT files; "
          f"quant bits {cfg['quantization_config'].get('bits')} / head {cfg['quantization_config'].get('head_bits')}")


if __name__ == "__main__":
    main()
