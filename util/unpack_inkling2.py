"""Write a BF16 copy of Inkling-2-Mini with the routed experts unpacked per expert, as input for exllamav3.

The original checkpoint stores the experts fused and quantized (NVFP4 on 20 layers, MXFP8 on 4) with gate/up rows
interleaved. This writes, per layer N and expert E,
    model.llm.layers.N.mlp.experts.E.{gate_proj,up_proj}.weight  (intermediate, hidden)  bf16
    model.llm.layers.N.mlp.experts.E.down_proj.weight            (hidden, intermediate)  bf16
using the dequantization of HF transformers (inkling2_loop_math.dequant_experts). Every other tensor
(language model, vision tower) is copied unchanged; mtp.safetensors and the tokenizer files are copied as is;
quantization_config is dropped from config.json.

Usage: python util/unpack_inkling2.py SRC_DIR DST_DIR

Then convert DST_DIR as usual (convert.py -i DST_DIR ...) and copy mtp.safetensors from SRC_DIR to the output
if the DSpark MTP head is wanted (it is not quantized).
"""
from __future__ import annotations
import json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3", "architecture"))
from inkling2_loop_math import dequant_experts, deinterleave_rows  # noqa: E402

SHARD_BYTES = 4 << 30


def main(src, dst):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(n for n in os.listdir(src) if n.startswith("model") and n.endswith(".safetensors"))
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists")
    if not files:
        raise SystemExit(f"no model*.safetensors in {src}")
    handles = [safe_open(os.path.join(src, n), "pt", device="cpu") for n in files]
    where = {k: h for h in handles for k in h.keys()}
    keys = list(where)

    class _All:  # one reader over every shard
        def get_tensor(self, k):
            return where[k].get_tensor(k)

    f = _All()
    os.makedirs(dst)
    expert_prefixes = sorted({k[: k.index(".experts.") + len(".experts.")] for k in keys if ".mlp.experts." in k})

    weight_map, shard, shard_bytes, n_shard = {}, {}, 0, 0

    def flush():
        nonlocal shard, shard_bytes, n_shard
        if not shard:
            return
        n_shard += 1
        name = f"model-{n_shard:05d}.safetensors"
        save_file(shard, os.path.join(dst, name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        print(f"wrote {name}: {len(shard)} tensors, {shard_bytes / 2**30:.2f} GiB", flush=True)
        shard, shard_bytes = {}, 0

    def add(k, t):
        nonlocal shard_bytes
        shard[k] = t.contiguous()
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes >= SHARD_BYTES:
            flush()

    for k in keys:
        if ".mlp.experts." not in k:
            add(k, f.get_tensor(k))

    for e in expert_prefixes:
        g = lambda n: f.get_tensor(e + n).to(dev) if (e + n) in where else None
        w13 = deinterleave_rows(dequant_experts(g("w13_weight"), g("w13_weight_scale"),
                                                g("w13_weight_amax"), g("w13_weight_outer_scale_denominator")))
        w2 = dequant_experts(g("w2_weight"), g("w2_weight_scale"),
                             g("w2_weight_amax"), g("w2_weight_outer_scale_denominator"))
        inter = w13.shape[1] // 2
        assert w2.shape[2] == inter, (e, w13.shape, w2.shape)
        for x in range(w13.shape[0]):
            add(f"{e}{x}.gate_proj.weight", w13[x, :inter].cpu())
            add(f"{e}{x}.up_proj.weight", w13[x, inter:].cpu())
            add(f"{e}{x}.down_proj.weight", w2[x].cpu())
        print(f"unpacked {e} ({w13.shape[0]} experts, intermediate {inter})", flush=True)
        del w13, w2
    flush()

    total = sum(os.path.getsize(os.path.join(dst, n)) for n in set(weight_map.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg.pop("quantization_config", None)
    cfg["dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if name.startswith("model") or name in ("config.json", "hf_quant_config.json", "ORIGINAL.txt") or name.startswith("."):
            continue
        (shutil.copytree if os.path.isdir(p) else shutil.copy2)(p, os.path.join(dst, name))
    print("UNPACK-FIN")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
