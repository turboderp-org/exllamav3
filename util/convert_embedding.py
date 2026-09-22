"""
Convert the trunk embed_tokens tensor of an EXL3 model from 16-bit storage to
int8 with per-32-element block scales (q8_0 convention, 8.50 bpw). Rebuilds only
the shard containing the embedding tensor (raw byte re-pack: no re-quantization
of anything else, bit-exact for every kept tensor); all other files are
hardlinked (copied if the filesystem has no hardlinks). model.safetensors.index.json
(weight_map) and quantization_config.json (tensor_storage) are updated for
consistency with external tooling - the runtime ignores both and globs
*.safetensors in the directory.

    python util/convert_embedding.py -m <model_dir> -o <out_dir>

Quant math (identical to Embedding.convert_int8 in the runtime):
    d = max|block| / 127 in fp16 (zero blocks: d = 1, q = 0)
    q = round(w / d).clamp(-127, 127)  -> int8
    dequant: q.half() * d

Stdlib + torch only. The model is only read.
"""
import argparse
import glob
import json
import os
import shutil
import struct
import sys

import torch

CHUNK = 32 * 1024 * 1024
EMB_KEY = "model.language_model.embed_tokens"
BLOCK = 32


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n   # data_offsets are relative to the data section start


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def die(msg):
    print(f" ## {msg}", file=sys.stderr)
    sys.exit(1)


def quantize_int8(w: torch.Tensor):
    """w: fp16 [V, H] -> (q: int8 [V, H], scale: fp16 [V, H // BLOCK])"""
    assert w.dim() == 2 and w.shape[1] % BLOCK == 0
    assert w.dtype == torch.half
    nb = w.shape[1] // BLOCK
    wb = w.view(w.shape[0], nb, BLOCK)
    d = wb.abs().amax(dim = 2) / 127.0
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = torch.round(wb / d.unsqueeze(2)).clamp(-127, 127).to(torch.int8)
    return q.view(w.shape).contiguous(), d.contiguous()


def main():
    ap = argparse.ArgumentParser(description = "Convert embed_tokens to int8 per-32 in an EXL3 model")
    ap.add_argument("-m", "--model_dir", type = str, required = True, help = "Source model directory (read-only)")
    ap.add_argument("-o", "--out_dir", type = str, required = True, help = "Output directory (must not exist or be empty)")
    args = ap.parse_args()

    mdir = os.path.abspath(args.model_dir)
    odir = os.path.abspath(args.out_dir)
    assert os.path.isdir(mdir), f"Not a directory: {mdir}"
    assert mdir != odir, "Output dir must differ from model dir"
    if os.path.exists(odir):
        assert not os.listdir(odir), f"Output dir exists and is not empty: {odir}"

    # Tied-embedding models: lm_head aliases the embedding tensor; not supported
    cfg_path = os.path.join(mdir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding = "utf8") as f:
            cfg = json.load(f)
        tied = cfg.get("tie_word_embeddings")
        if tied is None and isinstance(cfg.get("text_config"), dict):
            tied = cfg["text_config"].get("tie_word_embeddings")
        if tied:
            die("tie_word_embeddings is true; not supported (lm_head aliases the embedding tensor)")
        if tied is None:
            print(" !! Warning: tie_word_embeddings not set in config.json; assuming untied "
                  "(some archs default it to True when absent)")

    st_files = sorted(glob.glob(os.path.join(mdir, "*.safetensors")))
    assert st_files, f"No .safetensors files in {mdir}"

    # Pre-scan all headers before writing anything; fail early
    headers = {}
    emb_file = None
    n_emb = 0
    for st in st_files:
        h, ds = read_header(st)
        headers[st] = (h, ds)
        keys = [k for k in h if k != "__metadata__"]
        if EMB_KEY + ".weight" in keys:
            emb_file = st
            n_emb += 1
        if EMB_KEY + ".weight_i8" in keys or EMB_KEY + ".scale" in keys:
            die(f"{os.path.basename(st)} already contains int8 embedding tensors; model appears already converted")
    if n_emb != 1:
        cands = sorted(k for h, _ in headers.values() for k in h
                       if k.endswith(".embed_tokens.weight") and k != "__metadata__")
        extra = f"; found instead: {cands}" if cands else "; no *.embed_tokens.weight tensor in any shard"
        die(f"Expected exactly one '{EMB_KEY}.weight' tensor, found {n_emb}{extra}")

    src = emb_file
    header, data_start = headers[src]
    e = header[EMB_KEY + ".weight"]
    if e["dtype"] not in ("BF16", "F16"):
        die(f"Embedding dtype is {e['dtype']}; expected BF16 or F16")
    shape = e["shape"]
    if len(shape) != 2 or shape[1] % BLOCK != 0:
        die(f"Embedding shape {shape}; expected 2-D with hidden size divisible by {BLOCK}")
    beg, end = e["data_offsets"]
    w_bytes = end - beg
    numel = shape[0] * shape[1]
    assert w_bytes == numel * 2, f"Embedding byte size {w_bytes} inconsistent with shape {shape}"

    # Load and quantize
    print(f" -- Loading {EMB_KEY}.weight from {os.path.basename(src)} ({w_bytes / 1024**2:.1f} MiB, {e['dtype']})")
    with open(src, "rb") as f:
        f.seek(data_start + beg)
        buf = bytearray(f.read(w_bytes))
    w = torch.frombuffer(buf, dtype = torch.bfloat16 if e["dtype"] == "BF16" else torch.float16,
                         count = numel).reshape(shape)
    if w.dtype != torch.half:
        w = w.to(torch.half)
    q, scale = quantize_int8(w)
    del w
    q_bytes = q.numel()
    s_bytes = scale.numel() * 2
    print(f" -- Quantized: int8 per-{BLOCK}  (weight_i8 {q_bytes / 1024**2:.1f} MiB, "
          f"scale {s_bytes / 1024**2:.1f} MiB, 8.50 bpw)")

    os.makedirs(odir, exist_ok=True)

    # Rebuild the embedding shard: keep every other tensor bit-exact in original order,
    # drop .weight, append .weight_i8 and .scale
    name = os.path.basename(src)
    keys = [k for k in header if k != "__metadata__"]
    keep = [k for k in keys if k != EMB_KEY + ".weight"]
    keep_bytes = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in keep)

    new_header = {}
    md = header.get("__metadata__")
    if md:
        new_header["__metadata__"] = md
    off = 0
    for k in keep:
        s_, e_ = header[k]["data_offsets"]
        new_header[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"],
                         "data_offsets": [off, off + (e_ - s_)]}
        off += e_ - s_
    new_header[EMB_KEY + ".weight_i8"] = {"dtype": "I8", "shape": shape, "data_offsets": [off, off + q_bytes]}
    off += q_bytes
    new_header[EMB_KEY + ".scale"] = {"dtype": "F16", "shape": [shape[0], shape[1] // BLOCK],
                                      "data_offsets": [off, off + s_bytes]}
    off += s_bytes

    out_path = os.path.join(odir, name)
    with open(out_path, "wb") as dst:
        blob = json.dumps(new_header).encode()
        blob += b" " * (-len(blob) % 8)   # keep the data section 8-aligned (safetensors spec)
        dst.write(struct.pack("<Q", len(blob)) + blob)
        with open(src, "rb") as f:
            for k in keep:
                s_, e_ = header[k]["data_offsets"]
                f.seek(data_start + s_)
                remaining = e_ - s_
                while remaining:
                    b = f.read(min(CHUNK, remaining))
                    dst.write(b)
                    remaining -= len(b)
        dst.write(q.numpy().tobytes())
        dst.write(scale.numpy().tobytes())

    # Verify the rebuilt file: header key set, exact size, and a byte-exact sample
    check, chk_data_start = read_header(out_path)
    want_keys = keep + [EMB_KEY + ".weight_i8", EMB_KEY + ".scale"]
    assert [k for k in check if k != "__metadata__"] == want_keys, f"Key set mismatch in {name}"
    assert os.path.getsize(out_path) == chk_data_start + off, f"Size mismatch in {name}"
    with open(src, "rb") as f, open(out_path, "rb") as chk:
        for k in keep[:: max(1, len(keep) // 20)]:
            s_, e_ = header[k]["data_offsets"]
            ns, ne = check[k]["data_offsets"]
            f.seek(data_start + s_)
            chk.seek(chk_data_start + ns)
            a, b = f.read(e_ - s_), chk.read(ne - ns)
            assert a == b, f"Data mismatch for {k} in {name}"
        for k, ref in ((EMB_KEY + ".weight_i8", q), (EMB_KEY + ".scale", scale)):
            ns, ne = check[k]["data_offsets"]
            chk.seek(chk_data_start + ns)
            got = torch.frombuffer(bytearray(chk.read(ne - ns)), dtype = ref.dtype, count = ref.numel())
            assert got.equal(ref.view(-1)), f"Data mismatch for {k} in {name}"
    del q, scale
    print(f" -- {name}: stripped .weight ({w_bytes / 1024**2:.1f} MiB), "
          f"added weight_i8 + scale ({(q_bytes + s_bytes) / 1024**2:.1f} MiB), "
          f"kept {len(keep)} tensors ({keep_bytes / 1024**2:.1f} MiB) -> rebuilt, verified")

    # All other files
    new_headers = {k: v for k, v in headers.items() if k != src}
    new_headers[out_path] = (check, chk_data_start)
    for fn in sorted(os.listdir(mdir)):
        s_path = os.path.join(mdir, fn)
        if not os.path.isfile(s_path) or fn == name:
            continue
        d_path = os.path.join(odir, fn)
        if fn.endswith(".safetensors"):
            link_or_copy(s_path, d_path)
            print(f" -- {fn}: untouched -> linked")
            continue
        if fn == "model.safetensors.index.json":
            with open(s_path) as f:
                idx = json.load(f)
            wm = idx.get("weight_map", {})
            if EMB_KEY + ".weight" in wm:
                del wm[EMB_KEY + ".weight"]
            wm[EMB_KEY + ".weight_i8"] = name
            wm[EMB_KEY + ".scale"] = name
            if isinstance(idx.get("metadata"), dict) and "total_size" in idx["metadata"]:
                idx["metadata"]["total_size"] = sum(
                    h[k]["data_offsets"][1] - h[k]["data_offsets"][0]
                    for h, _ in new_headers.values() for k in h if k != "__metadata__")
            with open(d_path, "w") as f:
                f.write(json.dumps(idx, indent = 4))
            print(f" -- {fn}: weight_map updated (dropped .weight, added .weight_i8 + .scale)")
        elif fn == "quantization_config.json":
            with open(s_path) as f:
                qcfg = json.load(f)
            ts = qcfg.get("tensor_storage")
            if isinstance(ts, dict) and EMB_KEY in ts:
                ts[EMB_KEY] = {
                    "stored_tensors": {
                        EMB_KEY + ".weight_i8": {"shape": shape, "n_bytes": q_bytes, "dtype": "torch.int8"},
                        EMB_KEY + ".scale": {"shape": [shape[0], shape[1] // BLOCK], "n_bytes": s_bytes,
                                             "dtype": "torch.float16"},
                    },
                    "quant_format": "int8",
                    "bits_per_weight": round((q_bytes + s_bytes) / numel * 8, 3),
                    "block_size": BLOCK,
                }
                qcfg["embedding_bits"] = 8
                print(f" -- {fn}: tensor_storage['{EMB_KEY}'] updated, embedding_bits = 8")
            else:
                print(f" !! {fn}: no tensor_storage entry for {EMB_KEY}; left unchanged")
            with open(d_path, "w") as f:
                f.write(json.dumps(qcfg, indent = 4))
        else:
            link_or_copy(s_path, d_path)

    net = (q_bytes + s_bytes) - w_bytes
    print(f" -- Net delta: {net / 1024**2:+.1f} MiB ({net / 1024**2 / 1024:+.3f} GiB)")
    print(f" -- Output: {odir}")


if __name__ == "__main__":
    main()