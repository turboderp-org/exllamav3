"""
Trellis quantization of the Hy4-flash Engram tables.

Hy4-flash carries one table per Engram layer (four in the release), each a single BF16 tensor of
96-wide rows, <layer>.engram_layer.multi_head_embedding.embedding.weight. The reference reads a
row through an e4m3 round trip (one amax / 448 scale per row), so the encode target is the
fp8-rounded row, not the stored one. Each row becomes one tail-biting mul1 ring of length 96 at K
bits a value with an fp16 scale and a per-hash-head bias, the same ring format as the 160-wide
n-gram rows of exl3_ngram_trellis:

    row[i] = decode_mul1(state_i) * scale + head_bias[head]            (2 + 96 * K / 8 bytes a row)

One output file per table, engram.layers.<idx>.safetensors, holding

    <key>.head_bias          F16 (heads, 96)
    <key>.head_offsets       I64 (heads)       first row of each head's bucket range
    <key>.head_vocab_sizes   I64 (heads)       the hash moduli (config engram_layer_head_vocab_sizes)
    <key>.layer_multipliers  I64 (max_ngram)   the hash multipliers (drawn from the config's seed)
    <key>.trellis            I16 (rows, 1 + 6 * K)

with the small tensors first, so a partial file is resumable (StreamingSafetensorsWriter).
"""

from __future__ import annotations
import json
import math
import os
import struct
import time
import torch

from .ngram import StreamingSafetensorsWriter, quantize_rows, read_table_tensor, CS_HEURISTIC
from ..modules.quant.exl3_lib.ngram_codec import words_per_row
from ..modules.arch_specific.hy_v4_flash import HyV4EngramTable

FORMAT_VERSION = 1


def table_file_name(table: HyV4EngramTable) -> str:
    return f"engram.layers.{table.layer_id - 1}.safetensors"


class _Source:
    """Row-addressable view of one stored table tensor"""

    def __init__(self, stc, key: str):
        from safetensors import safe_open
        self.key = key
        self.filename = stc.find_stc(key).tensor_file_map[key] if hasattr(stc, "find_stc") else stc.tensor_file_map[key]
        self.handle = safe_open(self.filename, framework = "pt")
        self.slice = self.handle.get_slice(key)
        self.shape = self.slice.get_shape()

    def read(self, lo: int, hi: int) -> torch.Tensor:
        return self.slice[lo:hi]


def _target(rows: torch.Tensor, fp8: bool) -> torch.Tensor:
    rows = torch.nan_to_num(rows.float(), nan = 0.0, posinf = 0.0, neginf = 0.0)
    return HyV4EngramTable.fp8_round_trip(rows) if fp8 else rows


def compute_head_bias(src: _Source, table: HyV4EngramTable, device, fp8: bool, rows_per_head: int = 131072) -> torch.Tensor:
    """Per-head channel mean of the encode target over evenly spaced stripes, (heads, row_dim) fp16"""
    bias = torch.zeros(table.num_heads, table.row_dim, dtype = torch.float64)
    for h in range(table.num_heads):
        off, size = int(table.head_offsets[h]), int(table.head_vocab_sizes[h])
        stripe = min(8192, size)
        n_str = max(1, min(rows_per_head, size) // stripe)
        acc = torch.zeros(table.row_dim, dtype = torch.float64, device = device)
        n = 0
        for j in range(n_str):
            lo = off + (size - stripe) * j // max(1, n_str - 1)
            r = _target(src.read(lo, lo + stripe).to(device), fp8)
            acc += r.double().sum(dim = 0)
            n += r.shape[0]
        bias[h] = (acc / n).cpu()
    return bias.to(torch.float16)


def quantize_engram_table(
    stc,
    table: HyV4EngramTable,
    out_path: str,
    K: int,
    device,
    fp8: bool = True,
    cs: float | None = None,
    cs_search: int = 1,
    chunk_rows: int = 262144,
    limit_rows: int | None = None,
    resume: bool = True,
    verbose: bool = True,
    multipliers: list[int] | None = None,
) -> dict:
    """Quantize one table into out_path. Returns a stats dict (rfn against the fp8-rounded target
    and against the stored BF16 rows, rows a second)."""
    assert 1 <= K <= 8 and (table.row_dim * K) % 16 == 0
    key = table.key
    src = _Source(stc, f"{key}.weight")
    assert src.shape[1] == table.row_dim and src.shape[0] == table.num_rows, \
        f"{key}: stored {src.shape}, config says ({table.num_rows}, {table.row_dim})"
    total = table.num_rows if limit_rows is None else min(limit_rows, table.num_rows)
    if multipliers is None:
        lut = stc.get_tensor(table.lookup_key, "cpu", allow_bf16 = True, no_defer = True)
        multipliers = HyV4EngramTable.draw_multipliers(table.seed, table.layer_id, table.max_ngram_size, int(lut.max()) + 1)
    dev = torch.device(device)

    bias = read_table_tensor(out_path, f"{key}.head_bias", torch.float16) if resume else None
    if bias is None:
        if verbose:
            print(f" -- {key}: computing per-head bias")
        bias = compute_head_bias(src, table, dev, fp8)
    gamma, cs_hi = CS_HEURISTIC[K]
    cs_desc = f"heuristic(gamma={gamma:.2f}, hi={cs_hi:.2f})" if cs is None else f"{cs:.6f}"
    if cs_search > 1:
        cs_desc += f" x{cs_search} search"

    writer = StreamingSafetensorsWriter(
        out_path,
        stream_tensors = [(f"{key}.trellis", (total, words_per_row(K, table.row_dim)))],
        stream_dtype_str = "I16",
        small_tensors = {
            f"{key}.head_bias": bias,
            f"{key}.head_offsets": table.head_offsets.clone(),
            f"{key}.head_vocab_sizes": table.head_vocab_sizes.clone(),
            f"{key}.layer_multipliers": torch.tensor(multipliers, dtype = torch.long),
        },
        metadata = {
            "format": "exl3_ngram_trellis",
            "version": str(FORMAT_VERSION),
            "K": str(K),
            "codebook": "mul1",
            "codebook_scale": cs_desc,
            "row_dim": str(table.row_dim),
            "rows": str(total),
            "target": "fp8_e4m3_row_round_trip" if fp8 else "stored",
            "layer_id": str(table.layer_id),
        },
        resume = resume,
        chunk_rows = chunk_rows,
    )
    start = writer.resume_rows
    if verbose:
        print(f" -- {key}: {table.num_rows} rows x {table.row_dim}, K = {K}, target "
              f"{'fp8-rounded rows' if fp8 else 'stored rows'}, cs = {cs_desc}"
              + (f", resuming at row {start} ({start / total * 100:.1f}%)" if start else ""))
    bias_d = bias.to(dev)
    offs = table.head_offsets.to(dev)
    err = src_sq = err_raw = 0.0
    t0 = time.time()
    last = t0
    with torch.cuda.device(dev):
        for lo in range(start, total, chunk_rows):
            hi = min(lo + chunk_rows, total)
            raw = src.read(lo, hi).to(dev)
            tgt = _target(raw, fp8)
            heads = (torch.searchsorted(offs, torch.arange(lo, hi, device = dev), right = True) - 1) \
                .clamp(0, table.num_heads - 1)
            packed, deq = quantize_rows(tgt, bias_d[heads], K, cs, cs_search)
            err += (deq - tgt).square().sum().double().item()
            err_raw += (deq - raw.float()).square().sum().double().item()
            src_sq += tgt.square().sum().double().item()
            writer.write_chunk(packed.cpu())
            if verbose and time.time() - last > 60:
                last = time.time()
                rate = (hi - start) / (last - t0)
                print(f"    {key}: row {hi} of {total} ({hi / total * 100:.1f}%), {rate:,.0f} rows/s, "
                      f"eta {(total - hi) / max(rate, 1):,.0f} s, rfn {math.sqrt(err / max(src_sq, 1e-30)):.5f}", flush = True)
    writer.finalize()
    elapsed = time.time() - t0
    done = total - start
    stats = {
        "key": key, "rows": total, "processed_rows": done, "K": K, "row_dim": table.row_dim,
        "rfn_vs_target": math.sqrt(err / src_sq) if src_sq > 0 else 0.0,
        "rfn_vs_stored": math.sqrt(err_raw / src_sq) if src_sq > 0 else 0.0,
        "elapsed": elapsed, "rows_per_s": done / max(elapsed, 1e-9), "bytes": os.path.getsize(out_path),
        "resumed": bool(start),
    }
    if verbose:
        print(f" -- {key}: {done} rows in {elapsed:.0f} s ({stats['rows_per_s']:,.0f} rows/s), rfn "
              f"{stats['rfn_vs_target']:.5f} against the fp8-rounded rows, {stats['rfn_vs_stored']:.5f} against the "
              f"stored rows{' (resumed part)' if start else ''}, {stats['bytes'] / 1e9:.2f} GB -> {out_path}")
    return stats


def table_complete(path: str, table: HyV4EngramTable, K: int) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) < 16:
        return False
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    info = header.get(f"{table.key}.trellis")
    if info is None or info["shape"] != [table.num_rows, words_per_row(K, table.row_dim)]:
        return False
    return os.path.getsize(path) == 8 + hlen + info["data_offsets"][1]


def prepare_engram_tables_for_conversion(args: dict, config, model) -> list[str]:
    """Conversion step for Hy4-flash: one quantized file per Engram table in the output directory
    (resumable, skipped when complete), or copies of pre-quantized files from the directory given
    as --ngram_file. The files join the tensor collection so the calibration forwards read the
    quantized tables."""
    tables = [m for mod in model.modules for m in mod if isinstance(m, HyV4EngramTable)]
    if not tables:
        return []
    os.makedirs(args["out_dir"], exist_ok = True)
    K = args.get("ngram_bits") or max(1, min(8, round(args["bits"])))
    pre = args.get("ngram_file")
    devices = [int(d) for d in args["devices"].split(",")]
    out, stats = [], []
    for table in tables:
        name = table_file_name(table)
        path = os.path.join(args["out_dir"], name)
        if pre:
            src = os.path.join(pre, name)
            assert os.path.exists(src), f"--ngram_file directory has no {name}"
            if os.path.abspath(src) != os.path.abspath(path) and not table_complete(path, table, K):
                import shutil
                print(f" -- Copying pre-quantized Engram table: {src}")
                shutil.copyfile(src, path)
        if table_complete(path, table, K):
            print(f" -- Engram table in place: {name}")
        else:
            assert not pre, f"{name} in --ngram_file does not hold {table.num_rows} rows at K = {K}"
            stats.append(quantize_engram_table(
                config.stc, table, path, K, f"cuda:{devices[0]}", fp8 = table.fp8_rows, resume = True))
        out.append(path)
    if stats:
        with open(os.path.join(args["out_dir"], "engram_quant.json"), "w") as f:
            json.dump(stats, f, indent = 1)
    for path in out:
        config.stc.add_tensor_files(path, warn_if_override = False)
    return out
