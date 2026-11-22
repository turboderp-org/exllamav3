"""
Trellis quantization of hashed n-gram embedding tables (Qwen3.8-Flash-Next ple_embedding and kin).

Each embedding row (160-D in Qwen3.8-Flash-Next, 96-D in Hy4 flash) is quantized as a single
tail-biting trellis ring (mul1 codebook, the row instances of quantize_tiles), after subtracting a
per-hash-head bias vector and applying a per-row scale. Storage per row is 16 + dim*K bits, packed
as (1 + dim*K/16) little-endian uint16 words: word 0 holds the fp16 row scale, the rest hold the
ring bitstream where stream bits [i*K, (i+1)*K) are the low K bits of position i's 16-bit trellis
state.

Reconstruction: row[i] = decode_mul1(state_i) * scale + head_bias[head], where state_i is read as
the 16 ring bits ending at stream bit (i+1)*K - 1 (mod dim*K).

The file produced by quantize_ngram_table() / util/convert_ngram.py is a regular safetensors file
holding the packed table plus the bias vectors and the source model's hashing buffers, with the
quantization parameters in the metadata. NgramTableReader reads rows back from disk on demand
without ever loading the table (deferred-load backing store for the model's embedding module).
"""

from __future__ import annotations
import json
import math
import os
import queue
import struct
import threading
import time
import torch

from ..modules.quant.exl3_lib.quantize import quantize_tiles
from ..modules.quant.exl3_lib.ngram_codec import (  # noqa: F401  (re-exported)
    ROW_DIM, GROUP_DIM, MUL1, words_per_row, mul1_codebook, pack_rows, unpack_rows, dequant_rows,
    hadamard_rotate,
)

NGRAM_FORMAT_VERSION = 1

# Codebook-scale multiplier per K: input rows are scaled to rms = cs before the trellis search.
# The random codebook's tail coverage makes the optimum shrink with K; swept on the
# qwen3.8-flash-next table (benchmarks/ngram_quant/quant160.py)
DEFAULT_CS = {1: 1.16, 2: 0.98, 3: 0.98, 4: 0.92, 5: 0.92, 6: 0.86, 7: 0.86, 8: 0.80}

# Per-row heuristic (the default): cs = clamp(gamma / (absmax/rms), CS_MIN, cs_hi), i.e. scale each
# row so its largest element lands near the codebook edge (~3.35), capped at cs_hi for clean rows.
# (gamma, cs_hi) fitted per K on the qwen3.8-flash-next table against a per-row grid oracle;
# captures the predictable (clipping-driven) part of the per-row optimum in a single encode
CS_HEURISTIC = {
    1: (4.0, 1.16), 2: (3.6, 0.98), 3: (3.2, 0.98), 4: (3.0, 0.98),
    5: (3.0, 0.95), 6: (3.0, 0.92), 7: (3.0, 0.90), 8: (3.0, 0.86),
}
CS_MIN = 0.55
CS_SEARCH_STEP = 0.06


def quantize_rows(
    rows: torch.Tensor,
    bias: torch.Tensor | None,
    K: int,
    cs: float | None = None,
    cs_search: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    rows: (N, dim) source rows, or (N, 256) rotated groups (any float dtype) on a CUDA device
    bias: (N, dim) per-row bias to subtract (gathered per head), fp16 values, or None
    cs: fixed codebook-scale multiplier, or None for the per-row heuristic (default)
    cs_search: number of encodes per row; > 1 sweeps cs in steps of CS_SEARCH_STEP centered on the
        heuristic/fixed value and keeps the lowest-error encode per row (decode-invisible)
    Returns (packed (N, words) int16, deq float reconstruction incl. bias).
    """
    # Guard against corrupt source values: a single inf/NaN element would make every Viterbi cost
    # non-finite for its row (degenerate garbage output; before the kernel-side argmin clamp it
    # was an out-of-bounds access)
    w = torch.nan_to_num(rows.float(), nan = 0.0, posinf = 0.0, neginf = 0.0)
    if bias is not None:
        w = w - bias.float()
    rms = w.square().mean(dim = 1, keepdim = True).sqrt()
    if cs is None:
        gamma, cs_hi = CS_HEURISTIC[K]
        absmax_norm = w.abs().max(dim = 1).values / rms.squeeze(1).clamp(min = 1e-12)
        cs_row = (gamma / absmax_norm.clamp(min = 1e-6)).clamp(min = CS_MIN, max = cs_hi)
    else:
        cs_row = torch.full((w.shape[0],), cs, device = w.device)

    best_err = best_q = best_states = best_scale = None
    for c in range(cs_search):
        offset = (c - (cs_search - 1) / 2) * CS_SEARCH_STEP
        cs_c = (cs_row + offset).clamp(min = CS_MIN)
        scale0 = (rms / cs_c.unsqueeze(1)).clamp(min = 1e-8).to(torch.float16).float()
        # an rms below fp16 range rounds the pre-scale to zero; encode such rows against scale 1
        # instead of dividing by zero (their stored LS scale comes out ~0 either way)
        scale0 = torch.where(scale0 > 0, scale0, 1.0)
        q, states = quantize_tiles((w / scale0).contiguous(), {"K": K, "mul1": True})
        # least-squares per-row scale (w and q are in original resp. codebook units, so this
        # replaces scale0 rather than refining it), stored as the row's fp16 scale
        scale = ((w * q).sum(dim = 1) / (q * q).sum(dim = 1).clamp(min = 1e-12)).to(torch.float16)
        if cs_search == 1:
            best_q, best_states, best_scale = q, states, scale
            break
        err = (w - q * scale.float().unsqueeze(1)).square().sum(dim = 1)
        if best_err is None:
            best_err, best_q, best_states, best_scale = err, q, states, scale
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better.unsqueeze(1), q, best_q)
            best_states = torch.where(better.unsqueeze(1), states, best_states)
            best_scale = torch.where(better, scale, best_scale)

    packed = pack_rows(best_states, best_scale, K)
    deq = best_q * best_scale.float().unsqueeze(1)
    if bias is not None:
        deq = deq + bias.float()
    return packed, deq


def quantize_embedding(
    key: str,
    weight: torch.Tensor,
    K: int,
    device,
    chunk_rows: int = 8192,
    verbose: bool = True,
) -> tuple[dict, float]:
    """
    Trellis quantization of a token embedding table: rows are zero-padded to a multiple of 256,
    split into 256-wide groups and rotated (random signs + Hadamard), and each group is encoded
    like an n-gram row. Returns the module's tensors and the relative error of the table:

        <key>.trellis   (rows, groups * (1 + 16 * K)) int16, the groups' rings back to back
        <key>.signs     (groups, 256) fp16 sign vectors
    """
    assert 1 <= K <= 8
    rows, dim = weight.shape
    groups = (dim + GROUP_DIM - 1) // GROUP_DIM
    gen = torch.Generator().manual_seed(dim)
    signs = (torch.randint(0, 2, (groups, GROUP_DIM), generator = gen) * 2 - 1).to(torch.float16)
    signs_d = signs.to(device).float().view(1, -1)
    trellis = torch.empty((rows, groups * words_per_row(K, GROUP_DIM)), dtype = torch.int16)
    err_sq = src_sq = 0.0
    from ..util.progress import ProgressBar
    with ProgressBar(f" -- Quantizing: {key}" if verbose else None, rows) as pb:
        for a in range(0, rows, chunk_rows):
            w = weight[a : a + chunk_rows].to(device).float()
            padded = torch.nn.functional.pad(w, (0, groups * GROUP_DIM - dim))
            rotated = hadamard_rotate((padded * signs_d).view(-1, GROUP_DIM))
            packed, deq = quantize_rows(rotated, None, K)
            trellis[a : a + chunk_rows] = packed.view(w.shape[0], -1).cpu()
            deq = (hadamard_rotate(deq).view(w.shape[0], -1) * signs_d)[:, :dim]
            err_sq += (deq - w).square().sum().double().item()
            src_sq += w.square().sum().double().item()
            pb.update(a + w.shape[0])
    rfn = math.sqrt(err_sq / src_sq) if src_sq > 0 else 0.0
    return {f"{key}.trellis": trellis, f"{key}.signs": signs}, rfn


class StreamingSafetensorsWriter:
    """
    Writes a safetensors file front to back without holding the large tensors in memory. All
    shapes must be known up front. The small tensors are placed FIRST in the file and written at
    open time, then the streamed tensors' data is appended in row order — so a partial file
    from an interrupted run already contains a valid header and the small tensors, and a run with
    resume = True continues from the last complete chunk (the header must match exactly, which
    also guarantees the same bias vectors and quantization parameters).

    stream_tensors is a list of (name, shape) sharing one dtype and row width: they are laid out
    consecutively, so the byte stream is a single contiguous row array and write_chunk() never
    needs to know which shard a chunk lands in (a chunk may span a shard boundary).
    """

    def __init__(self, path: str, stream_tensors: list, stream_dtype_str: str,
                 small_tensors: dict[str, torch.Tensor], metadata: dict[str, str],
                 resume: bool = False, chunk_rows: int = 1):
        dtype_size = {"I16": 2, "U16": 2, "F16": 2, "BF16": 2, "F32": 4, "I64": 8}
        header = {"__metadata__": dict(metadata)}
        offset = 0
        self.small_tensors = {}
        for name, t in small_tensors.items():
            t = t.contiguous().cpu()
            dts = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32",
                   torch.int64: "I64", torch.int16: "I16"}[t.dtype]
            nbytes = t.numel() * t.element_size()
            header[name] = {"dtype": dts, "shape": list(t.shape), "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
            self.small_tensors[name] = t
        assert all(s[1] == stream_tensors[0][1][1] for _, s in stream_tensors), \
            "streamed tensors must share one row width"
        stream_bytes = 0
        for name, shape in stream_tensors:
            nbytes = int(torch.tensor(shape).prod().item()) * dtype_size[stream_dtype_str]
            header[name] = {
                "dtype": stream_dtype_str,
                "shape": [int(s) for s in shape],
                "data_offsets": [offset + stream_bytes, offset + stream_bytes + nbytes],
            }
            stream_bytes += nbytes
        hj = json.dumps(header, separators = (",", ":")).encode("utf-8")
        hj += b" " * (-len(hj) % 8)
        self.stream_bytes = stream_bytes
        stream_abs = 8 + len(hj) + offset
        row_bytes = int(stream_tensors[0][1][1]) * dtype_size[stream_dtype_str]
        self.resume_rows = 0

        if resume and os.path.exists(path) and os.path.getsize(path) > 8:
            with open(path, "rb") as f:
                ex_hlen = struct.unpack("<Q", f.read(8))[0]
                ex_hj = f.read(ex_hlen)
            if ex_hj != hj:
                raise ValueError(
                    f"cannot resume {path}: existing header does not match this run's parameters")
            done = max(0, os.path.getsize(path) - stream_abs)
            if done >= stream_bytes:
                done = stream_bytes    # complete file (final chunk may be smaller than chunk_rows)
            else:
                done -= done % (chunk_rows * row_bytes)
            self.file = open(path, "r+b")
            self.file.truncate(stream_abs + done)
            self.file.seek(stream_abs + done)
            self.streamed = done
            self.resume_rows = done // row_bytes
        else:
            self.file = open(path, "wb")
            self.file.write(struct.pack("<Q", len(hj)))
            self.file.write(hj)
            for t in self.small_tensors.values():
                self.file.write(t.numpy().tobytes())
            self.streamed = 0

    def write_chunk(self, t: torch.Tensor):
        b = t.contiguous().cpu().numpy().tobytes()
        self.streamed += len(b)
        assert self.streamed <= self.stream_bytes, "streamed tensor overflow"
        self.file.write(b)

    def finalize(self):
        assert self.streamed == self.stream_bytes, \
            f"streamed tensor underflow ({self.streamed} of {self.stream_bytes} bytes)"
        self.file.close()


class NgramSource:
    """
    Row-addressable view of one source table: the {prefix}.shard_N.weight tensors back to back, or a
    single {prefix}.weight. The hash layout comes from the caller (models that derive it from their
    config) or from the checkpoint's buffers next to the table.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        file_map: dict | None = None,
        prefix: str | None = None,
        row_dim: int = ROW_DIM,
        head_vocab_sizes: list | None = None,
        layer_multipliers: list | None = None,
        row_filter = None,
    ):
        """
        model_dir: read the tensor locations from its model.safetensors.index.json; or
        file_map: tensor name -> file path. prefix: the embedding module's key, found by its shard
        tensors when omitted. row_filter: applied to source rows (on the device, source dtype) before
        they are encoded, for tables the model reads through a transform
        """
        self.model_dir = model_dir
        if file_map is None:
            with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
                weight_map = json.load(f)["weight_map"]
            file_map = {k: os.path.join(model_dir, v) for k, v in weight_map.items()}

        def find(suffix):
            keys = [k for k in file_map if k.endswith(suffix)]
            assert len(keys) == 1, f"expected exactly one *{suffix} tensor, found {len(keys)}"
            return keys[0]

        if prefix is None:
            # model-scoped key prefix of the embedding module, e.g.
            # "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
            assert any(".ngram_embedding.shard_" in k for k in file_map), "no ngram_embedding shards found in model"
            prefix = find(".ngram_embedding.shard_0.weight").rsplit(".shard_", 1)[0]
        self.prefix = prefix
        num_shards = len([k for k in file_map if k.startswith(prefix + ".shard_") and k.endswith(".weight")])
        self.sharded = num_shards > 0
        self.shard_keys = [f"{prefix}.shard_{s}.weight" for s in range(num_shards)] if self.sharded else [f"{prefix}.weight"]
        self.shard_files = [file_map[k] for k in self.shard_keys]
        self.row_filter = row_filter
        self._local = threading.local()

        from safetensors import safe_open
        with safe_open(self.shard_files[0], framework = "pt") as f:
            shape = f.get_slice(self.shard_keys[0]).get_shape()
        self.rows_per_shard = shape[0]
        self.row_dim = row_dim
        assert shape[1] == row_dim, f"{prefix}: rows are {shape[1]} wide, expected {row_dim}"
        self.num_rows = self.rows_per_shard * len(self.shard_keys)

        if head_vocab_sizes is not None:
            self.head_vocab_sizes = torch.tensor(head_vocab_sizes, dtype = torch.long)
            self.head_offsets = self.head_vocab_sizes.cumsum(0) - self.head_vocab_sizes
            self.layer_multipliers = torch.tensor(layer_multipliers, dtype = torch.long)
        else:
            def load_aux(suffix):
                key = find(suffix)
                with safe_open(file_map[key], framework = "pt") as f:
                    return f.get_tensor(key)
            self.head_offsets = load_aux(".ngram_heads_offsets")
            self.head_vocab_sizes = load_aux(".ngram_heads_vocab_sizes")
            self.layer_multipliers = load_aux(".layer_multipliers")
        self.num_heads = self.head_offsets.shape[0]

    def _handles(self):
        # safetensors handles are opened once per thread
        h = getattr(self._local, "handles", None)
        if h is None:
            from safetensors import safe_open
            h = [safe_open(f, framework = "pt") for f in self.shard_files]
            self._local.handles = h
        return h

    def read_rows(self, start: int, end: int) -> torch.Tensor:
        """Contiguous global row range, may span shard boundaries."""
        parts = []
        handles = self._handles()
        while start < end:
            s, lo = divmod(start, self.rows_per_shard)
            hi = min(self.rows_per_shard, lo + (end - start))
            parts.append(handles[s].get_slice(self.shard_keys[s])[lo:hi])
            start += hi - lo
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def rows_on(self, start: int, end: int, device) -> torch.Tensor:
        """The rows as the model reads them, float on the device"""
        rows = self.read_rows(start, end).to(device)
        if self.row_filter is not None:
            rows = self.row_filter(rows)
        return rows.float()

    def head_of_rows(self, start: int, end: int, device) -> torch.Tensor:
        idx = torch.arange(start, end, device = device)
        h = torch.searchsorted(self.head_offsets.to(device), idx, right = True) - 1
        return h.clamp(0, self.num_heads - 1)

    def stream_tensors(self, K: int, total_rows: int | None = None) -> list:
        """(name, shape) of the packed tensors this table is written as: one per source shard (so
        consumers can keep them as separate tensors in RAM instead of one concatenated table), or one
        for an unsharded table"""
        total_rows = self.num_rows if total_rows is None else total_rows
        words = words_per_row(K, self.row_dim)
        if not self.sharded:
            return [(f"{self.prefix}.trellis", (total_rows, words))]
        out, r0 = [], 0
        while r0 < total_rows:
            r1 = min(r0 + self.rows_per_shard, total_rows)
            out.append((f"{self.prefix}.shard_{len(out)}.trellis", (r1 - r0, words)))
            r0 = r1
        return out


def copy_tensors_streamed(stc, names: list[str], out_path: str, chunk: int = 256 << 20):
    """Copy tensors from the checkpoint into one safetensors file without reading them into memory
    whole (an Engram table is tens of GB). Skipped when the file already holds them at full size"""
    entries = []
    for name in names:
        src = stc.tensor_file_map[name]
        with open(src, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            info = json.loads(f.read(hlen))[name]
        a, b = info["data_offsets"]
        entries.append((name, src, 8 + hlen + a, b - a, info["dtype"], info["shape"]))
    total = sum(e[3] for e in entries)
    header, off = {}, 0
    for name, _, _, size, dtype, shape in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + size]}
        off += size
    hb = json.dumps(header).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    if os.path.exists(out_path) and os.path.getsize(out_path) == 8 + len(hb) + total:
        print(f" -- N-gram table already in place: {out_path}")
        return
    bpw = 8 * total / max(sum(e[3] for e in entries if not e[0].endswith(".scale")), 1)
    print(f" -- Copying n-gram table ({total >> 20} MiB) to {out_path}, kept as stored: {bpw:.2f} bpw")
    with open(out_path + ".tmp", "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for name, src, start, size, _, _ in entries:
            with open(src, "rb") as f:
                f.seek(start)
                left = size
                while left:
                    buf = f.read(min(chunk, left))
                    out.write(buf)
                    left -= len(buf)
    os.replace(out_path + ".tmp", out_path)


def read_table_tensor(path: str, name: str, dtype: torch.dtype) -> torch.Tensor | None:
    """Read one (small) tensor from a quantized table file, tolerating a truncated stream."""
    try:
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
            info = header[name]
            f.seek(8 + hlen + info["data_offsets"][0])
            raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
        return torch.frombuffer(bytearray(raw), dtype = dtype).view(*info["shape"])
    except Exception:
        return None


def compute_head_bias(source: NgramSource, device, rows_per_head: int = 131072, verbose: bool = False) -> torch.Tensor:
    """Per-head channel mean over sampled stripes, returned as (num_heads, row_dim) float16."""
    bias = torch.zeros(source.num_heads, source.row_dim, dtype = torch.float64)
    stripe = 8192
    for h in range(source.num_heads):
        off = source.head_offsets[h].item()
        size = source.head_vocab_sizes[h].item()
        num_stripes = max(1, rows_per_head // stripe)
        acc = torch.zeros(source.row_dim, dtype = torch.float64, device = device)
        n = 0
        for j in range(num_stripes):
            lo = off + max(0, size - stripe) * j // max(1, num_stripes - 1)
            rows = source.rows_on(lo, min(lo + stripe, off + size), device)
            acc += rows.double().sum(dim = 0)
            n += rows.shape[0]
        bias[h] = (acc / n).cpu()
        if verbose:
            print(f" -- head {h:2d}: bias energy {(bias[h].square().sum()).item():.3e} over {n} rows")
    return bias.to(torch.float16)


def quantize_ngram_table(
    model_dir: str,
    out_path: str,
    K: int,
    devices: list[int],
    **kwargs,
) -> dict:
    """
    Quantize the n-gram embedding table of the model at model_dir (found by its shard tensors) into
    out_path. util/convert_ngram.py is the CLI wrapper.
    """
    return quantize_ngram_tables([NgramSource(model_dir)], out_path, K, devices, **kwargs)


def quantize_ngram_tables(
    sources: list[NgramSource],
    out_path: str,
    K: int,
    devices: list[int],
    cs: float | None = None,
    cs_search: int = 1,
    chunk_rows: int = 131072,
    limit_rows: int | None = None,
    bias_sample_rows: int = 131072,
    resume: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Quantize the given tables into one file, back to back. Returns a stats dict. cs = None (default)
    selects the per-row heuristic; a float fixes the multiplier. limit_rows caps every table.
    """
    assert 1 <= K <= 8
    assert cs_search >= 1
    row_dim = sources[0].row_dim
    assert all(s.row_dim == row_dim for s in sources), "tables of one file must share a row width"
    assert row_dim * K % 16 == 0, f"{row_dim}-D rows at K = {K} do not pack into whole words"
    rows_of = [s.num_rows if limit_rows is None else min(limit_rows, s.num_rows) for s in sources]
    starts = [sum(rows_of[:i]) for i in range(len(sources))]
    total_rows = sum(rows_of)

    if cs is None:
        gamma, cs_hi = CS_HEURISTIC[K]
        cs_desc = f"heuristic(gamma={gamma:.2f}, hi={cs_hi:.2f})"
    else:
        cs_desc = f"{cs:.6f}"
    if cs_search > 1:
        cs_desc += f" x{cs_search} search"
    if verbose:
        for s, n in zip(sources, rows_of):
            print(f" -- source table {s.prefix}: {s.num_rows} rows x {row_dim} in {len(s.shard_files)} tensor(s), "
                  f"quantizing {n} rows at K = {K} (cs = {cs_desc}) on devices {devices}")

    # On resume, reuse the interrupted run's bias vectors (already in the partial file) so all rows
    # of the finished table share one bias; the writer's header-equality check then guarantees the
    # remaining parameters match too
    biases = []
    for s in sources:
        bias = read_table_tensor(out_path, f"{s.prefix}.head_bias", torch.float16) if resume else None
        if bias is None:
            if verbose:
                print(f" -- computing per-head bias for {s.prefix} ({bias_sample_rows} sampled rows/head)")
            bias = compute_head_bias(s, f"cuda:{devices[0]}", bias_sample_rows, verbose = False)
        biases.append(bias)

    stream_tensors, small_tensors = [], {}
    for s, n, bias in zip(sources, rows_of, biases):
        stream_tensors += s.stream_tensors(K, n)
        small_tensors.update({
            f"{s.prefix}.head_bias": bias,
            f"{s.prefix}.head_offsets": s.head_offsets,
            f"{s.prefix}.head_vocab_sizes": s.head_vocab_sizes,
            f"{s.prefix}.layer_multipliers": s.layer_multipliers,
        })

    writer = StreamingSafetensorsWriter(
        out_path,
        stream_tensors = stream_tensors,
        stream_dtype_str = "I16",
        small_tensors = small_tensors,
        metadata = {
            "format": "exl3_ngram_trellis",
            "version": str(NGRAM_FORMAT_VERSION),
            "K": str(K),
            "codebook": "mul1",
            "codebook_scale": cs_desc,  # informational only; the chosen cs is folded into row scales
            "row_dim": str(row_dim),
            "rows": str(total_rows),
            "shard_rows": str(sources[0].rows_per_shard),
            "source": os.path.abspath(sources[0].model_dir) if sources[0].model_dir else "",
        },
        resume = resume,
        chunk_rows = chunk_rows,
    )
    start_row = writer.resume_rows
    if verbose and start_row:
        print(f" -- resuming at row {start_row} of {total_rows} ({start_row / total_rows * 100:.1f}% done)")

    def parts_of(lo, hi):
        """A chunk of the file's row space as (table, first row, last row) pieces: tables are stored
        back to back and a chunk may cross from one into the next"""
        out = []
        for t, (s0, n) in enumerate(zip(starts, rows_of)):
            a, b = max(lo, s0), min(hi, s0 + n)
            if a < b:
                out.append((t, a - s0, b - s0))
        return out

    chunks = [(i, lo, min(lo + chunk_rows, total_rows))
              for i, lo in enumerate(range(start_row, total_rows, chunk_rows))]
    work = queue.Queue()
    for c in chunks:
        work.put(c)
    results = {}
    cv = threading.Condition()
    state = {"next_write": 0, "error": None}
    max_pending = 3 * len(devices)
    # plain python accumulators: torch tensors created here would be inference tensors when the
    # caller runs under torch.inference_mode() (thread-local), and the worker threads could not
    # update them in place
    acc = {"err_sq": 0.0, "src_sq": 0.0}
    dev_rows = {d: 0 for d in devices}

    def worker(dev_idx):
        dev = f"cuda:{dev_idx}"
        try:
            biases_d = [b.to(dev) for b in biases]
            while True:
                try:
                    chunk_i, lo, hi = work.get_nowait()
                except queue.Empty:
                    return
                with cv:
                    cv.wait_for(lambda: len(results) < max_pending or state["error"])
                    if state["error"]:
                        return
                packed_parts, e, s = [], 0.0, 0.0
                for t, a, b in parts_of(lo, hi):
                    rows = sources[t].rows_on(a, b, dev)
                    heads = sources[t].head_of_rows(a, b, dev)
                    with torch.cuda.device(dev):
                        packed, deq = quantize_rows(rows, biases_d[t][heads], K, cs, cs_search)
                    e += (deq - rows).square().sum().double().item()
                    s += rows.square().sum().double().item()
                    packed_parts.append(packed.cpu())
                packed_cpu = packed_parts[0] if len(packed_parts) == 1 else torch.cat(packed_parts)
                with cv:
                    acc["err_sq"] += e
                    acc["src_sq"] += s
                    dev_rows[dev_idx] += hi - lo
                    results[chunk_i] = packed_cpu
                    cv.notify_all()
        except Exception as ex:
            with cv:
                state["error"] = ex
                cv.notify_all()

    t0 = time.time()
    threads = [threading.Thread(target = worker, args = (d,), daemon = True) for d in devices]
    for t in threads:
        t.start()

    from ..util.progress import ProgressBar
    with ProgressBar("Quantizing n-gram table" if verbose else None, len(chunks)) as pb:
        for i in range(len(chunks)):
            with cv:
                cv.wait_for(lambda: i in results or state["error"])
                if state["error"]:
                    raise state["error"]
                packed_cpu = results.pop(i)
                cv.notify_all()
            writer.write_chunk(packed_cpu)
            pb.update(i + 1)
    for t in threads:
        t.join()
    writer.finalize()

    elapsed = time.time() - t0
    processed = total_rows - start_row
    rfn = math.sqrt(acc["err_sq"] / acc["src_sq"]) if acc["src_sq"] > 0 else 0.0
    stats = {
        "rows": total_rows,
        "processed_rows": processed,
        "K": K,
        "cs": cs_desc,
        "rfn": rfn,  # over this run's rows only when resuming
        "elapsed": elapsed,
        "bytes": os.path.getsize(out_path),
        "dev_rows": dev_rows,
    }
    if verbose:
        print(f" -- done: {processed} rows in {elapsed:.0f} s "
              f"({processed * row_dim / max(elapsed, 1e-9) / 1e6:.1f}M weights/s), "
              f"rfn {rfn:.5f}{' (resumed portion)' if start_row else ''}, "
              f"{stats['bytes'] / 1024**3:.2f} GB -> {out_path}")
        for d in devices:
            print(f"      cuda:{d}: {dev_rows[d]} rows")
    return stats


class NgramTableReader:
    """
    On-demand reader for one table of a quantized n-gram table file: parses the safetensors header
    once, keeps file offsets and the small tensors, and serves arbitrary row indices with pread + GPU
    decode. prefix names the table in a file that holds several.
    """

    def __init__(self, path: str, prefix: str | None = None):
        self.path = path
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        self.metadata = header.pop("__metadata__", {})
        assert self.metadata.get("format") == "exl3_ngram_trellis", "not an exl3 ngram table file"
        self.K = int(self.metadata["K"])
        self.row_dim = int(self.metadata.get("row_dim", ROW_DIM))
        self.cs = self.metadata.get("codebook_scale")  # informational string, not needed for decode
        self.data_base = 8 + hlen
        if prefix is not None:
            header = {k: v for k, v in header.items() if k.startswith(prefix + ".")}
        self.tensors = header

        def resolve(suffix):
            # tolerate both model-prefixed keys (current converter output) and the bare
            # "ngram_embedding.*" keys of earlier files
            keys = [k for k in header if k.endswith(suffix)]
            assert len(keys) == 1, f"expected one *{suffix} tensor in {path}, found {len(keys)}"
            return keys[0]

        def load_small(suffix, dtype):
            info = header[resolve(suffix)]
            with open(path, "rb") as f:
                f.seek(self.data_base + info["data_offsets"][0])
                raw = f.read(info["data_offsets"][1] - info["data_offsets"][0])
            return torch.frombuffer(bytearray(raw), dtype = dtype).view(*info["shape"])

        self.head_bias = load_small(".head_bias", torch.float16)
        self.head_offsets = load_small(".head_offsets", torch.int64)
        self.head_vocab_sizes = load_small(".head_vocab_sizes", torch.int64)
        self.layer_multipliers = load_small(".layer_multipliers", torch.int64)

        # Table data: sharded shard_N.trellis tensors or a single .trellis tensor. Shards are laid
        # out consecutively, so either way the data is one contiguous row array
        shard_keys = [k for k in header if k.endswith(".trellis") and ".shard_" in k]
        if shard_keys:
            shard_keys.sort(key = lambda k: int(k.rsplit(".shard_", 1)[1].split(".")[0]))
            infos = [header[k] for k in shard_keys]
            self.num_rows = sum(i["shape"][0] for i in infos)
            assert all(i["shape"][1] == infos[0]["shape"][1] for i in infos)
            assert all(a["data_offsets"][1] == b["data_offsets"][0]
                       for a, b in zip(infos, infos[1:])), "table shards are not contiguous"
            self.row_words = infos[0]["shape"][1]
            info0 = infos[0]
        else:
            info0 = header[resolve(".trellis")]
            self.num_rows, self.row_words = info0["shape"]
        assert self.row_words == words_per_row(self.K, self.row_dim)
        self.row_bytes = self.row_words * 2
        self.trellis_offset = self.data_base + info0["data_offsets"][0]
        self.fd = os.open(path, os.O_RDONLY)
        self._codebooks = {}

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_rows_packed(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: 1D global row indices (CPU). Returns (N, row_words) int16 (CPU)."""
        idx = indices.cpu().to(torch.int64)
        buf = bytearray(idx.numel() * self.row_bytes)
        mv = memoryview(buf)
        for j, i in enumerate(idx.tolist()):
            data = os.pread(self.fd, self.row_bytes, self.trellis_offset + i * self.row_bytes)
            mv[j * self.row_bytes : (j + 1) * self.row_bytes] = data
        return torch.frombuffer(buf, dtype = torch.int16).view(idx.numel(), self.row_words)

    def _codebook(self, device):
        if device not in self._codebooks:
            self._codebooks[device] = mul1_codebook(device)
        return self._codebooks[device]

    def dequant(self, indices: torch.Tensor, device, out_dtype = torch.half) -> torch.Tensor:
        """Gather + decode rows: returns (N, row_dim) on device."""
        packed = self.read_rows_packed(indices).to(device)
        idx = indices.to(device, torch.int64)
        heads = (torch.searchsorted(self.head_offsets.to(device), idx, right = True) - 1) \
            .clamp(0, self.head_offsets.shape[0] - 1)
        bias = self.head_bias.to(device)[heads]
        return dequant_rows(packed, self.K, self._codebook(device), bias).to(out_dtype)


def prepare_ngram_table_for_conversion(args: dict, config, model) -> str | None:
    """
    First conversion step for models with hashed n-gram embedding tables: produce the quantized
    table file in the output directory (resumable), or copy a pre-quantized file given via
    --ngram_file, then add it to the model's tensor collection so the table modules pick up the
    .trellis format for the calibration forward pass. Returns the table file path, or None when
    the model has no n-gram table.
    """
    from ..modules.ngram_embedding import NGramEmbedding

    ngram_modules = [m for mod in model.modules for m in mod if isinstance(m, NGramEmbedding)]
    if not ngram_modules:
        return None

    out_path = os.path.join(args["out_dir"], "ngram_embedding.safetensors")
    os.makedirs(args["out_dir"], exist_ok = True)

    ngram_file = args.get("ngram_file")
    stc = config.stc

    def source_dtype(m):
        name = f"{m.key}.shard_0.weight" if stc.has_tensor(f"{m.key}.shard_0.weight") else f"{m.key}.weight"
        return stc.file_headers[stc.tensor_file_map[name]][name]["dtype"]

    fixed = [m.key for m in ngram_modules if source_dtype(m) in ("F8_E4M3", "F8_E5M2")]
    if not ngram_file and fixed:
        # Tables stored as one (fp8 + E8M0 scale) tensor each (DeepSeek-V4.1's Engram layers) have no
        # trellis codec yet: they go into the output as they are, streamed file to file
        assert len(fixed) == len(ngram_modules), "mixed n-gram table formats are not supported"
        if args.get("ngram_bits"):
            print(f" !! --ngram_bits ignored: {', '.join(fixed)} are fixed-width tables and are copied unquantized")
        names = []
        for prefix in fixed:
            names += [f"{prefix}.weight"] + ([f"{prefix}.scale"] if stc.has_tensor(f"{prefix}.scale") else [])
        copy_tensors_streamed(stc, names, out_path)
        config.stc.add_tensor_files(out_path, warn_if_override = False)
        return out_path
    if ngram_file:
        with open(ngram_file, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        for m in ngram_modules:
            assert f"{m.key}.trellis" in header or f"{m.key}.shard_0.trellis" in header, \
                f"--ngram_file does not contain {m.key}[.shard_0].trellis (run util/rekey_ngram.py on older files)"
        if os.path.abspath(ngram_file) != os.path.abspath(out_path):
            if not os.path.exists(out_path) or os.path.getsize(out_path) != os.path.getsize(ngram_file):
                print(f" -- Copying pre-quantized n-gram table: {ngram_file}")
                import shutil
                shutil.copyfile(ngram_file, out_path)
            else:
                print(f" -- Pre-quantized n-gram table already in place")
    else:
        K = args.get("ngram_bits") or max(1, min(8, round(args["bits"])))
        devices = [int(d) for d in args["devices"].split(",")]
        print(f" -- Quantizing n-gram embedding table{'s' if len(ngram_modules) > 1 else ''} at K = {K}")
        quantize_ngram_tables(
            [ngram_source_for(m, stc) for m in ngram_modules],
            out_path = out_path,
            K = K,
            devices = devices,
            resume = True,
        )

    # calibration forwards must see the quantized table: the modules detect .trellis over the
    # source weight tensors automatically
    config.stc.add_tensor_files(out_path, warn_if_override = False)
    return out_path


def ngram_source_for(module, stc) -> NgramSource:
    """The source table of an NGramEmbedding module, with the module's own hash layout and row transform"""
    sizes, mult = module.hash_params
    return NgramSource(
        file_map = stc.tensor_file_map,
        prefix = module.key,
        row_dim = module.head_dim,
        head_vocab_sizes = sizes,
        layer_multipliers = mult,
        row_filter = module._fp8_round_trip if module.fp8_rows else None,
    )
