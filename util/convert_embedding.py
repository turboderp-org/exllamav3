"""
Standalone quantizer for token-embedding tables: trellis VQ in the QTIP-transformed space
(exl3_trellis_embed format, 16*G + D*K bits per row), producing the embedding-trellis.safetensors
file that the model's Embedding module loads in place of the fp16 table (host-resident,
device-mapped fused dequant; see exllamav3/conversion/embed.py for the format contract).

The output file goes next to the model's other .safetensors files (any directory the loader
scans); the tensor key stem must match the model's embedding module key (default stem:
model.embed_tokens).

Example:
    python util/convert_embedding.py \
        -i /path/to/model \
        -o /path/to/model/embedding-trellis.safetensors \
        -K 8 --seed 0
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # fork package must win over any installed copy
import argparse
import json
import math
import torch

from exllamav3.conversion.embed import (
    quantize_trellis_table, TrellisEmbedTableReader, DEFAULT_TENSOR_KEY,
)

def main():
    parser = argparse.ArgumentParser(description = "Quantize token-embedding table to the exl3_trellis_embed format")
    parser.add_argument("-i", "--in_dir", type = str, required = True,
                        help = "Input HF model directory (or directory containing the source .safetensors)")
    parser.add_argument("-o", "--out_file", type = str, required = True, help = "Output .safetensors file")
    parser.add_argument("-K", "--trellis_bits", type = int, required = True, help = "Bits per position (K), 6..8")
    parser.add_argument("--seed", type = int, default = 0, help = "LCG sign-stream seed (pinned in metadata)")
    parser.add_argument("-t", "--tensor_key", type = str, default = DEFAULT_TENSOR_KEY,
                        help = f"Source embedding tensor key (default: {DEFAULT_TENSOR_KEY})")
    parser.add_argument("-d", "--devices", type = str, default = "0",
                        help = "CUDA device list for the transform/quality passes; only the "
                               "FIRST listed device is used (default: 0)")
    parser.add_argument("--chunk_rows", type = int, default = 8192, help = "Rows per work chunk")
    parser.add_argument("--limit_rows", type = int, default = None, help = "Quantize only the first N rows (for testing)")
    parser.add_argument("--threads", type = int, default = 0,
                        help = "Encoder threads (0 = all cores, capped so the per-thread history scratch stays under ~512 MiB)")
    parser.add_argument("--quality_sample", type = int, default = None,
                        help = "Report quality on a deterministic sample of at most N encoded rows across the "
                               "whole run, allocated proportionally per chunk (minimum 1 per chunk; "
                               "0 = skip the quality report; default: all encoded rows)")
    parser.add_argument("-r", "--resume", action = "store_true",
                        help = "Continue an interrupted run: reuse the partial output file's column scales "
                               "and restart from the last complete chunk (parameters must match)")
    parser.add_argument("--force", action = "store_true",
                        help = "Overwrite an existing output file (without --resume the file is truncated)")
    parser.add_argument("--verify_rows", type = int, default = 16384,
                        help = "Random rows to re-read from the output file and verify (0 = skip)")
    parser.add_argument("--quantization_config_emit", action = "store_true",
                        help = "Print the informational embedding_trellis quantization_config dict")
    args = parser.parse_args()

    if not 6 <= args.trellis_bits <= 8:
        parser.error("trellis_bits must be in range 6..8 (fused dequant kernel range)")
    if args.chunk_rows <= 0:
        parser.error("chunk_rows must be > 0")
    if args.limit_rows is not None and args.limit_rows <= 0:
        parser.error("limit_rows must be > 0 (0 rows is not a table)")
    if args.verify_rows < 0:
        parser.error("verify_rows must be >= 0 (0 = skip verification)")
    if args.quality_sample is not None and args.quality_sample < 0:
        parser.error("quality_sample must be >= 0 (0 = skip the quality report)")
    try:
        devices = [int(d) for d in args.devices.split(",")]
    except ValueError:
        parser.error(f"invalid device list: {args.devices!r} (expected e.g. '0' or '0,1')")
    if torch.cuda.is_available():
        for d in devices:
            if d < 0 or d >= torch.cuda.device_count():
                parser.error(f"device {d} does not exist ({torch.cuda.device_count()} CUDA device(s) available)")
            print(f" -- cuda:{d}: {torch.cuda.get_device_name(d)}")
    elif devices:
        print(" -- no CUDA available: the requested device list is ignored (CPU transform)")

    if os.path.exists(args.out_file) and not args.resume and not args.force:
        parser.error(f"{args.out_file} exists (use --resume to continue it or --force to overwrite)")
    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    os.makedirs(out_dir, exist_ok = True)
    if args.limit_rows is not None:
        print(f" -- NOTE: --limit_rows = {args.limit_rows}: the output is a TRUNCATED test table "
              f"(inference raises IndexError for ids >= {args.limit_rows}); not a loadable full-vocab table")

    # encoder decision (measured 2026-09-23): the production quantize_tiles CUDA kernel
    # only accepts tile lengths 256/160 (quantize.cu), not the D-length full-row rings of
    # this format, so quantize_trellis_table always encodes with the AVX2 CPU Viterbi port
    # in exllamav3/conversion/embed.py
    print(" -- encoder: CPU AVX2 Viterbi port (exllamav3/conversion/embed.py)")

    stats = quantize_trellis_table(
        source = args.in_dir,
        out_path = args.out_file,
        K = args.trellis_bits,
        seed = args.seed,
        key = args.tensor_key,
        chunk_rows = args.chunk_rows,
        limit_rows = args.limit_rows,
        resume = args.resume,
        threads = args.threads,
        devices = devices,
        quality_sample = args.quality_sample,
    )
    print(f" -- effective bpw: {stats['bpw']:.3f}")
    if stats["sqnr_db"] is not None:
        print(f" -- quality (decoder-vs-source, {stats['measured_rows']} rows measured): "
              f"SQNR {stats['sqnr_db']:.2f} dB, rfn {stats['rfn']:.5f}, encoder {stats['encoder']}")

    if args.verify_rows:
        print(f" -- verifying {args.verify_rows} random rows against source")
        with TrellisEmbedTableReader(args.out_file) as reader:
            gen = torch.Generator().manual_seed(0)
            idx = torch.randperm(reader.num_rows, generator = gen)[:min(args.verify_rows, reader.num_rows)]
            # EmbedSource.read_rows_indexed coalesces only SORTED indices into slice reads;
            # sorting once keeps both sides aligned (each read returns rows in its own idx
            # order) and avoids ~verify_rows single-row slices + a huge torch.cat in the source
            idx, _ = idx.sort()
            deq = reader.dequant(idx)
            from exllamav3.conversion.embed import EmbedSource
            source = EmbedSource(args.in_dir, args.tensor_key)
            try:
                src = source.read_rows_indexed(idx).float()
            finally:
                source.close()
            rfn = ((deq - src).square().sum().sqrt() / src.square().sum().sqrt()).item()
            if stats["rfn"] is not None:
                # gate, not just a report: the read-back estimates the same rfn as the
                # in-line measurement (both decode-vs-source), so a codec/format drift
                # (wrong seed, transform, codebook, scale layout) would decode to
                # near-garbage and land far above it; the margin is far above the
                # sampling noise of a 16k-row read-back, and the absolute floor covers a
                # quality_sample-limited in-line measurement
                limit = max(stats["rfn"] * 4.0, stats["rfn"] + 0.02)
                if not math.isfinite(rfn) or rfn > limit:
                    raise SystemExit(
                        f" -- verification FAILED: read-back rfn {rfn:.5f} > {limit:.5f} "
                        f"(in-line measurement {stats['rfn']:.5f}): the output file does not "
                        f"decode to the source table - do not use it")
            ref = f" (in-line measurement was {stats['rfn']:.5f})" if stats["rfn"] is not None else ""
            print(f" -- read-back rfn: {rfn:.5f}{ref} - OK")

    if args.quantization_config_emit:
        with TrellisEmbedTableReader(args.out_file) as reader:
            cfg = {"embedding_trellis": {"K": reader.K, "G": reader.G, "seed": reader.seed,
                                         "codebook": "mul1", "transform": "qtip1"}}
            print(json.dumps(cfg, indent = 2))


if __name__ == "__main__":
    main()
