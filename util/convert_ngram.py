import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch

from exllamav3.conversion.ngram import quantize_ngram_table, NgramTableReader, ROW_DIM, words_per_row

"""
Standalone quantizer for hashed n-gram embedding tables (Qwen3.8-Flash-Next ple_embedding).

Quantizes every row of the table as a 160-D tail-biting trellis ring (mul1 codebook, per-row fp16
scale, per-hash-head bias vector) into a single ngram_embedding.safetensors file that the model's
embedding module can either preload or stream rows from at runtime. Storage is 160*K + 16 bits per
row. convert_model.py can reuse the same routine, or skip it when this file is supplied.

Example:
    python util/convert_ngram.py \
        -i /path/to/model \
        -o /path/to/ngram_embedding_k3.safetensors \
        -ngb 3 \
        -d 0,1,2
"""

def main():
    parser = argparse.ArgumentParser(description = "Quantize n-gram embedding table to trellis format")
    parser.add_argument("-i", "--in_dir", type = str, required = True, help = "Input HF model directory")
    parser.add_argument("-o", "--out_file", type = str, required = True, help = "Output .safetensors file")
    parser.add_argument("-ngb", "--ngram_bits", type = int, required = True, help = "Bits per weight (K), 1..8")
    parser.add_argument("-d", "--devices", type = str, default = "0", help = "CUDA devices, e.g. 0,1,2 (default: 0)")
    parser.add_argument("-cs", "--codebook_scale", type = float, default = None,
                        help = "Fixed codebook scale multiplier (default: per-row heuristic, "
                               "cs = clamp(gamma / (absmax/rms)))")
    parser.add_argument("--cs_search", type = int, default = 1,
                        help = "Encodes per row: sweep cs in 0.06 steps around the heuristic/fixed value "
                               "and keep the lowest-error encode per row (default: 1, no search)")
    parser.add_argument("--chunk_rows", type = int, default = 131072, help = "Rows per work chunk")
    parser.add_argument("--bias_sample_rows", type = int, default = 131072,
                        help = "Sampled rows per hash head for the bias vectors")
    parser.add_argument("--limit_rows", type = int, default = None,
                        help = "Quantize only the first N rows (for testing)")
    parser.add_argument("-r", "--resume", action = "store_true",
                        help = "Continue an interrupted run: reuse the partial output file's bias vectors "
                               "and restart from the last complete chunk (parameters must match)")
    parser.add_argument("--verify_rows", type = int, default = 16384,
                        help = "Random rows to re-read from the output file and verify (0 = skip)")
    args = parser.parse_args()

    if not 1 <= args.ngram_bits <= 8:
        parser.error("ngram_bits must be in range 1..8")
    devices = [int(d) for d in args.devices.split(",")]
    for d in devices:
        name = torch.cuda.get_device_name(d)
        print(f" -- cuda:{d}: {name}")

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    os.makedirs(out_dir, exist_ok = True)

    if args.cs_search < 1:
        parser.error("cs_search must be >= 1")
    stats = quantize_ngram_table(
        model_dir = args.in_dir,
        out_path = args.out_file,
        K = args.ngram_bits,
        devices = devices,
        cs = args.codebook_scale,
        cs_search = args.cs_search,
        chunk_rows = args.chunk_rows,
        limit_rows = args.limit_rows,
        bias_sample_rows = args.bias_sample_rows,
        resume = args.resume,
    )
    bpw = (16 + ROW_DIM * args.ngram_bits) / ROW_DIM
    print(f" -- effective bpw: {bpw:.3f} ({words_per_row(args.ngram_bits)} words/row)")

    if args.verify_rows:
        print(f" -- verifying {args.verify_rows} random rows against source")
        from exllamav3.conversion.ngram import NgramSource
        reader = NgramTableReader(args.out_file)
        source = NgramSource(args.in_dir)
        dev = f"cuda:{devices[0]}"
        torch.manual_seed(0)
        idx = torch.randint(0, reader.num_rows, (args.verify_rows,)).unique().sort().values
        deq = reader.dequant(idx, dev, out_dtype = torch.float)
        src = torch.cat([source.read_rows(i, i + 1) for i in idx.tolist()]).to(dev).float()
        rfn = ((deq - src).square().sum().sqrt() / src.square().sum().sqrt()).item()
        print(f" -- read-back rfn: {rfn:.5f} (in-line measurement was {stats['rfn']:.5f})")
        reader.close()


if __name__ == "__main__":
    main()
