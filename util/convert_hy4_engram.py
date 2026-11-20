"""
Quantize the Engram tables of a Hy4-flash checkpoint into exl3_ngram_trellis files, one per
table (see exllamav3/conversion/hy4_engram.py). The output directory is what convert.py takes as
--ngram_file for this architecture. Resumable.

    python util/convert_hy4_engram.py -i <model_dir> -o <out_dir> -K 5 [-l 1,11] [-d 0] [--limit_rows N]
"""
import argparse, json, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model
from exllamav3.conversion.hy4_engram import quantize_engram_table, table_file_name, table_complete
from exllamav3.modules.arch_specific.hy_v4_flash import HyV4EngramTable

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in_dir", required = True)
parser.add_argument("-o", "--out_dir", required = True)
parser.add_argument("-K", "--bits", type = int, default = 5)
parser.add_argument("-l", "--layers", type = str, default = None, help = "layer indices of the tables to do, default all")
parser.add_argument("-d", "--device", type = int, default = 0)
parser.add_argument("--limit_rows", type = int, default = None, help = "first N rows only (speed probe; not a usable table)")
parser.add_argument("--chunk_rows", type = int, default = 262144)
parser.add_argument("--cs_search", type = int, default = 1)
parser.add_argument("--stored_target", action = "store_true", help = "encode the stored rows instead of the fp8-rounded ones")
args = parser.parse_args()

torch.set_grad_enabled(False)
config = Config.from_directory(args.in_dir)
model = Model.from_config(config)
tables = [m for mod in model.modules for m in mod if isinstance(m, HyV4EngramTable)]
want = None if args.layers is None else {int(x) for x in args.layers.split(",")}
os.makedirs(args.out_dir, exist_ok = True)
stats = []
for t in tables:
    if want is not None and (t.layer_id - 1) not in want:
        continue
    path = os.path.join(args.out_dir, table_file_name(t) + (".probe" if args.limit_rows else ""))
    if not args.limit_rows and table_complete(path, t, args.bits):
        print(f" -- complete: {path}")
        continue
    stats.append(quantize_engram_table(
        config.stc, t, path, args.bits, f"cuda:{args.device}", fp8 = t.fp8_rows and not args.stored_target,
        cs_search = args.cs_search, chunk_rows = args.chunk_rows, limit_rows = args.limit_rows, resume = True))
    sp = os.path.join(args.out_dir, "engram_quant.json")
    old = json.load(open(sp)) if os.path.exists(sp) else []
    json.dump(old + [stats[-1]], open(sp, "w"), indent = 1)
print(json.dumps(stats, indent = 1))
