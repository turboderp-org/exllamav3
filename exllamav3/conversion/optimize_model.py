import argparse
import torch
from .. import Config, Model, Tokenizer
from ..util.progress import ProgressBar
from .calibration_data import get_default_calibration
from ..modules import Linear
import json
import torch.nn.functional as F
import math
from .compile import compile_model
from ..loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection

col_default = "\u001b[0m"
col_red = "\u001b[31;1m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_purple = "\u001b[35;1m"
col_cyan = "\u001b[36;1m"
col_white = "\u001b[37;1m"

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--measurement", type = str, default = None, help = "Input measurement.json file from measure.py")
parser.add_argument("-o", "--out_dir", type = str, default = None, help = "Output directory for compiled model")
parser.add_argument("-b", "--bitrate", type = float, default = None, help = "Target bitrate")
parser.add_argument("-ss", "--shard_size", type = int, help = "Max shard size in MB, default: 8192", default = 8192)
parser.add_argument("-i", "--in_dir", nargs = "+", type = str, default = None, help = "Optional input (quantized model) directories")

def prepare(args) -> (dict, dict, bool, str):
    if not args.measurement:
        return None, None, False, "Please specify --measurement"
    if not args.out_dir:
        return None, None, False, "Please specify --out_dir"
    if args.bitrate is None:
        return None, None, False, "No bitrate specified"
    if args.in_dir and len(args.in_dir) < 2:
        return None, None, False, "Must specify at least two input models (or none)"

    in_args = {
        "measurement": args.measurement,
        "out_dir": args.out_dir,
        "bitrate": args.bitrate,
        "shard_size": args.shard_size,
        "in_dir": args.in_dir,
    }

    job_state = {}

    warn_experimental = False
    print(f"    Input measurement: {in_args['measurement']}")
    print(f"    Output directory: {in_args['out_dir']}")
    print(f"    Target bitrate: {in_args['bitrate']:.2f}")

    if warn_experimental:
        print(
            f" !! {col_red}WARNING, experimental options are selected. The quantized model may not work in future "
            f"versions of ExLlamaV3 {col_default}"
        )

    return in_args, job_state, True, None


def optimize(meas, base_numel, base_cost, target_cost, base_kld, num_q):
    groups = meas["groups"]
    num_groups = len(groups)
    solution = [0] * num_groups
    budget = target_cost - base_cost

    def adjust(dkld):
        if dkld > 0:
            return dkld
        return -((-dkld) ** 0.69)

    print(" -- Optimizing...")
    while True:
        best = None
        best_r = 0.0
        for i, g in enumerate(groups):
            cand = g["candidates"]
            s = solution[i]
            for j, c in enumerate(cand):
                if j < s or j >= num_q: continue
                dk = adjust(c["dkld"])
                db = c["dbits"]
                if s > 0:
                    dk -= adjust(cand[s - 1]["dkld"])
                    db -= cand[s - 1]["dbits"]
                r = 1e10 * dk / (db + 1)
                if r < best_r and budget > db:
                    best = i, j, db
                    best_r = r
        if best is None:
            break
        i, j, k = best
        solution[i] = j + 1
        budget -= k

    return solution


@torch.inference_mode()
def main(args, job_state):

    torch.set_grad_enabled(False)

    # Load measurement
    with open(args["measurement"], "r", encoding = "utf8") as f:
        meas = json.load(f)

    # Get models
    if not args["in_dir"]:
        dir_base = meas["base"]["dir"]
        dir_q = [x["dir"] for x in meas["alts"]]
    else:
        dir_base = args["in_dir"][0]
        dir_q = args["in_dir"][1:]
    num_q = len(dir_q)
    config_base = Config.from_directory(dir_base)
    config_q = [Config.from_directory(d) for d in dir_q]
    print(f" -- Loaded model config")
    print(f"    Architecture: {config_base.architecture}")
    model_base = Model.from_config(config_base)
    model_q = [Model.from_config(c) for c in config_q]
    print(f" -- Created model instances:")
    r_layout = model_base.get_layout_tree(4)
    print(r_layout)
    tokenizer = Tokenizer.from_config(config_base)
    print(f" -- Loaded tokenizer")
    print(f"    Vocab size: {tokenizer.actual_vocab_size}")

    # Costs
    base_numel = 0
    base_cost = 0
    for g in meas["groups"]:
        for l in g["layers"]:
            module = model_base.find_module(l)
            base_cost += module.storage_size() * 8
            base_numel += module.weights_numel()
    target_cost = args["bitrate"] * base_numel

    # Optimize
    solution = optimize(meas, base_numel, base_cost, target_cost, meas["base_kld"], num_q)
    # for i, s in enumerate(solution):
    #     print(i, s)

    # Tensor overrides
    groups = meas["groups"]
    overrides = [[] for _ in range(num_q)]
    for s, g in zip(solution, groups):
        if s > 0:
            overrides[s - 1] += g["layers"]
    vstc = VariantSafetensorsCollection(config_base.stc)
    for config, keys in zip(config_q, overrides):
        keysp = [k + ".*" for k in keys]
        vstc.add_stc(keysp, config.stc)
    config_base.stc = vstc

    # New bpw etc.
    bpw_layer, bpw_head, vram_bits = model_base.get_storage_info()
    bpw_layer = round(bpw_layer, 2)
    bpw_head = round(bpw_head)
    print(f" -- New estimated model bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")

    # Recompile model
    compile_args = {
        "bits": bpw_layer,
        "head_bits": bpw_head,
        "in_dir": dir_base,
        "out_dir": args["out_dir"],
        "shard_size": args["shard_size"],
        "model_stc": True
    }

    compile_model(compile_args, model_base, config_base, None)