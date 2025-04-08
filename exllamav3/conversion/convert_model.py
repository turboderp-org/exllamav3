import argparse
import torch
import time
import sys
from .. import Config, Model, Tokenizer
from ..modules import Linear
from ..modules.quant import LinearFP16
from ..util.progress import ProgressBar
from ..util.memory import free_mem
from ..util import Timer
from .calibration_data import get_default_calibration
from .compile import compile_model, dsize
from safetensors.torch import save_file
from safetensors import safe_open
import os, shutil
import json

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--in_dir", type = str, default = None, help = "Input (model) directory")
parser.add_argument("-w", "--work_dir", type = str, default = None, help = "Working directory")
parser.add_argument("-o", "--out_dir", type = str, default = None, help = "Output directory")
parser.add_argument("-ss", "--shard_size", type = int, help = "Max shard size in MB, default: 8192")
parser.add_argument("-b", "--bits", type = float, help = "Bits per weight")
parser.add_argument("-hb", "--head_bits", type = int, default = None, help = "Bits per weight, output (head) layer, default: 6")
parser.add_argument("-resume", "--resume", action = "store_true", help = "Resume interrupted job from working directory")
parser.add_argument("-cr", "--cal_rows", type = int, help = "Calibration data size, rows, default: 100")
parser.add_argument("-cc", "--cal_cols", type = int, help = "Calibration data size, columns, default: 2048")
parser.add_argument("-cpi", "--checkpoint_interval", type = int, default = 60, help = "Minimum checkpoint interval, in seconds")
parser.add_argument("-lcpi", "--last_checkpoint_index", type = int, default = None, help = "Last module index to checkpoint (for debug purposes)")
parser.add_argument("-v", "--verbose", action = "store_true", help = "Verbose mode")

num_ref_states = 5

def save_dict(filename, dict_, args):
    path = os.path.join(args["work_dir"], filename)
    with open(path, "w", encoding = "utf8") as f:
        f.write(json.dumps(dict_, indent = 4))


def load_dict(filename, args):
    path = os.path.join(args["work_dir"], filename)
    with open(path, "r", encoding = "utf8") as f:
        return json.load(f)


def load_tensor(filename, args):
    path = os.path.join(args["work_dir"], filename)
    with safe_open(path, framework = "pt", device = "cpu") as f:
        if "tensor" in f.keys():
            return f.get_tensor("tensor")
        else:
            tensors = []
            i = 0
            while f"tensor.{i}" in f.keys():
                tensors.append(f.get_tensor(f"tensor.{i}"))
                i += 1
            return tensors


def save_tensor(tensor, filename: str, args):
    path = os.path.join(args["work_dir"], filename)
    if isinstance(tensor, dict):
        save_file({
            k: v for k, v in tensor.items()
        }, path)
    elif isinstance(tensor, list):
        save_file({
            f"tensor.{i}": t for i, t in enumerate(tensor)
        }, path)
    else:
        save_file({
            f"tensor": tensor
        }, path)


def prepare_env(args):
    qtensors_dir = os.path.join(args["work_dir"], "qtensors")
    ckpt_dir = os.path.join(args["work_dir"], "ckpt")
    os.makedirs(args["work_dir"], exist_ok = True)
    os.makedirs(qtensors_dir, exist_ok = True)
    os.makedirs(ckpt_dir, exist_ok = True)


def prepare(args) -> (dict, bool, str, str):
    if not args.work_dir:
        return None, None, False, "Must specify --work_dir"
    if not args.in_dir and not args.resume:
        return None, None, False, "Specify either --in_dir to start a new job or --resume to resume an interrupted job"
    if not args.out_dir and not args.resume:
        return None, None, False, "Must specify --out_dir or --resume"

    in_args = { "work_dir": args.work_dir }
    if args.resume:
        in_args = load_dict("args.json", in_args)
        in_args["work_dir"] = args.work_dir

    prepare_env(in_args)

    def override(arg, can_override, default):
        if (arg not in args or vars(args)[arg] is None) and arg not in in_args:
            if default is not None:
                in_args[arg] = default
            else:
                raise ValueError(f" ## Missing required argument: {arg}")
        if arg in args and vars(args)[arg] is not None:
            if arg in in_args and vars(args)[arg] and in_args[arg] != vars(args)[arg]:
                if can_override:
                    print(
                        f" !! Warning: Overriding {arg} from existing job, was: {in_args[arg]}, "
                        f"new value: {vars(args)[arg]}"
                    )
                else:
                    raise ValueError(
                        f" ## Error: Resuming job with {arg} = {in_args[arg]}, "
                        f"cannot override with new value of {vars(args)[arg]}. "
                        f"Please start a new job to change this value."
                    )
            in_args[arg] = vars(args)[arg]

    for arg_, can_override, default in [
        ("in_dir", True, None),
        ("out_dir", True, None),
        ("shard_size", True, 8192),
        ("bits", False, None),
        ("head_bits", False, 6),
        ("cal_rows", False, 100),
        ("cal_cols", False, 2048),
        ("checkpoint_interval", True, None),
        ("last_checkpoint_index", True, -1),
    ]:
        override(arg_, can_override, default)

    # Momentary args
    in_args["verbose"] = args.verbose

    if args.resume:
        job_state = load_dict("ckpt/job.json", in_args)
        print(f" -- Resuming existing job")
    else:
        print(f" -- Creating new job")
        job_state = {
            "next_module_idx": 0,
            "surplus_bits": 0,
        }
        save_dict("args.json", in_args, in_args)
        save_dict("ckpt/job.json", job_state, in_args)

    print(f"    Input directory: {in_args['in_dir']}")
    print(f"    Working directory: {in_args['work_dir']}")
    print(f"    Output directory: {in_args['out_dir']}")
    print(f"    Calibration size: {in_args['cal_rows']} rows, {in_args['cal_cols']} columns")
    print(f"    Target bitrate: {in_args['bits']} (decoder), {in_args['head_bits']} (head)")

    return in_args, job_state, True, None


def get_base_model(args):
    config = Config.from_directory(args["in_dir"])
    print(f" -- Loaded model config")
    print(f"    Architecture: {config.architecture}")
    model = Model.from_config(config)
    print(f" -- Created model instance:")
    print(model.get_layout_tree(4))
    tokenizer = Tokenizer.from_config(config)
    print(f" -- Loaded tokenizer")
    print(f"    Vocab size: {tokenizer.actual_vocab_size}")
    return config, model, tokenizer


def prepare_state(args, job_state, config, model, tokenizer):
    idx = job_state["next_module_idx"]
    if idx == 0:
        print(f" -- Preparing input state")
        state = get_default_calibration(args, tokenizer)
    else:
        if idx < len(model.modules):
            print(f" -- Resuming at: {model.modules[idx].key}")
        else:
            print(f" -- Resuming after: {model.modules[idx - 1].key}")
        state = load_tensor("ckpt/state.safetensors", args)
    return state


def get_state_error(x, ref):
     x = x.view(-1, x.shape[-1]).float()
     ref = ref.view(-1, ref.shape[-1]).float()
     err = torch.linalg.norm(x - ref, 'fro') / torch.linalg.norm(ref, 'fro')
     return err.item()


@torch.inference_mode()
def main(args, job_state):

    torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")
    last_checkpoint_time = time.time()

    # Get model
    config, model, tokenizer = get_base_model(args)

    # Get initial state or resume state
    state = prepare_state(args, job_state, config, model, tokenizer)

    # Iterate over modules
    for idx, module in enumerate(model.modules):

        # If resuming, skip along to checkpoint index
        if idx < job_state["next_module_idx"]:
            continue

        # Load current module
        print(f" -- Loading unquantized module: {module.key}")
        module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
        for m in module:
            if m.used_alt_key:
                print(f"     - Cloned {m.key} from {m.alt_key}")

        # Skip modules without quant targets
        qmaps = module.get_qmaps()
        if len(qmaps) > 0:

            # Capture calibration input states during forward pass
            with ProgressBar(f" -- Capturing: {module.key}", len(state)) as progress:
                capture_H = {}
                ref_states = []
                for i in range(len(state)):
                    progress.update(i)
                    params = {
                        "attn_mode": "flash_attn_nc",
                        "capture": capture_H
                    }
                    rs = module.prepare_for_device(state[i], params)
                    rs = module.forward(rs, params)
                    if i < num_ref_states:
                        ref_states.append(rs.cpu())
                    rs = None
            print(f" -- Captured: {module.key}")
            sys.stdout.flush()

            # Swap captured H to system RAM
            for k, v in capture_H.items():
                v["H_swap_device"] = v["H"].device
                v["H"] = v["H"].cpu()

        # Get submodules to quantize
        linears = [m for m in module if isinstance(m, Linear) and m.qmap]

        # Move original tensors to system RAM (load to GPU one by one when quantizing)
        for linear in linears:
            assert isinstance(linear.inner, LinearFP16)
            linear.inner.swap_cpu()

        # Quantization strategy
        if linears:
            strategy, surplus = module.allocate_q(
                {
                    "bits": args["bits"],
                    "head_bits": args["head_bits"],
                },
                job_state["surplus_bits"],
            )
            job_state["surplus_bits"] = surplus
            assert all(m.key in strategy for m in linears), \
                f" ## Logic error, no quantization strategy for {m.key}"

        # Quantize module
        for linear in linears:
            linear.inner.unswap_cpu()
            quant_args = {
                "seed": idx,
                "K": strategy[linear.key],
            }
            with Timer() as t:
                proxy_err = linear.convert_exl3(
                    capture_H[linear.qmap],
                    quant_args = quant_args,
                    progress_str = f" -- <step>: {linear.key}",
                    verbose = args["verbose"]
                )
            print(
                f" -- Quantized: {linear.key:{config.stc.max_key_len()}}"
                f"  bpw: {quant_args['K']:.2f}"
                f"  proxy_err: {proxy_err:8.6f}"
                f"  [{t.interval:4.2f} s]"
            )
            sys.stdout.flush()

        # Save converted module tensors
        tensors = {}
        for m in module:
            tensors.update(m.get_tensors())
        save_tensor(tensors, f"qtensors/{module.key}.safetensors", args)

        # Output final bpw for layer
        num_bytes = dsize(tensors)
        num_bits = num_bytes * 8
        final_bpw = num_bits / module.weights_numel()
        print(
            f" -- Quantized: {module.key:{config.stc.max_key_len()}}"
            f"  bpw: {final_bpw:.2f}"
        )

        del tensors
        free_mem()

        # Advance state
        error = 0
        with ProgressBar(f" -- Forward pass: {module.key}", len(state)) as progress:
            for i in range(len(state)):
                progress.update(i)
                params = {
                    "attn_mode": "flash_attn_nc",
                }
                state[i] = module.prepare_for_device(state[i], params)
                if i < num_ref_states or idx < len(model.modules) - 1:
                    state[i] = module.forward(state[i], params).cpu()
                if i < num_ref_states and len(linears):
                    ref_states[i] = ref_states[i].to(state[i].device)
                    error += get_state_error(state[i], ref_states[i])
                    ref_states[i] = None
        error /= num_ref_states
        print(f" -- Finished module: {module.key}, rfn: {error:.6f}")
        sys.stdout.flush()

        # Unload current module
        module.unload()
        free_mem()

        # Checkpoint
        job_state["next_module_idx"] = idx + 1
        if time.time() > last_checkpoint_time + args["checkpoint_interval"] and \
            (args.get("last_checkpoint_index", -1) < 0 or idx <= args["last_checkpoint_index"]):
            print(f" -- Saving checkpoint")
            ckpt_dir = os.path.join(args["work_dir"], "ckpt")
            ckpt_dir_old = os.path.join(args["work_dir"], "ckpt_old")
            ckpt_dir_new = os.path.join(args["work_dir"], "ckpt_new")
            os.makedirs(ckpt_dir_new, exist_ok = True)
            save_dict("ckpt_new/job.json", job_state, args)
            save_tensor(state, "ckpt_new/state.safetensors", args)
            if os.path.exists(ckpt_dir_old):
                shutil.rmtree(ckpt_dir_old)
            os.rename(ckpt_dir, ckpt_dir_old)
            os.rename(ckpt_dir_new, ckpt_dir)
            last_checkpoint_time = time.time()

    # Compile model
    compile_model(args, model, config, tokenizer)

    # All done
    print(" -- All done")