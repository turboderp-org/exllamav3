import argparse
import torch
import time
import sys
from .. import Config, Model, Tokenizer
from ..modules import Linear
from ..modules.linear import convert_exl3_group
from ..modules.quant.exl3_lib.quantize import auto_split
from ..modules.quant import LinearFP16, LinearEXL3
from ..util.progress import ProgressBar
from ..util.memory import free_mem
from ..util import Timer, human_time
from ..util.tensor import save_tensor_image
from ..util.measures import cosine_error, sqnr
from .calibration_data import get_default_calibration
from .compile import compile_model, dsize
from .allocation import create_q_strategy, print_strategy
from ..loader.safetensors_alt import save_file, safe_open
import os, shutil
import json
import threading
from pathlib import Path
from collections import deque
import re

col_default = "\u001b[0m"
col_red = "\u001b[31;1m"

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

parser = argparse.ArgumentParser(allow_abbrev = False)
parser.add_argument("-i", "--in_dir", type = str, default = None, help = "Input (model) directory")
parser.add_argument("-w", "--work_dir", type = str, default = None, help = "Working directory")
parser.add_argument("-o", "--out_dir", type = str, default = None, help = "Output directory")
parser.add_argument("-ss", "--shard_size", type = int, help = "Max shard size in MB, default: 8192")
parser.add_argument("-b", "--bits", type = float, help = "Bits per weight")
parser.add_argument("-hb", "--head_bits", type = int, default = None, help = "Bits per weight, output (head) layer, default: 6")
parser.add_argument("-mb", "--mtp_bits", type = int, default = None, help = "Bits per weight, MTP layers, default: 4")
parser.add_argument("-hq", "--hq", action = "store_true", help = "Increase bitrate of select layers for supported models (MoE mostly)")
parser.add_argument("-r", "--resume", action = "store_true", help = "Resume interrupted job from working directory")
parser.add_argument("-cr", "--cal_rows", type = int, help = "Calibration data size, rows, default: 250")
parser.add_argument("-cc", "--cal_cols", type = int, help = "Calibration data size, columns, default: 2048")
parser.add_argument("-cpi", "--checkpoint_interval", type = int, default = 120, help = "Minimum checkpoint interval, in seconds")
parser.add_argument("-lcpi", "--last_checkpoint_index", type = int, default = None, help = "Last module index to checkpoint (for debug purposes)")
parser.add_argument("-v", "--verbose", action = "store_true", help = "Verbose mode")
parser.add_argument("-d", "--devices", type = str, default = "0", help = "List of devices to use for quantization, e.g. --devices 0,1,2")
parser.add_argument("-dr", "--device_ratios", type = str, default = "", help = "Split ratio for devices, e.g. --device_ratio 2,2,4")
parser.add_argument("-img", "--image_dump", action = "store_true", help = "Save model tensors as images (saved to working directory)")
parser.add_argument("-cb", "--codebook", type = str, default = "mul1", help = "Codebook: mul1 (default), mcg or 3inst")
parser.add_argument("-pm", "--parallel_mode", action = "store_true", help = "When possible, use new parallel mode for small tensors (MoE layers especially)")
parser.add_argument("--max_module", type = int, help = "End quantization after this many modules, includes embedding and norm layers (for debug purposes)", default = None)

group = parser.add_mutually_exclusive_group()
group.add_argument("--out_scales", type = str, default = "always", help = "Enable out channel scales (always/never/auto, default: always)")

parser.add_argument("--override_anyway", action = "store_true", help = "Allow resuming even when overriding settings that will break the existing job.")

num_ref_states = 5

progress_lock = threading.Lock()
curr_progress = 0
max_progress = 0

def check_system():
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") is not None:
        print(
            "\n"
            f" !! {col_red}IMPORTANT: The environment variable TORCH_ALLOW_TF32_CUBLAS_OVERRIDE is set!{col_default}\n"
            f" !! {col_red}This causes Torch to run in reduced precision mode, which is likely to cause this "
            f"script to fail or result in broken models.{col_default}\n"
            "\n"
        )


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
    images_dir = os.path.join(args["work_dir"], "images")
    os.makedirs(args["work_dir"], exist_ok = True)
    os.makedirs(qtensors_dir, exist_ok = True)
    os.makedirs(ckpt_dir, exist_ok = True)
    os.makedirs(images_dir, exist_ok = True)


def prepare(args) -> (dict, dict, bool, str):
    check_system()

    if not args.work_dir:
        return None, None, False, "Must specify --work_dir"
    if not args.in_dir and not args.resume:
        return None, None, False, "Specify either --in_dir to start a new job or --resume to resume an interrupted job"
    if not args.out_dir and not args.resume:
        return None, None, False, "Must specify --out_dir or --resume"
    if args.codebook not in ["mcg", "mul1", "3inst"]:
        return None, None, False, "Codebook must be 'mcg', 'mul1' or '3inst'"
    if args.bits is not None and (args.bits > 8 or args.bits < 1):
        return None, None, False, "--bits must be between 1 and 8"
    if args.head_bits is not None and (args.head_bits > 8 or args.head_bits < 1) and args.head_bits != 16:
        return None, None, False, "--head_bits must be between 1 and 8, or 16"

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
        ("mtp_bits", False, 4),
        ("hq", False, False),
        ("cal_rows", False, 250),
        ("cal_cols", False, 2048),
        ("checkpoint_interval", True, None),
        ("last_checkpoint_index", True, -1),
        ("devices", True, None),
        ("device_ratios", True, None),
        ("codebook", True, "mul1"),
        ("parallel_mode", True, False),
    ]:
        override(arg_, can_override if not args.override_anyway else True, default)

    # Momentary args
    in_args["image_dump"] = args.image_dump
    in_args["verbose"] = args.verbose
    in_args["apply_out_scales"] = {"always": True, "never": False, "auto": None}[args.out_scales]
    in_args["max_module"] = args.max_module

    if args.resume:
        job_state = load_dict("ckpt/job.json", in_args)
        print(f" -- Resuming existing job")
    else:
        print(f" -- Creating new job")
        job_state = {
            "next_module_idx": 0,
            "q_strategy": None,
        }
        save_dict("args.json", in_args, in_args)
        save_dict("ckpt/job.json", job_state, in_args)

    warn_experimental = False
    print(f"    Input directory: {in_args['in_dir']}")
    print(f"    Output directory: {in_args['out_dir']}")
    print(f"    Working directory: {in_args['work_dir']}")
    print(f"    Calibration size: {in_args['cal_rows']} rows, {in_args['cal_cols']} columns")
    print(f"    Target bitrate: {in_args['bits']} (decoder), {in_args['head_bits']} (head)")
    print(f"    Output scales: " + {True: "always", False: "never", None: "auto"}[in_args["apply_out_scales"]])
    print(f"    Codebook: {in_args['codebook']}")

    if warn_experimental:
        print(
            f" !! {col_red}WARNING, experimental options are selected. The quantized model may not work in future "
            f"versions of ExLlamaV3 {col_default}"
        )

    if not args.resume:
        qt_dir = os.path.join(in_args["work_dir"], "qtensors")
        clear = [p for p in Path(qt_dir).glob("*.safetensors") if p.is_file()]
        if len(clear):
            print(f" !! Deleting {len(clear)} safetensors file(s) from {qt_dir}")
        in_args["clear_files"] = clear

    return in_args, job_state, True, None


def get_base_model(args):
    config = Config.from_directory(args["in_dir"])
    print(f" -- Loaded model config")
    print(f"    Architecture: {config.architecture}")
    model = Model.from_config(config)
    use_reference_state = not model.caps.get("uncalibrated_quantize", False)
    assert model.caps.get("can_quantize", True), "Cannot quantize this model type."
    print(f" -- Created model instance:")
    print(model.get_layout_tree(4))
    mtp_model = model.from_config(config, component = "mtp") if "mtp" in config.model_classes else None
    if mtp_model:
        print(f" -- Created MTP model instance:")
        print(mtp_model.get_layout_tree(4))
    if use_reference_state:
        tokenizer = Tokenizer.from_config(config)
        print(f" -- Loaded tokenizer")
        print(f"    Vocab size: {tokenizer.actual_vocab_size}")
    else:
        tokenizer = None
    if hasattr(config, "rope_settings"):
        config.rope_settings.print()
    return config, model, mtp_model, tokenizer, use_reference_state


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
     sq = sqnr(x, ref)
     cos = cosine_error(x, ref)
     return err.item(), cos, sq


def make_quant_args(args, idx, K, devices, device_ratios = None):
    quant_args = {
        "seed": idx,
        "K": K,
        "devices": devices,
        "device_ratios": device_ratios,
        "apply_out_scales": args["apply_out_scales"],
        "debug_dir": os.path.join(args["work_dir"], "debug"),
    }
    if args["codebook"] == "mcg":
        quant_args.update({"mcg": True})
    elif args["codebook"] == "mul1":
        quant_args.update({"mul1": True})
    return quant_args


def print_quantized_linear(config, linear, quant_args, proxy_err, time_str = ""):
    flags = "o" if quant_args["apply_out_scales"] else "."
    flags += "f" if quant_args["q_fallback"] else "."
    proxy_err_str = (
        "(zero)  " if quant_args["zeros"] else
        "(big)   " if proxy_err >= 9.9 else
        f"{proxy_err:8.6f}" if proxy_err >= 0.0 else
        "(OoM)   "
    )
    proxy_err_label = "proxy_err" if not quant_args["q_fallback"] else "rmse     "
    print(
        f" -- Quantized: {linear.key:{config.stc.max_key_len() + 8}}"
        f"  bpw: {quant_args['K']:5.2f}"
        f"  {proxy_err_label}: {proxy_err_str}"
        f"  {flags}"
        f"  g_sc: {quant_args['g_scale']:.6f}"
        f"{time_str}",
        flush = True
    )


def group_quant_linears(linears, strategy, capture_H, max_cat_cols = 32768, max_stack = 16):
    """
    Partition linears into groups for batched quantization. Tensors sharing a Hessian (same qmap) at the same
    bitrate concatenate along out_features; leftover same-shape, same-bitrate tensors with individual Hessians
    (block-sparse down projections) stack with per-tensor L. Group sizes are capped so the LDLQ work buffers
    stay modest. Returns a list of groups; singleton groups take the plain per-tensor path.
    """
    if capture_H is None:
        return [[linear] for linear in linears]

    groups = []
    shared = {}
    for linear in linears:
        K = strategy[linear.key]
        if K == 16 or linear.qmap not in capture_H:
            groups.append([linear])
        else:
            shared.setdefault((linear.qmap, K, linear.in_features), []).append(linear)

    loners = {}
    for (qmap, K, k), ls in shared.items():
        if len(ls) == 1:
            linear = ls[0]
            loners.setdefault((linear.in_features, linear.out_features, K), []).append(linear)
            continue
        # Shared-Hessian group: chunk by total column count
        cap = max(ls[0].out_features, min(max_cat_cols, int(3.5e8) // k))
        chunk, cols = [], 0
        for linear in ls:
            if chunk and cols + linear.out_features > cap:
                groups.append(chunk)
                chunk, cols = [], 0
            chunk.append(linear)
            cols += linear.out_features
        if chunk:
            groups.append(chunk)

    for (k, n, K), ls in loners.items():
        # Same-shape stacking group: chunk by batch size
        cap = max(1, min(max_stack, int(3.5e8) // (k * n)))
        for i in range(0, len(ls), cap):
            groups.append(ls[i : i + cap])

    return groups


def group_label(group):
    prefix = os.path.commonprefix([linear.key for linear in group])
    return f"{prefix}* ({len(group)} tensors)"


def quantize_linears_single(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state):

    allow_grouping = state is not None and not args["image_dump"] and not args["verbose"]
    groups = group_quant_linears(linears, strategy, capture_H if allow_grouping else None)

    for group in groups:
        if len(group) > 1:
            quant_args_list = [make_quant_args(args, idx, strategy[l.key], devices, device_ratios) for l in group]
            with Timer() as t:
                proxy_errs = convert_exl3_group(
                    group,
                    [capture_H[l.qmap] for l in group],
                    quant_args_list,
                    progress_str = f" -- <step>: {group_label(group)}",
                )
                for linear in group:
                    assert isinstance(linear.inner, LinearEXL3)
                    linear.inner.swap_cpu()
            time_str = f"  [{t.interval / len(group):4.2f} s]"
            for linear, quant_args, proxy_err in zip(group, quant_args_list, proxy_errs):
                print_quantized_linear(config, linear, quant_args, proxy_err, time_str)
            continue

        linear = group[0]
        if strategy[linear.key] == 16:
            print(
                f" -- Unquantized: {linear.key:{config.stc.max_key_len() + 6}}"
                f"  bpw: {16:5.2f}",
                flush = True
            )
        else:
            quant_args = make_quant_args(args, idx, strategy[linear.key], devices, device_ratios)

            with Timer() as t:
                sr = os.path.join(args["work_dir"], f"images/{linear.key}.reg.jpg") \
                    if args["image_dump"] else None
                proxy_err = linear.convert_exl3(
                    capture_H[linear.qmap] if state else linear.init_H_data(False),
                    quant_args = quant_args,
                    progress_str = f" -- <step>: {linear.key}",
                    verbose = args["verbose"],
                    save_reg = sr,
                )
                assert isinstance(linear.inner, LinearEXL3)
                linear.inner.swap_cpu()
            print_quantized_linear(config, linear, quant_args, proxy_err, f"  [{t.interval:4.2f} s]")


def quantize_linears_parallel(args, linears, config, strategy, idx, devices, device_ratios, capture_H, state):
    assert not args["image_dump"], "Parallel mode is incompatible with --image_dump"
    global curr_progress, max_progress

    allow_grouping = state is not None and not args["verbose"]
    groups = group_quant_linears(linears, strategy, capture_H if allow_grouping else None)

    # Split workload by group
    all_dev_groups = [[] for _ in devices]

    tot_numel = sum(linear.weights_numel() for linear in linears)
    if device_ratios is None:
        dev_numel = [tot_numel // len(devices) for _ in devices]
    else:
        tot_split = sum(device_ratios)
        dev_numel = [tot_numel * r // tot_split for _, r in zip(devices, device_ratios)]

    for group in groups:
        g_numel = sum(linear.weights_numel() for linear in group)
        fit = [d_numel - g_numel for d_numel in dev_numel]
        bestfit = max(range(len(fit)), key = lambda x: fit[x])
        dev_numel[bestfit] -= g_numel
        all_dev_groups[bestfit].append(group)

    with progress_lock:
        curr_progress = 0
        max_progress = len(linears)

    # Worker thread
    def work_thread(device_idx, dev_groups):
        global curr_progress

        t0 = time.time()
        work_numel = sum(l.weights_numel() for g in dev_groups for l in g)

        for group in dev_groups:
            if len(group) > 1:
                quant_args_list = [make_quant_args(args, idx, strategy[l.key], [device_idx]) for l in group]
                proxy_errs = convert_exl3_group(
                    group,
                    [capture_H[l.qmap] for l in group],
                    quant_args_list,
                )
                for linear, quant_args_local, proxy_err in zip(group, quant_args_list, proxy_errs):
                    assert isinstance(linear.inner, LinearEXL3)
                    linear.inner.swap_cpu()
                    print_quantized_linear(config, linear, quant_args_local, proxy_err)
                    with progress_lock:
                        curr_progress += 1
                continue

            linear = group[0]
            quant_args_local = make_quant_args(args, idx, strategy[linear.key], [device_idx])

            proxy_err = linear.convert_exl3(
                capture_H[linear.qmap] if state else linear.init_H_data(False),
                quant_args = quant_args_local,
                verbose = args["verbose"],
                save_reg = False,
                override_swap_device = device_idx
            )
            assert isinstance(linear.inner, LinearEXL3)
            linear.inner.swap_cpu()

            print_quantized_linear(config, linear, quant_args_local, proxy_err)
            with progress_lock:
                curr_progress += 1

        # The device is idle from here until the slowest thread finishes; its measured speed
        # steers the next module's split
        torch.cuda.synchronize(torch.device(device_idx))
        auto_split.report("quant_thread", device_idx, work_numel, time.time() - t0)

    # Launch
    threads = []
    for i, device_idx in enumerate(devices):
        if len(all_dev_groups[i]):
            t = threading.Thread(target = work_thread, args = (device_idx, all_dev_groups[i]))
            t.daemon = True
            threads.append(t)
    for t in threads:
        t.start()

    try:
        with ProgressBar(" -- Quantizing (parallel)", max_progress, transient = True) as progress:
            while any(t.is_alive() for t in threads):
                progress.update(curr_progress)
                time.sleep(0.1)
    except KeyboardInterrupt as e:
        # TODO: This is too hacky
        from signal import pthread_kill, SIGTSTP, SIGKILL
        for t in threads:
            pthread_kill(t.ident, SIGTSTP)
            pthread_kill(t.ident, SIGKILL)
        print("Aborted.")
        sys.exit()

    for t in threads:
        t.join(timeout = 0.1)


def check_bad_rows(bad_rows, num_rows, max_fraction = 0.10):
    """Abort the job when too much of the calibration set has been excluded as non-finite: past
    this point the remaining rows no longer represent the calibration distribution, and the
    breakage itself indicates something structurally wrong worth investigating (see the Hessian
    debug dumps in <work_dir>/debug/)."""
    if len(bad_rows) > max_fraction * num_rows:
        raise RuntimeError(
            f"{len(bad_rows)} of {num_rows} calibration rows have produced non-finite states "
            f"(> {max_fraction:.0%}), aborting job. Rows: {sorted(bad_rows)}"
        )


def calibration_row_shards(num_rows, devices, device_ratios):
    """
    Split calibration row indices into one contiguous shard per device, proportional to device_ratios if given.
    Row 0..num_ref_states-1 land in the first device's shard whenever it is non-empty, which keeps the reference
    states on the primary device.
    """
    w = device_ratios if device_ratios else [1] * len(devices)
    tot = sum(w)
    counts = [int(num_rows * r / tot) for r in w]   # ratios may be learned floats
    counts[0] += num_rows - sum(counts)
    shards, start = [], 0
    for c in counts:
        shards.append(range(start, start + c))
        start += c
    return shards


def load_parallel_calib_modules(replica_models, idx, devices, load_slice, source = None):
    """
    Load replicas of module idx onto secondary devices for parallel calibration passes. Returns the list of
    replica modules, or None if replication is unavailable (caller falls back to the serial path).
    """
    if not replica_models:
        return None
    replicas = []
    try:
        for rm, dev in zip(replica_models, devices[1:]):
            rep = rm.modules[idx]
            rep.load(torch.device(dev), load_slice = load_slice, **({"source": source} if source is not None else {}))
            replicas.append(rep)
        return replicas
    except Exception as e:
        print(f" !! Parallel calibration disabled for module {idx}: {e}")
        for rep in replicas:
            rep.unload()
        free_mem()
        return None


def run_row_workers(title, num_rows, workers, progress_count):
    """
    Run one worker thread per device with a shared progress bar, mirroring the interrupt handling of
    quantize_linears_parallel. Worker exceptions are collected and re-raised.
    """
    errors = []
    def guard(fn):
        def inner():
            try:
                fn()
            except Exception as e:
                errors.append(e)
        return inner
    threads = [threading.Thread(target = guard(w), daemon = True) for w in workers]
    for t in threads:
        t.start()
    try:
        with ProgressBar(title, num_rows) as progress:
            while any(t.is_alive() for t in threads):
                progress.update(progress_count[0])
                time.sleep(0.05)
            progress.update(progress_count[0])
    except KeyboardInterrupt:
        # TODO: This is too hacky (same as quantize_linears_parallel)
        from signal import pthread_kill, SIGTSTP, SIGKILL
        for t in threads:
            pthread_kill(t.ident, SIGTSTP)
            pthread_kill(t.ident, SIGKILL)
        print("Aborted.")
        sys.exit()
    for t in threads:
        t.join(timeout = 0.1)
    if errors:
        raise errors[0]


def capture_module_parallel(
    model,
    modules,
    devices,
    device_ratios,
    state,
    original_input_ids,
    get_preserve,
    put_preserve,
    slicing,
    current_slice,
    title,
    bad_rows,
):
    """
    Run the Hessian-capture forward pass with calibration rows split across devices, each device forwarding its
    shard through its own replica of the module. Per-device capture dicts are merged by summing H (the proxy is
    a plain sum over token batches), counts and inf/nan counters onto the primary device.
    """
    shards = calibration_row_shards(len(state), devices, device_ratios)
    captures = [{} for _ in modules]
    ref_map = {}
    lock = threading.Lock()
    progress_count = [0]

    def make_worker(t_idx):
        def worker():
            module = modules[t_idx]
            t0 = time.time()
            done_rows = 0
            for i in shards[t_idx]:
                if i in bad_rows:
                    with lock:
                        progress_count[0] += 1
                    continue
                params = {
                    "attn_mode": "flash_attn_nc",
                    "capture": captures[t_idx],
                    "activate_all_experts": model.calibration_all_experts,
                    "input_ids": original_input_ids[i],
                }
                if slicing:
                    params["q_mlp_slice"] = current_slice
                get_preserve(i, params)
                model.per_layer_quant_preamble(params)
                rs = module.prepare_for_device(state[i], params)
                rs = module.forward(rs, params)
                put_preserve(i, params)
                if i < num_ref_states:
                    if model.calibration_all_experts:
                        # Do not activate all experts for reference state, for error measurement
                        params = {
                            "attn_mode": "flash_attn_nc",
                            "input_ids": original_input_ids[i],
                        }
                        if slicing:
                            params["q_mlp_slice"] = current_slice
                        get_preserve(i, params)
                        model.per_layer_quant_preamble(params)
                        rs = module.prepare_for_device(state[i], params)
                        rs = module.forward(rs, params)
                        put_preserve(i, params)
                    if torch.isfinite(rs).all().item():
                        with lock:
                            ref_map[i] = rs.cpu()
                    else:
                        with lock:
                            bad_rows.add(i)
                        print(f" !! Non-finite reference state in calibration row {i}, excluding row")
                rs = None
                done_rows += 1
                with lock:
                    progress_count[0] += 1
            torch.cuda.synchronize(torch.device(devices[t_idx]))
            auto_split.report("calib", devices[t_idx], done_rows, time.time() - t0)
        return worker

    run_row_workers(title, len(state), [make_worker(i) for i in range(len(modules))], progress_count)

    # Merge secondary capture dicts into the primary one
    capture_H = captures[0]
    device = torch.device(devices[0])
    for cap in captures[1:]:
        for qmap, hd in cap.items():
            if qmap not in capture_H:
                hd["H"] = hd["H"].to(device)
                hd["inf_nan"] = hd["inf_nan"].to(device)
                hd["device"] = device
                capture_H[qmap] = hd
            else:
                m = capture_H[qmap]
                m["H"] += hd["H"].to(device)
                m["count"] += hd["count"]
                m["num_total"] += hd["num_total"]
                m["inf_nan"] += hd["inf_nan"].to(device)
        cap.clear()

    # Keyed by row index: rows excluded as non-finite leave gaps, so a list would misalign
    return capture_H, ref_map


def advance_state_parallel(
    model,
    modules,
    devices,
    device_ratios,
    state,
    original_input_ids,
    get_preserve,
    put_preserve,
    ref_states,
    have_linears,
    is_last_module,
    title,
    bad_rows,
):
    """
    Advance the calibration state through the (re-quantized) module with rows split across devices. Returns
    summed rfn/cos/sqnr over the reference rows and the number of rows measured. Rows in bad_rows are
    skipped; rows whose advanced state comes out non-finite are added to it.
    """
    shards = calibration_row_shards(len(state), devices, device_ratios)
    lock = threading.Lock()
    progress_count = [0]
    sums = [0.0, 0.0, 0.0, 0]

    def make_worker(t_idx):
        def worker():
            module = modules[t_idx]
            t0 = time.time()
            done_rows = 0
            for i in shards[t_idx]:
                if i in bad_rows:
                    with lock:
                        progress_count[0] += 1
                    continue
                params = {
                    "attn_mode": "flash_attn_nc",
                    "input_ids": original_input_ids[i],
                }
                state[i] = module.prepare_for_device(state[i], params)
                row_bad = False
                if i < num_ref_states or not is_last_module:
                    get_preserve(i, params)
                    model.per_layer_quant_preamble(params)
                    rs = module.forward(state[i], params)
                    if not torch.isfinite(rs).all().item():
                        row_bad = True
                        with lock:
                            bad_rows.add(i)
                        print(f" !! Non-finite hidden state in calibration row {i}, excluding row")
                    state[i] = rs.cpu()
                    put_preserve(i, params)
                ref = ref_states.get(i) if i < num_ref_states else None
                if ref is not None and have_linears and not row_bad:
                    ref = ref.to(state[i].device)
                    rfn, cos, sq = get_state_error(state[i], ref)
                    ref_states[i] = None
                    with lock:
                        sums[0] += rfn
                        sums[1] += cos
                        sums[2] += sq
                        sums[3] += 1
                done_rows += 1
                with lock:
                    progress_count[0] += 1
            torch.cuda.synchronize(torch.device(devices[t_idx]))
            auto_split.report("calib", devices[t_idx], done_rows, time.time() - t0)
        return worker

    run_row_workers(title, len(state), [make_worker(i) for i in range(len(modules))], progress_count)
    return sums[0], sums[1], sums[2], sums[3]


def image_dump(args, linears):
    if args["image_dump"]:
        for linear in linears:
            filename = f"images/{linear.key}.jpg"
            print(f" -- Saving image: {filename}")
            w = linear.inner.get_weight_tensor()
            assert w.dim() == 2
            save_tensor_image(w, os.path.join(args["work_dir"], filename))


# Per-layer host allocation churn (fp32 weight copies, H/L stashes, q_tensors, state rows)
# leaves freed memory stranded in glibc's per-thread arenas, where RSS ratchets up over a long
# job even though nothing is referenced. Explicitly return what can be returned after each
# module; no-op where glibc is unavailable
_libc = None
def malloc_trim():
    global _libc
    if _libc is False:
        return
    try:
        if _libc is None:
            import ctypes
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except Exception:
        _libc = False


def host_rss_str():
    """Resident set size of this process, for the per-module feedback line (empty string where
    /proc is unavailable)."""
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return f"  rss: {rss_pages * os.sysconf('SC_PAGE_SIZE') / 1024**3:.2f} GB"
    except Exception:
        return ""


def feedback_module(state, module, config, final_bpw, error, cos_error, sqnr_, module_time):
    if state:
        print(
            f" -- Quantized: {module.key:{config.stc.max_key_len() + 8}}" +
            (f"  bpw: {final_bpw:5.2f}" if final_bpw else f"  no_weights") +
            (f"  rfn: {error:.6f}" if module.num_slices == 1 else "        rfn: N/A     ") +
            f"  cos: {cos_error:.6f}"
            f"  sqnr: {sqnr_:.6f}"
            f"  [{module_time:.2f} s]" +
            host_rss_str(),
            flush = True
        )
    else:
        print(
            f" -- Quantized: {module.key:{config.stc.max_key_len() + 8}}" +
            (f"  bpw: {final_bpw:5.2f}" if final_bpw else f"  no_weights") +
            f"  [{module_time:.2f} s]" +
            host_rss_str(),
            flush = True
        )

timed_blocks = 0
eta_window = deque(maxlen = 8)

def feedback_eta(idx, model, module_time):
    global timed_blocks, eta_window
    if idx >= model.first_block_idx:
        timed_blocks += 1
        eta_window.append(module_time)
        remaining_blocks = max(0, len(model.modules) - idx - 1)
        if timed_blocks < 3:
            print(
                " -- Estimated remaining time: warming up "
                f"({timed_blocks}/3 timed blocks)"
            )
        else:
            avg_block_time = sum(eta_window) / len(eta_window)
            est_remaining = avg_block_time * remaining_blocks
            print(
                f" -- Estimated remaining time: {human_time(est_remaining)} "
                f"(avg over last {len(eta_window)} blocks)"
            )


def clear_temp_files(args):
    clear_files = args.get("clear_files")
    if clear_files:
        for p in clear_files:
            p.unlink()
        del args["clear_files"]


@torch.inference_mode()
def main(args, job_state):
    # TODO: Refactor this, split into functions
    global max_progress, curr_progress, timed_blocks

    torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

    torch.set_grad_enabled(False)

    devices = [int(d) for d in args["devices"].split(",")]
    device = torch.device(devices[0])
    if args.get("device_ratios"):
        device_ratios = [int(d) for d in args["device_ratios"].split(",")]
        assert len(devices) == len(device_ratios), "--devices and --device_ratios must be same length"
    else:
        device_ratios = None

    # Without explicit --device_ratios, split workloads start even and adapt: each split
    # workload reports per-device busy times, and subsequent modules divide work in proportion
    # to the measured speeds (see AutoSplit). Workload kinds are tracked separately since
    # per-unit costs differ between quantization and calibration forwards
    def eff_ratios(kind):
        if device_ratios is not None or len(devices) == 1:
            return device_ratios
        return auto_split.ratios(kind, devices)

    last_auto_split = [None]
    def report_auto_split():
        if device_ratios is not None or len(devices) == 1:
            return
        parts = []
        for kind, label in (("quant_thread", "quant"), ("quant_tiles", "tiles"), ("calib", "calib")):
            if not auto_split.has_measured(kind, devices):
                continue
            r = auto_split.ratios(kind, devices)
            if r:
                tot = sum(r)
                parts.append(label + " " + ":".join(f"{100 * x / tot:.0f}" for x in r))
        if parts:
            line = ", ".join(parts)
            if line != last_auto_split[0]:
                last_auto_split[0] = line
                print(f" -- Auto device split: {line}")

    last_checkpoint_time = time.time()

    # Get model
    config, model, mtp_model, tokenizer, use_reference_state = get_base_model(args)

    # Check caps
    can_resume_quant = model.caps.get("can_resume_quant", use_reference_state)
    if not can_resume_quant:
        print(" !! Warning, resuming an interrupted quant is not possible for this architecture.")
        print(" !! Checkpoints will not be saved because they are redundant.")
        assert job_state["next_module_idx"] == 0, \
            "Current implementation of this architecture does not support resuming an interrupted quant job."

    # Get initial state or resume state
    if use_reference_state:
        state = prepare_state(args, job_state, config, model, tokenizer)
        original_input_ids = (
            state.copy()
            if job_state["next_module_idx"] == 0 else
            [{} for _ in range(len(state))]
        )
        quant_preserves = [{} for _ in range(len(state))]
        # Rows whose hidden state (or unquantized reference) has gone non-finite are excluded
        # from all further capture, state advancement and error measurement. Persisted through
        # checkpoints; the job aborts if more than 10% of all rows break
        bad_rows = set(job_state.get("bad_rows") or [])
        if bad_rows:
            print(f" -- Resuming with {len(bad_rows)} excluded calibration rows")
    else:
        print(" -- Performing uncalibrated quantization")
        state = None
        bad_rows = set()

    def get_preserve(si, params_):
        params_.update(quant_preserves[si])
        params_["quant_preserve"] = quant_preserves[si]

    def put_preserve(si, params_):
        quant_preserves[si] = params_["quant_preserve"]

    # Get quantization strategy for model @bitrate
    print(" -- Deciding quantization strategy")
    hq = args["hq"]
    strategy, final_bpw = create_q_strategy(model, mtp_model, config, args["bits"], args["head_bits"], args["mtp_bits"], hq)
    args["final_bits"] = round(final_bpw, 2)
    print(" -- Quantization strategy, summary:")
    print(print_strategy(strategy))
    print(f" -- Final bitrate (excluding head): {final_bpw:4.2f}" + (" (--hq enabled)" if hq else ""))

    # With multiple devices, run the capture and state-advance forward passes with calibration rows split
    # across replicas of the current module, one per device. Calibration rows are independent and the H proxy
    # is a plain sum over token batches, so shards merge exactly. Models with a custom per-layer preamble
    # (module state prepared on the primary device) fall back to the serial path.
    parallel_calib = (
        state is not None and
        len(devices) > 1 and
        type(model).per_layer_quant_preamble is Model.per_layer_quant_preamble
    )
    replica_models = [Model.from_config(config) for _ in devices[1:]] if parallel_calib else []

    # Iterate over modules
    for idx, module in enumerate(model.modules):

        start_module_time = time.time()
        if idx == model.first_block_idx:
            timed_blocks = 0

        # If resuming, skip along to checkpoint index
        if idx < job_state["next_module_idx"]:
            continue

        if args["max_module"] is not None and idx > args["max_module"]:
            print(" !! --max_module reached, stopping job")
            sys.exit()

        # Collect output tensors
        q_tensors = {}
        capture_H = None
        ref_states = {}

        # Slice module if necessary
        slicing = module.num_slices > 1
        for current_slice in range(module.num_slices):

            # Load current module
            slice_str = f" (slice {current_slice + 1}/{module.num_slices})" if slicing else ""
            print(f" -- Loading unquantized module: {module.key}" + slice_str)
            module.load(
                torch.device("cpu") if module.caps.get("prefer_cpu") else device,
                load_slice = current_slice if slicing else None
            )
            for m in module:
                if m.used_alt_key and not slicing:
                    print(f"     - Cloned {m.key} from {m.alt_key}")
            module.config.stc.close()

            # Skip modules without quant targets
            qmaps = module.get_qmaps()
            if len(qmaps) > 0:

                # Capture calibration input states during forward pass. For block-sparse models, all expert layers
                # are activated to ensure all down projections capture at least some calibration data. When the
                # state is advanced later, only selected experts will be used.
                if state is not None:
                    capture_replicas = None
                    if parallel_calib and not module.caps.get("prefer_cpu"):
                        capture_replicas = load_parallel_calib_modules(
                            replica_models, idx, devices, current_slice if slicing else None)
                    if capture_replicas is not None:
                        capture_H, ref_states = capture_module_parallel(
                            model,
                            [module] + capture_replicas,
                            devices,
                            eff_ratios("calib"),
                            state,
                            original_input_ids,
                            get_preserve,
                            put_preserve,
                            slicing,
                            current_slice,
                            f" -- Capturing: {module.key}" + slice_str,
                            bad_rows,
                        )
                        for rep in capture_replicas:
                            rep.unload()
                        free_mem()
                    else:
                        with ProgressBar(f" -- Capturing: {module.key}" + slice_str, len(state)) as progress:
                            capture_H = {}
                            ref_states = {}
                            for i in range(len(state)):
                                progress.update(i)
                                if i in bad_rows:
                                    continue
                                params = {
                                    "attn_mode": "flash_attn_nc",
                                    "capture": capture_H,
                                    "activate_all_experts": model.calibration_all_experts,
                                    "input_ids": original_input_ids[i],
                                }
                                if slicing:
                                     params["q_mlp_slice"] = current_slice
                                get_preserve(i, params)
                                model.per_layer_quant_preamble(params)
                                rs = module.prepare_for_device(state[i], params)
                                rs = module.forward(rs, params)
                                put_preserve(i, params)
                                if i < num_ref_states:
                                    if model.calibration_all_experts:
                                        # Do not activate all experts for reference state, for error measurement
                                        params = {
                                            "attn_mode": "flash_attn_nc",
                                            "input_ids": original_input_ids[i],
                                        }
                                        if slicing:
                                            params["q_mlp_slice"] = current_slice
                                        get_preserve(i, params)
                                        model.per_layer_quant_preamble(params)
                                        rs = module.prepare_for_device(state[i], params)
                                        rs = module.forward(rs, params)
                                        put_preserve(i, params)
                                    if torch.isfinite(rs).all().item():
                                        ref_states[i] = rs.cpu()
                                    else:
                                        bad_rows.add(i)
                                        print(f" !! Non-finite reference state in calibration row {i}, excluding row")
                                rs = None
                    print(f" -- Captured: {module.key}" + slice_str, flush = True)

                    # More feedback
                    if len(capture_H):
                        hfb_keys = [re.sub(r'\d+', '*', k) for k in capture_H.keys()]
                        hfb_keys = sorted(list(set(hfb_keys)))
                        print(f" -- Hessians: " + ", ".join(hfb_keys))
                    else:
                        print(f" !! Hessians: None")

                    # Check for infs or NaNs in H
                    for k, v in capture_H.items():
                        infs, nans = v["inf_nan"][0].item(), v["inf_nan"][1].item()
                        if infs or nans:
                            numel = v["num_total"]
                            print(f" !! Warning: {k} state has {infs:,} inf values and {nans:,} NaN values (out of {numel:,})")

                    # Swap captured H to system RAM
                    for k, v in capture_H.items():
                        v["H_swap_device"] = v["H"].device
                        v["H"] = v["H"].cpu()

            # Get submodules to quantize
            linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]
            for linear in linears:
                assert linear.key in strategy, \
                    f" ## Logic error, no quantization strategy for module {linear.key}"
            assert all(isinstance(linear.inner, LinearFP16) for linear in linears)

            # Write images
            image_dump(args, linears)

            # Move original tensors to system RAM (load to GPU one by one when quantizing)
            for linear in linears:
                linear.inner.swap_cpu()

            # Quantize
            if (
                args["parallel_mode"] and
                len(linears) >= len(devices) and
                all(b <= 8 for _, b in strategy.items())
            ):
                quantize_linears_parallel(args, linears, config, strategy, idx, devices, eff_ratios("quant_thread"), capture_H, state)
            else:
                quantize_linears_single(args, linears, config, strategy, idx, devices, eff_ratios("quant_tiles"), capture_H, state)

            # Collect converted module tensors
            for m in module:
                q_tensors.update(m.get_tensors())

            # Unload module
            if not module.caps.get("retain_during_quant"):
                module.unload()
            capture_H = None
            config.stc.close()

        # Save layer tensors to working directory
        clear_temp_files(args)
        save_tensor(q_tensors, f"qtensors/{module.key}.safetensors", args)

        # Output final bpw for layer
        num_bytes = dsize(q_tensors)
        num_bits = num_bytes * 8
        final_bpw = num_bits / module.weights_numel() if module.weights_numel() else None

        # Reload module from memory
        config.stc.set_new_tensors(q_tensors)
        module.load(
            torch.device("cpu") if module.caps.get("prefer_cpu") else device,
            source = q_tensors
        )
        advance_replicas = None
        if state is not None and parallel_calib and not module.caps.get("prefer_cpu"):
            advance_replicas = load_parallel_calib_modules(
                replica_models, idx, devices, None, source = q_tensors)
        config.stc.set_new_tensors(None)
        del q_tensors

        # Advance state
        error = 0
        cos_error = 0
        sqnr_ = 0
        if state is not None:
            if advance_replicas is not None:
                error, cos_error, sqnr_, num_measured = advance_state_parallel(
                    model,
                    [module] + advance_replicas,
                    devices,
                    eff_ratios("calib"),
                    state,
                    original_input_ids,
                    get_preserve,
                    put_preserve,
                    ref_states,
                    len(linears) > 0,
                    idx >= len(model.modules) - 1,
                    f" -- Forward pass: {module.key}",
                    bad_rows,
                )
                for rep in advance_replicas:
                    rep.unload()
                free_mem()
                n = max(num_measured, 1)
                error /= n
                cos_error /= n
                sqnr_ /= n
                check_bad_rows(bad_rows, len(state))
            else:
                num_measured = 0
                with ProgressBar(f" -- Forward pass: {module.key}", len(state)) as progress:
                    for i in range(len(state)):
                        progress.update(i)
                        if i in bad_rows:
                            continue
                        params = {
                            "attn_mode": "flash_attn_nc",
                            "input_ids": original_input_ids[i],
                        }
                        state[i] = module.prepare_for_device(state[i], params)
                        if i < num_ref_states or idx < len(model.modules) - 1:
                            get_preserve(i, params)
                            model.per_layer_quant_preamble(params)
                            rs = module.forward(state[i], params)
                            if not torch.isfinite(rs).all().item():
                                bad_rows.add(i)
                                print(f" !! Non-finite hidden state in calibration row {i}, excluding row")
                            state[i] = rs.cpu()
                            put_preserve(i, params)
                        ref = ref_states.get(i) if i < num_ref_states else None
                        if ref is not None and len(linears) and i not in bad_rows:
                            ref = ref.to(state[i].device)
                            rfn, cos, sq = get_state_error(state[i], ref)
                            error += rfn
                            cos_error += cos
                            sqnr_ += sq
                            num_measured += 1
                            ref_states[i] = None
                n = max(num_measured, 1)
                error /= n
                cos_error /= n
                sqnr_ /= n
                check_bad_rows(bad_rows, len(state))

        # Feedback after module. Trim first so the reported RSS reflects what the job actually
        # retains, not what the allocator happens to be holding
        malloc_trim()
        module_time = time.time() - start_module_time
        feedback_module(state, module, config, final_bpw, error, cos_error, sqnr_, module_time)
        report_auto_split()
        feedback_eta(idx, model, module_time)

        # Unload current module
        if not module.caps.get("retain_during_quant"):
            module.unload()

        # Checkpoint
        job_state["next_module_idx"] = idx + 1
        if can_resume_quant and \
            time.time() > last_checkpoint_time + args["checkpoint_interval"] and \
            (args.get("last_checkpoint_index", -1) < 0 or idx <= args["last_checkpoint_index"]):
            print(f" -- Saving checkpoint")
            ckpt_dir = os.path.join(args["work_dir"], "ckpt")
            ckpt_dir_old = os.path.join(args["work_dir"], "ckpt_old")
            ckpt_dir_new = os.path.join(args["work_dir"], "ckpt_new")
            os.makedirs(ckpt_dir_new, exist_ok = True)
            job_state["bad_rows"] = sorted(bad_rows)
            save_dict("ckpt_new/job.json", job_state, args)
            save_tensor(state, "ckpt_new/state.safetensors", args)
            if os.path.exists(ckpt_dir_old):
                shutil.rmtree(ckpt_dir_old)
            os.rename(ckpt_dir, ckpt_dir_old)
            os.rename(ckpt_dir_new, ckpt_dir)
            last_checkpoint_time = time.time()

    # Quantize additional modules
    if mtp_model:
        print(" -- Quantizing MTP tensors")

        for idx, module in enumerate(mtp_model.modules):
            assert module.num_slices <= 1
            start_module_time = time.time()

            q_tensors = {}
            print(f" -- Loading unquantized module: {module.key}")
            module.load(torch.device("cpu") if module.caps.get("prefer_cpu") else device)
            for m in module:
                if m.used_alt_key:
                    print(f"     - Cloned {m.key} from {m.alt_key}")
            module.config.stc.close()

            linears = [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]
            for linear in linears:
                assert linear.key in strategy, \
                    f" ## Logic error, no quantization strategy for module {linear.key}"
            assert all(isinstance(linear.inner, LinearFP16) for linear in linears)

            # Write images
            image_dump(args, linears)

            # Move original tensors to system RAM (load to GPU one by one when quantizing)
            for linear in linears:
                linear.inner.swap_cpu()

            # Quantize
            if (
                args["parallel_mode"] and
                len(linears) >= len(devices) and
                all(b <= 8 for _, b in strategy.items())
            ):
                quantize_linears_parallel(args, linears, config, strategy, idx, devices, eff_ratios("quant_thread"), None, None)
            else:
                quantize_linears_single(args, linears, config, strategy, idx, devices, eff_ratios("quant_tiles"), None, None)

            # Collect converted module tensors
            for m in module:
                q_tensors.update(m.get_tensors())

            # Unload module
            module.unload()
            config.stc.close()

            # Save layer tensors to working directory
            clear_temp_files(args)
            save_tensor(q_tensors, f"qtensors/{module.key}.safetensors", args)

            # Output final bpw for layer
            num_bytes = dsize(q_tensors)
            num_bits = num_bytes * 8
            final_bpw = num_bits / module.weights_numel() if module.weights_numel() else None

            # Feedback after module
            malloc_trim()
            module_time = time.time() - start_module_time
            feedback_module(state, module, config, final_bpw, 0, 0, 0, module_time)

            # Unload current module
            module.unload()
            del q_tensors

    # Compile model
    compile_model(args, model, config, tokenizer, mtp_model)

    # All done
    print(" -- All done")
