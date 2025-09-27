import argparse
import torch
from .. import Config, Model, Tokenizer
from ..util.progress import ProgressBar
from .calibration_data import get_default_calibration
import json
import torch.nn.functional as F
import math
from collections import defaultdict

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
parser.add_argument("-i", "--in_dir", nargs = "+", type = str, default = None, help = "Input (model) directories")
parser.add_argument("-r", "--ref_dir", type = str, default = None, help = "Reference unquantized model")
parser.add_argument("-l", "--level", type = int, default = 2, help = "Optimization level, 0-3, default: 2 (recommended)")
parser.add_argument("-o", "--out_file", type = str, default = None, help = "Output file")
parser.add_argument("-cr", "--cal_rows", type = int, default = 10, help = "Calibration data size, rows, default: 8")
parser.add_argument("-cc", "--cal_cols", type = int, default = 1024, help = "Calibration data size, columns, default: 1024")
# parser.add_argument("-d", "--devices", type = str, default = "0", help = "List of devices to use, e.g. --devices 0,1,2")
parser.add_argument("-d", "--device", type = int, default = 0, help = "Device index")
parser.add_argument("-ms", "--max_sys", type = float, default = 8, help = "Max system memory for state data, in GB")

def prepare(args) -> (dict, dict, bool, str):
    if not args.in_dir:
        return None, None, False, "Please specify --in_dir"
    if len(args.in_dir) < 2:
        return None, None, False, "Please specify at least two quantized input models"
    if not args.ref_dir:
        return None, None, False, "Please specify --ref_dir"
    if not args.out_file:
        return None, None, False, "No output file specified"

    in_args = {
        "ref_dir": args.ref_dir,
        "in_dir": args.in_dir,
        "level": args.level,
        "out_file": args.out_file,
        "cal_rows": args.cal_rows,
        "cal_cols": args.cal_cols,
        "device": args.device,
        "max_sys": args.max_sys
    }

    job_state = {}

    warn_experimental = False
    print(f"    Input directories:")
    for d in in_args["in_dir"]:
        print(f"    - {d}")
    print(f"    Output file: {in_args['out_file']}")
    print(f"    Calibration size: {in_args['cal_rows']} rows, {in_args['cal_cols']} columns")

    if warn_experimental:
        print(
            f" !! {col_red}WARNING, experimental options are selected. The quantized model may not work in future "
            f"versions of ExLlamaV3 {col_default}"
        )

    return in_args, job_state, True, None


def prepare_state(args, job_state, config, model, tokenizer):
    print(f" -- Preparing input state")
    state = get_default_calibration(args, tokenizer)
    return state[:args["cal_rows"]]


def kldiv(s, ref):
    r = ref.view(-1, ref.shape[-1])
    s = s.view(-1, s.shape[-1])
    bsz, _ = s.shape
    max_batch = 512
    kld = 0
    for i in range(0, bsz, max_batch):
        ref_probs = torch.softmax(r[i : i+max_batch, ...].to(s.device), dim = -1, dtype = torch.float)
        s_probs = torch.softmax(s[i : i+max_batch, ...], dim = -1, dtype = torch.float)
        kld += F.kl_div(torch.log(s_probs + 1e-10), ref_probs, reduction = "sum").item()
    return kld / bsz


def lcp(seqs):
    if not seqs: return []
    m = min(len(s) for s in seqs)
    i = 0
    while i < m and all(s[i] == seqs[0][i] for s in seqs):
        i += 1
    return seqs[0][:i]


def compact_alt(strings):
    seqs = [s.split(".") for s in strings]

    def render(seqs):
        if not seqs: return ""
        prefix = lcp(seqs)
        tails = [s[len(prefix):] for s in seqs]
        if all(not t for t in tails):
            return " ".join(prefix)

        groups = defaultdict(list)
        for t in tails:
            head, rest = (t[0], t[1:]) if t else ("", [])
            groups[head].append(rest)
        parts = []
        for head, rest_lists in groups.items():
            if head == "":
                parts.append("")
            else:
                sub = render(rest_lists)
                parts.append(f"{head} {sub}" if sub else head)

        inside = ", ".join(p for p in parts if p != "")
        out = " ".join(prefix)
        if len(parts) == 1 and inside:
            return f"{out} {inside}".strip()
        return (f"{out} [{inside}]" if out else f"[{inside}]").strip()

    return render(seqs)


@torch.inference_mode()
def main(args, job_state):

    torch.set_grad_enabled(False)
    device = torch.device(args["device"])

    # Get model
    dir_ref = args["ref_dir"]
    dir_q = args["in_dir"]
    config_ref = Config.from_directory(dir_ref)
    config_q = [Config.from_directory(d) for d in dir_q]
    print(f" -- Loaded model config")
    print(f"    Architecture: {config_ref.architecture}")
    model_ref = Model.from_config(config_ref)
    model_q = [Model.from_config(c) for c in config_q]
    print(f" -- Created model instances:")
    r_layout = model_ref.get_layout_tree(4)
    print(r_layout)
    tokenizer = Tokenizer.from_config(config_ref)
    print(f" -- Loaded tokenizer")
    print(f"    Vocab size: {tokenizer.actual_vocab_size}")
    num_q = len(model_q)
    num_cand = num_q - 1

    # Sanity checks
    storage_info = [mq.get_storage_info() for mq in model_q]
    if any(storage_info[i + 1][0] <= storage_info[i][0] for i in range(num_cand)):
        print(f" !! {col_red}Warning, input quantized models should be in order of increasing bitrate{col_default}")
    if any(cq.hidden_size != config_ref.hidden_size for cq in config_q) \
        or any(cq.arch_string != config_ref.arch_string for cq in config_q) \
        or any(mq.get_layout_tree(4) != r_layout for mq in model_q):
        print(f" !! {col_red}Warning, input models do not match reference model{col_default}")

    # Optimizer targets
    targets_per_layer = []
    total_targets = 0
    for module in model_ref.modules[:-1]:
        r_targets = module.optimizer_targets()
        s_targets = []
        def flatten(node, maxdepth, depth = 0):
            nonlocal s_targets
            l = []
            for n in node:
                if isinstance(n, str):
                    l.append(n)
                else:
                    if depth >= maxdepth:
                        l += flatten(n, maxdepth, depth + 1)
                    else:
                        l.append(flatten(n, maxdepth, depth + 1))
            if depth == maxdepth:
                s_targets.append(l)
            return l
        flatten(r_targets, args["level"])
        targets_per_layer.append(s_targets)
        total_targets += len(s_targets)
    targets_per_layer.append([])

    # Get initial state
    init_states_ref = prepare_state(args, job_state, config_ref, model_ref, tokenizer)
    init_states_ref = torch.cat(init_states_ref, dim = 0)

    # Chunk to limit system RAM
    print(f" -- Total optimized params: {total_targets}")
    state_size = config_ref.hidden_size * init_states_ref.numel() * 4
    r_sys = int(args["max_sys"] * 1024**3) - state_size * 2
    t_sys_cand = total_targets * state_size
    num_chunks = int(math.ceil(t_sys_cand / r_sys))
    tpl = targets_per_layer
    lpc = int(math.ceil(len(tpl) / num_chunks))
    tpl_chunks = []
    for i in range(0, len(tpl), lpc):
        tpl_c = [(t if i <= j < i + lpc else []) for j, t in enumerate(tpl)]
        if any(t for t in tpl_c):
            tpl_chunks.append(tpl_c)

    all_cand_groups = []
    all_cand_costs = [[] for _ in range(num_cand)]
    all_cand_kld = [[] for _ in range(num_cand)]

    # Chunk loop
    for chunk_idx, targets_per_layer in enumerate(tpl_chunks):
        print(f" -- Starting pass {chunk_idx + 1}/{len(tpl_chunks)}...")

        # States/intermediates
        states_ref = init_states_ref.clone()
        states_q = init_states_ref.clone()
        cand_states = [[] for _ in range(num_cand)]
        cand_groups = []
        cand_costs = [[] for _ in range(num_cand)]
        cand_kld = [[] for _ in range(num_cand)]
        base_kld = 0

        # Streaming forward pass
        num_modules = len(model_ref.modules)
        pb_total = 0
        pb_prog = 0
        l = 2
        for idx in range(num_modules):
            for t in targets_per_layer[idx]:
                if len(t) > 0: l += num_cand
            pb_total += l

        with ProgressBar(f"Measuring ({chunk_idx + 1}/{len(tpl_chunks)})", pb_total) as pb:
            for idx in range(num_modules):
                last_fwd = idx == len(model_ref.modules) - 1

                # Load next reference module
                module = model_ref.modules[idx]
                print(f" -- Loading module (ref): {col_green}{module.key}{col_default}")
                config_ref.stc.begin_deferred_load()
                module.load(device if not module.caps.get("prefer_cpu") else "cpu")
                config_ref.stc.end_deferred_load()
                config_ref.stc.close()

                # Reference forward pass
                params = {"activate_all_experts": True, "attn_mode": "flash_attn_nc"}
                s = module.prepare_for_device(states_ref, params)
                s = module.forward(s, params)
                new_states_ref = s.cpu()
                del s
                torch.cuda.empty_cache()
                pb.update(pb_prog)
                pb_prog += 1

                # Unload reference
                module.unload()
                del module
                torch.cuda.empty_cache()

                # Load next quantized modules
                modules = []
                for q in range(num_q):
                    if q > 0 and (not targets_per_layer[idx] or not targets_per_layer[idx][0]):
                        modules.append(None)
                        continue
                    module = model_q[q].modules[idx]
                    print(f" -- Loading module ({col_purple}{q}{col_default}): {col_green}{module.key}{col_default}")
                    config = config_q[q]
                    config.stc.begin_deferred_load()
                    module.load(device if not module.caps.get("prefer_cpu") else "cpu")
                    config.stc.end_deferred_load()
                    config.stc.close()
                    modules.append(module)
                    del module

                # Advance base state
                params = {"activate_all_experts": True, "attn_mode": "flash_attn_nc"}
                s = modules[0].prepare_for_device(states_q, params)
                s = modules[0].forward(s, params)
                if last_fwd:
                    base_kld = kldiv(s, new_states_ref)
                    new_states_q = None
                else:
                    new_states_q = s.cpu()
                del s
                torch.cuda.empty_cache()
                pb.update(pb_prog)
                pb_prog += 1

                # Iterate over options
                for k in range(num_cand):
                    last_k = k == num_cand - 1

                    # Propagate candidates
                    for i in range(len(cand_states[k])):
                        states_c = cand_states[k][i]
                        params = {"activate_all_experts": True, "attn_mode": "flash_attn_nc"}
                        s = modules[0].prepare_for_device(states_c, params)
                        s = modules[0].forward(s, params)
                        if last_fwd:
                            cand_kld[k][i] += kldiv(s, new_states_ref) - base_kld
                            cand_states[k][i] = None
                        else:
                            cand_states[k][i] = s.cpu()
                        del s
                        torch.cuda.empty_cache()
                        pb.update(pb_prog)
                        pb_prog += 1

                    # New candidate states
                    targets = targets_per_layer[idx]
                    for t in targets:
                        if len(t) == 0:
                            continue
                        group_idx = len(cand_groups)
                        tt = compact_alt((b[:-5] if b.endswith("_proj") else b) for b in t)
                        if len(tt) > 150:
                            tt = tt[:97] + "..."
                        print(
                            f" -- Group {col_blue}{group_idx} {col_purple}{k + 1}{col_default}: "
                            f"{col_yellow}{tt}{col_default}"
                        )
                        cand_kld[k].append(0)
                        params = {
                            "activate_all_experts": True,
                            "attn_mode": "flash_attn_nc",
                            "ovr": {key : model_q[k + 1].find_module(key) for key in t}
                        }
                        s = modules[0].prepare_for_device(states_q, params)
                        s = modules[0].forward(s, params)
                        if last_fwd:
                            cand_kld[k][i] = kldiv(s, new_states_ref) - base_kld
                        else:
                            cand_states[k].append(s.cpu())
                        del s
                        torch.cuda.empty_cache()
                        pb.update(pb_prog)
                        pb_prog += 1
                        cand_costs[k].append(
                            sum(model_q[k + 1].find_module(tt).storage_size() for tt in t) * 8 -
                            sum(model_q[0].find_module(tt).storage_size() for tt in t) * 8
                        )
                        if last_k:
                            cand_groups.append(t)

                # Unload quantized modules
                for module in modules:
                    if module is not None:
                        module.unload()
                del modules
                torch.cuda.empty_cache()

                # Store ref states
                states_ref = new_states_ref
                del new_states_ref
                states_q = new_states_q
                del new_states_q

            pb.update(pb_prog)

        # Collect results from chunk
        all_cand_groups += cand_groups
        all_cand_costs = [a + b for a, b in zip(all_cand_costs, cand_costs)]
        all_cand_kld = [a + b for a, b in zip(all_cand_kld, cand_kld)]

    cand_groups = all_cand_groups
    cand_costs = all_cand_costs
    cand_kld = all_cand_kld

    # Pretty feedback
    all_groups = []
    for c in cand_groups:
        all_groups += c
    prefix = lcp(all_groups)
    print(" -- Results")
    print(f"    idx      | parts                                                                |              d. bits |  d. KL-div")
    print(f"    ---------|----------------------------------------------------------------------|----------------------|-----------")
    print(
        f"       {col_blue}-{col_default}     |                                                                      |"
        f"                      | {col_green}{base_kld:10.5f}{col_default}"
    )
    for i, g in enumerate(cand_groups):
        sg = []
        gstr = compact_alt([sg[len(prefix):].replace("_proj", "") for sg in g])
        if len(gstr) > 68: gstr = gstr[:65] + "..."
        for j in range(num_cand):
            print(
                f"     {col_blue}{i:3} {col_purple}{j + 1:3}{col_default} | "
                f"{col_yellow}{gstr:69}{col_default}| "
                f"{cand_costs[j][i]:20,}"
                f"{col_default} | {col_green}{cand_kld[j][i]:+10.5f}{col_default}"
            )

    # Compile and write results
    results = {
        "base": {
            "dir": dir_q[0],
            "bpw": storage_info[0][0],
        },
        "alts": [
            {
                "dir": dir_q[c + 1],
                "bpw": storage_info[c + 1][0],
            }
            for c in range(num_cand)
        ],
        "groups": [
            {
                "idx": idx,
                "layers": layers,
                "candidates": [
                    {
                        "dkld": cand_kld[j][idx],
                        "dbits": cand_costs[j][idx],
                    }
                    for j in range(num_cand)
                ],
            }
            for idx, layers in enumerate(cand_groups)
        ],
        "base_kld": base_kld,
        "arch_string": config_ref.arch_string
    }
    filename = args["out_file"]
    print(f" -- Writing {filename}")
    with open(filename, "w", encoding = "utf8") as f:
        f.write(json.dumps(results, indent = 4))

    # Done
    print(" -- Done")