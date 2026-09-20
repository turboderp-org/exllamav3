"""
Kronecker-factored Hessians for two-sided LDLQ, after YAQA (Tseng, Sun, De Sa 2025, arXiv 2505.22988), sketch A.

Runs the unquantized model in Transformers over a packed calibration file and writes, for each linear layer, the two
factors of H ~ Hout (x) Hin, an approximation to the Hessian of the full-model KL divergence w.r.t. that layer's weight.
convert.py ingests the result with -hess/--hessians.

The Hessian of the KL at the unquantized weights is the Fisher information, estimated here from gradients of the
cross-entropy against targets sampled from the model's own output distribution. With tokens treated as independent,
each token t contributes (d_t d_t^T) (x) (x_t x_t^T) for the layer y = x W^T, where x_t is the layer's input and
d_t = dL/dy_t. The best Kronecker fit to that sum is found by power iteration, one side per pass over the data:

    forward pass:  Hin  = sum_t x_t x_t^T                                 (the usual GPTQ/LDLQ Hessian)
    then:          Hout = sum_t (x_t^T Hin x_t)  d_t d_t^T / |Hin|_F^2
    then:          Hin  = sum_t (d_t^T Hout d_t) x_t x_t^T / |Hout|_F^2   and so on, alternating

What to collect is set by -s/--sides:

    out (default)  Hout only, from one forward and one backward pass. On Qwen3.5-2B the whole benefit came from the
                   output side: the online (quantized-stream) Hin of a regular conversion together with this Hout beat
                   the fully iterated pair, and convert.py keeps its calibration pass when the files have no hin.
    both           Both factors, alternating for -p/--passes passes (default 6, ending on Hout). Conversion then runs
                   without calibration data.

With --unweighted the sample weights are dropped (Hout = sum_t d_t d_t^T), which needs no Hin at all and only the one
backward pass. -n/--samples draws several targets per row and runs a backward pass for each off the same forward
pass; Hout is a noisy estimate (see below) and an extra sample costs about a third of an extra row.

Output is one safetensors file per tensor, named by its key in the model checkpoint:

    {out_dir}/{key}.safetensors:  "hin": in x in,  "hout": out x out,  fp32

Both are symmetric and stored as the packed upper triangle, row-major: a 1-D tensor of n (n + 1) / 2 values (convert.py
also accepts full square matrices). The output head gets hin only, in "both" mode, since its hout would be vocab x
vocab. Only nn.Linear modules are covered; fused expert tensors (MoE) are not.

Memory: the factors are accumulated on the GPU, in_features^2 + out_features^2 fp32 values per tensor (half that
on disk). For large models that is far more than fits alongside the weights, so collect a range of layers at a time
with -l/--layers; ranges write disjoint files into the same output directory. No gradients are computed below the
first layer in the range, and the forward-only pass stops after the last, so later ranges run somewhat faster. With several devices
the model is placed on them in the order given, leaving -rv/--reserve_vram GB free on each, which has to cover the
activations kept for the backward pass (roughly as much as the weights of the layers on that device, at 2048 tokens),
the factors (on the devices given by -hd, otherwise with each layer) and ~10 GB for logits on the last device.
--tf32 speeds up the accumulation (about 40% off the whole run on a 2B model, 20% on a 27B); the inputs are BF16 and
the running sums stay FP32. A separate -hd device lets the accumulation overlap with the backward pass, but without P2P
the copies cost more than that gains back (measured on a 27B), so it's mainly a way to make room.

Sampling noise: Hout depends on the sampled targets and is dominated by relatively few high-gradient tokens. On
Qwen3.5-2B two seeds differ by ~20% (Frobenius) at 250 rows, ~40% at 64, falling as 1/sqrt(rows), and the same applies
between GPU models, since the sampling kernel isn't reproducible across them. At 250 rows the quantized model doesn't
seem to care: weighted, unweighted and 4-sample variants all land within run-to-run variation of each other.

    python util/yaqa_hessians.py -m /models/hf/model -c cal_trace.safetensors -o /scratch/hess
    python util/yaqa_hessians.py -m /models/hf/model -c cal_trace.safetensors -o /scratch/hess -d 0,1,2,3 -hd 0 -rv 73,16,12,14 -l 0:16 --tf32 --unweighted
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import re
import time
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Quantized linear layers inside a decoder block, by the last component of the module name. Small projections that
# EXL3 leaves unquantized (GDN a/b, routers etc.) are not matched
default_pattern = (
    r"(q_proj|k_proj|v_proj|o_proj|qkv_proj|gate_proj|up_proj|down_proj|gate_up_proj|"
    r"in_proj_qkv|in_proj_z|in_proj_qkvz|out_proj|"
    r"q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)$"
)
skip_names = ("visual", "vision", "mtp", "audio")


class StopForward(Exception):
    pass


def load_rows(args):
    """
    Read calibration rows from a packed token file: "input_ids" (rows, cols) and optionally "lengths" (rows,), as
    written by sc_trace.py, in which case each row is one example from position 0, right-padded. Rows are cropped to
    their own length (same convention as convert.py) so the model never sees the padding.
    """
    data = load_file(args.cal_data)
    packed = data["input_ids"]
    lengths = data.get("lengths")
    if args.skip_rows:
        packed = packed[args.skip_rows:]
        lengths = lengths[args.skip_rows:] if lengths is not None else None
    rows = packed.shape[0] if args.rows is None else args.rows
    if packed.shape[0] < rows:
        print(f" !! Calibration file contains {packed.shape[0]} rows, less than requested ({rows})")
        rows = packed.shape[0]
    cols = packed.shape[1] if args.cols is None else min(args.cols, packed.shape[1])
    out = []
    for i in range(rows):
        n = cols if lengths is None else min(cols, int(lengths[i]))
        if n < 1:
            raise ValueError(f"Calibration file {args.cal_data}: row {i} is empty")
        out.append(packed[i : i + 1, :n].to(torch.long).contiguous())
    num_tokens = sum(r.shape[-1] for r in out)
    print(
        f" -- Calibration: {len(out)} rows, {num_tokens} tokens" +
        (f", mixed length {min(r.shape[-1] for r in out)}-{max(r.shape[-1] for r in out)}" if lengths is not None else "")
    )
    return out


def load_model(args, devices):
    """
    Load the unquantized model in BF16. With more than one device the layers are distributed by accelerate, leaving
    reserve_vram GB per device for activations and the Hessian factors. Devices are filled in the order given.
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    kwargs = {"dtype": torch.bfloat16}
    if len(devices) == 1:
        kwargs["device_map"] = {"": devices[0]}
    else:
        reserve = args.reserve_vram or [8.0]
        reserve += [reserve[-1]] * (len(devices) - len(reserve))
        kwargs["device_map"] = "sequential"  # fill devices in order; "auto" balances and ignores most of a large device
        kwargs["max_memory"] = {
            d: max(int(torch.cuda.mem_get_info(d)[0] - r * 1024**3), 0)
            for d, r in zip(devices, reserve)
        }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **kwargs)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **kwargs)
    offloaded = sorted(set(str(v) for v in getattr(model, "hf_device_map", {}).values() if v in ("cpu", "disk")))
    if offloaded:
        print(f" !! Model doesn't fit with the given --reserve_vram, part of it is offloaded to {'/'.join(offloaded)} (slow)")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def checkpoint_keys(model_dir):
    """
    Weight keys in the source checkpoint. Module names in the loaded model don't necessarily match (a multimodal
    checkpoint loaded as a causal LM drops a prefix), and the output files must be named like the checkpoint.
    """
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return list(json.load(f)["weight_map"].keys())
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        with safe_open(single, "pt") as f:
            return list(f.keys())
    return []


def find_targets(args, model):
    """
    Collect the modules to compute Hessians for.

    returns:
        targets: dict of module name -> (output key, module), in model order
        first, last: decoder layer modules bounding the selected range (None when no layers are selected)
    """
    pattern = re.compile(args.pattern)
    layer_re = re.compile(r"(^|\.)layers\.(\d+)(\.|$)")

    def layer_idx(name):
        m = layer_re.search(name)
        return int(m.group(2)) if m and not any(s in name for s in skip_names) else None

    blocks = {}
    for name, mod in model.named_modules():
        if re.search(r"(^|\.)layers\.\d+$", name) and layer_idx(name) is not None:
            blocks[layer_idx(name)] = mod
    num_layers = max(blocks) + 1
    begin, end = args.layers if args.layers else (0, num_layers)
    end = min(end, num_layers)
    head = args.head if args.head is not None else (end == num_layers)

    # Map the tail of each module name (from "layers." on) to its checkpoint key
    ckpt = {}
    for k in checkpoint_keys(args.model_dir):
        if k.endswith(".weight") and layer_idx(k) is not None:
            ckpt[k[k.index("layers."):-len(".weight")]] = k[:-len(".weight")]

    targets = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        idx = layer_idx(name)
        if idx is not None and begin <= idx < end and pattern.search(name):
            tail = name[name.index("layers."):]
            key = args.key_prefix + tail if args.key_prefix is not None else ckpt.get(tail, name)
            targets[name] = (key, mod)
        elif head and name.split(".")[-1] == "lm_head":
            targets[name] = ("lm_head", mod)

    in_range = [i for i in sorted(blocks) if begin <= i < end]
    first = blocks[in_range[0]] if in_range else None
    last = blocks[in_range[-1]] if in_range else None
    print(f" -- Layers {begin}:{end} of {num_layers}" + (", head" if head else "") + f", {len(targets)} tensors")
    return targets, first, last, head


def pack_sym(h):
    """
    Upper triangle of a symmetric matrix as a 1-D tensor, row-major
    """
    n = h.shape[0]
    mask = torch.ones((n, n), dtype = torch.bool, device = h.device).triu_()
    return h[mask].contiguous()


class Collector:
    """
    Accumulates the factors for one linear layer.

    Each pass is one of:
        "fwd":  Hin = sum x x^T, forward only
        "out":  Hout = sum w d d^T, w = x^T Hin x / |Hin|^2 (or 1)
        "in":   Hin = sum w x x^T, w = d^T Hout d / |Hout|^2

    Modules fed the same tensor (q/k/v, gate/up etc.) have the same Hin after the forward pass. Only the first of them
    accumulates it, as the leader, and the rest end up referring to the same tensor. That holds until a sibling does
    its own weighted "in" update, which gets its own storage. Hin is allocated when first needed for that reason.
    """

    def __init__(self, name, key, module, device, want_in, want_out):
        self.name = name
        self.key = key
        self.device = device or module.weight.device
        self.in_features = module.in_features
        self.hin = None
        self.hin_shared = False
        self.leader = None
        self.hout = torch.zeros(module.out_features, module.out_features, device = self.device) if want_out else None
        self.weigh_by = None
        self.weigh_scale = 1.0
        self.count = 0
        self.x = None

    def new_hin(self):
        self.hin = torch.zeros(self.in_features, self.in_features, device = self.device)
        self.hin_shared = False

    def begin_pass(self, kind):
        """
        Reset the side being updated in this pass. The other side weights the samples, normalized by a scalar
        rather than in a copy, which would be another 50% on top of the factors for the large side
        """
        self.count = 0
        self.weigh_by = None
        self.weigh_scale = 1.0
        if kind == "out" and self.hout is not None:
            self.hout.zero_()
            self.weigh_by = self.hin
        elif kind == "in" and self.hout is not None:
            if self.hin_shared:
                self.new_hin()
            else:
                self.hin.zero_()
            self.weigh_by = self.hout
        if self.weigh_by is not None:
            self.weigh_scale = 1.0 / self.weigh_by.norm().item() ** 2

    def end_pass(self, kind):
        self.weigh_by = None
        if kind == "out" and self.hout is not None:
            self.hout /= max(self.count, 1)
        elif kind == "fwd" and self.leader is not None:
            self.hin = self.leader.hin
            self.hin_shared = True
        elif kind == "fwd" or (kind == "in" and self.hout is not None):
            self.hin /= max(self.count, 1)

    def end_row(self):
        self.x = None

    def forward(self, kind, x, leader):
        """
        Module input. leader is the collector that already received this same tensor, if any
        """
        x = x.detach().reshape(-1, x.shape[-1])
        if kind == "fwd":
            self.leader = leader
            if leader is None:
                if self.hin is None:
                    self.new_hin()
                x = x.to(self.device, non_blocking = True).float()
                self.hin.addmm_(x.T, x)
                self.count += x.shape[0]
        elif self.hout is not None and (kind == "in" or self.weigh_by is not None):
            # Kept until the gradient arrives, and moved right away so the copy overlaps with the rest of the forward
            # pass. A shared Hin may be on another device, and the weights are computed where it is
            self.x = x.to(self.device if kind == "in" else self.weigh_by.device, non_blocking = True)

    def backward(self, kind, d):
        """
        Gradient w.r.t. the module output
        """
        if kind == "fwd" or self.hout is None:
            return
        d = d.detach().reshape(-1, d.shape[-1]).to(self.device, non_blocking = True).float()
        x = self.x.float() if self.x is not None else None
        if kind == "out":
            if self.weigh_by is not None:
                w = ((x @ self.weigh_by) * x).sum(-1, keepdim = True) * self.weigh_scale  # x_t^T Hin x_t / |Hin|^2
                self.hout.addmm_(d.T, d * w.to(self.device, non_blocking = True))
            else:
                self.hout.addmm_(d.T, d)
        else:
            w = ((d @ self.weigh_by) * d).sum(-1, keepdim = True) * self.weigh_scale  # d_t^T Hout d_t / |Hout|^2
            self.hin.addmm_(x.T, x * w)
        self.count += d.shape[0]

    def save(self, out_dir, sides):
        tensors = {}
        if self.hin is not None and (sides == "both"):
            tensors["hin"] = pack_sym(self.hin).cpu()
        if self.hout is not None:
            tensors["hout"] = pack_sym(self.hout).cpu()
        save_file(tensors, os.path.join(out_dir, self.key + ".safetensors"))


def main(args):
    devices = [int(d) for d in args.device.split(",")]
    hess_devices = [torch.device(f"cuda:{d}") for d in args.hessian_device.split(",")] if args.hessian_device else None
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    # Pass schedule
    if args.sides == "both":
        assert not args.unweighted, "--unweighted only applies to --sides out"
        num_passes = args.passes or 6
        schedule = (["fwd"] + ["out", "in"] * num_passes)[:num_passes]
    elif args.unweighted:
        schedule = ["out"]
    else:
        num_passes = args.passes or 2
        assert num_passes % 2 == 0, "--sides out needs an even number of passes to end on Hout"
        schedule = (["fwd"] + ["out", "in"] * num_passes)[:num_passes]
    want_in = schedule[0] == "fwd"

    rows = load_rows(args)
    model = load_model(args, devices)
    targets, first, last, head = find_targets(args, model)
    if args.sides == "out":
        targets = {k: v for k, v in targets.items() if v[0] != "lm_head"}
        head = False
    if not targets:
        print(" !! Nothing to do")
        return
    if max(r.max().item() for r in rows) >= model.get_input_embeddings().num_embeddings:
        raise ValueError(f"Calibration file {args.cal_data} contains token ids outside the model's vocab")
    in_device = model.get_input_embeddings().weight.device

    # Factors go with their layers, or on the given devices, largest first onto the least loaded
    collectors = {}
    load = {d: 0 for d in hess_devices or []}
    for name, (key, mod) in sorted(targets.items(), key = lambda t: -(t[1][1].in_features ** 2 + t[1][1].out_features ** 2)):
        dev = min(load, key = load.get) if load else None
        c = Collector(name, key, mod, dev, want_in, key != "lm_head")
        collectors[name] = c
        if load:
            load[dev] += mod.in_features ** 2 * want_in + mod.out_features ** 2
    print(f" -- Passes: {', '.join(schedule)}" + (f", {args.samples} targets per row" if args.samples > 1 else ""))

    # Hooks: inputs on the way forward, output gradients on the way back (a tensor hook on the output, which is
    # lighter than a module backward hook and does the same job here). Siblings are recognized by arriving right after
    # one another with the same input; holding on to that input in the meantime keeps its address from being reused
    kind = [schedule[0]]
    last_input = [None, None]

    def forward_hook(c, inp, out):
        x = inp[0]
        lx, leader = last_input
        same = lx is not None and (x is lx or (
            x.data_ptr() == lx.data_ptr() and x.shape == lx.shape and x.stride() == lx.stride() and
            x.dtype == lx.dtype and x.device == lx.device
        ))
        if not same:
            last_input[:] = x, c
            leader = None
        c.forward(kind[0], x, leader)
        if kind[0] != "fwd" and out.requires_grad:
            out.register_hook(lambda d, c = c: c.backward(kind[0], d))

    for name, (key, mod) in targets.items():
        mod.register_forward_hook(lambda m, inp, out, c = collectors[name]: forward_hook(c, inp, out))

    # All parameters are frozen, so the graph starts where the hidden state first requires a gradient. Doing that at
    # the first collected layer rather than at the embeddings skips the backward pass through everything below it
    def start_graph(mod, a, kw):
        if kind[0] == "fwd":
            return None
        if len(a):
            return (a[0].detach().requires_grad_(True),) + tuple(a[1:]), kw
        kw = dict(kw)
        kw["hidden_states"] = kw["hidden_states"].detach().requires_grad_(True)
        return a, kw

    # The forward-only pass needs nothing past the last collected layer, unless the head is included
    def stop_forward(mod, a, out):
        if kind[0] == "fwd" and not head:
            raise StopForward()

    if first is not None:
        first.register_forward_pre_hook(start_graph, with_kwargs = True)
        last.register_forward_hook(stop_forward)
    else:
        schedule = schedule[:1]  # head only

    for p, kind[0] in enumerate(schedule):
        t0 = time.time()
        for c in collectors.values():
            c.begin_pass(kind[0])

        for r, ids in enumerate(rows):
            ids = ids.to(in_device)
            if kind[0] == "fwd":
                try:
                    with torch.no_grad():
                        model(input_ids = ids, use_cache = False)
                except StopForward:
                    pass
            else:
                with torch.enable_grad():
                    logits = model(input_ids = ids, use_cache = False).logits[0]
                # Fisher, not empirical Fisher: targets are samples from the model's own distribution. The gradient
                # of the summed cross-entropy w.r.t. the logits is softmax - onehot, applied directly. Summed, not
                # averaged, or the tokens of short rows would count for more
                probs = torch.softmax(logits.detach().float() / args.temperature, dim = -1)
                g = torch.Generator(device = probs.device).manual_seed(args.seed + r)
                samples = torch.multinomial(probs, args.samples, replacement = True, generator = g)
                pos = torch.arange(probs.shape[0], device = probs.device)
                for k in range(args.samples):
                    grad = probs.to(logits.dtype, copy = True)
                    grad[pos, samples[:, k]] -= 1.0
                    logits.backward(grad, retain_graph = k < args.samples - 1)
                del logits, probs, grad
            for c in collectors.values():
                c.end_row()
            last_input[:] = None, None
            if p == 0 and r == 0:
                unique = {h.data_ptr(): h.numel() * 4 for c in collectors.values() for h in (c.hin, c.hout) if h is not None}
                shared = sum(c.leader is not None for c in collectors.values())
                print(f" -- Hessian factors: {sum(unique.values()) / 1024**3:.2f} GB" + (f", {shared} tensors share Hin with a sibling" if shared else ""))

            if (r + 1) % args.log_interval == 0 or r + 1 == len(rows):
                el = time.time() - t0
                eta = el / (r + 1) * (len(rows) - r - 1)
                print(f" -- Pass {p + 1}/{len(schedule)} ({kind[0]}): row {r + 1}/{len(rows)}, {el:.0f} s, ETA {eta:.0f} s", flush = True)

        for c in collectors.values():
            c.end_pass(kind[0])
        torch.cuda.synchronize()
        print(f" -- Pass {p + 1}/{len(schedule)} ({kind[0]}): {time.time() - t0:.0f} s")

    used = sorted(set(devices) | set(d.index for d in hess_devices or []))
    print(f" -- Peak VRAM: " + ", ".join(f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 1024**3:.1f} GB" for d in used))
    if args.dry_run:
        return
    os.makedirs(args.out_dir, exist_ok = True)
    for c in collectors.values():
        c.save(args.out_dir, args.sides)
    print(f" -- Wrote {len(collectors)} tensors to {args.out_dir}")


def parse_layers(s):
    m = re.fullmatch(r"(\d*):(\d*)", s)
    if not m:
        raise argparse.ArgumentTypeError("expected begin:end")
    return int(m.group(1) or 0), int(m.group(2) or 1 << 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Collect Kronecker-factored (YAQA) Hessians with Transformers, for convert.py -hess")
    parser.add_argument("-m", "--model_dir", type = str, required = True, help = "Unquantized (HF) model directory")
    parser.add_argument("-c", "--cal_data", type = str, required = True, help = "Packed calibration rows (safetensors with input_ids and optionally lengths, e.g. from sc_trace.py)")
    parser.add_argument("-o", "--out_dir", type = str, required = True, help = "Output directory, one file per tensor")
    parser.add_argument("-d", "--device", type = str, default = "0", help = "Device index, or comma-separated list to split the model across, default: 0")
    parser.add_argument("-hd", "--hessian_device", type = str, default = None, help = "Device index (or list) to accumulate Hessians on, default: same device as each layer")
    parser.add_argument("-rv", "--reserve_vram", type = lambda s: [float(x) for x in s.split(",")], default = None, help = "With multiple devices: GB to keep free of weights per device (list, last value repeats), default: 8")
    parser.add_argument("-l", "--layers", type = parse_layers, default = None, help = "Range of decoder layers to collect, begin:end (end exclusive), default: all")
    parser.add_argument("--head", action = argparse.BooleanOptionalAction, default = None, help = "With --sides both: collect the output head (input side only), default: when the range includes the last layer")
    parser.add_argument("-r", "--rows", type = int, default = None, help = "Calibration rows to use, default: all rows in file")
    parser.add_argument("--skip_rows", type = int, default = 0, help = "Skip this many rows at the start of the file (e.g. to collect from disjoint halves)")
    parser.add_argument("-cc", "--cols", type = int, default = None, help = "Max tokens per row, default: width of file")
    parser.add_argument("-s", "--sides", choices = ["out", "both"], default = "out", help = "Factors to collect: out (Hout only, conversion keeps its own calibrated Hin) or both, default: out")
    parser.add_argument("-p", "--passes", type = int, default = None, help = "Passes over the data, the first forward-only, then alternating Hout/Hin updates, default: 2 for out, 6 for both")
    parser.add_argument("--unweighted", action = "store_true", help = "With --sides out: plain sum of gradient outer products, a single backward pass and no Hin")
    parser.add_argument("-n", "--samples", type = int, default = 1, help = "Sampled targets (backward passes) per row, default: 1")
    parser.add_argument("-t", "--temperature", type = float, default = 1.0, help = "Temperature of the output distribution the Fisher is taken at. Above 1 gives weight to tokens the model is confident about, default: 1")
    parser.add_argument("--seed", type = int, default = 0, help = "Base RNG seed for target sampling")
    parser.add_argument("--tf32", action = "store_true", help = "Allow TF32 in the accumulation matmuls (faster on GeForce, inputs are BF16 anyway)")
    parser.add_argument("--pattern", type = str, default = default_pattern, help = "Regex selecting linear modules within the decoder layers")
    parser.add_argument("--key_prefix", type = str, default = None, help = "Name outputs as prefix + 'layers.N...' instead of matching module names to the checkpoint index")
    parser.add_argument("--dry_run", action = "store_true", help = "Don't write anything, for timing and memory tests")
    parser.add_argument("--log_interval", type = int, default = 25, help = "Rows between progress lines, default: 25")
    _args = parser.parse_args()
    main(_args)
