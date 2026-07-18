import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import math
from collections import defaultdict
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.measures import compute_kl_div
from model_diff import get_test_tokens

"""
Quantization error attribution by single-module swap.

Runs the *reference* model (B), but substitutes the quantized weights (from A) for exactly one
module at a time. The KL divergence of the resulting logits against the clean reference isolates
that module's marginal contribution to the end-to-end error, propagated through an otherwise
noise-free network. Because one full forward per module would be expensive, the reference pass
caches the hidden state at every top-level module boundary and each experiment only runs the
suffix of the model from the swap point.

To the extent the per-module contributions add up to the full-model KL divergence (reported as
the additivity ratio; measured ~0.8-1.05 on models with KL 0.02-0.5), the output is a per-module
error budget: which tensors cause the final error, in units of the final metric. Contrast with
model_diff's per-layer state metrics, which measure accumulated drift and conflate injected
noise with propagated noise.

With --iso, each swap is followed by a control experiment that replaces the actual propagated
error at the swapped module's output with Gaussian noise of identical per-token norm. The ratio
kld/kld_iso measures how much more harmful the real (data-correlated) quantization error is than
random noise of the same size: ~1 means the noise is effectively isotropic and only its
magnitude (i.e. more bits) can help; >> 1 indicates direction-aligned error that a better
quantization objective could in principle avoid (observed strongly in the last layer, where
random directions are filtered by the final norm + head but real error survives).

Implementation notes, learned the hard way:
 - The residual stream is mutated in place by downstream modules. Every cached boundary state
   and every state handed to a forward pass must be cloned, or all mid-stream experiments
   silently run on corrupted inputs. The no-swap control per start index (ctrl column) must be
   exactly 0; anything else means the restart machinery is broken.
 - Module children may be referenced through list attributes (e.g. GatedMLP.gates/ups/downs),
   not just scalar attributes. Both are patched; if no reference is found the swap would be a
   silent no-op, so that is an error.
 - Suffix passes that start at the head must not retain per-row output states (full-vocab logit
   tensors).
"""


def main(args):
    device = torch.device("cuda", args.device)
    torch.manual_seed(0)

    config_a = Config.from_directory(args.model_a)
    config_a.override_dynamic_seq_len(args.length)
    config_b = Config.from_directory(args.model_b)
    config_b.override_dynamic_seq_len(args.length)
    tokenizer = Tokenizer.from_config(config_a)
    vocab_size = tokenizer.actual_vocab_size

    model_a = Model.from_config(config_a)
    model_b = Model.from_config(config_b)
    print(f" -- Loading A (quantized): {args.model_a}")
    model_a.load(device = device)
    print(f" -- Loading B (reference): {args.model_b}")
    model_b.load(device = device)

    ids = get_test_tokens(tokenizer, args.rows, args.length, args.length)
    rows = list(ids.split(1))

    mods_a = model_a.modules
    mods_b = model_b.modules
    num_mods = len(mods_b)

    @torch.inference_mode()
    def forward_rows(start_idx, states, ref_logits, noise = None, mods = None, collect_states = False):
        """
        Forward every row from module start_idx to the end (through model B unless mods is
        given), streaming the KL vs the reference per row. Returns (mean kld, per-row hidden
        state after module start_idx if collect_states).
        """
        mods = mods or mods_b
        kld_sum = 0.0
        out_states = []
        for r, state in enumerate(states):
            params = {}
            x = state.clone() if state.is_floating_point() else state
            for i in range(start_idx, num_mods):
                mod = mods[i]
                x = mod.prepare_for_device(x, params)
                x = mod.forward(x, params)
                if noise is not None and i == start_idx:
                    eps, gen = noise
                    n = torch.randn(x.shape, generator = gen, device = x.device, dtype = torch.float)
                    n = n / n.norm(dim = -1, keepdim = True) * eps[r].to(x.device).unsqueeze(-1)
                    x = (x.float() + n).to(x.dtype)
                if collect_states and i == start_idx:
                    out_states.append(x.clone())
            ref = ref_logits[r]
            kl_vocab = min(vocab_size, x.shape[-1], ref.shape[-1])
            # Chunk over tokens: full-vocab fp32 logits can be GBs when both models are resident
            x2 = x.view(-1, x.shape[-1])
            ref2 = ref.view(-1, ref.shape[-1])
            kl_row, n_row = 0.0, 0
            for a in range(0, x2.shape[0], 256):
                b = min(a + 256, x2.shape[0])
                kl_row += compute_kl_div(x2[a:b].float(), ref2[a:b].to(x.device), kl_vocab).sum().item()
                n_row += b - a
            kld_sum += kl_row / n_row
            del x
        return kld_sum / len(states), out_states

    @torch.inference_mode()
    def ref_pass():
        """Full reference pass, caching the input state to every module and the final logits"""
        boundary = [[] for _ in range(num_mods)]
        ref_logits = []
        for ids_row in rows:
            params = {}
            x = ids_row
            for i in range(num_mods):
                mod = mods_b[i]
                x = mod.prepare_for_device(x, params)
                boundary[i].append(x.clone() if x.is_floating_point() else x)
                x = mod.forward(x, params)
            ref_logits.append(x.half().cpu())
        return boundary, ref_logits

    print(" -- Reference pass")
    boundary, ref_logits = ref_pass()

    # Pair up modules between the two (structurally identical) trees
    def walk(mod, out):
        out.append(mod)
        for ch in getattr(mod, "modules", []):
            walk(ch, out)

    def find_parent(root, target):
        for ch in getattr(root, "modules", []):
            if ch is target:
                return root
            p = find_parent(ch, target)
            if p is not None:
                return p
        return None

    swaps = []   # (top_idx, key, mod_a, mod_b)
    for i in range(num_mods):
        tree_a, tree_b = [], []
        walk(mods_a[i], tree_a)
        walk(mods_b[i], tree_b)
        assert len(tree_a) == len(tree_b), f"module tree mismatch at index {i}"
        if args.level == "block":
            swaps.append((i, mods_b[i].key, mods_a[i], mods_b[i]))
        else:
            for ma, mb in zip(tree_a, tree_b):
                assert ma.key == mb.key, f"module tree mismatch: {ma.key} != {mb.key}"
                if type(ma).__name__ == "Linear" or len(tree_a) == 1:
                    swaps.append((i, ma.key, ma, mb))

    def patch(top_idx, mod_a, mod_b):
        """Replace mod_b with mod_a inside model B's tree; returns an undo closure"""
        if mod_b is mods_b[top_idx]:
            mods_b[top_idx] = mod_a
            def undo(): mods_b[top_idx] = mod_b
            return undo
        parent = find_parent(mods_b[top_idx], mod_b)
        attrs = [k for k, v in vars(parent).items() if v is mod_b]
        lists = [(v, j) for k, v in vars(parent).items() if isinstance(v, list)
                 for j, e in enumerate(v) if e is mod_b]
        assert attrs or lists, f"no reference to {mod_b.key} found in parent {parent.key}"
        for k in attrs: setattr(parent, k, mod_a)
        for lst, j in lists: lst[j] = mod_a
        def undo():
            for k in attrs: setattr(parent, k, mod_b)
            for lst, j in lists: lst[j] = mod_b
        return undo

    print(f" -- {len(swaps)} swap experiments at level '{args.level}'")
    results = []
    gen = torch.Generator(device = device)
    control_kld = {}

    for top_idx, key, mod_a, mod_b in swaps:
        # No-swap control: restarting from the cached boundary must reproduce the reference
        if top_idx not in control_kld:
            control_kld[top_idx], _ = forward_rows(top_idx, boundary[top_idx], ref_logits)

        undo = patch(top_idx, mod_a, mod_b)
        try:
            kld, states = forward_rows(top_idx, boundary[top_idx], ref_logits,
                                       collect_states = top_idx + 1 < num_mods)
        finally:
            undo()

        # Injected error at the swapped top-level module's output, vs the clean boundary state
        inj_sq, ref_sq, eps_rows = 0.0, 0.0, []
        if top_idx + 1 < num_mods:
            for st, clean in zip(states, boundary[top_idx + 1]):
                d = st.float().to(clean.device) - clean.float()
                eps_rows.append(d.norm(dim = -1).squeeze(0).cpu())
                inj_sq += d.square().sum().item()
                ref_sq += clean.float().square().sum().item()
        inj_rfn = (inj_sq / ref_sq) ** 0.5 if ref_sq else 0.0
        del states
        res = dict(idx = top_idx, key = key, kld = kld, inj_rfn = inj_rfn, ctrl = control_kld[top_idx])

        # Isotropic control: random noise at the same module output, same per-token norms
        if args.iso and eps_rows and inj_sq > 0:
            gen.manual_seed(hash(key) & 0x7fffffff)
            kld_iso, _ = forward_rows(top_idx + 1, boundary[top_idx + 1], ref_logits,
                                      noise = (eps_rows, gen))
            res["kld_iso"] = kld_iso

        results.append(res)
        line = f"    {key:55} kld {kld:11.8f}   ctrl {control_kld[top_idx]:10.8f}   inj_rfn {inj_rfn:.6f}"
        if "kld_iso" in res:
            line += f"   kld_iso {res['kld_iso']:11.8f}"
        print(line)

    total_sum = sum(r["kld"] for r in results)

    # Full-model KL for the additivity check
    print(" -- Full A pass")
    kld_full, _ = forward_rows(0, rows, ref_logits, mods = mods_a)

    # Summary
    contrib = [r for r in results if r["kld"] > 0 or r["inj_rfn"] > 0]
    print(f"\n -- Contributing modules: {len(contrib)}")
    print(f" -- Sum of per-swap KLD:  {total_sum:.8f}")
    print(f" -- Full-model KLD:       {kld_full:.8f}")
    print(f" -- Additivity ratio:     {kld_full / max(total_sum, 1e-12):.3f}")

    def layer_of(key):
        for part in key.split("."):
            if part.isdigit():
                return int(part)
        return None

    print("\n -- By layer:")
    by_layer = defaultdict(float)
    for r in contrib:
        l = layer_of(r["key"])
        by_layer["-" if l is None else l] += r["kld"]
    for l, k in by_layer.items():
        bar = "#" * int(150 * k / max(total_sum, 1e-12))
        print(f"      {str(l):10} {k:.6f}  ({100 * k / total_sum:5.1f}%)  {bar}")

    if args.level == "linear":
        print("\n -- By module type:")
        by_type = defaultdict(lambda: [0.0, 0.0, 0])
        for r in contrib:
            t = r["key"].split(".")[-1]
            by_type[t][0] += r["kld"]
            by_type[t][1] += r.get("kld_iso", 0.0)
            by_type[t][2] += 1
        for t, (k, kiso, n) in sorted(by_type.items(), key = lambda x: -x[1][0]):
            iso = f"   act/iso {k / kiso:5.2f}" if kiso > 0 else ""
            print(f"      {t:20} n={n:3}  kld_sum {k:.6f}  ({100 * k / total_sum:5.1f}%){iso}")

    print("\n -- Top contributors:")
    for r in sorted(contrib, key = lambda r: -r["kld"])[:args.top]:
        iso = f"   act/iso {r['kld'] / r['kld_iso']:6.2f}" if r.get("kld_iso") else ""
        print(f"      {r['key']:55} kld {r['kld']:.6f}  ({100 * r['kld'] / total_sum:4.1f}%){iso}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(
                model_a = args.model_a,
                model_b = args.model_b,
                rows = args.rows,
                length = args.length,
                level = args.level,
                kld_full = kld_full,
                results = results,
            ), f, indent = 2)
        print(f"\n -- Saved: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("-ma", "--model_a", type = str, required = True, help = "Model A (quantized)")
    parser.add_argument("-mb", "--model_b", type = str, required = True, help = "Model B (unquantized reference)")
    parser.add_argument("-r", "--rows", type = int, default = 20, help = "Number of eval rows, default: 20")
    parser.add_argument("-l", "--length", type = int, default = 2048, help = "Tokens per row, default: 2048")
    parser.add_argument("-d", "--device", type = int, default = 0, help = "CUDA device index")
    parser.add_argument("-lv", "--level", type = str, default = "linear", choices = ["block", "linear"],
                        help = "Swap granularity: whole top-level modules or individual Linear tensors")
    parser.add_argument("-iso", "--iso", action = "store_true",
                        help = "Isotropic-noise control per swap (doubles runtime)")
    parser.add_argument("-t", "--top", type = int, default = 15, help = "Top contributors to print")
    parser.add_argument("-o", "--out", type = str, default = None, help = "Save raw results (JSON)")
    main(parser.parse_args())
