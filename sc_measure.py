import sys, os

import argparse
import json
import math
import zlib
from collections import defaultdict
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.measures import compute_kl_div
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant.fp16 import LinearFP16
from exllamav3.modules.quant.exl3_lib.quantize import finalize_capture_H, get_hadamard_dt, had_k
from datasets import load_dataset

"""
Single-model quantization sensitivity measurement by weight-space noise injection.

For every quantizable Linear (qmap set), perturbs the fp16 weights in place with seeded noise of
a given relative Frobenius norm, runs the rest of the model from a cached boundary state, and
records the KL divergence of the final logits against the clean reference.

Two noise models:

 - iid (default): isotropic Gaussian dW. Validated on qwen3.8-27b vs swap attribution of a real
   2bpw conversion: Spearman 0.61, median 1.95x KL overestimate with strong per-type structure
   (up to 5.8x on v_proj). The bias is almost entirely the missing LDLQ error shaping. Real
   quantization error is steered away from data-covariant input directions, so it produces
   1.5-7x less output error per unit weight error than iid noise.

 - shaped (--shaped): mimics the LDLQ error distribution. A capture pass over --h_rows extra
   calibration rows accumulates the same Hessians the quantizer uses (via Linear.capture_H and
   finalize_capture_H, so damping, sign flips, block Hadamard and the block-16 LDL are all
   byte-identical to conversion). Per LDLQ theory, the quantizer's weight error is
   dW_rot = L^-T eta with eta white and H_rot = L D L^T, so shaped noise is sampled as
   dW = P^T L^-T eta (P = block-Hadamard x sign flips), with per-output-channel scaling
   mimicking out_scales, then normalized to the target weight rfn. Falls back to iid where the
   capture is unusable (q_fallback).

Noise levels are either global (--rfn 0.29,0.145) or per-tensor, anchored to the measured error
of an actual quantized model (--rfn_ref rfn.json from sc_rfn_probe.py, --rfn_scale 1.0,0.5).
Measuring two levels an octave apart gives a per-tensor scaling exponent as a sanity check on
the quadratic law.

Output JSON feeds sc_optimize.py, which turns the sensitivities into a per-tensor bitrate
recipe. Results are written incrementally and the script resumes from a partial output file.

NOTE: Currently the script requires the ability to load the full model under measurement on a 
      single GPU. Not validate don MoE models.
"""

# TODO: Add layer-streaming option
# TODO: Validate on MoE models

@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    dataset_text = "\n\n".join(
        load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )
    return dataset_text


def get_test_tokens(tokenizer, rows, eval_len):
    eval_tokens = tokenizer.encode(get_dataset_text({"dataset": "wiki2"}))
    num_tokens = eval_tokens.shape[-1]
    seqs = []
    for a in range(0, num_tokens - eval_len, eval_len):
        seqs.append(eval_tokens[:, a : a + eval_len])
        if len(seqs) >= rows:
            break
    assert len(seqs) >= rows, f"not enough calibration text for {rows} rows of {eval_len}"
    return torch.cat(seqs, dim = 0)


@torch.inference_mode()
def main(args):
    device = torch.device("cuda", args.device)
    torch.manual_seed(0)

    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(args.length)
    tokenizer = Tokenizer.from_config(config)
    vocab_size = tokenizer.actual_vocab_size

    model = Model.from_config(config)
    print(f" -- Loading model: {args.model}")
    model.load(device = device)

    total_rows = args.rows + (args.h_rows if args.shaped else 0)
    if args.trace:
        # Packed self-sampled trace from sc_trace.py: in-distribution for the model,
        # and the same data the quantizer can calibrate on (convert.py --cal_data)
        from safetensors.torch import load_file
        packed = load_file(args.trace)["input_ids"]
        assert packed.shape[0] >= total_rows, \
            f"trace has {packed.shape[0]} rows, need {total_rows} (rows + h_rows)"
        assert packed.shape[1] >= args.length, \
            f"trace rows are {packed.shape[1]} tokens, need {args.length}"
        ids = packed[:total_rows, :args.length]
    else:
        ids = get_test_tokens(tokenizer, total_rows, args.length)
    rows = list(ids[:args.rows].split(1))
    cap_states = list(ids[args.rows:].split(1)) if args.shaped else None

    mods = model.modules
    num_mods = len(mods)

    @torch.inference_mode()
    def forward_rows(start_idx, states, collect_states = False):
        """
        Forward every row from module start_idx to the end, streaming the KL vs the reference
        per row. Returns (mean kld, per-row hidden state after module start_idx if requested).
        """
        kld_sum = 0.0
        out_states = []
        for r, state in enumerate(states):
            params = {}
            x = state.clone() if state.is_floating_point() else state
            for i in range(start_idx, num_mods):
                mod = mods[i]
                x = mod.prepare_for_device(x, params)
                x = mod.forward(x, params)
                if collect_states and i == start_idx:
                    out_states.append(x.clone())
            ref = ref_logits[r]
            kl_vocab = min(vocab_size, x.shape[-1], ref.shape[-1])
            # Chunk over tokens: full-vocab fp32 logits can be GBs
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
        logits = []
        for ids_row in rows:
            params = {}
            x = ids_row
            for i in range(num_mods):
                mod = mods[i]
                x = mod.prepare_for_device(x, params)
                # Residual stream is mutated in place downstream; every cached state must be
                # cloned or all mid-stream experiments run on corrupted inputs
                boundary[i].append(x.clone() if x.is_floating_point() else x)
                x = mod.forward(x, params)
            logits.append(x.half().cpu())
        return boundary, logits

    print(" -- Reference pass")
    boundary, ref_logits = ref_pass()

    # Collect perturbation targets: every quantizable Linear, grouped by top-level module
    def walk(mod, out):
        out.append(mod)
        for ch in getattr(mod, "modules", []):
            walk(ch, out)

    targets_by_idx = defaultdict(list)
    num_targets = 0
    for i in range(num_mods):
        tree = []
        walk(mods[i], tree)
        for t in tree:
            if isinstance(t, Linear) and t.qmap is not None:
                assert isinstance(t.inner, LinearFP16), \
                    f"{t.key}: expected unquantized (fp16) tensor, got {type(t.inner).__name__}"
                targets_by_idx[i].append(t)
                num_targets += 1

    # Per-target noise levels
    if args.rfn_ref:
        with open(args.rfn_ref, "r") as f:
            ref_data = json.load(f)
        ref_rfn = {r["key"]: r["rfn"] for r in ref_data["results"]}
        scales = [float(s) for s in args.rfn_scale.split(",")]
        missing = [t.key for tl in targets_by_idx.values() for t in tl if t.key not in ref_rfn]
        assert not missing, f"keys missing from {args.rfn_ref}: {missing[:5]}..."
        levels = {key: [ref_rfn[key] * s for s in scales] for key in ref_rfn}
        num_levels = len(scales)
    else:
        rfns = [float(s) for s in args.rfn.split(",")]
        levels = defaultdict(lambda: rfns)
        num_levels = len(rfns)

    # Resume from partial output
    results = []
    done = set()
    if args.out and os.path.exists(args.out):
        with open(args.out, "r") as f:
            prev = json.load(f)
        results = prev["results"]
        done = {r["key"] for r in results}
        print(f" -- Resuming: {len(done)} tensors already measured")

    def save():
        if not args.out:
            return
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(
                model = args.model,
                rows = args.rows,
                length = args.length,
                draws = args.draws,
                mode = "shaped" if args.shaped else "iid",
                trace = args.trace,
                h_rows = args.h_rows if args.shaped else None,
                rfn_ref = args.rfn_ref,
                rfn_scale = args.rfn_scale if args.rfn_ref else None,
                rfn = None if args.rfn_ref else args.rfn,
                results = results,
            ), f, indent = 2)
        os.replace(tmp, args.out)

    @torch.inference_mode()
    def perturb_iid(lin, rfn, seed):
        """
        Add iid Gaussian noise with ||n|| = rfn * ||W|| to the weights in place (fp32 math,
        chunked over input features). Returns (saved original weights, realized rfn, ||W||).
        """
        w = lin.inner.weight
        rows_per_chunk = max(1, 2 ** 24 // w.shape[1])
        w_sq = 0.0
        for a in range(0, w.shape[0], rows_per_chunk):
            w_sq += w[a : a + rows_per_chunk].float().square().sum().item()
        sigma = rfn * math.sqrt(w_sq / w.numel())
        gen = torch.Generator(device = w.device)
        gen.manual_seed(seed)
        saved = w.clone()
        err_sq = 0.0
        for a in range(0, w.shape[0], rows_per_chunk):
            b = min(a + rows_per_chunk, w.shape[0])
            n = torch.randn((b - a, w.shape[1]), generator = gen, device = w.device, dtype = torch.float)
            w[a:b] = (w[a:b].float() + n.mul_(sigma)).to(w.dtype)
            # Realized error after rounding to storage dtype
            err_sq += (w[a:b].float() - saved[a:b].float()).square().sum().item()
            del n
        return saved, (err_sq / w_sq) ** 0.5, w_sq ** 0.5

    @torch.inference_mode()
    def perturb_shaped(lin, rfn, seed, L, su):
        """
        Add LDLQ-shaped noise: dW = P^T L^-T eta (eta white, P the quantizer's sign-flip +
        block-Hadamard rotation), with per-output-channel scaling mimicking out_scales,
        normalized to ||dW|| = rfn * ||W||. L is finalize_capture_H's unit-block-lower factor
        (diagonal zeroed) of the rotated, regularized H; solve_triangular treats the unit
        diagonal implicitly. Two passes with a re-seeded generator avoid holding the full fp32
        noise tensor (lm_head would be 5 GB).
        """
        w = lin.inner.weight                            # (k, n) = (in, out)
        k, n = w.shape
        assert L.shape[0] == k, f"{lin.key}: H dim {L.shape[0]} != in_features {k}"
        assert k % had_k == 0
        Lt = L.mT                                       # upper, unit diagonal implicit
        had = get_hadamard_dt(had_k, w.device, torch.float, 1.0 / math.sqrt(had_k))
        had_t = had.T.contiguous()
        su_col = su.view(k, 1).to(w.device)

        # Column norms: total weight norm + out_scales-like per-channel error weighting
        col_sq = torch.zeros(n, dtype = torch.float, device = w.device)
        rows_per_chunk = max(1, 2 ** 24 // n)
        for a in range(0, k, rows_per_chunk):
            col_sq += w[a : a + rows_per_chunk].float().square().sum(dim = 0)
        w_sq = col_sq.sum().item()
        col_scale = col_sq.sqrt()
        col_scale /= col_scale.square().mean().sqrt().clamp(min = 1e-20)
        col_scale.clamp_(min = 1e-4)

        cols_per_chunk = max(had_k, 2 ** 26 // k)

        def noise_chunks(apply_factor):
            # apply_factor None: accumulate ||dW||^2 only. Else: add scaled noise to w in place.
            # The generator is re-seeded so both passes see identical noise
            gen = torch.Generator(device = w.device)
            gen.manual_seed(seed)
            total_sq = 0.0
            for a in range(0, n, cols_per_chunk):
                b = min(a + cols_per_chunk, n)
                eta = torch.randn((k, b - a), generator = gen, device = w.device, dtype = torch.float)
                x = torch.linalg.solve_triangular(Lt, eta, upper = True, unitriangular = True)
                del eta
                x = (had_t @ x.view(k // had_k, had_k, b - a)).view(k, b - a)
                x *= su_col
                x *= col_scale[a:b].unsqueeze(0)
                if apply_factor is None:
                    total_sq += x.square().sum().item()
                else:
                    w[:, a:b] = (w[:, a:b].float() + x * apply_factor).to(w.dtype)
                del x
            return total_sq

        dw_sq = noise_chunks(None)
        factor = rfn * math.sqrt(w_sq / dw_sq)
        saved = w.clone()
        noise_chunks(factor)
        err_sq = 0.0
        for a in range(0, k, rows_per_chunk):
            b = min(a + rows_per_chunk, k)
            err_sq += (w[a:b].float() - saved[a:b].float()).square().sum().item()
        return saved, (err_sq / w_sq) ** 0.5, w_sq ** 0.5

    @torch.inference_mode()
    def advance_capture(top_idx, capture):
        """Advance the capture rows through module top_idx, accumulating H per qmap if capture
        is a dict (shared across rows)"""
        nonlocal cap_states
        new_states = []
        for st in cap_states:
            params = {} if capture is None else {"capture": capture}
            x = mods[top_idx].prepare_for_device(st, params)
            x = mods[top_idx].forward(x, params)
            # Retaining the final module's outputs would hold full-vocab logits for every
            # capture row; they are never needed
            if top_idx + 1 < num_mods:
                new_states.append(x)
            del x
        cap_states = new_states

    print(f" -- {num_targets} target tensors, {args.draws} draw(s) at {num_levels} noise level(s), "
          f"{'shaped' if args.shaped else 'iid'} noise")

    control_kld = {}
    quant_args = {"sigma_reg": 0.025}

    for top_idx in range(num_mods):
        tlist = targets_by_idx.get(top_idx, [])
        todo = [lin for lin in tlist if lin.key not in done]

        # Advance the capture rows through every module; accumulate H only where needed
        shaping = {}
        if args.shaped:
            capture = {} if todo else None
            advance_capture(top_idx, capture)
            if todo:
                torch.manual_seed(zlib.crc32(f"su|{top_idx}".encode()) & 0x7fffffff)
                for qmap, h_data in capture.items():
                    q_fallback, H, L, su, H_diag = finalize_capture_H(h_data, quant_args, False)
                    if q_fallback or L is None:
                        print(f" !! q_fallback for {qmap}, using iid noise")
                        shaping[qmap] = None
                    else:
                        shaping[qmap] = (L, su)
                    del H, H_diag
                del capture

        if not todo:
            continue

        # No-perturbation control: restarting from the cached boundary must reproduce the
        # reference exactly
        if top_idx not in control_kld:
            control_kld[top_idx], _ = forward_rows(top_idx, boundary[top_idx])
            assert control_kld[top_idx] == 0.0, \
                f"ctrl {control_kld[top_idx]} != 0 at module {top_idx}, restart machinery broken"

        for lin in todo:
            shape_lin = shaping.get(lin.qmap) if args.shaped else None
            res = dict(
                idx = top_idx,
                key = lin.key,
                qbits_key = lin.qbits_key,
                numel = lin.weights_numel(),
                shaped = shape_lin is not None,
                levels = [],
            )
            for li, rfn in enumerate(levels[lin.key]):
                for draw in range(args.draws):
                    seed = zlib.crc32(f"{lin.key}|{li}|{draw}".encode()) & 0x7fffffff
                    if shape_lin is not None:
                        saved, rfn_actual, w_norm = perturb_shaped(lin, rfn, seed, *shape_lin)
                    else:
                        saved, rfn_actual, w_norm = perturb_iid(lin, rfn, seed)
                    try:
                        collect = top_idx + 1 < num_mods
                        kld, states = forward_rows(top_idx, boundary[top_idx], collect_states = collect)
                    finally:
                        lin.inner.weight.copy_(saved)
                        del saved

                    # Injected error at the top-level module output, relative to the clean state
                    inj_sq, ref_sq = 0.0, 0.0
                    for st, clean in zip(states, boundary[top_idx + 1] if collect else []):
                        d = st.float() - clean.float().to(st.device)
                        inj_sq += d.square().sum().item()
                        ref_sq += clean.float().square().sum().item()
                        del d
                    del states
                    inj_rfn = (inj_sq / ref_sq) ** 0.5 if ref_sq else 0.0

                    res["levels"].append(dict(
                        rfn = rfn,
                        rfn_actual = rfn_actual,
                        draw = draw,
                        kld = kld,
                        inj_rfn = inj_rfn,
                    ))
                    res["w_norm"] = w_norm
                    print(f"    {lin.key:60} rfn {rfn_actual:.5f}   kld {kld:11.8f}   inj_rfn {inj_rfn:.6f}")

            results.append(res)
            save()

        for v in shaping.values():
            del v
        shaping.clear()
        torch.cuda.empty_cache()

    # Summary
    total = sum(r["levels"][0]["kld"] for r in results)
    print(f"\n -- Sum of per-tensor KLD at level 0: {total:.6f}")

    def layer_of(key):
        for part in key.split("."):
            if part.isdigit():
                return int(part)
        return None

    print("\n -- By layer (level 0):")
    by_layer = defaultdict(float)
    for r in results:
        l = layer_of(r["key"])
        by_layer["-" if l is None else l] += r["levels"][0]["kld"]
    for l, k in by_layer.items():
        bar = "#" * int(150 * k / max(total, 1e-12))
        print(f"      {str(l):10} {k:.6f}  ({100 * k / total:5.1f}%)  {bar}")

    print("\n -- Top contributors (level 0):")
    for r in sorted(results, key = lambda r: -r["levels"][0]["kld"])[:args.top]:
        k = r["levels"][0]["kld"]
        print(f"      {r['key']:60} kld {k:.6f}  ({100 * k / total:4.1f}%)")

    if num_levels > 1:
        alphas = []
        for r in results:
            l0, l1 = r["levels"][0], r["levels"][-1]
            if l0["kld"] > 0 and l1["kld"] > 0 and l0["rfn_actual"] != l1["rfn_actual"]:
                alphas.append(
                    math.log(l0["kld"] / l1["kld"]) / math.log(l0["rfn_actual"] / l1["rfn_actual"]))
        if alphas:
            alphas.sort()
            print(f"\n -- Scaling exponent (kld ~ rfn^a): "
                  f"min {alphas[0]:.2f}  median {alphas[len(alphas) // 2]:.2f}  max {alphas[-1]:.2f}")

    save()
    if args.out:
        print(f"\n -- Saved: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("-m", "--model", type = str, required = True, help = "Unquantized model directory")
    parser.add_argument("-r", "--rows", type = int, default = 10, help = "Number of eval rows, default: 10")
    parser.add_argument("-l", "--length", type = int, default = 1024, help = "Tokens per row, default: 1024")
    parser.add_argument("-d", "--device", type = int, default = 0, help = "CUDA device index")
    parser.add_argument("-tr", "--trace", type = str, default = None, help = "Packed self-sampled trace (safetensors from sc_trace.py) to use as eval")
    parser.add_argument("-sh", "--shaped", action = "store_true", help = "LDLQ-shaped noise from captured Hessians (recommended; extra capture pass)")
    parser.add_argument("-hr", "--h_rows", type = int, default = 64, help = "Calibration rows for Hessian capture in shaped mode, default: 64")
    parser.add_argument("-rfn", "--rfn", type = str, default = "0.29,0.145", help = "Comma-separated global noise levels (relative weight Frobenius norm)")
    parser.add_argument("-rr", "--rfn_ref", type = str, default = None, help = "Per-tensor noise anchors from sc_rfn_probe.py JSON (overrides --rfn)")
    parser.add_argument("-rs", "--rfn_scale", type = str, default = "1.0,0.5", help = "Scale factors applied to --rfn_ref anchors, default: 1.0,0.5")
    parser.add_argument("-dr", "--draws", type = int, default = 1, help = "Noise draws per level, default: 1")
    parser.add_argument("-t", "--top", type = int, default = 15, help = "Top contributors to print")
    parser.add_argument("-o", "--out", type = str, default = None, help = "Output file (JSON), resumes if present")
    main(parser.parse_args())
