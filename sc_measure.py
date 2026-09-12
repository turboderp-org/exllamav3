import sys, os

import argparse
import json
import math
import zlib
from collections import defaultdict
from contextlib import contextmanager
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.measures import compute_kl_div
from exllamav3.util.tensor import g_tensor_cache
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

With --streaming, load one top-level module at a time and keep cached states in system RAM.
Perturbed states fan out at each noise site and share module loads. --max-sys budgets pending
states by splitting the work into passes, with results saved after each pass.
The largest module plus Hessian/noise workspace must still fit on the device.
Not validated on MoE models.
"""

# TODO: Validate on MoE models


class ModuleRunner:
    """Run rows through one module at a time. Clone states because modules may mutate inputs,
    including modules that run on CPU."""

    def __init__(self, config, modules, device, streaming):
        self.config = config
        self.modules = modules
        self.device = device
        self.streaming = streaming

    @contextmanager
    def loaded(self, idx):
        mod = self.modules[idx]
        try:
            if self.streaming:
                defer = mod.can_defer_load()
                if defer:
                    self.config.stc.begin_deferred_load()
                try:
                    mod.load("cpu" if mod.caps.get("prefer_cpu") else self.device)
                    if defer:
                        self.config.stc.end_deferred_load()
                except BaseException:
                    if defer:
                        self.config.stc.abort_deferred_load()
                    raise
            yield mod
        finally:
            if self.streaming:
                mod.unload()
                g_tensor_cache.drop_all()

    def rows(self, mod, states, capture=None, consume=None):
        outputs = []
        for r, state in enumerate(states):
            params = {} if capture is None else {"capture": capture}
            x = mod.prepare_for_device(state.clone(), params)
            x = mod.forward(x, params)
            if consume is None:
                outputs.append(x.to("cpu", copy=True))
            else:
                consume(r, x)
            del x, params
        return outputs


def estimate_resident_bytes(modules, rows, length, vocab_size, shaped, h_rows, stc):
    """Estimate resident VRAM for weights, cached states, logits and noise/Hessian workspace.
    Use checkpoint headers since unloaded modules may return None from weights_numel()."""
    def walk(mod):
        yield mod
        for child in getattr(mod, "modules", []):
            yield from walk(child)

    weights = 0
    for key, filename in stc.tensor_file_map.items():
        tensor = stc.file_headers[filename][key]
        start, end = tensor["data_offsets"]
        # FP8 weights expand to FP16; retain the larger size for FP32 tensors
        weights += max(2 * math.prod(tensor["shape"]), end - start)
    linear_weights = 0
    width = 0
    largest_weight = 0
    hessian_peak = 0
    for mod in modules:
        qmaps = {}
        for lin in walk(mod):
            if isinstance(lin, Linear):
                weight_bytes = 2 * lin.in_features * lin.out_features
                linear_weights += weight_bytes
                width = max(width, lin.in_features)
                largest_weight = max(largest_weight, weight_bytes)
                if lin.qmap is not None:
                    qmaps[lin.qmap] = max(qmaps.get(lin.qmap, 0), lin.in_features)
        if qmaps:
            hessian_peak = max(hessian_peak,
                               4 * sum(k * k for k in qmaps.values())
                               + 16 * max(qmaps.values()) ** 2)
    weights = max(weights, linear_weights)  # account for padded/repeated linear modules
    states = 2 * length * width * (rows * (len(modules) + 2) + (h_rows if shaped else 0))
    logits = 2 * length * vocab_size + 16 * min(length, 256) * vocab_size
    workspace = largest_weight + (hessian_peak if shaped else 0) + 3 * 1024 ** 3
    return math.ceil(1.15 * (weights + states + logits + workspace))


def sysmem_budget(max_sys_gb):
    """Pending-state budget in bytes; default to half the RAM available after the reference pass"""
    if max_sys_gb is not None:
        return int(max_sys_gb * 1024 ** 3)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 // 2
    except OSError:
        pass
    return 8 * 1024 ** 3


def plan_fanout_passes(measurements_by_idx, state_bytes, budget):
    """Group modules by pending-state budget, counting control measurements too.
    Keep each module's measurements together, even if that module exceeds the budget."""
    passes, current, pending = [], [], 0
    for idx in sorted(measurements_by_idx):
        count = measurements_by_idx[idx]
        if current and (pending + count) * state_bytes > budget:
            passes.append(current)
            current, pending = [], 0
        current.append(idx)
        pending += count
    if current:
        passes.append(current)
    return passes

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
    mode = args.load_mode
    streaming = mode == "streaming"
    if mode == "auto":
        required = estimate_resident_bytes(model.modules, args.rows, args.length,
                                           vocab_size, args.shaped, args.h_rows, config.stc)
        free, _ = torch.cuda.mem_get_info(device)
        streaming = required > free
        print(f" -- VRAM preflight: {free / 1024 ** 3:.1f} GiB free, "
              f"{required / 1024 ** 3:.1f} GiB estimated for resident measurement")
    print(f" -- {'Streaming' if streaming else 'Loading'} model: {args.model}")
    if not streaming:
        try:
            model.load(device = device)
        except torch.cuda.OutOfMemoryError:
            if mode != "auto":
                raise
            config.stc.abort_deferred_load()
            model.unload()
            g_tensor_cache.drop_all()
            # Release the OOM traceback before retrying so it cannot retain weight buffers
            streaming = True
        if streaming:
            torch.cuda.empty_cache()
            print(" -- Full-model load exceeded VRAM; switching to layer streaming")
    runner = ModuleRunner(config, model.modules, device, streaming)

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

    def row_kld(r, x):
        ref = ref_logits[r]
        kl_vocab = min(vocab_size, x.shape[-1], ref.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])
        ref2 = ref.reshape(-1, ref.shape[-1])
        kl_row = 0.0
        for a in range(0, x2.shape[0], 256):
            b = min(a + 256, x2.shape[0])
            kl_row += compute_kl_div(x2[a:b].float(), ref2[a:b].to(x.device), kl_vocab).sum().item()
        return kl_row / x2.shape[0]

    @torch.inference_mode()
    def forward_rows(start_idx, states, collect_states = False):
        """
        Forward every row from module start_idx to the end, streaming the KL vs the reference
        per row. Returns (mean kld, per-row hidden state after module start_idx if requested).
        Resident mode only.
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
        if streaming:
            states = rows
            for i in range(num_mods):
                boundary[i] = states
                with runner.loaded(i) as mod:
                    states = runner.rows(
                        mod, states,
                        consume=(lambda r, x: logits.append(x.half().to("cpu", copy=True)))
                        if i == num_mods - 1 else None,
                    )
            return boundary, logits
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
                if not streaming:
                    assert isinstance(t.inner, LinearFP16), \
                        f"{t.key}: expected unquantized (fp16) tensor, got {type(t.inner).__name__}"
                targets_by_idx[i].append(t)
                num_targets += 1

    expected_keys = {lin.key for targets in targets_by_idx.values() for lin in targets}

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
        results = [r for r in prev["results"] if r["key"] in expected_keys]
        done = {r["key"] for r in results}
        print(f" -- Resuming: {len(done)} tensors already measured")

    def save():
        if not args.out:
            return
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            output = dict(
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
            )
            json.dump(output, f, indent = 2)
        os.replace(tmp, args.out)

    if expected_keys <= done:
        save()
        model.unload()
        config.stc.close()
        print(" -- All tensors already measured")
        return

    save()
    print(" -- Reference pass")
    boundary, ref_logits = ref_pass()

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
        is a dict (shared across rows). Resident mode only."""
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

    def finalize_shaping(top_idx, capture):
        """Finalize per-qmap shaping factors, falling back to iid noise for unusable captures"""
        shaping = {}
        torch.manual_seed(zlib.crc32(f"su|{top_idx}".encode()) & 0x7fffffff)
        for qmap in list(capture):
            h_data = capture.pop(qmap)
            q_fallback, H, L, su, H_diag = finalize_capture_H(h_data, quant_args, False)
            if q_fallback or L is None:
                print(f" !! q_fallback for {qmap}, using iid noise")
                shaping[qmap] = None
            else:
                shaping[qmap] = (L.cpu(), su.cpu()) if streaming else (L, su)
            del H, H_diag, L, su, h_data
        return shaping

    def measurement_plan(lin):
        """(level index, rfn, draw, seed) for every measurement on one tensor"""
        return [(li, rfn, draw, zlib.crc32(f"{lin.key}|{li}|{draw}".encode()) & 0x7fffffff)
                for li, rfn in enumerate(levels[lin.key]) for draw in range(args.draws)]

    def new_result(top_idx, lin, shaped):
        return dict(
            idx = top_idx,
            key = lin.key,
            qbits_key = lin.qbits_key,
            numel = lin.weights_numel(),
            shaped = shaped,
            levels = [],
        )

    def injected_rfn(states, top_idx):
        """Injected error at the top-level module output, relative to the clean state"""
        if states is None or top_idx + 1 >= num_mods:
            return 0.0
        inj_sq, ref_sq = 0.0, 0.0
        for st, clean in zip(states, boundary[top_idx + 1]):
            d = st.float() - clean.float().to(st.device)
            inj_sq += d.square().sum().item()
            ref_sq += clean.float().square().sum().item()
            del d
        return (inj_sq / ref_sq) ** 0.5 if ref_sq else 0.0

    def record(res, rfn, rfn_actual, draw, kld, inj_rfn, w_norm):
        res["levels"].append(dict(
            rfn = rfn,
            rfn_actual = rfn_actual,
            draw = draw,
            kld = kld,
            inj_rfn = inj_rfn,
        ))
        res["w_norm"] = w_norm
        print(f"    {res['key']:60} rfn {rfn_actual:.5f}   kld {kld:11.8f}   inj_rfn {inj_rfn:.6f}")

    def check_control(top_idx, kld):
        # No-perturbation control: restarting from the cached boundary must reproduce the
        # reference exactly
        control_kld[top_idx] = kld
        assert kld == 0.0, f"ctrl {kld} != 0 at module {top_idx}, restart machinery broken"

    if streaming:
        # Load each module once per pass, advancing pending states and spawning new perturbed states
        todo_by_idx = {idx: [lin for lin in targets_by_idx[idx] if lin.key not in done]
                       for idx in sorted(targets_by_idx)}
        todo_by_idx = {idx: lins for idx, lins in todo_by_idx.items() if lins}
        measurements_by_idx = {idx: 1 + sum(len(measurement_plan(lin)) for lin in lins)
                              for idx, lins in todo_by_idx.items()}
        state_bytes = max((sum(t.numel() * t.element_size() for t in states)
                           for states in boundary[1:]), default = 0)
        budget = sysmem_budget(args.max_sys)
        passes = plan_fanout_passes(measurements_by_idx, state_bytes, budget)
        peak = max((sum(measurements_by_idx[i] for i in p) for p in passes), default = 0)
        print(f" -- Fan-out: {sum(measurements_by_idx.values())} measurements over "
              f"{len(todo_by_idx)} module(s) in {len(passes)} pass(es); peak pending states "
              f"{peak * state_bytes / 1024 ** 3:.1f} GiB, budget {budget / 1024 ** 3:.1f} GiB")

        cap_idx = 0  # next module for capture rows
        for pass_no, members in enumerate(passes):
            # Capture rows stop where the next pass starts, so each module is captured once
            cap_stop = passes[pass_no + 1][0] - 1 if pass_no + 1 < len(passes) else members[-1]
            pending = []
            pass_results = []
            start = min(cap_idx, members[0]) if args.shaped else members[0]
            for idx in range(start, num_mods):
                last_mod = idx == num_mods - 1
                need_cap = args.shaped and cap_idx == idx and idx <= cap_stop
                spawn = todo_by_idx[idx] if idx in members else []
                assert not spawn or not args.shaped or need_cap
                if not (need_cap or pending or spawn):
                    continue
                print(f" -- [pass {pass_no + 1}/{len(passes)}] module {idx + 1}/{num_mods}: "
                      f"{len(pending)} pending, {len(spawn)} new tensor(s)")
                with runner.loaded(idx) as mod:

                    def run(states):
                        """Forward rows through the loaded module: (states, None), or
                        (None, mean KL) on the last module"""
                        scores = []
                        out = runner.rows(
                            mod, states,
                            consume = (lambda r, x: scores.append(row_kld(r, x))) if last_mod else None,
                        )
                        return (None, sum(scores) / len(scores)) if last_mod else (out, None)

                    shaping = {}
                    if need_cap:
                        capture = {} if spawn else None
                        cap_states = runner.rows(mod, cap_states, capture = capture,
                                                 consume = (lambda r, x: None) if last_mod else None)
                        cap_idx += 1
                        if spawn:
                            shaping = finalize_shaping(idx, capture)
                        del capture

                    # Advance pending states through the unperturbed module
                    for exp in pending:
                        exp["states"], exp["kld"] = run(exp["states"])

                    if spawn:
                        states, kld = run(boundary[idx])
                        pending.append(dict(control = idx, states = states, kld = kld))

                    for lin in spawn:
                        assert isinstance(lin.inner, LinearFP16), \
                            f"{lin.key}: expected unquantized (fp16) tensor, got {type(lin.inner).__name__}"
                        shape_lin = shaping.get(lin.qmap) if args.shaped else None
                        factors = None if shape_lin is None else \
                            tuple(t.to(lin.inner.weight.device) for t in shape_lin)
                        res = new_result(idx, lin, shape_lin is not None)
                        for li, rfn, draw, seed in measurement_plan(lin):
                            if factors is not None:
                                saved, rfn_actual, w_norm = perturb_shaped(lin, rfn, seed, *factors)
                            else:
                                saved, rfn_actual, w_norm = perturb_iid(lin, rfn, seed)
                            try:
                                states, kld = run(boundary[idx])
                            finally:
                                # Restore before unloading, which discards lin.inner
                                lin.inner.weight.copy_(saved)
                                del saved
                            pending.append(dict(
                                res = res, rfn = rfn, rfn_actual = rfn_actual, draw = draw,
                                w_norm = w_norm, inj_rfn = injected_rfn(states, idx),
                                states = states, kld = kld,
                            ))
                        pass_results.append(res)
                        del factors
                    for v in shaping.values():
                        del v
                    shaping.clear()

            # Check controls and save the completed pass
            for exp in pending:
                if "control" in exp:
                    check_control(exp["control"], exp["kld"])
                else:
                    record(exp["res"], exp["rfn"], exp["rfn_actual"], exp["draw"], exp["kld"],
                           exp["inj_rfn"], exp["w_norm"])
            del pending
            results.extend(pass_results)
            save()
            torch.cuda.empty_cache()

    else:
        for top_idx in range(num_mods):
            if top_idx > max(targets_by_idx, default=-1):
                break
            todo = [lin for lin in targets_by_idx.get(top_idx, []) if lin.key not in done]

            # Advance the capture rows through every module; accumulate H only where needed
            shaping = {}
            if args.shaped:
                capture = {} if todo else None
                advance_capture(top_idx, capture)
                if todo:
                    shaping = finalize_shaping(top_idx, capture)
                del capture

            if not todo:
                continue

            if top_idx not in control_kld:
                kld, _ = forward_rows(top_idx, boundary[top_idx])
                check_control(top_idx, kld)

            for lin in todo:
                shape_lin = shaping.get(lin.qmap) if args.shaped else None
                res = new_result(top_idx, lin, shape_lin is not None)
                for li, rfn, draw, seed in measurement_plan(lin):
                    if shape_lin is not None:
                        saved, rfn_actual, w_norm = perturb_shaped(lin, rfn, seed, *shape_lin)
                    else:
                        saved, rfn_actual, w_norm = perturb_iid(lin, rfn, seed)
                    try:
                        kld, states = forward_rows(top_idx, boundary[top_idx],
                                                   collect_states = top_idx + 1 < num_mods)
                    finally:
                        lin.inner.weight.copy_(saved)
                        del saved
                    record(res, rfn, rfn_actual, draw, kld, injected_rfn(states, top_idx), w_norm)
                    del states

                results.append(res)
                save()

            for v in shaping.values():
                del v
            shaping.clear()
            shape_lin = None
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
    model.unload()
    config.stc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    parser.add_argument("-m", "--model", type = str, required = True, help = "Unquantized model directory")
    parser.add_argument("-r", "--rows", type = int, default = 10, help = "Number of eval rows, default: 10")
    parser.add_argument("-l", "--length", type = int, default = 1024, help = "Tokens per row, default: 1024")
    parser.add_argument("-d", "--device", type = int, default = 0, help = "CUDA device index")
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument("--load-mode", choices = ("auto", "resident", "streaming"),
                         default = "auto", help = "auto (default): check free VRAM before choosing full-model loading or streaming")
    loading.add_argument("--streaming", dest = "load_mode", action = "store_const", const = "streaming",
                         help = "Force one-module-at-a-time loading with states and shaping factors in system RAM")
    loading.add_argument("--no-streaming", dest = "load_mode", action = "store_const", const = "resident",
                         help = "Force full-model loading, bypassing the VRAM preflight")
    parser.add_argument("-ms", "--max-sys", dest = "max_sys", type = float, default = None,
                        help = "Streaming: system RAM for pending fan-out states, in GB (default: half of what is available)")
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
