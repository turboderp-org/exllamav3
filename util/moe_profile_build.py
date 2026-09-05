"""Build a precomputed MoE expert-placement profile.

Runs prompts through the model, counts per-expert routing hits, and writes an .npz in the
same layout as the usage census, so profiles from either tool are interchangeable:

    counts_decode  int64 [n_prompts, n_layers, n_experts]
    counts_prefill int64 [n_prompts, n_layers, n_experts]

Decode is the allocation signal -- that is the regime CPU offload actually runs in.

  # from a bundled calibration corpus
  python util/moe_profile_build.py -m /models/GLM -corpus code -o out/code.npz
  # from your own prompts (one JSON list of strings, or a raw text file)
  python util/moe_profile_build.py -m /models/GLM -corpus ./my_rust.txt -o out/rust.npz
  # re-score an existing profile without loading a model
  python util/moe_profile_build.py -score out/code.npz -resident 108

Install it where the loader looks:
  <model_dir>/moe_profiles/<name>.npz   or  ~/.cache/exllamav3/moe_profiles/
then serve with:  --moe_cpu_profile <name>

WHY THIS SAMPLES MANY PROMPTS, AND WHY IT SCORES HELD-OUT
---------------------------------------------------------
A profile is only useful if the ranking it fits on one corpus still holds on text it has never
seen. Fitting on a SINGLE prompt and then reporting how well that prompt's own hot set covers
that same prompt's routing measures nothing: the hot set is chosen to cover it, so the number
is ~93% no matter how badly the ranking generalizes. Measured live, such a profile delivered
42-45% capture -- barely above the 37.5% that placing experts at random would give.

So this tool samples many independent windows, fits the ranking on a subset, and reports
capture on the DISJOINT remainder, against two reference points that make the number readable:

  uniform  = R/E      what random placement gets; a profile at this level is worthless
  oracle   = fit on the held-out set itself; the ceiling ANY static profile can reach here

If held-out sits near uniform, routing on this model is too input-dependent for static
placement and only a dynamic cache will help. If it sits near oracle, the profile is as good
as static placement gets and the remaining gap is inherent.
"""
import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

BUNDLED = {"wiki": "wiki.utf8", "c4": "c4.utf8", "code": "code.utf8",
           "multilingual": "multilingual.utf8", "technical": "technical.utf8",
           "tiny": "tiny.utf8"}


def cal_dirs():
    """The bundled .utf8 corpora live in the source tree; they are not installed as package
    data, so an installed exllamav3 has the module but not the files. Search both."""
    out = []
    try:
        from exllamav3.conversion import calibration_data
        out.append(os.path.join(os.path.dirname(os.path.abspath(calibration_data.__file__)),
                                "standard_cal_data"))
    except Exception:
        pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
    out.append(os.path.join(here, "exllamav3", "conversion", "standard_cal_data"))
    env = os.environ.get("EXL3_CAL_DATA_DIR")
    if env:
        out.insert(0, env)
    return out


def load_corpus(spec):
    """-> (list of texts, label). A single blob stays a single element here; it is split into
    independent prompts later, after tokenization, where the boundaries can be exact."""
    if spec in BUNDLED:
        tried = []
        for d in cal_dirs():
            fp = os.path.join(d, BUNDLED[spec])
            tried.append(fp)
            if os.path.isfile(fp):
                return [open(fp, encoding="utf8").read()], spec
        raise SystemExit(f"bundled corpus '{spec}' not found. Looked in:\n  " +
                         "\n  ".join(tried) +
                         "\nSet EXL3_CAL_DATA_DIR to the directory holding the .utf8 files.")
    if os.path.isfile(spec):
        if spec.endswith(".json"):
            v = json.load(open(spec))
            return ([v] if isinstance(v, str) else list(v)), os.path.basename(spec)
        return [open(spec, encoding="utf8", errors="replace").read()], os.path.basename(spec)
    raise SystemExit(f"corpus '{spec}' is neither bundled ({', '.join(BUNDLED)}) nor a file")


def make_windows(texts, tokenizer, n_prompts, plen):
    """Split the corpus into n_prompts independent token windows.

    Independence is the point. One long prompt yields one trajectory: its greedy continuation
    settles into a narrow routing pattern, and the resulting ranking encodes that trajectory
    rather than the corpus. Disjoint windows spread across the whole corpus give the ranking
    something to average over, and -- because each window is a separate row in the [P, L, E]
    census -- they are what makes a held-out split possible at all.
    """
    ids = []
    for t in texts:
        enc = tokenizer.encode(t)
        ids.append(enc.reshape(-1))
    import torch
    flat = torch.cat(ids) if len(ids) > 1 else ids[0]
    total = flat.shape[0]
    need = n_prompts * plen
    if total < need:
        n_fit = max(1, total // plen)
        print(f" !! corpus holds {total:,} tokens; {n_prompts} x {plen} needs {need:,}. "
              f"Using {n_fit} window(s).")
        n_prompts = n_fit
    # Evenly spaced, disjoint. Spread beats contiguous: a corpus is usually ordered by
    # document, so contiguous windows would all land in the same few documents.
    stride = max(plen, (total - plen) // max(n_prompts, 1)) if n_prompts > 1 else plen
    out = []
    for i in range(n_prompts):
        s = min(i * stride, max(0, total - plen))
        out.append(flat[s:s + plen].reshape(1, -1))
    return out


def find_moe(model):
    seen, out = set(), []
    def walk(m):
        if id(m) in seen: return
        seen.add(id(m))
        if hasattr(m, "routing_fn") and hasattr(m, "num_experts") and hasattr(m, "key"):
            out.append(m)
        for a in ("modules", "children", "layers"):
            k = getattr(m, a, None)
            if isinstance(k, (list, tuple)):
                for c in k:
                    if hasattr(c, "__dict__"): walk(c)
    walk(model)
    return out


# ---------------------------------------------------------------------------------------
# Scoring. Kept free of torch/exllamav3 so an existing .npz can be re-scored offline.
# ---------------------------------------------------------------------------------------

def capture_at(fit, test, R):
    """Fraction of `test` routing hits that land on the top-R experts ranked by `fit`.

    Both are [L, E]. Layers are weighted by their own hit count rather than averaged, so a
    layer that routes more often counts for more -- traffic is what we are trying to cut.
    """
    top = np.argsort(-fit, axis=1, kind="stable")[:, :R]              # [L, R]
    hit = np.take_along_axis(test, top, axis=1).sum()
    tot = test.sum()
    return float(hit) / float(tot) if tot else 0.0


def capture_report(counts, R_head, holdout_frac=0.34, seed=1234):
    """-> dict of capture numbers from a [P, L, E] census, using a disjoint prompt split.

    Returns None when there are too few prompts to split, which is exactly the condition that
    made the old single-prompt profiles unfalsifiable.
    """
    P, L, E = counts.shape
    if P < 4:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(P)
    n_test = max(1, int(round(P * holdout_frac)))
    test_i, fit_i = perm[:n_test], perm[n_test:]
    fit, test = counts[fit_i].sum(axis=0), counts[test_i].sum(axis=0)
    if fit.sum() == 0 or test.sum() == 0:
        return None
    curve = {}
    for frac in (0.25, 0.375, 0.5, 0.625):
        R = max(1, int(round(E * frac)))
        curve[frac] = {"R": R,
                       "held_out": capture_at(fit, test, R),
                       "in_sample": capture_at(fit, fit, R),
                       "oracle": capture_at(test, test, R),
                       "uniform": R / E}
    head = {"R": R_head,
            "held_out": capture_at(fit, test, R_head),
            "in_sample": capture_at(fit, fit, R_head),
            "oracle": capture_at(test, test, R_head),
            "uniform": R_head / E}
    return {"n_fit": len(fit_i), "n_test": len(test_i), "head": head, "curve": curve,
            "experts": E, "layers": L}


def print_capture(rep, bank):
    if rep is None:
        print("\n !! too few prompts to score held-out; capture numbers are unavailable.\n"
              "    Re-run with -nprompts 8 or more -- a profile that cannot be scored on text\n"
              "    it did not see is not a profile, it is a fit to one sample.")
        return
    h = rep["head"]
    print(f"\n -- capture ({bank}), fit on {rep['n_fit']} prompts, scored on {rep['n_test']} held out")
    print(f"    {'residency':>12}  {'held-out':>9}  {'uniform':>8}  {'oracle':>7}  {'in-sample':>10}")
    for frac in sorted(rep["curve"]):
        c = rep["curve"][frac]
        print(f"    {c['R']:>4}/{rep['experts']:<4} {100*frac:4.1f}%  {100*c['held_out']:8.1f}%  "
              f"{100*c['uniform']:7.1f}%  {100*c['oracle']:6.1f}%  {100*c['in_sample']:9.1f}%")
    lift = (h["held_out"] - h["uniform"]) / max(h["oracle"] - h["uniform"], 1e-9)
    print(f"\n -- at R={h['R']}: held-out capture {100*h['held_out']:.1f}%, "
          f"cold rate {100*(1-h['held_out']):.1f}%")
    print(f"    uniform (random placement) {100*h['uniform']:.1f}%   "
          f"oracle (best possible static) {100*h['oracle']:.1f}%")
    print(f"    profile realizes {100*lift:.0f}% of the available static headroom")
    if h["held_out"] - h["uniform"] < 0.03:
        print("\n !! held-out capture is within 3 points of RANDOM placement. Routing on this\n"
              "    model is too input-dependent for a static profile to help; a dynamic cache\n"
              "    (upstream's -mcs sweeps, or a demand-paged expert pool) is the only lever.")
    if h["in_sample"] - h["held_out"] > 0.25:
        print(f"\n !! overfit: in-sample {100*h['in_sample']:.1f}% vs held-out "
              f"{100*h['held_out']:.1f}%. Trust the held-out number; the in-sample one is\n"
              "    what this tool used to report, and it is what made a weak profile look strong.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("-score", "--score", type=str, default=None,
                   help="Re-score an existing .npz offline (no model load) and exit.")
    p.add_argument("-resident", "--resident", type=int, default=None,
                   help="Residency to headline, i.e. the -mcs complement (E - mcs). "
                        "Defaults to 37.5%% of experts.")
    p.add_argument("-holdout", "--holdout", type=float, default=0.34)
    known, _ = p.parse_known_args()

    if known.score:
        if known.score.lower().endswith((".exl3moe", ".safetensors")):
            # A packed profile carries its census as one tensor; read it and score exactly
            # as if it had come from the .npz. Serving never touches these bytes -- only
            # -score does -- so one file can be both the shipped artifact and the audit
            # trail.
            import struct
            with open(known.score, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n).decode("utf-8"))
                info = hdr.get("census")
                if info is None:
                    raise SystemExit(f"{known.score}: no census tensor "
                                     f"(packed with --no-census?). Score the .npz instead.")
                s0, s1 = info["data_offsets"]
                fh.seek(8 + n + s0)
                buf = fh.read(s1 - s0)
            dt = {"I64": "<i8", "I32": "<i4", "I16": "<i2"}[info["dtype"]]
            z = {"census": np.frombuffer(buf, dtype=np.dtype(dt)).reshape(info["shape"])}
            banks = ("census",)
        else:
            z = np.load(known.score, allow_pickle=False)
            banks = ("counts_decode", "counts_prefill")
        for bank in banks:
            if bank not in z:
                continue
            c = np.asarray(z[bank]).astype(np.float64)
            if c.ndim != 3 or c.sum() == 0:
                continue
            E = c.shape[2]
            R = known.resident or max(1, int(round(E * 0.375)))
            print(f"\n=== {os.path.basename(known.score)} :: {bank}  "
                  f"[{c.shape[0]} prompts, {c.shape[1]} layers, {E} experts] ===")
            print_capture(capture_report(c, R, known.holdout), bank)
        sys.exit(0)

    import torch
    from exllamav3 import model_init, Generator, Job
    from exllamav3.generator.sampler import GreedySampler

    model_init.add_args(p, cache=True, default_cache_size=98304,
                        default_autosplit_max_batch_size=1)
    p.add_argument("-corpus", "--corpus", type=str, required=True)
    p.add_argument("-o", "--out", type=str, required=True)
    p.add_argument("-nprompts", "--num_prompts", type=int, default=48,
                   help="Independent corpus windows to profile. Each becomes one row of the\n"
                        "[P, L, E] census, which is what makes held-out scoring possible.")
    p.add_argument("-plen", "--prompt_tokens", type=int, default=1024,
                   help="Prefill tokens per window.")
    p.add_argument("-gen", "--gen_tokens", type=int, default=128,
                   help="Decode tokens per window. Total decode hits = nprompts * gen * topk\n"
                        "* layers; ranking E experts needs many hits each, so prefer more\n"
                        "windows over a longer generation -- diversity beats length.")
    args = p.parse_args()

    # Never let an existing profile permute experts while we are measuring them.
    for v in ("EXL3_MOE_PROFILE", "EXL3_MOE_CPU_SPLIT_STATS"):
        os.environ.pop(v, None)

    texts, label = load_corpus(args.corpus)
    model, config, cache, tokenizer, *_ = model_init.init(args, max_chunk_size=4096)
    mods = find_moe(model)
    if not mods:
        raise SystemExit("no MoE layers found -- nothing to profile")
    # Model-independent: geometry comes from the model, never from constants. The only
    # requirement is a uniform expert count across MoE layers, which the [L, E] census layout
    # implies; a model that varies it needs a ragged format, so fail loudly rather than
    # silently truncate.
    counts_seen = sorted({int(m.num_experts) for m in mods})
    if len(counts_seen) != 1:
        raise SystemExit(
            f"this model has MoE layers with differing expert counts {counts_seen}; the "
            f"[layers, experts] profile layout cannot represent it")
    L, E = len(mods), mods[0].num_experts
    R_head = args.resident or max(1, int(round(E * 0.375)))

    windows = make_windows(texts, tokenizer, args.num_prompts, args.prompt_tokens)
    P = len(windows)
    print(f" -- {L} MoE layers x {E} experts; corpus '{label}', "
          f"{P} windows x {args.prompt_tokens} tok + {args.gen_tokens} decode")

    dec = np.zeros((P, L, E), dtype=np.int64)
    pre = np.zeros((P, L, E), dtype=np.int64)
    state = {"pi": 0}
    idx_of = {m.key: i for i, m in enumerate(mods)}

    for m in mods:
        orig = m.routing_fn
        def make(mod, fn):
            def wrapper(bsz, cfg, z, params):
                sel, w = fn(bsz, cfg, z, params)
                try:
                    e = sel.detach().reshape(-1).to(torch.int64).cpu().numpy()
                    rows = sel.shape[0] if sel.dim() > 1 else 1
                    bank = dec if rows == 1 else pre      # bsz 1 == decode step
                    np.add.at(bank[state["pi"], idx_of[mod.key]], e, 1)
                except Exception:
                    pass
                return sel, w
            return wrapper
        m.routing_fn = make(m, orig)

    gen = Generator(model=model, cache=cache, tokenizer=tokenizer, max_chunk_size=4096)
    for pi, ids in enumerate(windows):
        state["pi"] = pi
        if pi % 8 == 0 or pi == P - 1:
            print(f" -- window {pi+1}/{P}"); sys.stdout.flush()
        gen.enqueue(Job(input_ids=ids, max_new_tokens=args.gen_tokens,
                        stop_conditions=[], sampler=GreedySampler()))
        while gen.num_remaining_jobs():
            gen.iterate()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # .exl3moe is the default and the format to ship: safetensors carrying the ranking
    # precomputed, the summed counts, and the per-prompt census, each in its own byte range.
    # Serving preads only "ranking" (0.053 ms; the embedded census costs 0.007 ms of that
    # and nothing at all in the bytes actually read), while -score reads "census" to re-run
    # the held-out split. .npz is still written on request for interop with external
    # tooling, but it is 60x slower to load and needs decompressing in full.
    want_packed = not args.out.lower().endswith(".npz")
    npz_path = (os.path.splitext(args.out)[0] + ".npz") if want_packed else args.out
    np.savez_compressed(npz_path, counts_decode=dec, counts_prefill=pre)
    from exllamav3.model.moe_profile import model_fingerprint
    fp = model_fingerprint(config, E)
    print(" -- fingerprint: " + ", ".join(f"{k}={v}" for k, v in sorted(fp.items())))

    # Sample-size guard. Ranking E experts from a handful of hits produces noise that LOOKS
    # like extreme skew (most experts never sampled => the "hot" head trivially covers 100%).
    hits_per_expert = float(dec.sum()) / max(L * E, 1)
    MIN_HITS = 20.0
    bank_name, bank = "decode", dec
    if hits_per_expert < MIN_HITS:
        print(f"\n !! decode sample is too small: {int(dec.sum()):,} hits = "
              f"{hits_per_expert:.1f} per expert (want >= {MIN_HITS:.0f}).")
        pre_per = float(pre.sum()) / max(L * E, 1)
        if pre_per >= MIN_HITS:
            print(f" !! falling back to the PREFILL bank ({int(pre.sum()):,} hits = "
                  f"{pre_per:.1f}/expert) for the capture report. The saved .npz keeps BOTH\n"
                  f"    banks; re-run with more -nprompts/-gen for a decode-quality profile.")
            bank_name, bank = "prefill (decode too sparse)", pre
        else:
            print(" !! prefill is sparse too -- this profile is NOT usable. "
                  "Increase -nprompts/-gen/-plen.")

    rep = capture_report(bank.astype(np.float64), R_head, args.holdout)
    print_capture(rep, bank_name)

    meta = {"corpus": label, "layers": L, "experts": E, "prompts": P, "fingerprint": fp,
            "prompt_tokens": args.prompt_tokens, "gen_tokens": args.gen_tokens,
            "hits_per_expert_decode": round(float(dec.sum()) / max(L * E, 1), 2),
            "hits_per_expert_prefill": round(float(pre.sum()) / max(L * E, 1), 2),
            "layer_keys": [m.key for m in mods],
            "decode_hits": int(dec.sum()), "prefill_hits": int(pre.sum())}
    if rep is not None:
        meta["capture"] = {"resident": R_head, "bank": bank_name,
                           "held_out": round(rep["head"]["held_out"], 4),
                           "in_sample": round(rep["head"]["in_sample"], 4),
                           "oracle": round(rep["head"]["oracle"], 4),
                           "uniform": round(rep["head"]["uniform"], 4),
                           "n_fit": rep["n_fit"], "n_test": rep["n_test"]}
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if want_packed:
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from moe_profile_pack import pack
            info = pack(npz_path, args.out)
            os.remove(npz_path)                      # the packed file supersedes it
            print(f"\n -- wrote {args.out}  ({info['bytes']:,} bytes, census embedded)")
        except Exception as e:
            print(f"\n !! could not pack ({e}); keeping {npz_path}")
            args.out = npz_path
    print(f" -- {int(dec.sum()):,} decode hits, {int(pre.sum()):,} prefill")
    print("PROFILE_BUILD_DONE")
