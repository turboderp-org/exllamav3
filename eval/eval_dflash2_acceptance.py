"""
DFlash2 draft-quality validation: acceptance rate on a real checkpoint pair.

Needs a target model and a DFlash2 draft checkpoint (GPU). Runs a fixed
prompt list through generate() with the draft attached and reports, per
prompt: accepted/rejected draft tokens, mean accepted per target-forward
(tau), and wall tok/s. Exit 0 with a summary; exit 2 when zero drafts were
accepted anywhere (something is structurally broken — wrong taps, dead
selector, verify mismatch).

Cells (same seed/prompts, collate by --tag):
    A: no -dm, no MTP          -> autoregressive baseline tok/s
    B: --mtp only               -> in-checkpoint MTP
    C: -dm greedy               -> DFlash2 greedy (add -dds -dc 0.4 for the
      cropped variant; measured at 8k, full-block wins on tau and tok/s)
    D: -dm sampled (serve sampler) -> the lossless path you will ship

Usage:
    PYTHONPATH=. python eval/eval_dflash2_acceptance.py -m <target_dir> -dm <draft_dir> [-dds] [--temperature 0.8]

Phase profile (needs a CUDA build; process-local wrappers, nothing added
to Generator):
    PYTHONPATH=. python eval/eval_dflash2_acceptance.py -m <target> -dm <draft> --profile ...

Context sweep: --prefix-tokens 8000 repeats a filler paragraph to roughly
that prompt length before each prompt.

What good looks like (rule of thumb, checkpoint-dependent):
    - greedy: tau clearly above v1 DFlash on the same pair (diffusion
      drafter should clear ~2).
    - sampled: accept rate in the 0.5-0.8 band; bonus tokens on rejects.
    - tok/s vs cell A at the context you will serve.

Reading guide: DFlash2 wins only if accepted per target-forward stays high.
Flat tok/s with tau ~ 1 is a working draft not paying rent. Phase means
(with --profile): verify dominating with 3-6 accepted means memory-bound on
the target — stop, the next win is KV/cache, not DFlash2. Only if select
rivals verify is a fused walk on the table. Cropping (-dds/-dc) is negative
EV where measured (8k: full-block tau 2.29 vs 0.4-crop 1.73) — verify
positions are cheap; do not tighten -dc before kernel work, loosen it.
Short prefixes (2k) read low from prompt-1 warmup; the weak 2k cell was
additionally content noise (a different 2k crop reads 1.78) — note the cell,
don't average it into long-ctx conclusions. Energy (optional): locked clocks, nvidia-smi dmon -s
p during generate; J/token approx P_avg x t_wall / N_out, A vs C vs D.
"""
import argparse
import sys
import time

import torch

from exllamav3 import model_init
from exllamav3.generator import Generator

PROMPTS = [
    "Explain why the sky is blue in one paragraph.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Summarize the causes of the French Revolution in three sentences.",
    "Translate 'The quick brown fox jumps over the lazy dog' into French.",
    "List the planets of the solar system in order from the Sun.",
    "Write a haiku about rain.",
    "What is the capital of Japan?",
    "Compute 15% of 240.",
    "Name three benefits of regular exercise.",
    "Who was the first person to walk on the Moon?",
    "Translate 'Good morning' into Spanish.",
    "What is 7 times 8?",
    "Write a Python function that reverses a string.",
    "Summarize photosynthesis in two sentences.",
    "What is the largest ocean on Earth?",
    "Who wrote 'Hamlet'?",
    "Convert 100 degrees Celsius to Fahrenheit.",
    "Name the first three prime numbers.",
    "Write a Python function that checks if a number is even.",
    "What gas do plants absorb from the atmosphere?",
]

FILLER = (
    "The history of cartography traces how humans have mapped their world, from "
    "Babylonian clay tablets through Ptolemaic projections to modern satellite "
    "surveys. Each era added precision: the compass, the chronometer, aerial "
    "photography, and finally orbital geodesy. "
)


def build_prompt(tokenizer, prompt, prefix_tokens, offset = 0):
    if not prefix_tokens:
        return prompt
    prefix = ""
    ids = tokenizer.encode(prefix)
    while ids.numel() < prefix_tokens + offset:
        prefix += FILLER
        ids = tokenizer.encode(prefix)
    flat = ids.flatten()
    tail = flat[-prefix_tokens - offset : len(flat) - offset]
    decoded = tokenizer.decode(tail.unsqueeze(0))
    if isinstance(decoded, list):
        decoded = decoded[0]
    return decoded + "\n\n" + prompt


def main(args):
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
        model_init.init(args)

    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        draft_model = draft_model,
        draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens,
        dynamic_draft_tokens = args.dynamic_draft,
        draft_confidence = args.draft_confidence,
    )
    if args.draft_model_dir and not getattr(args, "mtp", False):
        assert generator.dflash2_draft, "draft model did not attach as a DFlash2 drafter"

    # Process-local phase profiler (--profile): monkeypatch timing wrappers
    # onto this process's model objects only. Nothing is added to Generator.
    # verify_us counts target forwards inside a draft round (propose sets the
    # window, update_kv closes it), so prefill is excluded.
    phase = {}
    if args.profile:
        # Draft-gated wrappers (verify/propose/accept) attach only when a draft
        # is present; prefill/TTFT timing is draft-independent and always valid.

        def _wrap_gpu(obj, name, key):
            fn = getattr(obj, name)

            def wrapped(*a, **k):
                e0 = torch.cuda.Event(enable_timing = True)
                e1 = torch.cuda.Event(enable_timing = True)
                e0.record()
                try:
                    return fn(*a, **k)
                finally:
                    e1.record()
                    e1.synchronize()
                    phase[key] = phase.get(key, 0.0) + e0.elapsed_time(e1) * 1000.0
                    phase[key + "_n"] = phase.get(key + "_n", 0) + 1

            setattr(obj, name, wrapped)

        _wrap_gpu(model, "prefill", "prefill_us")

        if draft_model is not None:
            _wrap_gpu(draft_model, "forward", "draft_us")

            propose_fn = draft_model.propose
            kv_fn = draft_model.update_kv_from_target
            verify_fn = model.forward

            def wrapped_propose(*a, **k):
                e0 = torch.cuda.Event(enable_timing = True)
                e1 = torch.cuda.Event(enable_timing = True)
                e0.record()
                try:
                    return propose_fn(*a, **k)
                finally:
                    e1.record()
                    e1.synchronize()
                    phase["select_us"] = phase.get("select_us", 0.0) + e0.elapsed_time(e1) * 1000.0
                    phase["select_us_n"] = phase.get("select_us_n", 0) + 1
                    phase["_active"] = True

            def wrapped_kv(*a, **k):
                try:
                    return kv_fn(*a, **k)
                finally:
                    phase["_active"] = False

            def wrapped_verify(*a, **k):
                if not phase.get("_active"):
                    return verify_fn(*a, **k)
                e0 = torch.cuda.Event(enable_timing = True)
                e1 = torch.cuda.Event(enable_timing = True)
                e0.record()
                try:
                    return verify_fn(*a, **k)
                finally:
                    e1.record()
                    e1.synchronize()
                    phase["verify_us"] = phase.get("verify_us", 0.0) + e0.elapsed_time(e1) * 1000.0
                    phase["verify_us_n"] = phase.get("verify_us_n", 0) + 1

            draft_model.propose = wrapped_propose
            draft_model.update_kv_from_target = wrapped_kv
            model.forward = wrapped_verify

            accept_fn = generator._dflash2_accept

            def wrapped_accept(*a, **k):
                t0 = time.perf_counter()
                try:
                    return accept_fn(*a, **k)
                finally:
                    # sync-inclusive wall time (the accept path syncs by design)
                    phase["accept_us"] = phase.get("accept_us", 0.0) + (time.perf_counter() - t0) * 1e6
                    phase["accept_us_n"] = phase.get("accept_us_n", 0) + 1

            # instance attribute shadows the class method for this process only
            generator._dflash2_accept = wrapped_accept

    sampler = model_init.get_arg_sampler(args)

    if args.failure_modes:
        # Failure-path checks on silicon (needs loaded models, no prompts):
        # a CFG-shaped tuple prompt must die in Job construction (single-
        # sequence assert), before any drafting; and a block-starved draft
        # cache must fail at Generator construction with the documented
        # ValueError. Neither may reach the draft loop.
        assert draft_model is not None and generator.dflash2_draft
        try:
            generator.generate(
                (PROMPTS[0], PROMPTS[1]),
                max_new_tokens = 8,
                seed = args.seed,
                sampler = sampler,
            )
            print("CFG check: NO ERROR (unexpected)")
            return 2
        except AssertionError as e:
            print(f"CFG check: PASS (dies in Job construction: {e})")
        # Page budget: no public Cache constructor can build a block-starved
        # cache (every layer type requires max_num_tokens % 256 == 0, and
        # block_size is 8), so the Generator guard is defense-in-depth. Pin
        # the live helper directly with starved numbers plus the real floor.
        from exllamav3.generator.generator import _dflash2_check_page_budget
        from exllamav3.constants import PAGE_SIZE
        bs = draft_config.block_size
        for tokens, pages in ((bs - 1, 1), (bs, 0)):
            try:
                _dflash2_check_page_budget(tokens, pages, bs)
                print(f"PAGE check ({tokens}tok/{pages}pg): NO ERROR (unexpected)")
                return 2
            except ValueError as e:
                print(f"PAGE check ({tokens}tok/{pages}pg): PASS ({e})")
        print(f"PAGE floor: min constructible cache {PAGE_SIZE} tokens "
              f"= {PAGE_SIZE // bs} draft blocks; guard untriggerable via "
              f"public constructors by design")
        return 0

    if args.draft_model_dir and not getattr(args, "mtp", False):
        from exllamav3.generator.generator import _dflash2_sampler_params
        if _dflash2_sampler_params(sampler) is None:
            print("NOTE: configured sampler is q-aware-ineligible "
                  "(min-p/penalties/mixed) — greedy walk + token-match verify.")

    def run_cell(prefix_tokens, cell_tag):
        # One sweep cell: all prompts at one prefix length. Totals reset per
        # cell so multi-prefix runs (--prefix-tokens 32000 2000) and repeats
        # (--repeat 2: rep1 cold vs rep2 warm) report independently.
        total_acc, total_rej, total_out, total_wall = 0, 0, 0, 0.0
        rows = []  # per-tally (acc, rej, out-tokens, wall) for the STEADY line
        prompts = PROMPTS[:args.num_prompts]
        if getattr(args, "prompt_idxs", None):
            idxs = [int(i) for i in args.prompt_idxs.split(",")]
            prompts = [PROMPTS[i] for i in idxs]
        if getattr(args, "mtp", False):
            print(f"tag={cell_tag} draft: in-checkpoint MTP")
        else:
            print(f"tag={cell_tag} draft: "
                  f"{'none' if draft_model is None else f'block {draft_config.block_size}, window {generator.num_draft_tokens}, dynamic={generator.dynamic_draft}'}")
        # Per-run settings echo: every cell self-describes its config so
        # cross-chain comparisons can't silently differ in crop/chunks.
        print(f"cfg={cell_tag} prefix={prefix_tokens} "
              f"chunk={getattr(args, 'chunk_size', '?')} "
              f"crop={bool(getattr(args, 'dynamic_draft', False))}/"
              f"{getattr(args, 'draft_confidence', '?')} "
              f"new={args.max_new_tokens} prompts={len(prompts)}")

        def tally(texts, results, wall, label, gen_t0 = None):
            nonlocal total_acc, total_rej, total_out, total_wall
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(results, dict):
                results = [results]
            n_out = sum(tokenizer.encode(t).numel() for t in texts)
            acc = rej = 0
            if results:
                for r in results:
                    acc += r.get("accepted_draft_tokens", 0)
                    rej += r.get("rejected_draft_tokens", 0)
            total_acc += acc
            total_rej += rej
            total_out += n_out
            total_wall += wall
            rows.append((acc, rej, n_out, wall))
            rounds = n_out - acc  # each verify forward emits accepted drafts + 1 token
            tau = acc / rounds if rounds > 0 else 0.0
            warn = "  <-- WARN: nothing accepted" if (acc + rej) and acc == 0 else ""
            print(f"[{acc:3d} acc / {rej:3d} rej | tau {tau:.2f} | "
                  f"{n_out / wall:.1f} tok/s]{warn} {label[:60]}")
            if gen_t0 is not None and results:
                # Time-to-first-token off the job clock: load-excluded
                # prefill measure. Guarded: any shape change -> silently omit.
                # gen_t0 must be time.time() domain (job clock), NOT
                # perf_counter (monotonic) — mixing domains prints garbage.
                try:
                    jobs = [r.get("job") for r in results
                            if isinstance(r, dict) and r.get("job") is not None]
                    fts = [j.time_first_token for j in jobs
                           if getattr(j, "time_first_token", None)]
                    if fts and gen_t0 > 1e9:
                        print(f"    ttft~{min(fts) - gen_t0:.1f}s")
                except Exception:
                    pass
            for t in texts:
                print(f"    -> {t[:160]!r}")
            if args.profile and phase.get("select_us_n"):
                n = phase["select_us_n"]
                print(f"    phases x{n}: " + ", ".join(
                    f"{k}={phase.get(k, 0.0) / max(phase.get(k + '_n', 1), 1):.1f}us"
                    for k in ("draft_us", "select_us", "verify_us", "accept_us", "kv_us")
                ) + f", drafted~{(acc + rej) / max(n, 1):.1f}, accepted~{acc / max(n, 1):.1f}")

        if args.batch_all or args.mixed:
            # One generate() call with all prompts: concurrent jobs in a single
            # batch. --mixed gives job 2 a greedy sampler so consensus fails and
            # the whole batch must fall back to the greedy walk.
            prompts = [build_prompt(tokenizer, p, prefix_tokens, args.prefix_offset)
                       for p in prompts]
            if args.mixed:
                from exllamav3.generator.sampler.presets import ComboSampler
                samplers = [sampler, ComboSampler(temperature = 0.0),
                            sampler, sampler]
            else:
                samplers = sampler
            if args.profile:
                phase.clear()
            t0 = time.perf_counter()
            wt0 = time.time()
            out = generator.generate(
                prompts,
                max_new_tokens = args.max_new_tokens,
                seed = args.seed,
                sampler = samplers,
                return_last_results = True,
                completion_only = True,
            )
            wall = time.perf_counter() - t0
            completions, results = out if isinstance(out, tuple) else (out, None)
            tally(completions, results, wall, f"batch x{len(prompts)}"
                  + (" mixed-sampler" if args.mixed else ""), wt0)
        else:
            for prompt in prompts:
                full_prompt = build_prompt(tokenizer, prompt, prefix_tokens,
                                           args.prefix_offset)
                if args.profile:
                    phase.clear()
                t0 = time.perf_counter()
                wt0 = time.time()
                out = generator.generate(
                    full_prompt,
                    max_new_tokens = args.max_new_tokens,
                    seed = args.seed,
                    sampler = sampler,
                    return_last_results = True,
                    completion_only = True,
                )
                wall = time.perf_counter() - t0
                if isinstance(out, tuple):
                    completions, results = out
                else:
                    completions, results = out, None
                text = completions[0] if isinstance(completions, list) else completions
                tally(text, results, wall, prompt, wt0)

        denom = total_acc + total_rej
        rate = total_acc / denom if denom else 0.0
        rounds = total_out - total_acc
        # Blend tau: total accepts over total rounds. Low-tau prompts run more
        # rounds per token, so the blend weights toward weak cells — it is not
        # the mean of the per-prompt taus.
        tau = total_acc / rounds if rounds > 0 else 0.0
        print(f"\nTOTAL [{cell_tag}]: {total_acc} accepted / {total_rej} rejected "
              f"(accept rate {rate:.3f} over {denom} draft positions, "
              f"tau {tau:.2f}, {total_out / total_wall:.1f} tok/s)")
        if len(rows) > 1:
            # STEADY drops the first prompt (first-touch compile/alloc +
            # prefill), which otherwise dominates blends at low prompt counts.
            s_acc = sum(r[0] for r in rows[1:])
            s_rej = sum(r[1] for r in rows[1:])
            s_out = sum(r[2] for r in rows[1:])
            s_wall = sum(r[3] for r in rows[1:])
            s_rounds = s_out - s_acc
            s_tau = s_acc / s_rounds if s_rounds > 0 else 0.0
            print(f"STEADY [{cell_tag}]: {s_acc} accepted / {s_rej} rejected "
                  f"(tau {s_tau:.2f}, {s_out / s_wall:.1f} tok/s, "
                  f"{len(rows) - 1} prompts)")
        if draft_model is not None and not getattr(args, "mtp", False):
            if denom == 0:
                print("FAIL: no draft positions were verified at all")
                return 2
            if total_acc == 0:
                print("FAIL: zero drafts accepted — taps, selector, or verify path broken")
                return 2
        return 0

    code = 0
    for rep in range(args.repeat):
        for px in args.prefix_tokens:
            suffix = f"{args.tag}-{px}" + (f"-r{rep + 1}" if args.repeat > 1 else "")
            code = run_cell(px, suffix) or code
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, add_sampling_args = True, add_draft_model_args = True)
    parser.add_argument("--max_new_tokens", type = int, default = 64)
    parser.add_argument("--seed", type = int, default = 1234)
    parser.add_argument("--tag", type = str, default = "cell",
                        help = "Cell label for collating runs (A/B/C/D)")
    parser.add_argument("--prefix-tokens", type = int, nargs = "+", default = [0],
                        help = "One sweep cell per value, run in-process in listed order "
                             "(cache must fit the max; --prefix-tokens 32000 2000 puts 2k last)")
    parser.add_argument("--repeat", type = int, default = 1,
                        help = "Repeat the sweep in-process; rep2+ is warm (isolates first-touch numerics)")
    parser.add_argument("--prefix-offset", type = int, default = 0,
                        help = "Shift the filler window: same length, different content")
    parser.add_argument("--profile", action = "store_true",
                        help = "Timing wrappers for draft/select/verify/accept/kv phases (this process only)")
    parser.add_argument("--batch-all", action = "store_true",
                        help = "One generate() call with all prompts: concurrent batch jobs")
    parser.add_argument("--mixed", action = "store_true",
                        help = "Batched run with a greedy sampler on job 2 (consensus must fall back)")
    parser.add_argument("--failure-modes", action = "store_true",
                        help = "CFG tuple-job and block-starved cache checks only (needs -dm)")
    parser.add_argument("--num-prompts", type = int, default = 4,
                        help = "Use the first N of PROMPTS (default 4 reproduces the published table; "
                             "more prompts + the STEADY line for methodology upgrades)")
    parser.add_argument("--prompt-idxs", type = str, default = None,
                        help = "Comma-separated PROMPTS indices overriding --num-prompts "
                             "(e.g. prose-only subset)")
    _args = parser.parse_args()
    sys.exit(main(_args))
