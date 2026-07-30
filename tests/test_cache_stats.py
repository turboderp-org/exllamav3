"""
Tests for cache statistics (Generator.get_cache_stats).

The unit test is a plain pytest test:

    pytest tests/test_cache_stats.py

Integration runs in script mode and requires an exl3 model (any model for the snapshot checks; the
recurrent-capped section runs on models with recurrent states, e.g. Gemma-4 with SWA in recurrent mode):

    python tests/test_cache_stats.py --model <exl3 model dir>

The recurrent-capped section reproduces the silent warm-resume degradation: with every KV page held in
the CPU tier but the unlocking recurrent checkpoint evicted from an undersized recurrent cache, resumes
fall back to full prefill. get_cache_stats must report this as recurrent_capped_evicted (the budget
signal) rather than recurrent_uncovered (the policy signal). The recurrent budget is sized at runtime to
about three checkpoints, so the flood evicts deterministically on any model geometry.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job, GreedySampler
from exllamav3.constants import PAGE_SIZE

checks_passed = 0
checks_failed = 0

def check(name, cond, detail = ""):
    global checks_passed, checks_failed
    if cond:
        checks_passed += 1
        print(f"  PASS  {name}")
    else:
        checks_failed += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


def test_evicted_tracking():
    """
    RecurrentCache must remember recently evicted keys so restore-time misses can distinguish an evicted
    checkpoint (budget undersized) from one that never existed (interval/state policy), and must forget a
    key again once it is re-stashed.
    """
    import exllamav3.cache.recurrent as rec

    class _Dummy:
        loaded_tp = False

    class _DummyState:
        def __init__(self, position):
            self.position = position
        def stash(self):
            return {"position": self.position, "checkpoint_size": 8}

    rc = rec.RecurrentCache(_Dummy(), max_size = 20)   # fits two 8-byte checkpoints
    rc.put(b"a", _DummyState(1))
    rc.put(b"b", _DummyState(2))
    rc.put(b"c", _DummyState(3))                        # evicts a
    assert rc.was_evicted(b"a"), "evicted key must be remembered"
    assert not rc.was_evicted(b"b") and not rc.was_evicted(b"c"), "resident keys must not be flagged"
    rc.put(b"a", _DummyState(1))                        # evicts b, re-stashes a
    assert not rc.was_evicted(b"a"), "re-stashed key must be forgotten"
    assert rc.was_evicted(b"b"), "newly evicted key must be remembered"
    assert rc.evicted_count == 2
    assert rc.stats()["evicted_age_min_window"] is not None
    assert rc.stats(reset_window = True)["evicted_count"] == 2
    assert rc.stats()["evicted_age_min_window"] is None, "window reset must clear the age minimum"


def make_long_prompt(tokenizer, seed_text, target_pages):
    text = seed_text
    filler = (
        " The archive records further details, cross-referenced against seasonal observations"
        " and the incidental notes of several field assistants over many years of study."
    )
    while tokenizer.encode(text).shape[-1] < target_pages * PAGE_SIZE + 16:
        text += filler
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True, help = "exl3 model directory")
    ap.add_argument("--cache_size", type = int, default = 4096)
    ap.add_argument("--cpu_cache_mb", type = int, default = 1024)
    ap.add_argument("--max_batch_size", type = int, default = 2,
                    help = "Recurrent state slots per cache; the test is batch-1")
    args = ap.parse_args()

    try:
        test_evicted_tracking()
        check("test_evicted_tracking", True)
    except AssertionError as e:
        check("test_evicted_tracking", False, str(e))

    print(f" -- Loading model: {args.model}")
    config = Config.from_directory(args.model)
    model = Model.from_config(config, swa_full = False)
    cache_a = Cache(model, max_num_tokens = args.cache_size, max_batch_size = args.max_batch_size)
    cache_b = Cache(model, max_num_tokens = args.cache_size, max_batch_size = args.max_batch_size)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)
    sampler = GreedySampler()

    prompt_a = make_long_prompt(
        tokenizer,
        "The migration of songbirds across continents follows magnetic fields and stellar cues.", 6)
    a_ids = tokenizer.encode(prompt_a)
    flood_prompts = [
        make_long_prompt(tokenizer, f"Flood text number {j} concerns the industrial history of a city.", 5)
        for j in range(6)
    ]

    def run_job_ids(gen, ids, max_new):
        job = Job(input_ids = ids, max_new_tokens = max_new, sampler = sampler)
        gen.enqueue(job)
        out, eos_r = [], None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                t = r.get("token_ids")
                if t is not None:
                    out.append(t)
                if r.get("eos"):
                    eos_r = r
        out_ids = torch.cat(out, dim = -1) if out else torch.empty((1, 0), dtype = torch.long)
        return out_ids, eos_r

    def gen_prompt(gen, prompt):
        return run_job_ids(gen, tokenizer.encode(prompt), 16)

    # Snapshot sanity on a generator with the CPU tier: prompt A, flood past cache capacity, re-run A
    print(" -- Stats snapshot: requests, GPU cache, CPU tier")
    stats_gen = Generator(model = model, cache = cache_a, tokenizer = tokenizer,
                          cpu_cache_size = args.cpu_cache_mb * 1024**2)
    run_job_ids(stats_gen, a_ids, 24)
    for p in flood_prompts:
        gen_prompt(stats_gen, p)
    _, eos_a = run_job_ids(stats_gen, a_ids, 8)

    stats = stats_gen.get_cache_stats(reset_window = True)
    r, cc, gpu = stats["requests"], stats["cpu_cache"], stats["gpu_cache"]
    check("stats: requests counted", r["completed"] == 8 and r["prompt_tokens"] > 0,
          f"requests = {r}")
    check("stats: cached tokens bounded by prompt tokens", 0 <= r["cached_tokens"] <= r["prompt_tokens"],
          f"requests = {r}")
    check("stats: prefill time accumulated", r["prefill_time_sum"] > 0,
          f"requests = {r}")
    check("stats: GPU pool accounted", 0 <= gpu["referenced"] + gpu["retired_valid"] <= gpu["pages"],
          f"gpu_cache = {gpu}")
    check("stats: CPU tier activity recorded",
          cc is not None and cc["pushes"] > 0 and 0 <= cc["entries"] <= cc["slots"],
          f"cpu_cache = {cc}")
    check("stats: window reset clears minima",
          stats_gen.get_cache_stats()["requests"]["cached_ratio_min_window"] is None)

    if stats_gen.recurrent_cache is None:
        print(" -- Model has no recurrent states; skipping recurrent-capped section.")
        print(f"\n -- {checks_passed} passed, {checks_failed} failed")
        sys.exit(1 if checks_failed else 0)

    # Probe the model's checkpoint size so the capped scenario's budget (~3 checkpoints) is geometry-independent
    probe_rc = stats_gen.recurrent_cache
    checkpoint_bytes = probe_rc.current_size // max(len(probe_rc), 1) if len(probe_rc) else 0
    del stats_gen
    torch.cuda.empty_cache()
    if not checkpoint_bytes:
        print(" -- No recurrent checkpoints were created by the probe; skipping recurrent-capped section.")
        print(f"\n -- {checks_passed} passed, {checks_failed} failed")
        sys.exit(1 if checks_failed else 0)

    # Recurrent-capped detection: an undersized recurrent cache silently degrades warm resumes to full
    # prefill even with every KV page in the CPU tier. That condition must be visible in stats.
    print(f" -- Stats: recurrent-capped restore detection (budget = 3 checkpoints of ~{checkpoint_bytes} bytes)")
    capped_gen = Generator(model = model, cache = cache_b, tokenizer = tokenizer,
                           cpu_cache_size = args.cpu_cache_mb * 1024**2,
                           recurrent_cache_size = 3 * checkpoint_bytes + checkpoint_bytes // 2)
    run_job_ids(capped_gen, a_ids, 40)
    for p in flood_prompts:
        gen_prompt(capped_gen, p)
    _, eos_c = run_job_ids(capped_gen, a_ids, 8)
    capped_stats = capped_gen.get_cache_stats()
    check("stats: evicted-checkpoint capped restores detected",
          capped_stats["gpu_cache"]["recurrent_capped_evicted"] > 0,
          f"recurrent_capped_evicted = {capped_stats['gpu_cache']['recurrent_capped_evicted']}, "
          f"uncovered = {capped_stats['gpu_cache']['recurrent_uncovered']}, "
          f"cached_tokens = {eos_c['cached_tokens']}")
    check("stats: recurrent evictions aged",
          capped_stats["recurrent_cache"]["evicted_count"] > 0,
          f"recurrent_cache = {capped_stats['recurrent_cache']}")

    print(f"\n -- {checks_passed} passed, {checks_failed} failed")
    sys.exit(1 if checks_failed else 0)


if __name__ == "__main__":
    main()
