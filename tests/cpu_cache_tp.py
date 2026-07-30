"""
Integration test for the CPU page cache tier, including tensor-parallel mode. Script-only (not collected
by pytest: everything here needs a real model):

    python tests/cpu_cache_tp.py --model <exl3 model dir>              # single-device tier
    python tests/cpu_cache_tp.py --model <exl3 model dir> --tensor_p   # per-rank pools in TP workers

Verifies, with greedy sampling:

1. GPU prompt cache sanity: an immediate re-run of a multi-page prompt reuses cached pages.
2. Tier restore after eviction: flooding a small GPU cache pushes evicted pages to the tier; a re-run
   restores them, and every restored page's K/V is byte-identical to what was offloaded. The byte
   comparison runs through the same data plane as the feature (local tensor reads, or per-rank worker
   dispatch under TP), so it validates the offload/restore path independently of token outputs.
3. Mixed batch: a restored-prefix job, a fresh job and a recently flooded job generated concurrently
   reproduce their solo outputs.
4. Lifecycle: replacing a tier-backed generator without an intervening collection must not leak or
   double-free the worker-side pools (deferred frees drain at the next tier's allocation).

Output-identity checks are gated by a determinism probe: linear-attention models (e.g. GDN) are not
run-to-run deterministic, so correctness there rests on the byte-level fidelity checks, which do not
depend on determinism. On hybrid recurrent models the restorable prefix is capped by stashed recurrent
checkpoints, so restore counts are asserted loosely (> 0) rather than exactly.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, GreedySampler
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


def mp_test_fetch_page_bytes(local_context, cache_id, page_index):
    """
    Worker-side helper: raw bytes of this rank's shard of one GPU cache page, in deterministic tensor order.
    Runs in the TP worker processes; .cpu() serializes against the rank's stream, so pending tier copies
    ordered before this command are reflected.
    """
    from exllamav3.model.model_tp_fn import _host_cache_shard_tensors
    return b"".join(
        t[page_index].contiguous().cpu().view(torch.uint8).numpy().tobytes()
        for t in _host_cache_shard_tensors(local_context, cache_id)
    )


def fetch_page_bytes(cpc, page_index):
    """
    Raw bytes of one GPU cache page across every cache the tier covers - local tensors directly, TP shards
    via worker dispatch. Byte-compare of store-time vs fetch-time snapshots is the architecture-independent
    fidelity check for the offload/restore data plane.
    """
    parts = []
    for t, _, _, _ in cpc.segments:
        parts.append(t[page_index].contiguous().cpu().view(torch.uint8).numpy().tobytes())
    for model, cache_id, pool_id in cpc.tp_caches:
        for device in model.active_devices:
            parts.append(model.tp_worker_dispatch_single(
                device, mp_test_fetch_page_bytes, (cache_id, page_index)))
    return b"".join(parts)


def install_kv_fidelity_hooks(cpc, max_snapshots = 8):
    """
    Wrap store()/fetch() to snapshot page bytes at offload time and byte-compare them at restore time.
    Returns the dict of phash -> bool comparison results, filled in as restores happen.
    """
    snapshots = {}
    results = {}
    orig_store = cpc.store
    orig_fetch = cpc.fetch

    def snap_store(page, serial, protect = None):
        orig_store(page, serial, protect)
        # The page's GPU contents survive until the claiming job's prefill, which is dispatched after this
        # returns, so the fetch below still observes the exact bytes the store offloaded
        if page.phash in cpc.entries and page.phash not in snapshots and len(snapshots) < max_snapshots:
            snapshots[page.phash] = fetch_page_bytes(cpc, page.page_index)

    def check_fetch(phash, page_index, serial):
        e = orig_fetch(phash, page_index, serial)
        if phash in snapshots:
            results[phash] = fetch_page_bytes(cpc, page_index) == snapshots[phash]
        return e

    cpc.store = snap_store
    cpc.fetch = check_fetch
    return results


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
    ap.add_argument("--max_new_tokens", type = int, default = 32)
    ap.add_argument("--cache_size", type = int, default = 4096)
    ap.add_argument("--cpu_cache_mb", type = int, default = 1024)
    ap.add_argument("--tensor_p", action = "store_true",
                    help = "Load the model tensor-parallel across all visible devices; the tier then "
                           "offloads/restores per-rank cache shards through the TP workers")
    args = ap.parse_args()

    print(f" -- Loading model: {args.model}{' (tensor-parallel)' if args.tensor_p else ''}")
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache_a = Cache(model, max_num_tokens = args.cache_size)
    cache_b = Cache(model, max_num_tokens = args.cache_size)
    model.load(progressbar = True, tensor_p = args.tensor_p)
    tokenizer = Tokenizer.from_config(config)
    sampler = GreedySampler()

    prompt_a = make_long_prompt(
        tokenizer,
        "The migration of songbirds across continents follows magnetic fields and stellar cues.", 6)
    prompt_c = "The most surprising property of liquid helium below the lambda point is"
    flood_prompts = [
        make_long_prompt(tokenizer, f"Flood text number {j} concerns the industrial history of a city.", 5)
        for j in range(10)
    ]
    a_tokens = tokenizer.encode(prompt_a).shape[-1]
    a_pages = a_tokens // PAGE_SIZE
    flood_tokens = sum(tokenizer.encode(p).shape[-1] for p in flood_prompts)
    print(f" -- prompt A: {a_tokens} tokens ({a_pages} full pages); "
          f"flood: {flood_tokens} tokens vs {args.cache_size}-token cache")
    assert flood_tokens > args.cache_size * 1.5, "flood too small to guarantee eviction"

    def gen(generator, prompts):
        completions, results = generator.generate(
            prompt = prompts,
            max_new_tokens = args.max_new_tokens,
            sampler = sampler,
            completion_only = True,
            return_last_results = True,
        )
        if not isinstance(prompts, list):
            return completions, results["cached_tokens"]
        return completions, [r["cached_tokens"] for r in results]

    # Determinism probe: byte-identity of generated text across replays assumes a run-to-run deterministic
    # forward pass. Linear-attention scans are not; their correctness rests on the byte-fidelity checks.
    if model.caps.get("linear_attn"):
        deterministic = False
        print(" -- Determinism probe: skipped (linear-attention scan is not run-to-run deterministic)")
    else:
        probe_gen = Generator(model = model, cache = cache_b, tokenizer = tokenizer)
        det_a, _ = gen(probe_gen, prompt_c)
        del probe_gen
        probe_gen = Generator(model = model, cache = cache_b, tokenizer = tokenizer)
        det_b, _ = gen(probe_gen, prompt_c)
        del probe_gen
        deterministic = det_a == det_b
        print(f" -- Determinism probe: model is {'' if deterministic else 'NOT '}run-to-run deterministic")

    tier_gen = Generator(model = model, cache = cache_a, tokenizer = tokenizer,
                         cpu_cache_size = args.cpu_cache_mb * 1024**2)
    cpc = tier_gen.cpu_page_cache
    fidelity = install_kv_fidelity_hooks(cpc)

    print(" -- Tier generator: reference run")
    ref_a, cached_first = gen(tier_gen, prompt_a)
    ref_c, _ = gen(tier_gen, prompt_c)
    check("first run of prompt A has no cached prefix", cached_first == 0,
          f"cached_tokens = {cached_first}")

    print(" -- Tier generator: immediate re-run (GPU prompt cache sanity)")
    rerun_a, cached_rerun = gen(tier_gen, prompt_a)
    if tier_gen.recurrent_cache is None:
        check("immediate re-run reuses GPU-cached pages", cached_rerun >= (a_pages - 1) * PAGE_SIZE,
              f"cached_tokens = {cached_rerun}, expected >= {(a_pages - 1) * PAGE_SIZE}")
    else:
        check("immediate re-run reuses GPU-cached pages (hybrid: capped by checkpoints)", cached_rerun > 0,
              f"cached_tokens = {cached_rerun}")
    if deterministic:
        check("immediate re-run output identical", rerun_a == ref_a,
              f"got: {rerun_a!r}\nref: {ref_a!r}")

    print(f" -- Tier generator: flooding cache with {len(flood_prompts)} prompts")
    flood_refs = []
    for p in flood_prompts:
        fr, _ = gen(tier_gen, p)
        flood_refs.append(fr)
    check("flood evictions pushed pages to the tier", cpc.metrics["pushes"] > 0,
          f"pushes = {cpc.metrics['pushes']}")

    print(" -- Tier generator: post-flood re-run of prompt A")
    restored_a, cached_restored = gen(tier_gen, prompt_a)
    check("post-flood re-run restored pages from the tier", cpc.metrics["restores"] > 0,
          f"restores = {cpc.metrics['restores']}, cached_tokens = {cached_restored}")
    if deterministic:
        check("post-flood output identical to reference", restored_a == ref_a,
              f"got: {restored_a!r}\nref: {ref_a!r}")
    check("all restored pages byte-identical to their offloaded contents",
          len(fidelity) > 0 and all(fidelity.values()),
          f"fidelity = {fidelity}")

    print(" -- Tier generator: mixed batch after flood")
    batch, _ = gen(tier_gen, [prompt_a, prompt_c, flood_prompts[0]])
    if deterministic:
        check("mixed batch: restored-prefix job matches solo output", batch[0] == ref_a)
        check("mixed batch: fresh job matches solo output", batch[1] == ref_c)
        check("mixed batch: recent flood job matches solo output", batch[2] == flood_refs[0])
    else:
        check("mixed batch: all jobs produced output", all(len(b) > 0 for b in batch))

    # Lifecycle: replace the tier-backed generator without a prompt collection. The fidelity hooks above
    # form a reference cycle (cpc.store -> closure -> cpc), so the old tier's finalizer only runs at a
    # cyclic GC point - possibly after the replacement tier has already allocated its pools. Its deferred
    # frees are keyed by its own pool generation and must never touch the new tier's pools. Regression for
    # double-free/stale-pool bugs across store generations.
    print(" -- Lifecycle: replace tier-backed generator, old tier finalized late")
    import gc
    import weakref
    cpc_ref = weakref.ref(cpc)
    del tier_gen, cpc
    tier_gen2 = Generator(model = model, cache = cache_b, tokenizer = tokenizer,
                          cpu_cache_size = args.cpu_cache_mb * 1024**2)
    lc_a, _ = gen(tier_gen2, prompt_a)
    for p in flood_prompts[:4]:
        gen(tier_gen2, p)
    # Finalize the old tier now; its frees queue against dead pools only. The alloc thread briefly holds a
    # strong reference once per polling cycle, so allow a couple of retries before declaring it retained.
    import time as time_mod
    for _ in range(4):
        gc.collect()
        if cpc_ref() is None:
            break
        time_mod.sleep(0.6)
    check("dead tier is collectable (alloc thread must not retain it)", cpc_ref() is None,
          "the background pinning thread is keeping the abandoned CPUPageCache alive; its pinned slabs "
          "and any worker-side pools leak until process exit")
    lc_a2, _ = gen(tier_gen2, prompt_a)   # next dispatches drain the frees, then keep working
    check("replacement tier generator stores and restores",
          tier_gen2.cpu_page_cache.metrics["pushes"] > 0,
          f"pushes = {tier_gen2.cpu_page_cache.metrics['pushes']}")
    if deterministic:
        check("replacement tier generator output correct", lc_a == lc_a2 == restored_a)

    # Abandon a tier mid-preallocation: the pinning worker holds no reference to the (cyclic) cache, so
    # collection must not be deferred until the full budget is pinned. _make_slab is gated after the first
    # slab, guaranteeing preallocation is provably in flight when the generator is dropped; the finalizer
    # must stop the worker, which then exits without pinning to capacity.
    print(" -- Lifecycle: abandon tier mid-preallocation")
    import threading
    import exllamav3.generator.cpu_cache as cc_mod
    if tier_gen2.cpu_page_cache._alloc_state is None:
        print("        (pure-TP tier: no local slabs to preallocate; covered by the single-device run)")
    else:
        del tier_gen2
        gate = threading.Event()
        made = []
        orig_make = cc_mod._make_slab
        def gated_make(slot_size, layouts):
            if made:
                gate.wait(timeout = 30)
            made.append(1)
            return orig_make(slot_size, layouts)
        cc_mod._make_slab = gated_make
        try:
            g3 = Generator(model = model, cache = cache_a, tokenizer = tokenizer,
                           cpu_cache_size = args.cpu_cache_mb * 1024**2)
            state3 = g3.cpu_page_cache._alloc_state
            thread3 = g3.cpu_page_cache._alloc_thread
            ref3 = weakref.ref(g3.cpu_page_cache)
            del g3
            collected = False
            for _ in range(4):
                gc.collect()
                if ref3() is None:
                    collected = True
                    break
                time_mod.sleep(0.3)
            check("tier abandoned mid-preallocation is collectable", collected,
                  "the pinning worker is retaining the cache; its full budget would pin before cleanup")
            check("finalizer stopped the pinning worker", state3.stopped)
            n_before = len(made)
            gate.set()
            thread3.join(timeout = 10)
            check("worker exited without preallocating to capacity",
                  not thread3.is_alive() and len(made) <= n_before + 1,
                  f"thread alive = {thread3.is_alive()}, slabs made = {len(made)}, "
                  f"target = {state3.target}")
        finally:
            cc_mod._make_slab = orig_make
            gate.set()

    print(f"\n -- {checks_passed} passed, {checks_failed} failed")
    sys.exit(1 if checks_failed else 0)


if __name__ == "__main__":
    main()
