"""
Tests for asynchronous recurrent tail stashing: background finalization of the pinned->pageable clone
(RecurrentCache worker) and event-gated recurrent slot release (side-stream D2H, Cache.release_state_deferred).

Unit tests (no model required):
1. Cross-RecurrentCache staging guard: two caches sharing one Cache's pinned staging buffers must serialize
   through the shared guard, or the second put() overwrites staging mid-clone.
2. Event-gated slot reclaim state machine: polling reclaims only fired slots and never blocks; backpressure
   blocks on the oldest pending release and frees exactly one slot.
3. Failure paths synchronize: a put() failure after staging joins/synchronizes before propagating, and a
   partial staging failure synchronizes the side streams it started, so cleanup's finally-block slot release
   can never recycle a slot still being read.

Integration (requires a hybrid SWA model, e.g. Gemma-4):
4. Pending-join: a follow-up arriving while its conversation's tail stash is still finalizing must join the
   pending future before restoring, verified with a slowed clone and a reader-side spy.
5. Single-slot lifecycle: with one recurrent slot, a second thread's allocation must ride the blocking
   backpressure path to claim the first thread's slot, and the first thread's stash must survive the slot
   reuse (byte-matched against cold full prefill).

The unit tests are plain pytest tests:

    pytest tests/test_tail_async.py

The integration tests run in script mode (units included):

    python tests/test_tail_async.py --model <exl3 model dir>
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


def test_staging_guard():
    """
    Multiple RecurrentCache instances can share one Cache and therefore its pinned staging buffers. put()
    must join the staging guard on the shared Cache before restaging — a guard held only per RecurrentCache
    lets a second cache overwrite staging while the first one's worker is still cloning from it.
    """
    print(" -- Unit: cross-RecurrentCache staging guard")
    import exllamav3.cache.recurrent as rec
    import time as time_mod

    class _Dummy:
        loaded_tp = False

    class _DummyState:
        def __init__(self, cache, position):
            self.cache = cache
            self.position = position
        def stash(self, pinned_staging = False):
            stashed = {"position": self.position, "checkpoint_size": 8,
                       ("layer", 0): torch.full((4,), float(self.position))}
            return (stashed, []) if pinned_staging else stashed

    shared_cache = _Dummy()
    order = []
    orig_finalize = rec._finalize_stash
    def slow_finalize(stashed, events = None):
        time_mod.sleep(0.3)
        order.append(("clone_done", stashed["position"]))
        orig_finalize(stashed, events)
    rec._finalize_stash = slow_finalize
    try:
        rc_a = rec.RecurrentCache(_Dummy(), max_size = 1024)
        rc_b = rec.RecurrentCache(_Dummy(), max_size = 1024)
        rc_a.put(b"a", _DummyState(shared_cache, 1), pinned_staging = True)
        rc_b.put(b"b", _DummyState(shared_cache, 2), pinned_staging = True)
        order.append(("put_b_done", None))
        rc_b.drain()
        rc_a.drain()
        assert ("clone_done", 1) in order and order.index(("clone_done", 1)) < order.index(("put_b_done", None)), \
            f"second cache's put must join the shared staging guard: order = {order}"
    finally:
        rec._finalize_stash = orig_finalize


def test_slot_reclaim():
    """
    State machine for event-gated slot release: slots return to the pool only when their stash events have
    fired; polling never blocks; backpressure (_ensure_free_slot) blocks on the oldest pending release and
    frees exactly one slot.
    """
    print(" -- Unit: event-gated slot reclaim")
    from types import SimpleNamespace
    from exllamav3.cache.cache import Cache

    class _FakeEvent:
        def __init__(self):
            self.fired = False
            self.sync_calls = 0
        def query(self):
            return self.fired
        def synchronize(self):
            self.sync_calls += 1
            self.fired = True

    pool = SimpleNamespace(free_list = __import__("collections").deque(), pending_slot_releases = [])
    pool.reclaim_slots = lambda **kw: Cache.reclaim_slots(pool, **kw)
    ev_a, ev_b = _FakeEvent(), _FakeEvent()
    Cache.release_state_deferred(pool, SimpleNamespace(slot = 0), [ev_a])
    Cache.release_state_deferred(pool, SimpleNamespace(slot = 1), [ev_b])

    Cache.reclaim_slots(pool)
    assert len(pool.free_list) == 0 and len(pool.pending_slot_releases) == 2, "unfired events keep slots pending"

    ev_b.fired = True
    Cache.reclaim_slots(pool)
    assert list(pool.free_list) == [1] and len(pool.pending_slot_releases) == 1, "polling reclaims only fired slots"

    pool.free_list.clear()
    Cache._ensure_free_slot(pool)
    assert ev_a.sync_calls == 1 and list(pool.free_list) == [0] and not pool.pending_slot_releases, \
        f"backpressure must block on the oldest pending release: sync_calls = {ev_a.sync_calls}, free = {list(pool.free_list)}"


def test_stash_failure_synchronizes():
    """
    Once a pinned stash has returned events, any later put() failure must synchronize them before propagating:
    cleanup releases the state in a finally block and otherwise could recycle a slot still being read.
    """
    print(" -- Unit: failed stash setup synchronizes queued events")
    import exllamav3.cache.recurrent as rec

    class _Dummy:
        loaded_tp = False

    class _FakeEvent:
        def __init__(self):
            self.sync_calls = 0
        def synchronize(self):
            self.sync_calls += 1

    event = _FakeEvent()
    shared_cache = _Dummy()

    class _DummyState:
        cache = shared_cache
        def stash(self, pinned_staging = False):
            stashed = {"position": 1, "checkpoint_size": 8,
                       ("layer", 0): torch.zeros(1)}
            return (stashed, [event]) if pinned_staging else stashed

    # max_size smaller than one checkpoint fails during eviction, after state.stash() returned its events
    rc = rec.RecurrentCache(_Dummy(), max_size = 0)
    try:
        rc.put(b"oversized", _DummyState(), pinned_staging = True)
    except (AssertionError, KeyError):
        pass
    else:
        raise AssertionError("oversized recurrent stash did not raise")
    assert event.sync_calls == 1, \
        f"failed recurrent stash must synchronize queued events: sync_calls = {event.sync_calls}"

    class _FakeSide:
        def __init__(self):
            self.sync_calls = 0
        def wait_stream(self, other):
            pass
        def synchronize(self):
            self.sync_calls += 1

    class _FakeStreamContext:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

    class _Layer:
        device = "cuda:0"
        def __init__(self, fail = False):
            self.fail = fail
        def stash(self, *args, **kwargs):
            if self.fail:
                raise RuntimeError("injected staging failure")
            return torch.zeros(1)

    class _LayerCache:
        def get_all_recurrent_layers(self):
            return {(0, 0): _Layer(), (1, 0): _Layer(fail = True)}

    fake_side = _FakeSide()
    orig_stream_ctor = rec.torch.cuda.Stream
    orig_current_stream = rec.torch.cuda.current_stream
    orig_stream_context = rec.torch.cuda.stream
    rec.torch.cuda.Stream = lambda **kwargs: fake_side
    rec.torch.cuda.current_stream = lambda device = None: object()
    rec.torch.cuda.stream = lambda side: _FakeStreamContext()
    try:
        try:
            rec.stash_recurrent_layers(_LayerCache(), 0, pinned_staging = True)
        except RuntimeError as exc:
            assert str(exc) == "injected staging failure"
        else:
            raise AssertionError("partial recurrent staging did not raise")
    finally:
        rec.torch.cuda.Stream = orig_stream_ctor
        rec.torch.cuda.current_stream = orig_current_stream
        rec.torch.cuda.stream = orig_stream_context
    assert fake_side.sync_calls == 1, \
        f"partial recurrent staging must synchronize its side stream: sync_calls = {fake_side.sync_calls}"


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
    ap.add_argument("--model", required = True,
                    help = "exl3 model directory of a hybrid-attention model with recurrent SWA layers "
                           "(e.g. Gemma-4)")
    ap.add_argument("--cache_size", type = int, default = 8192)
    ap.add_argument("--max_batch_size", type = int, default = 16,
                    help = "Recurrent state slots per cache; the test itself is batch-1, so this can be "
                           "lowered to fit large hybrid models on smaller GPUs")
    args = ap.parse_args()

    for unit in (test_staging_guard, test_slot_reclaim, test_stash_failure_synchronizes):
        try:
            unit()
            check(unit.__name__, True)
        except AssertionError as e:
            check(unit.__name__, False, str(e))

    print(f" -- Loading model: {args.model}")
    config = Config.from_directory(args.model)
    model = Model.from_config(config, swa_full = False)
    cache_main = Cache(model, max_num_tokens = args.cache_size, max_batch_size = args.max_batch_size)
    cache_ctrl = Cache(model, max_num_tokens = args.cache_size, max_batch_size = args.max_batch_size)
    # Single recurrent slot: forces the event-gated slot-release backpressure path in hybrid mode
    cache_slot = Cache(model, max_num_tokens = args.cache_size, max_batch_size = 1)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)
    sampler = GreedySampler()

    generator = Generator(model = model, cache = cache_main, tokenizer = tokenizer)
    if generator.recurrent_cache is None:
        print(" -- Model has no recurrent states; tail checkpoints do not apply. Skipping integration.")
        sys.exit(1 if checks_failed else 0)
    if not getattr(generator.cache.recurrent_state_cls, "guaranteed_rollback", 0):
        print(" -- Recurrent state has no guaranteed rollback (e.g. GDN); tail checkpoints keep default")
        print("    behavior on this architecture by design. Skipping integration.")
        sys.exit(1 if checks_failed else 0)

    prompt_a = make_long_prompt(
        tokenizer,
        "The migration of songbirds across continents follows magnetic fields and stellar cues.", 6)
    a_ids = tokenizer.encode(prompt_a)
    extra_ids = tokenizer.encode(" Now summarize the above in one sentence.", add_bos = False)
    print(f" -- prompt A: {a_ids.shape[-1]} tokens ({a_ids.shape[-1] // PAGE_SIZE} full pages)")

    def drive_enqueued_job(gen, cleanup_tail = True, leave_deferred = False):
        out, eos_r = [], None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                t = r.get("token_ids")
                if t is not None:
                    out.append(t)
                if r.get("eos"):
                    eos_r = r
            if eos_r is not None and leave_deferred:
                break
            if eos_r is not None and not cleanup_tail:
                # A cold correctness reference does not need to populate the recurrent cache. Release its
                # deferred state without allocating another model-sized pinned tail-staging buffer.
                gen.clear_queue()
                break
        out_ids = torch.cat(out, dim = -1) if out else torch.empty((1, 0), dtype = torch.long)
        return out_ids, eos_r

    def run_job_ids(gen, ids, max_new, cleanup_tail = True, leave_deferred = False):
        # Token-level Job API: continuation prompts are built from exact ids, avoiding tokenization-seam
        # ambiguity at the prompt/response join
        job = Job(input_ids = ids, max_new_tokens = max_new, sampler = sampler)
        gen.enqueue(job)
        return drive_enqueued_job(gen, cleanup_tail, leave_deferred)

    # Background-clone join: slow the pinned->pageable clone so the follow-up request provably arrives while
    # its thread's tail stash is still pending. The follow-up's enqueue() submits the clone and the same
    # request then restores the entry, so wait_stash_ready must block and join.
    print(" -- Pending-join: restore while background finalization is in flight")
    import exllamav3.cache.recurrent as recurrent_mod
    import time as time_mod
    orig_clone = recurrent_mod.clone_staged_tensors
    def slow_clone(tensors):
        time_mod.sleep(0.25)
        return orig_clone(tensors)
    recurrent_mod.clone_staged_tensors = slow_clone
    # Spy on the reader-side join: the warm resume must observe a genuinely pending finalization, otherwise
    # this test cannot distinguish a working join from no join at all
    join_saw_pending = []
    orig_wait = recurrent_mod.wait_stash_ready
    def spy_wait(stashed):
        fut = stashed.get("pending")
        join_saw_pending.append(fut is not None and not fut.done())
        orig_wait(stashed)
    recurrent_mod.wait_stash_ready = spy_wait
    try:
        pend_base = torch.cat([
            a_ids, tokenizer.encode(" Pending-join case follows now.", add_bos = False)], dim = -1)
        gen_n = (pend_base.shape[-1] // PAGE_SIZE + 1) * PAGE_SIZE - pend_base.shape[-1] + 1 + 30
        resp_p, _ = run_job_ids(generator, pend_base, gen_n)
        pend_ids = torch.cat([pend_base, resp_p, extra_ids], dim = -1)
        cold_gen = Generator(model = model, cache = cache_ctrl, tokenizer = tokenizer)
        cold_p, _ = run_job_ids(cold_gen, pend_ids, 12, cleanup_tail = False)
        warm_p, eos_p = run_job_ids(generator, pend_ids, 12)
        pend_boundary = (pend_base.shape[-1] + resp_p.shape[-1]) // PAGE_SIZE * PAGE_SIZE
        check("pending-join: resumed from the tail checkpoint",
              eos_p["cached_tokens"] >= pend_boundary,
              f"cached_tokens = {eos_p['cached_tokens']}, expected >= {pend_boundary}")
        check("pending-join: resumed output matches cold full-prefill",
              torch.equal(warm_p, cold_p),
              f"warm ids = {warm_p.tolist()}\ncold ids = {cold_p.tolist()}")
        check("pending-join: restore joined a genuinely pending finalization",
              any(join_saw_pending),
              f"join_saw_pending = {join_saw_pending} (no restore observed an unfinished clone)")
    finally:
        recurrent_mod.clone_staged_tensors = orig_clone
        recurrent_mod.wait_stash_ready = orig_wait

    # Single-slot lifecycle: make A's release events report pending to nonblocking polls until the allocator
    # explicitly synchronizes them. B must therefore ride the backpressure path when claiming A's only slot;
    # resuming A afterwards also proves its stash retained A's ring bytes across the reuse.
    print(" -- Single-slot lifecycle: blocking backpressure and stash survival across slot reuse")
    slot_gen = Generator(model = model, cache = cache_slot, tokenizer = tokenizer)
    sa_ids = torch.cat([
        a_ids[:, :640], tokenizer.encode(" Single-slot thread A continues here.", add_bos = False)], dim = -1)
    sb_ids = torch.cat([
        a_ids[:, :640], tokenizer.encode(" Single-slot thread B differs entirely.", add_bos = False)], dim = -1)
    gen_a = (sa_ids.shape[-1] // PAGE_SIZE + 1) * PAGE_SIZE - sa_ids.shape[-1] + 1 + 20
    resp_sa, _ = run_job_ids(slot_gen, sa_ids, gen_a, leave_deferred = True)
    job_sb = Job(input_ids = sb_ids, max_new_tokens = 24, sampler = sampler)
    slot_gen.enqueue(job_sb)  # runs A's cleanup and registers its deferred slot release
    check("single-slot: A's slot release was deferred",
          len(cache_slot.pending_slot_releases) == 1,
          f"pending releases = {len(cache_slot.pending_slot_releases)}")

    import threading
    main_thread = threading.current_thread()
    backpressure_syncs = []

    class _PendingUntilAllocatorSync:
        def __init__(self, inner):
            self.inner = inner
            self.allocator_synced = False
        def query(self):
            return self.allocator_synced and self.inner.query()
        def synchronize(self):
            self.inner.synchronize()
            if threading.current_thread() is main_thread:
                self.allocator_synced = True
                backpressure_syncs.append(1)

    # Give the pending-slot queue a wrapped list while leaving the clone future's event list untouched.
    # Polling reports pending until _ensure_free_slot synchronizes the wrappers on the main thread.
    if cache_slot.pending_slot_releases:
        release_slot, release_events = cache_slot.pending_slot_releases[0]
        wrapped_events = [_PendingUntilAllocatorSync(ev) for ev in release_events]
        cache_slot.pending_slot_releases[0] = (release_slot, wrapped_events)
    resp_sb, _ = drive_enqueued_job(slot_gen)
    check("single-slot: allocation used blocking backpressure",
          bool(backpressure_syncs),
          "the only slot was reclaimed without allocator-side event synchronization")
    check("single-slot: second thread generates through the reused slot", resp_sb.shape[-1] > 0)
    cont_sa = torch.cat([sa_ids, resp_sa, extra_ids], dim = -1)
    cold_gen = Generator(model = model, cache = cache_ctrl, tokenizer = tokenizer)
    cold_sa, _ = run_job_ids(cold_gen, cont_sa, 12, cleanup_tail = False)
    warm_sa, eos_sa = run_job_ids(slot_gen, cont_sa, 12)
    sa_boundary = (sa_ids.shape[-1] + resp_sa.shape[-1]) // PAGE_SIZE * PAGE_SIZE
    check("single-slot: thread A resumes from its tail checkpoint",
          eos_sa["cached_tokens"] >= sa_boundary,
          f"cached_tokens = {eos_sa['cached_tokens']}, expected >= {sa_boundary}")
    check("single-slot: resumed output matches cold full-prefill",
          torch.equal(warm_sa, cold_sa),
          f"warm ids = {warm_sa.tolist()}\ncold ids = {cold_sa.tolist()}")

    print(f"\n -- {checks_passed} passed, {checks_failed} failed")
    sys.exit(1 if checks_failed else 0)


if __name__ == "__main__":
    main()
