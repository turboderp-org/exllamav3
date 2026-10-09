import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from collections import deque
from types import SimpleNamespace
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.pagetable import DraftRing
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX


# The drafter's window is rebuilt (refilled) after a prefix resume or a deep rewind:
# Job.refill_draft_window runs the target over the window's tokens (an idempotent rewrite
# of the main K/V) and projects the exported hidden states into the ring. The rebuild is
# skipped when the window is provably fine: a shallow rewind never reaches the ring's
# wrap floor, a replay starting at or below the window projects it itself, and on a
# resume the foreign-row ceiling (Sequence.draft_foreign_ceiling) can already sit at
# or below the window start. These tests pin the resume arming in Sequence.allocate_pages
# (through the real page table, including the requeue leg), the resume call sites in
# Job.prefill (once per resume, not per chunk), the rewind guard (depth against the wrap
# floor, checked against a simulated ring, plus the replay-coverage skip), the throwaway
# state's lifecycle and the state-pool/cost bail-outs, the image-span guard, the mrope
# table regrow, and the rewind call site driven through real banned-string rewinds in
# both directions (shallow skip, deep rebuild).


class NoteFakePage:
    def __init__(self, page_index, kv_position = 0, prev_hash = None):
        self.page_index = page_index
        self.kv_position = kv_position
        self.prev_hash = prev_hash
        self.phash = b"\x00" * 16
        self.can_revert = True
        self.sequence = torch.zeros((1, PAGE_SIZE), dtype = torch.long)

    def update_hash(self, h = None):
        # rewind_checkpoint invalidates completed pages below the rewind target this
        # way (exercised by the deep-rewind end-to-end test)
        self.phash = h if h is not None else b"\x01" * 16


class RecordingTierCache:
    def __init__(self):
        self.copies = []
        # Like the real Cache.free_list: refill_draft_window checks a spare slot exists
        # before allocating (at a full batch it is empty and the refill bails out)
        self.free_list = deque([0])

    def copy_page(self, other, src_page, dst_page, num_tokens):
        self.copies.append((src_page, dst_page, num_tokens))

    def get_new_state(self):
        return SimpleNamespace(free = lambda: None)


class FakeRecurrentState:
    # The skip loop's guard (0 <= cp_pos <= kv_position) breaks on the first page while
    # the restored state sits at or below kv_position - every round of a recurrent
    # resume - and the partial-page copy is skipped for recurrent models, so the
    # prefill-side arming site can't fire there
    def __init__(self, position):
        self.position = position

    def rewind(self, n):
        self.position -= n

    def free(self):
        pass


class PrefillTargetModel:
    # Records each prefill chunk and exports position-tagged hidden states (each row carries
    # its absolute position), so the refill's projection into the ring - including the span
    # clip - is observable from the outside. export=False models a target that runs the
    # forward but exports no states: the refill then neither projects nor credits the
    # ceiling, which is what makes the completion-site guard load-bearing
    caps = {}

    def __init__(self, export = True):
        self.calls = []
        self.inputs = []
        self.export = export
        # Only Job.refill_draft_window's alt_rope regrow branch reaches for g_rope; the
        # stub records the (ids, embeddings, seq_len) call and hands back a longer table
        self.g_rope = SimpleNamespace(calls = [], freqs = None, get_mrope_freqs = self._mrope)

    def _mrope(self, ids, embeddings, seq_len):
        self.g_rope.calls.append((ids, embeddings, seq_len))
        return self.g_rope.freqs, None

    def prefill(self, *, input_ids, params):
        start = params["cache_seqlens"][0].item()
        positions = torch.arange(start, start + input_ids.shape[-1], dtype = torch.float32)
        if self.export:
            params["export_states"] = [positions.view(1, -1, 1)]
        # Like a real recurrent state, the state advances to the end of the forward
        for st in params.get("recurrent_states") or []:
            st.position = start + input_ids.shape[-1]
        self.calls.append(params)
        self.inputs.append(input_ids.clone())

    forward = prefill


class PrefillDraftModel:
    draft_verifier_params = {}

    def __init__(self):
        self.calls = []

    def update_kv_from_target(self, *, target_hidden, cache, params):
        self.calls.append({"target_hidden": target_hidden, "cache": cache, "params": params})


class TestRefillCallSites:

    RING = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047, spec_rows = 17)

    def make_prefill_job(self, prompt_tokens, *, page_kvs = None, all_pages = ()):
        from exllamav3.generator.job import Job
        ids = torch.arange(1, prompt_tokens + 1, dtype = torch.long).view(1, prompt_tokens)
        job = Job(input_ids = ids, embeddings = [])
        seq = job.sequences[0]
        page_count = (prompt_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        kvs = page_kvs if page_kvs is not None else [0] * page_count
        seq.allocated_pages = [NoteFakePage(i, kvs[i]) for i in range(page_count)]
        seq.page_hashes = []  # empty: prefill never re-hashes a completed page here
        seq.block_index_tensor = torch.tensor([list(range(page_count))], dtype = torch.int32)
        # What allocate_pages does for a ring-backed sequence: bind the ring to the slot
        seq.draft_ring = self.RING
        seq.draft_slot = 0
        seq.draft_block_index_tensor = self.RING.table(0, page_count)
        job.pagetable = SimpleNamespace(all_pages = list(all_pages))
        job.generator = SimpleNamespace(
            model = PrefillTargetModel(),
            max_chunk_size = 4096,
            recurrent_cache = None,
            cache = RecordingTierCache(),
            draft_model = PrefillDraftModel(),
            draft_cache = RecordingTierCache(),
            draft_ring = self.RING,
            dflash_draft = True,
            mtp_draft = False,
        )
        return job

    def test_cached_resume_refills_the_draft_window(self):
        # The first two prompt pages are cached: allocation consumes them (the resume
        # point sits at 2*PAGE_SIZE, the refill armed, the resume point the foreign
        # ceiling). The window at completion starts below the resume point, so the
        # refill runs the target over the window, once, at prefill completion (a
        # per-chunk refill would add a re-run at every later chunk tip)
        job = self.make_prefill_job(1000)
        seq = job.sequences[0]
        # What allocate_pages leaves behind for a two-page resume:
        seq.kv_position = 2 * PAGE_SIZE
        seq.draft_refill_pending = True
        seq.raise_draft_floor(2 * PAGE_SIZE)
        job.prefill([])
        assert seq.kv_position >= 2 * PAGE_SIZE
        # exactly one refill: the chunk forward plus the completion refill, nothing else
        seqlens = sorted(p["cache_seqlens"][0].item() for p in job.generator.model.calls)
        assert seqlens == [0, 2 * PAGE_SIZE]

    def test_fully_cached_resume_refills_once(self):
        # Every prompt page is cached and full - a multiple-of-page prompt, so the
        # allocator can actually present this state (a short prompt's tail page is
        # never full): allocation consumes the whole prompt, so prefill never runs a
        # chunk forward and the fully-cached branch is the only refill call site that
        # can fire. It must fire exactly once: the branch re-enters on every later
        # prefill round (the sequence never completes without a forward), and a
        # per-round refill would re-run the target over the window for the whole
        # generation
        job = self.make_prefill_job(4 * PAGE_SIZE)
        seq = job.sequences[0]
        # What allocate_pages leaves behind for a fully-cached resume:
        seq.kv_position = 4 * PAGE_SIZE
        seq.draft_refill_pending = True
        seq.raise_draft_floor(4 * PAGE_SIZE)
        job.prefill([])
        assert seq.kv_position >= len(seq.sequence_ids) - 1
        # the refill's target forward covers the window below the resume point
        assert any(p["cache_seqlens"][0].item() == 0 for p in job.generator.model.calls)
        calls = len(job.generator.model.calls)
        # a later prefill round (the decode path re-enters prefill) must not refill again
        job.prefill([])
        assert len(job.generator.model.calls) == calls

    def test_fully_cached_resume_without_exported_states_refills_once(self):
        # The behavioral pin for the completion-site guard: a target that exports no
        # states gives the rebuild no ceiling credit, so the ceiling exit stays
        # unavailable for the same window and an UNGUARDED branch would re-run the
        # target over the window on every re-entered prefill round. The guard (guard
        # on the flag, clear it after the call) is the only thing that keeps the
        # refill to one forward
        job = self.make_prefill_job(4 * PAGE_SIZE)
        job.generator.model = PrefillTargetModel(export = False)
        seq = job.sequences[0]
        seq.kv_position = 4 * PAGE_SIZE
        seq.draft_refill_pending = True
        seq.raise_draft_floor(4 * PAGE_SIZE)
        for _ in range(3):
            job.prefill([])
        # one refill forward over the window, not one per re-entered round
        seqlens = [p["cache_seqlens"][0].item() for p in job.generator.model.calls]
        assert seqlens == [0]
        # the rebuild projected nothing: no states to project, no ceiling credit
        assert job.generator.draft_model.calls == []
        assert seq.draft_foreign_ceiling == 4 * PAGE_SIZE

    def test_recurrent_resume_refills_at_completion(self):
        # A recurrent resume restores the state at the resume point, so the skip
        # loop's guard breaks on the first page every round and the prefill-side
        # arming sites never fire. The flag armed at allocation (pinned in
        # TestRefillArming) must still be consumed exactly once, at prefill
        # completion, or the window keeps the slot's previous rows for the whole
        # generation. The refill runs on a THROWAWAY state (the job's own state
        # sits at the resume point and must not advance) and must return it to
        # the pool afterwards: a dropped free leaks one state per resume/deep
        # rewind out of a pool sized to max_batch_size
        job = self.make_prefill_job(1000, page_kvs = [PAGE_SIZE, PAGE_SIZE, PAGE_SIZE, 0])
        seq = job.sequences[0]
        # What Job.allocate_pages leaves behind for a three-page recurrent resume:
        # the resume point armed pending AND marked the rows below it foreign
        seq.kv_position = 3 * PAGE_SIZE
        seq.draft_refill_pending = True
        seq.raise_draft_floor(3 * PAGE_SIZE)
        job.recurrent_state = FakeRecurrentState(3 * PAGE_SIZE)
        job.generator.recurrent_cache = RecordingTierCache()
        freed = []
        job.find_recurrent_stash = lambda pos: {"position": 2 * PAGE_SIZE}
        job.generator.cache.new_from_stashed = lambda stashed, pos: (
            SimpleNamespace(free = lambda: freed.append(1), position = pos))
        job.prefill([])
        assert seq.prefill_complete
        # the tail forward plus the single completion refill (re-running from the
        # stash at 2*PAGE_SIZE), nothing else
        seqlens = sorted(p["cache_seqlens"][0].item() for p in job.generator.model.calls)
        assert seqlens == [2 * PAGE_SIZE, 3 * PAGE_SIZE]
        # the refill ran on the throwaway state, and only on it: the job's own state
        # sits where the tail forward left it (the prompt end), not further
        refill_call = next(p for p in job.generator.model.calls
                           if p["cache_seqlens"][0].item() == 2 * PAGE_SIZE)
        assert refill_call["recurrent_states"][0] is not job.recurrent_state
        assert job.recurrent_state.position == 999
        assert freed == [1]
        calls = len(job.generator.model.calls)
        # a later prefill round must not refill again
        job.prefill([])
        assert len(job.generator.model.calls) == calls

    def test_partial_page_copy_refills_the_draft_window(self):
        # A cached page shares the prompt's first 100 tokens: their K/V is copied into
        # the main cache, but the draft rows belong to the other sequence's ring. The
        # refill rewrites the window from the target's hidden states
        prompt = 1000
        ids = torch.arange(1, prompt + 1, dtype = torch.long)
        candidate = NoteFakePage(90, kv_position = 100, prev_hash = None)
        candidate.sequence[0, :100] = ids[:100]
        job = self.make_prefill_job(prompt, all_pages = [candidate])
        job.prefill([])
        # only the main cache is copied for a ring-backed draft cache; the draft pool is
        # sequence-private and stays untouched
        assert job.generator.cache.copies == [(90, 0, 100)]
        assert job.generator.draft_cache.copies == []
        # the refill's target forward covers the window below the resume point
        assert any(p["cache_seqlens"][0].item() == 0 for p in job.generator.model.calls)
        # exactly one refill: the chunk forward plus the completion refill
        seqlens = sorted(p["cache_seqlens"][0].item() for p in job.generator.model.calls)
        assert seqlens == [0, 100]

    def test_fresh_multichunk_prefill_never_refills(self):
        # A fresh prefill has no cached prefix: every chunk forward projects its own
        # ring rows, so no refill may fire (the per-chunk refill regression would
        # re-run the target over the window on every chunk after the first)
        job = self.make_prefill_job(10000)
        job.generator.max_chunk_size = 4096
        seq = job.sequences[0]
        while seq.kv_position < len(seq.sequence_ids) - 1:
            job.prefill([])
        # the three chunk forwards, nothing else
        seqlens = sorted(p["cache_seqlens"][0].item() for p in job.generator.model.calls)
        assert seqlens == [0, 4096, 8192]

    def test_resume_below_the_window_never_refills(self):
        # A cached prefix is consumed at allocation: the resume is armed and the
        # foreign ceiling sits at the resume point (512), far below the window start
        # at completion (7953). Every window row is this job's own projection, so the
        # completion refill must not fire - the redundant re-run is the whole window
        # of target forward per prefix-cache hit (a per-request TTFT cost)
        job = self.make_prefill_job(10000)
        job.generator.max_chunk_size = 4096
        seq = job.sequences[0]
        seq.kv_position = 2 * PAGE_SIZE
        seq.draft_refill_pending = True
        seq.raise_draft_floor(2 * PAGE_SIZE)
        while seq.kv_position < len(seq.sequence_ids) - 1:
            job.prefill([])
        # the three chunk forwards (resume point + n*max_chunk_size), nothing else
        seqlens = sorted(p["cache_seqlens"][0].item() for p in job.generator.model.calls)
        assert seqlens == [2 * PAGE_SIZE, 2 * PAGE_SIZE + 4096, 2 * PAGE_SIZE + 2 * 4096]
        assert seq.draft_foreign_ceiling == 2 * PAGE_SIZE

    def test_resume_inside_the_window_refills_once(self):
        # Thirty-two cached pages (8192 tokens) reach INSIDE the window (start 7952
        # at the final tip): the rows between the window start and the resume point
        # are the slot's previous sequence's, so the refill must fire exactly once,
        # at completion, over the window
        job = self.make_prefill_job(10000)
        job.generator.max_chunk_size = 4096
        seq = job.sequences[0]
        # What allocate_pages leaves behind for a resume at 8192:
        seq.kv_position = 32 * PAGE_SIZE
        seq.draft_refill_pending = True
        seq.raise_draft_floor(32 * PAGE_SIZE)
        while seq.kv_position < len(seq.sequence_ids) - 1:
            job.prefill([])
        tip = len(seq.sequence_ids) - 1
        # the rebuild credited the window back: the ceiling now sits at the window
        # start, so a later completion-branch re-entry recognizes it as covered
        assert seq.draft_foreign_ceiling == tip - 2047
        # the tail chunk forward plus the single completion refill over the window
        seqlens = sorted(p["cache_seqlens"][0].item() for p in job.generator.model.calls)
        assert seqlens == [tip - 2047, 32 * PAGE_SIZE]
        # the single completion refill projects the window into the ring
        draft = job.generator.draft_model
        assert len(draft.calls) == 2  # chunk plus refill
        assert draft.calls[-1]["params"]["cache_seqlens"].tolist() == [tip - 2047]


class TestRefillArming:
    # Sequence.allocate_pages arms the completion refill for any cached resume. The
    # prefill-side arming site (partial-page copy) can't fire on a recurrent target
    # (the copy is skipped for recurrent models), so the resume must be armed where
    # it is known, at allocation. Whether the armed refill then RUNS is the
    # foreign-row ceiling's decision (Sequence.draft_foreign_ceiling), pinned below
    # through the real refill entry.
    RING = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047, spec_rows = 17)

    def make_seq(self, page_count, cached_pages):
        from exllamav3.generator.pagetable import Sequence
        prompt = page_count * PAGE_SIZE
        ids = torch.arange(1, prompt + 1, dtype = torch.long).view(1, prompt)
        seq = Sequence(ids, ids)
        seq.page_hashes = list(range(page_count))
        seq.new_unique_pages = 0
        pages = [NoteFakePage(i, PAGE_SIZE if i < cached_pages else 0) for i in range(page_count)]
        return seq, pages

    def allocate(self, seq, pages, cached_pages):
        pagetable = SimpleNamespace(
            draft_ring = self.RING,
            draft_slots = deque([0]),
            allocate_pages = lambda *a: (pages, cached_pages * PAGE_SIZE, cached_pages, 0),
        )
        seq.allocate_pages(pagetable, None)

    def test_cached_resume_arms_the_refill(self):
        seq, pages = self.make_seq(4, 3)
        self.allocate(seq, pages, 3)
        assert seq.kv_position == 3 * PAGE_SIZE
        assert seq.draft_refill_pending
        assert seq.draft_foreign_ceiling == 3 * PAGE_SIZE

    def test_real_page_table_arms_resume_and_requeue(self):
        # The stubbed allocate above pins the Sequence-side reaction; this pins the
        # real PageTable that produces it: a cached resume arms the refill and marks
        # the resume point foreign, and a requeued sequence - its slot released with
        # its pages - reallocates over the same sealed prefix, takes a slot again and
        # is armed once more
        from exllamav3.generator.pagetable import PageTable, Sequence
        prompt = 40 * PAGE_SIZE
        pt = PageTable(SimpleNamespace(), SimpleNamespace(max_num_tokens = 128 * PAGE_SIZE))
        pt.draft_ring = self.RING
        pt.draft_slots = deque(range(self.RING.num_slots))

        def make_seq():
            ids = torch.arange(1, prompt + 1, dtype = torch.long).view(1, prompt)
            seq = Sequence(ids[0].clone(), ids[0].clone())
            seq.prepare(has_prefix_token = False, max_new_tokens = PAGE_SIZE)
            return seq

        seq = make_seq()
        seq.allocate_pages(pt, None)
        assert not seq.draft_refill_pending
        # Seal every page (complete, un-revertable) and let go, i.e. what a finished
        # job leaves in the table as a resumable prefix (same as the seal helper in
        # test_draft_cache_ring.py)
        for page in seq.allocated_pages:
            page.kv_position = PAGE_SIZE
            page.can_revert = False
        pt.deallocate_pages(seq.allocated_pages)
        pt.release_draft_slot(seq)

        resume = make_seq()
        _, cached, _, _ = resume.allocate_pages(pt, None)
        assert cached == (prompt - 1) // PAGE_SIZE
        assert resume.draft_refill_pending
        assert resume.draft_foreign_ceiling == cached * PAGE_SIZE

        # Requeue: the same sequence object allocates again after its slot was released
        pt.deallocate_pages(resume.allocated_pages)
        pt.release_draft_slot(resume)
        assert resume.draft_slot is None
        resume.draft_refill_pending = False
        resume.draft_foreign_ceiling = 0
        _, cached, _, _ = resume.allocate_pages(pt, None)
        assert cached == (prompt - 1) // PAGE_SIZE
        assert resume.draft_slot is not None
        assert resume.draft_refill_pending
        assert resume.draft_foreign_ceiling == cached * PAGE_SIZE

    def test_fresh_prefill_leaves_the_refill_disarmed(self):
        seq, pages = self.make_seq(4, 0)
        self.allocate(seq, pages, 0)
        assert not seq.draft_refill_pending
        assert seq.draft_foreign_ceiling == 0

    def make_refill_job(self):
        # A ring-backed job (own generator, not the call-site fixtures') with the
        # 10000-token prompt fully in pages, ready for a direct refill entry
        from exllamav3.generator.job import Job
        prompt = 10000
        ids = torch.arange(1, prompt + 1, dtype = torch.long).view(1, prompt)
        job = Job(input_ids = ids, embeddings = [])
        seq = job.sequences[0]
        seq.kv_position = prompt
        page_count = (prompt + PAGE_SIZE - 1) // PAGE_SIZE
        seq.allocated_pages = [NoteFakePage(i, PAGE_SIZE) for i in range(page_count)]
        seq.page_hashes = []
        seq.block_index_tensor = torch.tensor([list(range(page_count))], dtype = torch.int32)
        seq.draft_ring = self.RING
        seq.draft_slot = 0
        seq.draft_block_index_tensor = self.RING.table(0, page_count)
        job.generator = SimpleNamespace(
            model = PrefillTargetModel(),
            recurrent_cache = None,
            cache = RecordingTierCache(),
            draft_model = PrefillDraftModel(),
            draft_cache = RecordingTierCache(),
            draft_ring = self.RING,
            dflash_draft = True,
            mtp_draft = False,
        )
        return job

    def test_completion_call_sites_gate_the_refill_on_the_flag(self):
        # The completion call sites (the fully-cached skip branch and the post-forward
        # completion) must consume the refill exactly once: each guards on
        # draft_refill_pending and clears it right after the call. The once-per-resume
        # clearing is pinned behaviorally at the fully-cached branch, where the branch
        # re-enters on every later prefill round (the sequence never completes without
        # a forward) - and only where the rebuild leaves no ceiling credit, in
        # test_fully_cached_resume_without_exported_states_refills_once (with an
        # exporting target the credit keeps the ceiling exit armed for the same
        # window, so an unconditional call is a no-op). The post-forward site can't be
        # pinned the same way: it sets prefill_complete right before the guard, so no
        # later round re-enters it, and its blocked-when-disarmed direction is pinned
        # by test_fresh_multichunk_prefill_never_refills. Pin the remaining contract
        # directly: guard on the flag, clear it after the call, at both sites
        import inspect
        import re
        from exllamav3.generator.job import Job
        src = inspect.getsource(Job.prefill)
        sites = re.findall(
            r"if seq\.draft_refill_pending[^\n]*\n\s+self\.refill_draft_window\(seq\)\n"
            r"\s+seq\.draft_refill_pending = False", src)
        assert len(sites) == 2, (
            "both completion refill call sites must guard on draft_refill_pending and "
            "clear it after the call (once-per-resume)")
        # And the armed flag must actually be consumed: the once-per-resume behavior
        # across re-entered prefill rounds is pinned in
        # TestRefillCallSites.test_fully_cached_resume_without_exported_states_refills_once

    def test_resume_with_foreign_rows_refills(self):
        # A resume whose foreign ceiling sits above the window start (e.g. a
        # fully-cached resume or a recurrent resume, where the whole window holds
        # the slot's previous rows): the armed refill must run. draft_refill_pending is
        # armed alongside the ceiling in production (allocate_pages / the prefill arming
        # sites); the direct entry itself keys on the ceiling
        job = self.make_refill_job()
        job.sequences[0].draft_foreign_ceiling = 10000
        job.refill_draft_window(job.sequences[0])
        assert len(job.generator.model.calls) == 1

    def test_resume_skips_when_own_forwards_cover_the_window(self):
        # The ceiling at or below the window start (tip 10000 - window 2047 = 7953):
        # every window row is this job's own projection, so the rebuild is redundant
        for floor in (0, 7953):
            job = self.make_refill_job()
            job.sequences[0].draft_foreign_ceiling = floor
            job.refill_draft_window(job.sequences[0])
            assert job.generator.model.calls == []

    def test_resume_refills_when_the_ceiling_is_inside_the_window(self):
        # One token above the window start: the bottom row of the window is the
        # slot's previous sequence's rows, so the rebuild must run
        job = self.make_refill_job()
        job.sequences[0].draft_foreign_ceiling = 7954
        job.refill_draft_window(job.sequences[0])
        assert len(job.generator.model.calls) == 1

    def test_ceiling_does_not_guard_the_replay_path(self):
        # A rewind that reaches the wrap floor rebuilds even when the ceiling says
        # the window is own-projected: those rows were projected, but the pre-rewind
        # tip advanced past the ring's wrap floor and clobbered them
        job = self.make_refill_job()
        job.sequences[0].draft_foreign_ceiling = 0
        job.refill_draft_window(job.sequences[0], tip = 10000, replay_from = 9400,
                                rewind_depth = 1000)
        assert len(job.generator.model.calls) == 1


class TestRefillRewindGuard:
    # The guards in Job.refill_draft_window: a rewind is shallow - and skips the rebuild -
    # when it doesn't reach back to the ring's wrap floor (tip + depth reaches past
    # window_start + headroom, the span headroom below the window plus the spec_rows
    # transient write); a deep rewind rebuilds UNLESS the replay prefill itself starts at
    # or below the window start and projects the window. The rewind depth and the replay
    # gap are independent inputs: depth decides clobbering, the gap decides coverage.

    # span 2560, window 2047, spec_rows 17: headroom = 2560 - 2047 - 17 = 496
    RING = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047, spec_rows = 17)
    HEADROOM = 496

    def make_job(self):
        from exllamav3.generator.job import Job
        prompt = 10000
        ids = torch.arange(1, prompt + 1, dtype = torch.long).view(1, prompt)
        job = Job(input_ids = ids, embeddings = [])
        seq = job.sequences[0]
        seq.kv_position = prompt
        page_count = (prompt + PAGE_SIZE - 1) // PAGE_SIZE
        seq.allocated_pages = [NoteFakePage(i, PAGE_SIZE) for i in range(page_count)]
        seq.page_hashes = []
        seq.block_index_tensor = torch.tensor([list(range(page_count))], dtype = torch.int32)
        seq.draft_ring = self.RING
        seq.draft_slot = 0
        seq.draft_block_index_tensor = self.RING.table(0, page_count)
        job.generator = SimpleNamespace(
            model = PrefillTargetModel(),
            max_chunk_size = 4096,
            recurrent_cache = None,
            cache = RecordingTierCache(),
            draft_model = PrefillDraftModel(),
            draft_cache = RecordingTierCache(),
            draft_ring = self.RING,
            dflash_draft = True,
            mtp_draft = False,
        )
        return job

    def refill(self, job, tip, replay_from, rewind_depth = None):
        job.refill_draft_window(job.sequences[0], tip = tip, replay_from = replay_from,
                                rewind_depth = rewind_depth)
        return job.generator.model.calls

    def test_deep_rewind_refills(self):
        # tip 10000, depth 1000 (pre-rewind tip 11000 reaches below the wrap floor),
        # replay from 9400: the window bottom holds clobbered rows -> rebuild
        job = self.make_job()
        calls = self.refill(job, tip = 10000, replay_from = 9400, rewind_depth = 1000)
        assert len(calls) == 1
        # the refill runs the target over the window [10000 - 2047, 10000)
        assert calls[0]["cache_seqlens"][0].item() == 10000 - 2047
        # ...and over exactly that slice: the window's tokens, no more and no less
        window = job.generator.model.inputs[0]
        assert window.shape[-1] == 2047
        assert window[0, 0].item() == 10000 - 2047 + 1
        assert window[0, -1].item() == 10000

    def test_shallow_rewind_skips(self):
        # depth 100: the pre-rewind tip never passed the ring's wrap floor, so every
        # window row still holds its own last write and generation rewrites it anyway
        # - even though the replay gap (tip - replay_from = 600) exceeds the headroom
        job = self.make_job()
        assert self.refill(job, tip = 10000, replay_from = 9400, rewind_depth = 100) == []

    def test_depth_boundary(self):
        # pre-rewind tip exactly at window_start + headroom stays shallow (<=); one
        # more token of rewind depth is deep
        job = self.make_job()
        assert self.refill(job, tip = 10000, replay_from = 9400,
                           rewind_depth = self.HEADROOM) == []
        job = self.make_job()
        assert len(self.refill(job, tip = 10000, replay_from = 9400,
                               rewind_depth = self.HEADROOM + 1)) == 1

    def test_deep_rewind_with_near_stash_refills(self):
        # The regression: the rewind reaches into the wrap floor while the recurrent
        # stash sits right below the rewind target (gap 100 <= headroom). The replay
        # covers almost none of the window and the window bottom is clobbered, so the
        # rebuild must run - a guard keyed on the replay GAP instead of the rewind
        # DEPTH would skip it and leave foreign rows in the window
        job = self.make_job()
        assert len(self.refill(job, tip = 10000, replay_from = 9900,
                               rewind_depth = 3000)) == 1

    def test_shallow_rewind_with_far_stash_skips(self):
        # The other direction: a big replay gap (tip - replay_from = 6000) but the
        # rewind itself never reached the wrap floor - nothing in the window was
        # clobbered, so no rebuild. The old gap-keyed guard rebuilt the whole window
        # on every such rewind
        job = self.make_job()
        assert self.refill(job, tip = 10000, replay_from = 4000, rewind_depth = 50) == []

    def test_replay_covering_the_window_skips(self):
        # replay at or below the window start: the replay prefill projects the window
        # itself, however deep the rewind reached
        job = self.make_job()
        assert self.refill(job, tip = 10000, replay_from = 10000 - 2047,
                           rewind_depth = 1000) == []
        job = self.make_job()
        assert self.refill(job, tip = 10000, replay_from = 0, rewind_depth = 1000) == []

    def test_deep_rewind_without_replay_refills(self):
        # A rewind with no replay prefill: a non-recurrent target, or a recurrent state
        # rewound in place. The ceiling is 0 (fresh prompt, own forwards projected the
        # window), but the pre-rewind tip passed the wrap floor and clobbered the bottom
        # rows, so the rebuild must run - the resume ceiling must not vouch for them
        job = self.make_job()
        assert job.sequences[0].draft_foreign_ceiling == 0
        assert len(self.refill(job, tip = 10000, replay_from = None, rewind_depth = 1000)) == 1

    def test_shallow_rewind_without_replay_skips(self):
        # The wrap-floor check decides even without a replay: nothing was clobbered
        job = self.make_job()
        assert self.refill(job, tip = 10000, replay_from = None, rewind_depth = 100) == []

    def test_tip_below_the_window_clamps_the_headroom(self):
        # The guard's min(window, tip) clamp, driven through the real entry: at tip
        # 900 the clamped headroom is 2560 - 17 - 900 = 1643, so depths that the
        # unclamped headroom (496) would call deep must still skip - the wrap floor
        # (tip + spec_rows - span) sits below position 0, so the advance to the tip
        # lost no row. Without the clamp, the same calls would rebuild
        job = self.make_job()
        assert self.refill(job, tip = 900, replay_from = None, rewind_depth = 497) == []
        job = self.make_job()
        assert self.refill(job, tip = 900, replay_from = None, rewind_depth = 600) == []

    def test_resume_with_foreign_rows_refills_from_rewind_entry(self):
        # no replay and the ceiling above the window start (a fully-cached/resumed
        # window holding the slot's previous rows): the armed refill rebuilds
        job = self.make_job()
        job.sequences[0].draft_foreign_ceiling = 10000
        assert len(self.refill(job, tip = 10000, replay_from = None)) == 1

    def test_non_dflash_generator_skips(self):
        job = self.make_job()
        job.generator.dflash_draft = False
        assert self.refill(job, tip = 10000, replay_from = 9400) == []

    def test_refill_uses_the_tip_argument_not_kv_position(self):
        # In the real rewind, seq.kv_position is reset to replay_from before the refill
        # call, so the explicit tip argument (the post-rewind position) is load-bearing:
        # if it were dropped, tip would fall back to kv_position == replay_from, the
        # window would be computed below 9400, and the rebuild would cover the wrong
        # token range
        job = self.make_job()
        job.sequences[0].kv_position = 9400  # the post-rewind state: kv_position == replay_from
        calls = self.refill(job, tip = 10000, replay_from = 9400, rewind_depth = 1000)
        assert len(calls) == 1
        # the refill ran over the window below the POST-REWIND tip, not kv_position (9400)
        assert calls[0]["cache_seqlens"][0].item() == 10000 - 2047

    def test_deep_rewind_projects_the_window_into_the_ring(self):
        # The refill's observable effect: project the window's exported hidden states into the
        # draft ring's own table, positioned at the window start (no clip: the window fits the span)
        job = self.make_job()
        self.refill(job, tip = 10000, replay_from = 9400, rewind_depth = 1000)
        draft = job.generator.draft_model
        assert len(draft.calls) == 1
        call = draft.calls[0]
        seq = job.sequences[0]
        # the projection writes into the draft ring's table, not the main block table
        assert call["params"]["block_table"] is seq.draft_block_index_tensor
        assert call["cache"] is job.generator.draft_cache
        # no clip: the window (2047) fits the span (2560), so the write is at the window start
        assert call["params"]["cache_seqlens"].tolist() == [10000 - 2047]
        # the exported states are the window's absolute positions, unclipped
        assert call["target_hidden"][0].flatten().tolist() == list(range(10000 - 2047, 10000))
        # the rebuild's TARGET forward rewrote the main cache through the MAIN block
        # table: pointing it at the draft pool/table would scatter the window's main
        # K/V rows across the shared draft pool under ring page indices and leave the
        # main cache silently un-rewritten (mirror of test_no_ring_backing_skips)
        assert len(job.generator.model.calls) == 1
        target = job.generator.model.calls[0]
        assert target["block_table"] is seq.block_index_tensor
        assert target["cache"] is job.generator.cache

    def test_recurrent_refill_from_far_stash_clips_the_projection_to_the_span(self):
        # A recurrent target whose stash sits more than a span below the tip re-runs
        # further than the span, so the projection is clipped to the last span_tokens
        # rows, positioned at the absolute clip start (the kernel writes
        # cache_seqlens[i] + j for row j of the exported chunk). The throwaway state
        # comes back to the pool when the re-run is done
        job = self.make_job()
        job.sequences[0].draft_foreign_ceiling = 10000
        job.recurrent_state = object()
        job.find_recurrent_stash = lambda pos: {"position": 7000}
        freed = []
        job.generator.cache.new_from_stashed = \
            lambda stashed, pos: SimpleNamespace(free = lambda: freed.append(1))
        self.refill(job, tip = 10000, replay_from = None, rewind_depth = 600)
        span = self.RING.span_tokens
        clip_start = 10000 - span
        draft = job.generator.draft_model
        assert len(draft.calls) == 1
        call = draft.calls[0]
        assert call["params"]["cache_seqlens"].tolist() == [clip_start]
        # the exported states are trimmed to the span, from the absolute clip start
        assert call["target_hidden"][0].flatten().tolist() == list(range(clip_start, 10000))
        assert freed == [1]

    def test_missing_stash_far_below_the_window_skips(self):
        # No checkpoint at or below the window start survives: rebuilding would re-run
        # the target from 0 (the whole context) to restore a window's worth of rows,
        # inside prefill (delays the first token) or receive_sample (stalls the round)
        # and needing a spare state slot. Beyond the cost cap the window keeps its
        # current rows and slides out as generation advances
        job = self.make_job()
        job.sequences[0].draft_foreign_ceiling = 10000
        job.recurrent_state = object()
        job.find_recurrent_stash = lambda pos: None
        job.generator.cache.get_new_state = lambda: pytest.fail(
            "the far re-run must not allocate a state")
        assert self.refill(job, tip = 10000, replay_from = None) == []

    def test_full_batch_bails_out_without_crashing(self):
        # A full batch holds every recurrent state slot (the pool is exactly
        # max_batch_size wide): the refill must bail out instead of tripping the
        # pool assertion in new_from_stashed/get_new_state
        job = self.make_job()
        job.sequences[0].draft_foreign_ceiling = 10000
        job.recurrent_state = object()
        job.find_recurrent_stash = lambda pos: {"position": 7000}
        job.generator.cache.free_list = deque()
        assert self.refill(job, tip = 10000, replay_from = None, rewind_depth = 600) == []

    def test_no_ring_backing_skips(self):
        # Without a ring the draft cache spans the pool and the sequence's draft
        # table aliases the MAIN block table: projecting through it would scatter the
        # window's rows into the shared draft cache under main-cache page indices
        job = self.make_job()
        job.sequences[0].draft_ring = None
        assert self.refill(job, tip = 10000, replay_from = 9400, rewind_depth = 1000) == []
        assert job.generator.draft_model.calls == []


class TestRefillImageSpanGuard:
    # The image-span guard in Job.refill_draft_window: a re-run that covers an image span
    # would compute the span's K/V causally (the refill passes no mm_span_prefix),
    # diverging from the original prefill's non-causal span attention and corrupting the
    # main cache. The guard skips the rebuild when ANY token of the re-run range is
    # multimodal - not just when run_start itself sits inside a span.

    RING = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047, spec_rows = 17)

    def make_job(self, mm_span = None):
        from exllamav3.generator.job import Job
        prompt = 10000
        ids = torch.arange(1, prompt + 1, dtype = torch.long).view(1, prompt)
        if mm_span is not None:
            ids[0, mm_span[0]:mm_span[1]] = FIRST_MM_EMBEDDING_INDEX
        job = Job(input_ids = ids, embeddings = [SimpleNamespace()])
        seq = job.sequences[0]
        seq.kv_position = prompt
        page_count = (prompt + PAGE_SIZE - 1) // PAGE_SIZE
        seq.allocated_pages = [NoteFakePage(i, PAGE_SIZE) for i in range(page_count)]
        seq.page_hashes = []
        seq.block_index_tensor = torch.tensor([list(range(page_count))], dtype = torch.int32)
        seq.draft_ring = self.RING
        seq.draft_slot = 0
        seq.draft_block_index_tensor = self.RING.table(0, page_count)
        job.generator = SimpleNamespace(
            model = PrefillTargetModel(),
            max_chunk_size = 4096,
            recurrent_cache = None,
            cache = RecordingTierCache(),
            draft_model = PrefillDraftModel(),
            draft_cache = RecordingTierCache(),
            draft_ring = self.RING,
            dflash_draft = True,
            mtp_draft = False,
        )
        return job

    def refill(self, job, tip, replay_from, rewind_depth = 600):
        # depth 600 > headroom 496: the guard chain always reaches the image-span
        # guard instead of short-circuiting on a shallow rewind
        job.refill_draft_window(job.sequences[0], tip = tip, replay_from = replay_from,
                                rewind_depth = rewind_depth)
        return job.generator.model.calls

    def test_span_covering_run_start_skips(self):
        # The guard's original case: the re-run would start inside the span
        # (run_start = 10000 - 2047 = 7953)
        job = self.make_job(mm_span = (7900, 8000))
        assert self.refill(job, tip = 10000, replay_from = 9400) == []

    def test_span_inside_the_rerun_range_skips(self):
        # A span that starts after run_start is still covered by the re-run: its K/V
        # would be recomputed causally and the main cache corrupted
        job = self.make_job(mm_span = (9000, 9500))
        assert self.refill(job, tip = 10000, replay_from = 9400) == []

    def test_span_below_the_rerun_range_refills(self):
        # A span the re-run does not cover does not block the rebuild
        job = self.make_job(mm_span = (100, 200))
        calls = self.refill(job, tip = 10000, replay_from = 9400)
        assert len(calls) == 1
        assert calls[0]["cache_seqlens"][0].item() == 10000 - 2047

    def test_stash_inside_a_span_skips_and_frees_state(self):
        # The re-run starts at the stash, which sits inside an image span: the guard
        # must fire there and free the throwaway state it already allocated
        job = self.make_job(mm_span = (6900, 7100))
        # a resume whose window is foreign: the ceiling keeps the guard chain from
        # skipping the rebuild before the image-span guard is reached
        job.sequences[0].draft_foreign_ceiling = 10000
        job.recurrent_state = object()
        job.find_recurrent_stash = lambda pos: {"position": 7000}
        freed = []
        job.generator.cache.new_from_stashed = \
            lambda stashed, pos: SimpleNamespace(free = lambda: freed.append(1))
        assert self.refill(job, tip = 10000, replay_from = None) == []
        assert freed == [1]


class TestRefillAltRopeRegrow:
    # A rewind prefill can exceed the prompt-length mrope table, and the RoPE kernel
    # reads it unchecked: when the refill's tip passes the table's token length it must
    # re-derive the table from the full sequence and hand the new one to the forward.

    RING = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047, spec_rows = 17)

    def make_job(self, rope_len):
        from exllamav3.generator.job import Job
        prompt = 10000
        ids = torch.arange(1, prompt + 1, dtype = torch.long).view(1, prompt)
        job = Job(input_ids = ids, embeddings = [])
        seq = job.sequences[0]
        seq.kv_position = prompt
        page_count = (prompt + PAGE_SIZE - 1) // PAGE_SIZE
        seq.allocated_pages = [NoteFakePage(i, PAGE_SIZE) for i in range(page_count)]
        seq.page_hashes = []
        seq.block_index_tensor = torch.tensor([list(range(page_count))], dtype = torch.int32)
        seq.draft_ring = self.RING
        seq.draft_slot = 0
        seq.draft_block_index_tensor = self.RING.table(0, page_count)
        job.generator = SimpleNamespace(
            model = PrefillTargetModel(),
            recurrent_cache = None,
            cache = RecordingTierCache(),
            draft_model = PrefillDraftModel(),
            draft_cache = RecordingTierCache(),
            draft_ring = self.RING,
            dflash_draft = True,
            mtp_draft = False,
        )
        # token dim last, like the real per-position table the kernel indexes
        job.alt_rope_freqs = torch.zeros((1, rope_len, 1))
        job.generator.model.g_rope.freqs = torch.ones((1, prompt, 1))
        return job

    def refill(self, job):
        # depth 600 > headroom 496 at tip 10000: always reach the rebuild path
        job.refill_draft_window(job.sequences[0], tip = 10000, replay_from = None,
                                rewind_depth = 600)
        return job.generator.model.calls

    def test_tip_past_the_table_regrows_mrope_freqs(self):
        job = self.make_job(rope_len = 9000)
        calls = self.refill(job)
        assert len(calls) == 1
        g_rope = job.generator.model.g_rope
        assert len(g_rope.calls) == 1
        ids_arg, embeddings_arg, seq_len_arg = g_rope.calls[0]
        # re-derived from the FULL sequence, not the re-run window
        assert seq_len_arg == 10000
        assert ids_arg.shape[-1] == 10000
        assert embeddings_arg == []
        # the forward ran with the regenerated table
        assert job.alt_rope_freqs is g_rope.freqs
        assert calls[0]["inv_freq"] is g_rope.freqs

    def test_tip_within_the_table_keeps_the_table(self):
        job = self.make_job(rope_len = 10000)
        before = job.alt_rope_freqs
        calls = self.refill(job)
        assert len(calls) == 1
        assert job.generator.model.g_rope.calls == []
        assert job.alt_rope_freqs is before
        assert calls[0]["inv_freq"] is before


class TestRefillWrapFloorGuard:
    # The refill guard's premise is geometric, so test the geometry against a model
    # of the ring's physical rows (logical position p lands on row p % span; the
    # advance writes every accepted row [0, tip) plus the drafter's transient
    # spec_rows past the tip):
    # - a row is PERMANENTLY lost only once tip > p + span + spec_rows: between the
    #   two thresholds a congruent TRANSIENT write occupies the row, but the next
    #   round's accepted write restores it - the window is never rewound into it
    # - never skip while the rewind target reaches a permanently lost row
    # - the guard fires precisely when the rewind target (tip - depth) reaches the
    #   highest row the advance's writes (transient spec_rows included) cover
    #   congruently; one depth shallower the target provably sits above every lost
    #   row (the skip side, asserted below) - so no skip ever aims the rewind at a
    #   row the ring has lost, and no rebuild runs while the target is still clean
    # (test_draft_cache_ring.py's TestWriteClip models the same modulo map; the deleted
    # warmup suite's row-level simulator pinned this rule's predecessor, the credit floor)

    def test_guard_matches_simulated_ring_contents(self):
        for ring_pages, window, spec_rows in ((10, 2047, 17), (4, 700, 9), (10, 2500, 17)):
            ring = DraftRing(ring_pages = ring_pages, num_slots = 1,
                             window_tokens = window, spec_rows = spec_rows)
            span = ring.span_tokens
            headroom = span - spec_rows - window
            assert headroom > 0, "geometry must leave headroom to sweep"
            for tip in (3000, 5000, 10000):
                window_start = tip - min(window, tip)
                # each row's content is its highest congruent write so far
                latest = {}
                for p in range(tip + spec_rows + 1):
                    latest[p % span] = p
                # classify every below-window row by the model's own thresholds
                # (a row the advance never reached - the ring head before the first
                # wrap - is neither lost nor covered; prefill owns it long before the
                # window does)
                permanently_lost = [p for p in range(min(window_start, tip))
                                    if p + span + spec_rows < tip]
                transiently_covered = [p for p in range(min(window_start, tip))
                                       if p + span - spec_rows <= tip < p + span + spec_rows]
                clean = [p for p in range(min(window_start, tip))
                         if p + span > tip + spec_rows]
                # the model agrees with the thresholds it is built from
                assert all(latest[p % span] != p for p in permanently_lost)
                assert all(latest[p % span] != p for p in transiently_covered)
                assert all(latest[p % span] == p for p in clean)
                for depth in range(headroom + 4):
                    fires = depth > headroom  # Job's guard: skip iff depth <= headroom
                    if not fires:
                        # skipping is sound: the rewind target stays above every
                        # permanently lost row (transient rows restore themselves)
                        assert window_start - depth > max(permanently_lost, default=-1)
                    else:
                        # tight the other way: the target reached the highest row
                        # the advance covers congruently (transient included) - one
                        # depth shallower it provably sits above every lost row
                        covered = max(permanently_lost + transiently_covered,
                                      default=-1)
                        assert window_start - depth <= covered


    def test_tip_below_the_window_cannot_lose_a_row(self):
        # tip + spec_rows < span: the wrap floor (tip + spec - span) sits below
        # position 0, so the advance to the tip lost nothing - no row is permanently
        # lost or even transiently covered, and the window is the whole context
        ring = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047, spec_rows = 17)
        span = ring.span_tokens
        tip = 400
        assert tip + ring.spec_rows < span
        latest = {}
        for p in range(tip + ring.spec_rows + 1):
            latest[p % span] = p
        assert all(latest[p % span] == p for p in range(tip + ring.spec_rows + 1))
        # the guard's min(window, tip) clamp grows the headroom to span - spec - tip,
        # so every rewind depth reachable at this tip (<= tip) stays shallow; the
        # real-entry side of the clamp is pinned in
        # TestRefillRewindGuard.test_tip_below_the_window_clamps_the_headroom
        headroom = span - ring.spec_rows - min(ring.window_tokens, tip)
        assert headroom == span - ring.spec_rows - tip
        assert headroom >= tip



class FakePieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def get_id_to_piece_list(self, decode_special_tokens):
        return self.pieces


class TestRewindRefillCallSite:
    # The rewind call site (receive_sample -> rewind_checkpoint -> refill_draft_window)
    # driven end to end through a real banned-string rewind: the checkpoint hold and
    # the two-token rollback are the banned-string filter's. The call site must pass
    # the post-rewind tip and the checkpoint offset as the rewind depth (the rewind
    # resets kv_position to replay_from before the call, so neither can be derived
    # from the sequence afterwards). A two-token rollback is far too shallow to reach
    # the ring's wrap floor, so the depth guard must skip the rebuild outright.

    RING = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047, spec_rows = 17)

    def make_job(self, banned = ("ab",), prompt = 999, kv = 900, max_new = 100,
                 pieces = None):
        from exllamav3.generator.job import Job
        from exllamav3.util.tensor import SeqTensor
        ids = torch.arange(1, prompt + 1, dtype = torch.long).view(1, prompt)
        job = Job(input_ids = ids, embeddings = [], max_new_tokens = max_new,
                  banned_strings = list(banned))
        seq = job.sequences[0]
        seq.kv_position = kv
        page_count = (prompt + PAGE_SIZE - 1) // PAGE_SIZE
        kvs = [min(PAGE_SIZE, max(kv - i * PAGE_SIZE, 0)) for i in range(page_count)]
        seq.allocated_pages = [NoteFakePage(i, kvs[i]) for i in range(page_count)]
        seq.page_hashes = []
        seq.block_index_tensor = torch.tensor([list(range(page_count))], dtype = torch.int32)
        seq.draft_ring = self.RING
        seq.draft_slot = 0
        seq.draft_block_index_tensor = self.RING.table(0, page_count)
        # A recurrent target that cannot roll back in place: the rewind restores the
        # stash at 768 (stashes are page-aligned; the skip loop can then start the
        # replay exactly at the stash). The rewind is 2 tokens deep - nowhere near
        # the ring's wrap floor - so the refill must skip the rebuild
        job.recurrent_state = SimpleNamespace(
            position = 902,
            rollback_capacity = lambda: 0,
            rewind = lambda rw: None,
            free = lambda: None,
        )
        job.find_recurrent_stash = lambda pos: {"position": 768}
        job.generator = SimpleNamespace(
            tokenizer = FakePieceTokenizer(pieces or ["", "a", "b", "c", "d"]),
            draft_model = PrefillDraftModel(),
            ngram_match_min = None,
            model = PrefillTargetModel(),
            max_chunk_size = 4096,
            recurrent_cache = None,
            cache = RecordingTierCache(),
            draft_cache = RecordingTierCache(),
            draft_ring = self.RING,
            dflash_draft = True,
            mtp_draft = False,
        )
        job.pagetable = SimpleNamespace(all_pages = [])
        # The restored state: like the real one it carries its position (prefill
        # reads it to decide what the replay still owes), its rollback capacity (a
        # second rewind can roll it back again), and returns to the pool
        job.generator.cache.new_from_stashed = lambda stashed, pos: SimpleNamespace(
            free = lambda: None, position = pos, rollback_capacity = lambda: 0,
            rewind = lambda rw: None)
        job.serial_number = 1
        job.rq_margin = 0
        job.max_rq_tokens = 10 ** 6
        for name in ("held_tokens", "held_k_tokens"):
            setattr(job, name, SeqTensor((1, 0), dtype = torch.long, seq_dim = -1))
        for name in ("held_probs", "held_k_probs", "held_logits"):
            setattr(job, name, SeqTensor((1, 0), dtype = torch.float, seq_dim = -1))
        return job

    def test_banned_string_rewind_passes_tip_and_depth_and_skips(self):
        # "ab" is banned: the first accept is a partial match (hold, checkpoint at
        # offset 1), the second completes the string and rewinds both tokens through
        # the real rewind_checkpoint path
        job = self.make_job()
        seq = job.sequences[0]
        seen = []
        real_refill = job.refill_draft_window
        job.refill_draft_window = lambda s, **kw: (seen.append(dict(kw)),
                                                   real_refill(s, **kw))[1]
        results = []
        job.receive_sample(None, torch.tensor([[1]]), None, None, None, results)  # "a"
        assert job.checkpoint is not None and job.checkpoint["offset"] == 1
        job.receive_sample(None, torch.tensor([[2]]), None, None, None, results)  # "b"
        assert job.checkpoint_rewound
        assert job.checkpoint["offset"] == 0
        assert results[-1]["suppressed_text"] == "ab"
        # the rewind reset kv_position to the stash position (replay_from)
        assert seq.kv_position == 768
        # the call site passed the POST-REWIND tip (900) and the checkpoint offset:
        # with kv_position already at replay_from, neither is recoverable inside
        # refill_draft_window
        assert len(seen) == 1
        assert seen[0]["tip"] == 900
        assert seen[0]["replay_from"] == 768
        assert seen[0]["rewind_depth"] == 2
        # depth 2 cannot reach the wrap floor (headroom 1643 at this tip): no rebuild
        assert job.generator.model.calls == []
        assert job.generator.draft_model.calls == []
        # the replay prefill runs on the restored state from the stash position
        job.prefill([])
        assert seq.prefill_complete
        seqlens = [p["cache_seqlens"][0].item() for p in job.generator.model.calls]
        assert seqlens == [768]
        # and no completion refill: the replay's own forward covers the window
        assert len(job.generator.model.calls) == 1

    def test_shallow_banned_string_rewind_skips_the_refill(self):
        # The same two-token rollback without the state-pool stash-restore machinery
        # (the state rolls back in place, replay_from stays None): depth 2 can never
        # reach the ring's wrap floor, so the depth guard must skip the rebuild
        job = self.make_job()
        job.recurrent_state = SimpleNamespace(
            position = 902,
            rollback_capacity = lambda: 100,
            rewind = lambda rw: None,
            free = lambda: None,
        )
        results = []
        job.receive_sample(None, torch.tensor([[1]]), None, None, None, results)  # "a"
        job.receive_sample(None, torch.tensor([[2]]), None, None, None, results)  # "b"
        assert job.checkpoint_rewound
        assert job.generator.model.calls == []
        assert job.generator.draft_model.calls == []

    def test_second_banned_string_rewind_uses_the_truncated_tip(self):
        # The first rewind leaves the checkpoint in place at offset 0 "in case the
        # resampled token is rejected too": a second completion re-enters the call
        # site through the restored held-buffer path, a different shape than the
        # first - the tip comes from the already-truncated sequence, and the state
        # (restored at the stash by the first rewind) now rolls back in place, so
        # replay_from stays None
        job = self.make_job()
        seq = job.sequences[0]
        seen = []
        real_refill = job.refill_draft_window
        job.refill_draft_window = lambda s, **kw: (seen.append(dict(kw)),
                                                   real_refill(s, **kw))[1]
        results = []
        job.receive_sample(None, torch.tensor([[1]]), None, None, None, results)  # "a"
        job.receive_sample(None, torch.tensor([[2]]), None, None, None, results)  # "b"
        assert job.checkpoint_rewound
        assert seen[0]["tip"] == 900
        assert seen[0]["replay_from"] == 768
        # the resampled "a" is held again, and the next "b" completes the string a
        # second time, rewinding through the offset-0 checkpoint
        job.receive_sample(None, torch.tensor([[1]]), None, None, None, results)  # "a"
        job.receive_sample(None, torch.tensor([[2]]), None, None, None, results)  # "b"
        assert len(seen) == 2
        assert seen[1]["tip"] == 768
        assert seen[1]["replay_from"] is None
        assert seen[1]["rewind_depth"] == 2
        assert seq.kv_position == 768
        # the restored held buffer is the empty post-emit state: both held tokens
        # are suppressed again
        assert results[-1]["suppressed_text"] == "ab"
        # both rewinds are far too shallow to reach the wrap floor: no rebuild
        assert job.generator.model.calls == []
        assert job.generator.draft_model.calls == []

    def test_deep_banned_string_rewind_rebuilds_end_to_end(self):
        # The deep direction of the same call site, at a full-window tip: a 500-token
        # banned string holds every token as a growing partial match, and the completed
        # match rewinds depth 500 - past the shipped geometry's headroom (2560 - 17 -
        # 2047 = 496), so the pre-rewind tip (3500) passed the ring's wrap floor and
        # the bottom window row was clobbered. The rebuild must run through the real
        # receive_sample -> rewind_checkpoint path: one target forward over the window
        # below the post-rewind tip, projected into the ring through the draft table,
        # on a freed throwaway state. (Only long banned-string holds rewind deep: the
        # checkpoint offset grows one token per consecutive partial match, and at tips
        # below the window the guard's min(window, tip) clamp makes every possible
        # depth shallow - no live row can be above the wrap floor there.)
        hold = 500
        job = self.make_job(banned = ["a" * hold], prompt = 3999, kv = 3000,
                            max_new = hold + 100, pieces = [""] + ["a"] * (hold + 1))
        seq = job.sequences[0]
        # mid-decode: the prompt is prefilled and complete (this fixture hand-builds
        # the decode-time state), so only the rewind's own refill can fire
        seq.prefill_complete = True
        # the state rolls back in place: replay_from stays None, and the ceiling (0:
        # a fresh prompt) must NOT vouch for the window - the depth guard decides
        job.recurrent_state = SimpleNamespace(
            position = 3000 + hold,
            rollback_capacity = lambda: hold * 2,
            rewind = lambda rw: None,
            free = lambda: None,
        )
        job.find_recurrent_stash = lambda pos: None
        freed = []
        job.generator.cache.get_new_state = lambda: SimpleNamespace(
            free = lambda: freed.append(1))
        seen = []
        real_refill = job.refill_draft_window
        job.refill_draft_window = lambda s, **kw: (seen.append(dict(kw)),
                                                   real_refill(s, **kw))[1]
        results = []
        for _ in range(hold):  # 'aaaa...' held as a growing partial match
            job.receive_sample(None, torch.tensor([[1]]), None, None, None, results)
        assert job.checkpoint_rewound
        assert len(seen) == 1
        assert seen[0]["rewind_depth"] == hold
        assert seen[0]["tip"] == 3000
        assert seen[0]["replay_from"] is None
        # the rebuild ran. No recurrent stash at or below the window start survives in
        # this fixture, so the re-run starts from 0 (within the cost cap) and the
        # projection is span-clipped to the last 2560 rows at the absolute clip start
        calls = job.generator.model.calls
        assert len(calls) == 1
        assert calls[0]["cache_seqlens"][0].item() == 0
        assert job.generator.model.inputs[0].shape[-1] == 3000
        # ...projected into the ring, and the main cache rewritten through the main
        # table; the clip start (440) sits below the window start (953), so the whole
        # window is projected
        draft = job.generator.draft_model
        assert len(draft.calls) == 1
        assert draft.calls[0]["params"]["block_table"] is seq.draft_block_index_tensor
        assert draft.calls[0]["params"]["cache_seqlens"].tolist() == [3000 - 2560]
        assert 3000 - 2560 <= 3000 - 2047
        assert draft.calls[0]["cache"] is job.generator.draft_cache
        assert calls[0]["block_table"] is seq.block_index_tensor
        assert calls[0]["cache"] is job.generator.cache
        # the throwaway state returned to the pool
        assert freed == [1]

    def test_rewind_during_prefill_never_double_rebuilds(self):
        # The rewind during prompt prefill sets prefill_complete False, so the
        # replay prefill runs a real forward (768 to the prompt end) and completes
        # the prefill: the completion refill branch is reached exactly once, and
        # later rounds skip at the top of the loop. The shallow rewind already
        # skipped its rebuild (the replay covers the window), and the completion
        # branch must not turn that single skip into a rebuild
        job = self.make_job()
        seq = job.sequences[0]
        results = []
        job.receive_sample(None, torch.tensor([[1]]), None, None, None, results)  # "a"
        job.receive_sample(None, torch.tensor([[2]]), None, None, None, results)  # "b"
        assert job.checkpoint_rewound
        assert len(job.generator.model.calls) == 0
        for _ in range(3):
            job.prefill([])
        assert len(job.generator.model.calls) == 1  # the replay chunk, no rebuild
        assert seq.prefill_complete
        job.prefill([])
        assert len(job.generator.model.calls) == 1