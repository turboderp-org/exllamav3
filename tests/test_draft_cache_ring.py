import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from types import SimpleNamespace
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.pagetable import DraftRing, draft_cache_ring
from exllamav3.cache.cache import Cache, CacheLayer


# A windowed draft cache is a per-sequence ring: logical page p of a sequence lands on physical page
# slot * ring_pages + p % ring_pages. Two properties make that sound, and both are what these tests
# pin: a sequence's pages never leave its own slice of the pool (otherwise one sequence's draft rows
# overwrite another's), and no page a draft forward can read has a newer congruent page (otherwise
# the slot holds the newer page's rows and the drafter attends over the wrong context).

def fake_draft_model(windows, block_size = 8, loaded_tp = False, dflash_draft = True):
    class Config:
        pass
    class Module:
        def __init__(self, w):
            self.sliding_window = w
    model = SimpleNamespace()
    model.loaded_tp = loaded_tp
    model.config = Config()
    model.config.block_size = block_size
    # Every ring-capable drafter in the tree (dflash, dflash2, dflash_laguna, deepseek_v4_mtp)
    # advertises dflash_draft; the ring is gated on it
    model.caps = {"dflash_draft": True} if dflash_draft else {}
    model.get_cache_layers = lambda: [Module(w) for w in windows]
    return model


class TestRingGeometry:

    def test_windowed_draft_model_gets_a_ring(self):
        ring = draft_cache_ring(fake_draft_model([2047] * 5), num_draft_tokens = 8, num_slots = 2)
        assert ring is not None
        # Must cover the window plus the drafted block plus a page of slack, page-aligned
        assert ring.ring_pages * PAGE_SIZE >= 2047 + 8 + 1 + PAGE_SIZE
        assert ring.ring_pages * PAGE_SIZE < 2047 + 8 + 1 + 2 * PAGE_SIZE
        assert ring.num_tokens == ring.ring_pages * PAGE_SIZE * ring.num_slots
        assert ring.window_tokens == 2047

    def test_global_attention_draft_model_keeps_pool_cache(self):
        # AR/MTP drafters and DFlash checkpoints with a full_attention layer read draft KV from
        # position 0, which is exactly what a ring discards
        assert draft_cache_ring(fake_draft_model([2047, -1]), 8, 2) is None
        assert draft_cache_ring(fake_draft_model([None, 2047]), 8, 2) is None
        assert draft_cache_ring(fake_draft_model([]), 8, 2) is None

    def test_tp_draft_model_keeps_pool_cache(self):
        # Its cache tensors live in the TP workers, so they cannot be reallocated from here
        assert draft_cache_ring(fake_draft_model([2047], loaded_tp = True), 8, 2) is None

    def test_non_dflash_draft_model_keeps_pool_cache(self):
        # The ring is DFlash-family-only by capability: a plain AR drafter runs its own
        # forward over the draft cache (each row's values derive from the cached left
        # window), so even an all-sliding-window AR config must not get a ring
        assert draft_cache_ring(fake_draft_model([2047] * 5, dflash_draft = False), 8, 2) is None

    def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setenv("EXL3_DRAFT_RING", "0")
        assert draft_cache_ring(fake_draft_model([2047]), 8, 2) is None

    def test_spec_rows_and_token_footprint(self):
        # spec_rows is the widest transient write one draft round lands past the accepted
        # tip (drafted block rows + verify row): max(num_draft_tokens, block_size) + 1. It
        # feeds the span (keep = window + spec_rows + a page of slack), so pin how the ring
        # builder derives it and what footprint the ring declares (the Generator resizes the
        # draft cache to exactly ring.num_tokens).
        model = fake_draft_model([2047] * 5)  # block_size 8
        ring = draft_cache_ring(model, num_draft_tokens = 16, num_slots = 2)
        assert ring.spec_rows == 17  # max(16, 8) + 1
        assert ring.ring_pages * PAGE_SIZE >= 2047 + 17 + PAGE_SIZE
        assert ring.span_tokens == 2560  # the shipped dflash2 geometry
        assert ring.num_tokens == ring.span_tokens * ring.num_slots
        # the model's own block_size dominates when the drafted block is smaller
        assert draft_cache_ring(fake_draft_model([2047], block_size = 16), 4, 1).spec_rows == 17
        # and the drafted block dominates when it is larger
        assert draft_cache_ring(fake_draft_model([2047], block_size = 8), 20, 1).spec_rows == 21
        # a zero block_size (no block_size on the config) still credits the drafted block
        model_no_block = fake_draft_model([2047], block_size = 0)
        assert draft_cache_ring(model_no_block, 6, 1).spec_rows == 7


class TestRingTable:

    @pytest.mark.parametrize("ring_pages,num_slots,page_count", [
        (10, 2, 10), (10, 2, 640), (10, 2, 1024), (16, 4, 4096), (2, 1, 1), (2, 1, 7),
    ])
    def test_pages_stay_inside_the_owning_slice(self, ring_pages, num_slots, page_count):
        ring = DraftRing(ring_pages = ring_pages, num_slots = num_slots, window_tokens = 2047)
        tables = [ring.table(slot, page_count) for slot in range(num_slots)]
        for slot, table in enumerate(tables):
            assert table.dtype == torch.int32 and table.is_contiguous()
            assert table.shape == (1, page_count)
            pages = table[0]
            assert int(pages.min()) >= slot * ring_pages
            assert int(pages.max()) < (slot + 1) * ring_pages
        # Sequence-private: no physical page is claimed by two sequences
        flat = [set(t[0].tolist()) for t in tables]
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                assert not flat[i] & flat[j]


    def test_writes_from_older_pages_cannot_hit_a_live_slot(self):
        # Page-granular stream ordering: chunks land oldest page first, so once every chunk of
        # a long prompt has been written, each of the last ring_pages pages owns its slot - a
        # newer page only ever overwrites a page it is congruent with, which is by definition
        # not live. This is a property of the modulo map plus write order; no write-range
        # clipping is involved (the loop iterates page indices, clip_write is token-granular)
        # and clipping itself is pinned by TestWriteClip
        ring_pages = 10
        ring = DraftRing(ring_pages = ring_pages, num_slots = 1, window_tokens = 2047)
        table = ring.table(0, 4000)[0].tolist()
        written = {}
        for chunk_start in range(0, 4000, 333):
            chunk_end = min(chunk_start + 333, 4000)
            for p in range(chunk_start, chunk_end):
                written[table[p]] = p
        for p in range(4000 - ring_pages, 4000):
            assert written[table[p]] == p


class TestWriteClip:
    # The kernel maps a whole chunk's positions to rows in one unordered launch, so two
    # positions span_tokens apart in one write range are a data race: the older congruent
    # write can win and leave a live-window row holding a position from a span ago. Clipping
    # each write range to the span removes the congruent pairs without touching the rows the
    # drafter can actually read.

    ring = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047)

    def test_unclipped_chunk_does_contain_congruent_pairs(self):
        # The hazard the clip removes: at the shipped defaults (span 2560, chunk 4096) a
        # single prefill chunk writes positions x and x + span to the same physical row.
        # Congruence is whatever the production block table says, so the predicate is read off
        # ring.table: two positions collide when their pages map to one physical page and their
        # page offsets agree (the same row key the write kernel uses)
        chunk = 4096
        span = self.ring.span_tokens
        assert span < chunk
        table = self.ring.table(0, (chunk + PAGE_SIZE - 1) // PAGE_SIZE)[0].tolist()
        row = lambda p: (table[p // PAGE_SIZE], p % PAGE_SIZE)
        for p in range(0, chunk - span):
            assert row(p + span) == row(p)

    def test_clipped_range_has_no_congruent_pairs(self):
        span = self.ring.span_tokens
        # Same predicate as the sibling test: the row key is whatever the production block
        # table maps a position to (physical page, page offset), read off ring.table rather
        # than re-derived inline, so the test cannot drift from the real mapping
        max_end = 10_000
        table = self.ring.table(0, (max_end + PAGE_SIZE - 1) // PAGE_SIZE)[0].tolist()
        row = lambda p: (table[p // PAGE_SIZE], p % PAGE_SIZE)
        for start, end in [(0, 4096), (3000, 7096), (0, 2560), (1, 2561), (5000, 5200), (0, 10_000)]:
            clip = self.ring.clip_write(start, end)
            assert start <= clip <= end
            assert end - clip <= span, (start, end)
            # a range of at most span consecutive positions can never hit one row twice
            rows = {row(p) for p in range(clip, end)}
            assert len(rows) == end - clip, (start, end)

    def test_short_chunks_are_not_clipped(self):
        assert self.ring.clip_write(100, 900) == 100
        assert self.ring.clip_write(100, 100) == 100
        assert self.ring.clip_write(0, self.ring.span_tokens) == 0

    def test_clipped_chunked_prefill_leaves_the_window_current(self):
        # Every row the final window reads must hold the newest congruent position written
        # across all chunks: clipping must not skip a row the drafter will read
        span = self.ring.span_tokens
        total = 10_000
        table = self.ring.table(0, (total + PAGE_SIZE - 1) // PAGE_SIZE)[0].tolist()
        written = {}
        for chunk_start in range(0, total, 4096):
            chunk_end = min(chunk_start + 4096, total)
            for p in range(self.ring.clip_write(chunk_start, chunk_end), chunk_end):
                written[(table[p // PAGE_SIZE], p % PAGE_SIZE)] = p
        for p in range(total - self.ring.window_tokens, total):
            assert written[(table[p // PAGE_SIZE], p % PAGE_SIZE)] == p, p


class TestAdmissionSlotCap:
    # The pool was sized at load time; the ring is clamped to it, and admission must stop at
    # the slot count so allocate_pages' slot claim can never run dry (F2)

    def fake_job(self, serial):
        from types import SimpleNamespace
        return SimpleNamespace(
            sequences = [object()],
            serial_number = serial,
            identifier = None,
            skips = 0,
            max_skips = 100,
            current_new_pages_required = lambda: 1,
            activate = lambda: None,
            allocate_pages = lambda: None,
        )

    def make_gen(self, slots):
        from types import SimpleNamespace
        from exllamav3.generator.generator import Generator
        gen = object.__new__(Generator)
        gen.max_batch_size = 8
        gen.draft_ring = DraftRing(ring_pages = 10, num_slots = slots, window_tokens = 2047)
        gen.active_jobs = []
        gen.pending_jobs = [self.fake_job(i) for i in range(4)]
        gen.pagetable = SimpleNamespace(num_unreferenced_pages = lambda: 10 ** 6)
        return gen

    def test_admission_capped_by_ring_slots(self):
        gen = self.make_gen(slots = 2)
        results = []
        gen.iterate_start_jobs(results)
        assert len(gen.active_jobs) == 2
        assert len(gen.pending_jobs) == 2
        assert len(results) == 2

    def test_max_batch_size_still_bounds_admission_without_a_ring(self):
        gen = self.make_gen(slots = 8)
        gen.draft_ring = None
        gen.max_batch_size = 3
        gen.iterate_start_jobs([])
        assert len(gen.active_jobs) == 3



class FakeCache:
    def __init__(self, max_num_tokens):
        self.max_num_tokens = max_num_tokens


class FakeGenerator:
    pass


def make_page_table(max_pages = 128, ring = None):
    from collections import deque
    from exllamav3.generator.pagetable import PageTable
    pt = PageTable(FakeGenerator(), FakeCache(max_pages * PAGE_SIZE))
    pt.draft_ring = ring
    pt.draft_slots = deque(range(ring.num_slots)) if ring is not None else deque()
    return pt


def make_sequence(n_tokens):
    from exllamav3.generator.pagetable import Sequence
    ids = torch.arange(1, n_tokens + 1, dtype = torch.long).view(1, n_tokens)
    return Sequence(ids[0].clone(), ids[0].clone())


class TestSequenceRing:
    # Same properties as above, but driven through the real page table: slot claim/release,
    # table width against the sequences pages and cached-prefix reuse as the generator sees it

    ring = DraftRing(ring_pages = 10, num_slots = 2, window_tokens = 2047)

    def test_slot_claim_and_positional_table(self):
        pt = make_page_table(ring = self.ring)
        seq = make_sequence(4000)
        seq.prepare(has_prefix_token = False, max_new_tokens = 1000)
        pages, cached, _, _ = seq.allocate_pages(pt, None)
        assert seq.draft_slot == 0
        assert cached == 0
        assert seq.draft_block_index_tensor is not seq.block_index_tensor
        assert seq.draft_block_index_tensor.shape == seq.block_index_tensor.shape == (1, pages)
        assert pages == (4000 + 1000 + PAGE_SIZE - 1) // PAGE_SIZE > self.ring.ring_pages
        values = seq.draft_block_index_tensor[0].tolist()
        assert all(0 <= v < self.ring.ring_pages for v in values)
        assert values[:self.ring.ring_pages] == list(range(self.ring.ring_pages))
        assert values[self.ring.ring_pages] == 0  # wraps back onto the ring

    def test_two_sequences_do_not_share_ring_pages(self):
        pt = make_page_table(ring = self.ring)
        seqs = []
        for idx in range(2):
            seq = make_sequence(4000 + idx * PAGE_SIZE)
            seq.prepare(has_prefix_token = False, max_new_tokens = 1000)
            seq.allocate_pages(pt, None)
            seqs.append(seq)
        assert [s.draft_slot for s in seqs] == [0, 1]
        assert not set(seqs[0].draft_block_index_tensor[0].tolist()) & \
               set(seqs[1].draft_block_index_tensor[0].tolist())
        # No slot left for a third live sequence
        third = make_sequence(1000)
        third.prepare(has_prefix_token = False, max_new_tokens = 100)
        with pytest.raises(AssertionError, match = "ring slot"):
            third.allocate_pages(pt, None)
        # Release returns the slot
        pt.release_draft_slot(seqs[0])
        assert seqs[0].draft_slot is None and seqs[0].draft_block_index_tensor is None
        third.allocate_pages(pt, None)
        assert third.draft_slot == 0

    def seal(self, pt, seq):
        """Mark every page complete and sealed, i.e. resumable by a later job, then let go of it"""
        for page in seq.allocated_pages:
            page.kv_position = PAGE_SIZE
            page.can_revert = False
        self.release(pt, seq)

    def release(self, pt, seq):
        pt.deallocate_pages(seq.allocated_pages)
        pt.release_draft_slot(seq)

    def test_full_reuse_without_a_ring(self):
        pt = make_page_table(ring = None)
        seq = make_sequence(40 * PAGE_SIZE)
        seq.prepare(has_prefix_token = False, max_new_tokens = PAGE_SIZE)
        seq.allocate_pages(pt, None)
        self.seal(pt, seq)
        assert seq.draft_block_index_tensor is seq.block_index_tensor
        resume = make_sequence(40 * PAGE_SIZE)
        resume.prepare(has_prefix_token = False, max_new_tokens = PAGE_SIZE)
        _, cached, _, _ = resume.allocate_pages(pt, None)
        assert cached == (40 * PAGE_SIZE - 1) // PAGE_SIZE
        assert resume.kv_position == cached * PAGE_SIZE
        assert resume.draft_block_index_tensor is resume.block_index_tensor

    def test_prefix_reuse_still_works_with_a_ring(self):
        # Cached-prefix reuse must work the same with a ring present: a sealed
        # 40-page prefix resumes with cached == 39 over the sealed pages and kv_position at the
        # page boundary, and the ring's draft table is bound (distinct from the main block table)
        pt = make_page_table(ring = self.ring)
        seq = make_sequence(40 * PAGE_SIZE)
        seq.prepare(has_prefix_token = False, max_new_tokens = PAGE_SIZE)
        seq.allocate_pages(pt, None)
        self.seal(pt, seq)
        resume = make_sequence(40 * PAGE_SIZE)
        resume.prepare(has_prefix_token = False, max_new_tokens = PAGE_SIZE)
        _, cached, _, _ = resume.allocate_pages(pt, None)
        assert cached == (40 * PAGE_SIZE - 1) // PAGE_SIZE
        assert resume.kv_position == cached * PAGE_SIZE
        # the ring's draft table is bound and distinct from the main block table
        assert resume.draft_slot is not None
        assert resume.draft_block_index_tensor is not resume.block_index_tensor
        # A longer prompt on the same cached prefix keeps its full reuse: the shared
        # 40-page prefix resumes over the sealed pages and the extra pages allocate fresh
        longer = make_sequence(60 * PAGE_SIZE)
        longer.prepare(has_prefix_token = False, max_new_tokens = PAGE_SIZE)
        _, cached, _, _ = longer.allocate_pages(pt, None)
        assert cached == (40 * PAGE_SIZE - 1) // PAGE_SIZE
        assert longer.kv_position == cached * PAGE_SIZE
        # A prompt shorter than the ring span starts at kv_position 0 on a FRESH
        # table: nothing in it is sealed, so the prefix offers no reuse. A separate
        # table: the sequences above still hold their ring slots, and this prompt
        # shares the sealed prefix, which would otherwise be reused
        fresh = make_page_table(ring = self.ring)
        short = make_sequence(8 * PAGE_SIZE)
        short.prepare(has_prefix_token = False, max_new_tokens = PAGE_SIZE)
        _, cached, _, _ = short.allocate_pages(fresh, None)
        assert cached == 0 and short.kv_position == 0


class TestDFlashPrefillClip:
    # DFlash2 fills the ring by projecting the target's exported hidden states, so the clip
    # call-site is Job.prefill: a prefill chunk longer than the ring span must hand
    # update_kv_from_target exactly the last span_tokens rows, positioned at the ABSOLUTE clip
    # start (the kernel writes cache_seqlens[i] + j for row j of the exported chunk), while a
    # chunk that fits the span is passed through untouched. Driven through the real Job and
    # Sequence with a stub target model that exports position-tagged states and a stub draft
    # model that records what it was asked to write; no weights, no GPU.

    ring = DraftRing(ring_pages = 10, num_slots = 1, window_tokens = 2047)

    class FakePage:
        def __init__(self, page_index):
            self.page_index = page_index
            self.kv_position = 0
            self.prev_hash = None
            self.phash = b"\x00" * 16
            self.can_revert = True
            self.sequence = torch.zeros((1, PAGE_SIZE), dtype = torch.long)

    class TargetModel:
        # Real exporter contract: the tapped norms append the chunk's post-norm states to
        # params['export_states'] when the drafter asked for them through draft_verifier_params.
        # Each exported row carries its absolute position, so a wrong slice offset or a slice
        # that kept the wrong end of the chunk is visible in what the drafter receives.
        caps = {}

        def __init__(self):
            self.calls = []

        def prefill(self, *, input_ids, params):
            self.calls.append(params)
            if "export_state_norm_keys" in params:
                start = int(params["cache_seqlens"][0])
                positions = torch.arange(start, start + input_ids.shape[-1], dtype = torch.float32)
                params["export_states"] = [positions.view(1, -1, 1), positions.view(1, -1, 1) + 1000.0]

        forward = prefill

    class DraftModel:
        def __init__(self, exports = True):
            self.calls = []
            self.draft_verifier_params = {"export_state_norm_keys": ("target.final_norm",)} if exports else {}

        def update_kv_from_target(self, *, target_hidden, cache, params):
            self.calls.append({"target_hidden": target_hidden, "cache": cache, "params": params})

    def make_job(self, prompt_tokens, resume_at, max_chunk_size, ring, exports = True):
        from exllamav3.generator.job import Job
        ids = torch.arange(1, prompt_tokens + 1, dtype = torch.long).view(1, prompt_tokens)
        job = Job(input_ids = ids, embeddings = [])
        seq = job.sequences[0]
        seq.kv_position = resume_at
        page_count = (prompt_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        seq.allocated_pages = [self.FakePage(i) for i in range(page_count)]
        seq.page_hashes = []  # empty: prefill never re-hashes a completed page here
        seq.block_index_tensor = torch.tensor([list(range(page_count))], dtype = torch.int32)
        seq.draft_block_index_tensor = ring.table(0, page_count) if ring is not None else seq.block_index_tensor
        job.pagetable = SimpleNamespace(all_pages = [])  # no partial-page reuse candidates
        job.generator = SimpleNamespace(
            model = self.TargetModel(),
            max_chunk_size = max_chunk_size,
            recurrent_cache = None,
            cache = SimpleNamespace(),
            draft_model = self.DraftModel(exports = exports),
            draft_cache = SimpleNamespace(),
            draft_ring = ring,
            dflash_draft = True,
            mtp_draft = False,
        )
        return job

    def run_chunk(self, job):
        before = len(job.generator.draft_model.calls)
        job.prefill([])
        assert len(job.generator.draft_model.calls) == before + 1
        return job.generator.draft_model.calls[-1], job.generator.model.calls[-1]

    def test_long_prefill_chunk_is_clipped_to_the_span(self):
        ring = self.ring
        span = ring.span_tokens
        job = self.make_job(prompt_tokens = 5000, resume_at = 0, max_chunk_size = 4096, ring = ring)
        draft_call, target_params = self.run_chunk(job)
        clip_start = 4096 - span
        assert 0 < clip_start < 4096
        # The target still prefilled the whole chunk; only the draft write is clipped
        assert target_params["export_states"][0].shape[1] == 4096
        assert target_params["cache_seqlens"].tolist() == [0]
        assert target_params["block_table"] is job.sequences[0].block_index_tensor
        # Every exported state tensor is trimmed to the span, and the write is positioned at
        # the absolute clip start so rows land on the positions they were computed from
        assert draft_call["params"]["cache_seqlens"].tolist() == [clip_start]
        for tap, states in enumerate(draft_call["target_hidden"]):
            assert states.shape[1] == span
            assert float(states[0, 0, 0]) == clip_start + 1000.0 * tap
            assert float(states[0, -1, 0]) == 4095 + 1000.0 * tap
        assert draft_call["params"]["block_table"] is job.sequences[0].draft_block_index_tensor
        assert draft_call["cache"] is job.generator.draft_cache
        assert job.sequences[0].kv_position == 4096

    def test_clip_slice_is_chunk_relative_and_seqlens_is_absolute(self):
        # A job resuming mid-sequence separates the two quantities the clip produces: the slice
        # into the exported chunk is (clip_start - prefill_start) while cache_seqlens is the
        # absolute clip_start
        ring = self.ring
        span = ring.span_tokens
        job = self.make_job(prompt_tokens = 6000, resume_at = 1024, max_chunk_size = 4096, ring = ring)
        draft_call, target_params = self.run_chunk(job)
        clip_start = 5120 - span
        assert 1024 < clip_start < 5120
        assert target_params["cache_seqlens"].tolist() == [1024]
        assert draft_call["params"]["cache_seqlens"].tolist() == [clip_start]
        assert [states.shape[1] for states in draft_call["target_hidden"]] == [span, span]
        # Rows are positions [clip_start, 5120): slicing the exported chunk by the absolute
        # clip_start instead of the chunk-relative offset would drop the wrong rows
        assert float(draft_call["target_hidden"][0][0, 0, 0]) == clip_start
        assert float(draft_call["target_hidden"][0][0, -1, 0]) == 5119

    def test_short_prefill_chunk_passes_through_unclipped(self):
        ring = self.ring
        assert 1024 < ring.span_tokens
        job = self.make_job(prompt_tokens = 3000, resume_at = 0, max_chunk_size = 1024, ring = ring)
        draft_call, target_params = self.run_chunk(job)
        assert [states.shape[1] for states in draft_call["target_hidden"]] == [1024, 1024]
        assert draft_call["params"]["cache_seqlens"].tolist() == target_params["cache_seqlens"].tolist()
        assert float(draft_call["target_hidden"][0][0, 0, 0]) == 0
        assert float(draft_call["target_hidden"][0][0, -1, 0]) == 1023

    def test_chunked_prefill_clips_only_the_oversized_chunk(self):
        # Cross-chunk aliasing is safe because chunks land oldest first: the oversized first
        # chunk is clipped, the short tail chunk is written whole at its own absolute position
        ring = self.ring
        span = ring.span_tokens
        job = self.make_job(prompt_tokens = 5000, resume_at = 0, max_chunk_size = 4096, ring = ring)
        job.prefill([])
        job.prefill([])
        first, second = job.generator.draft_model.calls
        assert first["params"]["cache_seqlens"].tolist() == [4096 - span]
        assert [states.shape[1] for states in first["target_hidden"]] == [span, span]
        assert second["params"]["cache_seqlens"].tolist() == [4096]
        assert [states.shape[1] for states in second["target_hidden"]] == [903, 903]
        assert float(second["target_hidden"][0][0, 0, 0]) == 4096
        assert float(second["target_hidden"][0][0, -1, 0]) == 4998
        assert job.sequences[0].prefill_complete

    def test_pool_backed_draft_model_is_never_clipped(self):
        # Without a ring the draft cache is the pool-wide mirror of the main page table, so
        # there is no span to clip to and the whole chunk must be written
        job = self.make_job(prompt_tokens = 5000, resume_at = 0, max_chunk_size = 4096, ring = None)
        draft_call, target_params = self.run_chunk(job)
        assert [states.shape[1] for states in draft_call["target_hidden"]] == [4096, 4096]
        assert draft_call["params"]["cache_seqlens"].tolist() == target_params["cache_seqlens"].tolist()
        assert float(draft_call["target_hidden"][0][0, 0, 0]) == 0

    def test_draft_model_without_exported_states_is_passed_through(self):
        # A drafter that asked for no state taps leaves export_states unset; the clip guard on
        # it must not fire, and the call keeps the unclipped seqlens
        ring = self.ring
        job = self.make_job(prompt_tokens = 5000, resume_at = 0, max_chunk_size = 4096, ring = ring, exports = False)
        draft_call, target_params = self.run_chunk(job)
        assert "export_states" not in target_params
        assert draft_call["target_hidden"] is None
        assert draft_call["params"]["cache_seqlens"].tolist() == [0]


# The draft-cache ring is installed by resizing the draft Cache after the fact:
# Cache.resize_num_tokens (exllamav3/cache/cache.py) rebuilds every paged layer for a new
# capacity while preserving the Cache object every existing reference holds, and
# Generator._setup_draft_ring decides the target size. Both are pinned here with CPU fakes:
# a CacheLayer stand-in whose alloc/free track the device, and recording stand-ins for the
# Cache objects the Generator resizes.

class FakePagedLayer(CacheLayer):
    # CPU stand-in for CacheLayer_fp16: one plain tensor sized by max_num_tokens, with
    # alloc/free appending to a shared log so the resize sequence (free-all, rebuild,
    # alloc-all on the recorded device) and the loader-only no-allocation path are both
    # observable without a GPU.
    BYTES_PER_TOKEN = 16

    def __init__(self, config, attention, cache_id, max_num_tokens, **kwargs):
        super().__init__(config, attention, cache_id, max_num_tokens, **kwargs)
        self.log = kwargs["log"]
        self.device = None
        self.tensor = None

    def alloc(self, device):
        assert self.tensor is None, "alloc called on an already-allocated layer"
        self.device = device
        self.tensor = torch.zeros((self.max_num_tokens, self.BYTES_PER_TOKEN))
        self.log.append(("alloc", id(self), self.max_num_tokens, str(device)))

    def free(self):
        self.log.append(("free", id(self), self.max_num_tokens, self.device))
        self.tensor = None
        self.device = None

    def storage_size(self):
        return self.max_num_tokens * self.BYTES_PER_TOKEN

    def overhead_size(self):
        return 0

    def get_tensors(self):
        return (self.tensor,)

    def get_kv(self, cache_seqlens, block_table, sliding_window = -1):
        raise NotImplementedError

    def update_kv(self, cache_seqlens, block_table, k, v, length):
        raise NotImplementedError

    def update_kv_direct(self, cache_seqlens, block_table, k, v, length):
        raise NotImplementedError

    def copy_page(self, source, from_page, to_page, num_tokens):
        raise NotImplementedError

    def tp_export(self, plan):
        raise NotImplementedError


class FakeCacheModule:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.sliding_window = 2047
        self.cache_layers = []
        self.recurrent_layers = []


class FakeRecurrentLayerState:
    def __init__(self, layer, num_slots, max_history, cache_id):
        self.layer = layer


class FakeCacheModel:
    # The minimal Model surface Cache.__init__/attach/detach/resize touch: config, the
    # weakref table, the cache/recurrent layer module lists and the instance map.
    def __init__(self, num_layers = 3, recurrent = False):
        self.config = SimpleNamespace()
        self.loaded_tp = False
        self.cache_weakrefs = {}
        self.recurrent_state_cls = FakeRecurrentLayerState if recurrent else None
        self.modules = [FakeCacheModule(i) for i in range(num_layers)]
        self.recurrent_modules = []
        if recurrent:
            rec = FakeCacheModule(100)
            rec.layer_state_cls = FakeRecurrentLayerState
            self.recurrent_modules = [rec]

    def get_cache_layers(self):
        return self.modules

    def get_recurrent_layers(self):
        return self.recurrent_modules

    def get_layer_instances(self, layer_idx):
        return [(layer_idx, 0)]


class TestCacheResize:
    BYTES = FakePagedLayer.BYTES_PER_TOKEN

    def make_cache(self, log, num_layers = 3, tokens = 8 * PAGE_SIZE, **model_kwargs):
        model = FakeCacheModel(num_layers = num_layers, **model_kwargs)
        cache = Cache(model, tokens, layer_type = FakePagedLayer, log = log)
        return model, cache

    def test_building_and_storage_size_allocate_nothing(self):
        log = []
        model, cache = self.make_cache(log, num_layers = 3)
        assert cache.initialized is False
        assert log == []  # _build_layers sizes the layer objects; the loader allocates them
        assert cache.storage_size_total() == 3 * 8 * PAGE_SIZE * self.BYTES
        assert all(layer.tensor is None for layer in cache.layers.values())
        # attach bookkeeping: the model holds a weakref to the cache and each module
        # lists exactly this cache's layer
        assert model.cache_weakrefs[id(cache)]() is cache
        for module in model.modules:
            assert module.cache_layers == [cache.layers[(module.layer_idx, 0)]]

    def test_preload_resize_rebuilds_layers_and_registration(self):
        log = []
        model, cache = self.make_cache(log, num_layers = 3)
        old = dict(cache.layers)
        cache.resize_num_tokens(4 * PAGE_SIZE)
        assert cache.max_num_tokens == 4 * PAGE_SIZE
        # the rebuild frees the (never-allocated) old layer objects but allocates
        # nothing: the loader brings the new objects up at load time
        assert [entry[0] for entry in log] == ["free", "free", "free"]
        # the layer objects are rebuilt, keyed by the same instances, sized to the new
        # capacity and left unallocated
        assert set(cache.layers) == set(old)
        for key, layer in cache.layers.items():
            assert layer is not old[key]
            assert layer.max_num_tokens == 4 * PAGE_SIZE
            assert layer.tensor is None and layer.device is None
            assert old[key] not in model.modules[key[0]].cache_layers
        # re-registration followed the rebuild, and the weakref still resolves to the
        # same Cache object
        for module in model.modules:
            assert module.cache_layers == [cache.layers[(module.layer_idx, 0)]]
        assert model.cache_weakrefs[id(cache)]() is cache
        assert cache.storage_size_total() == 3 * 4 * PAGE_SIZE * self.BYTES

    def test_initialized_resize_preserves_identity_device_and_sizes(self):
        log = []
        model, cache = self.make_cache(log, num_layers = 2)
        cache.initialized = True
        device = torch.device("cpu")
        for layer in cache.layers.values():
            layer.alloc(device)
        old_tensors = {key: layer.tensor for key, layer in cache.layers.items()}
        log.clear()
        cache_ref = cache
        cache.resize_num_tokens(4 * PAGE_SIZE)
        # the Cache object identity survives, so params["cache"] and the model's weakref
        # table stay valid across the resize
        assert cache is cache_ref
        assert cache.initialized is True
        # every old layer is freed before any new layer is allocated: shrinking returns
        # the memory to the allocator before the grow asks for the new footprint
        assert [entry[0] for entry in log] == ["free", "free", "alloc", "alloc"]
        for entry in log:
            if entry[0] == "alloc":
                assert entry[3] == "cpu"  # reallocated on the device the layer lived on
        for key, layer in cache.layers.items():
            assert layer.device == device
            assert layer.tensor is not None and layer.tensor is not old_tensors[key]
            assert layer.tensor.shape[0] == 4 * PAGE_SIZE
            assert model.modules[key[0]].cache_layers == [layer]
        assert model.cache_weakrefs[id(cache)]() is cache
        assert cache.storage_size_total() == 2 * 4 * PAGE_SIZE * self.BYTES
        # storage_size_total is independent of allocation state
        for layer in cache.layers.values():
            layer.free()
        assert cache.storage_size_total() == 2 * 4 * PAGE_SIZE * self.BYTES

    def test_resize_rejects_bad_sizes_without_touching_state(self):
        log = []
        model, cache = self.make_cache(log, num_layers = 2)
        layers_before = dict(cache.layers)
        for bad in (0, -PAGE_SIZE, 4 * PAGE_SIZE + 1, PAGE_SIZE // 2):
            with pytest.raises(AssertionError, match = "multiple"):
                cache.resize_num_tokens(bad)
        assert cache.max_num_tokens == 8 * PAGE_SIZE
        assert dict(cache.layers) == layers_before
        for module in model.modules:
            assert module.cache_layers == [layers_before[(module.layer_idx, 0)]]
        assert model.cache_weakrefs[id(cache)]() is cache
        assert log == []

    def test_resize_refuses_recurrent_tp_and_unallocated(self):
        # Recurrent state layers are sized per batch slot, not per token: a cache that
        # has them cannot be resized through the paged path
        log = []
        model, cache = self.make_cache(log, recurrent = True)
        assert cache.recurrent_layers
        with pytest.raises(AssertionError, match = "recurrent"):
            cache.resize_num_tokens(4 * PAGE_SIZE)
        # TP workers own the layer tensors: rebuilding them from here would desync the
        # workers
        model, cache = self.make_cache(log)
        model.loaded_tp = True
        with pytest.raises(AssertionError, match = "tensor-parallel"):
            cache.resize_num_tokens(4 * PAGE_SIZE)
        # initialized but never allocated: there is no device to reallocate onto
        model, cache = self.make_cache(log)
        cache.initialized = True
        with pytest.raises(AssertionError, match = "not allocated"):
            cache.resize_num_tokens(4 * PAGE_SIZE)
        assert log == []


class RecordingSizedCache:
    # Stand-in for Cache at the Generator._setup_draft_ring seam: the method only ever
    # touches max_num_tokens, storage_size_total() and resize_num_tokens(), so recording
    # those calls pins the resize contract without any tensors.
    def __init__(self, max_num_tokens):
        self.max_num_tokens = max_num_tokens
        self.resizes = []

    def storage_size_total(self):
        return self.max_num_tokens * 4096

    def resize_num_tokens(self, max_num_tokens):
        self.resizes.append(max_num_tokens)
        self.max_num_tokens = max_num_tokens


def make_ring_setup_gen(num_draft_tokens = 16):
    from exllamav3.generator.generator import Generator
    gen = object.__new__(Generator)
    gen.num_draft_tokens = num_draft_tokens
    gen.draft_ring = None
    gen.pagetable = SimpleNamespace(draft_ring = None, draft_slots = None)
    return gen


class TestSetupDraftRing:
    # Generator._setup_draft_ring owns the draft cache's final size: with a ring it
    # resizes the pool-wide cache the caller passed down to exactly ring.num_tokens;
    # without one the draft cache must span the pool like the main cache. Geometry
    # throughout: window 2047, spec_rows 17 -> span 2560, two slots = 5120.

    def test_pool_wide_draft_cache_shrinks_to_the_ring(self, monkeypatch):
        monkeypatch.delenv("EXL3_DRAFT_RING", raising = False)
        gen = make_ring_setup_gen()
        draft = RecordingSizedCache(100 * PAGE_SIZE)
        main = RecordingSizedCache(100 * PAGE_SIZE)
        gen._setup_draft_ring(fake_draft_model([2047] * 5), draft, main, max_batch_size = 2)
        ring = gen.draft_ring
        assert ring is not None
        assert ring.spec_rows == 17 and ring.span_tokens == 2560
        assert ring.num_slots == 2
        # exactly one resize, straight to the ring footprint; the main cache is untouched
        assert draft.resizes == [ring.num_tokens] == [5120]
        assert draft.max_num_tokens == 5120
        assert main.resizes == []
        # the page table learns the ring, one slot per live sequence
        assert gen.pagetable.draft_ring is ring
        assert list(gen.pagetable.draft_slots) == [0, 1]

    def test_ring_slots_clamped_to_the_pool(self, monkeypatch):
        monkeypatch.delenv("EXL3_DRAFT_RING", raising = False)
        gen = make_ring_setup_gen()
        # the pool holds exactly one span: admission must cap at one sequence even
        # though max_batch_size is 4, and the already-correct size is not resized
        draft = RecordingSizedCache(2560)
        main = RecordingSizedCache(2560)
        gen._setup_draft_ring(fake_draft_model([2047] * 5), draft, main, max_batch_size = 4)
        assert gen.draft_ring.num_slots == 1
        assert draft.resizes == []
        assert list(gen.pagetable.draft_slots) == [0]

    def test_pool_below_one_span_is_refused(self, monkeypatch):
        monkeypatch.delenv("EXL3_DRAFT_RING", raising = False)
        gen = make_ring_setup_gen()
        draft = RecordingSizedCache(2048)  # < one 2560-token span
        with pytest.raises(RuntimeError, match = "smaller than one"):
            gen._setup_draft_ring(fake_draft_model([2047] * 5), draft,
                                  RecordingSizedCache(2048), max_batch_size = 2)
        assert draft.resizes == []

    def test_no_ring_resizes_draft_cache_to_the_main_cache(self, monkeypatch):
        # The caller may have pre-sized the draft cache for a ring this Generator
        # declined (e.g. EXL3_DRAFT_RING=0): the pool-wide path must resize it to match
        # the main cache instead of faulting on the mismatch
        monkeypatch.setenv("EXL3_DRAFT_RING", "0")
        gen = make_ring_setup_gen()
        draft = RecordingSizedCache(100 * PAGE_SIZE)
        main = RecordingSizedCache(80 * PAGE_SIZE)
        gen._setup_draft_ring(fake_draft_model([2047] * 5), draft, main, max_batch_size = 2)
        assert gen.draft_ring is None
        assert gen.pagetable.draft_ring is None
        assert draft.resizes == [80 * PAGE_SIZE]
        # equal sizes are left alone
        draft2 = RecordingSizedCache(80 * PAGE_SIZE)
        gen2 = make_ring_setup_gen()
        gen2._setup_draft_ring(fake_draft_model([2047] * 5), draft2,
                               RecordingSizedCache(80 * PAGE_SIZE), max_batch_size = 2)
        assert draft2.resizes == []

    def test_non_dflash_draft_model_takes_the_pool_path(self, monkeypatch):
        # Capability refusal (a plain AR drafter) must reach the same resize-to-main
        # contract as the env kill switch, without the env var
        monkeypatch.delenv("EXL3_DRAFT_RING", raising = False)
        gen = make_ring_setup_gen()
        draft = RecordingSizedCache(100 * PAGE_SIZE)
        gen._setup_draft_ring(fake_draft_model([2047] * 5, dflash_draft = False), draft,
                              RecordingSizedCache(64 * PAGE_SIZE), max_batch_size = 2)
        assert gen.draft_ring is None
        assert draft.resizes == [64 * PAGE_SIZE]