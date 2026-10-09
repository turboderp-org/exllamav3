"""Draft/verify row alignment in iterate_gen, driven end to end over a mixed batch.

test_draft_row_map.py pins the mapping function in isolation; this pins the wiring that uses it.
The regression: the draft pass emits one row per drafting job, in its active-set order, while
iterate_gen builds the verify batch from every prefill-done job. A prefill-done job absent from
the active set (no draft row) must take no row, or every drafting job behind it is fed a foreign
row and compared against one - the accepted token is always the target's sample, so the output
stays correct while acceptance collapses.

Generator._draft_active_jobs is currently exactly the prefill-done set, so the rowless case is
defensive-only here (it fires if a future filter drops a job from the draft active set); the
wiring below pins that a rowless job never steals or shifts a row when it does occur.

Runs the real iterate_gen over fake jobs, a fake model and a staging stand-in.
No model weights, no CUDA allocations.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from collections import deque
from types import SimpleNamespace

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.generator import Generator
from exllamav3.generator.pagetable import DraftRing


VOCAB = 64
PENDING = 5  # the one real position a prefill-done job still owes the forward


class Staging:
    """Stands in for Generator._staging: same (name, width) keying and row growth, but plain
    pageable memory so the test needs no CUDA context. Records every call, so a regression to
    an unstaged batch_ids assembly shows up as a missing "batch_ids" request."""

    def __init__(self):
        self.buffers = {}
        self.calls = []

    def __call__(self, name, rows, width = None, dtype = torch.int32):
        self.calls.append((name, rows, width, dtype))
        key = (name, width)
        buf = self.buffers.get(key)
        if buf is None or buf.shape[0] < rows:
            shape = (max(rows, 32),) if width is None else (max(rows, 32), width)
            buf = torch.zeros(shape, dtype = dtype)
            self.buffers[key] = buf
        return buf[:rows]

    def requested(self, name):
        return [c for c in self.calls if c[0] == name]


class FakePage:
    def __init__(self, kv_position):
        self.kv_position = kv_position


class FakeSequenceIds:
    """Minimal stand-in for the generated token ids: the calibration pass only reads the last one."""

    def __init__(self, tail, length = 5):
        self._ids = torch.zeros(length, dtype = torch.long)
        self._ids[-1] = tail
        self._length = length

    def __len__(self):
        return self._length

    def torch_slice(self, start, end):
        return self._ids[start:end]


class FakeSeq:
    def __init__(self, kv_position, drafted = 0, tail = 7, block_pages = None,
                 draft_block_pages = None):
        self.kv_position = kv_position
        self.sequence_ids = FakeSequenceIds(tail)
        # Distinct page values per table (default zeros keep existing tests unchanged) so a
        # refill wired to the wrong table is visible instead of silently identical
        self.block_index_tensor = torch.tensor([block_pages or [0] * 16], dtype = torch.int32)
        self.draft_block_index_tensor = torch.tensor(
            [draft_block_pages or [0] * 16], dtype = torch.int32
        )
        # Enough pages for the drafted block to land in the second one, so a rejection has
        # somewhere to roll back to
        self.allocated_pages = [FakePage(kv_position % PAGE_SIZE), FakePage((kv_position + drafted) % PAGE_SIZE)]


class FakeState:
    """Records rewind calls so the padded-state vs real-cache split is observable from the outside."""

    def __init__(self):
        self.rewinds = []

    def rewind(self, n):
        self.rewinds.append(n)


class FakeJob:
    """A prefill-done job with one sequence, scripted to sample a fixed token per verify position.
    Records which draft row it was handed, so the fed row and the compared row are both observable
    from the outside (the compared row shows up as accepted_draft_tokens).

    eos_at / rq_at / rewind_at script the exits that break out of the acceptance loop before the
    draft comparison runs, each at the position index it fires on.
    """

    def __init__(self, script, drafted = 0, eos_at = None, rq_at = None, rewind_at = None,
                 prefill_done = True, block_pages = None, draft_block_pages = None):
        self.sequences = [FakeSeq(300, drafted, block_pages = block_pages,
                                  draft_block_pages = draft_block_pages)]
        self.script = script
        self.eos_at = eos_at
        self.rq_at = rq_at
        self.rewind_at = rewind_at
        self.prefill_done = prefill_done
        self.deallocated = False
        self.stashed = []
        self.rq_job = None
        self.time_first_token = 1.0
        self.embeddings = []
        self.new_tokens = 0
        self.filters = []
        self.filter_futures = []
        self.logit_masks = []
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.checkpoint_rewound = False
        self.recurrent_state = FakeState()
        self.draft_stats = []
        self.fed_draft_rows = []
        self.fed_widths = []
        self._pos = 0

    def is_prefill_done(self):
        return self.prefill_done

    def get_max_seq_len(self):
        return 301

    def is_checkpoint_boundary(self):
        return False

    def prepare_logit_mask(self):
        pass

    def prepare_sampling_past_ids(self):
        pass

    def deallocate_pages(self):
        self.deallocated = True

    def get_input_ids_list(self, draft_tokens = None, idx = 0, add_to_cache = False):
        ids = torch.full((1, 1), PENDING, dtype = torch.long)
        self.fed_draft_rows.append(None if draft_tokens is None else idx)
        if draft_tokens is not None:
            ids = torch.cat((ids, draft_tokens[idx:idx + 1, :]), dim = -1)
        self.fed_widths.append(ids.shape[-1])
        return [ids]

    def receive_logits(self, token_logits):
        return torch.tensor([self.script[self._pos]], dtype = torch.long), None, None, None

    def receive_sample(self, token_logits, next_token, *args):
        sampled = next_token
        pos = self._pos
        self._pos += 1
        if pos == self.rewind_at:
            self.checkpoint_rewound = True
        return pos == self.eos_at, sampled, pos == self.rq_at

    def maybe_stash_recurrent(self, cache, interval = None):
        self.stashed.append(interval)

    def prepare_for_requeue(self):
        self.rq_job = object()
        return self.rq_job


class FakeModel:
    def __init__(self, rows, width):
        self.caps = set()
        self.seen = {}
        self.logits = torch.zeros((rows, width, VOCAB), dtype = torch.float)

    def forward(self, input_ids, params):
        self.seen["input_ids"] = input_ids.clone()
        self.seen["params"] = params
        return self.logits

    def prefetch_tokens(self, tokens):
        pass


class FakeDraftModel:
    """Stands in for a DFlash/MTP draft model. Records the params dict of every accepted-row
    cache refill (update_kv_from_target / prefill), so which block table the refill was
    wired to - the staged per-sequence ring table or the main block_index - is observable
    from the outside. export_states stands in for the target hidden states the real forward
    exports for the MTP refill."""

    def __init__(self, export_states = None):
        self.draft_verifier_params = {} if export_states is None else {"export_states": export_states}
        self.refills = []
        self.prefills = []

    def update_kv_from_target(self, target_hidden, cache, lengths, params):
        self.refills.append({"target_hidden": target_hidden, "cache": cache,
                             "lengths": list(lengths), "params": params})

    def prefill(self, ids, params):
        self.prefills.append((ids.clone(), params))


def make_generator(jobs, rows, width):
    gen = object.__new__(Generator)
    gen.active_jobs = list(jobs)  # iterate_gen retires jobs from this list; keep the caller's intact
    gen.pending_jobs = []
    gen.pagetable = None
    gen.staging_buffers = {}
    gen.staging = Staging()
    gen._staging = gen.staging
    gen.cache = None
    gen.draft_cache = None
    gen.recurrent_cache = None
    gen.num_draft_tokens = width - 1
    gen.draft_model = None
    gen.draft_ring = None
    gen.mtp_draft = False
    gen.dflash_draft = False
    gen.draft_calibrator = None
    gen._draft_conf_round = None
    gen.record_draft_stats = True
    gen.model = FakeModel(rows, width)
    # A round that retires every job drains the queue; the real hook touches the cache allocator
    gen.on_queue_drained = lambda: None
    return gen


MAX_BATCH_SIZE = 8
STALE = 99  # what a row past the end of this round's draft block still holds


def draft_buffer(draft_rows, width):
    """
    iterate_gen is handed self.draft_ids_pinned[:, :window]: a max_batch_size-row buffer of which
    only the first len(draft_active) rows were written this round. Reading past them is therefore
    a silent stale read, never an IndexError - which is exactly why the row skew went unnoticed.
    """
    buf = torch.full((MAX_BATCH_SIZE, width - 1), STALE, dtype = torch.long)
    buf[:len(draft_rows)] = torch.tensor(draft_rows, dtype = torch.long)
    return buf


class RecordingCalibrator:
    """Stands in for the draft-confidence calibrator, recording every (confidence, matched) label
    the calibration pass submits, so a label attributed to the wrong job's round is visible."""

    def __init__(self):
        self.labels = []
        self.decays = 0

    def add_label(self, conf, matched):
        self.labels.append((conf, matched))

    def decay_step(self):
        self.decays += 1


def conf_round(draft_rows, window):
    """
    The confidence round the draft pass stashes for iterate_gen's calibration update: one row per
    drafting job, in draft-row order, with values distinct per row so a mis-indexed row shows up
    instead of silently matching.
    """
    conf = torch.tensor([[10.0 * (r + 1) + i for i in range(window)] for r in range(draft_rows)])
    ids = torch.tensor(
        [[100 * (r + 1) + i for i in range(window)] for r in range(draft_rows)], dtype = torch.long
    )
    return {"ids": ids, "conf": conf, "window": window}



class TestDraftRowWiring:
    """
    A draft row r is only correct if the job that is *fed* draft_tokens[r] is also the job whose
    logits are *compared* against draft_tokens[r]. The job scripts below accept exactly their own
    draft row, so any row skew shows up as a dropped acceptance.
    """

    def run_round(self, jobs, draft_active, draft_tokens, rows, width, recurrent = False,
                  calibrator = None, ring = None, dflash = False, mtp = False,
                  export_states = None):
        gen = make_generator(jobs, rows, width)
        if calibrator is not None:
            gen.draft_calibrator = calibrator
            gen._draft_conf_round = conf_round(len(draft_active), width - 1)
        if recurrent:
            gen.recurrent_cache = object()
        if ring is not None:
            # Mirror exactly what Generator._setup_draft_ring installs: the ring on the
            # generator (the switch iterate_gen reads) and on the pagetable, plus the free
            # slot deque the pagetable hands out.
            gen.draft_ring = ring
            gen.pagetable = SimpleNamespace(
                draft_ring = ring,
                draft_slots = deque(range(ring.num_slots)),
                metrics = {},
            )
        if dflash or mtp:
            gen.draft_model = FakeDraftModel(export_states)
            gen.dflash_draft = dflash
            gen.mtp_draft = mtp
        gen.iterate_gen([], draft_tokens, draft_active)
        return gen

    def test_rowless_job_does_not_shift_the_draft_rows_behind_it(self):
        # Draft pass order: job1 -> row 0, job2 -> row 1. job0 has no draft row and sits at
        # compact-batch index 0, exactly the offset that used to skew every row behind it.
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        job0 = FakeJob([7, 7, 7, 7])
        job1 = FakeJob([11, 12, 13, 13], drafted = 3)
        job2 = FakeJob([50, 50, 50, 50], drafted = 3)
        jobs = [job0, job1, job2]
        before = job2.sequences[0].allocated_pages[1].kv_position
        self.run_round(jobs, [job1, job2], draft_tokens, rows = 3, width = 4)

        # Each drafting job was handed its own row, and only the drafting jobs were handed one
        assert job0.fed_draft_rows == [None]
        assert job1.fed_draft_rows == [0]
        assert job2.fed_draft_rows == [1]

        # job1's script accepts only row 0, job2's accepts only row 1. Reading either row off by
        # one - onto a neighbour's row, or onto a stale row past the end of this round's block -
        # collapses the count to 0.
        assert job1.accepted_draft_tokens == 3
        assert job2.accepted_draft_tokens == 0

        # Rejected at i=0, so the drafted tail is dropped: counted as rejected, and the page rolled
        # back by the job's own width (4 - 1 - 0), not the batch width
        assert job2.rejected_draft_tokens == 3
        assert before - job2.sequences[0].allocated_pages[1].kv_position == 3

    def test_rowless_job_is_padded_to_the_batch_width_and_staged(self):
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        job0 = FakeJob([7, 7, 7, 7])
        job1 = FakeJob([11, 12, 13, 13], drafted = 3)
        job2 = FakeJob([21, 22, 23, 23], drafted = 3)
        jobs = [job0, job1, job2]
        gen = self.run_round(jobs, [job1, job2], draft_tokens, rows = 3, width = 4)

        # The rowless job contributed one real position; the rest is zero padding, which the
        # forward needs only to stay rectangular
        assert [j.fed_widths for j in jobs] == [[1], [4], [4]]
        ids = gen.model.seen["input_ids"]
        assert ids.shape == (3, 4)
        assert ids[0].tolist() == [PENDING, 0, 0, 0]
        assert ids[1].tolist() == [PENDING, 11, 12, 13]
        assert ids[2].tolist() == [PENDING, 21, 22, 23]

        # ...and it went through the reusable staging buffer rather than a fresh pageable tensor
        assert gen.staging.requested("batch_ids") == [("batch_ids", 3, 4, torch.long)]

        # A 1-token verify is not a draft round, so it must not land in the acceptance stats
        assert job0.draft_stats == []
        assert job1.draft_stats == [(job1.new_tokens, 3, 3)]
        assert job2.draft_stats == [(job2.new_tokens, 3, 3)]

    def test_all_jobs_have_rows(self):
        # Happy path: every job has a row, every row is its compact-batch index, nothing is padded
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23], [31, 32, 33]], width = 4)
        jobs = [FakeJob([11, 12, 13, 13]), FakeJob([21, 22, 23, 23]), FakeJob([31, 32, 33, 33])]
        gen = self.run_round(jobs, jobs, draft_tokens, rows = 3, width = 4)

        assert [j.fed_draft_rows for j in jobs] == [[0], [1], [2]]
        assert [j.accepted_draft_tokens for j in jobs] == [3, 3, 3]
        ids = gen.model.seen["input_ids"]
        assert ids.shape == (3, 4)
        assert ids[0].tolist() == [PENDING, 11, 12, 13]
        assert gen.staging.requested("batch_ids") == [("batch_ids", 3, 4, torch.long)]

    def test_rowless_job_at_the_tail_does_not_consume_a_row(self):
        # Same skew from the other end: the drafting jobs keep rows 0 and 1, and the rowless job
        # at the tail must not take row 2, which is where its compact index would point
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        job0 = FakeJob([11, 12, 13, 13])
        job1 = FakeJob([21, 22, 23, 23])
        job2 = FakeJob([7, 7, 7, 7])
        jobs = [job0, job1, job2]
        self.run_round(jobs, [job0, job1], draft_tokens, rows = 3, width = 4)

        assert job0.fed_draft_rows == [0]
        assert job1.fed_draft_rows == [1]
        assert job2.fed_draft_rows == [None]
        assert job0.accepted_draft_tokens == 3
        assert job1.accepted_draft_tokens == 3

    def test_recurrent_state_rewinds_by_padded_width_while_pages_rollback_by_job_width(self):
        # The verify forward advances every recurrent state by the padded row width (4), but the
        # cache only moves by the job's real tokens. The rejected drafting job rewinds its state by
        # the padded remainder (4 - 1 - 0) while its page rolls back by the job width (3); the
        # rowless job consumed one real position, so its state rewinds the other three; the
        # fully-accepted job lands on the width and rewinds nothing
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        job0 = FakeJob([7, 7, 7, 7])
        job1 = FakeJob([11, 12, 13, 13], drafted = 3)
        job2 = FakeJob([50, 50, 50, 50], drafted = 3)
        jobs = [job0, job1, job2]
        before = job2.sequences[0].allocated_pages[1].kv_position
        self.run_round(jobs, [job1, job2], draft_tokens, rows = 3, width = 4, recurrent = True)

        assert job0.recurrent_state.rewinds == [3]
        assert job1.recurrent_state.rewinds == [0]
        assert job2.recurrent_state.rewinds == [3]
        assert before - job2.sequences[0].allocated_pages[1].kv_position == 3

    def test_draft_active_none_falls_back_to_the_compact_batch_index(self):
        # Without an active set (draft_rows is None), each drafting job takes its compact-batch
        # index as its row, exactly as before the row map existed
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23], [31, 32, 33]], width = 4)
        jobs = [FakeJob([11, 12, 13, 13]), FakeJob([21, 22, 23, 23]), FakeJob([31, 32, 33, 33])]
        self.run_round(jobs, None, draft_tokens, rows = 3, width = 4)

        assert [j.fed_draft_rows for j in jobs] == [[0], [1], [2]]
        assert [j.accepted_draft_tokens for j in jobs] == [3, 3, 3]

    def test_mid_window_rejection_counts_accepted_and_rejected(self):
        # Accept at i=0, mismatch at i=1: the draft comparison runs past the first position, the
        # accepted token is counted, and the two unresolved positions are rejected and rolled
        # back by the job's own width
        draft_tokens = draft_buffer([[11, 12, 13]], width = 4)
        job = FakeJob([11, 99, 13, 13], drafted = 3)
        before = job.sequences[0].allocated_pages[1].kv_position
        self.run_round([job], [job], draft_tokens, rows = 1, width = 4)

        assert job.accepted_draft_tokens == 1
        assert job.rejected_draft_tokens == 2
        assert before - job.sequences[0].allocated_pages[1].kv_position == 2
        assert job.draft_stats == [(0, 3, 1)]

    def test_requeue_rewinds_state_by_the_padded_remainder_and_pages_by_the_job_width(self):
        # Requeue breaks out before the draft comparison, so reject_remainder is the only thing
        # that unwinds the round. A rowless job requeued at i=0 rewinds three state positions
        # (the padded remainder) and zero cache positions (its real row was one token); a drafting
        # job requeued at i=0 rewinds three of each. Swapping the two desyncs the recurrent state
        # from the K/V the replacement segment resumes over.
        draft_tokens = draft_buffer([[11, 12, 13]], width = 4)
        job0 = FakeJob([7, 7, 7, 7], rq_at = 0)
        job1 = FakeJob([11, 12, 13, 13], drafted = 3, rq_at = 0)
        jobs = [job0, job1]
        pages0 = job0.sequences[0].allocated_pages[0].kv_position
        pages1 = job1.sequences[0].allocated_pages[1].kv_position
        gen = self.run_round(jobs, [job1], draft_tokens, rows = 2, width = 4, recurrent = True)

        assert job0.recurrent_state.rewinds == [3]
        assert job0.sequences[0].allocated_pages[0].kv_position == pages0
        assert job0.rejected_draft_tokens == 0
        assert job1.recurrent_state.rewinds == [3]
        assert pages1 - job1.sequences[0].allocated_pages[1].kv_position == 3
        assert job1.rejected_draft_tokens == 3

        # Both yielded: state stashed at the page boundary, pages released, replacements queued
        assert job0.stashed == [PAGE_SIZE] and job1.stashed == [PAGE_SIZE]
        assert [j.rq_job is not None for j in jobs] == [True, True]
        assert gen.active_jobs == []

    def test_eos_rewinds_the_padded_remainder_and_releases_the_job(self):
        # EOS also breaks out before the comparison, so the outgoing-state normalizer at the end of
        # the round is what unwinds it: the rowless job resolved one position and rewinds the
        # other three, the drafting job resolved one draft token before stopping and rewinds two.
        # Positions fed but never resolved are counted as rejected, and their cache is not rolled
        # back - the pages go with the job.
        draft_tokens = draft_buffer([[11, 12, 13]], width = 4)
        job0 = FakeJob([7, 7, 7, 7], eos_at = 0)
        job1 = FakeJob([11, 12, 99, 99], drafted = 3, eos_at = 1)
        jobs = [job0, job1]
        before = job1.sequences[0].allocated_pages[1].kv_position
        gen = self.run_round(jobs, [job1], draft_tokens, rows = 2, width = 4, recurrent = True)

        assert job0.recurrent_state.rewinds == [3]
        assert job0.rejected_draft_tokens == 0
        assert job1.recurrent_state.rewinds == [2]
        assert job1.rejected_draft_tokens == 2
        assert before == job1.sequences[0].allocated_pages[1].kv_position
        assert job0.deallocated and job1.deallocated
        assert gen.active_jobs == []

    def test_banned_string_rewind_abandons_the_window_untouched(self):
        # A banned-string match already reset the job's pages and recurrent state inside
        # receive_sample, so the round must not normalize or roll anything on top of it: the -1
        # sentinel skips both the padded-state rewind and the draft stats. Letting it through would
        # rewind a state that no longer exists, and count an abandoned window as a draft round.
        draft_tokens = draft_buffer([[11, 12, 13]], width = 4)
        job0 = FakeJob([7, 7, 7, 7])
        job1 = FakeJob([11, 12, 99, 99], drafted = 3, rewind_at = 1)
        jobs = [job0, job1]
        before = job1.sequences[0].allocated_pages[1].kv_position
        self.run_round(jobs, [job1], draft_tokens, rows = 2, width = 4, recurrent = True)

        assert job1.recurrent_state.rewinds == []
        assert before == job1.sequences[0].allocated_pages[1].kv_position
        assert job1.draft_stats == []
        assert job1.rejected_draft_tokens == 2
        assert not job1.deallocated

    def test_calibration_labels_only_drafting_jobs_with_their_own_conf_row(self):
        # The calibration pass reads accepted_lengths with a counter that must advance for every
        # job, including the rowless ones it skips. If the skip forgot to count job0, job1 and
        # job2 would each read the accepted length of the job in front of them - and, since the
        # conf rows are indexed by draft row, the labels would land on the wrong round as well.
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        job0 = FakeJob([7, 7, 7, 7])
        job1 = FakeJob([11, 12, 13, 13], drafted = 3)   # accepts its whole draft
        job2 = FakeJob([50, 50, 50, 50], drafted = 3)   # rejects at i=0
        cal = RecordingCalibrator()
        self.run_round([job0, job1, job2], [job1, job2], draft_tokens, rows = 3, width = 4,
                       calibrator = cal)

        # job1's three accepted positions label its own row True; job2 resolved nothing, so only
        # the first position of its row is labelled, False (its id 100 is not the sequence's last
        # generated token 7). The rowless job submitted nothing.
        assert cal.labels == [(10.0, True), (11.0, True), (12.0, True), (20.0, False)]
        assert cal.decays == 1

    def test_still_prefilling_job_takes_no_row_and_no_width(self):
        # A job that has not finished prefill must be skipped by the compact counters in
        # iterate_gen: if the guarded loop counted it, the drafting jobs behind it would read
        # the previous job's width or row - the original skew's shape, this time from the
        # prefill guard rather than the active-set filter. The batch shape and row mapping below pin
        # the skip for the input-ids loop and batch; batch_states and accepted_lengths build
        # from the same guarded batch_jobs. Production never puts a prefilling job in
        # draft_active, matching here
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        pre = FakeJob([7, 7, 7, 7], prefill_done = False)
        job1 = FakeJob([11, 12, 13, 13], drafted = 3)
        job2 = FakeJob([21, 22, 23, 23], drafted = 3)
        gen = self.run_round([pre, job1, job2], [job1, job2], draft_tokens, rows = 2, width = 4)

        assert pre.fed_draft_rows == [] and pre.fed_widths == []
        assert job1.fed_draft_rows == [0]
        assert job2.fed_draft_rows == [1]
        assert [j.accepted_draft_tokens for j in (job1, job2)] == [3, 3]
        ids = gen.model.seen["input_ids"]
        assert ids.shape == (2, 4)
        assert ids[0].tolist() == [PENDING, 11, 12, 13]
        assert ids[1].tolist() == [PENDING, 21, 22, 23]

    @pytest.mark.parametrize("draft_pass", [
        "iterate_draftmodel_dflash_gen", "iterate_draftmodel_mtp_gen", "iterate_draftmodel_gen",
    ])
    def test_iterate_computes_the_active_set_once_for_both_passes(self, draft_pass):
        # iterate() is the only place that guarantees the draft pass and the verify pass see
        # the same active set. If the wiring regressed - iterate_gen called without the
        # third argument, or either pass recomputing its own set - every other test here
        # still passes (they call iterate_gen directly and bless the None fallback), while
        # production re-skews the rows with acceptance silently collapsed. This drives the
        # real iterate() and real _draft_active_jobs with the passes stubbed, pinning the
        # membership (every prefill-done job drafts, a not-prefill-done one does not), the
        # identity (one list, both passes) and the token handoff (the verify pass receives
        # exactly the buffer the draft pass returned), for each of the three draft branches
        from types import SimpleNamespace
        from exllamav3.generator.pagetable import DraftRing
        job0 = FakeJob([7, 7, 7, 7])
        job1 = FakeJob([11, 12, 13, 13], drafted = 3)
        job2 = FakeJob([7, 7, 7, 7], prefill_done = False)
        for job in (job0, job1, job2):
            job.prefill = lambda results: None
        gen = make_generator([job0, job1, job2], rows = 1, width = 4)
        gen.cache = SimpleNamespace(initialized = True)
        gen.draft_cache = None
        gen.visualizer = None
        gen.draft_model = object()
        gen.dflash_draft = draft_pass.endswith("dflash_gen")
        gen.mtp_draft = draft_pass.endswith("mtp_gen")
        gen.pagetable = SimpleNamespace(
            draft_ring = DraftRing(ring_pages = 10, num_slots = 2, window_tokens = 2047),
            # mirror what _setup_draft_ring installs, so a future slot dependency in
            # iterate() surfaces as an intentional assertion, not a missing attribute
            draft_slots = deque(range(2)),
            metrics = {},
        )
        gen.iterate_start_jobs = lambda results: None
        seen = {}
        def fake_draft_pass(results, active):
            seen["draft"] = active
            seen["draft_tokens"] = draft_buffer([[11, 12, 13]], width = 4)
            return seen["draft_tokens"]
        def fake_iterate_gen(results, tokens, active):
            seen["verify"] = (tokens, active)
        setattr(gen, draft_pass, fake_draft_pass)
        gen.iterate_gen = fake_iterate_gen
        gen.iterate()

        # job2 is not prefill-done, so _draft_active_jobs excludes it: it must not draft
        assert seen["draft"] == [job0, job1]
        # The buffer the draft pass returned must be the object the verify pass is handed:
        # a verify call wired to None or to a stale buffer keeps the output correct (the
        # verifier re-samples every position) while silently drafting against garbage rows
        assert seen["verify"][0] is seen["draft_tokens"]
        assert seen["verify"][1] is seen["draft"]

    # ---- the draft-cache ring table wiring ----------------------------------------------------
    #
    # With a ring-backed draft cache, the accepted-row refill must write into the draft
    # pool's own page space: iterate_gen stages a per-sequence block table
    # ("verify_draft_block_index", built from seq.draft_block_index_tensor) and the
    # DFlash/MTP refills receive that table, not the main block_index. Feeding the main
    # table instead scatters the accepted rows across unrelated physical pages of the
    # draft pool - the output stays correct (the verifier re-samples everything) while
    # every later draft attends garbage.

    RING = DraftRing(ring_pages = 8, num_slots = 2, window_tokens = 2047, spec_rows = 4)
    # 301 prompt + 3 drafted + 3 verify positions -> 2 pages, padded to the staging width
    MAX_PAGES_BATCH = 16

    def _ring_jobs(self):
        job0 = FakeJob([11, 12, 13, 13], drafted = 3,
                       block_pages = list(range(100, 100 + self.MAX_PAGES_BATCH)),
                       draft_block_pages = self.RING.table(0, self.MAX_PAGES_BATCH)[0].tolist())
        job1 = FakeJob([21, 22, 23, 23], drafted = 3,
                       block_pages = list(range(200, 200 + self.MAX_PAGES_BATCH)),
                       draft_block_pages = self.RING.table(1, self.MAX_PAGES_BATCH)[0].tolist())
        return job0, job1

    def test_dflash_refill_gets_the_staged_ring_table(self):
        job0, job1 = self._ring_jobs()
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        gen = self.run_round([job0, job1], [job0, job1], draft_tokens, rows = 2, width = 4,
                             ring = self.RING, dflash = True)
        dm = gen.draft_model
        assert len(dm.refills) == 1
        table = dm.refills[0]["params"]["block_table"]
        # One row per compact-batch sequence, at the batch's padded page width
        assert table.shape == (2, self.MAX_PAGES_BATCH)
        # Rows are the per-sequence ring tables, in batch order - built through the ring's
        # own page mapping, not the main pool's
        assert table[0].tolist() == job0.sequences[0].draft_block_index_tensor[0].tolist()
        assert table[1].tolist() == job1.sequences[0].draft_block_index_tensor[0].tolist()
        assert table[0].tolist() == self.RING.table(0, self.MAX_PAGES_BATCH)[0].tolist()
        assert table[1].tolist() == self.RING.table(1, self.MAX_PAGES_BATCH)[0].tolist()
        # ...and demonstrably not the main block_index the target forward was given
        main = gen.model.seen["params"]["block_table"]
        assert main[0].tolist() == job0.sequences[0].block_index_tensor[0].tolist()
        assert not torch.equal(table, main)
        assert dm.refills[0]["lengths"] == [4, 4]
        assert dm.refills[0]["cache"] is gen.draft_cache

    def test_dflash_refill_falls_back_to_the_main_table_without_a_ring(self):
        # draft_ring = None: the swap must hand the refill the main block_index itself -
        # a pool-wide draft cache mirrors the main page space one for one
        job0, job1 = self._ring_jobs()
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        gen = self.run_round([job0, job1], [job0, job1], draft_tokens, rows = 2, width = 4,
                             dflash = True)
        dm = gen.draft_model
        assert len(dm.refills) == 1
        table = dm.refills[0]["params"]["block_table"]
        assert table is gen.model.seen["params"]["block_table"]
        assert table[0].tolist() == job0.sequences[0].block_index_tensor[0].tolist()

    def test_mtp_refill_gets_its_own_ring_table_row(self):
        # MTP refills per job with a per-row slice; row a_idx:b_idx of the staged table
        # must be the sequence's own ring table, so a swapped row or the main table shows
        # up as foreign pages in that job's draft cache.
        job0, job1 = self._ring_jobs()
        draft_tokens = draft_buffer([[11, 12, 13], [21, 22, 23]], width = 4)
        export_states = [torch.zeros(2, 4, 3)]  # per-layer (rows, ids_width, hidden), as exported
        gen = self.run_round([job0, job1], [job0, job1], draft_tokens, rows = 2, width = 4,
                             ring = self.RING, mtp = True, export_states = export_states)
        dm = gen.draft_model
        assert len(dm.prefills) == 2
        for r, job in enumerate((job0, job1)):
            ids, params = dm.prefills[r]
            table = params["block_table"]
            assert table.shape == (1, self.MAX_PAGES_BATCH)
            assert table[0].tolist() == job.sequences[0].draft_block_index_tensor[0].tolist()
            assert table[0].tolist() != job.sequences[0].block_index_tensor[0].tolist()
            assert params["cache"] is gen.draft_cache
            # The refill advances the draft cache over the accepted positions only
            assert params["cache_seqlens"].tolist() == [job.sequences[0].kv_position + 1]
            assert ids.shape == (1, 3)
