import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from exllamav3.generator.generator import draft_row_map


# The draft pass emits one draft row per drafting job, in its active-set order. iterate_gen must
# hand each drafting job its own row and no row to a job absent from the draft active set.
# draft_row_map is the alignment between the two: a rowless job ahead of a drafting job must
# not shift the drafting job's row (the P1 regression - a shifted row verifies a foreign job's
# draft, and every job behind it reads a shifted row too).
# Note: Generator._draft_active_jobs is currently exactly the prefill-done set, so the rowless
# branch is defensive-only (it fires if a future filter ever drops a job from the draft
# active set); these tests pin the alignment that keeps it safe.

class FakeJob:
    def __init__(self, prefill_done = True):
        self._done = prefill_done

    def is_prefill_done(self):
        return self._done


class TestDraftRowMap:

    def test_all_drafting_jobs_get_sequential_rows(self):
        jobs = [FakeJob() for _ in range(4)]
        rows = draft_row_map(jobs, jobs)
        assert [rows[id(j)] for j in jobs] == [0, 1, 2, 3]

    def test_rowless_job_gets_no_row_and_does_not_shift_the_rest(self):
        # The regression: a rowless job (prefill-done, absent from the draft active set) ahead of a
        # drafting job must not consume a row, so the drafting job keeps row 0
        jobs = [FakeJob(), FakeJob(), FakeJob()]
        active = [jobs[0], jobs[2]]  # jobs[1] is rowless
        rows = draft_row_map(jobs, active)
        assert rows[id(jobs[0])] == 0
        assert rows[id(jobs[1])] is None
        assert rows[id(jobs[2])] == 1  # not 2 - the rowless job did not take a row

    def test_rowless_job_at_the_end(self):
        jobs = [FakeJob(), FakeJob()]
        active = [jobs[0]]
        rows = draft_row_map(jobs, active)
        assert rows[id(jobs[0])] == 0
        assert rows[id(jobs[1])] is None

    def test_prefill_incomplete_jobs_are_skipped(self):
        jobs = [FakeJob(prefill_done = False), FakeJob(), FakeJob(prefill_done = False), FakeJob()]
        active = [jobs[1], jobs[3]]
        rows = draft_row_map(jobs, active)
        assert id(jobs[0]) not in rows
        assert rows[id(jobs[1])] == 0
        assert id(jobs[2]) not in rows
        assert rows[id(jobs[3])] == 1

    def test_all_rowless_gives_no_rows(self):
        jobs = [FakeJob(), FakeJob()]
        rows = draft_row_map(jobs, [])
        assert rows[id(jobs[0])] is None
        assert rows[id(jobs[1])] is None

    def test_empty(self):
        assert draft_row_map([], []) == {}

    def test_active_entry_outside_the_jobs_raises_instead_of_misaligning(self):
        # The guard on the alignment invariant: an active entry that is not one of the jobs at
        # the cursor (e.g. a stale job) must raise, not silently map every drafting job to None
        jobs = [FakeJob(), FakeJob()]
        with pytest.raises(AssertionError, match = "subsequence"):
            draft_row_map(jobs, [FakeJob()])
