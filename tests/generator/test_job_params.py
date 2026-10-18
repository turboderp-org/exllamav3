"""
Job construction parameters and requeue bookkeeping: max_new_tokens is taken exactly (None defers to enqueue,
0 is rejected), a job holds one sequence, and a requeued job carries the token count of all earlier segments.
Checked against the documented Job contract, on CPU without a model.
"""

from unittest.mock import patch

import pytest
import torch

from exllamav3.generator.job import Job

pytestmark = pytest.mark.nogpu


def test_max_new_tokens_none_and_exact():
    ids = torch.tensor([[1, 2, 3]])
    assert Job(input_ids = ids).max_new_tokens is None                     # documented default: resolved at enqueue
    assert Job(input_ids = ids, max_new_tokens = None).max_new_tokens is None
    for k in (1, 2, 3, 17):
        assert Job(input_ids = ids, max_new_tokens = k).max_new_tokens == k   # no off-by-one, 1 != 2
    with pytest.raises(AssertionError):
        Job(input_ids = ids, max_new_tokens = 0)


def test_single_sequence_only():
    ids = torch.tensor([[1, 2, 3]])
    assert len(Job(input_ids = [ids]).sequences) == 1
    with pytest.raises(AssertionError):
        Job(input_ids = [ids, ids.clone()])


def test_requeue_carries_token_count():
    """new_tokens restarts in every requeued segment, so the count handed to the next segment has to include
    what earlier segments handed to this one"""
    job = Job(input_ids = torch.tensor([[1, 2, 3]]), max_new_tokens = 1000, max_rq_tokens = 256)
    job.cached_pages, job.cached_tokens = 0, 0
    total = 0
    with patch.object(Job, "prepare_for_queue"):
        for segment in (250, 256, 100):
            job.new_tokens = segment
            job.sequences[0].sequence_ids.append(torch.zeros((1, segment), dtype = torch.long))
            total += segment
            job = job.prepare_for_requeue()
            assert job.is_requeued
            assert job.new_tokens == 0
            assert job.rq_new_tokens == total
            assert job.max_new_tokens == 1000 - total
            assert job.rq_prompt_tokens == 3
