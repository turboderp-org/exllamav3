"""
Job construction parameters and requeue bookkeeping: max_new_tokens is taken exactly (None defers to enqueue,
0 is rejected), a job holds one sequence, and a requeued job carries the token count of all earlier segments.
Checked against the documented Job contract, on CPU without a model.
"""

from unittest.mock import patch

import pytest
import torch

from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import PAGE_SIZE

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


def _queued_job(mtp_draft, prompt_len = 100, max_rq_tokens = 1900):
    """A single-sequence job queued on a stand-in generator with a recurrent cache (2048-token checkpoints)"""
    from types import SimpleNamespace
    generator = SimpleNamespace(
        max_total_tokens = 65536, max_draft_tokens = 3, recurrent_checkpoint_interval = 2048,
        recurrent_cache = object(), mtp_draft = mtp_draft, pagetable = SimpleNamespace(max_pages = 1 << 20),
        max_batch_size = 1, padded_vocab_size = 32, ngram_match_min = 0, hybrid_draft_tokens = 0,
        ngram_corpus = None, model = SimpleNamespace(caps = {}),
    )
    job = Job(input_ids = torch.zeros((1, prompt_len), dtype = torch.long), max_new_tokens = 10000,
              max_rq_tokens = max_rq_tokens)
    job.prepare_for_queue(generator, 0)
    return job, generator


def _reserved(job):
    """Tokens the queued job reserved pages for: the segment plus the speculative window written past its end"""
    return len(job.sequences[0].input_ids) + job.max_rq_tokens + job.rq_headroom


def _first_requeue(job):
    """Sequence length at the first accepted token that asks for a requeue"""
    n = 0
    while True:
        n += 1
        job.new_tokens = n
        if job.requeue_due():
            return len(job.sequences[0].input_ids) + n


def test_mtp_requeue_keeps_the_boundary_page_cached():
    """An MTP job's next segment caches all but the last two tokens of its prompt, so a requeue exactly on the
    aligned checkpoint boundary would leave the boundary page uncached and replay back to the previous checkpoint"""
    job, generator = _queued_job(mtp_draft = True)
    seq_len = _first_requeue(job)
    assert (seq_len - 2) // PAGE_SIZE * PAGE_SIZE == 2048
    assert _reserved(job) >= seq_len - 1 + generator.max_draft_tokens


def test_non_mtp_requeue_lands_on_the_boundary():
    job, generator = _queued_job(mtp_draft = False)
    seq_len = _first_requeue(job)
    assert seq_len - 1 == 2048
    assert _reserved(job) >= seq_len - 1 + generator.max_draft_tokens


@pytest.mark.parametrize("mtp_draft", [False, True])
def test_final_segment_fits_the_cache(mtp_draft):
    """The final segment plus everything reserved past it (draft window, MTP token) ends exactly at the end of the
    cache, and never asks for a requeue there"""
    job, generator = _queued_job(mtp_draft, prompt_len = 65536 - 1000)
    assert _reserved(job) == generator.max_total_tokens
    job.new_tokens = job.max_new_tokens
    assert not job.requeue_due()
