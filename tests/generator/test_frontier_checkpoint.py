"""
Frontier checkpoints: the recurrent state at the exact end of a prompt or request.

Job.frontier_checkpoint() stores the state at the sequence's current position. On a page boundary that is an
ordinary checkpoint; inside a page it stores the partially filled page in the page tiers under the content
hash of the prefix and the state under the same key, so a later request that extends exactly that prefix can
resume there. These tests drive the job method with stand-in pages, a stand-in page table and a stand-in
recurrent cache, without a model.
"""

from types import SimpleNamespace

import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.job import Job
from exllamav3.generator.pagetable import tensor_hash_checksum
from exllamav3.util.tensor import SeqTensor

K = 3
OFFSET = 100


class FakePagetable:
    cpu_tier = object()

    def __init__(self, store = True):
        self.store = store
        self.calls = []

    def store_partial_page(self, page, key, protect = None):
        self.calls.append((page, key, protect))
        return self.store


class FakeCache:
    def __init__(self):
        self.puts = []
        self.partials = []

    def put(self, key, state):
        self.puts.append((key, state))

    def put_partial(self, key, state, anchor, offset):
        self.partials.append((key, state, anchor, offset))


def fake_page(phash, prev_hash = None, kv_position = PAGE_SIZE, tokens = None):
    if tokens is None:
        tokens = torch.zeros((1, PAGE_SIZE), dtype = torch.long)
    return SimpleNamespace(phash = phash, prev_hash = prev_hash, kv_position = kv_position,
                           sequence = tokens, persistable = True)


def fake_job(pos, pages, state_position = None, enabled = True, cpu_tier = True, store = True):
    tokens = torch.arange(0, pos + 1, dtype = torch.long)
    job = Job.__new__(Job)
    job.sequences = [SimpleNamespace(
        kv_position = pos,
        allocated_pages = pages,
        sequence_ids = SeqTensor.from_tensor(tokens.unsqueeze(0), seq_dim = -1),
    )]
    job.recurrent_state = SimpleNamespace(
        position = pos if state_position is None else state_position,
        checkpoint_size = 64,
        stash = lambda: {"position": job.recurrent_state.position, "checkpoint_size": 64},
    )
    job.last_recurrent_checkpoint_pos = None
    job.new_tokens = 0
    job.pagetable = FakePagetable(store = store)
    if not cpu_tier:
        job.pagetable.cpu_tier = None
    job.generator = SimpleNamespace(recurrent_frontier_checkpoints = enabled)
    return job


def complete_pages(count = K):
    pages = []
    prev = None
    for i in range(count):
        tokens = torch.arange(i * PAGE_SIZE, (i + 1) * PAGE_SIZE, dtype = torch.long)
        h = tensor_hash_checksum(tokens, prev)
        pages.append(fake_page(h, prev, PAGE_SIZE, tokens.unsqueeze(0)))
        prev = h
    return pages


def test_page_aligned_frontier_is_an_ordinary_checkpoint():
    pages = complete_pages()
    job = fake_job(K * PAGE_SIZE, pages)
    cache = FakeCache()

    assert job.frontier_checkpoint(cache)
    assert cache.puts == [(pages[-1].phash, job.recurrent_state)]
    assert cache.partials == []
    assert job.last_recurrent_checkpoint_pos == K * PAGE_SIZE


def test_mid_page_frontier_stores_the_partial_page_under_its_prefix_hash():
    pages = complete_pages()
    anchor = pages[-1].phash
    pos = K * PAGE_SIZE + OFFSET
    partial_tokens = torch.arange(K * PAGE_SIZE, pos, dtype = torch.long)
    partial = fake_page(
        b"\x99" * 16, anchor, OFFSET,
        torch.cat((partial_tokens, torch.zeros(PAGE_SIZE - OFFSET, dtype = torch.long))).unsqueeze(0),
    )
    job = fake_job(pos, pages + [partial])
    cache = FakeCache()

    assert job.frontier_checkpoint(cache)
    key, state, got_anchor, got_offset = cache.partials[0]
    assert key == tensor_hash_checksum(partial_tokens, anchor)
    assert state is job.recurrent_state
    assert got_anchor == anchor and got_offset == OFFSET
    assert cache.puts == []
    assert job.pagetable.calls[0][1] == key
    assert anchor in job.pagetable.calls[0][2], "the anchor page must be protected during the store"
    assert job.last_recurrent_checkpoint_pos == pos


def test_frontier_requires_state_and_kv_to_describe_the_same_prefix():
    pages = complete_pages()
    job = fake_job(K * PAGE_SIZE, pages, state_position = K * PAGE_SIZE - 1)
    assert not job.frontier_checkpoint(FakeCache())


def test_frontier_is_disabled_by_the_generator_option():
    job = fake_job(K * PAGE_SIZE, complete_pages(), enabled = False)
    assert not job.frontier_checkpoint(FakeCache())


def test_mid_page_frontier_needs_a_cpu_tier():
    pages = complete_pages()
    partial = fake_page(b"\x99" * 16, pages[-1].phash, OFFSET,
                        torch.zeros((1, PAGE_SIZE), dtype = torch.long))
    job = fake_job(K * PAGE_SIZE + OFFSET, pages + [partial], cpu_tier = False)
    assert not job.frontier_checkpoint(FakeCache())


def test_mid_page_frontier_skips_a_deferred_anchor():
    # A random (deferred) anchor hash has no stable identity across requests, so no exact-position checkpoint
    pages = complete_pages()
    pages[-1].phash = b"\x00" * 8 + b"\x01" * 8
    partial = fake_page(b"\x99" * 16, pages[-1].phash, OFFSET,
                        torch.zeros((1, PAGE_SIZE), dtype = torch.long))
    job = fake_job(K * PAGE_SIZE + OFFSET, pages + [partial])
    assert not job.frontier_checkpoint(FakeCache())


def test_mid_page_frontier_skips_a_broken_page_link():
    pages = complete_pages()
    partial = fake_page(b"\x99" * 16, b"\x77" * 16, OFFSET,
                        torch.zeros((1, PAGE_SIZE), dtype = torch.long))
    job = fake_job(K * PAGE_SIZE + OFFSET, pages + [partial])
    assert not job.frontier_checkpoint(FakeCache())


def test_a_failed_page_store_does_not_leave_a_dangling_state():
    pages = complete_pages()
    partial = fake_page(b"\x99" * 16, pages[-1].phash, OFFSET,
                        torch.zeros((1, PAGE_SIZE), dtype = torch.long))
    job = fake_job(K * PAGE_SIZE + OFFSET, pages + [partial], store = False)
    cache = FakeCache()
    assert not job.frontier_checkpoint(cache)
    assert cache.partials == []


def test_prompt_end_is_captured_mid_page():
    # maybe_stash_recurrent() is called after every prefill round; with no token sampled yet and prefill done,
    # it captures the exact prompt end through the frontier path
    pages = complete_pages()
    anchor = pages[-1].phash
    pos = K * PAGE_SIZE + OFFSET
    partial_tokens = torch.arange(K * PAGE_SIZE, pos, dtype = torch.long)
    partial = fake_page(
        b"\x99" * 16, anchor, OFFSET,
        torch.cat((partial_tokens, torch.zeros(PAGE_SIZE - OFFSET, dtype = torch.long))).unsqueeze(0),
    )
    job = fake_job(pos, pages + [partial])
    cache = FakeCache()

    job.maybe_stash_recurrent(cache)
    assert len(cache.partials) == 1
    assert cache.partials[0][0] == tensor_hash_checksum(partial_tokens, anchor)
