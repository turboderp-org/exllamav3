"""
Model.warmup(): the pass schedule must cover the GEMM autotuner's row buckets (1..16 and beyond),
a chunk, batched and single-row decode steps (with a history step only when the cache reserves
history), and a long-context family only when the cache can hold it; every family must stay within
the cache and release the recurrent-state slots it took; failing passes are reported, not raised.
Uses a stub model whose forward records the shapes it was asked for and asserts the per-step invariants
(family within the cache, state positions tracking past_len, history within max_history).
"""

from unittest.mock import patch

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.model.model import Model

pytestmark = pytest.mark.nogpu

# Row counts warmup() runs for the GEMM autotuner's buckets: 1..16 and the 16-row steps past it
# (32/48/64-row tiles, commit 3bbc7777)
ROW_BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64)


def rup(n):
    return -(-n // PAGE_SIZE) * PAGE_SIZE


class FakeState:
    def __init__(self, slot): self.slot = slot; self.position = 0


class FakeCache:
    def __init__(self, max_num_tokens, max_history = 0, num_slots = 16):
        self.max_num_tokens = max_num_tokens
        self.max_history = max_history
        self.free_list = list(range(num_slots)); self.taken = 0
    @property
    def free(self): return self.free_list
    def get_new_state(self):
        assert len(self.free_list) > 0, "Cannot create new state: no available slots"
        self.taken += 1
        return FakeState(self.free_list.pop(0))
    def release_state(self, s):
        self.free_list.append(s.slot)


class StubModel(Model):
    """Records (bsz, q, past_len, history) per forward; creates recurrent states like a recurrent model."""
    def __init__(self, vocab = 1000, fail_on = None, recurrent = True):
        self.caps = {}
        self.cache_weakrefs = {}
        self.config = type("C", (), {"vocab_size": vocab})()
        self.calls = []
        self.fail_on = fail_on
        self.recurrent = recurrent
    def get_recurrent_layers(self):
        return [object()] if self.recurrent else []
    def forward(self, input_ids, params):
        b, q = input_ids.shape
        assert input_ids.dtype == torch.long and int(input_ids.max()) < self.config.vocab_size
        assert params.get("tp_warmup") is True
        cache = params.get("cache")
        if cache is not None:
            cb, seq_cap = params["batch_shape"]
            assert seq_cap % PAGE_SIZE == 0 and cb * seq_cap <= cache.max_num_tokens, "family exceeds the cache"
            assert params["past_len"] + q <= seq_cap, "step runs past the family's rows"
            if self.recurrent:
                rs = params.get("recurrent_states")
                if rs is None:
                    assert params["past_len"] == 0
                    rs = [cache.get_new_state() for _ in range(b)]
                    params["recurrent_states"] = rs
                for r in rs:
                    assert r.position == params["past_len"], "state position must track past_len"
                if params.get("recurrent_history"):
                    assert q - 1 <= cache.max_history, "history step beyond the cache's max_history"
                for r in rs: r.position += q
        self.calls.append((b, q, params.get("past_len", 0), bool(params.get("recurrent_history"))))
        if self.fail_on and self.fail_on == (b, q):
            raise RuntimeError("boom")
        return torch.zeros(b, q, 8)


def test_schedule_covers_buckets_decode_and_long_context():
    m = StubModel(); c = FakeCache(16384, max_history = 15)
    fails = m.warmup(cache = c, max_chunk_size = 2048, max_batch_size = 8, max_q_len = 16)
    assert fails == []
    rows_at_zero = sorted(set(b * q for b, q, p, h in m.calls if p == 0))
    for n in ROW_BUCKETS + (2048,):
        assert n in rows_at_zero, f"missing row bucket {n}"
    assert (8, 1, 32, False) in m.calls, "batched one-token decode step"
    assert (8, 16, 33, True) in m.calls, "batched history step"
    assert (1, 1, 32, False) in m.calls
    assert (1, 64, 2048, False) in m.calls, "long-context step past the chunk"
    assert (1, 1, 2112, False) in m.calls, "long-context decode step"
    assert len(c.free) == 16, "recurrent-state slots must all be released"
    assert c.taken > 0


def test_history_step_respects_max_history():
    m = StubModel(); c = FakeCache(16384, max_history = 0)
    m.warmup(cache = c, max_chunk_size = 512)
    assert not any(h for *_, h in m.calls), "no history step without reserved history"
    m = StubModel(); c = FakeCache(16384, max_history = 3)
    m.warmup(cache = c, max_chunk_size = 512, max_q_len = 16)
    assert (8, 4, 33, True) in m.calls, "history step clamped to max_history + 1"


def test_small_cache_drops_families_it_cannot_hold():
    m = StubModel(); c = FakeCache(2 * PAGE_SIZE)
    fails = m.warmup(cache = c, max_chunk_size = 2048, max_batch_size = 8)
    assert fails == []
    for b, q, p, h in m.calls:
        assert b * rup(p + q) <= 2 * PAGE_SIZE, f"step (bsz {b}, q {q}, past {p}) does not fit the cache"
    assert not any(q == 2048 for b, q, p, h in m.calls), "chunk larger than the cache is clipped"
    assert any(q == 2 * PAGE_SIZE for b, q, p, h in m.calls), "chunk clipped to the cache"
    assert not any(b == 8 for b, q, p, h in m.calls), "batched family too large for the cache"
    assert not any(p >= 2 * PAGE_SIZE for b, q, p, h in m.calls), "no long-context family"


def test_no_cache_runs_cacheless_rows():
    m = StubModel(recurrent = False)
    fails = m.warmup(cache = None, max_chunk_size = 256)
    assert fails == []
    assert [q for b, q, p, h in m.calls] == list(ROW_BUCKETS) + [256]


def test_failure_is_reported_not_raised():
    m = StubModel(fail_on = (8, 1)); c = FakeCache(16384, max_history = 15)
    with patch("builtins.print"):
        fails = m.warmup(cache = c, max_chunk_size = 256)
    assert len(fails) == 1
    assert "batch 8 decode" in fails[0]
    assert (8, 16, 33, True) not in m.calls, "a failed family stops at the failing step"
    assert (1, 1, 32, False) in m.calls, "later families still run"
    assert len(c.free) == 16


def test_batched_family_bounded_by_state_slots():
    # chat.py-style cache: one recurrent-state slot. The batched family must be dropped (not
    # attempted and leaked), the single-row families still run, and every slot comes back
    m = StubModel(); c = FakeCache(16384, max_history = 15, num_slots = 1)
    with patch("builtins.print"):
        fails = m.warmup(cache = c, max_chunk_size = 2048, max_batch_size = 8)
    assert fails == []
    assert not any(b > 1 for b, q, p, h in m.calls)
    assert (1, 1, 32, False) in m.calls
    assert (1, 64, 2048, False) in m.calls, "long-context family still runs"
    assert len(c.free_list) == 1
    # Three slots: the batched family runs at batch 3
    m = StubModel(); c = FakeCache(16384, max_history = 15, num_slots = 3)
    m.warmup(cache = c, max_chunk_size = 256, max_batch_size = 8)
    assert (3, 1, 32, False) in m.calls
    assert len(c.free_list) == 3


def test_draft_models_are_skipped():
    m = StubModel(); m.caps["dflash2_draft"] = True
    assert m.warmup(cache = FakeCache(4096)) == []
    assert m.calls == []
