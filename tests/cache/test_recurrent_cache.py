"""
Recurrent checkpoint storage (cache/recurrent.py). HostPool: stashes copy into pooled host buffers keyed by
exact shape and dtype that eviction returns, so the allocator sees no per-checkpoint churn (issue #432), and
host_copy matches a plain .cpu() for strided device sources. RecurrentCache.close(): drops every checkpoint,
counts a stash shared by two keys once, returns the buffers and empties the pool so a replacement cache
allocates fresh memory. Checked against the pool's allocation/reuse counters and buffer identity.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache import recurrent
from exllamav3.cache.recurrent import HostPool, RecurrentCache, host_copy, host_pool


@pytest.fixture(autouse = True)
def fresh_pool(monkeypatch):
    # host_pool is process-global: start every test from an empty free list and freed-bytes counter
    monkeypatch.setattr(recurrent, "_freed_bytes", 0)
    host_pool.free.clear()
    yield
    host_pool.free.clear()


def free_count(pool):
    return sum(len(v) for v in pool.free.values())


# HostPool

@pytest.mark.nogpu
def test_take_reuses_exact_shape_and_dtype():
    pool = HostPool()
    a = pool.take((4, 8), torch.float32)
    b = pool.take((4, 8), torch.float32)
    assert pool.allocated == 2
    pool.give(a)
    c = pool.take((4, 8), torch.float32)
    assert c is a
    assert (pool.allocated, pool.reused) == (2, 1)
    # neither a different shape nor a different dtype may be served from a free buffer
    pool.give(c)
    d = pool.take((8, 4), torch.float32)
    e = pool.take((4, 8), torch.float16)
    assert pool.allocated == 4
    assert d.shape == (8, 4) and e.dtype == torch.float16
    assert e is not c
    assert pool.take((4, 8), torch.float32) is c


@pytest.mark.nogpu
def test_give_walks_a_stashed_structure():
    pool = HostPool()
    t = [pool.take((3,), torch.float32) for _ in range(4)]
    stashed = {"position": 2048, "checkpoint_size": 12345, "tp_handle": 7,
               "layer_a": (t[0], t[1]), "layer_b": [t[2], (t[3],)]}
    pool.give(stashed)
    assert free_count(pool) == 4
    ids = {id(x) for x in t}
    for _ in range(4):
        assert id(pool.take((3,), torch.float32)) in ids
    assert pool.reused == 4


def test_give_does_not_pool_device_tensors(device):
    pool = HostPool()
    pool.give(torch.zeros(3, device = device))
    assert free_count(pool) == 0


@pytest.mark.nogpu
def test_release_drops_idle_buffers_only():
    pool = HostPool()
    held = pool.take((5,), torch.float32)
    idle = pool.take((5,), torch.float32)
    pool.give(idle)
    pool.release()
    assert free_count(pool) == 0
    assert pool.take((5,), torch.float32) is not idle
    held.fill_(1.0)                       # the buffer still in use is untouched


def test_host_copy_matches_cpu_for_strided_sources(device):
    before = host_pool.allocated
    state = torch.randn(4, 6, 16, 16, device = device)
    conv = torch.randn(4, 24, 8, device = device)
    for src in (state[2, :1], conv[1, :, :4], state[3]):
        out = host_copy(src)
        assert out.device.type == "cpu"
        assert torch.equal(out, src.cpu())
        assert out.shape == src.shape
    assert host_pool.allocated == before + 3
    # a second stash of the same shapes reuses the buffers once the first is returned
    host_pool.give([host_copy(state[0, :1])])
    host_copy(state[1, :1])
    assert host_pool.reused >= 1
    host_pool.release()


# RecurrentCache.close()

class FakeState:
    """Stand-in for a recurrent layer state: stash() copies two tensors into pooled host buffers."""

    def __init__(self, position, rows = 64):
        self.position = position
        self.rows = rows
        self.checkpoint_size = 2 * rows * 1024 * 2

    def stash(self):
        return {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
            "a": recurrent.host_copy(torch.ones(self.rows, 1024, dtype = torch.float16)),
            "b": recurrent.host_copy(torch.ones(self.rows, 1024, dtype = torch.float16)),
        }


def make_cache(max_size = 10 * 1024**2):
    return RecurrentCache(SimpleNamespace(loaded_tp = False), max_size)


@pytest.mark.nogpu
def test_close_drops_checkpoints_and_releases_the_pool():
    cache = make_cache()
    for i in range(5):
        cache.put(i, FakeState(position = i * 256))
    assert len(cache) == 5 and cache.current_size == 5 * FakeState(0).checkpoint_size
    cache.pagetable = object()

    cache.close()

    assert len(cache) == 0
    assert cache.current_size == 0
    assert cache.pagetable is None
    # The buffers were handed back to the pool and the pool emptied, so the RAM is free
    assert host_pool.free == {}
    assert cache.get_stashed(0) is None


@pytest.mark.nogpu
def test_close_is_idempotent_and_the_cache_stays_usable_for_bookkeeping():
    cache = make_cache()
    cache.put(1, FakeState(position = 0))
    cache.close()
    cache.close()
    assert len(cache) == 0 and cache.current_size == 0


@pytest.mark.nogpu
def test_close_counts_a_shared_stash_once():
    cache = make_cache()
    cache.put(1, FakeState(position = 0))
    cache[2] = cache[1]  # two keys, one stash
    cache.update_total_size()
    assert cache.current_size == FakeState(0).checkpoint_size

    cache.close()
    assert len(cache) == 0 and host_pool.free == {}


@pytest.mark.nogpu
def test_closed_cache_frees_memory_for_a_replacement():
    """A new cache's stashes don't come out of the old cache's buffers once they are released."""
    allocated, reused = host_pool.allocated, host_pool.reused
    old = make_cache()
    old.put(1, FakeState(position = 0))
    old.close()
    assert host_pool.allocated == allocated + 2

    new = make_cache()
    new.put(1, FakeState(position = 0))
    # Allocated fresh, since the pool was emptied and the old buffers' memory returned
    assert host_pool.allocated == allocated + 4 and host_pool.reused == reused
