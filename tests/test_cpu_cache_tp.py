import pytest
import torch
from collections import namedtuple
from types import SimpleNamespace

from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.model.model_tp_fn import (
    mp_cpu_cache_init,
    mp_cpu_cache_store,
    mp_cpu_cache_fetch,
)

# In TP mode the main process holds no cache tensors, so the whole path can be driven on the host: the ranks
# below are ordinary dicts standing in for the worker processes' local_context, and the fake model routes the
# dispatch calls straight into the same functions the workers run. What that leaves untested is the IPC and the
# CUDA stream ordering; what it does test is the part this patch actually adds, which is the main process
# handing out slot indices that every rank agrees on, and the data surviving a round trip through them.

PAGE_SIZE = 32
NUM_PAGES = 8


class FakeCacheLayer:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensors(self):
        return self.tensors


class FakeModule:
    def __init__(self, lookup):
        self.tp_cache_lookup = lookup


def make_rank(cache_ids, shards, num_pages = NUM_PAGES):
    """One worker's local_context: a shard of each cache, page-major, filled with a rank-specific pattern."""

    kv_modules = []
    for width in shards:
        lookup = {}
        for cache_id in cache_ids:
            t = torch.zeros((num_pages, width), dtype = torch.float16)
            lookup[cache_id] = FakeCacheLayer([t, None])  # a None tensor must be skipped, as in a real layer
        kv_modules.append(FakeModule(lookup))
    return {"kv_modules": kv_modules, "cpu_page_cache": None}


class FakeTPModel:
    """Dispatches to the rank functions in-process, in place of the worker pipes."""

    loaded_tp = True

    def __init__(self, ranks):
        self.ranks = ranks
        self.dispatches = 0
        self.caches = []

    def cache(self):
        """A cache belonging to this model, which is what decides where its transfers are dispatched."""
        cache = SimpleNamespace(layers = {}, model = self)
        self.caches.append(cache)
        return cache

    def tp_cpu_cache_init(self, cache_ids, max_slots = 0):
        return sum(mp_cpu_cache_init(r, cache_ids, max_slots) for r in self.ranks)

    def tp_cpu_cache_store(self, cache_ids, slot, page_index):
        self.dispatches += 1
        return sum(mp_cpu_cache_store(r, cache_ids, slot, page_index) for r in self.ranks)

    def tp_cpu_cache_fetch(self, cache_ids, slot, page_index):
        self.dispatches += 1
        for r in self.ranks:
            mp_cpu_cache_fetch(r, cache_ids, slot, page_index)


Page = namedtuple("Page", ["phash", "prev_hash", "page_index", "sequence"])


def page(tag, page_index, prev = None):
    return Page(bytes([tag]), prev, page_index, torch.full((PAGE_SIZE,), tag, dtype = torch.long))


def rank_tensors(rank, cache_id):
    return [m.tp_cache_lookup[cache_id].get_tensors()[0] for m in rank["kv_modules"]]


def fill(ranks, cache_ids, page_index, value):
    for rank_no, rank in enumerate(ranks):
        for cache_no, cache_id in enumerate(cache_ids):
            for module_no, t in enumerate(rank_tensors(rank, cache_id)):
                t[page_index] = value + rank_no * 100 + cache_no * 10 + module_no


def snapshot(ranks, cache_ids, page_index):
    return [
        t[page_index].clone()
        for rank in ranks
        for cache_id in cache_ids
        for t in rank_tensors(rank, cache_id)
    ]


def same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.fixture(autouse = True)
def inference_mode():
    # The generator and the TP workers both drive the tier under inference mode, and the pinned slabs the
    # background threads produce are inference tensors; writing to them outside the mode is a RuntimeError
    with torch.inference_mode():
        yield


@pytest.fixture
def tp_cache():
    """One TP model over two ranks with uneven shards, holding two caches, two cache modules each."""

    model = FakeTPModel(None)
    caches = [model.cache(), model.cache()]
    # The ranks are keyed by id() of the cache objects, which is what a real cache layer carries as its cache_id
    cache_ids = [id(cache) for cache in caches]
    model.ranks = [make_rank(cache_ids, [64, 48]), make_rank(cache_ids, [32, 16])]
    return model.ranks, cache_ids, model, caches


def build(model, caches, max_size):
    return CPUPageCache(caches, max_size)


def test_slot_size_is_the_page_summed_over_ranks(tp_cache):
    ranks, cache_ids, model, caches = tp_cache
    cache = build(model, caches, 64 * 4096)

    # (64 + 48 + 32 + 16) shard widths x 2 caches x 2 bytes, rounded up to the 4k slot alignment
    assert cache.slot_size == 4096
    assert cache.max_slots == 64


def test_a_page_survives_the_round_trip_on_every_rank(tp_cache):
    ranks, cache_ids, model, caches = tp_cache
    cache = build(model, caches, 64 * 4096)

    fill(ranks, cache_ids, 3, 7.0)
    original = snapshot(ranks, cache_ids, 3)
    cache.store(page(1, 3), serial = 1)

    # The page is destroyed in VRAM, then restored into a different page slot
    fill(ranks, cache_ids, 3, 0.0)
    fill(ranks, cache_ids, 5, 0.0)
    entry = cache.fetch(bytes([1]), 5, serial = 2)

    assert same(snapshot(ranks, cache_ids, 5), original)
    assert entry["tokens"].tolist() == [1] * PAGE_SIZE


def test_every_rank_stores_the_same_slot(tp_cache):
    # The main process owns the slot table; a rank disagreeing about which slot a page lives in would restore
    # one rank's shard into another rank's page
    ranks, cache_ids, model, caches = tp_cache
    cache = build(model, caches, 64 * 4096)

    for tag in range(4):
        cache.store(page(tag, tag), serial = tag)

    slots = [e["slot"] for e in cache.entries.values()]
    assert sorted(slots) == [0, 1, 2, 3]
    for rank in ranks:
        assert sorted(rank["cpu_page_cache"].slots) == [0, 1, 2, 3]


def test_eviction_recycles_the_slot_on_every_rank(tp_cache):
    ranks, cache_ids, model, caches = tp_cache
    # Room for exactly two pages
    cache = build(model, caches, 2 * 4096)
    assert cache.max_slots == 2

    cache.store(page(1, 0), serial = 1)
    cache.store(page(2, 1), serial = 2)
    fill(ranks, cache_ids, 2, 9.0)
    expected = snapshot(ranks, cache_ids, 2)
    cache.store(page(3, 2), serial = 3)

    assert len(cache.entries) == 2, "the budget must be a ceiling, not a suggestion"
    assert cache.metrics["evictions"] == 1
    for rank in ranks:
        assert len(rank["cpu_page_cache"].slots) == 2, "a rank kept buffers for an evicted slot"

    # The recycled slot holds the newest page on every rank, not the evicted one
    fill(ranks, cache_ids, 6, 0.0)
    cache.fetch(bytes([3]), 6, serial = 4)
    assert same(snapshot(ranks, cache_ids, 6), expected)


def test_duplicate_store_does_not_dispatch(tp_cache):
    # Deduplication happens in the main process, so a repeat store must not cost a round trip to the ranks
    ranks, cache_ids, model, caches = tp_cache
    cache = build(model, caches, 64 * 4096)

    cache.store(page(1, 0), serial = 1)
    before = model.dispatches
    cache.store(page(1, 0), serial = 2)

    assert model.dispatches == before
    assert cache.metrics["dedup_hits"] == 1


def test_ranks_pin_their_buffers(tp_cache):
    ranks, cache_ids, model, caches = tp_cache
    cache = build(model, caches, 64 * 4096)
    cache.store(page(1, 0), serial = 1)

    for rank in ranks:
        buffers = rank["cpu_page_cache"].slots[0]
        assert buffers, "rank allocated no buffers for the slot"
        if not rank["cpu_page_cache"].pageable:
            assert all(b.is_pinned() for b in buffers)


def test_a_draft_cache_on_its_own_model_is_dispatched_separately():
    # The failure this exists for: a draft cache belongs to the draft model, which has its own workers and its
    # own view of which cache ids exist. Dispatching both caches to the main model's ranks raised KeyError on
    # the draft cache id, and skipping it instead would have restored pages whose draft KV was stale.
    main_model = FakeTPModel(None)
    draft_model = FakeTPModel(None)
    main_cache, draft_cache = main_model.cache(), draft_model.cache()
    main_model.ranks = [make_rank([id(main_cache)], [64, 48]), make_rank([id(main_cache)], [32, 16])]
    draft_model.ranks = [make_rank([id(draft_cache)], [8]), make_rank([id(draft_cache)], [8])]

    cache = CPUPageCache([main_cache, draft_cache], 64 * 4096)

    # One page image spans both models: (64 + 48 + 32 + 16) + (8 + 8) widths, x 2 bytes
    assert cache.slot_size == 4096
    assert len(cache.tp_groups) == 2

    for model, cache_id in ((main_model, id(main_cache)), (draft_model, id(draft_cache))):
        fill(model.ranks, [cache_id], 2, 5.0)
    original = [
        snapshot(m.ranks, [cid], 2)
        for m, cid in ((main_model, id(main_cache)), (draft_model, id(draft_cache)))
    ]

    cache.store(page(1, 2), serial = 1)
    for model, cache_id in ((main_model, id(main_cache)), (draft_model, id(draft_cache))):
        fill(model.ranks, [cache_id], 4, 0.0)
    cache.fetch(bytes([1]), 4, serial = 2)

    # Both models' shards come back, and both used the same slot index
    for (model, cache_id), expected in zip(
        ((main_model, id(main_cache)), (draft_model, id(draft_cache))), original
    ):
        assert same(snapshot(model.ranks, [cache_id], 4), expected)
        for rank in model.ranks:
            assert list(rank["cpu_page_cache"].slots) == [0]


def test_a_local_draft_cache_shares_the_slot_with_a_tp_main_cache():
    # The configuration this exists for: a TP main model whose MTP draft model, being a single layer, is not
    # TP-loaded. Its cache tensors are here in the main process while the main cache's are in the workers, and
    # both halves of the page have to travel together under one slot index.
    main_model = FakeTPModel(None)
    main_cache = main_model.cache()
    main_model.ranks = [make_rank([id(main_cache)], [64, 48]), make_rank([id(main_cache)], [32, 16])]

    local_model = SimpleNamespace(loaded_tp = False)
    draft_tensor = torch.zeros((NUM_PAGES, 24), dtype = torch.float16, device = "cuda")
    draft_cache = SimpleNamespace(
        model = local_model,
        layers = {0: FakeCacheLayer([draft_tensor, None])},
    )

    cache = CPUPageCache([main_cache, draft_cache], 64 * 4096)
    assert len(cache.tp_groups) == 1
    assert cache.segments, "the local draft cache must still be copied here"
    assert cache.slot_size == 4096

    fill(main_model.ranks, [id(main_cache)], 2, 5.0)
    draft_tensor[2] = 3.0
    tp_original = snapshot(main_model.ranks, [id(main_cache)], 2)

    cache.store(page(1, 2), serial = 1)
    fill(main_model.ranks, [id(main_cache)], 4, 0.0)
    draft_tensor[4] = 0.0
    cache.fetch(bytes([1]), 4, serial = 2)

    assert same(snapshot(main_model.ranks, [id(main_cache)], 4), tp_original)
    assert torch.equal(draft_tensor[4], torch.full((24,), 3.0, dtype = torch.float16, device = "cuda"))
