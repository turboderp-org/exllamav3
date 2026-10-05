import pytest
import torch
from collections import namedtuple
from types import SimpleNamespace

from exllamav3.generator.cpu_cache import CPUPageCache

# The eviction policy is plain host-side bookkeeping over the slot table, so it can be driven directly with a
# stand-in cache: one paged tensor is enough to give the tier a slot size and something to copy.

PAGE_SIZE = 32
NUM_PAGES = 8


@pytest.fixture(autouse = True)
def inference_mode():
    # The generator drives the tier under inference mode, and the slabs the background thread pins are
    # inference tensors; writing to them outside the mode is a RuntimeError
    with torch.inference_mode():
        yield


class FakeCacheLayer:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensors(self):
        return self.tensors


Page = namedtuple("Page", ["phash", "prev_hash", "page_index", "sequence"])


def page(tag, page_index, prev = None):
    return Page(tag.to_bytes(2, "big"), prev, page_index, torch.full((PAGE_SIZE,), tag, dtype = torch.long))


def build(slots):
    tensor = torch.zeros((NUM_PAGES, 64), dtype = torch.float16, device = "cuda")
    cache_obj = SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})
    cache = CPUPageCache([cache_obj], slots * 4096)
    assert cache.max_slots == slots
    return cache


def test_a_large_protect_set_does_not_livelock_a_full_cache():
    # A restore protects every page of the chain it is bringing back. When that set is larger than the eviction
    # order's rebuild interval, _evict_one used to defer a page, hit the rebuild threshold, forget it had
    # deferred anything, and start over -- spinning on the GIL forever with the GPU idle. At the interval's
    # floor of 64 pages, any prompt over ~16k tokens is enough once the tier is full.
    slots = 200
    cache = build(slots)

    for tag in range(slots):
        cache.store(page(tag, tag % NUM_PAGES), serial = tag)
    assert len(cache.entries) == slots, "test needs a saturated cache to be meaningful"

    # Larger than _order_rebuild, which is what made the deferral counter reset before it could terminate
    protect = {tag.to_bytes(2, "big") for tag in range(slots)}
    assert len(protect) > cache._order_rebuild

    # Everything is protected, so the policy has to give up and take one anyway rather than spin
    slot = cache._evict_one(protect)

    assert 0 <= slot < slots
    assert len(cache.entries) == slots - 1
    assert cache.metrics["evictions"] == 1


def test_a_large_protect_set_does_not_claim_protected_pages():
    # The counterpart of the livelock test: the protect set again outlasts the rebuild interval, but this time
    # unprotected entries exist behind the protected run. The loop must walk past the whole run and take one of
    # those -- not trip the rebuild threshold on deferrals and eventually give up on a page the restore in
    # progress is about to fetch.
    slots = 400
    n_protect = 300
    cache = build(slots)

    # The protected chain is oldest (lowest serials), so the eviction order leads with it
    prev = None
    for tag in range(n_protect):
        cache.store(page(tag, tag % NUM_PAGES, prev), serial = tag)
        prev = tag.to_bytes(2, "big")
    for tag in range(n_protect, slots):
        cache.store(page(tag, tag % NUM_PAGES), serial = tag)
    assert len(cache.entries) == slots

    protect = {tag.to_bytes(2, "big") for tag in range(n_protect)}
    assert len(protect) > cache._order_rebuild

    cache._evict_one(protect)

    assert all(h in cache.entries for h in protect), "evicted a page the allocation in progress claimed"
    assert len(cache.entries) == slots - 1
    assert cache.metrics["evictions"] == 1


def test_dropped_tier_releases_cache_tensors_and_arena():
    # A tier dropped without close() (the Generator going away) must not keep the GPU cache tensors in its
    # segment table or its pinned arena alive
    import gc, weakref
    cache = build(8)
    cache.store(page(1, 0), serial = 1)
    tensor = cache.segments[0][0]
    unregister = cache.arena._unregister
    ref_cache, ref_tensor, ref_arena = weakref.ref(cache), weakref.ref(tensor), weakref.ref(cache.arena)
    del cache, tensor
    gc.collect()
    assert ref_cache() is None, "tier kept alive after its last reference was dropped"
    assert ref_tensor() is None, "GPU cache tensor kept alive by a dropped tier"
    assert ref_arena() is None and not unregister.alive, "arena still locked after the tier was collected"


def test_close_releases_arena():
    cache = build(8)
    cache.store(page(1, 0), serial = 1)
    arena = cache.arena
    cache.close()
    assert arena.map is None and not arena.pinned
    assert cache.arena is None and not cache.slot_views and not cache.segments and not cache.entries


def test_tier_occupies_exactly_its_budget():
    # The slabs are slices of one registered arena. Pinned tensors from torch's allocator would each be
    # rounded up to a power of two (and kept by it after the tier is gone), so none may come from there
    torch._C._host_emptyCache()
    before = torch.cuda.host_memory_stats()["allocated_bytes.current"]
    tensor = torch.zeros((NUM_PAGES, 1536 + 64), dtype = torch.float16, device = "cuda")   # 3200-byte pages
    cache = CPUPageCache([SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})], 5 * 4096)
    assert cache.arena.size == cache.max_slots * cache.slab_size == 5 * 4096
    for i in range(cache.max_slots + 2):   # past the budget: slots are recycled, never added
        cache.store(page(i + 1, i % NUM_PAGES), serial = i + 1)
    assert len(cache.slot_views) == cache.max_slots
    assert torch.cuda.host_memory_stats()["allocated_bytes.current"] == before
    cache.close()


def test_slots_are_pinned_and_round_trip():
    # Views into the arena must take the asynchronous copy path, and every slot must come back intact
    tensor = torch.zeros((NUM_PAGES, 64), dtype = torch.float16, device = "cuda")
    cache = CPUPageCache([SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})], NUM_PAGES * 4096)
    for i in range(NUM_PAGES):
        tensor[i] = i + 1
        cache.store(page(i + 1, i), serial = i + 1)
    assert cache.arena.pinned and all(v.is_pinned() for views in cache.slot_views for v in views)
    tensor.zero_()
    for i in range(NUM_PAGES):
        cache.fetch(page(i + 1, i).phash, NUM_PAGES - 1 - i, serial = 100 + i)
    torch.cuda.synchronize()
    assert torch.equal(tensor[:, 0].cpu(), torch.arange(NUM_PAGES, 0, -1, dtype = torch.float16))
    cache.close()
