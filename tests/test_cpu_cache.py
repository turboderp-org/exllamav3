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


def _wait_pinned(cache, timeout = 10.0):
    # The worker pins the whole budget in the background; wait until it has, so the test covers the state the
    # tier spends its life in (worker parked on the condition)
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        with cache._spare_cond:
            if len(cache._spare) + len(cache.slot_slabs) >= cache.max_slots:
                return
        time.sleep(0.01)
    raise AssertionError("pinning thread did not fill the spare pool")


def test_dropped_tier_releases_cache_tensors_and_worker():
    # The pinning thread used to run a bound method and wait on the condition forever once the budget was pinned,
    # which kept the tier alive after the Generator was dropped, and with it every GPU cache tensor in the
    # segment table and the whole pinned budget (one KV cache leaked per model reload with cpu_cache_size set)
    import gc, weakref
    cache = build(8)
    _wait_pinned(cache)
    thread = cache._alloc_thread
    tensor = cache.segments[0][0]
    ref_cache, ref_tensor = weakref.ref(cache), weakref.ref(tensor)
    del cache, tensor
    gc.collect()
    assert ref_cache() is None, "tier kept alive after its last reference was dropped"
    assert ref_tensor() is None, "GPU cache tensor kept alive by a dropped tier"
    thread.join(5.0)
    assert not thread.is_alive(), "pinning thread did not exit when the tier was collected"


def test_close_stops_worker_and_releases_slabs():
    cache = build(8)
    _wait_pinned(cache)
    cache.store(page(1, 0), serial = 1)
    thread = cache._alloc_thread
    cache.close()
    assert not thread.is_alive()
    assert not cache._spare and not cache.slot_slabs and not cache.segments and not cache.entries
