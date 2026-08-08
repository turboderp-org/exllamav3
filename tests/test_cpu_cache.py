import torch
from collections import namedtuple
from types import SimpleNamespace

from exllamav3.generator.cpu_cache import CPUPageCache

# The eviction policy is plain host-side bookkeeping over the slot table, so it can be driven directly with a
# stand-in cache: one paged tensor is enough to give the tier a slot size and something to copy.

PAGE_SIZE = 32
NUM_PAGES = 8


class FakeCacheLayer:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensors(self):
        return self.tensors


Page = namedtuple("Page", ["phash", "prev_hash", "page_index", "sequence"])


def page(tag, page_index, prev = None):
    return Page(bytes([tag]), prev, page_index, torch.full((PAGE_SIZE,), tag, dtype = torch.long))


def test_a_large_protect_set_does_not_livelock_a_full_cache():
    # A restore protects every page of the chain it is bringing back. When that set is larger than the eviction
    # order's rebuild interval, _evict_one used to defer a page, hit the rebuild threshold, forget it had
    # deferred anything, and start over -- spinning on the GIL forever with the GPU idle. At the interval's
    # floor of 64 pages, any prompt over ~16k tokens is enough once the tier is full.
    tensor = torch.zeros((NUM_PAGES, 64), dtype = torch.float16, device = "cuda")
    cache_obj = SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})

    slots = 200
    cache = CPUPageCache([cache_obj], slots * 4096)
    assert cache.max_slots == slots

    for tag in range(slots):
        cache.store(page(tag, tag % NUM_PAGES), serial = tag)
    assert len(cache.entries) == slots, "test needs a saturated cache to be meaningful"

    # Larger than _order_rebuild, which is what made the deferral counter reset before it could terminate
    protect = {bytes([tag]) for tag in range(slots)}
    assert len(protect) > cache._order_rebuild

    # Everything is protected, so the policy has to give up and take one anyway rather than spin
    slot = cache._evict_one(protect)

    assert 0 <= slot < slots
    assert len(cache.entries) == slots - 1
    assert cache.metrics["evictions"] == 1
