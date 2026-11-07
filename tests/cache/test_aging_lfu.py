"""
Aging-LFU prompt-cache weights: the SQLite sidecar (generator/aging_lfu.py), the page table's hit credit and
eviction ordering, and the CPU tier ordering it inherits.

The policy is pure host-side bookkeeping over page records, so these drive a real PageTable over a stand-in
cache (testlib.cache) and a real CPUPageCache over a stand-in cache layer. No model is loaded and no tokens
are generated: a "hit" is a second allocation of hashes the page table already holds.
"""

import time
from types import SimpleNamespace

import pytest
import torch

from exllamav3.generator.aging_lfu import PrefixWeightStore
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.generator.pagetable import PAGE_SIZE, PageTable
from testlib.cache import FakeCacheLayer, page, phash

NUM_PAGES = 8


def fake_cache(pages = 64):
    # PageTable only reads max_num_tokens from the cache at construction
    return SimpleNamespace(max_num_tokens = pages * PAGE_SIZE)


def cached_chain(pt: PageTable, hashes: list[bytes], kv_complete = True):
    """Allocate a chain of pages, link it root -> leaf and release it back into the prompt cache."""

    pages, _, _, _ = pt.allocate_pages(hashes, 0, None)
    for p in pages:
        p.kv_position = PAGE_SIZE if kv_complete else 0
    for p, prev in zip(pages[1:], hashes[:-1]):
        p.prev_hash = prev
    pt.deallocate_pages(pages)
    return pages


def test_weight_store_round_trips(tmp_path):
    # Weights and prefix token IDs must survive the process that earned them
    store = PrefixWeightStore(str(tmp_path))
    store.put_weights([(b"\x00\x01", 1.5, 100.0), (b"\x00\x02", 2.0, 200.0)])
    store.put_prefix("system", torch.arange(5, dtype = torch.int64).contiguous().numpy().tobytes())
    store.close()

    reopened = PrefixWeightStore(str(tmp_path))
    assert reopened.load_weights() == {b"\x00\x01": (1.5, 100.0), b"\x00\x02": (2.0, 200.0)}
    (label, blob), = reopened.load_prefixes()
    assert label == "system"
    assert torch.frombuffer(bytearray(blob), dtype = torch.int64).tolist() == [0, 1, 2, 3, 4]
    reopened.close()


def test_hit_credit_is_capped_per_hit_and_per_page():
    # A 245K one-off document covers ~960 pages and must not dominate after a single read: the hit earns at
    # most aging_hit_cap_pages total, spread over the pages it covered
    pt = PageTable(None, fake_cache(), aging_lfu = True, aging_hit_cap_pages = 64)
    pages = [SimpleNamespace(phash = bytes([i % 256]), weight = 0.0, last_hit = 0.0) for i in range(960)]

    pt._credit_hit(pages)
    assert all(p.weight == pytest.approx(64 / 960) for p in pages)
    assert pt.metrics["aging_hits"] == 1 and pt.metrics["aging_credit_pages"] == 960

    pt._credit_hit(pages)
    assert all(p.weight == pytest.approx(128 / 960) for p in pages)


def test_hit_credit_respects_the_per_page_cap():
    pt = PageTable(None, fake_cache(), aging_lfu = True, aging_weight_cap = 3.0, aging_hit_cap_pages = 64)
    pages = [SimpleNamespace(phash = b"\x00\x01", weight = 2.5, last_hit = 0.0)]
    pt._credit_hit(pages)
    assert pages[0].weight == 3.0


def test_effective_weight_decays_with_the_half_life():
    pt = PageTable(None, fake_cache(), aging_lfu = True, aging_half_life_hours = 72.0)
    past = time.time() - 72 * 3600
    page_rec = SimpleNamespace(phash = b"\x00\x03", weight = 8.0, last_hit = past)
    assert pt._effective_weight(page_rec) == pytest.approx(4.0)

    pt._weights[b"\x00\x04"] = (8.0, past)
    assert pt.effective_weight_hash(b"\x00\x04") == pytest.approx(4.0)
    assert pt.effective_weight_hash(b"\x00\x05") == 0.0


def test_allocation_credits_reused_pages():
    pt = PageTable(None, fake_cache(), aging_lfu = True)
    hashes = [b"\x10\x01", b"\x10\x02", b"\x10\x03"]
    cached_chain(pt, hashes)

    pages, _, cached, _ = pt.allocate_pages(hashes, 0, None)
    assert cached == 3
    # One hit over three pages earns three pages of credit, one per page
    assert all(p.weight == pytest.approx(1.0) for p in pages)
    assert pt.metrics["aging_hits"] == 1


def test_weights_survive_a_restart(tmp_path):
    hashes = [b"\x20\x01", b"\x20\x02"]
    first = PageTable(None, fake_cache(), aging_lfu = True, aging_persist_dir = str(tmp_path))
    cached_chain(first, hashes)
    first.allocate_pages(hashes, 0, None)
    first.close()

    # A new process: the pages are gone, the weights keyed by hash are not, and the pages recreated by a
    # fresh prefill of the same token sequence inherit them
    second = PageTable(None, fake_cache(), aging_lfu = True, aging_persist_dir = str(tmp_path))
    pages, _, _, _ = second.allocate_pages(hashes, 0, None)
    assert all(p.weight == pytest.approx(1.0) for p in pages)
    second.close()


def test_eviction_takes_cold_tails_before_hot_ones():
    # Two cached sequences, the older one hot (its pages were repeatedly reused) and the newer one cold.
    # Raw recency would eat the older tree first; aging-LFU has to prune the cold tail instead
    pt = PageTable(None, fake_cache(), aging_lfu = True)
    hot = [b"\x30\x01", b"\x30\x02"]
    cold = [b"\x31\x01", b"\x31\x02"]
    hot_pages = cached_chain(pt, hot)
    cold_pages = cached_chain(pt, cold)

    for p in hot_pages:
        p.weight = 10.0
        p.last_hit = time.time()

    order = list(pt.build_eviction_order())
    positions = {p.phash: i for i, p in enumerate(order)}
    assert positions[cold[1]] < positions[cold[0]]
    assert positions[cold[1]] < positions[hot[1]], "cold tail should go before the hot sequence"


class FakePageTable:
    """The CPU/disk tiers only need live-page lookup and hash weights from the page table."""

    def __init__(self, weights = None):
        self.weights = weights or {}

    def get_live_page(self, h):
        return None

    def effective_weight_hash(self, h):
        return self.weights.get(h, 0.0)


def test_cpu_tier_orders_leaves_by_weight(device):
    slots = 8
    tensor = torch.zeros((NUM_PAGES, 64), dtype = torch.float16, device = device)
    cache = CPUPageCache([SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})], slots * 4096)
    assert cache.max_slots == slots

    # Chain A (older): root 1 -> leaf 2, and chain B (newer): root 3 -> leaf 4
    cache.store(page(1, 0), serial = 1)
    cache.store(page(2, 1, phash(1)), serial = 2)
    cache.store(page(3, 2), serial = 3)
    cache.store(page(4, 3, phash(3)), serial = 4)

    # Make the older chain hot; with plain recency its leaf would lead the order
    cache.attach(FakePageTable({phash(1): 10.0, phash(2): 10.0}))
    cache._build_order()
    order = list(cache._order)

    assert order.index(phash(4)) < order.index(phash(2))
    assert order.index(phash(3)) < order.index(phash(1))
    cache.close()
