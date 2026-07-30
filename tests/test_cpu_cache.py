import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "requires CUDA")

NUM_PAGES = 4


class _FakeLayer:
    def __init__(self, tensors):
        self.tensors = tensors
    def get_tensors(self):
        return self.tensors


class _FakeCache:
    """Minimal stand-in for Cache: one layer with paged K/V tensors on the GPU."""
    def __init__(self):
        k = torch.randn((NUM_PAGES, 8, 16), dtype = torch.float16, device = "cuda")
        v = torch.randn((NUM_PAGES, 8, 16), dtype = torch.float16, device = "cuda")
        self.layers = {0: _FakeLayer([k, v])}


class _FakePage:
    """Minimal stand-in for CachePage: the fields CPUPageCache.store reads."""
    def __init__(self, phash, page_index, sequence, prev_hash = None):
        self.phash = phash
        self.prev_hash = prev_hash
        self.page_index = page_index
        self.kv_position = PAGE_SIZE
        self.sequence = sequence


class _FakePageTable:
    """Minimal stand-in for PageTable: hash lookup of complete VRAM pages."""
    def __init__(self, pages = ()):
        self.pages = {p.phash: p for p in pages}
        self.max_pages = 1024
        self.lookups = 0

    def get_live_page(self, phash):
        self.lookups += 1
        page = self.pages.get(phash)
        if page is not None and page.kv_position == PAGE_SIZE:
            return page
        return None


def _text_sequence():
    return torch.arange(PAGE_SIZE, dtype = torch.long).unsqueeze(0)


def _mm_sequence():
    seq = _text_sequence()
    seq[0, PAGE_SIZE // 2] = FIRST_MM_EMBEDDING_INDEX + 42
    return seq


@requires_cuda
def test_mm_pages_not_stored():
    cache = _FakeCache()
    cpc = CPUPageCache([cache], max_size = 64 * 1024)

    text_page = _FakePage(b"\x01" * 16, 0, _text_sequence())
    cpc.store(text_page, serial = 1)
    assert text_page.phash in cpc
    assert len(cpc) == 1

    mm_page = _FakePage(b"\x02" * 16, 1, _mm_sequence())
    cpc.store(mm_page, serial = 2)
    assert mm_page.phash not in cpc
    assert len(cpc) == 1
    assert cpc.metrics["mm_rejects"] == 1
    assert cpc.metrics["pushes"] == 1


@requires_cuda
def test_mm_taint_propagates_through_descendants():
    # A text-only page downstream of an MM page carries K/V that depends on the image content (through
    # attention) under a hash that only commits to token IDs, so the whole descendant chain must be rejected.
    cache = _FakeCache()
    cpc = CPUPageCache([cache], max_size = 64 * 1024)

    mm_page = _FakePage(b"\x10" * 16, 0, _mm_sequence())
    child = _FakePage(b"\x11" * 16, 1, _text_sequence(), prev_hash = mm_page.phash)
    grandchild = _FakePage(b"\x12" * 16, 2, _text_sequence(), prev_hash = child.phash)
    cpc.attach(_FakePageTable([mm_page, child, grandchild]))

    cpc.store(child, serial = 1)
    cpc.store(grandchild, serial = 2)
    assert len(cpc) == 0
    assert cpc.metrics["mm_rejects"] == 2


@requires_cuda
def test_clean_ancestry_through_tier_entry():
    # An ancestor already stored in the tier proves its own chain was verified MM-free, so the walk may stop
    # there; a chain that cannot be followed at all is conservatively rejected.
    cache = _FakeCache()
    cpc = CPUPageCache([cache], max_size = 64 * 1024)
    cpc.attach(_FakePageTable())

    parent = _FakePage(b"\x20" * 16, 0, _text_sequence())
    cpc.store(parent, serial = 1)

    child = _FakePage(b"\x21" * 16, 1, _text_sequence(), prev_hash = parent.phash)
    cpc.store(child, serial = 2)     # parent absent from VRAM but present in the tier
    assert child.phash in cpc

    orphan = _FakePage(b"\x22" * 16, 2, _text_sequence(), prev_hash = b"\x99" * 16)
    cpc.store(orphan, serial = 3)    # ancestry unverifiable
    assert orphan.phash not in cpc
    assert cpc.metrics["mm_rejects"] == 1


@requires_cuda
def test_taint_walk_is_memoized():
    # Tearing down an N-page chain must not walk N^2/2 ancestors: the deepest store memoizes taint verdicts
    # for its whole path (verdicts are pure functions of the chained hash), so later stores resolve without
    # any further chain lookups.
    chain_len = 8
    cache = _FakeCache()
    cpc = CPUPageCache([cache], max_size = 64 * 1024)

    pages = []
    for i in range(chain_len):
        prev_hash = pages[-1].phash if pages else None
        pages.append(_FakePage(bytes([0x30 + i]) * 16, i % NUM_PAGES, _text_sequence(), prev_hash = prev_hash))
    pt = _FakePageTable(pages)
    cpc.attach(pt)

    cpc.store(pages[-1], serial = 1)                # deepest leaf: walks to root once
    assert len(cpc._taint_memo) == chain_len
    walk_lookups = pt.lookups
    assert walk_lookups == chain_len - 1

    for i, page in enumerate(reversed(pages[:-1])): # leaf-first teardown of the rest
        cpc.store(page, serial = 2 + i)
    assert pt.lookups == walk_lookups               # every verdict came from the memo
    assert cpc.metrics["pushes"] == chain_len


@requires_cuda
def test_store_fetch_roundtrip():
    cache = _FakeCache()
    cpc = CPUPageCache([cache], max_size = 64 * 1024)
    tensors = cache.layers[0].get_tensors()
    originals = [t[0].clone() for t in tensors]

    page = _FakePage(b"\x03" * 16, 0, _text_sequence())
    cpc.store(page, serial = 1)

    # Destroy the source page, then restore the stored copy into a different page index
    for t in tensors:
        t[0].fill_(0.0)
    entry = cpc.fetch(page.phash, page_index = 2, serial = 2)
    torch.cuda.synchronize()

    for t, orig in zip(tensors, originals):
        assert torch.equal(t[2], orig)
    assert torch.equal(entry["tokens"], page.sequence)
    assert entry["prev_hash"] is None
    assert page.phash in cpc  # fetch is non-consuming
