"""
FlashStates-style arbitrary state checkpointing (ASC) for hybrid models.

A recurrent checkpoint at an arbitrary token position is keyed by the content hash of the partial page it
ends in, chained onto the last complete page, and that partial page's K/V is stored in the page tiers under
the same key. These tests drive the pieces without a model: the recurrent cache's partial index and its
GreedyDual-Size eviction, the CPU/disk round trip of a partial page, and the allocation path that resumes a
sequence at a non-page-aligned position.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache.recurrent import RecurrentCache, host_pool
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.generator.disk_cache import DiskPageCache
from exllamav3.generator.pagetable import PageTable, Sequence, tensor_hash_checksum
from exllamav3.util.tensor import SeqTensor
from testlib.cache import FakeCacheLayer

NUM_PAGES = 16
K = 4           # complete pages before the partial one
OFFSET = 100    # tokens of the partial page that the checkpoint covers


# -- helpers -------------------------------------------------------------------------------------------------

class FakeState:
    """The subset of PairedState the recurrent cache uses"""

    def __init__(self, size = 64, position = 0):
        self.checkpoint_size = size
        self.position = position

    def stash(self):
        return {"position": self.position, "checkpoint_size": self.checkpoint_size}


def recurrent_cache(max_size = 1 << 20, **kwargs):
    return RecurrentCache(SimpleNamespace(loaded_tp = False), max_size, **kwargs)


@pytest.fixture(autouse = True)
def fresh_pool(monkeypatch):
    from exllamav3.cache import recurrent
    monkeypatch.setattr(recurrent, "_freed_bytes", 0)
    host_pool.free.clear()
    yield
    host_pool.free.clear()


def fake_cache_tier(pages = NUM_PAGES, slots = 32, device = "cuda:0"):
    tensor = torch.zeros((pages, 64), dtype = torch.float16, device = device)
    cache_obj = SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})
    tier = CPUPageCache([cache_obj], slots * 4096)
    return tier, tensor


def page_table(tier = None, pages = NUM_PAGES):
    pt = PageTable(None, SimpleNamespace(max_num_tokens = pages * PAGE_SIZE))
    if tier is not None:
        pt.cpu_tier = tier
        tier.attach(pt)
    return pt


def complete_chain(pt, tokens, base = 0x10):
    """Cache K complete pages holding `tokens` (whole pages) and return their content hashes"""
    assert len(tokens) >= K * PAGE_SIZE
    pages, _, _, _ = pt.allocate_pages([], K, None)
    prev = None
    hashes = []
    for i, p in enumerate(pages):
        p.sequence[0, :PAGE_SIZE] = tokens[i * PAGE_SIZE:(i + 1) * PAGE_SIZE]
        p.kv_position = PAGE_SIZE
        p.prev_hash = prev
        p.update_hash()
        hashes.append(p.phash)
        prev = p.phash
    return hashes


def make_partial(pt, tokens, anchor, offset = OFFSET, slab = None, fill = None):
    """Create and store the partial page for a checkpoint offset into page K, then drop it from VRAM"""
    pages, _, _, _ = pt.allocate_pages([], 1, None)
    p = pages[0]
    if slab is not None and fill is not None:
        slab[p.page_index].fill_(fill)
    p.sequence[:, :] = 0
    p.sequence[0, :offset] = tokens[K * PAGE_SIZE:K * PAGE_SIZE + offset]
    p.prev_hash = anchor
    p.kv_position = offset
    key = tensor_hash_checksum(p.sequence[0, :offset].contiguous(), anchor)
    del pt.referenced_pages[p.phash]
    p.phash = key
    pt.referenced_pages[key] = p
    assert pt.store_partial_page(p, key)
    page_index = p.page_index
    pt.deallocate_pages([p])
    p.clear()
    return key, page_index


class Recurrent:
    """The subset of RecurrentCache Sequence.allocate_pages consults"""

    def __init__(self, partial = None, boundary = None):
        self.partial = dict(partial or {})
        self.boundary = dict(boundary or {})

    def has(self, h):
        return h in self.partial or h in self.boundary

    def get_stashed(self, h):
        return self.partial.get(h) or self.boundary.get(h)

    def partial_keys(self, anchor):
        return [k for k, v in self.partial.items() if v["anchor"] == anchor]

    def position(self, key):
        return self.partial[key]["position"] if key in self.partial else None


def make_sequence(tokens, page_hashes, new_unique_pages = 2):
    seq = Sequence.__new__(Sequence)
    seq.page_hashes = page_hashes
    seq.max_cached_pages = None
    seq.new_unique_pages = new_unique_pages
    seq.mm_first_page = None
    seq.sequence_ids = SeqTensor.from_tensor(tokens.unsqueeze(0), seq_dim = -1)
    return seq


# -- recurrent cache: partial index and eviction -------------------------------------------------------------

def test_partial_checkpoint_is_indexed_under_its_anchor():
    rc = recurrent_cache()
    anchor = b"\xaa" * 16
    key = b"\xbb" * 16
    state = FakeState(size = 64, position = K * PAGE_SIZE + OFFSET)
    rc.put_partial(key, state, anchor, OFFSET)

    assert rc.partial_keys(anchor) == [key]
    assert rc.partial_keys(b"\xcc" * 16) == []
    assert rc.position(key) == K * PAGE_SIZE + OFFSET
    stashed = rc.get_stashed(key)
    assert stashed["offset"] == OFFSET
    assert stashed["anchor_hash"] == anchor.hex()
    assert rc.metrics["stash_partial"] == 1


def test_partial_index_is_maintained_across_eviction():
    rc = recurrent_cache(max_size = 100)
    anchor = b"\xaa" * 16
    rc.put_partial(b"\x01" * 16, FakeState(size = 60, position = 1000), anchor, 10)
    # Forces the first checkpoint out; its index entry must go with it
    rc.put_partial(b"\x02" * 16, FakeState(size = 60, position = 2000), anchor, 20)
    assert len(rc) == 1
    assert rc.partial_keys(anchor) == [b"\x02" * 16]
    rc.close()


def test_partial_index_rebuilds_from_the_disk_index():
    anchor = b"\xaa" * 16
    key = b"\xbb" * 16
    disk = SimpleNamespace(
        states = {
            key: {"position": K * PAGE_SIZE + OFFSET, "size": 1, "last_access": 0},
            b"\xee" * 16: {"position": (K + 1) * PAGE_SIZE, "size": 1, "last_access": 0},  # page-aligned
        },
        has_page = lambda h: h == key,
        page_prev_hash = lambda h: anchor,
        note_state = lambda h, s: True,
    )
    rc = recurrent_cache()
    rc.disk_tier = disk
    rc.rebuild_partial_index()
    assert rc.partial_keys(anchor) == [key]


def test_gds_evicts_the_lowest_prefill_value_per_byte():
    # A: long prefix but large state -> 1.0/unit. B: short prefix and small state -> 1.0. C: long prefix and
    # small state -> 100.0. The budget forces one eviction; C must survive, A (the older of the two ties) goes.
    rc = recurrent_cache(max_size = 115, eviction_policy = "gds")
    rc.put(b"\x01" * 16, FakeState(size = 100, position = 100))
    rc.put(b"\x02" * 16, FakeState(size = 10, position = 10))
    rc.put(b"\x03" * 16, FakeState(size = 10, position = 1000))

    assert b"\x03" * 16 in rc
    assert b"\x01" * 16 not in rc
    assert b"\x02" * 16 in rc
    assert rc.metrics["stash_evictions"] == 1
    rc.close()


def test_lru_still_drops_the_oldest_entry():
    rc = recurrent_cache(max_size = 115)
    rc.put(b"\x01" * 16, FakeState(size = 100, position = 1000))
    rc.put(b"\x02" * 16, FakeState(size = 10, position = 10))
    rc.put(b"\x03" * 16, FakeState(size = 10, position = 10))
    assert b"\x01" * 16 not in rc and len(rc) == 2
    rc.close()


# -- partial page in the CPU tier and exact-position resume --------------------------------------------------

@pytest.fixture(autouse = True)
def inference_mode():
    with torch.inference_mode():
        yield


def test_partial_page_round_trips_through_the_cpu_tier(device):
    tokens = torch.arange(0, (K + 2) * PAGE_SIZE, dtype = torch.long)
    tier, slab = fake_cache_tier(device = device)
    pt = page_table(tier)
    hashes = complete_chain(pt, tokens)
    anchor = hashes[-1]
    restores = tier.metrics["restores"]

    # The K/V slab is filled so the copy is checked bit-exactly; the page is then dropped from VRAM, leaving
    # only the tier copy
    key, _ = make_partial(pt, tokens, anchor, slab = slab, fill = 0.5)

    # A later prompt completes page K, so page K's full hash is in the prompt's page hashes; the checkpoint
    # inside it is found by its own key
    full = tensor_hash_checksum(tokens[K * PAGE_SIZE:(K + 1) * PAGE_SIZE], anchor)
    seq = make_sequence(tokens, hashes + [full], new_unique_pages = 2)
    rc = Recurrent(partial = {key: {
        "position": K * PAGE_SIZE + OFFSET,
        "offset": OFFSET,
        "anchor": anchor,
    }})

    allocated, cached, _, stashed = seq.allocate_pages(pt, rc)

    assert seq.kv_position == K * PAGE_SIZE + OFFSET
    assert cached == K
    assert stashed["offset"] == OFFSET
    assert seq.allocated_pages[K].kv_position == OFFSET
    assert seq.allocated_pages[K].phash == key
    assert torch.equal(seq.allocated_pages[K].sequence[0, :OFFSET],
                       tokens[K * PAGE_SIZE:K * PAGE_SIZE + OFFSET])
    assert tier.metrics["restores"] == restores + 1
    assert torch.all(slab[seq.allocated_pages[K].page_index] == 0.5), \
        "partial page K/V was not restored bit-exactly"
    assert pt.metrics["alloc_partial_pages"] == 1


def test_partial_page_is_shared_when_it_is_still_in_vram(device):
    tokens = torch.arange(0, (K + 2) * PAGE_SIZE, dtype = torch.long)
    tier, _ = fake_cache_tier(device = device)
    pt = page_table(tier)
    hashes = complete_chain(pt, tokens)
    key, _ = make_partial(pt, tokens, hashes[-1])
    # Put the partial page back into VRAM under its key (as a resume would leave it)
    seq0 = make_sequence(tokens, hashes + [tensor_hash_checksum(tokens[K * PAGE_SIZE:(K + 1) * PAGE_SIZE],
                                                                 hashes[-1])], new_unique_pages = 1)
    rc = Recurrent(partial = {key: {"position": K * PAGE_SIZE + OFFSET, "offset": OFFSET, "anchor": hashes[-1]}})
    seq0.allocate_pages(pt, rc)
    page = seq0.allocated_pages[K]

    # A second request resuming from the same checkpoint references the same page object
    seq1 = make_sequence(tokens, hashes + [tensor_hash_checksum(tokens[K * PAGE_SIZE:(K + 1) * PAGE_SIZE],
                                                                 hashes[-1])], new_unique_pages = 1)
    seq1.allocate_pages(pt, rc)
    assert seq1.allocated_pages[K] is page
    assert page.ref_count == 2


def test_frontier_checkpoint_key_is_found_by_a_later_request(device):
    # The key the job stores under and the key a later request rebuilds from its own prompt must agree, or
    # the exact-position checkpoint is never hit
    tokens = torch.arange(0, (K + 2) * PAGE_SIZE, dtype = torch.long)
    tier, _ = fake_cache_tier(device = device)
    pt = page_table(tier)
    hashes = complete_chain(pt, tokens)
    anchor = hashes[-1]
    pos = K * PAGE_SIZE + OFFSET

    partial_pages, _, _, _ = pt.allocate_pages([], 1, None)
    page = partial_pages[0]
    page.sequence[:, :] = 0
    page.sequence[0, :OFFSET] = tokens[K * PAGE_SIZE:pos]
    page.prev_hash = anchor
    page.kv_position = OFFSET

    seq = make_sequence(tokens, hashes + [tensor_hash_checksum(tokens[K * PAGE_SIZE:(K + 1) * PAGE_SIZE],
                                                                 anchor)], new_unique_pages = 2)
    seq.kv_position = pos
    # Point the sequence at the real chain plus the partial page
    chain = pt.allocate_pages(hashes, 0, None)[0]
    seq.allocated_pages = chain + [page]
    for p, prev in zip(chain[1:], hashes[:-1]):
        p.prev_hash = prev

    from exllamav3.generator.job import Job

    class Cache:
        def __init__(self):
            self.partial = None
        def put(self, key, state):
            raise AssertionError("unexpected page-aligned checkpoint")
        def put_partial(self, key, state, anchor_hash, offset):
            self.partial = (key, state, anchor_hash, offset)

    job = Job.__new__(Job)
    job.sequences = [seq]
    job.recurrent_state = SimpleNamespace(
        position = pos, checkpoint_size = 64,
        stash = lambda: {"position": pos, "checkpoint_size": 64},
    )
    job.last_recurrent_checkpoint_pos = None
    job.pagetable = pt
    job.generator = SimpleNamespace(recurrent_frontier_checkpoints = True)
    cache = Cache()

    assert job.frontier_checkpoint(cache)
    key, state, got_anchor, offset = cache.partial
    assert offset == OFFSET and got_anchor == anchor

    # Drop the VRAM page; the tier copy is what the next request restores
    pt.deallocate_pages([page])
    page.clear()

    # The real cache indexes a put_partial() state under its anchor and answers find_resume()
    rc = recurrent_cache()
    rc.put_partial(key, FakeState(size = 64, position = pos), anchor, OFFSET)
    later = make_sequence(tokens, hashes + [tensor_hash_checksum(tokens[K * PAGE_SIZE:(K + 1) * PAGE_SIZE],
                                                                 anchor)], new_unique_pages = 2)
    _, cached, _, found = later.allocate_pages(pt, rc)

    assert found is not None and found["offset"] == OFFSET and found["position"] == pos
    assert later.kv_position == pos
    assert cached == K
    rc.close()


def test_exact_position_is_not_selected_when_the_page_chain_is_gone(device):
    # The partial page is stored and then evicted from both VRAM and the CPU tier: the checkpoint is
    # stranded, so a boundary checkpoint (or replay) must be used instead
    tokens = torch.arange(0, (K + 2) * PAGE_SIZE, dtype = torch.long)
    tier, _ = fake_cache_tier(device = device)
    pt = page_table(tier)
    hashes = complete_chain(pt, tokens)
    key, _ = make_partial(pt, tokens, hashes[-1])
    tier.entries.pop(key)                       # drop the partial page from the tier as well

    boundary_position = K * PAGE_SIZE
    boundary = {hashes[K - 1]: {"position": boundary_position}}
    full = tensor_hash_checksum(tokens[K * PAGE_SIZE:(K + 1) * PAGE_SIZE], hashes[-1])
    seq = make_sequence(tokens, hashes + [full], new_unique_pages = 2)
    rc = Recurrent(partial = {key: {"position": K * PAGE_SIZE + OFFSET, "offset": OFFSET,
                                    "anchor": hashes[-1]}}, boundary = boundary)

    _, cached, _, stashed = seq.allocate_pages(pt, rc)
    assert seq.kv_position == boundary_position
    assert cached == K
    assert stashed is boundary[hashes[K - 1]]
    assert pt.metrics["alloc_partial_pages"] == 0


def test_unreadable_partial_state_falls_back_to_a_page_boundary(device):
    # The partial page is in the tier, so the checkpoint is selected, but its state cannot be read back (a
    # damaged store). The sequence must resume at a page boundary, not at the partial position with no state
    tokens = torch.arange(0, (K + 2) * PAGE_SIZE, dtype = torch.long)
    tier, _ = fake_cache_tier(device = device)
    pt = page_table(tier)
    hashes = complete_chain(pt, tokens)
    key, _ = make_partial(pt, tokens, hashes[-1])

    class Unreadable(Recurrent):
        def get_stashed(self, h):
            return None if h == key else super().get_stashed(h)

    boundary = {hashes[K - 1]: {"position": K * PAGE_SIZE}}
    rc = Unreadable(partial = {key: {"position": K * PAGE_SIZE + OFFSET, "offset": OFFSET,
                                     "anchor": hashes[-1]}}, boundary = boundary)
    full = tensor_hash_checksum(tokens[K * PAGE_SIZE:(K + 1) * PAGE_SIZE], hashes[-1])
    seq = make_sequence(tokens, hashes + [full], new_unique_pages = 2)

    _, cached, _, stashed = seq.allocate_pages(pt, rc)
    assert stashed is boundary[hashes[K - 1]]
    assert seq.kv_position == K * PAGE_SIZE
    assert seq.allocated_pages[K].kv_position == 0 or seq.allocated_pages[K].kv_position == OFFSET


# -- disk tier -----------------------------------------------------------------------------------------------

def build_disk(slots, device, root, disk_size = 1 << 30, fingerprint = "asc"):
    tensor = torch.zeros((NUM_PAGES, 64), dtype = torch.float16, device = device)
    cache_obj = SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})
    tier = CPUPageCache([cache_obj], slots * 4096)
    disk = DiskPageCache(str(root), [cache_obj], tier, disk_size, fingerprint = fingerprint,
                         min_state_distance = 0)
    tier.attach_disk(disk)
    return tier, disk


def test_partial_page_and_checkpoint_survive_a_restart(device, tmp_path):
    tokens = torch.arange(0, (K + 2) * PAGE_SIZE, dtype = torch.long)
    tier, disk = build_disk(8, device, tmp_path)
    pt = page_table()
    tier.attach(pt)
    pt.cpu_tier = tier
    hashes = complete_chain(pt, tokens)
    anchor = hashes[-1]
    key, page_index = make_partial(pt, tokens, anchor)
    stashed = {
        "position": K * PAGE_SIZE + OFFSET,
        "checkpoint_size": 64,
        "anchor_hash": anchor.hex(),
        "offset": OFFSET,
        "target": {"position": K * PAGE_SIZE + OFFSET, "checkpoint_size": 64, "x": torch.ones(8)},
    }
    assert disk.write_state(key, stashed)
    disk.flush()
    assert disk.has_page(key) and disk.has_state(key)
    disk.close()
    tier.close()

    # A new process: page and checkpoint are on disk; the recurrent cache indexes the checkpoint from the
    # disk metadata alone and the page comes back under its partial key
    tier2, disk2 = build_disk(8, device, tmp_path)
    rc = RecurrentCache(SimpleNamespace(loaded_tp = False), 1 << 20)
    rc.disk_tier = disk2
    rc.rebuild_partial_index()
    assert rc.partial_keys(anchor) == [key]
    assert rc.get_stashed(key)["offset"] == OFFSET

    e = tier2.promote(key, serial = 1)
    assert e is not None and torch.equal(e["tokens"][:OFFSET], tokens[K * PAGE_SIZE:K * PAGE_SIZE + OFFSET])
    assert disk2.page_prev_hash(key) == anchor
    rc.close()
    disk2.close()
    tier2.close()
