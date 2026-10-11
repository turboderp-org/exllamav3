"""
DiskPageCache (generator/disk_cache.py), the persistent tier behind the CPU page tier: pages written through
from the CPU tier must come back bit-exact in a fresh tier (a new process in effect), the chain metadata the
resumability walk needs must survive with them, multimodal pages must never be written, the directory must
stay within its budget by the tree policy, checkpoints must round-trip through pooled buffers with their
container structure intact, a stash whose write is still queued must not return its buffers to the pool, and
a fingerprint mismatch must land in a different directory.

Driven with the stand-in cache of testlib.cache, as the CPU tier tests are: one paged tensor gives the tiers a
slot size and something to copy. The fingerprint is overridden so no model is needed.
"""

import errno
import os
from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache.recurrent import host_pool
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.generator.disk_cache import DiskPageCache
from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from testlib.cache import FakeCacheLayer, page, phash, PAGE_SIZE

NUM_PAGES = 8


@pytest.fixture(autouse = True)
def inference_mode():
    with torch.inference_mode():
        yield


def build(slots, device, root, disk_size = 1 << 30, fingerprint = "test"):
    tensor = torch.zeros((NUM_PAGES, 64), dtype = torch.float16, device = device)
    cache_obj = SimpleNamespace(layers = {0: FakeCacheLayer([tensor, None])})
    tier = CPUPageCache([cache_obj], slots * 4096)
    # Checkpoint spacing off unless a test sets it: the small positions used here would otherwise be refused
    disk = DiskPageCache(str(root), [cache_obj], tier, disk_size, fingerprint = fingerprint, min_state_distance = 0)
    tier.attach_disk(disk)
    return tier, disk, tensor


def fill(tensor, page_index, value):
    tensor[page_index].fill_(value)


def test_pages_round_trip_through_a_fresh_tier(device, tmp_path):
    tier, disk, tensor = build(4, device, tmp_path)
    prev = None
    for tag in range(1, 4):
        fill(tensor, tag, float(tag))
        tier.store(page(tag, tag, prev), serial = tag)
        prev = phash(tag)
    disk.flush()
    assert disk.metrics["page_writes"] == 3 and len(disk) == 3
    assert not disk.pending_pages
    disk.close()
    tier.close()

    # A new process: nothing in VRAM or RAM, everything on disk
    tier2, disk2, tensor2 = build(4, device, tmp_path)
    assert len(disk2) == 3
    assert disk2.page_prev_hash(phash(2)) == phash(1) and disk2.page_prev_hash(phash(1)) is None
    for tag in range(1, 4):
        e = tier2.promote(phash(tag), serial = 10 + tag)
        assert e is not None and torch.equal(e["tokens"], torch.full((PAGE_SIZE,), tag, dtype = torch.long))
        tier2.fetch(phash(tag), 0, serial = 10 + tag)
        torch.cuda.synchronize()
        assert torch.all(tensor2[0] == float(tag)), f"page {tag} did not restore bit-exact from disk"
    assert tier2.metrics["promotions"] == 3 and disk2.metrics["page_reads"] == 3
    disk2.close()
    tier2.close()


def test_prefetch_promotes_asynchronously_and_fetch_waits(device, tmp_path):
    tier, disk, tensor = build(8, device, tmp_path)
    for tag in range(1, 6):
        fill(tensor, tag, float(tag))
        tier.store(page(tag, tag), serial = tag)
    disk.flush()
    tier.close()
    disk.close()

    tier2, disk2, tensor2 = build(8, device, tmp_path)
    disk2.prefetch([phash(t) for t in range(1, 6)], serial = 1)
    # Entries exist from the moment the read is queued, pending or not; the prefetch stops at its in-flight
    # bound and the allocation promotes the rest itself
    queued = sum(phash(t) in tier2 for t in range(1, 6))
    assert 2 <= queued <= 5
    for tag in range(1, 6):
        if phash(tag) not in tier2:
            assert tier2.promote(phash(tag), serial = 20) is not None
        e = tier2.fetch(phash(tag), 0, serial = 20)
        assert e is not None and e["pending"].is_set()
        torch.cuda.synchronize()
        assert torch.all(tensor2[0] == float(tag))
    disk2.close()
    tier2.close()


def test_a_slot_with_a_queued_write_is_not_recycled(device, tmp_path):
    # Two slots, writer paused: the second store would have to evict the first, whose write is still queued
    tier, disk, tensor = build(2, device, tmp_path)
    tier.max_slots = 1     # force eviction on the second store
    with disk._cv:
        disk._queue.append(("block",))      # unknown op: the writer logs an error and moves on, but first...
    tier.store(page(1, 1), serial = 1)
    assert phash(1) in disk.pending_pages
    # Eviction must not hand out slot 0 while its write is pending: it waits for the flush instead
    tier.store(page(2, 2), serial = 2)
    assert phash(1) not in disk.pending_pages, "store returned before the pending write completed"
    disk.flush()
    assert disk.has_page(phash(1)) and disk.has_page(phash(2))
    disk.close()
    tier.close()


def test_multimodal_pages_are_not_written(device, tmp_path):
    tier, disk, tensor = build(4, device, tmp_path)
    p = page(1, 1)
    p.sequence[3] = FIRST_MM_EMBEDDING_INDEX + 7
    tier.store(p, serial = 1)
    disk.flush()
    assert len(disk) == 0 and disk.metrics["skipped_mm"] == 1
    assert phash(1) in tier     # still a perfectly good CPU-tier entry
    disk.close()
    tier.close()


def test_budget_evicts_tail_first_oldest_root_first(device, tmp_path):
    tier, disk, tensor = build(16, device, tmp_path)
    page_bytes = disk.image_size + 512
    # Room for about six pages: two chains of four would overflow by two
    disk.max_size = 6 * page_bytes
    prev = None
    for tag in range(1, 5):                 # chain A, older
        tier.store(page(tag, tag % NUM_PAGES, prev), serial = tag)
        prev = phash(tag)
    disk.flush()
    prev = None
    for tag in range(11, 15):               # chain B, newer
        tier.store(page(tag, tag % NUM_PAGES, prev), serial = tag)
        prev = phash(tag)
    disk.flush()
    assert disk.total_size <= disk.max_size
    kept = {h for h in disk.pages}
    # Chain B intact; chain A lost its tail, never its root
    assert all(phash(t) in kept for t in range(11, 15))
    assert phash(1) in kept
    assert phash(4) not in kept
    disk.close()
    tier.close()


def test_state_round_trip_and_deferred_release(device, tmp_path):
    tier, disk, tensor = build(4, device, tmp_path)
    stashed = {
        "position": 2048,
        "checkpoint_size": 4096,
        "layer.a": (torch.arange(16, dtype = torch.float32).reshape(1, 16), torch.ones(4, 3, dtype = torch.bfloat16)),
        "layer.b": [torch.full((8,), 3.0), torch.full((2, 2), 5.0)],
        "layer.c": torch.full((5,), 9, dtype = torch.int32),
    }
    key = phash(7)
    released = []
    disk.write_state(key, stashed)
    assert disk.has_state(key)
    # Released before the write is done: must be deferred, not run
    disk.release_after_write(stashed, lambda s: released.append(s))
    disk.flush()
    assert released == [stashed], "release did not run after the write completed"
    disk.close()
    tier.close()

    tier2, disk2, _ = build(4, device, tmp_path)
    assert disk2.has_state(key)
    back = disk2.read_state(key)
    assert back["position"] == 2048 and back["checkpoint_size"] == 4096
    assert isinstance(back["layer.a"], tuple) and isinstance(back["layer.b"], list)
    for k in ("layer.a", "layer.b"):
        for a, b in zip(stashed[k], back[k]):
            assert a.dtype == b.dtype and torch.equal(a, b)
    assert torch.equal(stashed["layer.c"], back["layer.c"])
    # Buffers came from the stash pool so RecurrentCache can give them back
    host_pool.give(back)
    assert host_pool.free
    disk2.close()
    tier2.close()


def test_fingerprint_selects_the_directory(device, tmp_path):
    tier, disk, _ = build(4, device, tmp_path, fingerprint = "aaaa")
    tier2, disk2, _ = build(4, device, tmp_path, fingerprint = "bbbb")
    assert disk.dir != disk2.dir and os.path.isdir(disk.dir) and os.path.isdir(disk2.dir)
    for d in (disk, disk2):
        with open(os.path.join(d.dir, "manifest.json")) as f:
            assert f.read().count(d.fingerprint) >= 1
    disk.close(); tier.close(); disk2.close(); tier2.close()


def test_index_is_rebuilt_from_files(device, tmp_path):
    tier, disk, tensor = build(4, device, tmp_path)
    for tag in range(1, 3):
        tier.store(page(tag, tag), serial = tag)
    disk.flush()
    disk.close()
    tier.close()
    os.remove(os.path.join(disk.dir, "index.sqlite"))
    tier2, disk2, _ = build(4, device, tmp_path)
    assert len(disk2) == 2 and disk2.has_page(phash(1)) and disk2.has_page(phash(2))
    disk2.close()
    tier2.close()


def test_root_budget_evicts_least_recently_used_sibling_directories(device, tmp_path):
    # Three configurations share one root. The budget is the root's; the newest open removes the oldest
    # sibling whole and keeps the one used more recently
    tier_a, disk_a, _ = build(8, device, tmp_path, disk_size = 1 << 40, fingerprint = "aaaa")
    for tag in range(1, 5):
        tier_a.store(page(tag, tag), serial = tag)
    disk_a.flush(); size_a = disk_a.total_size; disk_a.close(); tier_a.close()
    os.utime(os.path.join(disk_a.dir, "index.sqlite"), (1, 1))     # used long ago

    tier_b, disk_b, _ = build(8, device, tmp_path, disk_size = 1 << 40, fingerprint = "bbbb")
    for tag in range(1, 5):
        tier_b.store(page(tag, tag), serial = tag)
    disk_b.flush(); size_b = disk_b.total_size; disk_b.close(); tier_b.close()
    dir_size_b = DiskPageCache._dir_size(disk_b.dir)      # pages plus the index files, which count too

    # Room for about two directories' worth of pages, not three
    tier_c, disk_c, _ = build(8, device, tmp_path, disk_size = size_a + size_b + size_b // 2, fingerprint = "cccc")
    assert not os.path.isdir(disk_a.dir), "oldest sibling should have been removed"
    assert os.path.isdir(disk_b.dir), "more recently used sibling should survive"
    assert disk_c.max_size == disk_c.root_size - dir_size_b
    assert disk_c.metrics["dirs_evicted"] == 1
    disk_c.close(); tier_c.close()


def test_a_full_disk_turns_the_tier_read_only_and_frees_the_slots(device, tmp_path, monkeypatch):
    tier, disk, tensor = build(4, device, tmp_path)
    tier.store(page(1, 1), serial = 1)
    disk.flush()
    assert disk.has_page(phash(1))
    size_before = disk.total_size

    # Every later write hits a full disk
    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(DiskPageCache, "_do_write_page", no_space)
    monkeypatch.setattr(DiskPageCache, "_do_write_state", no_space)
    for tag in range(2, 5):
        tier.store(page(tag, tag), serial = tag)
    disk.write_state(phash(9), {"position": 256, "checkpoint_size": 64, "x": torch.ones(16)})
    disk.flush()

    assert disk.write_disabled and disk.metrics["write_errors"] >= 1
    assert not disk.pending_pages and not disk.pending_states, "failed writes left slots or stashes pinned"
    assert disk.total_size == size_before, "reservations of failed writes were not rolled back"
    assert not any(n.endswith(".tmp") for _, _, names in os.walk(disk.dir) for n in names)
    # Nothing more is queued, the higher tiers are untouched, the page already on disk still restores
    assert disk.write_page(phash(6), None, torch.zeros(PAGE_SIZE, dtype = torch.long), 0, []) is False
    assert all(phash(t) in tier for t in range(1, 5))
    tier.close()
    disk.close()
    tier2, disk2, tensor2 = build(4, device, tmp_path)
    assert tier2.promote(phash(1), serial = 5) is not None
    disk2.close()
    tier2.close()


def test_checkpoints_follow_their_pages_and_are_pruned_by_spacing(device, tmp_path):
    tier, disk, tensor = build(16, device, tmp_path)

    def state(pos):
        return {"position": pos, "checkpoint_size": 64, "x": torch.full((16,), float(pos))}

    # The user's example: one chain of 36 pages (9216 tokens at PAGE_SIZE 256 in the real tier; here the
    # positions are what matter), checkpoints at 768, 2048, 2560, 4096, 4352, 9216 anchored on pages.
    # Written with spacing off, so the prune has something to decide
    disk.min_state_distance = 0
    positions = {3: 768, 8: 2048, 10: 2560, 16: 4096, 17: 4352, 36: 9216}
    # A candidate offered before its page exists is held, not written; it is written once the page lands
    held = state(768)
    assert disk.note_state(phash(3), held)
    assert phash(3) in disk.state_candidates and not disk.has_state(phash(3))
    prev = None
    for tag in range(1, 37):
        tier.store(page(tag, tag % NUM_PAGES, prev), serial = tag)
        prev = phash(tag)
        if tag in positions and tag != 3:
            assert disk.note_state(phash(tag), state(positions[tag]))
    disk.flush()
    disk.drain_ready_states()
    disk.flush()
    assert disk.has_state(phash(3)), "candidate not written after its page was published"
    assert all(disk.has_state(phash(t)) for t in positions)
    # An orphan: a checkpoint whose page is not on disk
    assert disk.write_state(phash(99), state(5000))
    disk.flush()

    disk.min_state_distance = 4096
    dropped = disk.prune_states()
    kept = sorted(positions[t] for t in positions if disk.has_state(phash(t)))
    # Counted from the root: 768, 2048, 2560 are within the first interval, 4096 is the first kept, 4352 is
    # too close behind it, 9216 is the next; the same answer for every leaf through this prefix
    assert kept == [4096, 9216], f"kept {kept}"
    assert not disk.has_state(phash(99)) and dropped == 5

    # The same rule at write time: offered again, the pruned positions are refused outright and a new one
    # far enough past 9216 is taken
    for tag in (3, 8, 10, 17):
        assert not disk.note_state(phash(tag), state(positions[tag]))
    tier.store(page(60, 4, phash(36)), serial = 60)
    assert disk.note_state(phash(60), state(9216 + 4096))
    disk.flush()
    disk.drain_ready_states()
    disk.flush()
    assert disk.has_state(phash(60)) and len(disk.states) == 3
    disk.close()
    tier.close()


def test_candidate_dropped_when_released_before_its_page(device, tmp_path):
    tier, disk, tensor = build(4, device, tmp_path)
    released = []
    st = {"position": 256, "checkpoint_size": 64, "x": torch.ones(8)}
    assert disk.note_state(phash(5), st)
    disk.release_after_write(st, lambda s: released.append(s))
    assert released == [st] and phash(5) not in disk.state_candidates and not disk.has_state(phash(5))
    assert disk.metrics["state_candidates_dropped"] == 1
    disk.close()
    tier.close()


def test_open_discards_leftovers_of_interrupted_transfers(device, tmp_path):
    tier, disk, _ = build(4, device, tmp_path)
    tier.store(page(1, 1), serial = 1)
    disk.write_state(phash(2), {"position": 256, "checkpoint_size": 64, "x": torch.ones(8)})
    disk.flush()
    disk.close()
    tier.close()
    # What a process killed mid-write leaves behind: partial temp files and rank markers, in both directories
    h = phash(7).hex()
    for name in (f"{h}.rs.tmp", f"{h}.rs.tmp.d0", f"{h}.rs.tmp.d1.err", "junk"):
        open(os.path.join(disk.states_dir, name), "w").close()
    pdir = os.path.join(disk.pages_dir, phash(7)[:1].hex())
    os.makedirs(pdir, exist_ok = True)
    for name in (f"{h}.kv.tmp", f"{h}.kv.tmp.d0", "junk"):
        open(os.path.join(pdir, name), "w").close()

    tier2, disk2, _ = build(4, device, tmp_path)
    assert disk2.has_page(phash(1)) and disk2.has_state(phash(2))
    assert sorted(os.listdir(disk2.states_dir)) == [phash(2).hex() + ".rs"]
    assert sorted(os.listdir(pdir)) == [] or all(n.endswith(".kv") for n in os.listdir(pdir))
    disk2.close()
    tier2.close()
