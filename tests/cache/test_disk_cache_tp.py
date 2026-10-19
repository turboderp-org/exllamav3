"""
DiskPageCache with tensor-parallel caches: a page image on disk is the canonical (all heads, model order)
layout, written and read by the ranks as strided chunks of their own shards, so a page written under one head
split restores under another split and into a non-TP tier over the full tensor, bit-exact. Completion goes
through rank marker files that the main process's writer thread polls.

Driven in-process like test_cpu_cache_tp.py: the ranks are dicts standing in for worker local_contexts and the
fake model routes the dispatch calls straight into the rank functions (model_tp_fn.mp_disk_*), so the IPC is
not covered, the layout arithmetic and the completion protocol are. Pages are the real PAGE_SIZE rows, since
the head split is a per-token-row slice.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.generator.disk_cache import DiskPageCache
from exllamav3.model.model_tp_fn import (
    mp_cpu_cache_init, mp_cpu_cache_store, mp_cpu_cache_fetch,
)
from testlib.cache import FakeCacheLayer, page, phash

NUM_PAGES = 4
HEADS, HEAD_DIM = 4, 8


@pytest.fixture(autouse = True)
def inference_mode():
    with torch.inference_mode():
        yield


class FakeRankModule:
    def __init__(self, key, lookup, first, last, total):
        self.key = key
        self.tp_cache_lookup = lookup
        self.num_kv_heads = last - first
        self.tp_kv_first = first
        self.tp_kv_total = total


class FakeTPModel:
    """Routes every dispatch into the rank functions in-process."""
    loaded_tp = True

    def __init__(self, ranks):
        self.ranks = ranks
        self.active_devices = [r["device"] for r in ranks]

    def tp_cpu_cache_init(self, cache_ids, max_slots = 0):
        return sum(mp_cpu_cache_init(r, cache_ids, max_slots) for r in self.ranks)

    def tp_cpu_cache_store(self, cache_ids, slot, page_index):
        for r in self.ranks:
            mp_cpu_cache_store(r, cache_ids, slot, page_index)

    def tp_cpu_cache_fetch(self, cache_ids, slot, page_index):
        for r in self.ranks:
            mp_cpu_cache_fetch(r, cache_ids, slot, page_index)

    def tp_worker_dispatch_wait_multi(self, devices, fn, args, dev_args = None):
        return [fn(r, *args, *(dev_args[i] if dev_args else ())) for i, r in enumerate(self.ranks)]

    def tp_dispatch_all(self, fn, args):
        for r in self.ranks:
            fn(r, *args)


def build_tp(full_k, full_v, split, root, fingerprint = "tp"):
    """A two-layer-tensor (k, v) cache sharded by kv heads per `split` (list of (first, last)), its CPU tier and
    disk tier. The rank tensors are views' copies of the full tensors, so the test knows the canonical image."""
    cache_id = 12345
    ranks = []
    for dev, (first, last) in enumerate(split):
        k = full_k[:, :, first:last].contiguous()
        v = full_v[:, :, first:last].contiguous()
        module = FakeRankModule("layer0", {cache_id: FakeCacheLayer([k, v])}, first, last, HEADS)
        ranks.append({"device": dev, "kv_modules": [module], "cpu_page_cache": None})
    model = FakeTPModel(ranks)
    # Cache.layers is keyed by placement tuples; the layer object carries the attention module (its key)
    cache = SimpleNamespace(layers = {("layer0", 0): SimpleNamespace(attention = FakeMainModule("layer0"))}, model = model)
    cache.fake_id = cache_id
    # CPUPageCache groups by id(cache) and dispatches with those ids; make id(cache) the id the ranks use
    _ids_patch(cache, cache_id)
    tier = CPUPageCache([cache], 4 * 1 << 20)
    disk = DiskPageCache(str(root), [cache], tier, 1 << 30, fingerprint = fingerprint, min_state_distance = 0)
    tier.attach_disk(disk)
    return ranks, tier, disk


class FakeMainModule:
    def __init__(self, key):
        self.key = key


def _ids_patch(cache, cache_id):
    # The ranks key their cache layers by the main process's id(cache); the fake ranks were built with a fixed
    # id, so rebind the lookup to the real one
    for m in cache.model.ranks:
        for module in m["kv_modules"]:
            module.tp_cache_lookup[id(cache)] = module.tp_cache_lookup.pop(cache_id)


def rank_views(ranks, slot):
    return [r["cpu_page_cache"].get(slot) for r in ranks]


def test_tp_page_round_trips_across_splits_and_into_a_non_tp_tier(device, tmp_path):
    g = torch.Generator(device = "cpu").manual_seed(0)
    full_k = torch.randn((NUM_PAGES, PAGE_SIZE, HEADS, HEAD_DIM), generator = g).half().to(device)
    full_v = torch.randn((NUM_PAGES, PAGE_SIZE, HEADS, HEAD_DIM), generator = g).half().to(device)

    # Written under a 2 + 2 split
    ranks, tier, disk = build_tp(full_k, full_v, [(0, 2), (2, 4)], tmp_path)
    tier.store(page(7, 1, page_size = PAGE_SIZE), serial = 1)
    disk.flush()
    assert disk.has_page(phash(7)) and disk.metrics["page_writes"] == 1
    image_size = disk.image_size
    disk.close()
    tier.close()

    # Read back under a 3 + 1 split: each rank's slab must hold its own heads of the same page
    ranks2, tier2, disk2 = build_tp(full_k, full_v, [(0, 3), (3, 4)], tmp_path)
    assert disk2.image_size == image_size and disk2.has_page(phash(7))
    e = tier2.promote(phash(7), serial = 2)
    assert e is not None and torch.equal(e["tokens"], torch.full((PAGE_SIZE,), 7, dtype = torch.long))
    for (first, last), views in zip([(0, 3), (3, 4)], rank_views(ranks2, e["slot"])):
        k_view, v_view = views
        assert torch.equal(k_view, full_k[1, :, first:last].cpu()), f"k shard {first}:{last} differs after restore"
        assert torch.equal(v_view, full_v[1, :, first:last].cpu()), f"v shard {first}:{last} differs after restore"
    # And through the ordinary fetch into another page of the rank tensors
    tier2.fetch(phash(7), 0, serial = 3)
    torch.cuda.synchronize()
    for (first, last), r in zip([(0, 3), (3, 4)], ranks2):
        k, v = r["kv_modules"][0].tp_cache_lookup[next(iter(r["kv_modules"][0].tp_cache_lookup))].get_tensors()
        assert torch.equal(k[0], full_k[1, :, first:last]) and torch.equal(v[0], full_v[1, :, first:last])
    disk2.close()
    tier2.close()

    # And into a non-TP tier over the full tensors (same fingerprint, same canonical layout)
    k3, v3 = torch.zeros_like(full_k), torch.zeros_like(full_v)
    cache3 = SimpleNamespace(layers = {0: FakeCacheLayer([k3, v3])})
    tier3 = CPUPageCache([cache3], 4 * 1 << 20)
    disk3 = DiskPageCache(str(tmp_path), [cache3], tier3, 1 << 30, fingerprint = "tp")
    tier3.attach_disk(disk3)
    assert disk3.image_size == image_size
    assert tier3.promote(phash(7), serial = 4) is not None
    tier3.fetch(phash(7), 2, serial = 5)
    torch.cuda.synchronize()
    assert torch.equal(k3[2], full_k[1]) and torch.equal(v3[2], full_v[1])
    disk3.close()
    tier3.close()


def test_non_tp_page_restores_into_tp_ranks(device, tmp_path):
    g = torch.Generator(device = "cpu").manual_seed(1)
    full_k = torch.randn((NUM_PAGES, PAGE_SIZE, HEADS, HEAD_DIM), generator = g).half().to(device)
    full_v = torch.randn((NUM_PAGES, PAGE_SIZE, HEADS, HEAD_DIM), generator = g).half().to(device)
    cache = SimpleNamespace(layers = {0: FakeCacheLayer([full_k, full_v])})
    tier = CPUPageCache([cache], 4 * 1 << 20)
    disk = DiskPageCache(str(tmp_path), [cache], tier, 1 << 30, fingerprint = "tp")
    tier.attach_disk(disk)
    tier.store(page(3, 2, page_size = PAGE_SIZE), serial = 1)
    disk.flush()
    disk.close()
    tier.close()

    ranks, tier2, disk2 = build_tp(torch.zeros_like(full_k), torch.zeros_like(full_v), [(0, 1), (1, 4)], tmp_path)
    assert tier2.promote(phash(3), serial = 2) is not None
    tier2.fetch(phash(3), 0, serial = 3)
    torch.cuda.synchronize()
    for (first, last), r in zip([(0, 1), (1, 4)], ranks):
        module = r["kv_modules"][0]
        k, v = module.tp_cache_lookup[next(iter(module.tp_cache_lookup))].get_tensors()
        assert torch.equal(k[0], full_k[2, :, first:last]) and torch.equal(v[0], full_v[2, :, first:last])
    disk2.close()
    tier2.close()


# ----------------------------------------------------------------------------------------------------------------
# Checkpoints: a layer state's tp_stash_pieces() maps a rank's stashed tensors into the canonical (full-width)
# checkpoint image. A GDN-like layer here: a recurrent state split by head and a conv state whose channels are
# three head-split projections in sequence, which is the hardest piece map in the tree.

GK, KD, G, VD = 4, 2, 2, 2          # k heads, k head dim, v heads per k head, v head dim
FDIM = 2 * GK * KD + GK * G * VD    # canonical conv channels: q | k | v


class FakeRecurrentLayer:
    """Rank-local state of a GDN-like layer for k heads [first, last)"""

    def __init__(self, first, last):
        self.nk = last - first
        self.nv = self.nk * G
        self.first = first
        self.module = SimpleNamespace(key = "layer0")

    def tp_stash_pieces(self):
        nk, nv, first = self.nk, self.nv, self.first
        return [
            {"dim": 2, "total": GK * G, "pieces": [(0, nv, first * G)]},
            {"dim": 0, "total": FDIM, "pieces": [
                (0, nk * KD, first * KD),
                (nk * KD, 2 * nk * KD, (GK + first) * KD),
                (2 * nk * KD, 2 * nk * KD + nv * VD, 2 * GK * KD + first * G * VD),
            ]},
        ]


def local_stash(full_rec, full_conv, first, last):
    """What a rank with k heads [first, last) would stash: its slice of the canonical tensors"""
    nk = last - first
    rec = full_rec[:, :, first * G : last * G].contiguous()
    conv = torch.cat([
        full_conv[first * KD : last * KD],
        full_conv[(GK + first) * KD : (GK + last) * KD],
        full_conv[2 * GK * KD + first * G * VD : 2 * GK * KD + last * G * VD],
    ]).contiguous()
    assert conv.shape[0] == 2 * nk * KD + nk * G * VD
    return (rec, conv)


def build_tp_states(full_k, full_v, split, root, handle = None, stash = None):
    """TP page tier plus recurrent modules on the ranks; with handle/stash, every rank holds its slice of
    `stash` (full canonical tensors) in its recurrent cache under handle"""
    ranks, tier, disk = build_tp(full_k, full_v, split, root)
    model = disk.cpu_tier.tp_groups[0][0]
    cache_id = disk.cpu_tier.tp_groups[0][1][0]
    for r, (first, last) in zip(ranks, split):
        layer = FakeRecurrentLayer(first, last)
        r["recurrent_modules"] = [SimpleNamespace(key = "layer0", tp_recurrent_lookup = {cache_id: layer})]
        r["recurrent_cache"] = {}
        if stash is not None:
            r["recurrent_cache"][handle] = [local_stash(*stash, first, last)]
    main_cache = SimpleNamespace(get_all_recurrent_layers = lambda: {("layer0", 0): SimpleNamespace(module = SimpleNamespace(key = "layer0"))})
    # id(cache) the ranks key by must be the one the describe/read dispatches carry
    disk.recurrent_sides = {"target": (model, SimpleNamespace(_id = cache_id, get_all_recurrent_layers = main_cache.get_all_recurrent_layers))}
    return ranks, tier, disk


def test_tp_checkpoint_round_trips_across_splits_and_configurations(device, tmp_path, monkeypatch):
    g = torch.Generator(device = "cpu").manual_seed(2)
    full_k = torch.randn((NUM_PAGES, PAGE_SIZE, HEADS, HEAD_DIM), generator = g).half().to(device)
    full_v = torch.randn((NUM_PAGES, PAGE_SIZE, HEADS, HEAD_DIM), generator = g).half().to(device)
    full_rec = torch.randn((1, 1, GK * G, 4, 4), generator = g)
    full_conv = torch.randn((FDIM, 3), generator = g).bfloat16()
    key = phash(5)
    stashed_ints = {"position": 512, "window_beg": 256, "checkpoint_size": full_rec.numel() * 4 + full_conv.numel() * 2}

    # The disk tier dispatches with id(cache) of the side's cache object; the fake ranks are keyed by the page
    # tier's cache id, so make the side's cache object report that id
    import exllamav3.generator.disk_cache as dc
    monkeypatch.setattr(dc, "id", lambda o: o._id if hasattr(o, "_id") else id(o), raising = False)

    # Written by ranks under a 2 + 2 k-head split
    ranks, tier, disk = build_tp_states(full_k, full_v, [(0, 2), (2, 4)], tmp_path, handle = 7, stash = (full_rec, full_conv))
    stashed = {**stashed_ints, "target": {**stashed_ints, "tp_handle": 7}}
    assert disk.write_state(key, stashed)
    disk.flush()
    assert disk.has_state(key) and disk.metrics["state_writes"] == 1
    disk.close(); tier.close()

    # Read by ranks under a 1 + 3 split: each rank's stash must be its slice of the canonical tensors
    ranks2, tier2, disk2 = build_tp_states(full_k, full_v, [(0, 1), (1, 4)], tmp_path)
    back = disk2.read_state(key)
    assert back is not None and back["position"] == 512 and back["target"]["window_beg"] == 256
    h = back["target"]["tp_handle"]
    for r, (first, last) in zip(ranks2, [(0, 1), (1, 4)]):
        rec, conv = r["recurrent_cache"][h][0]
        e_rec, e_conv = local_stash(full_rec, full_conv, first, last)
        assert rec.dtype == e_rec.dtype and torch.equal(rec, e_rec), f"recurrent state slice {first}:{last} differs"
        assert conv.dtype == e_conv.dtype and torch.equal(conv, e_conv), f"conv state slice {first}:{last} differs"
    disk2.close(); tier2.close()

    # Read without TP: the side comes back as a stash dict with the full tensors under the cache's own key
    k3, v3 = torch.zeros_like(full_k), torch.zeros_like(full_v)
    cache3 = SimpleNamespace(layers = {0: FakeCacheLayer([k3, v3])})
    tier3 = CPUPageCache([cache3], 4 * 1 << 20)
    disk3 = DiskPageCache(str(tmp_path), [cache3], tier3, 1 << 30, fingerprint = "tp", min_state_distance = 0)
    tier3.attach_disk(disk3)
    local_model = SimpleNamespace(loaded_tp = False)
    local_cache = SimpleNamespace(get_all_recurrent_layers = lambda: {("layer0", 0): SimpleNamespace(module = SimpleNamespace(key = "layer0"))})
    disk3.recurrent_sides = {"target": (local_model, local_cache)}
    back3 = disk3.read_state(key)
    rec, conv = back3["target"][("layer0", 0)]
    assert torch.equal(rec, full_rec) and torch.equal(conv, full_conv) and "tp_handle" not in back3["target"]

    # And the other direction: written without TP, read by ranks
    key2 = phash(6)
    local_stashed = {**stashed_ints, "target": {**stashed_ints, ("layer0", 0): (full_rec.clone(), full_conv.clone())}}
    assert disk3.write_state(key2, local_stashed)
    disk3.flush()
    disk3.close(); tier3.close()
    ranks4, tier4, disk4 = build_tp_states(full_k, full_v, [(0, 3), (3, 4)], tmp_path)
    back4 = disk4.read_state(key2)
    h4 = back4["target"]["tp_handle"]
    for r, (first, last) in zip(ranks4, [(0, 3), (3, 4)]):
        rec, conv = r["recurrent_cache"][h4][0]
        e_rec, e_conv = local_stash(full_rec, full_conv, first, last)
        assert torch.equal(rec, e_rec) and torch.equal(conv, e_conv)
    disk4.close(); tier4.close()
