from __future__ import annotations
import heapq
import threading
import torch
from collections import deque
from ..constants import PAGE_SIZE


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class CPUPageCache:
    """
    Second-tier page cache in pinned system memory.

    Stores complete, hashed K/V pages evicted from the GPU page cache, keyed by the same chained page hash the
    PageTable uses. Each entry occupies one fixed-size slot holding the concatenated per-layer cache state of one
    page across all attached caches (main and draft cache, so restored pages remain valid for speculative
    decoding). Pages are pushed when PageTable.evict repurposes them and restored during allocation whenever a
    missing page hash is found here, replacing a prefill pass over the page with one host-to-device copy per
    layer tensor.

    All transfers are enqueued on each cache tensor's current stream. Pushes are therefore ordered before
    anything that could overwrite the dying page, restores are ordered before any kernel that could read the
    restored page, and slot recycling is race-free because a given byte range of a slot only ever moves over the
    stream of its fixed segment device.

    Eviction mirrors the GPU tier's policy: orphaned chains first, then whole trees, least recently used root
    first, pruned tail-first. A chain whose parent page is still live in the GPU page table counts as rooted.
    The order is consumed as a snapshot and rebuilt after at most max(64, max_slots/8) evictions, so recency
    updates between rebuilds are approximated.
    """

    def __init__(
        self,
        caches: list,
        max_size: int,
    ):
        """
        :param caches:
            List of Cache objects whose paged layers make up one page image, i.e. [cache] or
            [cache, draft_cache]. The model must be loaded (cache tensors allocated). Each cache names its
            own model, which is what the dispatch follows: in tensor-parallel mode the cache is sharded
            across worker processes and the main process holds no tensors to copy from, so this object keeps
            only the slot table and the eviction policy and every transfer is dispatched per rank. A slot
            index means the same page image on every rank, and the budget counts whole pages across all of
            them

        :param max_size:
            Capacity in bytes of pinned system memory. Slots are allocated lazily as pages are pushed, so this
            is a ceiling, not an up-front allocation
        """

        # A draft cache belongs to the draft model, with its own workers and its own view of which cache ids
        # exist, so caches are grouped by owning model and each group is dispatched separately. Grouping is by
        # identity rather than by id() so that a model whose object is recreated is never conflated with the
        # old one.
        tp_groups = {}
        local_caches = []
        for cache in caches:
            cache_model = getattr(cache, "model", None)
            if cache_model is not None and cache_model.loaded_tp:
                group = tp_groups.setdefault(id(cache_model), (cache_model, []))
                group[1].append(id(cache))
            else:
                local_caches.append(cache)

        self.tp_groups = list(tp_groups.values())
        self.tp = bool(self.tp_groups)

        # The two are not exclusive: a model can be TP while its MTP draft model, being one layer, is not, in
        # which case a slot has a local slab for the draft cache and per-rank buffers for the main one. Both
        # halves are keyed by the same slot index, so a page is stored and restored as one image either way.
        #
        # Segment table: one entry per paged local cache tensor. Every cache tensor is page-major, so one page
        # is the contiguous slice tensor[page_index]. TP caches have no entry here, since their tensors live in
        # the workers and the ranks hold the host buffers themselves.
        self.segments = []
        offset = 0
        for cache in local_caches:
            for layer in cache.layers.values():
                for t in layer.get_tensors():
                    if t is None:
                        continue
                    assert t.device.type == "cuda", \
                        "Cannot build CPU page cache tier before the model (and its cache tensors) are loaded."
                    page_shape = tuple(t.shape[1:])
                    nbytes = t[0].numel() * t.element_size()
                    self.segments.append((t, offset, page_shape, t.dtype))
                    offset = _align(offset + nbytes, 256)

        # Size of this process's half of a slot, which is what a slab has to hold
        self.slab_size = _align(offset, 4096) if self.segments else 0

        tp_bytes = 0
        for cache_model, cache_ids in self.tp_groups:
            tp_bytes += cache_model.tp_cpu_cache_init(cache_ids)
        assert offset or tp_bytes, "No paged cache layers to attach CPU page cache tier to."

        # A whole page image, across every rank and this process, which is what the budget is spent in
        self.slot_size = _align(offset + tp_bytes, 4096)
        self.max_slots = int(max_size) // self.slot_size
        assert self.max_slots >= 2, \
            f"CPU page cache of {max_size} bytes is smaller than two pages ({self.slot_size} bytes per page)."

        for cache_model, cache_ids in self.tp_groups:
            cache_model.tp_cpu_cache_init(cache_ids, self.max_slots)

        self.pagetable = None

        # phash -> {slot, prev_hash, access_serial, tokens}
        self.entries = {}
        self.slot_slabs = []
        self.slot_views = []
        self.free_slots = deque()
        # Slots handed out so far. In TP mode this is the only record the main process keeps of them, since the
        # buffers themselves belong to the ranks
        self.num_slots = 0

        # Eviction order snapshot
        self._order = deque()
        self._order_pops = 0
        self._order_rebuild = max(64, self.max_slots // 8)

        self.metrics = {
            "pushes": 0,        # pages copied to the tier on GPU eviction
            "dedup_hits": 0,    # pushes skipped because the page was already stored
            "restores": 0,      # pages copied back into the GPU cache at allocation
            "evictions": 0,     # tier entries dropped to make room
            "cold_allocs": 0,   # pushes that had to pin a slab synchronously (spare pool was empty)
        }
        # cold_allocs is the sum of the two halves, since a slot can have a local slab and per-rank buffers
        self._local_cold_allocs = 0
        self._tp_cold_allocs = 0

        # Transfers run at PCIe speed, but pinning host memory only manages ~2.5 GB/s and serializes with copy
        # submission on the driver, so the full configured capacity is pinned up front by a background thread
        # (mirroring the GPU cache, whose full allocation is also committed at load). Pushes that outrun the
        # pinning thread early in the process fall back to pinning synchronously. Each rank runs the same for
        # its own shard, so there is nothing to pin here for a cache that is entirely TP.
        self._spare = deque()
        self._spare_cond = threading.Condition()
        if self.segments:
            self._alloc_thread = threading.Thread(target = self._alloc_worker, daemon = True)
            self._alloc_thread.start()


    def attach(self, pagetable):
        self.pagetable = pagetable


    def __contains__(self, phash: bytes):
        return phash in self.entries


    def __len__(self):
        return len(self.entries)


    def _make_slab(self):
        slab = torch.empty((self.slab_size,), dtype = torch.uint8, pin_memory = True)
        views = []
        for t, offset, page_shape, dtype in self.segments:
            nbytes = t[0].numel() * t.element_size()
            views.append(slab[offset : offset + nbytes].view(dtype).view(page_shape))
        return slab, views


    def _alloc_worker(self):
        with torch.inference_mode():
            while True:
                with self._spare_cond:
                    while len(self.slot_slabs) + len(self._spare) >= self.max_slots:
                        self._spare_cond.wait()
                sv = self._make_slab()  # slow part, outside the lock
                with self._spare_cond:
                    self._spare.append(sv)


    def _new_slot(self, protect: set | None):
        """
        Index of a slot to write into: a recycled one, a fresh one while the budget allows, or the eviction
        candidate. The index is the whole identity of a slot, since the ranks key their own buffers by it, so
        a local slab (when there is one) is appended in lockstep and stays at the matching position.
        """

        if self.free_slots:
            return self.free_slots.popleft()
        if self.num_slots >= self.max_slots:
            return self._evict_one(protect)

        if self.segments:
            sv = None
            with self._spare_cond:
                if self._spare:
                    sv = self._spare.popleft()
                    self._spare_cond.notify()
            if sv is None:
                sv = self._make_slab()  # slow part, outside the lock
                self._local_cold_allocs += 1
                self.metrics["cold_allocs"] = self._local_cold_allocs + self._tp_cold_allocs
                with self._spare_cond:
                    self._spare_cond.notify()
            self.slot_slabs.append(sv[0])
            self.slot_views.append(sv[1])

        self.num_slots += 1
        return self.num_slots - 1


    def _evict_one(self, protect: set | None):
        assert self.entries, "CPU page cache has no entries to evict (logic error)"
        # Entries claimed by the allocation in progress (a restore protects every page of the chain it is
        # bringing back, hundreds of pages on a long prompt) are passed over and requeued; they are only
        # taken once every entry has been deferred, i.e. everything is protected. A deferral does not count
        # as an order pop. Pops are consumed entries only, so one call walks past the whole protected run in
        # a single pass and reaches the first unprotected candidate
        deferred = 0
        while True:
            if not self._order or self._order_pops >= self._order_rebuild:
                self._build_order()
                deferred = 0
            h = self._order.popleft()
            if protect and h in protect and deferred < len(self.entries):
                self._order.append(h)
                deferred += 1
                continue
            self._order_pops += 1
            e = self.entries.pop(h, None)
            if e is not None:
                self.metrics["evictions"] += 1
                return e["slot"]


    def _build_order(self):
        """
        Snapshot of the eviction order over current entries; see class docstring.
        """
        entries = self.entries
        children = {h: [] for h in entries}
        roots = []
        orphan_roots = []
        pt = self.pagetable
        for h, e in entries.items():
            ph = e["prev_hash"]
            if ph is not None and ph in children:
                children[ph].append(h)
            elif ph is None or (pt is not None and pt.get_live_page(ph) is not None):
                roots.append(h)
            else:
                orphan_roots.append(h)

        order = []
        def prune(root):
            remaining = {}
            heap = []
            stack = [root]
            while stack:
                h = stack.pop()
                n = len(children[h])
                remaining[h] = n
                if n == 0:
                    e = entries[h]
                    heapq.heappush(heap, (e["access_serial"], e["slot"], h))
                stack.extend(children[h])
            while heap:
                _, _, h = heapq.heappop(heap)
                order.append(h)
                ph = entries[h]["prev_hash"]
                if ph is not None and ph in remaining:
                    remaining[ph] -= 1
                    if remaining[ph] == 0:
                        e = entries[ph]
                        heapq.heappush(heap, (e["access_serial"], e["slot"], ph))

        serial = lambda h: entries[h]["access_serial"]
        for root in sorted(orphan_roots, key = serial):
            prune(root)
        for root in sorted(roots, key = serial):
            prune(root)

        self._order = deque(order)
        self._order_pops = 0


    def store(self, page, serial: int, protect: set | None = None):
        """
        Copy a dying page's cache state into the tier (device-to-host, async on the current stream). Duplicate
        hashes only refresh the entry's recency.
        """
        e = self.entries.get(page.phash)
        if e is not None:
            e["access_serial"] = serial
            self.metrics["dedup_hits"] += 1
            return
        slot = self._new_slot(protect)
        for v, (t, _, _, _) in zip(self.slot_views[slot] if self.segments else (), self.segments):
            v.copy_(t[page.page_index], non_blocking = True)
        if self.tp:
            # Each rank copies its own shard of the page. The count of synchronous pins comes back from them,
            # since that stall happens in the workers, and is added to any this process incurred for a local
            # slab so the metric stays one number
            self._tp_cold_allocs = sum(
                m.tp_cpu_cache_store(ids, slot, page.page_index) for m, ids in self.tp_groups
            )
            self.metrics["cold_allocs"] = self._local_cold_allocs + self._tp_cold_allocs
        self.entries[page.phash] = {
            "slot": slot,
            "prev_hash": page.prev_hash,
            "access_serial": serial,
            "tokens": page.sequence.clone(),
        }
        self.metrics["pushes"] += 1


    def fetch(self, phash: bytes, page_index: int, serial: int) -> dict:
        """
        Copy a stored page back into the GPU cache at page_index (host-to-device, async on the current stream).
        The entry remains in the tier; the restored copy may well be evicted again before this one goes stale.
        Returns the entry so the caller can restore page metadata (token IDs).
        """
        e = self.entries[phash]
        e["access_serial"] = serial
        for v, (t, _, _, _) in zip(self.slot_views[e["slot"]] if self.segments else (), self.segments):
            t[page_index].copy_(v, non_blocking = True)
        for m, ids in self.tp_groups:
            m.tp_cpu_cache_fetch(ids, e["slot"], page_index)
        self.metrics["restores"] += 1
        return e
