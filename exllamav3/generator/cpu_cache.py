from __future__ import annotations
import heapq
import itertools
import threading
import weakref
import torch
from collections import deque
from ..constants import PAGE_SIZE


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


# Distinguishes worker-side pool generations across sequential CPUPageCache instances (see tp_caches)
_pool_id_serial = itertools.count()


class _SlabAllocState:
    """
    Shared state between a CPUPageCache and its slab-pinning worker. Holds no reference back to the cache,
    so an abandoned cache - reference cycles and all - stays collectable at every instant of preallocation;
    the cache's finalizer stops the worker instead of the worker noticing the cache is gone.
    """
    def __init__(self, slot_size, layouts, target):
        self.slot_size = slot_size
        self.layouts = layouts   # (offset, nbytes, dtype, shape) per segment; no tensors
        self.target = target     # total slabs that may ever exist (worker-made + cold-allocated)
        self.reserved = 0        # slabs made or being made; incremented before pinning starts
        self.spare = deque()
        self.cond = threading.Condition()
        self.stopped = False

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()


def _make_slab(slot_size, layouts):
    slab = torch.empty((slot_size,), dtype = torch.uint8, pin_memory = True)
    views = []
    for offset, nbytes, dtype, shape in layouts:
        views.append(slab[offset : offset + nbytes].view(dtype).view(shape))
    return slab, views


def _alloc_worker(state):
    """
    Pin slabs up to state.target, then exit. Receives only the shared allocation state - never the
    CPUPageCache, a bound method, or anything else referencing it - so the worker cannot delay collection of
    an abandoned cache at any point. The cache's finalizer sets state.stopped to end the thread early; cold
    allocations by the cache reserve against the same target, so the combined slab count never overshoots.
    """
    while True:
        with state.cond:
            if state.stopped or state.reserved >= state.target:
                return
            state.reserved += 1
        try:
            sv = _make_slab(state.slot_size, state.layouts)  # slow part, outside the lock
        except BaseException:
            # Return the reservation so a foreground allocation cannot wait forever for a slab that will
            # never be published. It will retry synchronously and surface the allocation error to the caller.
            with state.cond:
                state.reserved -= 1
                state.cond.notify_all()
            return
        with state.cond:
            if state.stopped:
                state.reserved -= 1
                state.cond.notify_all()
                return
            state.spare.append(sv)
            state.cond.notify_all()


def _finalize_tier(state, tp_frees):
    """
    Runs exactly once, when the CPUPageCache is collected (reference counting or cyclic GC) or explicitly
    via close(). Stops the slab-pinning worker and queues the worker-side TP pools for release. Queue only:
    this can run at any GC point, even mid-fan-out with a rank blocked in a collective, so no pipe I/O here;
    the frees drain at the next quiet boundary (another tier's dispatches, or a forward/prefill fan-out).
    Best effort at interpreter shutdown, when the TP context may already be gone.
    """
    if state is not None:
        state.stop()
    for model, pool_id in tp_frees:
        try:
            if model.loaded_tp:
                model.tp_host_cache_free_deferred(pool_id)
        except Exception:
            pass


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

    For a tensor-parallel model, the cache tensors live sharded across the TP worker processes, so the data
    plane moves there too: each rank allocates pinned pools covering its own shard up front, and store/fetch
    dispatch per-page commands to every rank under the same global slot index. All metadata (hashes, eviction
    order, token ids) stays in this class; the per-rank command pipes preserve ordering against forward passes
    exactly as the shared stream does in the single-process case. Local (non-TP) and TP caches can be mixed,
    e.g. a TP-loaded model with a single-device draft model.
    """

    def __init__(
        self,
        caches: list,
        max_size: int,
    ):
        """
        :param caches:
            List of Cache objects whose paged layers make up one page image, i.e. [cache] or
            [cache, draft_cache]. The model must be loaded (cache tensors allocated)

        :param max_size:
            Capacity in bytes of pinned system memory. Slots are allocated lazily as pages are pushed, so this
            is a ceiling, not an up-front allocation
        """

        # Segment table: one entry per paged cache tensor. Every cache tensor is page-major, so one page is the
        # contiguous slice tensor[page_index]. TP-loaded caches contribute no local segments; their per-rank
        # pools live in the worker processes, keyed as (model, cache_id, pool_id). pool_id is unique per store
        # generation: a predecessor store over the same cache may have its finalizer deferred by cyclic GC
        # until after this store has allocated, and its queued free must then target only its own dead pools.
        self.segments = []
        self.tp_caches = []
        offset = 0
        tp_page_bytes = 0
        for cache in caches:
            if getattr(getattr(cache, "model", None), "loaded_tp", False):
                cache.model.tp_host_cache_process_frees()
                self.tp_caches.append((cache.model, id(cache), next(_pool_id_serial)))
                tp_page_bytes += cache.model.tp_host_cache_page_bytes(id(cache))
                continue
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
        assert self.segments or self.tp_caches, "No paged cache layers to attach CPU page cache tier to."

        self.slot_size = _align(offset, 4096) if self.segments else 0
        page_bytes = self.slot_size + tp_page_bytes
        self.max_slots = int(max_size) // page_bytes
        assert self.max_slots >= 2, \
            f"CPU page cache of {max_size} bytes is smaller than two pages ({page_bytes} bytes per page)."

        # Per-rank pools are committed up front for the full slot count (pinning happens in the workers, in
        # parallel across ranks); the local slabs below stay lazily allocated as before
        for model, cache_id, pool_id in self.tp_caches:
            model.tp_host_cache_alloc(cache_id, pool_id, self.max_slots)
        self._slots_used = 0  # slots handed out when no local slabs exist to count

        self.pagetable = None

        # phash -> {slot, prev_hash, access_serial, tokens}
        self.entries = {}
        self.slot_slabs = []
        self.slot_views = []
        self.free_slots = deque()

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

        # Transfers run at PCIe speed, but pinning host memory only manages ~2.5 GB/s and serializes with copy
        # submission on the driver, so the full configured capacity is pinned up front by a background thread
        # (mirroring the GPU cache, whose full allocation is also committed at load), which exits once the
        # target is reached. Pushes that outrun it early in the process fall back to pinning synchronously.
        # With no local segments (pure-TP store) there are no slabs to pin and the thread is not started.
        self._alloc_state = None
        self._alloc_thread = None
        if self.segments:
            layouts = [
                (offset, t[0].numel() * t.element_size(), dtype, page_shape)
                for t, offset, page_shape, dtype in self.segments
            ]
            self._alloc_state = _SlabAllocState(self.slot_size, layouts, self.max_slots)
            self._alloc_thread = threading.Thread(
                target = _alloc_worker, args = (self._alloc_state,), daemon = True)
            self._alloc_thread.start()

        # The finalizer keeps itself alive, fires exactly once on any form of collection or via close(), and
        # captures no reference to self, so the worker thread and TP pool cleanup never depend on __del__
        # semantics or on when cyclic GC happens to run
        self._finalizer = weakref.finalize(
            self, _finalize_tier, self._alloc_state,
            [(model, pool_id) for model, cache_id, pool_id in self.tp_caches])


    def close(self):
        """
        Deterministically release this tier's background resources: stop the slab-pinning worker and queue
        any worker-side TP pools for release in the rank processes. Idempotent; runs automatically when the
        object is collected.
        """
        self._finalizer()


    def attach(self, pagetable):
        self.pagetable = pagetable


    def __contains__(self, phash: bytes):
        return phash in self.entries


    def __len__(self):
        return len(self.entries)


    def _new_slot(self, protect: set | None):
        if self.free_slots:
            return self.free_slots.popleft()
        if not self.segments:
            # No local slabs to materialize; slot indices only key the per-rank TP pools
            if self._slots_used < self.max_slots:
                self._slots_used += 1
                return self._slots_used - 1
            return self._evict_one(protect)
        if len(self.slot_slabs) < self.max_slots:
            state = self._alloc_state
            with state.cond:
                while True:
                    if state.stopped:
                        raise RuntimeError("CPU page cache tier is closed.")
                    if state.spare:
                        sv = state.spare.popleft()
                        self.slot_slabs.append(sv[0])
                        self.slot_views.append(sv[1])
                        return len(self.slot_slabs) - 1
                    if state.reserved < state.target:
                        state.reserved += 1
                        break
                    # The worker has reserved the remaining capacity but has not published its in-flight slab
                    # yet. Wait for that slab instead of exceeding the configured pinned-memory budget.
                    state.cond.wait()
            try:
                sv = _make_slab(state.slot_size, state.layouts)
            except BaseException:
                with state.cond:
                    state.reserved -= 1
                    state.cond.notify_all()
                raise
            with state.cond:
                if state.stopped:
                    state.reserved -= 1
                    state.cond.notify_all()
                    raise RuntimeError("CPU page cache tier was closed during slab allocation.")
            self.metrics["cold_allocs"] += 1
            self.slot_slabs.append(sv[0])
            self.slot_views.append(sv[1])
            return len(self.slot_slabs) - 1
        return self._evict_one(protect)


    def _evict_one(self, protect: set | None):
        assert self.entries, "CPU page cache has no entries to evict (logic error)"
        deferred = 0
        while True:
            if not self._order or self._order_pops >= self._order_rebuild:
                self._build_order()
                deferred = 0
            h = self._order.popleft()
            self._order_pops += 1
            # Entries claimed by the allocation in progress are skipped unless everything is protected
            if protect and h in protect and deferred < len(self.entries):
                self._order.append(h)
                deferred += 1
                continue
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
        if self.segments:
            for v, (t, _, _, _) in zip(self.slot_views[slot], self.segments):
                v.copy_(t[page.page_index], non_blocking = True)
        for model, cache_id, pool_id in self.tp_caches:
            model.tp_host_cache_process_frees()
            model.tp_host_cache_store(pool_id, slot, page.page_index)
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
        if self.segments:
            for v, (t, _, _, _) in zip(self.slot_views[e["slot"]], self.segments):
                t[page_index].copy_(v, non_blocking = True)
        for model, cache_id, pool_id in self.tp_caches:
            model.tp_host_cache_process_frees()
            model.tp_host_cache_restore(pool_id, e["slot"], page_index)
        self.metrics["restores"] += 1
        return e
