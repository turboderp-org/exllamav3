from collections import OrderedDict
from ..constants import PAGE_SIZE
import torch
from ..util.memory import malloc_trim
import threading


def is_content_hash(h: bytes) -> bool:
    # Mirrors generator.pagetable.is_content_hash (not imported: that module imports this one)
    return h[:8] != bytes(8)

# Checkpoint stashes are MB-scale host allocations with LRU (i.e. interleaved) lifetimes —
# exactly the churn glibc retains after free (issue #277). Return memory to the OS once
# enough has been released; per-event cost at this threshold is a few ms. The accumulator
# is per-process, which also gives each tensor-parallel rank its own (their stashes live
# in the child processes)
_TRIM_THRESHOLD = 256 * 1024**2
_freed_bytes = 0


class HostPool:
    """
    Reusable host buffers for recurrent checkpoints. A checkpoint is a few dozen multi-MiB tensors with
    lifetimes interleaved with everything else the process allocates; allocating them fresh per stash
    and freeing them on eviction is exactly the pattern that strands memory in glibc's arenas (the
    dynamic mmap threshold moves them into the heap after the first free, and what malloc_trim can
    then hand back depends on the allocator and the platform). Buffers are handed out by exact shape
    and dtype, returned on eviction, and never freed, so the pool's footprint is bounded by the peak
    checkpoint occupancy and the allocator sees no churn at all.
    """

    def __init__(self):
        self.free = {}
        self.allocated = 0
        self.reused = 0
        # The workers of a pipelined prefill take buffers for their devices' layers
        self.lock = threading.Lock()

    def take(self, shape, dtype):
        key = (tuple(shape), dtype)
        with self.lock:
            lst = self.free.get(key)
            if lst:
                self.reused += 1
                return lst.pop()
            self.allocated += 1
        return torch.empty(shape, dtype = dtype)

    def give(self, obj):
        """Return every tensor inside a stashed structure (dict / list / tuple of tensors) to the pool"""
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "cpu":
                with self.lock:
                    self.free.setdefault((tuple(obj.shape), obj.dtype), []).append(obj)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                self.give(o)
        elif isinstance(obj, dict):
            for k, o in obj.items():
                if k not in ("position", "checkpoint_size", "tp_handle"):
                    self.give(o)

    def release(self):
        """Drop the idle buffers (idle-transition housekeeping: the ones pruning stranded checkpoints
        just returned would otherwise hold their RAM for the whole idle period)"""
        with self.lock:
            self.free.clear()


host_pool = HostPool()


def mp_host_pool_release(local_context: dict):
    host_pool.release()


def host_copy(src: torch.Tensor) -> torch.Tensor:
    """Copy a device tensor (any strides) into a pooled host buffer; the stash-side replacement for .cpu()"""
    dst = host_pool.take(src.shape, src.dtype)
    dst.copy_(src)
    return dst

def note_freed(nbytes: int):
    global _freed_bytes
    _freed_bytes += nbytes
    if _freed_bytes >= _TRIM_THRESHOLD:
        _freed_bytes = 0
        malloc_trim()


class PairedState:
    """
    One sequence's recurrent state across the target model and the draft model, as the generator and
    jobs see it: each side that keeps recurrent state (GDN / Mamba2 pools, SWA rings, DSA pools) has its
    own state object from its own cache, and the pair moves together. Checkpoints stash both sides
    under one key at the target's page boundary, so a prompt-cache resume restores both, and one LRU
    eviction drops both. Either side may be absent (a dense target with a recurrent draft, or the
    common case today, a recurrent target and no recurrent draft).

    The verification pass advances the target side only; the draft side is advanced by the draft's own
    forwards, or by the projected rows a DFlash-family drafter writes, so rewind() after verification
    and after a prefill overshoot rewinds the target side alone. A checkpoint rewind (banned strings)
    brings every side to the same position (rewind_to), in place where each side's rollback capacity
    allows, otherwise the job restores the pair from a stash.
    """

    def __init__(self, target = None, draft = None):
        assert target is not None or draft is not None, "PairedState needs at least one side"
        self.target = target
        self.draft = draft

    @property
    def components(self):
        return [s for s in (self.target, self.draft) if s is not None]

    @property
    def primary(self):
        return self.target if self.target is not None else self.draft

    @property
    def position(self) -> int:
        return self.primary.position

    @property
    def last_history(self) -> int:
        return self.primary.last_history

    @property
    def checkpoint_size(self) -> int:
        return sum(s.checkpoint_size for s in self.components)

    def rewind(self, num_tokens: int):
        """Settle the target side after a speculative pass, or correct its position after a prefill
        overshoot (see the class note)"""
        if self.target is not None:
            self.target.rewind(num_tokens)

    @property
    def stash_by_rollback(self) -> bool:
        """Whether every side can reach a past page boundary in place when a job ends (SWA rings), so that no
        rolling checkpoint has to be copied while the job runs"""
        return all(getattr(s, "stash_by_rollback", False) for s in self.components)

    def can_stash_at(self, position: int) -> bool:
        return all(s.can_stash_at(position) for s in self.components)

    def settled_at(self, position: int) -> bool:
        return all(s.position == position and s.last_history == 0 for s in self.components)

    def can_rewind_to(self, position: int) -> bool:
        return all(0 <= s.position - position <= s.rollback_capacity() for s in self.components)

    def rewind_to(self, position: int):
        for s in self.components:
            s.rewind(s.position - position)

    def stash(self) -> dict:
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
        }
        if self.target is not None:
            stashed["target"] = self.target.stash()
        if self.draft is not None:
            stashed["draft"] = self.draft.stash()
        return stashed

    def free(self):
        for s in self.components:
            s.free()

    def reset(self):
        for s in self.components:
            s.reset()


class RecurrentCache(OrderedDict):
    """
    Host-memory store for recurrent checkpoints.

    A checkpoint is keyed by the chained content hash of the KV prefix it resumes from. When the state sits on
    a page boundary (the historical case) that key is the hash of the last complete page, which is also what
    Sequence.prepare computes for a later request. A checkpoint at the exact end of a prompt or request is
    usually *inside* its last page; it is keyed by the content hash of that partial page chained onto the last
    complete page, and indexed under the complete page's hash so a later request can enumerate the partial
    positions that share its prefix (`partial_keys`). The partial page's K/V is stored in the page tiers under
    the same partial key, so the existing chain walk (`PageTable.is_resumable`) covers it unchanged.

    Eviction is LRU by default, with stranded checkpoints (whose KV chain has been broken) always dropped
    first. With eviction_policy="gds" the cache instead applies GreedyDual-Size over the prefill seconds a
    checkpoint saves per byte it occupies (FlashStates' cost-aware hybrid state management), with the classic
    inflation term so recently inserted states are not immediately churned.
    """

    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
        draft_model = None,
        eviction_policy: str = "lru",
        prefill_cost_d: float = 1.0,
        prefill_cost_e: float = 0.0,
    ):
        super().__init__()
        assert eviction_policy in ("lru", "gds"), f"Unknown recurrent cache eviction policy: {eviction_policy}"
        self.max_size = max_size
        self.current_size = 0
        self.model = model
        # The models behind each side of a stashed pair, for the tensor-parallel ranks' copies
        self.models = {"target": model, "draft": draft_model}

        # Optionally set by the Generator; enables stranded-first eviction and staleness metrics
        self.pagetable = None
        # Optional disk tier (DiskPageCache): checkpoints are written through on put() and read back on a miss
        self.disk_tier = None
        # Optional PrefillPipeline: a checkpoint stashed behind chunks still in flight is filled in by it, and
        # goes to the disk tier once that's done
        self.pipeline = None
        self.unwritten = []

        # Arbitrary-position (partial-page) checkpoints: anchor page hash -> [checkpoint key]. Only partial
        # checkpoints are indexed; boundary checkpoints are found directly from a sequence's page hashes.
        self._partial = {}

        # GreedyDual-Size state: priority per key and the inflation term
        self.eviction_policy = eviction_policy
        self.prefill_cost_d = float(prefill_cost_d)
        self.prefill_cost_e = float(prefill_cost_e)
        self._gds_keys = {}
        self._gds_inflation = 0.0

        self.metrics = {
            "stash_evictions": 0,           # checkpoints dropped by eviction pressure
            "stash_evictions_stranded": 0,  # of those, checkpoints that were already unrestorable
            "stash_evictions_live_kv": 0,   # of those, checkpoints whose anchor KV page was still cached
            "stash_pruned": 0,              # stranded checkpoints dropped by prune_stranded()
            "stash_partial": 0,             # arbitrary-position checkpoints stored
        }


    def has(self, key) -> bool:
        """Whether a checkpoint for key can be produced, from RAM or from the disk tier"""
        return key in self or (self.disk_tier is not None and self.disk_tier.has_state(key))


    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue. A checkpoint that is only on disk is read
        back into the cache first.
        """
        if key in self:
            self.move_to_end(key)
            self._settle(self[key])
        if key in self:
            stashed = self[key]
            if self.eviction_policy == "gds":
                self._gds_keys[key] = self._gds_priority(stashed) + self._gds_inflation
            return stashed
        if self.disk_tier is not None:
            stashed = self.disk_tier.read_state(key)
            if stashed is not None:
                self._insert(key, stashed, persist = False)
                return stashed
        return default


    def position(self, key) -> int | None:
        """Absolute token position a checkpoint resumes from, from RAM or the disk index, or None"""
        stashed = OrderedDict.get(self, key)
        if stashed is not None:
            return stashed.get("position")
        if self.disk_tier is not None:
            meta = self.disk_tier.states.get(key)
            if meta is not None:
                return meta.get("position")
        return None


    def partial_keys(self, anchor_hash: bytes | None) -> list:
        """Keys of the arbitrary-position checkpoints anchored on a complete page hash (None = first page)"""
        return list(self._partial.get(anchor_hash, ()))


    def put_partial(self, key, state, anchor_hash: bytes | None, offset: int):
        """
        Add a checkpoint that ends `offset` tokens into the page following `anchor_hash` (0 < offset <
        PAGE_SIZE). The caller has already stored that partial page in the page tiers under `key`.
        """
        assert 0 < offset < PAGE_SIZE, "A partial checkpoint must sit inside a page"
        if key in self:
            self.move_to_end(key)
            self._index_partial(key, anchor_hash)
            if self.eviction_policy == "gds":
                self._gds_keys[key] = self._gds_priority(self[key]) + self._gds_inflation
            return
        self._insert(
            key,
            None,
            persist = True,
            state = state,
            meta = {
                "anchor_hash": anchor_hash.hex() if anchor_hash is not None else None,
                "offset": offset,
            },
        )
        if key in self:
            self.metrics["stash_partial"] += 1


    def rekey(self, old_key, new_key):
        """Move a checkpoint to the final hash of its anchor page (deferred pages are hashed once their
        tokens are final). Partial checkpoints are keyed by their own content and never re-key."""
        if old_key not in self or new_key in self:
            return
        stashed = self[old_key]
        if stashed.get("offset"):
            return
        self._pop_key(old_key)
        self._gds_keys[new_key] = self._gds_priority(stashed) + self._gds_inflation
        self[new_key] = stashed
        if self.disk_tier is not None:
            self.disk_tier.rekey_state(old_key, new_key, stashed)


    def put(self, key, state, owner = None):
        """
        Add state to cache. Behind prefill chunks in flight the stash is filled in as they finish, so the caller isn't
        held up; owner is the job that a failure doing that belongs to.
        """
        if key in self:
            self.move_to_end(key)
            if self.eviction_policy == "gds":
                self._gds_keys[key] = self._gds_priority(self[key]) + self._gds_inflation
        else:
            self._insert(key, None, persist = True, state = state, owner = owner)


    def adopt(self, key, stashed: dict):
        """Store a stash made outside the cache (a job's rolling checkpoint), taking ownership of it"""
        if key in self:
            self.move_to_end(key)
            self.recycle(stashed)
        elif stashed["checkpoint_size"] > self.max_size:
            self.recycle(stashed)
        else:
            self._insert(key, stashed, persist = True)


    def recycle(self, stashed: dict):
        """Hand a stash that never entered the cache back to the pool, for the next stash to reuse. Nothing
        reaches the allocator and no disk write can be pending, so this doesn't count towards a trim"""
        host_pool.give(stashed)
        self._release_tp(stashed, trim = False)


    def _settle(self, stashed: dict):
        """Wait for a stash that is still being filled in"""
        if self.pipeline is not None and id(stashed) in self.pipeline.pending_stashes:
            self.pipeline.flush()


    def write_settled(self):
        """
        Once the prefill pipeline has drained: hand the checkpoints filled in behind it to the disk tier, and take
        back any that didn't complete (the job failed along the way), which must never be restored
        """
        unwritten, self.unwritten = self.unwritten, []
        for key, stashed in unwritten:
            held = self.get(key) is stashed
            if id(stashed) in self.pipeline.failed_stashes:
                self.pipeline.failed_stashes.discard(id(stashed))
                if held:
                    del self[key]
                    self.update_total_size()
                    host_pool.give(stashed)
            elif held and self.disk_tier is not None and is_content_hash(key):
                self.disk_tier.note_state(key, stashed)


    def _gds_priority(self, stashed: dict) -> float:
        """Prefill seconds a checkpoint saves per byte it occupies (GreedyDual-Size value / cost)"""
        size = max(1, int(stashed.get("checkpoint_size", 0)))
        value = self.prefill_cost_d * max(0, int(stashed.get("position", 0))) + self.prefill_cost_e
        return (value if value > 0.0 else 1e-9) / size


    def _index_partial(self, key, anchor_hash):
        lst = self._partial.setdefault(anchor_hash, [])
        if key not in lst:
            lst.append(key)


    def _index_partial_from_stash(self, key, stashed: dict):
        offset = stashed.get("offset", 0)
        if not offset:
            return
        anchor = stashed.get("anchor_hash")
        if isinstance(anchor, str):
            anchor = bytes.fromhex(anchor)
        self._index_partial(key, anchor)


    def _unindex_partial(self, key, stashed: dict):
        if not stashed.get("offset", 0):
            return
        # A state the disk tier still holds stays findable: get_stashed() reads it back into RAM on demand.
        # Its position comes from the disk index, and find_resume() rejects it if the page chain is gone
        if self.disk_tier is not None and key in self.disk_tier.states:
            return
        anchor = stashed.get("anchor_hash")
        if isinstance(anchor, str):
            anchor = bytes.fromhex(anchor)
        lst = self._partial.get(anchor)
        if lst is not None and key in lst:
            lst.remove(key)
            if not lst:
                self._partial.pop(anchor, None)


    def _pop_key(self, key):
        stashed = self.pop(key)
        self._gds_keys.pop(key, None)
        self._unindex_partial(key, stashed)
        return stashed


    def _victim(self):
        """Key and stash to drop: stranded first, then the GDS minimum or the least recently used entry"""
        pt = self.pagetable

        # A checkpoint whose anchor page chain has been broken by KV eviction can never be restored by an
        # allocation, so drop stranded checkpoints before restorable ones. This is a pure win: if the
        # conversation returns, the replay prefill recreates the same checkpoint at no extra cost, since the
        # missing pages force a replay past this position either way.
        if pt is not None:
            for k in self:
                if not pt.is_resumable(k):
                    self.metrics["stash_evictions_stranded"] += 1
                    return k, self._pop_key(k)

        if self.eviction_policy == "gds" and self._gds_keys:
            k = min(self._gds_keys, key = self._gds_keys.get)
            if k in self:
                self._gds_inflation = max(self._gds_inflation, self._gds_keys[k])
                return k, self._pop_key(k)
            self._gds_keys.pop(k, None)

        k = next(iter(self))
        if pt is not None:
            page = pt.referenced_pages.get(k) or pt.unreferenced_pages.get(k)
            if page is not None and page.kv_position == PAGE_SIZE:
                self.metrics["stash_evictions_live_kv"] += 1
        return k, self._pop_key(k)


    def _insert(self, key, stashed, persist: bool, state = None, owner = None, meta: dict | None = None):
        """Make room for a checkpoint, then stash `state` (or take `stashed` as is) under key"""
        # Evict before stashing so the pool's peak occupancy is the cache limit, not the limit
        # plus the incoming checkpoint: the evicted buffers are what the new stash reuses
        state_size = state.checkpoint_size if state is not None else stashed["checkpoint_size"]
        # A checkpoint larger than the whole budget (e.g. -rcs 0) is not stored at all, without evicting or
        # stashing anything; a later restore misses and falls back to replay
        if state_size > self.max_size:
            return
        while self.update_total_size() + state_size > self.max_size:
            assert self.current_size >= 0, "Not enough space in cache for single state"
            _, popped = self._victim()
            self.metrics["stash_evictions"] += 1
            self._release(popped)

        deferred = state is not None and self.pipeline is not None and self.pipeline.busy
        if deferred:
            stashed = self.pipeline.stash(state, owner)
        elif state is not None:
            stashed = state.stash()
        if state is not None and meta:
            stashed.update(meta)
        self._index_partial_from_stash(key, stashed)
        if self.eviction_policy == "gds":
            self._gds_keys[key] = self._gds_priority(stashed) + self._gds_inflation
        self[key] = stashed
        self.update_total_size()
        if deferred:
            self.unwritten.append((key, stashed))
        elif persist and self.disk_tier is not None and is_content_hash(key):
            self.disk_tier.note_state(key, stashed)


    def rebuild_partial_index(self):
        """
        Rebuild the anchor index of arbitrary-position checkpoints from RAM and the disk tier. Called once
        the disk tier is attached: partial checkpoints written in an earlier process have no in-RAM index
        entry, and their positions (not page-aligned) identify them in the disk index without reading files.
        """
        self._partial.clear()
        for k, stashed in self.items():
            self._index_partial_from_stash(k, stashed)
        disk = self.disk_tier
        if disk is None:
            return
        for k, meta in list(disk.states.items()):
            if k in self or meta.get("position", 0) % PAGE_SIZE == 0:
                continue
            if not disk.has_page(k):
                continue
            self._index_partial(k, disk.page_prev_hash(k))


    def prune_stranded(self) -> int:
        """
        Drop all checkpoints whose anchor page chain has been broken by KV eviction. A stranded checkpoint can
        never be restored by an allocation, and if its conversation returns, the replay prefill recreates it at
        no extra cost, so this only frees system RAM that would otherwise sit dead until LRU pressure reaches it.
        Intended to be called when the generator goes idle.
        """
        if self.pagetable is None:
            return 0
        stranded = [k for k in self if not self.pagetable.is_resumable(k)]
        for k in stranded:
            popped = self._pop_key(k)
            self.metrics["stash_pruned"] += 1
            self._release(popped)
        if stranded:
            self.update_total_size()
        return len(stranded)


    def close(self):
        """
        Drop every checkpoint, return its buffers to the stash pool and release the pool, so the RAM goes
        back to the OS now rather than when this object is garbage collected. For a generator being retired:
        nothing restores from a closed cache, and a replacement generator's own cache would otherwise fill up
        alongside the checkpoints still stashed here. Safe to call more than once.
        """
        seen = set()
        freed = 0
        while len(self):
            k = next(iter(self))
            popped = self._pop_key(k)
            # Several keys may share one stash
            if id(popped) in seen:
                continue
            seen.add(id(popped))
            host_pool.give(popped)
            freed += popped["checkpoint_size"]
            self._release_tp(popped)
        self._partial.clear()
        self._gds_keys.clear()
        self._gds_inflation = 0.0
        self.current_size = 0
        self.pagetable = None
        if freed:
            note_freed(freed)
        host_pool.release()
        if self.model.loaded_tp:
            self.model.tp_dispatch_all(mp_host_pool_release, ())
        malloc_trim()


    def _release_tp(self, stashed: dict, trim: bool = True):
        """Drop the tensor-parallel ranks' copies of a stash: one handle per side of a pair (or at
        the top level of a bare state stash)"""
        for key in ("target", "draft", None):
            s = stashed if key is None else stashed.get(key)
            if not isinstance(s, dict) or "tp_handle" not in s:
                continue
            model = self.model if key is None else self.models[key]
            if model is not None and model.loaded_tp:
                model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), s["tp_handle"], trim))

    def _release(self, stashed: dict):
        self._settle(stashed)
        # A stash the disk tier is still writing keeps its buffers until the write is done
        if self.disk_tier is not None:
            self.disk_tier.release_after_write(stashed, self._release_now)
        else:
            self._release_now(stashed)

    def _release_now(self, stashed: dict):
        host_pool.give(stashed)
        note_freed(stashed["checkpoint_size"])
        self._release_tp(stashed)

    def update_total_size(self):
        seen = set()
        total = 0
        for v in self.values():
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += v["checkpoint_size"]
        self.current_size = total
        return total


# Checkpoint handles key the per-rank recurrent_cache dicts and must be unique across all
# recurrent module types (GDN, short-conv, SWA states all stash through the same dict)
_next_checkpoint_handle = 0

def new_checkpoint_handle() -> int:
    global _next_checkpoint_handle
    h = _next_checkpoint_handle
    _next_checkpoint_handle += 1
    return h


# Per-rank functions for tensor-parallel mode

def mp_cache_recurrent_clear(local_context: dict, cache_id: int, slot: int):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        recurrent_layer = module.tp_recurrent_lookup[cache_id]
        recurrent_layer.clear(slot)


def mp_cache_recurrent_stash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = []
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        stashed.append(l.stash(slot, position))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache[cp_handle]
    for module, s in zip(recurrent_modules, stashed):
        l = module.tp_recurrent_lookup[cache_id]
        l.unstash(slot, s, position)


def _stashed_bytes(obj) -> int:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_stashed_bytes(o) for o in obj)
    return 0


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int, trim: bool = True):
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache.pop(cp_handle)
    host_pool.give(stashed)
    if trim:
        note_freed(_stashed_bytes(stashed))
