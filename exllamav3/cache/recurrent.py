from collections import OrderedDict
from ..constants import PAGE_SIZE
import torch
from ..util.memory import malloc_trim

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

    def take(self, shape, dtype):
        key = (tuple(shape), dtype)
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
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
        draft_model = None,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model
        # The models behind each side of a stashed pair, for the tensor-parallel ranks' copies
        self.models = {"target": model, "draft": draft_model}

        # Optionally set by the Generator; enables stranded-first eviction and staleness metrics
        self.pagetable = None
        self.metrics = {
            "stash_evictions": 0,           # checkpoints dropped by LRU pressure
            "stash_evictions_stranded": 0,  # of those, checkpoints that were already unrestorable
            "stash_evictions_live_kv": 0,   # of those, checkpoints whose anchor KV page was still cached
            "stash_pruned": 0,              # stranded checkpoints dropped by prune_stranded()
        }


    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue
        """
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default


    def put(self, key, state):
        """
        Add state to cache
        """
        if key in self:
            self.move_to_end(key)
        else:
            # Evict before stashing so the pool's peak occupancy is the cache limit, not the limit
            # plus the incoming checkpoint: the evicted buffers are what the new stash reuses
            state_size = state.checkpoint_size
            while self.update_total_size() + state_size > self.max_size:
                assert self.current_size >= 0, "Not enough space in cache for single state"
                pt = self.pagetable

                # A checkpoint whose anchor page chain has been broken by KV eviction can never be restored by
                # an allocation, so drop stranded checkpoints (oldest first) before restorable ones. This is a
                # pure win: if the conversation returns, the replay prefill recreates the same checkpoint at no
                # extra cost, since the missing pages force a replay past this position either way.
                popped_key = None
                if pt is not None:
                    for k in self:
                        if not pt.is_resumable(k):
                            popped_key = k
                            break
                if popped_key is not None:
                    popped = self.pop(popped_key)
                    self.metrics["stash_evictions_stranded"] += 1
                else:
                    popped_key, popped = self.popitem(last = False)
                    if pt is not None:
                        page = pt.referenced_pages.get(popped_key) or pt.unreferenced_pages.get(popped_key)
                        if page is not None and page.kv_position == PAGE_SIZE:
                            self.metrics["stash_evictions_live_kv"] += 1

                self.metrics["stash_evictions"] += 1
                self._release(popped)

            self[key] = state.stash()
            self.update_total_size()


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
            popped = self.pop(k)
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
            _, popped = self.popitem(last = False)
            # Several keys may share one stash
            if id(popped) in seen:
                continue
            seen.add(id(popped))
            host_pool.give(popped)
            freed += popped["checkpoint_size"]
            self._release_tp(popped)
        self.current_size = 0
        self.pagetable = None
        if freed:
            note_freed(freed)
        host_pool.release()
        if self.model.loaded_tp:
            self.model.tp_dispatch_all(mp_host_pool_release, ())
        malloc_trim()


    def _release_tp(self, stashed: dict):
        """Drop the tensor-parallel ranks' copies of a stash: one handle per side of a pair (or at
        the top level of a bare state stash)"""
        for key in ("target", "draft", None):
            s = stashed if key is None else stashed.get(key)
            if not isinstance(s, dict) or "tp_handle" not in s:
                continue
            model = self.model if key is None else self.models[key]
            if model is not None and model.loaded_tp:
                model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), s["tp_handle"]))

    def _release(self, stashed: dict):
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


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int):
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache.pop(cp_handle)
    host_pool.give(stashed)
    note_freed(_stashed_bytes(stashed))
