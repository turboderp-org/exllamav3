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


class RecurrentCache(OrderedDict):
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model

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
                host_pool.give(popped)
                note_freed(popped["checkpoint_size"])
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))

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
            host_pool.give(popped)
            note_freed(popped["checkpoint_size"])
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
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
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        self.current_size = 0
        self.pagetable = None
        if freed:
            note_freed(freed)
        host_pool.release()
        if self.model.loaded_tp:
            self.model.tp_dispatch_all(mp_host_pool_release, ())
        malloc_trim()


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


# [Path A] Only the GDN/Mamba2 layer states index the doubled recurrent_state pool by parity
# (GDNLayerState.parity_indexed); every other recurrent layer ignores the argument. A state
# class that does not track parity (SWA, ShortConv, DSA) passes None instead of a literal 0:
# stashing a parity-indexed layer under a parity its state class never owned would silently
# checkpoint the wrong pool row, so that combination is refused rather than guessed.
def resolve_recurrent_parity(recurrent_layers, parity: int | None) -> int:
    if parity is not None:
        return parity
    for layer in recurrent_layers:
        if getattr(layer, "parity_indexed", False):
            raise RuntimeError(
                "recurrent checkpoint: this cache holds parity-indexed (GDN/Mamba2) layers, "
                "so the stashing state class must carry the pool parity instead of None")
    return 0


# Per-rank functions for tensor-parallel mode

def mp_cache_recurrent_clear(local_context: dict, cache_id: int, slot: int):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        recurrent_layer = module.tp_recurrent_lookup[cache_id]
        recurrent_layer.clear(slot)


def mp_cache_recurrent_stash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int, parity: int | None):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    layers = [m.tp_recurrent_lookup[cache_id] for m in recurrent_modules]
    parity = resolve_recurrent_parity(layers, parity)
    stashed = []
    for l in layers:
        stashed.append(l.stash(slot, position, parity))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int, parity: int | None):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    layers = [m.tp_recurrent_lookup[cache_id] for m in recurrent_modules]
    parity = resolve_recurrent_parity(layers, parity)
    stashed = recurrent_cache[cp_handle]
    for l, s in zip(layers, stashed):
        l.unstash(slot, s, position, parity)


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
