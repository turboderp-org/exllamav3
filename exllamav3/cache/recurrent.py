from collections import OrderedDict
from ..constants import PAGE_SIZE

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
            stashed_state = state.stash()
            state_size = stashed_state["checkpoint_size"]
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
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))

            self[key] = stashed_state
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
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        if stranded:
            self.update_total_size()
        return len(stranded)


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


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int):
    recurrent_cache = local_context["recurrent_cache"]
    del recurrent_cache[cp_handle]
