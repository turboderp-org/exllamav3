from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import torch
from ..constants import PAGE_SIZE


def wait_stash_ready(stashed: dict):
    """
    Block until a recurrent-cache entry's background finalization (pinned -> pageable clone) has completed.
    Entries written through the pinned-staging path carry a "pending" future until their layer tensors are
    owned pageable copies; restoring from such an entry must join it first. A completed future is a no-op.
    """
    fut = stashed.get("pending")
    if fut is not None:
        fut.result()


def _finalize_stash(stashed: dict, events: list = None):
    """
    Background-clone every staged layer value into pageable storage. Runs on the RecurrentCache worker thread;
    layer values are keyed by module instance (non-string keys), metadata by string keys. The events gate the
    side-stream D2H copies into staging: they must fire before the staged bytes are valid to read. Reading
    pinned inference tensors from a plain thread is safe; only in-place mutation requires inference mode.
    """
    if events:
        for ev in events:
            ev.synchronize()
    for k in list(stashed.keys()):
        if isinstance(k, str):
            continue
        stashed[k] = clone_staged_tensors(stashed[k])


def stage_tensors_pinned(owner, tensors):
    """
    Copy tensors into reusable pinned host buffers. The caller must synchronize the source device streams before
    reading or cloning the returned buffers.
    """
    single = isinstance(tensors, torch.Tensor)
    tensors = (tensors,) if single else tensors
    staging = getattr(owner, "_pinned_stash_staging", None)
    if staging is None or len(staging) != len(tensors) or any(
        b.shape != t.shape or b.dtype != t.dtype for b, t in zip(staging, tensors)
    ):
        staging = tuple(
            torch.empty(t.shape, dtype = t.dtype, device = "cpu", pin_memory = True)
            for t in tensors
        )
        owner._pinned_stash_staging = staging
    for buffer, tensor in zip(staging, tensors):
        buffer.copy_(tensor, non_blocking = True)
    return staging[0] if single else staging


def clone_staged_tensors(tensors):
    """
    Move a completed pinned staging copy into ordinary pageable storage owned by a recurrent-cache entry.
    """
    single = isinstance(tensors, torch.Tensor)
    tensors = (tensors,) if single else tensors
    cloned = tuple(
        torch.empty(t.shape, dtype = t.dtype, device = "cpu").copy_(t)
        for t in tensors
    )
    return cloned[0] if single else cloned


def stash_recurrent_layers(cache, slot: int, position: int = 0, pinned_staging: bool = False, cursor: int | None = None):
    """
    Stash all recurrent layers for one state slot. For the completion-tail path, enqueue D2H copies into reusable
    pinned buffers on per-device side streams and return events that gate both the pageable clone and slot reuse.
    cursor carries the SWA ring's true content extent; state types without a ring ignore it.
    """
    layers = cache.get_all_recurrent_layers()
    if not pinned_staging:
        return {key: layer.stash(slot, position, cursor = cursor) for key, layer in layers.items()}

    # Group layers by device and stage each device's D2H copies on a dedicated side stream that waits on the
    # main stream (which last wrote the ring state). The copies then overlap ongoing main-stream compute; the
    # recorded events gate both the background clone and the release of the state's slot back to the pool.
    by_device = {}
    for key, layer in layers.items():
        device = getattr(layer, "device", None)
        device = torch.device(device) if device is not None else None
        if device is not None and device.type != "cuda":
            device = None
        by_device.setdefault(device, []).append((key, layer))

    side_streams = getattr(cache, "_stash_side_streams", None)
    if side_streams is None:
        side_streams = {}
        cache._stash_side_streams = side_streams

    staged = {}
    events = []
    started_sides = []
    try:
        for device, dev_layers in by_device.items():
            if device is None:
                for key, layer in dev_layers:
                    staged[key] = layer.stash(slot, position, pinned_staging = True, cursor = cursor)
                continue
            side = side_streams.get(device)
            if side is None:
                side = torch.cuda.Stream(device = device)
                side_streams[device] = side
            started_sides.append(side)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                for key, layer in dev_layers:
                    staged[key] = layer.stash(slot, position, pinned_staging = True, cursor = cursor)
            ev = torch.cuda.Event()
            ev.record(side)
            events.append(ev)
    except Exception:
        # The caller cannot defer its slot release without a returned event list. Make the failure path
        # synchronous so any copies queued before the exception have stopped reading the slot.
        for side in started_sides:
            side.synchronize()
        raise

    # Staged pinned references become valid once the events fire, and stay valid until the next stash
    # reuses the staging buffers. The caller (RecurrentCache.put) clones them out on its background worker.
    return staged, events


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
        # Single worker serializes background clones, which also serializes reuse of the shared pinned
        # staging buffers: put() joins the previous stash's future before staging the next one
        self._clone_executor = None
        self._last_stash_future = None

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


    def put(self, key, state, pinned_staging: bool = False):
        """
        Add state to cache. With pinned_staging, the D2H copies are enqueued on side streams and gated by CUDA
        events; the clone into the entry's own pageable storage runs on a background worker after the events
        fire. The entry is inserted immediately with a "pending" future that wait_stash_ready joins before the
        entry can be restored. Returns the staging events (empty/None when nothing was staged): the caller must
        not release the state's slot until they have fired, since the side streams are still reading it.
        """
        if key in self:
            self.move_to_end(key)
            return None
        events = None
        fut = None
        try:
            if pinned_staging:
                # The pinned staging buffers are owned by the Cache's layer states, which multiple
                # RecurrentCache instances can share, so the reuse guard must live on the Cache: whichever
                # put() staged last must have finished cloning out of the buffers before they are overwritten
                guard = getattr(state.cache, "_stash_staging_guard", None)
                if guard is not None:
                    guard.result()
                stashed_state, events = state.stash(pinned_staging = True)
            else:
                stashed_state, events = state.stash(), None
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

            if pinned_staging:
                if self._clone_executor is None:
                    self._clone_executor = ThreadPoolExecutor(max_workers = 1)
                fut = self._clone_executor.submit(_finalize_stash, stashed_state, events)
                # Install the shared-buffer guard immediately after submission, before publishing the entry
                state.cache._stash_staging_guard = fut
                self._last_stash_future = fut
                # The completed future stays in the entry; wait_stash_ready treats it as a no-op. Evicting a
                # still-pending entry is safe: the worker fills an orphaned dict nobody will read
                stashed_state["pending"] = fut
            self[key] = stashed_state
            self.update_total_size()
            return events
        except Exception:
            # cleanup_completed_jobs releases the state in a finally block. If staging has already begun,
            # make that immediate release safe before allowing the exception to escape.
            if fut is not None:
                try:
                    fut.result()
                except Exception:
                    pass
            elif events:
                for ev in events:
                    ev.synchronize()
            raise


    def drain(self):
        """
        Join any in-flight background clone, for deterministic teardown. Errors surface to readers of the
        affected entry, not here.
        """
        if self._last_stash_future is not None:
            try:
                self._last_stash_future.result()
            except Exception:
                pass


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
