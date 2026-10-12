"""
Streamed frozen MoE experts for the native QLoRA path (Qwen3.8-Flash-Next
plan, Phase C -- the 16 GB tier). The routed experts' packed EXL3 trellis
tensors live in page-locked host RAM and are copied into a small ring of VRAM
slots one MoE layer at a time, just ahead of the reconstruct kernel that reads
them. The dense trunk, routers, shared experts, hyper-connection tables, PLE
projections, embedding and head stay resident on the GPU as before.

Why this is enough. The differentiable MoE forward (``NativeLlamaQLoRA._moe_out``)
never holds dense expert weights: every touched expert's ``[in, out]`` inner
weight is reconstructed on the fly by ``LinearEXL3.get_inner_weight_tensor``
(``backbone.frozen_trellis_parts`` -> ``EXL3LoRAHadFunction``, and the legacy
``get_weight_tensor`` path calls the same method), which reads ``inner.trellis``
BY ATTRIBUTE at call time and allocates its output on that tensor's device.
The sign vectors ``suh``/``svh`` (a few KB per expert) are pre-cast copies held
by the DiffLinear wrapper and stay in VRAM. So the only thing that has to
change is where ``inner.trellis`` points between uses: a host slab while
parked, a VRAM slot view while the layer is being computed.

Value-exact by construction: copies don't round, and the reconstruct kernel
and everything downstream are unchanged. A same-seed streamed run must produce
bit-identical losses to the resident run; that is the box gate.

Mechanics (CUDA):

  * ``park(inners)`` at load time, per MoE layer: one ``PinnedArena`` of
    exactly the layer's size (no power-of-two rounding -- torch's pinned
    allocator would turn 31 GB of experts into ~48 GB of locked pages) holds
    the layer's trellis tensors back to back, 256 B aligned. Each
    ``inner.trellis`` is rebound to its host slice and
    ``inner.get_inner_weight_tensor`` is wrapped so the first reconstruct of a
    layer calls ``ensure(layer)``. The device copies are released by the caller
    (``park_block`` drops the inference fast-path objects that also reference
    them; ``load_streaming`` measures that the VRAM actually came back).
  * ``ensure(layer)`` on a miss copies the layer's slab H2D into the least
    recently used of ``slots`` VRAM ring buffers (side stream, asynchronous
    from pinned memory) and rebinds every expert's ``trellis`` to a view of
    it; the compute stream waits on the copy's event before its first kernel.
    It then PREFETCHES the next ``slots - 1`` layers in the current traversal
    direction, so the copy of layer i+1 overlaps layer i's expert matmuls.
    Direction is inferred from the access order: layers climb in the forward
    (0, 1, 2, ...); the first repeat of the last layer, or any descent, marks
    the backward (under grad checkpointing a layer's recompute and its expert
    backward run back-to-back, so one residency serves both; without it the
    Functions' backward re-reconstructs through the same hooked method).
    Layer 0 always means "forward", so after a backward ends at layer 0 the
    ring already holds layers 0 and 1 for the next step: steady state has no
    synchronous copy at all.
  * Evicted layers are rebound to their HOST views, so a stale read fails
    loudly on the kernel's device check instead of reading another layer's
    bytes. The side stream waits on the compute stream before overwriting a
    slot, so a slot is never refilled while a kernel enqueued earlier still
    reads it.

Cost model (plan section 4): at training batch sizes top-10-of-512 over a few
hundred tokens touches nearly every expert, so each pass streams the whole
expert set (~31 GB at 2.05 bpw); with grad checkpointing that is 3 passes per
step (forward, recompute, backward; ``--dequant-cache`` makes it 2), a few
seconds per step over PCIe 4 x16, overlapped with the layer's compute.

On CPU (the unit tests) host and device coincide: copies are synchronous
``copy_`` calls and the same bookkeeping runs, so residency, direction,
eviction and the hook path are testable without a GPU.
"""

from __future__ import annotations
from typing import Optional
import collections
import torch

# Slab offset alignment in trellis ELEMENTS (int16): 256 B boundaries for
# every expert's view, so the reconstruct kernel's loads stay aligned.
_ALIGN = 128

# Routed-expert adapter target names (native_llama._MOE_EXPERT_TARGET_ALIASES);
# excluded on the streaming tier, see incompatible_flags.
_EXPERT_TARGETS = ("expert_gate_proj", "expert_up_proj", "expert_down_proj")


class _Layer:
    __slots__ = ("index", "inners", "views", "numel", "nbytes", "dtype", "host",
                 "arena", "orig_device", "slot", "hooks")

    def __init__(self, index: int):
        self.index = index
        self.inners = []        # LinearEXL3-like objects (have .trellis)
        self.views = []         # per inner: (offset, numel, shape) into the slab
        self.numel = 0          # slab length in elements (aligned)
        self.nbytes = 0
        self.dtype = None
        self.host = None        # 1-D host slab (pinned on CUDA)
        self.arena = None       # PinnedArena owning `host`, or None
        self.orig_device = None
        self.slot = None        # _Slot while resident
        self.hooks = []         # per inner: (had_own_attr, original attribute)


class _Slot:
    __slots__ = ("buf", "layer", "event", "waited")

    def __init__(self, buf, event):
        self.buf = buf          # 1-D device buffer, sized to the largest layer
        self.layer = None       # index of the layer it holds
        self.event = event      # CUDA event of the last fill (None on CPU)
        self.waited = set()     # streams that already waited on `event`


class _ReconstructHook:
    """Instance-level wrapper over ``inner.get_inner_weight_tensor``: makes the
    layer resident, then runs the original. Installed as an instance attribute
    so it shadows the class method for this inner only (``get_weight_tensor``
    reaches it through ``self`` too, covering the legacy dequant path)."""

    __slots__ = ("streamer", "layer", "orig")

    def __init__(self, streamer, layer: int, orig):
        self.streamer = streamer
        self.layer = layer
        self.orig = orig

    def __call__(self, *args, **kwargs):
        self.streamer.ensure(self.layer)
        return self.orig(*args, **kwargs)


class ExpertStreamer:
    """Residency manager for parked MoE layers (one ``park`` per layer, in
    forward order). ``ensure(i)`` is what the installed hooks call; everything
    else is setup / teardown / reporting."""

    def __init__(self, device, slots: int = 2, pin: Optional[bool] = None,
                 what: str = "expert streaming"):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if int(slots) < 2:
            raise ValueError(f"{what}: need at least 2 slots (current + prefetch), got {slots}")
        self.n_slots = int(slots)
        self.pin = self.cuda if pin is None else bool(pin)
        self.what = what
        self.layers: list[_Layer] = []
        self._slots: list[_Slot] = []
        self._slot_numel: Optional[int] = None     # fixed at the first ensure()
        self._lru: "collections.OrderedDict[int, None]" = collections.OrderedDict()
        self._last: Optional[int] = None
        self._dir = 1
        self._side = None
        self.stats = {"copies": 0, "misses": 0, "prefetches": 0, "hits": 0}

    # --- setup ------------------------------------------------------------

    def park(self, inners) -> int:
        """Move one layer's expert trellis tensors to a host slab, rebind each
        ``inner.trellis`` to its slice and install the reconstruct hook.
        Returns the layer index (its position in forward order). The caller
        drops any other references to the device copies."""
        inners = list(inners)
        if not inners:
            raise ValueError(f"{self.what}: park() needs at least one expert linear")
        if self._slot_numel is not None:
            raise RuntimeError(f"{self.what}: park() after the first ensure() -- park every "
                               f"layer before the first forward")
        layer = _Layer(len(self.layers))
        t0 = inners[0].trellis
        layer.dtype = t0.dtype
        layer.orig_device = t0.device
        off = 0
        for inner in inners:
            t = inner.trellis
            if t.dtype != layer.dtype or t.device != layer.orig_device:
                raise ValueError(f"{self.what}: layer {layer.index}: expert trellis tensors "
                                 f"differ in dtype/device ({t.dtype}/{t.device} vs "
                                 f"{layer.dtype}/{layer.orig_device})")
            n = t.numel()
            layer.views.append((off, n, tuple(t.shape)))
            off += (n + _ALIGN - 1) // _ALIGN * _ALIGN
        layer.numel = off
        layer.nbytes = off * t0.element_size()
        layer.host, layer.arena = self._alloc_host(layer)
        for inner, (o, n, shape) in zip(inners, layer.views):
            dst = layer.host[o:o + n].view(shape)
            dst.copy_(inner.trellis)                 # D2H, synchronous (load time)
            inner.trellis = dst
            had_own = "get_inner_weight_tensor" in getattr(inner, "__dict__", {})
            orig = inner.get_inner_weight_tensor
            inner.get_inner_weight_tensor = _ReconstructHook(self, layer.index, orig)
            layer.hooks.append((had_own, orig))
        layer.inners = inners
        self.layers.append(layer)
        return layer.index

    def _alloc_host(self, layer: _Layer):
        if self.pin:
            from ..util.pinned_arena import PinnedArena
            arena = PinnedArena(layer.nbytes, f"{self.what}: layer {layer.index} experts")
            return arena.tensor[:layer.nbytes].view(layer.dtype), arena
        return torch.empty(layer.numel, dtype=layer.dtype), None

    # --- residency --------------------------------------------------------

    def ensure(self, li: int) -> None:
        """Make layer ``li`` resident (synchronously if it was not prefetched),
        order the current stream after its copy, and prefetch the next layers
        in the traversal direction."""
        layer = self.layers[li]
        direction = self._direction(li)
        if layer.slot is None:
            self.stats["misses"] += 1
            self._fill(layer, self._victim((li,)))
        else:
            self.stats["hits"] += 1
        self._lru.move_to_end(li)
        self._wait(layer.slot)
        busy = {li}
        for k in range(1, self.n_slots):
            nxt = li + k * direction
            if not (0 <= nxt < len(self.layers)):
                break
            if self.layers[nxt].slot is None:
                slot = self._victim(busy)
                if slot is None:
                    break
                self._fill(self.layers[nxt], slot)
                self.stats["prefetches"] += 1
            busy.add(nxt)
        self._last = li
        self._dir = direction

    def _direction(self, li: int) -> int:
        n = len(self.layers)
        last = self._last
        if n == 1 or li == 0 or last is None or li > last:
            return 1
        if li < last:
            return -1
        # Same layer again. For the last layer that is the backward starting
        # (its forward touch was the previous call); elsewhere it is the next
        # expert projection of the same layer -- keep going.
        return -1 if li == n - 1 else self._dir

    def _victim(self, exclude) -> Optional[_Slot]:
        """A free slot: allocate while the ring is not full, else evict the
        least recently used resident layer not in ``exclude``."""
        if len(self._slots) < self.n_slots:
            return self._new_slot()
        for idx in self._lru:
            if idx not in exclude:
                slot = self.layers[idx].slot
                self._evict(idx)
                return slot
        return None

    def _new_slot(self) -> _Slot:
        if self._slot_numel is None:
            self._slot_numel = max(layer.numel for layer in self.layers)
        buf = torch.empty(self._slot_numel, dtype=self.layers[0].dtype, device=self.device)
        event = torch.cuda.Event() if self.cuda else None
        slot = _Slot(buf, event)
        self._slots.append(slot)
        return slot

    def _evict(self, idx: int) -> None:
        layer = self.layers[idx]
        for inner, (o, n, shape) in zip(layer.inners, layer.views):
            inner.trellis = layer.host[o:o + n].view(shape)     # loud on a stale read
        layer.slot.layer = None
        layer.slot = None
        del self._lru[idx]

    def _side_stream(self):
        if self._side is None:
            self._side = torch.cuda.Stream(device=self.device)
        return self._side

    def _fill(self, layer: _Layer, slot: _Slot) -> None:
        n = layer.numel
        if n > slot.buf.numel():
            raise RuntimeError(f"{self.what}: layer {layer.index} ({n} elements) exceeds the "
                               f"slot size ({slot.buf.numel()}); park every layer before the "
                               f"first forward")
        if self.cuda:
            cur = torch.cuda.current_stream(self.device)
            side = self._side_stream()
            # Every kernel already enqueued on the compute stream (the slot's
            # previous layer's reads) completes before the overwrite.
            side.wait_stream(cur)
            with torch.cuda.stream(side):
                slot.buf[:n].copy_(layer.host, non_blocking=True)
            slot.event.record(side)
            slot.waited = set()
        else:
            slot.buf[:n].copy_(layer.host)
        self.stats["copies"] += 1
        slot.layer = layer.index
        layer.slot = slot
        for inner, (o, m, shape) in zip(layer.inners, layer.views):
            inner.trellis = slot.buf[o:o + m].view(shape)
        self._lru[layer.index] = None

    def _wait(self, slot: _Slot) -> None:
        if not self.cuda:
            return
        cur = torch.cuda.current_stream(self.device)
        key = cur.cuda_stream
        if key not in slot.waited:
            cur.wait_event(slot.event)
            slot.waited.add(key)

    # --- teardown / reporting ---------------------------------------------

    def unpark_all(self) -> None:
        """Restore every parked layer: fresh device copies of the trellis
        tensors on their original device, hooks removed, slots and host
        slabs released. The inference fast-path objects the caller dropped
        are NOT rebuilt."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        for layer in self.layers:
            for inner, (o, n, shape), (had_own, orig) in zip(layer.inners, layer.views,
                                                             layer.hooks):
                inner.trellis = layer.host[o:o + n].view(shape).to(layer.orig_device, copy=True)
                if had_own:
                    inner.get_inner_weight_tensor = orig
                else:
                    del inner.get_inner_weight_tensor
            layer.slot = None
            layer.host = None
            if layer.arena is not None:
                layer.arena.close()
                layer.arena = None
        self.layers = []
        self._slots = []
        self._lru.clear()
        self._slot_numel = None
        self._last = None
        self._dir = 1

    @property
    def host_bytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    @property
    def slot_bytes(self) -> int:
        if not self.layers:
            return 0
        return max(layer.numel for layer in self.layers) * self.layers[0].host.element_size()

    def resident_layers(self) -> list[int]:
        return [i for i, layer in enumerate(self.layers) if layer.slot is not None]

    def is_resident_view(self, t: torch.Tensor) -> bool:
        """True when ``t`` views one of the VRAM slot buffers (test helper)."""
        p = t.data_ptr()
        for slot in self._slots:
            base = slot.buf.data_ptr()
            if base <= p < base + slot.buf.numel() * slot.buf.element_size():
                return True
        return False

    def describe(self) -> str:
        n = len(self.layers)
        per = (self.host_bytes / n / 2 ** 20) if n else 0.0
        # A PinnedArena that could not be page-locked falls back to pageable
        # memory (synchronous, slower copies) and says so once; report it here.
        pinned = bool(n) and all(layer.arena is not None and layer.arena.pinned
                                 for layer in self.layers)
        return (f"expert streaming: {n} MoE layers parked in "
                f"{'pinned' if pinned else 'pageable'} host RAM "
                f"({self.host_bytes / 2 ** 30:.1f} GiB, {per:.0f} MiB/layer), "
                f"{self.n_slots} VRAM slots x {self.slot_bytes / 2 ** 20:.0f} MiB on {self.device}")

    def stats_line(self) -> str:
        s = self.stats
        return (f"expert streaming: {s['copies']} layer copies "
                f"({s['prefetches']} prefetched, {s['misses']} synchronous), "
                f"{s['hits']} resident hits")


# --- exllamav3 glue ---------------------------------------------------------

def park_block(streamer: ExpertStreamer, mlp) -> int:
    """Park one ``BlockSparseMLP``'s routed experts and drop the inference
    fast-path objects that hold their own references to the device trellis
    tensors (``BC_LinearEXL3`` / ``BC_BlockSparseMLP`` keep ``at::Tensor``
    members; the ``MultiLinear`` / ``BatchReconLayer`` tables hold raw device
    addresses). They would keep the VRAM copies alive, and a stale address
    table would read freed memory; the training forward uses none of them."""
    linears = list(mlp.gates) + list(mlp.ups) + list(mlp.downs)
    bad = [l.key for l in linears if getattr(l, "quant_type", None) != "exl3"]
    if bad:
        raise RuntimeError(f"{streamer.what}: routed experts must be EXL3-quantized; "
                           f"{len(bad)} are not (e.g. {bad[0]})")
    inners = [l.inner for l in linears]
    li = streamer.park(inners)
    for inner in inners:
        inner.bc = None
        inner._fused_reconstruct = None
    for name in ("bc", "multi_gate", "multi_up", "multi_down", "batch_recon",
                 "fused_mode_buffers"):
        if hasattr(mlp, name):
            setattr(mlp, name, None)
    return li


def _moe_of(module):
    """The module's ``BlockSparseMLP`` when it is a decoder block with one."""
    from ..modules import BlockSparseMLP
    mlp = getattr(module, "mlp", None)
    return mlp if isinstance(mlp, BlockSparseMLP) else None


def estimate_expert_bytes(model) -> Optional[int]:
    """Bytes of routed-expert trellis the model will park, from the safetensors
    headers (before anything is loaded); None when a key can't be sized."""
    stc = model.config.stc
    total = 0
    try:
        for module in model.modules:
            mlp = _moe_of(module)
            if mlp is None:
                continue
            for lin in list(mlp.gates) + list(mlp.ups) + list(mlp.downs):
                key = f"{lin.key}.trellis"
                if not stc.has_tensor(key):
                    return None
                total += int(stc.get_tensor_size(key))
    except Exception:
        return None
    return total


def load_streaming(model, device, slots: int = 2, progressbar: bool = True,
                   pin: Optional[bool] = None) -> ExpertStreamer:
    """Single-device load of ``model`` (``Model.load(device=...)``'s protocol:
    deferred fills, prefer_cpu modules on CPU, shared scratch released at the
    end) that parks every MoE block's routed experts right after the block's
    tensors land, so the peak VRAM during loading is the dense trunk plus ONE
    layer's experts. MoE blocks load outside the deferred-load slab arena:
    their experts leave the device immediately, and slab slices would keep
    whole 128 MB blocks resident for the sake of the neighbouring small
    tensors. Returns the streamer (hooks installed, nothing resident yet)."""
    import gc
    from ..util.memory import free_mem, gc_paused, check_host_memory
    from ..util.progress import ProgressBar
    from ..util.tensor import g_tensor_cache

    device = torch.device(device)
    ip = model.config.infer_params
    if getattr(ip, "moe_cpu_offload", 0) or getattr(ip, "moe_cpu_split", 0):
        raise RuntimeError("expert streaming is the training path's own expert offload; "
                           "unset EXL3_MOE_CPU_OFFLOAD / EXL3_MOE_CPU_SPLIT (the inference "
                           "CPU-compute offload) to use it")
    est = estimate_expert_bytes(model)
    if est:
        # Fail before the load, not after 40 layers of it (PinnedArena checks
        # again per layer; the inference loader's reserve rules apply).
        check_host_memory(est, "expert streaming (routed experts in host RAM)")
    streamer = ExpertStreamer(device, slots=slots, pin=pin)
    stc = model.config.stc
    modules = model.modules
    warned = False
    freeze = gc.get_freeze_count() == 0
    if freeze:
        gc.freeze()
    try:
        with torch.inference_mode():
            with ProgressBar("Loading (streamed experts)" if progressbar else None,
                             len(modules)) as progress:
                for idx, module in enumerate(modules):
                    moe = _moe_of(module)
                    defer = module.can_defer_load()
                    with gc_paused():
                        if defer:
                            stc.begin_deferred_load(arena=moe is None)
                        module.load(torch.device("cpu") if module.caps.get("prefer_cpu")
                                    else device)
                        if defer:
                            stc.end_deferred_load()
                    if moe is not None:
                        before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                        li = park_block(streamer, moe)
                        if device.type == "cuda" and not warned:
                            freed = before - torch.cuda.memory_allocated(device)
                            want = streamer.layers[li].nbytes
                            if freed < want // 2:
                                print(f" !! expert streaming: parking layer {li} freed "
                                      f"{freed >> 20} MiB of VRAM, expected ~{want >> 20} MiB "
                                      f"-- something still references the device copies of "
                                      f"its experts; the run will not fit the planned budget",
                                      flush=True)
                                warned = True
                    progress.update(idx + 1)
            model.output_device = modules[-1].device
            free_mem()
            g_tensor_cache.drop_all()
            for ref in model.cache_weakrefs.values():
                cache = ref()
                if cache is not None:
                    cache.initialized = True
    finally:
        if freeze:
            gc.unfreeze()
    return streamer


def incompatible_flags(parallel: str, vram_spillover: bool, sample_every: int,
                       targets, expert_r) -> list[str]:
    """Trainer flags that cannot be combined with ``--stream-experts``, as
    user-facing reasons (empty when the combination is fine)."""
    problems = []
    if parallel != "single":
        problems.append(
            "--parallel split: streaming keeps the dense trunk on ONE device and feeds "
            "every MoE layer from host RAM, so one card is the point; use --parallel single")
    if vram_spillover:
        problems.append(
            "--vram-spillover: managed-memory allocations would silently page the streamed "
            "slots; streaming already does the explicit host<->device traffic")
    if sample_every:
        problems.append(
            "--sample-every N: live samples run the INFERENCE forward, whose fused MoE "
            "kernels read device-resident expert tables that parked experts no longer have. "
            "Pass --sample-every 0 and sample with the saved adapter afterwards")
    hit = sorted(set(targets or ()) & set(_EXPERT_TARGETS))
    if hit:
        problems.append(
            f"routed-expert adapters ({', '.join(hit)}): per-expert LoRA on 512 experts x 48 "
            f"layers is ~17 GB of fp32 adapter state at r=16, which is the whole card; the "
            f"streaming tier adapts attention / GDN / shared-expert projections only")
    if expert_r is not None:
        problems.append("--expert-r: no routed-expert adapters on the streaming tier")
    return problems
