"""
KV cache streaming (see doc/kv_streaming.md).

A streaming cache layer keeps the authoritative K/V history in pinned host memory (the mirror,
indexed by *pool* page, exactly like the regular paged cache tensors) and only a bounded GPU
working set on the device: a resident window of `R` pages (the most recent token pages of the
active sequence, addressed by token page index `i -> i % R`) and a shared ring of `S` chunks of
`C` pages for the older history. Attention consumes the history in chunks: the patched
paged-attention kernels take a KV span and a partial-slot range, so a chunk walk chains the
stock flash-decoding split kernel (or the prefill kernel) across placements and reduces
everything in one final combine, in KV order.

Placements are a pure function of the token page index:

    resident window:  pages [max(0, n_pages - R), n_pages)  ->  slot i % R
    streamed:         pages [0, max(0, n_pages - R))        ->  ring slot (i // C) % S * C + i % C

so the kernel table (token page index -> physical slot) does not depend on the block table at
all and is only rebuilt when the live page count changes. Writes always land in the resident
window (the driver sub-chunks appends so that the pages an append touches fit the window), are
published to the mirror right after the write, and reads of streamed pages come from the mirror
through the ring.

The layer keeps the stock attention interface out of the loop: `decode_flash_attn` calls
`stream_attn()` when the resolved cache layer is a streaming one, and any other caller that
reaches a streaming layer through `attn_dispatch` gets a loud error instead of a wrong result.

Supported: CUDA, bsz 1, causal, stock paged attention (full-attention and left-sliding-window
layers, logit softcapping, asymmetric V), fp16 and quantized caches. Not supported: right-hand
windows (window_right > 0), MLA/QSA geometries, TP, CUDA graphs over the chunk walk, CPU/disk
page tiers (they raise).

Environment (doc/env_vars.md):

    EXL3_KV_STREAM            0       enable streaming cache layers
    EXL3_KV_STREAM_MIRROR     auto    host mirror budget: reserved, not enforced (mirror is
                                      allocated for the full logical capacity; setting a smaller
                                      value warns)
    EXL3_KV_STREAM_RESIDENT   auto    resident window size: pages ("12") or bytes ("1G"), per
                                      full-attention layer. "auto" fills the free device memory at
                                      first use, keeping a reserve for the ring, the walk partials
                                      and workspace headroom
    EXL3_KV_STREAM_RING       auto    ring size in pages, shared across layers (default 64)
    EXL3_KV_STREAM_PREFETCH   2       ring slots in flight (chunks read ahead)
    EXL3_KV_STREAM_SERIAL     0       serialized debug mode: one stream, no overlap
    EXL3_KV_STREAM_TRACE      0       log chunk/slot/event transitions
"""

from __future__ import annotations

import math
import os
import re
import warnings

import torch
from typing_extensions import override

from ..constants import PAGE_SIZE
from .cache import CacheLayer
from .fp16 import CacheLayer_fp16
from .quant import CacheLayer_quant

# Partial-buffer budget per step (partial_o + partial_ml), matches the spirit of the stock prefill
# wrapper's 128 MB cap
_PARTIAL_BUDGET = 64 << 20

# -------------------------------------------------------------------------------------------------
# Environment

def stream_enabled() -> bool:
    return os.environ.get("EXL3_KV_STREAM", "0") == "1"


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name}: expected an integer, got {v!r}")


def _parse_bytes(name: str, s: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTkmgt]?)([Bb])?\s*", s)
    if m is None:
        raise ValueError(f"{name}: expected a byte size like '8G' or '512M', got {s!r}")
    mult = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


# Resolved per-cache settings (one placement policy per cache, shared by its stream layers)
_STREAM_CONF: dict = {}
_warned: set = set()


def _stream_config(cache_id, layer, capacity_pages: int) -> dict:
    cfg = _STREAM_CONF.get(cache_id)
    if cfg is not None:
        # One call per stream layer of this cache: the auto placement divides the free device
        # memory between the layers sharing it, so the count must be complete (every layer object
        # exists once the model is constructed) before the first step resolves the window
        cfg["n_layers"] += 1
        return cfg

    per_page = layer.stream_per_page_bytes()

    r_env = os.environ.get("EXL3_KV_STREAM_RESIDENT", "").strip()
    if r_env == "" or r_env.lower() == "auto":
        R = None                    # resolved at first use from the free device memory
    elif re.fullmatch(r"\d+\s*[pP]?", r_env):
        R = int(re.match(r"\d+", r_env).group(0))
    else:
        R = max(1, _parse_bytes("EXL3_KV_STREAM_RESIDENT", r_env) // per_page)
    if R is not None:
        R = max(1, min(R, capacity_pages))

    ring_pages = _env_int("EXL3_KV_STREAM_RING", 64)
    if ring_pages < 1:
        raise ValueError("EXL3_KV_STREAM_RING must be at least one page")
    prefetch = max(1, _env_int("EXL3_KV_STREAM_PREFETCH", 2))
    S = min(prefetch, ring_pages)
    C = max(1, ring_pages // S)

    mirror_env = os.environ.get("EXL3_KV_STREAM_MIRROR", "").strip()
    if mirror_env and mirror_env.lower() != "auto":
        budget = _parse_bytes("EXL3_KV_STREAM_MIRROR", mirror_env)
        full = capacity_pages * per_page
        if budget < full and "mirror" not in _warned:
            _warned.add("mirror")
            warnings.warn(
                f"EXL3_KV_STREAM_MIRROR={mirror_env} ({budget} bytes) is below the full mirror "
                f"({full} bytes): the mirror is allocated at full logical capacity and the "
                f"budget is not enforced"
            )

    serial = os.environ.get("EXL3_KV_STREAM_SERIAL", "0") == "1"
    trace = os.environ.get("EXL3_KV_STREAM_TRACE", "0") == "1"

    cfg = dict(R = R, ring_pages = ring_pages, prefetch = prefetch, S = S, C = C,
               serial = serial, trace = trace, per_page = per_page, n_layers = 1)
    _STREAM_CONF[cache_id] = cfg
    if trace:
        r_str = "auto" if R is None else str(R)
        print(f" [stream] cache {cache_id}: resident {r_str} pages, ring {ring_pages} pages "
              f"(S={S}, C={C}), prefetch {prefetch}, serial {int(serial)}")
    return cfg


# Shared ring registry: the ring is one GPU buffer per (device, component shapes) reused by every
# stream layer, with the ready/consumed events that order its slots across layers
_RING_REGISTRY: dict = {}


def _ring_entry(device: torch.device, spec: tuple, n_slots: int) -> dict:
    dev_key = device.index if device.index is not None else str(device)
    key = (dev_key, spec, n_slots)
    entry = _RING_REGISTRY.get(key)
    if entry is None:
        tensors = {name: torch.zeros(shape, dtype = dtype, device = device)
                   for name, shape, dtype in spec}
        # The zero-fill above is queued on the stream of whichever layer creates the ring, while
        # the first H2D copies into the ring run on the layers' copy streams: nothing orders the
        # two, and a fill that lands late zeroes the slots the first copies just filled. Record
        # the fill so every layer can order its copy stream after it before the first copy
        filled = torch.cuda.Event()
        filled.record()
        entry = dict(tensors = tensors, filled = filled,
                     used = [False] * n_slots,
                     ready = [torch.cuda.Event() for _ in range(n_slots)],
                     consumed = [torch.cuda.Event() for _ in range(n_slots)])
        _RING_REGISTRY[key] = entry
    return entry


_SM_COUNTS: dict = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index
    if idx not in _SM_COUNTS:
        _SM_COUNTS[idx] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SM_COUNTS[idx]


# -------------------------------------------------------------------------------------------------
# Layer mapping

def stream_cache_layer_type(attn, default, kwargs: dict):
    """
    Map a requested cache layer type to its streaming variant, or return None to leave it alone.
    Only exact fp16/quant layers stream (geometry-specific layers keep their own paths), and only
    attention modules whose decode path is the stock `Attention.decode_flash_attn` with plain
    geometry (no right-hand window, no cu_seqlens batching) qualify. Left sliding windows, logit
    softcapping and asymmetric V are handled by the chunked driver.
    """
    if not stream_enabled():
        return None
    from ..modules.attn import Attention
    if type(attn).decode_flash_attn is not Attention.decode_flash_attn:
        return None
    if getattr(attn, "has_split_cache", False):
        _warn_once("tp", "KV streaming is not supported in tensor-parallel mode")
        return None
    if attn.window_right > 0:
        _warn_once("window_right", "KV streaming does not cover right-hand windows; "
                                   "those layers keep their regular cache")
        return None
    if attn.use_cu_seqlens:
        _warn_once("cu_seqlens", "KV streaming does not cover cu_seqlens batching; "
                                 "those layers keep their regular cache")
        return None
    if default is CacheLayer_quant:
        if attn.head_dim % 32 != 0:
            _warn_once("quant_dim", "KV streaming needs head_dim to be a multiple of 32 on "
                                    "quantized caches")
            return None
        return CacheLayer_stream_quant, dict(kwargs)
    if default is CacheLayer_fp16:
        if attn.head_dim % 8 != 0:
            _warn_once("fp16_dim", "KV streaming needs head_dim to be a multiple of 8")
            return None
        return CacheLayer_stream, dict(kwargs)
    return None


def _warn_once(key: str, msg: str):
    if key not in _warned:
        _warned.add(key)
        warnings.warn(msg)


# -------------------------------------------------------------------------------------------------
# Shared streaming machinery

class StreamLayer(CacheLayer):
    """
    Base for the streaming cache layers. Holds the mirror/resident/ring machinery, the kernel
    table and the chunked attention driver; the subclasses supply the resident and mirror tensors
    for their storage format.
    """

    # Component names, in write order: ("k", "v") or ("qk", "sk", "qv", "sv")
    COMPONENTS: tuple = ()

    def __init__(self, config, attention, cache_id, max_num_tokens, **kwargs):
        super().__init__(config, attention, cache_id, max_num_tokens, **kwargs)
        assert attention is not None, "streaming cache layers need an attention module"
        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}"
        self.num_q_heads = attention.num_q_heads
        self.num_kv_heads = attention.num_kv_heads
        self.head_dim = attention.head_dim
        self.capacity_pages = max_num_tokens // PAGE_SIZE
        self.cfg = _stream_config(cache_id, self, self.capacity_pages)

        # Driver state (allocated in alloc())
        self.device = None
        self.mirror = None
        self._resident_ready = False
        self._bc_stream = None
        self._bt_identity = None
        self._valid_from = None
        self._kt = None
        self._kt_host = None
        self._pub_rows = None
        self._stale_pages = None
        self._table_key = None
        self._epoch = 0
        self._slot_owner = None
        self._tok2pool = None
        self._wrote_upto = -1
        self._pub_event = None
        self._ev_mark = None
        self._ring = None
        self._geo_dec = None
        self._geo_pre = None
        self._part_dec = None
        self._part_pre = None
        self._copy_stream = None
        self._stage = None

    # ---- layout hooks (subclasses) ----

    def stream_per_page_bytes(self) -> int:
        """Bytes the mirror/resident storage uses per page, for byte-valued budgets"""
        return 0

    def _alloc_resident(self, device: torch.device):
        raise NotImplementedError

    def _free_resident(self):
        raise NotImplementedError

    def _alloc_mirror(self, device: torch.device):
        raise NotImplementedError

    def _resident_tensors(self) -> dict:
        raise NotImplementedError

    def _write_rows(self, k, v, a: int, length: int):
        raise NotImplementedError

    # ---- allocation ----

    @override
    def alloc(self, device: torch.device):
        self.device = device
        self._resident_ready = False
        self._bc_stream = None
        self._bt_identity = None
        self._valid_from = None
        R = self.cfg["R"]
        if R is not None:
            self._alloc_resident(device)
            self._slot_owner = [None] * R
            self._resident_ready = True
        else:
            # Auto placement: sized from the free device memory at first use (all the model's
            # layers exist by then, and the weights are on the device)
            self._slot_owner = None
        self._alloc_mirror(device)
        self._kt_host = torch.zeros((self.capacity_pages,), dtype = torch.int32)
        self._kt = torch.zeros((1, self.capacity_pages), dtype = torch.int32, device = device)
        self._pub_rows = [0] * self.capacity_pages
        self._stale_pages = set()
        self._table_key = None
        self._epoch = 0
        self._tok2pool = []
        self._wrote_upto = -1
        self._pub_event = None
        self._ev_mark = None
        self._ring = None
        self._geo_dec = {}
        self._geo_pre = {}
        self._part_dec = None
        self._part_pre = None
        self._stage = torch.zeros((1,), dtype = torch.int32, device = device)
        self._copy_stream = None if self.cfg["serial"] else torch.cuda.Stream(device)

    @override
    def free(self):
        self.device = None
        self.mirror = None
        self._bc_stream = None
        self._bt_identity = None
        self._resident_ready = False
        self._free_resident()
        self._kt = self._kt_host = None
        self._pub_rows = None
        self._stale_pages = None
        self._slot_owner = None
        self._tok2pool = None
        self._pub_event = None
        self._ev_mark = None
        self._ring = None
        self._geo_dec = self._geo_pre = None
        self._part_dec = self._part_pre = None
        self._copy_stream = None
        self._stage = None

    def _auto_resident_pages(self) -> int:
        """Resident pages that fit the free device memory: everything else about the layer is
        bounded (ring, walk partials) or transient, so the window takes what is left after a
        reserve for those."""
        per_page = self.cfg["per_page"]
        n_layers = max(1, self.cfg.get("n_layers", 1))
        free, _ = torch.cuda.mem_get_info(self.device)
        ring = self.cfg["S"] * self.cfg["C"] * per_page
        # The decode and prefill walks each keep partial buffers bounded by _PARTIAL_BUDGET per
        # layer; cap the reserve so models with huge per-page KV cannot starve the window
        partials = min(2 * n_layers * _PARTIAL_BUDGET, free // 4)
        slack = max(512 << 20, free // 16)
        budget = max(0, free - ring - partials - slack)
        pages = budget // max(1, per_page * n_layers)
        return int(max(1, min(self.capacity_pages, pages)))

    def _ensure_resident(self):
        """Allocate the resident window on first use (auto placement resolves its size here)"""
        if self._resident_ready:
            return
        assert self.device is not None, \
            "streaming cache layer used before the model was loaded (alloc was not called)"
        if self.cfg["R"] is None:
            self.cfg["R"] = self._auto_resident_pages()
            if self.cfg["trace"]:
                free, _ = torch.cuda.mem_get_info(self.device)
                print(f" [stream]{self.attention.key if self.attention is not None else ''} "
                      f"auto resident: {self.cfg['R']} pages "
                      f"({self.cfg['R'] * self.cfg['per_page'] / 2**20:.0f} MiB/layer, "
                      f"free {free / 2**30:.1f} GiB, {self.cfg.get('n_layers', 1)} layers)")
        self._alloc_resident(self.device)
        self._slot_owner = [None] * self.cfg["R"]
        self._valid_from = None
        self._resident_ready = True

    # ---- interface surfaces that streaming replaces ----

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1) -> tuple:
        raise RuntimeError(
            f"{type(self).__name__}: streaming layers do not expose flat K/V tensors; "
            f"attention must go through the chunked driver")

    @override
    def update_kv(self, cache_seqlens, block_table, k, v, length):
        raise RuntimeError(
            f"{type(self).__name__}: streaming layers write through the chunked driver")

    @override
    def update_kv_direct(self, cache_seqlens, block_table, k, v, length):
        raise RuntimeError(
            f"{type(self).__name__}: streaming layers write through the chunked driver")

    @override
    def get_tensors(self):
        if self._resident_ready:
            return list(self._resident_tensors().values())
        return []

    @override
    def overhead_size(self):
        return 0

    @override
    def tp_export(self, plan):
        raise NotImplementedError("KV streaming does not support tensor parallelism")

    def _ensure_ring(self) -> dict:
        if self._ring is None:
            spec = []
            for name, t in self._resident_tensors().items():
                spec.append((name, (self.cfg["S"] * self.cfg["C"],) + tuple(t.shape[1:]), t.dtype))
            self._ring = _ring_entry(self.device, tuple(spec), self.cfg["S"])
            cs = self._copy_stream
            if cs is not None:
                # Order this layer's copy stream after the ring's creation fill (queued on the
                # creating layer's stream, or this one): the first copies must not race it
                cs.wait_event(self._ring["filled"])
        return self._ring

    # ---- kernel table and resident window ----

    def _trace(self, msg: str):
        if self.cfg["trace"]:
            print(f" [stream]{self.attention.key if self.attention is not None else ''} {msg}")

    def _ensure_state(self, host_bt, a: int, L: int):
        """
        Rebuild the kernel table if the live page count changed and refill the resident window
        slots whose contents are not known to match the block table's pages. `a` is the append
        start (host value), `L = a + chunk` the sequence length this sub-chunk serves.
        """
        self._ensure_resident()
        R = self.cfg["R"]
        n_pages = -(-L // PAGE_SIZE)
        window_start = max(0, n_pages - R)
        host_bt_row = host_bt[0] if host_bt.dim() == 2 else host_bt
        pool = [int(v) for v in host_bt_row[:n_pages]]
        self._tok2pool = pool

        table_key = (n_pages, window_start)
        if table_key != self._table_key:
            self._table_key = table_key
            C, S = self.cfg["C"], self.cfg["S"]
            vals = [0] * n_pages
            for i in range(window_start):
                vals[i] = (i // C) % S * C + i % C
            for i in range(window_start, n_pages):
                vals[i] = i % R
            self._kt_host[:n_pages].copy_(torch.tensor(vals, dtype = torch.int32))
            # The device table is generated on-device: a host-staged H2D would race with the next
            # table rewrite while the copy is still in flight (the CPU runs ahead of the stream)
            idx = torch.arange(n_pages, dtype = torch.int32, device = self.device)
            vals_dev = torch.where(idx < window_start,
                                   (idx // C) % S * C + idx % C,
                                   idx % R)
            self._kt[0, :n_pages].copy_(vals_dev)

        if self._valid_from is not None and self._valid_from <= window_start:
            # Every slot in the window was written (or refilled) by this sequence since the last
            # invalidation and nothing stale came in: no per-page verification needed
            self._trace(f"table: {n_pages} pages, window [{window_start}, {n_pages}), window valid")
            return

        write_page0 = a // PAGE_SIZE
        stale = self._stale_pages
        n_refill = 0
        for i in range(window_start, n_pages):
            p = pool[i]
            if i < write_page0:
                rows = min(self._pub_rows[p], PAGE_SIZE)
            elif i == write_page0:
                rows = max(0, min(self._pub_rows[p], a - i * PAGE_SIZE))
            else:
                rows = 0
            if rows <= 0:
                continue
            if self._slot_owner[i % R] == (self._epoch, i, p) and p not in stale:
                continue
            self._copy_mirror_to_slot(p, i % R, rows)
            self._slot_owner[i % R] = (self._epoch, i, p)
            n_refill += 1
        if stale:
            # The window's pages were verified or refilled above, so their entries can leave the
            # stale set; only pages outside the window (other cached sequences) stay
            stale.difference_update(pool[window_start:])
            for p in list(stale):
                if p not in pool:
                    stale.discard(p)
        self._valid_from = window_start
        self._trace(f"table: {n_pages} pages, window [{window_start}, {n_pages}), "
                    f"{n_refill} refill(s), epoch={self._epoch}")

    def _copy_mirror_to_slot(self, pool_page: int, slot: int, rows: int):
        """Refill a resident slot from the mirror. Runs on the compute stream, which the caller has already
        ordered after the publishes with _wait_publish() (also in the overlapped mode)"""
        for name, resident in self._resident_tensors().items():
            if rows > 0:
                resident[slot, :rows].copy_(self.mirror[name][pool_page, :rows], non_blocking = True)

    def _copy_page_to_ring(self, token_page: int, slot: int):
        ring = self._ring["tensors"]
        pool_page = self._tok2pool[token_page]
        for name, t in ring.items():
            t[slot].copy_(self.mirror[name][pool_page], non_blocking = True)

    def _runs(self, p0: int, p1: int):
        """
        Split token pages [p0, p1) into runs whose mirror pages and destination slots are both
        contiguous: one copy per run per component instead of one per page.
        """
        runs = []
        p = p0
        while p < p1:
            pool0 = self._tok2pool[p]
            slot0 = self._table_slot(p)
            q = p + 1
            while (q < p1 and self._tok2pool[q] == pool0 + (q - p) and
                   self._table_slot(q) == slot0 + (q - p)):
                q += 1
            runs.append((p, q, pool0, slot0))
            p = q
        return runs

    def _copy_pages_to_ring(self, p0: int, p1: int):
        """H2D pages [p0, p1) into their ring slots, one copy per contiguous run"""
        ring = self._ring["tensors"]
        for a, b, pool0, slot0 in self._runs(p0, p1):
            n = b - a
            for name, t in ring.items():
                t[slot0:slot0 + n].copy_(self.mirror[name][pool0:pool0 + n], non_blocking = True)

    # ---- publish ----

    def _quiesce(self):
        """Wait for every scheduled copy (mirror must be stable for host-side page surgery)"""
        if self.device is None:
            return
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        torch.cuda.synchronize(self.device)

    def _wait_publish(self):
        """Order the compute stream after the last D2H publish (mirror reads and slot reuse)"""
        if self._pub_event is not None:
            torch.cuda.current_stream().wait_event(self._pub_event)

    def _publish_rows(self, a: int, b: int):
        """
        D2H rows [a, b) of the just-written pages to the mirror (current stream). Full pages
        coalesce into contiguous runs; the partial head and tail pages copy on their own.
        """
        p0 = a // PAGE_SIZE
        p1 = (b - 1) // PAGE_SIZE
        for p in range(p0, p1 + 1):
            r1 = min(b - p * PAGE_SIZE, PAGE_SIZE)
            self._pub_rows[self._tok2pool[p]] = max(self._pub_rows[self._tok2pool[p]], r1)
        parts = []
        if p0 == p1:
            parts.append((p0, p0 + 1, a - p0 * PAGE_SIZE, b - p0 * PAGE_SIZE))
        else:
            parts.append((p0, p0 + 1, a - p0 * PAGE_SIZE, PAGE_SIZE))
            if p1 > p0 + 1:
                parts.append((p0 + 1, p1, 0, PAGE_SIZE))
            parts.append((p1, p1 + 1, 0, b - p1 * PAGE_SIZE))
        resident = self._resident_tensors()
        for q0, q1, r0, r1 in parts:
            for s0, s1, pool0, slot0 in self._runs(q0, q1):
                n = s1 - s0
                if r0 == 0 and r1 == PAGE_SIZE:
                    for name, t in resident.items():
                        self.mirror[name][pool0:pool0 + n].copy_(
                            t[slot0:slot0 + n], non_blocking = True)
                else:
                    for p in range(s0, s1):
                        for name, t in resident.items():
                            self.mirror[name][self._tok2pool[p], r0:r1].copy_(
                                t[self._table_slot(p), r0:r1], non_blocking = True)

    def _account_rollback(self, a: int):
        """Rows in [a, previous end) were rejected: forget them in the publication ledger"""
        if a >= self._wrote_upto:
            return
        for p in range(a // PAGE_SIZE, (self._wrote_upto - 1) // PAGE_SIZE + 1):
            self._pub_rows[p] = min(self._pub_rows[p], max(0, a - p * PAGE_SIZE))
        self._trace(f"rollback to {a}: publication clamped (was {self._wrote_upto})")

    def _publish_append(self, a: int, b: int):
        """Record ownership of the pages a step just wrote and publish their rows to the mirror"""
        for p in range(a // PAGE_SIZE, (b - 1) // PAGE_SIZE + 1):
            slot = self._table_slot(p)
            self._slot_owner[slot] = (self._epoch, p, self._tok2pool[p])
        if self._copy_stream is None:
            self._publish_rows(a, b)
            self._record_publish_event()
        else:
            # Reuse one marker event per layer: a wait snapshots the record that is current when it
            # is enqueued, so re-recording it on a later step cannot affect earlier waits. One
            # fewer event object per publish (this runs once per layer per step)
            ev = self._ev_mark
            if ev is None:
                ev = self._ev_mark = torch.cuda.Event()
            ev.record()
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(ev)
                self._publish_rows(a, b)
                self._record_publish_event()
        # Plain assignment: a new sequence can start below the previous one's end (the epoch bump
        # in the driver already invalidated every slot owner). Keeping a stale maximum here would
        # make the position mismatch persist for the whole sequence and bump the epoch on every
        # call, which refills the entire resident window every step
        self._wrote_upto = b

    def _write_append(self, k, v, a: int, length: int):
        """Write rows [a, a + length) into the resident window and publish them"""
        assert length > 0
        self._stage.fill_(a)
        self._write_rows(k, v, a, length)
        self._publish_append(a, a + length)

    def _record_publish_event(self):
        if self._copy_stream is not None:
            if self._pub_event is None:
                self._pub_event = torch.cuda.Event()
            self._pub_event.record(self._copy_stream)

    def _table_slot(self, token_page: int) -> int:
        return int(self._kt_host[token_page].item())

    # ---- chunk walk ----

    def _decode_splits(self, kv_len: int, programs: int, block_n: int) -> int:
        if kv_len <= 0:
            return 1
        import triton
        target = max(1, 2 * _sm_count(self.device))
        by_programs = max(1, target // max(1, programs))
        return int(max(1, min(by_programs, triton.cdiv(kv_len, max(1, block_n)), 128)))

    def _prefill_splits(self, kv_len: int, programs: int, block_n: int) -> int:
        if kv_len < 8192 or programs >= 2 * _sm_count(self.device):
            return 1
        import triton
        by_programs = max(1, (2 * _sm_count(self.device)) // max(1, programs))
        return int(min(8, by_programs, max(1, triton.cdiv(kv_len, max(1, block_n)))))

    def _plan_chunks(self, L: int, window_start: int, kv_lo: int, programs: int, block_n: int,
                     splits_fn, bytes_per_slot: int):
        """Launch plan: streamed chunks in ascending KV order, then the resident window. Chunks
        entirely below `kv_lo` (the earliest key any query row of this step can see) are skipped,
        including their H2D copies. The per-chunk split counts are capped so the step's partial
        buffers stay within budget"""
        C = self.cfg["C"]
        n_chunks = -(-window_start // C) if window_start else 0
        n_launches = n_chunks + (1 if L > window_start * PAGE_SIZE else 0)
        slots_cap = max(1, _PARTIAL_BUDGET // max(1, bytes_per_slot))
        per_launch_cap = max(1, -(-slots_cap // max(1, n_launches)))
        plan = []
        for ci in range(n_chunks):
            p0 = ci * C
            p1 = min(p0 + C, window_start)
            nb = p0 * PAGE_SIZE
            ne = min(p1 * PAGE_SIZE, L)
            if ne <= nb or ne <= kv_lo:
                continue
            splits = min(splits_fn(ne - nb, programs, block_n), per_launch_cap)
            plan.append((ci, nb, ne, splits))
        rb = max(window_start * PAGE_SIZE, kv_lo)
        if L > rb:
            splits = min(splits_fn(L - rb, programs, block_n), per_launch_cap)
            plan.append((None, rb, L, splits))
        total_slots = sum(e[3] for e in plan)
        final = len(plan) == 1 and plan[0][0] is None and plan[0][3] == 1
        return n_chunks, plan, total_slots, final

    def _issue_chunk_h2d(self, ci: int, window_start: int):
        """Schedule the H2D of streamed chunk `ci` into its ring slot"""
        C, S = self.cfg["C"], self.cfg["S"]
        s0 = ci * C
        s1 = min(s0 + C, window_start)
        self._ensure_ring()
        if self._copy_stream is None:
            self._copy_pages_to_ring(s0, s1)
            return
        ring = self._ensure_ring()
        cs = self._copy_stream
        if ring["used"][ci % S]:
            cs.wait_event(ring["consumed"][ci % S])
        with torch.cuda.stream(cs):
            self._copy_pages_to_ring(s0, s1)
            ring["ready"][ci % S].record()
        ring["used"][ci % S] = True

    def _wait_chunk_ready(self, ci: int):
        if self._copy_stream is None:
            return
        ring = self._ensure_ring()
        torch.cuda.current_stream().wait_event(ring["ready"][ci % self.cfg["S"]])

    def _record_chunk_consumed(self, ci: int):
        if self._copy_stream is None:
            return
        ring = self._ensure_ring()
        ring["consumed"][ci % self.cfg["S"]].record()

    def _split_len(self, kv_len: int, splits: int, geo) -> int:
        import triton
        block_n = getattr(geo, "block_n", None) or (geo.candidates[0] if geo.candidates else 64)
        return int(triton.cdiv(triton.cdiv(kv_len, splits), block_n) * block_n)

    # ---- public driver ----

    @torch.no_grad
    def stream_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        host_seqlens: torch.Tensor | None = None,
        host_block_table: torch.Tensor | None = None,
        causal: bool = True,
        sm_scale: float | None = None,
        sinks: torch.Tensor | None = None,
        non_causal_spans: list | None = None,
        window_left: int = -1,
        window_right: int = -1,
        softcap: float = 0.0,
    ) -> torch.Tensor:
        """
        Chunked attention over the streamed history. Writes rows [a, a + q_len), walks the KV
        history in placement chunks with the patched split/prefill kernels, and reduces the
        partials in KV order.

        `window_left` < 0 means full attention. A left sliding window only needs the keys
        [position - window_left, position], so chunks entirely below the earliest row's bound are
        skipped (no H2D copies); the kernels mask the partial boundary chunk. `softcap` is the
        logit softcapping value (0.0: none).
        """
        assert self.device is not None, "streaming cache layer is not allocated (model not loaded?)"
        assert non_causal_spans is None, "KV streaming does not support non-causal spans"
        assert causal, "KV streaming does not support non-causal attention"
        if window_left is None or window_left < 0:
            window_left, window_right = -1, -1
        elif window_right is None:
            window_right = 0
        softcap = float(softcap or 0.0)
        self._ensure_resident()
        bsz, q_len, n_q_heads, head_dim = q.shape
        assert bsz == 1, "KV streaming supports bsz == 1"
        assert n_q_heads == self.num_q_heads and head_dim == self.head_dim
        assert k.shape == v.shape and k.shape[0] == 1 and k.shape[1] == q_len
        assert k.shape[2] == self.num_kv_heads
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert q.device == self.device

        if host_seqlens is not None:
            a = int(host_seqlens[0])
        else:
            a = int(cache_seqlens[0].item())
        if host_block_table is None:
            host_block_table = block_table if block_table.device.type == "cpu" else block_table.cpu()

        assert a >= 0 and q_len > 0
        L = a + q_len
        assert L <= self.capacity_pages * PAGE_SIZE, \
            f"KV streaming: sequence length {L} exceeds the cache capacity " \
            f"{self.capacity_pages * PAGE_SIZE}"

        # A step that does not continue this cache's last published position is a new sequence
        # (new job, prompt-cache prefix, rollback): every resident slot is suspect
        self._account_rollback(a)
        if a != self._wrote_upto:
            self._epoch += 1
            self._slot_owner = [None] * self.cfg["R"]
            self._valid_from = None
            self._trace(f"epoch {self._epoch} at {a} (last published {self._wrote_upto})")

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(self.head_dim)
        nkv_override = self.num_kv_heads if self._quant else None

        out = torch.empty((bsz, q_len, n_q_heads, head_dim), dtype = torch.half, device = self.device)

        R = self.cfg["R"]
        pos, s0 = a, 0
        while s0 < q_len:
            take = min(q_len - s0, max(1, R * PAGE_SIZE - (pos % PAGE_SIZE)))
            s1 = s0 + take
            b_s = pos + take
            self._wait_publish()
            self._ensure_state(host_block_table, pos, b_s)
            self._write_append(k[:, s0:s1], v[:, s0:s1], pos, take)
            window_start = max(0, -(-b_s // PAGE_SIZE) - R)
            # Earliest key any query row of this step can see (left window): chunks entirely
            # below it carry no weight and are not copied or launched
            kv_lo = max(0, pos - window_left) if window_left >= 0 else 0
            self._stage.fill_(b_s)
            if take <= 16:
                self._walk_decode(q[:, s0:s1], out[:, s0:s1], b_s, window_start, kv_lo, sm_scale,
                                  sinks, nkv_override, window_left, window_right, softcap)
            else:
                self._walk_prefill(q[:, s0:s1], out[:, s0:s1], b_s, window_start, kv_lo, sm_scale,
                                   sinks, nkv_override, window_left, window_right, softcap)
            pos, s0 = b_s, s1
        return out

    def _identity_table(self):
        """Block table the graph reads over the resident window: slot == token page"""
        if self._bt_identity is None:
            self._bt_identity = torch.arange(self.cfg["R"], dtype = torch.int32,
                                             device = self.device).unsqueeze(0)
        return self._bt_identity

    @torch.no_grad
    def bc_attn_step(self, attn, x, params, block_table, cache_seqlens, host_seqlens,
                     position, positions, position_ids, inv_freq):
        """
        All-resident steps run the stock graph path. While the whole live history fits the
        resident window (so slot == token page for every page) the window is a plain page-major
        pool, and the stock BC graph reads and appends through the identity block table: same
        captured C++ call as a regular cache, projections and rope included. The mirror publish
        is the only streaming-specific work left. Returns None when the step must take the
        chunked driver instead (window slid, host inputs missing, graph unsupported).
        """
        if x.dtype != torch.float16 or x.dim() != 3 or x.shape[0] != 1 or not x.is_contiguous():
            return None
        if params.get("causal", True) is not True:
            # Non-causal callers (draft graphs) are not eligible: the streaming driver asserts
            # causal outright, so declining keeps the failure loud instead of silently wrong
            return None
        self._ensure_resident()
        if host_seqlens is None:
            cs_host = params.get("cache_seqlens")
            if not (isinstance(cs_host, torch.Tensor) and cs_host.device.type == "cpu" and
                    cs_host.numel() >= 1):
                return None
            host_seqlens = cs_host
        host_bt = params.get("block_table")
        if not (isinstance(host_bt, torch.Tensor) and host_bt.device.type == "cpu" and
                host_bt.dim() == 2):
            return None
        a = int(host_seqlens[0])
        q_len = x.shape[1]
        L = a + q_len
        if a < 0 or q_len < 1 or L > self.cfg["R"] * PAGE_SIZE:
            return None
        bca = self._bc_stream
        if bca is None:
            from ..modules.attention_fn.bc_attn import build_bc_stream
            bca = self._bc_stream = (build_bc_stream(attn, self) or False)
        if bca is False:
            return None

        self._account_rollback(a)
        if a != self._wrote_upto:
            self._epoch += 1
            self._slot_owner = [None] * self.cfg["R"]
            self._valid_from = None
            self._trace(f"epoch {self._epoch} at {a} (bc step, last published {self._wrote_upto})")
        self._wait_publish()
        self._ensure_state(host_bt, a, L)
        # The graph appends rows [a, L) itself, through the identity table (see docstring), and
        # counts the new tokens as part of the sequence
        self._stage.fill_(a)
        y = bca.step(x, self._stage, self._identity_table(), position, positions, position_ids,
                     inv_freq, causal = True, host_seqlens = None)
        self._publish_append(a, L)
        return y

    def _walk_decode(self, q, out, L, window_start, kv_lo, sm_scale, sinks, nkv_override,
                     window_left, window_right, softcap):
        from ..modules.attention_fn.triton_paged import PagedAttnStreamDecode
        q_len = q.shape[1]
        key = (q_len, window_left, window_right, softcap)
        geo = self._geo_dec.get(key)
        if geo is None:
            geo = self._geo_dec[key] = PagedAttnStreamDecode(
                q, self.num_kv_heads, self._qc_pair(),
                window_left = window_left, window_right = window_right, softcap = softcap)
        block_n = geo.candidates[0] if geo.candidates else 64
        bytes_per_slot = geo.programs * (geo.block_rows * geo.hd_pad * 4 + geo.block_rows * 8)
        n_chunks, plan, total_slots, final = self._plan_chunks(
            L, window_start, kv_lo, geo.programs, block_n, self._decode_splits, bytes_per_slot)
        self._trace(f"decode walk: L={L}, window_start={window_start}, kv_lo={kv_lo}, "
                    f"chunks={n_chunks}, slots={total_slots}, final={final}")
        if not final:
            no, nml = geo.partial_numel(total_slots)
            if self._part_dec is None or self._part_dec[2] < no:
                self._part_dec = (
                    torch.empty(no, dtype = torch.float32, device = self.device),
                    torch.empty(nml, dtype = torch.float32, device = self.device),
                    no,
                )
        resident = self._resident_tensors()
        self._run_walk(geo, q, out, resident, 0, window_start, plan, total_slots, final,
                       sm_scale, sinks, nkv_override, prefill = False)

    def _walk_prefill(self, q, out, L, window_start, kv_lo, sm_scale, sinks, nkv_override,
                      window_left, window_right, softcap):
        from ..modules.attention_fn.triton_paged import PagedAttnStreamPrefill
        q_len = q.shape[1]
        key = (q_len, window_left, window_right, softcap)
        geo = self._geo_pre.get(key)
        resident = self._resident_tensors()
        kc = resident["k"] if "k" in resident else resident["qk"]
        vc = resident["v"] if "v" in resident else resident["qv"]
        if geo is None:
            geo = self._geo_pre[key] = PagedAttnStreamPrefill(
                q, self.num_kv_heads, self._qc_pair(),
                window_left = window_left, window_right = window_right, softcap = softcap)
        geo.ensure_config(q, kc, vc, self._kt, self._stage, True,
                          n_kv_heads_override = nkv_override)
        block_m, block_n = geo.cfg[0], geo.cfg[1]
        q_blocks = -(-q_len // block_m)
        programs = q_blocks * self.num_q_heads
        bytes_per_slot = programs * (block_m * geo.hd_pad * 4 + block_m * 8)
        n_chunks, plan, total_slots, final = self._plan_chunks(
            L, window_start, kv_lo, programs, block_n, self._prefill_splits, bytes_per_slot)
        self._trace(f"prefill walk: L={L}, window_start={window_start}, kv_lo={kv_lo}, "
                    f"chunks={n_chunks}, slots={total_slots}, final={final}")
        if not final:
            no, nml = geo.partial_numel(q_blocks, total_slots)
            if self._part_pre is None or self._part_pre[2] < no:
                self._part_pre = (
                    torch.empty(no, dtype = torch.float32, device = self.device),
                    torch.empty(nml, dtype = torch.float32, device = self.device),
                    no,
                )
        self._run_walk(geo, q, out, resident, 0, window_start, plan, total_slots, final,
                       sm_scale, sinks, nkv_override, prefill = True)

    def _run_walk(self, geo, q, out, resident, kv_append_len, window_start, plan,
                  total_slots, final, sm_scale, sinks, nkv_override, prefill: bool):
        S = self.cfg["S"]
        # Chunk plan entries keep their page-group indices, which are not contiguous when a
        # windowed walk skips groups below the visible range: prefetch in plan order, not by
        # index arithmetic
        chunk_plan = [e for e in plan if e[0] is not None]
        for e in chunk_plan[:S]:
            self._issue_chunk_h2d(e[0], window_start)
        ring = self._ensure_ring()["tensors"] if chunk_plan else None
        if not final:
            part_o, part_ml = (self._part_pre if prefill else self._part_dec)[:2]
        else:
            part_o = part_ml = q
        slot = 0
        chunk_pos = 0
        q_blocks = None
        for ci, nb, ne, splits in plan:
            if ci is not None:
                self._wait_chunk_ready(ci)
                src = ring
            else:
                src = resident
            kc = src["k"] if "k" in src else src["qk"]
            vc = src["v"] if "v" in src else src["qv"]
            ks, vs = src.get("sk"), src.get("sv")   # quantized caches: scales ride the placement
            if prefill:
                q_blocks = geo.launch(
                    q, out, kc, vc, self._kt, self._stage, part_o, part_ml,
                    kv_append_len, nb, ne, slot, total_slots, splits, True, sm_scale,
                    sinks = sinks, final = final, n_kv_heads_override = nkv_override,
                    k_scales = ks, v_scales = vs,
                )
            else:
                split_len = self._split_len(ne - nb, splits, geo)
                geo.launch(
                    q, out, kc, vc, self._kt, self._stage, part_o, part_ml,
                    nb, ne, slot, total_slots, splits, split_len, True, sm_scale,
                    sinks = sinks, final = final, n_kv_heads_override = nkv_override,
                    k_scales = ks, v_scales = vs,
                )
            if ci is not None:
                self._record_chunk_consumed(ci)
                if chunk_pos + S < len(chunk_plan):
                    self._issue_chunk_h2d(chunk_plan[chunk_pos + S][0], window_start)
                chunk_pos += 1
            slot += splits
        if not final:
            if prefill:
                geo.combine(part_o, part_ml, out, total_slots, q, sinks = sinks,
                            q_blocks = q_blocks)
            else:
                geo.combine(part_o, part_ml, out, total_slots, q, sinks = sinks)

    def _qc_pair(self):
        if not self._quant:
            return None
        # Same layout the stock dispatch builds (common.py): scales plus bit widths
        _qk, sk, _qv, sv, k_bits, v_bits = self.get_qkv()
        return (sk, sv, k_bits, v_bits)

    # ---- prompt caching ----

    @override
    def copy_page(self, source, from_page: int, to_page: int, num_tokens: int):
        assert type(self) is type(source), "streaming page copies need matching layer types"
        assert num_tokens <= PAGE_SIZE
        self._quiesce()
        for name, t in self.mirror.items():
            t[to_page, :num_tokens].copy_(source.mirror[name][from_page, :num_tokens],
                                          non_blocking = False)
        self._pub_rows[to_page] = num_tokens
        self._stale_pages.add(to_page)
        self._valid_from = None

    def rotate_pages(self, order_cpu: torch.Tensor):
        """
        Permute the mirror pages with the same sequential (dst, src, -1 = temp) semantics as
        ext.cache_rotate, and permute the publication ledger with them. The resident slots are
        invalidated: they are refilled from the (rotated) mirror on the next step.
        """
        self._quiesce()
        self._ensure_resident()
        order = order_cpu.tolist()
        idx = list(range(self.capacity_pages))
        tmp = None
        for i in range(0, len(order), 2):
            a_, b_ = order[i], order[i + 1]
            src = tmp if b_ < 0 else idx[b_]
            if a_ < 0:
                tmp = src
            else:
                idx[a_] = src
        old_pub = self._pub_rows
        for name, t in self.mirror.items():
            tmp_page = torch.empty_like(t[0])
            for i in range(0, len(order), 2):
                a_, b_ = order[i], order[i + 1]
                dst = tmp_page if a_ < 0 else t[a_]
                src = tmp_page if b_ < 0 else t[b_]
                dst.copy_(src, non_blocking = False)
        self._pub_rows = [old_pub[i] for i in idx]
        self._epoch += 1
        self._slot_owner = [None] * self.cfg["R"]
        self._table_key = None
        self._stale_pages = set()
        self._valid_from = None

    # ---- accounting ----

    @override
    def storage_size(self):
        """GPU bytes (resident window); the mirror lives in pinned host memory"""
        if not self._resident_ready:
            return 0
        total = 0
        for t in self._resident_tensors().values():
            total += t.numel() * t.element_size()
        return total

    def mirror_size(self) -> int:
        """Host bytes of the pinned mirror"""
        return sum(t.numel() * t.element_size() for t in (self.mirror or {}).values())


# -------------------------------------------------------------------------------------------------
# Concrete layers

class CacheLayer_stream(StreamLayer, CacheLayer_fp16):

    COMPONENTS = ("k", "v")

    def __init__(self, config, attention, cache_id, max_num_tokens, **kwargs):
        super().__init__(config, attention, cache_id, max_num_tokens, **kwargs)
        self._quant = False

    @override
    def stream_per_page_bytes(self) -> int:
        return 2 * PAGE_SIZE * self.num_kv_heads * self.head_dim * torch.half.itemsize

    @override
    def _alloc_resident(self, device):
        sh = (self.cfg["R"], PAGE_SIZE, self.num_kv_heads, self.head_dim)
        self.k = torch.zeros(sh, dtype = torch.half, device = device)
        self.v = torch.zeros(sh, dtype = torch.half, device = device)

    @override
    def _free_resident(self):
        self.k = self.v = None

    @override
    def _alloc_mirror(self, device):
        sh = (self.capacity_pages, PAGE_SIZE, self.num_kv_heads, self.head_dim)
        self.mirror = {
            "k": torch.zeros(sh, dtype = torch.half, pin_memory = True),
            "v": torch.zeros(sh, dtype = torch.half, pin_memory = True),
        }

    @override
    def _resident_tensors(self):
        return {"k": self.k, "v": self.v}

    @override
    def _write_rows(self, k, v, a: int, length: int):
        from exllamav3.ext import exllamav3_ext as ext
        ext.paged_kv_cache_update(k, v, self.k, self.v, self._kt, self._stage)


class CacheLayer_stream_quant(StreamLayer, CacheLayer_quant):

    COMPONENTS = ("qk", "sk", "qv", "sv")

    def __init__(self, config, attention, cache_id, max_num_tokens, **kwargs):
        super().__init__(config, attention, cache_id, max_num_tokens, **kwargs)
        self._quant = True

    @override
    def stream_per_page_bytes(self) -> int:
        return (self.qshape_k[2] * torch.int.itemsize +
                self.qshape_v[2] * torch.int.itemsize +
                2 * self.qshape_s[2] * torch.half.itemsize) * PAGE_SIZE

    @override
    def _alloc_resident(self, device):
        R = self.cfg["R"]
        self.qk = torch.zeros((R, PAGE_SIZE, self.qshape_k[2]), dtype = torch.int, device = device)
        self.qv = torch.zeros((R, PAGE_SIZE, self.qshape_v[2]), dtype = torch.int, device = device)
        self.sk = torch.zeros((R, PAGE_SIZE, self.qshape_s[2]), dtype = torch.half, device = device)
        self.sv = torch.zeros((R, PAGE_SIZE, self.qshape_s[2]), dtype = torch.half, device = device)

    @override
    def _free_resident(self):
        self.qk = self.qv = self.sk = self.sv = None

    @override
    def _alloc_mirror(self, device):
        pages = self.capacity_pages
        self.mirror = {
            "qk": torch.zeros((pages, PAGE_SIZE, self.qshape_k[2]), dtype = torch.int, pin_memory = True),
            "qv": torch.zeros((pages, PAGE_SIZE, self.qshape_v[2]), dtype = torch.int, pin_memory = True),
            "sk": torch.zeros((pages, PAGE_SIZE, self.qshape_s[2]), dtype = torch.half, pin_memory = True),
            "sv": torch.zeros((pages, PAGE_SIZE, self.qshape_s[2]), dtype = torch.half, pin_memory = True),
        }

    @override
    def _resident_tensors(self):
        return {"qk": self.qk, "sk": self.sk, "qv": self.qv, "sv": self.sv}

    @override
    def _write_rows(self, k, v, a: int, length: int):
        from exllamav3.ext import exllamav3_ext as ext
        ext.quant_cache_paged(k, self.qk, self.sk, v, self.qv, self.sv,
                              self._stage, self._kt, PAGE_SIZE, length, self.compand_a, True)
