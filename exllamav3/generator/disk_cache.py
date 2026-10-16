from __future__ import annotations
import atexit
import errno
import hashlib
import heapq
import json
import logging
import os
import shutil
import sqlite3
import struct
import threading
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from ..constants import PAGE_SIZE
from ..version import __version__
from ..cache.recurrent import host_pool, new_checkpoint_handle
from ..model.model_tp_fn import (
    mp_disk_layout, mp_disk_set_layout, mp_disk_write, mp_disk_read,
    mp_disk_state_describe, mp_disk_state_write, mp_disk_state_read, pack_stash, unpack_stash,
)
from .cpu_cache import _align

logger = logging.getLogger(__name__)

PAGE_MAGIC = b"EXL3KVP1"
STATE_MAGIC = b"EXL3RCS1"
_HEADER = struct.Struct("<II")       # (header json length, payload-descriptor length)

# Several processes can use the same root. A process does not remove a file that it does not know (a temporary
# file, a rank marker) if the file changed in the last _STRAY_AGE seconds: the file can belong to a write of
# an other process. For the same reason, the root budget does not remove a directory whose index changed in
# the last _STRAY_AGE seconds. The writer touches the index at least each _TOUCH_INTERVAL seconds while it works
_STRAY_AGE = 600
_TOUCH_INTERVAL = 60

# Bump when a change to the code alters the bytes of a page image or a checkpoint (a kernel or module computing
# K/V differently, a new checkpoint layout). Part of the fingerprint, so existing caches are left alone until
# then; the version string and the model/cache identity cover the rest
DISK_CACHE_FORMAT_VERSION = 2


# ---------------------------------------------------------------------------------------------------------------
# Fingerprint: everything that changes the bytes of a page image or a checkpoint. A cache directory belongs to
# exactly one fingerprint, so a requantized model, a different cache layer or bit width, a different draft cache
# or a code change that alters K/V numerics (DISK_CACHE_FORMAT_VERSION) can never restore stale pages.
# ---------------------------------------------------------------------------------------------------------------

def model_identity(config) -> str:
    """
    config.json, the raw safetensors headers (names, shapes, dtypes, offsets) and samples of the weights. A
    requantization at the same bitrate can give the same headers and the same file sizes, but not the same
    weights. Thus each file adds its size and seven 64 kB samples, spread through the file. This reads less
    than 1 MB per file.
    """
    h = hashlib.blake2b(digest_size = 16)
    h.update(json.dumps(getattr(config, "config_dict", {}), sort_keys = True, default = str).encode())
    directory = getattr(config, "directory", None)
    if directory and os.path.isdir(directory):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(directory, name)
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h.update(name.encode())
                h.update(str(size).encode())
                h.update(f.read(n))
                for k in range(1, 8):
                    f.seek(size * k // 8)
                    h.update(f.read(65536))
    return h.hexdigest()


def _layer_key(key, layer) -> str | None:
    """The module key of a cache layer: Cache.layers is keyed by placement tuples, the layer holds its module"""
    module = getattr(layer, "attention", None)
    return getattr(module, "key", None) or getattr(key, "key", None)


def cache_fingerprint(caches: list, canonical: list[dict], image_size: int) -> tuple[str, dict]:
    """(fingerprint, its inputs in clear) for page images in the canonical layout `canonical` (one entry per
    cache tensor: rows, row bytes, dtype, image offset), which does not depend on a tensor-parallel split"""
    parts = {
        "exllamav3": __version__,
        "format": DISK_CACHE_FORMAT_VERSION,
        "page_size": PAGE_SIZE,
        "image_size": image_size,
        "segments": [[seg["rows"], seg["canonical_row_bytes"], seg["dtype"], seg["image_offset"]] for seg in canonical],
        "caches": [],
    }
    for cache in caches:
        model = getattr(cache, "model", None)
        c = {
            "model": model_identity(model.config) if model is not None else None,
            "layers": [
                [_layer_key(key, layer), type(layer).__name__,
                 getattr(layer, "k_bits", None), getattr(layer, "v_bits", None)]
                for key, layer in cache.layers.items()
            ],
            "recurrent": [
                [str(k), type(l).__name__, l.get_checkpoint_size(), str(getattr(getattr(l, "recurrent_state", None), "dtype", None))]
                for k, l in (cache.get_all_recurrent_layers() if hasattr(cache, "get_all_recurrent_layers") else {}).items()
            ],
        }
        parts["caches"].append(c)
    text = json.dumps(parts, sort_keys = True)
    return hashlib.blake2b(text.encode(), digest_size = 16).hexdigest(), parts


def _raw(t: torch.Tensor) -> memoryview:
    """The bytes of a host tensor of any dtype (bf16 included) without a copy"""
    return memoryview(t.contiguous().view(torch.uint8).numpy())


def _itemsize(dtype_name: str) -> int:
    return torch.empty((), dtype = getattr(torch, dtype_name)).element_size()


def _tensor_placeholders(struct) -> list[dict]:
    """The {"__t__", "shape", "dtype"} entries of a packed structure, in tensor order"""
    out = []
    def collect(obj):
        if isinstance(obj, dict):
            if "__t__" in obj:
                out.append(obj)
            else:
                for v in obj.values():
                    collect(v)
        elif isinstance(obj, list):
            for o in obj:
                collect(o)
    collect(struct)
    out.sort(key = lambda d: d["__t__"])
    return out


# ---------------------------------------------------------------------------------------------------------------

class DiskPageCache:
    """
    Third tier for the paged KV cache and the recurrent checkpoint cache: a directory of page images and
    checkpoints on disk, written through from the CPU page tier (CPUPageCache) and the recurrent cache, read
    back by promotion into the CPU tier when an allocation needs a page that is in neither VRAM nor system
    memory. Pages and checkpoints are keyed by the same chained content hash the page table uses (blake2b over
    previous hash and token ids, stable across processes), so a context that was generated in one process can
    be resumed in the next.

    Layout: <root>/<fingerprint>/manifest.json, index.sqlite, pages/<hh>/<phash>.kv, states/<phash>.rs. Each
    file is self-describing (its header carries the hash, the parent hash, the tokens and the fingerprint), so
    the SQLite index is a rebuildable cache of the metadata and the directory is reconciled against it at open.
    The fingerprint covers the model weights (safetensors headers and samples), the cache configuration (layer
    types, bit widths, the slot segment table), the checkpoint layout, the page size and the format version, so
    a directory can never serve pages computed under a different configuration.

    Write path: every page entering the CPU tier is queued for the writer thread, which waits for the tier's
    device-to-host copy (a CUDA event recorded by the store) and writes the slab straight from pinned memory.
    Writing on the way in rather than on the way out keeps the CPU tier's slot recycling free of any wait on
    the disk: an evicted slot has already been written, or was deliberately never queued. The only coupling is
    that a slot with a write still queued is passed over by the tier's eviction. Checkpoints are written the
    same way from RecurrentCache.put, which defers returning a stash's buffers to the pool until the write is
    done. Multimodal pages (placeholder ids are per-process) and pages without a content hash are never written.

    Read path: Generator.enqueue hands the tier a job's page hashes; those on disk but in no higher tier are
    promoted into CPU-tier slots on a reader pool, bounded so promotions never evict each other, and the
    allocation then restores them with the ordinary host-to-device copy, waiting on a read still in flight.
    Hashes the prefetch did not reach are promoted synchronously by the allocation when it meets them.

    Checkpoints are written only once their anchor page is on disk (a checkpoint without its page is
    unrestorable and the states of a hybrid model outweigh its pages several times over), and pruned at idle
    to the ones some root-to-leaf chain still needs at min_state_distance spacing counted from the root;
    orphans go with the same pass.

    Checkpoints are stored per layer in the same spirit: every stashed tensor of every recurrent layer at full
    width, keyed by the layer's module key, with the state's ints (position, window) alongside. A process
    without TP writes and reads them directly from the stash dict; a tensor-parallel side has its ranks write
    and read their own pieces (each layer state's tp_stash_pieces, model_tp_fn.mp_disk_state_*), so a
    checkpoint moves between splits and between TP and non-TP loads like a page does.

    Tensor-parallel caches: the page image on disk is the canonical layout regardless of how the heads are
    split across ranks, so a file written under one split restores under another, or without TP. The main
    process assigns every cache tensor its place in the image from the ranks' shard geometry (mp_disk_layout)
    and creates each page file; the ranks write and read their own strided chunks of it (model_tp_fn
    DiskRankWorker) and signal completion with marker files that this process's writer thread polls, since
    only the main thread may use the dispatch pipes. A local draft cache next to a TP main cache occupies
    the leading part of the image, written and read here as in the non-TP case.

    Eviction keeps the root within max_size: the budget is shared by every fingerprint directory under the
    root (other models, other cache modes, with or without a draft cache). At open, the least recently used
    sibling directories are removed whole until what remains fits, and this directory then runs within the
    rest. Within a directory: orphaned chains first (parent in no tier), then whole trees, least recently used
    root first, pruned tail-first; a page's checkpoint goes with the page. All SQLite traffic happens on the
    writer thread; the main thread works from an in-memory mirror of the index.
    """

    def __init__(
        self,
        root_dir: str,
        caches: list,
        cpu_tier,
        max_size: int,
        read_threads: int = 4,
        fingerprint: str | None = None,
        min_state_distance: int = 4096,
        min_free: int | None = None,
    ):
        """
        :param root_dir:
            Directory holding one subdirectory per fingerprint (created as needed)
        :param caches:
            The caches whose pages the CPU tier images ([cache] or [cache, draft_cache])
        :param cpu_tier:
            The CPUPageCache this tier backs; must not be tensor-parallel (the slabs live in the rank processes)
        :param max_size:
            Budget in bytes for the whole root directory, all fingerprints together
        :param read_threads:
            Reader pool size for promotions
        :param fingerprint:
            Override of the computed fingerprint (tests)
        :param min_free:
            Free space in bytes that the tier leaves on the file system. Below it, no new writes start. Default
            is 2% of the file system, but not more than 16 GB

        :param min_state_distance:
            Minimum spacing, in tokens, between the recurrent checkpoints kept along any root-to-leaf chain of
            pages on disk; closer ones are pruned at idle (see prune_states). 0 keeps every checkpoint
        """
        self.cpu_tier = cpu_tier
        self.root_dir = root_dir
        self.root_size = int(max_size)
        self.max_size = int(max_size)       # this directory's share: root budget minus the other directories
        self.pagetable = None

        # Canonical page image: this process's slab first (its segments at their slab offsets, so the local
        # part of the image is the slab byte for byte), then every tensor-parallel cache tensor at full width
        self.local_bytes = cpu_tier.slab_size
        canonical = []
        for t, offset, shape, dtype in cpu_tier.segments:
            nbytes = t[0].numel() * t.element_size()
            rows = PAGE_SIZE if len(shape) >= 1 and shape[0] == PAGE_SIZE else 1
            canonical.append({"rows": rows, "canonical_row_bytes": nbytes // rows, "dtype": str(dtype),
                              "image_offset": offset})
        self.tp_groups = []     # (model, cache_ids, devices)
        image_end = self.local_bytes
        rank_layouts = []       # (model, devices, per-device layout lists) to send back once offsets are known
        for model, cache_ids in cpu_tier.tp_groups:
            devices = list(model.active_devices)
            layouts = model.tp_worker_dispatch_wait_multi(devices, mp_disk_layout, (cache_ids,))
            # Union of the ranks' tensors, in the main process's layer order, each at full width
            entries = {}
            for layout in layouts:
                for seg in layout:
                    ident = (seg["cache_id"], seg["key"], seg["idx"])
                    known = entries.setdefault(ident, seg)
                    assert known["canonical_row_bytes"] == seg["canonical_row_bytes"] and known["rows"] == seg["rows"], \
                        f"ranks disagree on the page layout of {seg['key']}"
            order = []
            for cache in caches:
                if id(cache) not in cache_ids:
                    continue
                for lkey, layer in cache.layers.items():
                    key = _layer_key(lkey, layer)
                    idx = 0
                    while (id(cache), key, idx) in entries:
                        order.append((id(cache), key, idx))
                        idx += 1
            assert len(order) == len(entries), "tensor-parallel cache layout does not match the main process's layers"
            offsets = {}
            for ident in order:
                seg = entries[ident]
                offsets[ident] = image_end
                canonical.append({"rows": seg["rows"], "canonical_row_bytes": seg["canonical_row_bytes"],
                                  "dtype": seg["dtype"], "image_offset": image_end})
                image_end = _align(image_end + seg["rows"] * seg["canonical_row_bytes"], 256)
            for layout in layouts:
                for seg in layout:
                    seg["image_offset"] = offsets[(seg["cache_id"], seg["key"], seg["idx"])]
            self.tp_groups.append((model, cache_ids, devices))
            rank_layouts.append((model, devices, layouts))
        self.image_size = _align(image_end, 4096)
        for model, devices, layouts in rank_layouts:
            model.tp_worker_dispatch_wait_multi(devices, mp_disk_set_layout, (self.image_size,), [(l,) for l in layouts])
        assert self.image_size, "No paged cache layers to attach the disk cache tier to"

        if fingerprint is None:
            self.fingerprint, parts = cache_fingerprint(caches, canonical, self.image_size)
        else:
            self.fingerprint, parts = fingerprint, {"override": fingerprint}
        self.dir = os.path.join(root_dir, self.fingerprint[:16])
        self.pages_dir = os.path.join(self.dir, "pages")
        self.states_dir = os.path.join(self.dir, "states")
        os.makedirs(self.pages_dir, exist_ok = True)
        os.makedirs(self.states_dir, exist_ok = True)
        if min_free is None:
            usage = shutil.disk_usage(self.dir)
            min_free = min(16 * 1024**3, usage.total // 50)
        manifest = os.path.join(self.dir, "manifest.json")
        if not os.path.exists(manifest):
            with open(manifest, "w") as f:
                json.dump({"fingerprint": self.fingerprint, "inputs": parts}, f, indent = 2, default = str)

        # Mirrors of the index: phash -> {"prev_hash", "size", "last_access"} and phash -> {"size",
        # "last_access", "position"}. Updated by the main thread, persisted by the writer
        self.pages: dict[bytes, dict] = {}
        self.states: dict[bytes, dict] = {}
        self.total_size = 0
        self.access_serial = 0

        # Writes queued but not yet on disk. A page here must not have its slot recycled; a state here keeps a
        # reference to its stash so the buffers cannot return to the pool
        self.pending_pages: set[bytes] = set()
        self.pending_states: dict[bytes, dict] = {}
        # side ("target" / "draft") -> (model, cache) of the recurrent states stashed under that key, set by the
        # Generator; decides whether a side is written here or by the model's ranks
        self.recurrent_sides: dict[str, tuple] = {}
        # Checkpoints offered by the recurrent cache (note_state) whose anchor page is not on disk yet: written
        # when the page lands (the writer thread queues the hash in _ready_states, the main thread writes),
        # dropped if the checkpoint leaves RAM before that
        self.state_candidates: dict[bytes, dict] = {}
        self._ready_states = deque()
        self.min_state_distance = int(min_state_distance)
        self._deferred_release: dict[int, tuple] = {}     # id(stash) -> (stash, release fn)
        self._promotions = 0                              # reads in flight
        self.max_promotions = max(2, cpu_tier.max_slots // 4)

        self._order = deque()
        self._order_pops = 0
        self._order_rebuild = 64

        self.metrics = {
            "page_writes": 0, "page_write_bytes": 0, "page_reads": 0, "page_read_bytes": 0,
            "page_read_failures": 0, "state_writes": 0, "state_reads": 0, "evictions": 0, "skipped_mm": 0,
            "write_errors": 0, "states_pruned": 0, "state_candidates_dropped": 0, "skipped_low_space": 0,
        }

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._queue = deque()
        self._tp_waits = []         # tensor-parallel transfers in flight: (kind, phash, markers, payload)
        self._closed = False
        # A full disk (or a run of other write failures) turns the tier read-only for the rest of the session:
        # whatever is on disk still restores, nothing new is queued, one message says so
        self.write_disabled = False
        self._consecutive_failures = 0
        # A full disk pauses the writes for a time and does not stop them for the session: space comes back
        # when files are removed. Writes also stop while the free space is below min_free
        self.min_free = min_free
        self._space_pause_until = 0.0
        self._space_checked = 0.0
        self._space_ok = True

        # Temporary files carry the process id, so two processes never write the same temporary file
        self._tmp_suffix = f".{os.getpid()}.tmp"
        self._last_touch = 0.0
        # Activity summary for the log: metric values at the last report, and the times of the last report and
        # the last activity
        self._report_base = None
        self._report_time = time.time()
        self._activity_time = 0.0

        self._load_index()
        self._reclaim_root()
        self._writer = threading.Thread(target = self._writer_loop, name = "disk_cache_writer", daemon = True)
        self._writer.start()
        self._readers = ThreadPoolExecutor(read_threads, thread_name_prefix = "disk_cache_read")
        # A process that ends without Generator.close() would otherwise kill the writer mid-file (daemon
        # thread) and lose whatever was queued; drain it at exit instead. Held by weak reference so the hook
        # does not keep a dropped tier alive
        import weakref
        ref = weakref.ref(self)
        atexit.register(lambda: (lambda d: d and d.close())(ref()))


    # -- index -------------------------------------------------------------------------------------------------

    def _connect(self):
        # An other process can hold the write lock of the index for a short time
        conn = sqlite3.connect(os.path.join(self.dir, "index.sqlite"), timeout = 60)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE IF NOT EXISTS pages (phash BLOB PRIMARY KEY, prev_hash BLOB, size INTEGER, "
                     "last_access INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS states (phash BLOB PRIMARY KEY, size INTEGER, last_access INTEGER, "
                     "position INTEGER)")
        return conn


    def _page_path(self, phash: bytes) -> str:
        return os.path.join(self.pages_dir, phash[:1].hex(), phash.hex() + ".kv")


    def _state_path(self, phash: bytes) -> str:
        return os.path.join(self.states_dir, phash.hex() + ".rs")


    def _load_index(self):
        """Load the mirror from SQLite and reconcile it with the directory: rows without a file are dropped,
        files without a row are adopted from their headers (a crash between rename and insert)"""
        conn = self._connect()
        try:
            for phash, prev_hash, size, last_access in conn.execute("SELECT phash, prev_hash, size, last_access FROM pages"):
                if os.path.exists(self._page_path(phash)):
                    self.pages[phash] = {"prev_hash": prev_hash, "size": size, "last_access": last_access}
                else:
                    conn.execute("DELETE FROM pages WHERE phash = ?", (phash,))
            for phash, size, last_access, position in conn.execute("SELECT phash, size, last_access, position FROM states"):
                if os.path.exists(self._state_path(phash)):
                    self.states[phash] = {"size": size, "last_access": last_access, "position": position}
                else:
                    conn.execute("DELETE FROM states WHERE phash = ?", (phash,))
            for dirpath, _, filenames in os.walk(self.pages_dir):
                for name in filenames:
                    path = os.path.join(dirpath, name)
                    try:
                        phash = bytes.fromhex(name[:-3]) if name.endswith(".kv") else None
                    except ValueError:
                        phash = None
                    if phash is None:
                        # A .tmp, a rank marker of an interrupted transfer, or a stray. A new one can belong to
                        # a write of an other process
                        self._remove_stray(path)
                        continue
                    if phash in self.pages:
                        continue
                    meta = self._read_page_header(path)
                    if meta is None:
                        os.remove(path)
                        continue
                    self.pages[phash] = {"prev_hash": meta["prev_hash"], "size": os.path.getsize(path), "last_access": 0}
                    conn.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?)",
                                 (phash, meta["prev_hash"], self.pages[phash]["size"], 0))
            for name in os.listdir(self.states_dir):
                path = os.path.join(self.states_dir, name)
                try:
                    phash = bytes.fromhex(name[:-3]) if name.endswith(".rs") else None
                except ValueError:
                    phash = None
                if phash is None:
                    self._remove_stray(path)
                    continue
                if phash in self.states:
                    continue
                meta = self._read_state_header(path)
                if meta is None:
                    os.remove(path)
                    continue
                self.states[phash] = {"size": os.path.getsize(path), "last_access": 0, "position": meta.get("position", 0)}
                conn.execute("INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?)",
                             (phash, self.states[phash]["size"], 0, self.states[phash]["position"]))
            conn.commit()
        finally:
            conn.close()
        self.total_size = sum(m["size"] for m in self.pages.values()) + sum(m["size"] for m in self.states.values())
        self.access_serial = max([m["last_access"] for m in self.pages.values()] +
                                 [m["last_access"] for m in self.states.values()] + [0])


    @staticmethod
    def _remove_stray(path: str):
        try:
            if os.path.getmtime(path) < time.time() - _STRAY_AGE:
                os.remove(path)
        except OSError:
            pass


    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total


    def _reclaim_root(self):
        """
        The budget is for the root: other fingerprint directories (other models, other cache modes) count
        against it. Remove the least recently used of them, whole, until this directory can have at least its
        own current size plus whatever is left; a directory's recency is the mtime of its index, touched
        on every flush and close.
        """
        siblings = []
        active = 0
        for name in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir, name)
            if path == self.dir or not os.path.isfile(os.path.join(path, "manifest.json")):
                continue
            index = os.path.join(path, "index.sqlite")
            used = os.path.getmtime(index) if os.path.exists(index) else os.path.getmtime(path)
            if used > time.time() - _STRAY_AGE:
                # An other process can use this directory now: count its size, but do not remove it
                active += self._dir_size(path)
                continue
            siblings.append((used, path, self._dir_size(path)))
        siblings.sort()
        others = active + sum(sz for _, _, sz in siblings)
        while siblings and others + self.total_size > self.root_size:
            _, path, sz = siblings.pop(0)
            shutil.rmtree(path, ignore_errors = True)
            others -= sz
            self.metrics["dirs_evicted"] = self.metrics.get("dirs_evicted", 0) + 1
        self.max_size = max(0, self.root_size - others)
        if self.total_size > self.max_size:
            with self._lock:
                self._make_room(0)


    def _touch_dir(self):
        self._last_touch = time.time()
        try:
            os.utime(os.path.join(self.dir, "index.sqlite"))
        except OSError:
            pass


    def _read_page_header(self, path: str) -> dict | None:
        try:
            with open(path, "rb") as f:
                if f.read(len(PAGE_MAGIC)) != PAGE_MAGIC:
                    return None
                hlen, tlen = _HEADER.unpack(f.read(_HEADER.size))
                meta = json.loads(f.read(hlen))
                if meta.get("fingerprint") != self.fingerprint or meta.get("image_size") != self.image_size:
                    return None
                meta["prev_hash"] = bytes.fromhex(meta["prev_hash"]) if meta["prev_hash"] else None
                meta["phash"] = bytes.fromhex(meta["phash"])
                meta["tokens_len"] = tlen
                meta["data_offset"] = len(PAGE_MAGIC) + _HEADER.size + hlen
                return meta
        except (OSError, ValueError, KeyError):
            return None


    def _read_state_header(self, path: str) -> dict | None:
        try:
            with open(path, "rb") as f:
                if f.read(len(STATE_MAGIC)) != STATE_MAGIC:
                    return None
                hlen, _ = _HEADER.unpack(f.read(_HEADER.size))
                meta = json.loads(f.read(hlen))
                if meta.get("fingerprint") != self.fingerprint:
                    return None
                meta["data_offset"] = len(STATE_MAGIC) + _HEADER.size + hlen
                return meta
        except (OSError, ValueError, KeyError):
            return None


    def attach(self, pagetable):
        self.pagetable = pagetable


    # -- queries (main thread) ---------------------------------------------------------------------------------

    def has_page(self, phash: bytes) -> bool:
        return phash in self.pages


    def has_state(self, phash: bytes) -> bool:
        return phash in self.states or phash in self.pending_states


    def page_prev_hash(self, phash: bytes):
        return self.pages[phash]["prev_hash"]


    def __len__(self):
        return len(self.pages)


    def chain_prev(self, phash: bytes):
        """The hash of the page before phash, from VRAM, the CPU tier or the disk tier. None for a first page
        or for a page that is in no tier"""
        pt = self.pagetable
        page = pt.get_live_page(phash) if pt is not None else None
        if page is not None:
            return page.prev_hash
        e = self.cpu_tier.entries.get(phash)
        if e is not None:
            return e["prev_hash"]
        m = self.pages.get(phash)
        return m["prev_hash"] if m is not None else None


    # -- write path --------------------------------------------------------------------------------------------

    def _touch(self, meta: dict, kind: str, phash: bytes):
        self.access_serial += 1
        meta["last_access"] = self.access_serial
        self._enqueue(("touch", kind, phash, self.access_serial))


    def write_page(self, phash: bytes, prev_hash: bytes | None, tokens: torch.Tensor, slot: int, events: list,
                   persistable: bool = True):
        """
        Queue the page image in CPU-tier slot `slot` for writing (called by CPUPageCache.store once the
        device-to-host copies are enqueued; `events` are recorded after them, one per device). Returns True if
        queued, in which case the slot must stay untouched until the write completes (pending_pages).

        `persistable` is the page's CachePage.persistable: False when a multimodal placeholder id precedes the
        end of the page's chain. Placeholder ids are allocated per process, so after a restart a different
        image can take the same ids and so the same page hashes, and every text page hashed behind it would
        then serve K/V computed for the old image. Such pages stay in the in-process tiers only.
        """
        self.drain_ready_states()
        if phash in self.pages or phash in self.pending_pages or not self._can_write():
            return False
        if not persistable:
            self.metrics["skipped_mm"] += 1
            return False
        tokens_bytes = tokens.contiguous().numpy().tobytes()
        size = len(PAGE_MAGIC) + _HEADER.size + 256 + len(tokens_bytes) + self.image_size
        markers = None
        if self.tp_groups:
            # The ranks write straight into the file, so it has to exist before they are told; the dispatch
            # must come from this thread. Their chunks land asynchronously (rank worker threads), the
            # writer thread picks the file up once every rank's marker is there
            tmp = self._page_path(phash) + self._tmp_suffix
            try:
                data_offset = self._create_page_file(tmp, phash, prev_hash, tokens_bytes)
            except OSError as e:
                self._write_failed("page", None, e)
                return False
            markers = []
            for model, cache_ids, devices in self.tp_groups:
                model.tp_dispatch_all(mp_disk_write, (slot, tmp, data_offset))
                markers += [f"{tmp}.d{dev}" for dev in devices]
        with self._lock:
            self._make_room(size)
            self.total_size += size         # reserved; corrected to the file size once written
            self.pending_pages.add(phash)
            self._enqueue(("page", phash, prev_hash, tokens_bytes, slot, events, size, markers))
        return True


    def _page_header(self, phash: bytes, prev_hash: bytes | None, crc: int | None = None) -> bytes:
        meta = {
            "phash": phash.hex(),
            "prev_hash": prev_hash.hex() if prev_hash is not None else None,
            "fingerprint": self.fingerprint,
            "image_size": self.image_size,
            "page_size": PAGE_SIZE,
        }
        if crc is not None:
            meta["crc32"] = crc
        return json.dumps(meta).encode()


    def _create_page_file(self, tmp: str, phash: bytes, prev_hash: bytes | None, tokens_bytes: bytes,
                          crc: int | None = None) -> int:
        """Write the header and tokens and size the file for the whole image; returns the image's offset"""
        header = self._page_header(phash, prev_hash, crc)
        os.makedirs(os.path.dirname(tmp), exist_ok = True)
        with open(tmp, "wb") as f:
            f.write(PAGE_MAGIC)
            f.write(_HEADER.pack(len(header), len(tokens_bytes)))
            f.write(header)
            f.write(tokens_bytes)
            data_offset = f.tell()
            f.truncate(data_offset + self.image_size)
        return data_offset


    def _sides(self, stashed: dict):
        """(top-level ints, {side: side dict}) of a stash: a PairedState stash has "target"/"draft" sides, a
        bare dict is one unnamed side"""
        if any(isinstance(stashed.get(k), dict) for k in ("target", "draft")):
            sides = {k: stashed[k] for k in ("target", "draft") if isinstance(stashed.get(k), dict)}
            top = {k: v for k, v in stashed.items() if k not in sides}
            return top, sides
        return {}, {"__root__": stashed}


    def _side_keymap(self, side: str):
        """{stash key -> module key} for a side's recurrent layers (identity when unknown, as in tests)"""
        info = self.recurrent_sides.get(side)
        if info is None:
            return None
        _, cache = info
        return {k: l.module.key for k, l in cache.get_all_recurrent_layers().items()}


    def _side_is_tp(self, side: str) -> bool:
        info = self.recurrent_sides.get(side)
        return info is not None and bool(getattr(info[0], "loaded_tp", False))


    def write_state(self, phash: bytes, stashed: dict, force: bool = False):
        """
        Queue a checkpoint stash for writing. Non-TP sides are serialized from the stash on the writer thread;
        a tensor-parallel side is described by its ranks (dispatch, main thread), the file is created here and
        the ranks write their pieces, the writer thread publishing the file once their markers are in. The
        stash's buffers (and rank handles) stay alive until then, so the caller routes their release through
        release_after_write. With force = True, the spacing test (min_state_distance) does not apply: the
        caller writes the last checkpoint of a chain (see Generator.flush_cache_to_disk).
        """
        if phash in self.states or phash in self.pending_states or not self._can_write():
            return False
        if not force and not self._spaced(phash, stashed.get("position", 0)):
            self.metrics["state_candidates_dropped"] += 1
            return False
        top, sides = self._sides(stashed)
        size = stashed["checkpoint_size"] + 4096
        with self._lock:
            self._make_room(size)
            self.total_size += size
            self.pending_states[phash] = stashed
        try:
            layout = self._state_layout(phash, top, sides)
        except Exception as e:
            with self._lock:
                self.total_size -= size
                self._state_written(phash)
            self._write_failed("state", None, e)
            return False
        self._enqueue(("state", phash, stashed, size, layout))
        return True


    def _state_layout(self, phash: bytes, top: dict, sides: dict) -> dict:
        """
        Plan a checkpoint file: per side, the ints and per layer the structure and full-width tensors with
        their offsets in the image. Local sides are planned from the stash itself, TP sides from the ranks'
        descriptions (dispatched now, from the main thread); for TP sides the file is also created and the
        ranks' writes dispatched here, since neither may happen from the writer thread.
        """
        manifest = {"phash": phash.hex(), "fingerprint": self.fingerprint, "position": top.get("position", 0),
                    "top": {k: v for k, v in top.items() if not isinstance(v, (dict, list, tuple, torch.Tensor))},
                    "sides": {}}
        local_writes = []       # (offset, tensor) for the writer thread
        tp_writes = []          # (model, cache_id, handle, image_offset, layers)
        image_end = 0
        for side, d in sides.items():
            ints = {k: v for k, v in d.items() if isinstance(v, (int, float, str, bool)) and k != "tp_handle"}
            layers = {}
            if "tp_handle" in d:
                model, cache = self.recurrent_sides[side]
                descs = model.tp_worker_dispatch_wait_multi(
                    model.active_devices, mp_disk_state_describe, (id(cache), d["tp_handle"]))
                for desc in descs:
                    for layer in desc:
                        entry = layers.get(layer["key"])
                        shapes = []
                        for t in layer["tensors"]:
                            shape = list(t["shape"])
                            if t["split"] is not None:
                                shape[t["split"]["dim"]] = t["split"]["total"]
                            shapes.append(shape)
                        if entry is None:
                            layers[layer["key"]] = {
                                "struct": layer["struct"],
                                "tensors": [{"shape": sh, "dtype": t["dtype"]} for sh, t in zip(shapes, layer["tensors"])],
                            }
                        else:
                            assert [e["shape"] for e in entry["tensors"]] == shapes, \
                                f"ranks disagree on the checkpoint layout of {layer['key']}"
            else:
                keymap = self._side_keymap(side) or {}
                for k, v in d.items():
                    if k in ints:
                        continue
                    tensors = []
                    struct = pack_stash(v, tensors)
                    key = keymap.get(k, k if isinstance(k, str) else str(k))
                    layers[key] = {
                        "struct": struct,
                        "tensors": [{"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")} for t in tensors],
                        "_tensors": tensors,
                    }
            # Offsets, 64-byte aligned, one image for the side
            off = 0
            for entry in layers.values():
                for t in entry["tensors"]:
                    n = 1
                    for x in t["shape"]:
                        n *= x
                    t["nbytes"] = n * _itemsize(t["dtype"])
                    t["offset"] = off
                    off = _align(off + t["nbytes"], 64)
            image_offset = image_end
            image_end = _align(image_end + off, 256)
            for entry in layers.values():
                for t, tensor in zip(entry["tensors"], entry.pop("_tensors", [])):
                    local_writes.append((image_offset + t["offset"], tensor))
            manifest["sides"][side] = {"ints": ints, "image_offset": image_offset, "image_size": off, "layers": layers}
            if "tp_handle" in d:
                model, cache = self.recurrent_sides[side]
                tp_writes.append((model, id(cache), d["tp_handle"], image_offset, layers))
        header = json.dumps(manifest).encode()
        path = self._state_path(phash)
        tmp = path + self._tmp_suffix
        layout = {"header": header, "tmp": tmp, "path": path, "local": local_writes, "image_size": image_end, "markers": None}
        if tp_writes:
            data_offset = self._create_state_file(tmp, header, image_end)
            markers = []
            for model, cache_id, handle, image_offset, layers in tp_writes:
                model.tp_dispatch_all(mp_disk_state_write, (cache_id, handle, tmp, data_offset + image_offset, layers))
                markers += [f"{tmp}.d{dev}" for dev in model.active_devices]
            layout["markers"] = markers
            layout["data_offset"] = data_offset
        return layout


    def _create_state_file(self, tmp: str, header: bytes, image_size: int) -> int:
        os.makedirs(os.path.dirname(tmp), exist_ok = True)
        with open(tmp, "wb") as f:
            f.write(STATE_MAGIC)
            f.write(_HEADER.pack(len(header), 0))
            f.write(header)
            data_offset = f.tell()
            f.truncate(data_offset + image_size)
        return data_offset


    def note_state(self, phash: bytes, stashed: dict) -> bool:
        """
        Offer a checkpoint for persistence. Written right away if its anchor page is on disk or on its way
        there, else kept as a candidate until the page lands (or dropped if the checkpoint leaves RAM first).
        Returns True if written or kept.
        """
        self.drain_ready_states()
        if phash in self.states or phash in self.pending_states or not self._can_write():
            return False
        if phash in self.pages or phash in self.pending_pages:
            return self.write_state(phash, stashed)
        with self._lock:
            self.state_candidates[phash] = stashed
        return True


    def drain_ready_states(self):
        """Main thread: write the candidates whose anchor page the writer has published since the last call"""
        while True:
            with self._lock:
                if not self._ready_states:
                    return
                phash = self._ready_states.popleft()
                stashed = self.state_candidates.pop(phash, None)
            if stashed is not None:
                self.write_state(phash, stashed)


    def release_after_write(self, stashed: dict, release_fn):
        """Call release_fn(stashed) now, or once every queued write of this stash has completed. A candidate
        whose page has reached the disk (or is being written) is written first; one whose page has not is
        dropped, its page may still come later but the checkpoint will not"""
        with self._lock:
            cands = [k for k, v in self.state_candidates.items() if v is stashed]
            for k in cands:
                del self.state_candidates[k]
        for k in cands:
            if k in self.pages or k in self.pending_pages:
                self.write_state(k, stashed)
            else:
                self.metrics["state_candidates_dropped"] += 1
        with self._lock:
            pending = any(s is stashed for s in self.pending_states.values())
            if pending:
                self._deferred_release[id(stashed)] = (stashed, release_fn)
                return
        release_fn(stashed)


    def _spaced(self, phash: bytes, position: int) -> bool:
        """Whether a checkpoint at `position` anchored on page phash is at least min_state_distance past the
        nearest checkpoint below it on its chain (the root counting as position 0): the same test
        prune_states applies, so a checkpoint the prune would drop is not written in the first place"""
        if self.min_state_distance <= 0:
            return True
        with self._lock:
            h = self._prev_of(phash)
            steps = 0
            while h is not None and steps <= len(self.pages) + len(self.pending_pages):
                m = self.states.get(h)
                if m is not None:
                    return position - m["position"] >= self.min_state_distance
                if h in self.pending_states:
                    return position - self.pending_states[h].get("position", 0) >= self.min_state_distance
                h = self._prev_of(h)
                steps += 1
        return position >= self.min_state_distance


    def _prev_of(self, phash: bytes) -> bytes | None:
        """Lock held. The previous page of a page on disk or queued for it (a flush queues a whole chain at
        once, so the chain walks must see queued pages or the spacing test would take every candidate above
        them for a root), None for a root or a page in neither"""
        m = self.pages.get(phash)
        if m is not None:
            return m["prev_hash"]
        if phash in self.pending_pages:
            e = self.cpu_tier.entries.get(phash)
            if e is not None:
                return e["prev_hash"]
        return None


    def prune_states(self) -> int:
        """
        Drop the checkpoints no chain needs: orphans (anchor page not on disk) and, along every root-to-leaf
        chain of pages, those less than min_state_distance tokens past the previous kept checkpoint, the root
        counting as one at position 0. Walked from the root, the decision for a checkpoint depends only on
        the chain prefix below it, so every leaf through a shared prefix agrees on it and a branched cache
        prunes consistently; a resumed prompt then replays at most one interval. A checkpoint survives if any
        chain through it keeps it. Returns the number dropped.
        """
        with self._lock:
            if not self.states:
                return 0
            # Queued pages and checkpoints count as present (nothing queued is dropped here): the leaf a flush
            # has just queued, on pages it has just queued, then retires the previous leaf in this same prune
            # instead of lingering until the next one
            pages = dict(self.pages)
            for h in self.pending_pages:
                if h not in pages:
                    pages[h] = {"prev_hash": self._prev_of(h)}
            states = {h: m["position"] for h, m in self.states.items()}
            states.update({h: st.get("position", 0) for h, st in self.pending_states.items()})
            children = {h: [] for h in pages}
            for h, m in pages.items():
                ph = m["prev_hash"]
                if ph is not None and ph in children:
                    children[ph].append(h)
            leaves = [h for h, c in children.items() if not c]
            keep = set()
            for leaf in leaves:
                chain = []
                h = leaf
                while h is not None and len(chain) <= len(pages):
                    chain.append(h)
                    h = pages[h]["prev_hash"] if h in pages else None
                last_kept = 0
                deepest = None
                for h in reversed(chain):
                    pos = states.get(h)
                    if pos is None:
                        continue
                    deepest = h
                    if pos - last_kept >= self.min_state_distance:
                        keep.add(h)
                        last_kept = pos
                # Always keep the last checkpoint of the chain. A conversation continues from the end of its
                # chain, so this checkpoint is the resume point of the next request. Without it, the next
                # request processes up to min_state_distance tokens again
                if deepest is not None:
                    keep.add(deepest)
            drop = [h for h in states if h not in keep and h not in self.pending_states]
            for h in drop:
                self._drop("state", h)
            self.metrics["states_pruned"] += len(drop)
        return len(drop)


    def rekey_state(self, old: bytes, new: bytes, stashed: dict):
        """A deferred page's checkpoint moves to the page's final hash; on disk that is a fresh write under the
        new key (the old one, if it was written, is an orphan and goes with the next eviction pass)"""
        with self._lock:
            if old in self.state_candidates:
                self.state_candidates[new] = self.state_candidates.pop(old)
                return
            if old in self.pending_states:
                # Not written yet. The queued op holds a layout whose header and path carry the old hash, so
                # replace it: queue a new write under the new hash, then remove the old op. In this order, the
                # stash always has a pending write, so its buffers do not go back to the pool. A
                # tensor-parallel write is already dispatched to the ranks and cannot be replaced; it stays
                # under the old hash, which is then an orphan
                self.write_state(new, stashed)
                for i, op in enumerate(self._queue):
                    if op[0] == "state" and op[1] == old and op[4]["markers"] is None:
                        del self._queue[i]
                        self.total_size -= op[3]
                        self._state_written(old)
                        break
                return
        if old in self.states:
            with self._lock:
                self._drop("state", old)
        self.write_state(new, stashed)


    def _can_write(self) -> bool:
        """Main thread. Return False when no new write may start: closed, disabled, paused after a full disk,
        or free space below min_free (examined at most each 5 seconds)"""
        if self._closed or self.write_disabled:
            return False
        now = time.time()
        if now < self._space_pause_until:
            self.metrics["skipped_low_space"] = self.metrics.get("skipped_low_space", 0) + 1
            return False
        if now - self._space_checked > 5.0:
            self._space_checked = now
            try:
                usage = shutil.disk_usage(self.dir)
                self._space_ok = usage.free >= self.min_free
            except OSError:
                self._space_ok = True
        if not self._space_ok:
            self.metrics["skipped_low_space"] = self.metrics.get("skipped_low_space", 0) + 1
        return self._space_ok


    def _enqueue(self, op):
        with self._cv:
            self._queue.append(op)
            self._cv.notify()


    def _make_room(self, nbytes: int):
        # Lock held by the caller: the writer thread inserts into the mirrors while this walks them
        while self.total_size + nbytes > self.max_size and (self.pages or self.states):
            self._evict_one()


    def _evict_one(self):
        while True:
            if not self._order or self._order_pops >= self._order_rebuild:
                self._build_order()
                if not self._order:
                    return
            kind, phash = self._order.popleft()
            self._order_pops += 1
            if kind == "page" and phash in self.pages:
                self._drop("page", phash)
                if phash in self.states:
                    self._drop("state", phash)
                self.metrics["evictions"] += 1
                return
            if kind == "state" and phash in self.states:
                self._drop("state", phash)
                self.metrics["evictions"] += 1
                return


    def _drop(self, kind: str, phash: bytes):
        table = self.pages if kind == "page" else self.states
        meta = table.pop(phash)
        self.total_size -= meta["size"]
        self._enqueue(("delete", kind, phash))


    def _rooted_elsewhere(self, h: bytes) -> bool:
        pt = self.pagetable
        if pt is not None and pt.get_live_page(h) is not None:
            return True
        return h in self.cpu_tier


    def _build_order(self):
        """
        Eviction order over the mirror: orphaned states (anchor page in no tier), orphaned page chains, then
        rooted trees least recently used root first, each pruned leaf-first. Same policy as the CPU tier, over
        last_access instead of the page table's serial.
        """
        pages = self.pages
        children = {h: [] for h in pages}
        roots, orphan_roots = [], []
        for h, m in pages.items():
            ph = m["prev_hash"]
            if ph is not None and ph in children:
                children[ph].append(h)
            elif ph is None or self._rooted_elsewhere(ph):
                roots.append(h)
            else:
                orphan_roots.append(h)

        order = []
        for h in self.states:
            if h not in pages and not self._rooted_elsewhere(h):
                order.append(("state", h))

        def prune(root):
            remaining = {}
            heap = []
            stack = [root]
            while stack:
                h = stack.pop()
                n = len(children[h])
                remaining[h] = n
                if n == 0:
                    heapq.heappush(heap, (pages[h]["last_access"], h))
                stack.extend(children[h])
            while heap:
                _, h = heapq.heappop(heap)
                order.append(("page", h))
                ph = pages[h]["prev_hash"]
                if ph is not None and ph in remaining:
                    remaining[ph] -= 1
                    if remaining[ph] == 0:
                        heapq.heappush(heap, (pages[ph]["last_access"], ph))

        access = lambda h: pages[h]["last_access"]
        for root in sorted(orphan_roots, key = access):
            prune(root)
        for root in sorted(roots, key = access):
            prune(root)
        self._order = deque(order)
        self._order_pops = 0


    # -- writer thread -----------------------------------------------------------------------------------------

    def _writer_loop(self):
        conn = self._connect()
        try:
            while True:
                with self._cv:
                    while not self._queue and not self._tp_waits:
                        if self._report_base is not None:
                            # Unreported activity: wake up to write the summary when the activity stops.
                            # The write-behind of the generator queues pages each 2 seconds, so a shorter
                            # quiet time would give one line for each of its steps
                            if not self._cv.wait(timeout = 5.0):
                                self._report_activity()
                        else:
                            self._cv.wait()
                    if self._tp_waits and (not self._queue or self._queue[0][0] in ("flush", "stop")):
                        # Transfers the ranks are still doing: poll their markers, and hold a flush/stop
                        # behind them (they are what the flush is for)
                        self._cv.wait(timeout = 0.001)
                        self._poll_tp_waits(conn)
                        if self._tp_waits or not self._queue:
                            continue
                    op = self._queue.popleft()
                kind = op[0]
                if kind in ("page", "state", "touch"):
                    self._note_activity()
                try:
                    if kind == "page":
                        self._do_write_page(conn, *op[1:])
                    elif kind == "state":
                        self._do_write_state(conn, *op[1:])
                    elif kind == "touch":
                        _, k, phash, serial = op
                        conn.execute(f"UPDATE {'pages' if k == 'page' else 'states'} SET last_access = ? WHERE phash = ?",
                                     (serial, phash))
                    elif kind == "delete":
                        _, k, phash = op
                        path = self._page_path(phash) if k == "page" else self._state_path(phash)
                        try:
                            os.remove(path)
                        except FileNotFoundError:
                            pass
                        conn.execute(f"DELETE FROM {'pages' if k == 'page' else 'states'} WHERE phash = ?", (phash,))
                    elif kind == "flush":
                        conn.commit()
                        self._touch_dir()
                        op[1].set()
                    elif kind == "stop":
                        conn.commit()
                        self._touch_dir()
                        self._report_activity()
                        op[1].set()
                        return
                except Exception as e:
                    self._write_failed(kind, op, e)
                else:
                    if kind in ("page", "state"):
                        self._consecutive_failures = 0
                if not self._queue:
                    try:
                        conn.commit()
                    except sqlite3.Error as e:
                        self._write_failed("index commit", None, e)
                    if time.time() - self._last_touch > _TOUCH_INTERVAL:
                        # Shows other processes that this directory is in use (see _reclaim_root)
                        self._touch_dir()
        finally:
            conn.close()


    _REPORTED = ("page_writes", "page_write_bytes", "state_writes", "page_reads", "page_read_bytes", "state_reads")

    def _note_activity(self):
        """Writer thread. Start an activity summary, or write the current one if it is 10 seconds old"""
        now = time.time()
        self._activity_time = now
        if self._report_base is None:
            self._report_base = {k: self.metrics[k] for k in self._REPORTED}
            self._report_time = now
        elif now - self._report_time >= 10.0:
            self._report_activity(final = False)


    def _report_activity(self, final: bool = True):
        """Writer thread. Log what was written and read since the last summary"""
        base = self._report_base
        if base is None:
            return
        d = {k: self.metrics[k] - base[k] for k in self._REPORTED}
        now = time.time()
        self._report_base = None if final else {k: self.metrics[k] for k in self._REPORTED}
        self._report_time = now
        parts = []
        if d["page_writes"] or d["state_writes"]:
            parts.append(f"saved {d['page_writes']} pages ({d['page_writes'] * PAGE_SIZE} tokens) and "
                         f"{d['state_writes']} checkpoints, {d['page_write_bytes'] / 1024**2:.0f} MB of pages")
        if d["page_reads"] or d["state_reads"]:
            parts.append(f"restored {d['page_reads']} pages ({d['page_reads'] * PAGE_SIZE} tokens) and "
                         f"{d['state_reads']} checkpoints")
        if not parts:
            return
        logger.info(
            f"Disk cache tier: {'; '.join(parts)}{'' if final else ' (continues)'}. On disk: {len(self.pages)} "
            f"pages, {len(self.states)} checkpoints, {self.total_size / 1024**3:.1f} of "
            f"{self.max_size / 1024**3:.1f} GB"
        )


    def _write_failed(self, kind: str, op, e: Exception):
        """Writer thread. Undo the reservation of a failed page/state write and, on a disk-space error or a run
        of failures, stop writing for the rest of the session: the queue is drained of writes so slots and
        stash buffers are released, reads are unaffected"""
        self.metrics["write_errors"] += 1
        with self._lock:
            if op is not None and kind == "page":
                self.pending_pages.discard(op[1])
                self.total_size -= op[6]
                tmp = self._page_path(op[1]) + self._tmp_suffix
                self._remove_tmp(tmp)
                for model, cache_ids, devices in self.tp_groups:
                    for dev in devices:
                        self._remove_tmp(f"{tmp}.d{dev}")
                        self._remove_tmp(f"{tmp}.d{dev}.err")
            elif op is not None and kind == "state":
                self.total_size -= op[3]
                self._remove_tmp(self._state_path(op[1]) + self._tmp_suffix)
                self._state_written(op[1])
            out_of_space = isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT)
            if self.write_disabled:
                return
            if out_of_space:
                # Pause, do not stop: space comes back when files are removed. A full disk is not a fault of
                # the tier, so it does not count to the limit of consecutive failures
                if time.time() >= self._space_pause_until:
                    logger.warning(f"Disk cache tier: {kind} failed ({e}); writes paused for 60 seconds "
                          f"(restores from what is already on disk continue)")
                self._space_pause_until = time.time() + 60
                self._drop_queued_writes()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3 or (isinstance(e, OSError) and e.errno == errno.EROFS):
                self.write_disabled = True
                reason = "read-only file system" if isinstance(e, OSError) and e.errno == errno.EROFS else \
                    f"{self._consecutive_failures} consecutive failures"
                logger.warning(f"Disk cache tier: {kind} failed ({e}); {reason}, writes disabled for the rest of this "
                      f"session (restores from what is already on disk continue)")
                self._drop_queued_writes()
            else:
                logger.warning(f"Disk cache tier: {kind} failed: {e}")


    def _drop_queued_writes(self):
        """Lock held. Remove the queued page and state writes. Their slots and stash buffers are then free"""
        kept = deque()
        for q in self._queue:
            if q[0] == "page":
                self.pending_pages.discard(q[1])
                self.total_size -= q[6]
            elif q[0] == "state":
                self.total_size -= q[3]
                self._state_written(q[1])
            else:
                kept.append(q)
        self._queue = kept


    def abort(self):
        """
        Stop all writes for the rest of the session. Call this after a generator error. After an error, the
        contents of the cache are not known to be correct, and the disk tier must not keep them for the next
        process. Queued writes are removed. Restores from the files on disk continue.
        """
        with self._lock:
            if not self.write_disabled:
                logger.warning("Disk cache tier: generator error, writes disabled for the rest of this session")
            self.write_disabled = True
            self.state_candidates.clear()
            self._drop_queued_writes()


    @staticmethod
    def _remove_tmp(path: str):
        try:
            os.remove(path)
        except OSError:
            pass


    def _do_write_page(self, conn, phash, prev_hash, tokens_bytes, slot, events, size, markers = None):
        for ev in events:
            ev.synchronize()
        path = self._page_path(phash)
        tmp = path + self._tmp_suffix
        slab = self.cpu_tier.slab_view(slot)
        if markers is None:
            # The full image is here, so the header can carry a checksum of the tokens and the image. The file
            # is published by rename after the write, so a killed process leaves only a temp file; but the file
            # gets its full size before the image is written, and a power loss between the rename and the data
            # writeback can leave a published file of the right size with zeros in place of data. The checksum
            # finds this on read. (Tensor-parallel pages are written in parts by the ranks and have no checksum.)
            crc = zlib.crc32(memoryview(slab.numpy()), zlib.crc32(tokens_bytes)) if slab is not None else None
            data_offset = self._create_page_file(tmp, phash, prev_hash, tokens_bytes, crc)
        else:
            data_offset = len(PAGE_MAGIC) + _HEADER.size + len(self._page_header(phash, prev_hash)) + len(tokens_bytes)
        if slab is not None:
            # This process's part of the image: the local slab, at the front
            with open(tmp, "r+b") as f:
                f.seek(data_offset)
                f.write(memoryview(slab.numpy()))
        if markers is not None:
            # The ranks' chunks are still landing; finished by _poll_tp_waits
            self._tp_waits.append(("page", phash, markers, (prev_hash, size, tmp, path)))
            return
        self._publish_page(conn, phash, prev_hash, size, tmp, path)


    def _publish_page(self, conn, phash, prev_hash, size, tmp, path):
        os.replace(tmp, path)
        actual = os.path.getsize(path)
        with self._lock:
            self.pending_pages.discard(phash)
            self.access_serial += 1
            self.pages[phash] = {"prev_hash": prev_hash, "size": actual, "last_access": self.access_serial}
            self.total_size += actual - size
            if phash in self.state_candidates:
                self._ready_states.append(phash)
        conn.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?)", (phash, prev_hash, actual, self.access_serial))
        self.metrics["page_writes"] += 1
        self.metrics["page_write_bytes"] += actual


    def _poll_tp_waits(self, conn):
        """Writer thread. Finish the tensor-parallel transfers whose every rank marker has appeared"""
        still = []
        for wait in self._tp_waits:
            kind, phash, markers, payload = wait
            done = all(os.path.exists(m) or os.path.exists(m + ".err") for m in markers)
            if not done:
                still.append(wait)
                continue
            failed = any(os.path.exists(m + ".err") for m in markers)
            for m in markers:
                self._remove_tmp(m)
                self._remove_tmp(m + ".err")
            if kind == "page":
                prev_hash, size, tmp, path = payload
                if failed:
                    self._write_failed("page", ("page", phash, None, None, None, None, size), OSError("a rank failed to write its shard"))
                    continue
                try:
                    self._publish_page(conn, phash, prev_hash, size, tmp, path)
                except Exception as e:
                    self._write_failed("page", ("page", phash, None, None, None, None, size), e)
            elif kind == "state":
                tmp, path, size, position = payload
                if failed:
                    self._write_failed("state", ("state", phash, None, size), OSError("a rank failed to write its part"))
                    continue
                try:
                    self._publish_state(conn, phash, tmp, path, size, position)
                except Exception as e:
                    self._write_failed("state", ("state", phash, None, size), e)
            else:
                tokens, done_cb = payload
                with self._lock:
                    self._promotions -= 1
                if failed:
                    self.metrics["page_read_failures"] += 1
                    with self._lock:
                        if phash in self.pages:
                            self._drop("page", phash)
                    done_cb(None)
                else:
                    with self._lock:
                        if phash in self.pages:
                            self._touch(self.pages[phash], "page", phash)
                    self.metrics["page_reads"] += 1
                    self.metrics["page_read_bytes"] += self.image_size
                    done_cb(tokens)
        self._tp_waits = still


    def _do_write_state(self, conn, phash, stashed, size, layout):
        tmp, path = layout["tmp"], layout["path"]
        if layout["markers"] is None:
            data_offset = self._create_state_file(tmp, layout["header"], layout["image_size"])
        else:
            data_offset = layout["data_offset"]
        with open(tmp, "r+b") as f:
            for offset, tensor in layout["local"]:
                f.seek(data_offset + offset)
                f.write(_raw(tensor))
        if layout["markers"] is not None:
            self._tp_waits.append(("state", phash, layout["markers"], (tmp, path, size, stashed.get("position", 0))))
            return
        self._publish_state(conn, phash, tmp, path, size, stashed.get("position", 0))


    def _publish_state(self, conn, phash, tmp, path, size, position):
        os.replace(tmp, path)
        actual = os.path.getsize(path)
        with self._lock:
            self.access_serial += 1
            self.states[phash] = {"size": actual, "last_access": self.access_serial, "position": position}
            self.total_size += actual - size
            self._state_written(phash)
        conn.execute("INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?)", (phash, actual, self.access_serial, position))
        self.metrics["state_writes"] += 1


    def _state_written(self, phash):
        """Lock held. Drop the pending reference and run a deferred release once nothing else is queued for
        the same stash"""
        stashed = self.pending_states.pop(phash, None)
        if stashed is None:
            return
        if any(s is stashed for s in self.pending_states.values()):
            return
        deferred = self._deferred_release.pop(id(stashed), None)
        if deferred is not None:
            deferred[1](deferred[0])


    # -- read path ---------------------------------------------------------------------------------------------

    def _read_tokens(self, phash: bytes):
        """(header meta, token ids) of a page file, or None if missing or invalid (the entry is then dropped)"""
        path = self._page_path(phash)
        meta = self._read_page_header(path)
        try:
            if meta is None or meta["phash"] != phash:
                raise ValueError("bad header")
            with open(path, "rb") as f:
                f.seek(meta["data_offset"])
                tokens = torch.frombuffer(bytearray(f.read(meta["tokens_len"])), dtype = torch.long).clone()
            meta["image_offset"] = meta["data_offset"] + meta["tokens_len"]
            return meta, tokens
        except (OSError, ValueError):
            self.metrics["page_read_failures"] += 1
            with self._lock:
                if phash in self.pages:
                    self._drop("page", phash)
            return None


    def read_page_into(self, phash: bytes, slab: torch.Tensor | None) -> torch.Tensor | None:
        """Read this process's part of the page image for phash into `slab` (a CPU-tier slab view, None when
        there is no local part). Returns the page's token ids, or None if the file is missing or invalid (the
        entry is then dropped from the index)"""
        r = self._read_tokens(phash)
        if r is None:
            return None
        meta, tokens = r
        if slab is not None:
            try:
                with open(self._page_path(phash), "rb") as f:
                    f.seek(meta["image_offset"])
                    if f.readinto(memoryview(slab.numpy())) != self.local_bytes:
                        raise ValueError("short read")
                crc = meta.get("crc32")
                if crc is not None and zlib.crc32(memoryview(slab.numpy()), zlib.crc32(tokens.numpy().tobytes())) != crc:
                    raise ValueError("checksum mismatch")
            except (OSError, ValueError) as e:
                logger.warning(f"Disk cache tier: page {phash.hex()[:12]} unreadable ({e}), dropped")
                self.metrics["page_read_failures"] += 1
                with self._lock:
                    if phash in self.pages:
                        self._drop("page", phash)
                return None
        with self._lock:
            if phash in self.pages:
                self._touch(self.pages[phash], "page", phash)
        self.metrics["page_reads"] += 1
        self.metrics["page_read_bytes"] += self.image_size
        return tokens


    def read_state(self, phash: bytes) -> dict | None:
        """
        Load a checkpoint: local sides into pooled host buffers under the stash's own keys, tensor-parallel
        sides into the ranks' recurrent caches under fresh handles (dispatched and waited for here, on the
        main thread). None if not on disk or unreadable.
        """
        if phash in self.pending_states:
            return self.pending_states[phash]
        if phash not in self.states:
            return None
        path = self._state_path(phash)
        meta = self._read_state_header(path)
        handles = []    # (model, cache_id, handle) created for TP sides, dropped again on failure
        try:
            if meta is None or meta["phash"] != phash.hex():
                raise ValueError("bad header")
            data_offset = meta["data_offset"]
            stashed = dict(meta["top"])
            sides_out = {}
            markers = []
            with open(path, "rb") as f:
                for side, info in meta["sides"].items():
                    d = dict(info["ints"])
                    if self._side_is_tp(side):
                        model, cache = self.recurrent_sides[side]
                        handle = new_checkpoint_handle()
                        handles.append((model, id(cache), handle))
                        model.tp_dispatch_all(mp_disk_state_read,
                                              (id(cache), handle, path, data_offset + info["image_offset"], info["layers"]))
                        markers += [f"{path}.d{dev}" for dev in model.active_devices]
                        d["tp_handle"] = handle
                    else:
                        keymap = self._side_keymap(side)
                        inverse = {v: k for k, v in keymap.items()} if keymap else {}
                        for key, entry in info["layers"].items():
                            tensors = []
                            for t in entry["tensors"]:
                                tensor = host_pool.take(t["shape"], getattr(torch, t["dtype"]))
                                f.seek(data_offset + info["image_offset"] + t["offset"])
                                if f.readinto(_raw(tensor)) != t["nbytes"]:
                                    raise ValueError("short read")
                                tensors.append(tensor)
                            d[inverse.get(key, key)] = unpack_stash(entry["struct"], tensors)
                    sides_out[side] = d
            if markers:
                deadline = time.time() + 120
                while not all(os.path.exists(m) or os.path.exists(m + ".err") for m in markers):
                    if time.time() > deadline:
                        raise TimeoutError("ranks did not finish reading the checkpoint")
                    time.sleep(0.0005)
                failed = any(os.path.exists(m + ".err") for m in markers)
                for m in markers:
                    self._remove_tmp(m)
                    self._remove_tmp(m + ".err")
                if failed:
                    raise ValueError("a rank failed to read its part")
            if "__root__" in sides_out:
                stashed.update(sides_out["__root__"])
            else:
                stashed.update(sides_out)
        except (OSError, ValueError, AttributeError, KeyError, TimeoutError) as e:
            from ..cache.recurrent import mp_cache_recurrent_del
            for model, cache_id, handle in handles:
                try:
                    model.tp_dispatch_all(mp_cache_recurrent_del, (cache_id, handle))
                except Exception:
                    pass
            logger.warning(f"Disk cache tier: checkpoint {phash.hex()[:12]} unreadable ({e}), dropped")
            with self._lock:
                if phash in self.states:
                    self._drop("state", phash)
            return None
        with self._lock:
            self._touch(self.states[phash], "state", phash)
        self.metrics["state_reads"] += 1
        return stashed


    def submit_promotion(self, phash: bytes, slot: int, done, wait_events: list = ()):
        """Read a page into CPU-tier slot `slot`: this process's part on the reader pool after the stream
        copies `wait_events` mark have completed, the ranks' parts by the ranks themselves; done(tokens or
        None) runs on a background thread when everything has landed"""
        with self._lock:
            self._promotions += 1
        slab = self.cpu_tier.slab_view(slot)
        if self.tp_groups:
            r = self._read_tokens(phash)
            if r is None:
                with self._lock:
                    self._promotions -= 1
                done(None)
                return
            meta, tokens = r
            path = self._page_path(phash)
            markers = []
            for model, cache_ids, devices in self.tp_groups:
                model.tp_dispatch_all(mp_disk_read, (slot, path, meta["image_offset"]))
                markers += [f"{path}.d{dev}" for dev in devices]
            if slab is not None:
                # The local part is read here, synchronously: small next to the ranks' shards, and the
                # ranks are already on their way
                for ev in wait_events:
                    ev.synchronize()
                if self.read_page_into(phash, slab) is None:
                    tokens = None
            with self._cv:
                self._tp_waits.append(("read", phash, markers, (tokens, done)))
                self._cv.notify()
            return
        def task():
            try:
                for ev in wait_events:
                    ev.synchronize()
                tokens = self.read_page_into(phash, slab)
            except Exception:
                tokens = None
            finally:
                with self._lock:
                    self._promotions -= 1
            done(tokens)
        self._readers.submit(task)


    def prefetch(self, hashes: list, serial: int):
        """Promote the pages of a coming allocation from disk into the CPU tier ahead of time, in prefix order,
        up to the in-flight bound; the allocation promotes synchronously whatever this did not reach"""
        pt = self.pagetable
        for h in hashes:
            if self._promotions >= self.max_promotions:
                break
            if h not in self.pages or h in self.cpu_tier:
                continue
            if pt is not None and pt.get_live_page(h) is not None:
                continue
            self.cpu_tier.begin_promote(h, serial)


    # -- lifecycle ---------------------------------------------------------------------------------------------

    def flush(self):
        """Block until every queued write has reached the disk and the index"""
        ev = threading.Event()
        self._enqueue(("flush", ev))
        ev.wait()


    def close(self):
        """Drain the writer and reader pools; the directory stays for the next process"""
        if self._closed:
            return
        self.drain_ready_states()
        with self._lock:
            cands = list(self.state_candidates.items())
            self.state_candidates.clear()
        for k, stashed in cands:
            if k in self.pages or k in self.pending_pages:
                self.write_state(k, stashed)
            else:
                self.metrics["state_candidates_dropped"] += 1
        self._closed = True
        self._readers.shutdown(wait = True)
        ev = threading.Event()
        self._enqueue(("stop", ev))     # held behind any rank transfer still in flight
        ev.wait()
        self._writer.join()
        for stashed, fn in list(self._deferred_release.values()):
            fn(stashed)
        self._deferred_release.clear()
        self.pending_states.clear()
