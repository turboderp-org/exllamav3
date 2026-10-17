from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
import torch
from ..loader.safetensors import DiskTensorHandle
from ..ext import exllamav3_ext as ext

"""
Row-addressable table behind an embedding module (token embeddings, hashed n-gram embeddings),
held in system RAM or streamed from disk. Streaming is a first-class mode: the table's tensors
are never loaded, the table keeps DiskTensorHandles from the loader and gathers only the rows a
forward pass actually touches.

Lookups run on the CPU, where the token ids already live (pinned in the generator, so nothing
round-trips through the device): the owning module resolves ids to unique table rows, the rows
are gathered with threaded preads (disk) or index_select (RAM) into a pinned staging set, and
one non-blocking H2D feeds the module's decode and the inverse gather.

The staging step depends only on the token ids, so it can run ahead of the forward on a worker
thread (prefetch()). Each queued prefetch owns one staging set; lookup() takes the set whose
staged ids equal the ones it was given and stages inline otherwise, so a prefetch for the wrong
ids can only cost time. A set's CUDA event marks the last lookup's uploads from it; the next
writer waits on it before reusing the buffers.
"""

PREFETCH_ENABLED = os.environ.get("EXL3_NGRAM_PREFETCH", "1") != "0"   # debug/A-B switch
PREFETCH_MIN_TOKENS = 256   # positions (bsz * seq) below which prefetch() declines (decode-sized)
MAX_PIN_SETS = 2            # staging sets: one with the last forward's uploads in flight, one being staged
INLINE_MAX_ROWS = 16        # rows of a table that gets read-ahead hints, below which a gather reads in line


class _PinSet:
    """Pinned staging buffers for one resolve + gather: unique row ids, the inverse map, the
    per-row head index and the stored rows. `held` while a queued prefetch or a running lookup
    owns it; `event` marks the last lookup's uploads from it."""

    def __init__(self):
        self.uids = None
        self.inverse = None
        self.heads = None
        self.packed = None
        self.event = None
        self.held = False

    def grow(self, n: int, row_words: int, row_dtype: torch.dtype):
        if self.uids is None or self.uids.numel() < n or self.packed.shape[1] != row_words \
                or self.packed.dtype != row_dtype:
            self.packed = torch.empty((n, row_words), dtype = row_dtype, pin_memory = True)
            self.inverse = torch.empty(n, dtype = torch.int64, pin_memory = True)
            self.heads = torch.empty(n, dtype = torch.int32, pin_memory = True)
            self.uids = torch.empty(n, dtype = torch.int64, pin_memory = True)


class RowTable:

    def __init__(self, key: str, keys: list[str]):
        self.key = key
        self.keys = keys
        # The table is always kept as its individual shard tensors: a list of CPU tensors (RAM)
        # or of DiskTensorHandles (disk). Never concatenated by reference (a cat would
        # transiently double the footprint); lookups route rows to shards instead. All shards
        # but the last hold rows_per_shard rows
        self.stores = None
        self.on_disk = False
        self.rows_per_shard = None
        self.num_rows = 0
        self.row_words = None       # elements per stored row
        self.row_dtype = None
        self._locked = []           # mlock()ed (address, size) ranges of the stores
        self._pins = []             # pinned staging sets (see _PinSet)
        self._pending = []          # queued prefetches, oldest first: {"ids", "pin", "future"}
        self._executor = None
        self.prefetch_stats = {"hit": 0, "miss": 0, "retired": 0}
        self.hinted = False         # the owner announces rows ahead of their lookup (advise())

    @staticmethod
    def find(stc, key: str, suffix: str) -> RowTable | None:
        """Table stored as <key>.<suffix> or split into <key>.shard_N.<suffix>"""
        keys = []
        while stc.has_tensor(f"{key}.shard_{len(keys)}.{suffix}"):
            keys.append(f"{key}.shard_{len(keys)}.{suffix}")
        if not keys and stc.has_tensor(f"{key}.{suffix}"):
            keys = [f"{key}.{suffix}"]
        return RowTable(key, keys) if keys else None

    def _set_stores(self, stores: list):
        # all shards but the last must hold the same row count (row -> shard routing is a plain
        # division); the last may be short
        shapes = [list(s.shape) for s in stores]
        assert all(s[0] == shapes[0][0] for s in shapes[:-1]) and shapes[-1][0] <= shapes[0][0], \
            f"shards of {self.key} must have equal row counts (last may be short)"
        assert all(s[1:] == shapes[0][1:] for s in shapes)
        self.stores = stores
        self.on_disk = isinstance(stores[0], DiskTensorHandle)
        self.rows_per_shard = shapes[0][0] if len(stores) > 1 else sum(s[0] for s in shapes)
        self.num_rows = sum(s[0] for s in shapes)
        self.row_words = shapes[0][1]
        self.row_dtype = stores[0].dtype

    def open(self, stc, stream_from_disk: bool, allow_bf16: bool = False, what: str | None = None,
             lock: bool = False):
        """lock: hold the table in RAM and mlock() it there (stream_from_disk must be False); the locked-memory
        limit is checked before anything loads"""
        keys = self.keys
        if lock:
            assert not stream_from_disk
            from ..util.memory import prepare_host_lock, lock_host_tensors
            nbytes = sum(stc.get_tensor_meta(k)[k]["n_bytes"] for k in keys)
            prepare_host_lock(nbytes, f"Locking {self.key} in RAM")
        handles = [stc.get_tensor_handle(k, optional = True) for k in keys] if stream_from_disk else [None]
        if all(h is not None for h in handles):
            # Shards that sit back-to-back in one file (the layout convert_ngram.py writes)
            # collapse into a single handle spanning the whole table: _gather_rows issues one
            # synchronous gather call per handle segment
            h0 = handles[0]
            rows = sum(h.num_rows for h in handles)
            if len(handles) > 1 and all(
                h.filename == h0.filename and h.row_bytes == h0.row_bytes and
                h.abs_offset == h0.abs_offset + s * h0.num_rows * h0.row_bytes
                for s, h in enumerate(handles)
            ):
                merged = DiskTensorHandle(
                    key = self.key, filename = h0.filename, abs_offset = h0.abs_offset,
                    shape = [rows, *h0.row_shape], dtype = h0.dtype)
                stc.find_stc(keys[0]).disk_handles.append(merged)   # closed with the collection
                handles = [merged]
            if os.name == "nt":
                # Release the loader's handles to the table files now
                for f in set(h.filename for h in handles):
                    stc.release_file(f)
            self._set_stores(handles)
        elif len(keys) == 1:
            # no file entry: the tensor is already in host memory (conversion)
            meta = stc.find_stc(keys[0]).get_tensor_meta(keys[0])
            if what and meta is not None:
                from ..util.memory import check_host_memory
                check_host_memory(meta[keys[0]]["n_bytes"], what)
            self._set_stores([stc.get_tensor(keys[0], "cpu", allow_bf16 = allow_bf16, no_defer = True)])
        else:
            # Sharded table: one contiguous slab, each shard copied into its slice as it loads
            slab = None
            r0 = 0
            for k in keys:
                t = stc.get_tensor(k, "cpu", allow_bf16 = allow_bf16, no_defer = True)
                if slab is None:
                    rows = sum(stc.get_tensor_meta(k_)[k_]["shape"][0] for k_ in keys)
                    from ..util.memory import check_host_memory
                    check_host_memory(rows * t[0].numel() * t.element_size(), what or f"{self.key} held in RAM")
                    slab = torch.empty((rows, *t.shape[1:]), dtype = t.dtype)
                slab[r0 : r0 + t.shape[0]].copy_(t)
                r0 += t.shape[0]
                del t
            self._set_stores([slab])
        if lock:
            self._locked = lock_host_tensors(self.stores, f"Locking {self.key} in RAM")

    def close(self):
        self.drain_prefetch()       # queued workers still read the table; first
        if self._locked:
            from ..util.memory import unlock_host_ranges
            unlock_host_ranges(self._locked)
            self._locked = []
        self.stores = None
        self._pins = []

    def export(self, stc) -> dict:
        """
        Tensor-parallel: the table itself never travels. Every rank streams rows from disk
        through its own handles, so the export carries the shard locations. A table held in RAM
        here is still streamed from disk by the workers.
        """
        handles = [stc.get_tensor_handle(k) for k in self.keys]
        return {
            "key": self.key,
            "keys": self.keys,
            "handles": [(h.key, h.filename, h.abs_offset, list(h.shape), str(h.dtype)) for h in handles],
        }

    @staticmethod
    def from_export(exported: dict) -> RowTable:
        table = RowTable(exported["key"], exported["keys"])
        table._set_stores([
            DiskTensorHandle(key = k, filename = fn, abs_offset = off, shape = shape,
                             dtype = getattr(torch, d.split(".")[1]))
            for k, fn, off, shape, d in exported["handles"]
        ])
        return table

    def fetch(self, uids_cpu: torch.Tensor) -> torch.Tensor:
        """Gather rows of the backing store to CPU, in the order given. Reference form of the
        gather; lookup() runs the fast path."""
        def gather(s, local):
            return self.stores[s].read_rows(local) if self.on_disk else self.stores[s].index_select(0, local)

        if len(self.stores) == 1:
            return gather(0, uids_cpu)
        shard = uids_cpu // self.rows_per_shard
        local = uids_cpu - shard * self.rows_per_shard
        out = None
        for s in shard.unique().tolist():
            m = shard == s
            rows = gather(s, local[m])
            if out is None:
                out = torch.empty((uids_cpu.numel(), *rows.shape[1:]), dtype = rows.dtype)
            out[m] = rows
        return out

    def advise(self, uids: list[int]):
        """Hint that these rows are about to be read, for a table streamed from disk: the OS
        starts reading them into its page cache and returns at once. Issued when the next ids
        become known ahead of their forward (the generator, right after sampling), the read then
        overlaps the host work in between."""
        if not self.on_disk or not hasattr(os, "posix_fadvise"):
            return
        self.hinted = True
        for uid in uids:
            if not 0 <= uid < self.num_rows:
                continue            # (multimodal placeholder ids have no row)
            s, local = divmod(uid, self.rows_per_shard)
            h = self.stores[s]
            os.posix_fadvise(h._ensure_open(), h.abs_offset + local * h.row_bytes, h.row_bytes,
                             os.POSIX_FADV_WILLNEED)

    def drain_prefetch(self):
        for e in list(self._pending):
            self._retire(e)
        if self._executor is not None:
            self._executor.shutdown(wait = True)
            self._executor = None

    def _forget(self, entry: dict):
        # by identity: list.remove would compare the entries' id tensors
        self._pending = [e for e in self._pending if e is not entry]

    def _retire(self, entry: dict):
        # Drop a queued prefetch that no lookup will take. Its worker may still be writing the
        # staging set, so wait it out (a cold gather, at most) unless it hasn't started
        f = entry["future"]
        if not f.cancel():
            f.exception()       # an error of its own belongs to no lookup
        self._forget(entry)
        self.prefetch_stats["retired"] += 1
        entry["pin"].held = False

    def _acquire_pin(self, n: int) -> _PinSet:
        """A staging set not held by a queued prefetch, grown to n rows."""
        free = [p for p in self._pins if not p.held]
        if not free and len(self._pins) < MAX_PIN_SETS:
            free = [_PinSet()]
            self._pins += free
        if not free:
            # every set is held by a queued prefetch: retire one, preferably one whose staging
            # already finished (a stale guess), else the oldest
            done = [e for e in self._pending if e["future"].done()]
            self._retire(done[0] if done else self._pending[0])
            free = [p for p in self._pins if not p.held]
        pin = free[0]
        pin.grow(n, self.row_words, self.row_dtype)
        pin.held = True
        return pin

    @torch.inference_mode()
    def _stage(self, ids: torch.Tensor, pin: _PinSet, resolve) -> tuple:
        """Resolve + gather for ids into pin; returns the number of rows staged and whether they
        are in output order. Runs on the prefetch worker or inline (inference mode is
        thread-local, and the staging buffers are inference tensors)."""
        if pin.event is not None:
            # the previous lookup's non_blocking uploads read this set; the generator issues
            # chunk forwards back to back with no host sync, so wait before rewriting it
            pin.event.synchronize()
        uids, in_order = resolve(ids, pin)
        U = uids.numel()
        self._gather_rows(uids, pin.packed[:U])
        return U, in_order

    def _match(self, ids: torch.Tensor) -> dict | None:
        for e in self._pending:
            if e["ids"].shape == ids.shape and torch.equal(e["ids"], ids):
                return e
        return None

    def _gather_rows(self, uids: torch.Tensor, out: torch.Tensor):
        """Gather rows into the pinned staging buffer, routing shard segments (contiguous in
        the list, which must be sorted for a table of several shards) to their tensor/handle."""
        stores = self.stores
        if len(stores) > 1:
            # One searchsorted for every shard boundary
            bounds = torch.arange(1, len(stores), dtype = torch.int64) * self.rows_per_shard
            cuts = torch.searchsorted(uids, bounds).tolist() + [uids.numel()]
        else:
            cuts = [uids.numel()]
        i0 = 0
        for s, store in enumerate(stores):
            i1 = cuts[s]
            if i1 > i0:
                seg = uids[i0 : i1]
                base = s * self.rows_per_shard
                if self.on_disk:
                    # (index_select checks the range of a table in RAM)
                    lo, hi = torch.aminmax(seg)
                    assert base <= lo.item() and hi.item() < base + store.num_rows, \
                        f"Row index out of range for {self.key}"
                    if self.hinted and i1 - i0 <= INLINE_MAX_ROWS:
                        # Rows announced ahead are in the page cache by now: one read each on
                        # this thread, which the gather's worker pool only slows down
                        for i in range(i0, i1):
                            ext.ngram_gather_cpu(store._ensure_open(), store.abs_offset,
                                                 store.row_bytes, uids[i : i + 1], base, out[i : i + 1])
                    else:
                        ext.ngram_gather_cpu(store._ensure_open(), store.abs_offset,
                                             store.row_bytes, seg.contiguous(), base, out[i0 : i1])
                else:
                    torch.index_select(store, 0, seg - base if base else seg, out = out[i0 : i1])
            i0 = i1

    def prefetch(self, ids: torch.Tensor, n: int, resolve):
        """
        Stage the rows for a coming lookup over `ids`, the exact ids that lookup() will receive,
        on a worker thread; n is the number of rows they resolve to. Decode-sized inputs stage
        inline faster than the thread hop, so callers leave those out (PREFETCH_MIN_TOKENS); ids
        already queued are ignored.
        """
        if not PREFETCH_ENABLED or self.stores is None:
            return
        ids = ids.to("cpu", torch.int64).contiguous().clone()
        if self._match(ids) is not None:
            return
        if self.on_disk:
            for h in self.stores:
                h._ensure_open()    # lazy open isn't thread-safe; do it here, not on the worker
        pin = self._acquire_pin(n)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "row_prefetch")
        self._pending.append({
            "ids": ids,
            "pin": pin,
            "future": self._executor.submit(self._stage, ids, pin, resolve),
        })

    def lookup(
        self,
        ids: torch.Tensor,
        n: int,
        resolve,
        decode,
        device: torch.device,
        synced: bool = False,
    ) -> torch.Tensor:
        """
        ids: contiguous int64 CPU ids
        n: number of table rows the ids resolve to
        resolve(ids, pin) -> (uids, in_order): the rows to gather, either the n output rows in
            output order, or the sorted unique rows with the row of each output written to
            pin.inverse[:n] (pin.uids and pin.heads are free for the module's use)
        decode(rows, pin, U): stored rows on the device -> (U, dim) decoded rows
        synced: the caller guarantees a sync point before the next lookup (the generator's
            decode loop), so the uploads need no event to guard the staging set
        Returns the (n, dim) decoded rows on the device.
        """
        entry = self._match(ids)
        if entry is not None:
            self._forget(entry)
            self.prefetch_stats["hit"] += 1
            pin = entry["pin"]
        else:
            self.prefetch_stats["miss"] += 1
            pin = self._acquire_pin(n)
        try:
            U, in_order = entry["future"].result() if entry is not None else self._stage(ids, pin, resolve)
            out = decode(pin.packed[:U].to(device, non_blocking = True), pin, U)
            if not in_order:
                out = out.index_select(0, pin.inverse[:n].to(device, non_blocking = True))
            elif not out.is_cuda:
                out = out.clone()       # the decoded rows may still be the staging buffer itself
            if out.is_cuda and not synced:
                if pin.event is None:
                    pin.event = torch.cuda.Event()
                pin.event.record(torch.cuda.current_stream(out.device))
        except BaseException:
            # a failed gather, upload or decode must not keep its staging set; an interrupted wait
            # leaves the set with its prefetch, whose worker may still be writing it
            if entry is not None and not entry["future"].done():
                self._pending.append(entry)
            else:
                pin.held = False
            raise
        pin.held = False
        return out
