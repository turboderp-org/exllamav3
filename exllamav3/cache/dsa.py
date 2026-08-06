from __future__ import annotations
from dataclasses import dataclass
import torch
from ..constants import PAGE_SIZE
from .cache import Cache, CacheLayer
from .recurrent import new_checkpoint_handle, mp_cache_recurrent_stash, mp_cache_recurrent_unstash

"""
Cache state for DSA (DeepSeek-V4-style hybrid sparse attention) layers, split along the
framework's paged/recurrent line:

 - The compressed pool, rope pool and indexer-key pool are PAGED cache layers
   (CacheLayer_dsa): append-only, entry-indexed by position // m, immutable once written --
   semantically a KV cache growing at rate 1/m. Pool entries alias the token page table:
   token page i holds entries [i * epp, (i + 1) * epp), epp = PAGE_SIZE // m, addressed
   through the job's ordinary block table. Capacity therefore follows the Cache's
   max_num_tokens budget exactly (any mix of jobs fits iff total tokens fit), and prefix
   sharing, eviction, defragmentation and the CPU tier apply to pool pages for free.

 - Everything else is fixed-shape per-slot recurrent state (DSV4LayerState), with all
   compressor bookkeeping derived from absolute position (entry_count = position // m,
   buffer fill = position % m, overlap exists iff entry_count >= 1), so rewind is pure
   cursor arithmetic:

    - SWA ring: shifting linear buffer of raw roped K=V rows, page-aligned window_beg
      carried on the job state (identical mechanism to SWALayerState; window +
      overprovision slack gives guaranteed_rollback = PAGE_SIZE).
    - Compressor sub-window buffers: rings of the last (PAGE_SIZE + m) PROJECTED (kv, gate)
      rows, indexed by absolute token position % ring size. Rewind = cursor move; the rows
      for the new partial window are still present for any rollback <= PAGE_SIZE.
    - CSA overlap (previous window's Ca slice): ring of the last few window-boundary
      snapshots indexed by (entry number - 1) % depth, depth sized for PAGE_SIZE worth of
      windows.

Checkpoints (stash) carry only the recurrent residue; pool persistence across
pause/resume is governed by the same page-anchoring logic as K/V pages on hybrid
recurrent models.
"""


class CacheLayer_dsa(CacheLayer):
    """
    Paged pool storage for one CSA/HCA DSAttention layer: pool_c (compressed KV, nope part),
    pool_r (rope part) and, for CSA, pool_idx (indexer keys), each shaped
    (num_pages, PAGE_SIZE // m, D) so that one token page holds the pool entries produced by
    exactly that page's tokens. Tensors are page-major, so the defragmenter's rotation and
    the CPU tier's per-page slabs apply unchanged. Pools are always fp16 in v1; a quantized
    Cache quantizes only its transformer/MLA layers, like recurrent state.
    """

    def __init__(
        self,
        config: Config | None,
        attention,
        cache_id: int,
        max_num_tokens: int,
        **kwargs
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        m = attention.compress_rate
        assert m and PAGE_SIZE % m == 0, f"compress_rate {m} must divide {PAGE_SIZE}"
        self.compress_rate = m
        self.epp = PAGE_SIZE // m
        self.num_pages = max_num_tokens // PAGE_SIZE
        self.capacity = self.num_pages * self.epp
        D = attention.head_dim
        D_r = attention.rope_head_dim
        self.D_c = D - D_r
        self.D_r = D_r
        self.D_i = attention.index_head_dim if attention.layer_type == "csa" else 0
        self.pool_c = None
        self.pool_r = None
        self.pool_idx = None
        self.device = None
        self._slot_bt = None

    def alloc(self, device: torch.device):
        self.device = device
        self.pool_c = torch.zeros((self.num_pages, self.epp, self.D_c), dtype = torch.half, device = device)
        self.pool_r = torch.zeros((self.num_pages, self.epp, self.D_r), dtype = torch.half, device = device)
        if self.D_i:
            self.pool_idx = torch.zeros((self.num_pages, self.epp, self.D_i), dtype = torch.half, device = device)

    def free(self):
        self.device = None
        self.pool_c = self.pool_r = self.pool_idx = None
        self._slot_bt = None

    def get_kv(self, cache_seqlens, block_table, sliding_window = -1):
        return None, None

    def update_kv(self, cache_seqlens, block_table, k, v, length):
        pass

    def update_kv_direct(self, cache_seqlens, block_table, k, v, length):
        pass

    def copy_page(self, source: CacheLayer_dsa, from_page: int, to_page: int, num_tokens: int):
        ne = min(num_tokens // self.compress_rate, self.epp)
        if ne <= 0:
            return
        self.pool_c[to_page, :ne].copy_(source.pool_c[from_page, :ne], non_blocking = True)
        self.pool_r[to_page, :ne].copy_(source.pool_r[from_page, :ne], non_blocking = True)
        if self.pool_idx is not None:
            self.pool_idx[to_page, :ne].copy_(source.pool_idx[from_page, :ne], non_blocking = True)

    def get_tensors(self):
        return [t for t in [self.pool_c, self.pool_r, self.pool_idx] if t is not None]

    def storage_size(self):
        n = self.num_pages * self.epp * (self.D_c + self.D_r + self.D_i)
        return n * torch.half.itemsize

    def overhead_size(self):
        return 0

    def slot_bt(self, num_slots: int) -> torch.Tensor:
        """
        Slot-partitioned identity block table for direct-forward use without a page table
        (tests, benchmarks): slot s owns pages [s * pps, (s + 1) * pps),
        pps = num_pages // num_slots, reproducing isolated per-slot pool semantics.
        """
        if self._slot_bt is None or self._slot_bt.shape[0] != num_slots:
            pps = self.num_pages // num_slots
            assert pps > 0, "CacheLayer_dsa: cache too small for slot-partitioned fallback"
            self._slot_bt = torch.arange(num_slots * pps, dtype = torch.int32,
                                         device = self.device).view(num_slots, pps)
        return self._slot_bt

    def tp_export(self, plan):
        # Pools are shared-KV (MQA) state: replicated per rank, shapes derived from the
        # (replicated) compressor fields of the shard module passed as `attention` on import
        return {
            "cls": CacheLayer_dsa,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
            }
        }


@dataclass
class DSV4ExportedState:
    cache: int
    slot: int
    position: int
    window_beg: int
    serial: int
    wshift: int = 0
    last_history: int = 0


class DSV4State:
    """
    Job-level state (recurrent_state_cls): position bookkeeping shared by every DSA layer
    of the model. Mirrors SWAState's contract; all tensor work lives in the layer states.
    """

    exported = False
    guaranteed_rollback = PAGE_SIZE

    _serial = 0    # distinguishes successive jobs on the same slot (BC block-table refresh)

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict | None = None,
        test_state: bool = False,
    ):
        assert test_state or position == 0 or stashed is not None
        self.cache = cache
        self.slot = slot
        self.serial = DSV4State._serial
        DSV4State._serial += 1
        self.position = position
        self.last_history = 0
        self.window_beg = position // PAGE_SIZE * PAGE_SIZE
        self.wshift = 0
        if stashed is not None:
            self.unstash(stashed)
        self.checkpoint_size = sum(
            l.get_checkpoint_size() for l in cache.get_all_recurrent_layers().values()
        )
        if clear and stashed is None:
            for l in cache.get_all_recurrent_layers().values():
                l.clear(slot)

    def free(self):
        self.cache.release_state(self)

    def rewind(self, num_tokens: int):
        assert num_tokens <= self.rollback_capacity(), \
            f"DSV4State: rewind {num_tokens} exceeds capacity {self.rollback_capacity()}"
        self.position -= num_tokens
        self.last_history = 0

    def rollback_capacity(self):
        return self.position - self.window_beg

    def post_advance(self):
        self.window_beg += self.wshift
        self.wshift = 0

    def stash(self):
        stashed = {
            "position": self.position,
            "window_beg": self.window_beg,
            "checkpoint_size": self.checkpoint_size,
        }
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                stashed[k] = l.stash(self.slot, self.position)
        else:
            cp_handle = new_checkpoint_handle()
            self.cache.model.tp_dispatch_all(
                mp_cache_recurrent_stash, (id(self.cache), cp_handle, self.slot, self.position))
            stashed["tp_handle"] = cp_handle
        return stashed

    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        self.window_beg = stashed["window_beg"]
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                l.unstash(self.slot, stashed[k], self.position)
        else:
            cp_handle = stashed["tp_handle"]
            self.cache.model.tp_dispatch_all(
                mp_cache_recurrent_unstash, (id(self.cache), cp_handle, self.slot, self.position))

    def tp_export(self):
        return DSV4ExportedState(
            cache = id(self.cache),
            slot = self.slot,
            position = self.position,
            window_beg = self.window_beg,
            serial = self.serial,
        )

    def tp_readback(self, exported: DSV4ExportedState):
        # The forward decides the per-step ring shift; post_advance() applies it parent-side
        self.wshift = exported.wshift

    def reset(self):
        self.position = 0
        self.window_beg = 0
        self.wshift = 0
        self.last_history = 0


class DSV4LayerState:
    """Per-layer, per-cache recurrent state tensors for one DSAttention module. Allocated on
    meta at construction, materialized by the module's load. Component set depends on layer
    type: every layer has the SWA ring; CSA/HCA add compressor sub-window rings; CSA adds
    the indexer rings and overlap snapshots. The pools live in the paged CacheLayer_dsa."""

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.cache_id = cache_id
        self.max_batch_size = max_batch_size
        self.max_history = max_history
        self.device = None

        D = module.head_dim
        D_r = module.rope_head_dim
        self.layer_type = module.layer_type
        self.window = module.sliding_window
        overp = 2 * PAGE_SIZE
        self.ring_rows = -(-(self.window + overp) // PAGE_SIZE) * PAGE_SIZE

        mk = lambda *shape, dtype = torch.half: torch.zeros(shape, dtype = dtype, device = "meta")
        B = max_batch_size
        self.D_c = D - D_r
        self.D_r = D_r
        self.ring = mk(B, self.ring_rows, D)

        self.comp_buf_kv = self.comp_buf_gate = None
        self.idx_buf_kv = self.idx_buf_gate = None
        self.comp_ovl = self.idx_ovl = None

        if self.layer_type in ("csa", "hca"):
            m = module.compress_rate
            w = module.compressor.wkv.out_features_unpadded
            self.buf_rows = PAGE_SIZE + m
            self.comp_buf_kv = mk(B, self.buf_rows, w)
            self.comp_buf_gate = mk(B, self.buf_rows, w)
            if self.layer_type == "csa":
                # overlap snapshots: one per emitted window; PAGE_SIZE//m + slack of them.
                # fp32: the saved gate slice carries the fp32 position bias
                self.ovl_depth = PAGE_SIZE // m + 2
                self.comp_ovl = mk(B, self.ovl_depth, 2, m, D, dtype = torch.float)
                D_i = module.index_head_dim
                wi = module.indexer.wkv.out_features_unpadded
                self.idx_buf_kv = mk(B, self.buf_rows, wi)
                self.idx_buf_gate = mk(B, self.buf_rows, wi)
                self.idx_ovl = mk(B, self.ovl_depth, 2, m, D_i, dtype = torch.float)


    def _tensors(self):
        return [t for t in [
            self.ring,
            self.comp_buf_kv, self.comp_buf_gate, self.idx_buf_kv, self.idx_buf_gate,
            self.comp_ovl, self.idx_ovl,
        ] if t is not None]


    def _set_tensors(self, ts):
        names = ["ring", "comp_buf_kv", "comp_buf_gate",
                 "idx_buf_kv", "idx_buf_gate", "comp_ovl", "idx_ovl"]
        it = iter(ts)
        for n in names:
            if getattr(self, n) is not None:
                setattr(self, n, next(it))


    def get_checkpoint_size(self):
        # Window slice + buffers + overlaps (upper bound; the ring slice shrinks below the
        # window at low positions). Pools are paged cache state and not part of checkpoints
        n = self.window * self.module.head_dim * 2
        for t in self._tensors()[1:]:
            n += t[0].numel() * t.element_size()
        return n


    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in self._tensors())


    def alloc(self, device):
        self.device = device
        self._set_tensors([torch.zeros_like(t, device = device) for t in self._tensors()])


    def free(self):
        self.device = None
        self._set_tensors([torch.zeros_like(t, device = "meta") for t in self._tensors()])


    def clear(self, idx):
        for t in self._tensors():
            t[idx].zero_()


    def rewind(self, slot, last_history, num_tokens):
        pass  # all bookkeeping is position-derived; stale rows are overwritten on re-advance


    def stash(self, slot, position):
        out = [self.ring[slot, :min(self.ring_rows, position)].cpu()]
        if self.comp_buf_kv is not None:
            out.append(self.comp_buf_kv[slot].cpu())
            out.append(self.comp_buf_gate[slot].cpu())
            if self.idx_buf_kv is not None:
                out.append(self.idx_buf_kv[slot].cpu())
                out.append(self.idx_buf_gate[slot].cpu())
                out.append(self.comp_ovl[slot].cpu())
                out.append(self.idx_ovl[slot].cpu())
        return out


    def unstash(self, slot, stashed, position):
        it = iter(stashed)
        ring = next(it)
        self.ring[slot].zero_()  # never leave stale rows below the restored window
        self.ring[slot, :ring.shape[0]].copy_(ring)
        if self.comp_buf_kv is not None:
            self.comp_buf_kv[slot].copy_(next(it))
            self.comp_buf_gate[slot].copy_(next(it))
            if self.idx_buf_kv is not None:
                self.idx_buf_kv[slot].copy_(next(it))
                self.idx_buf_gate[slot].copy_(next(it))
                self.comp_ovl[slot].copy_(next(it))
                self.idx_ovl[slot].copy_(next(it))


    def tp_export(self, plan):
        return {
            "cls": DSV4LayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            },
        }


class CacheLayer_dspark(CacheLayer):
    """
    Paged storage for one DSpark drafter block's main-kv rows: one (head_dim,) fp16 row per
    TARGET position, derived from the trunk's tap states by update_kv_from_target. Rides
    the standard page table (rows for position p live at block_table[p // PAGE_SIZE],
    p % PAGE_SIZE), so draft and target cache layouts stay aligned for free. The drafter
    only ever reads the trailing attention window.
    """

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        **kwargs
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        assert max_num_tokens % PAGE_SIZE == 0
        self.num_pages = max_num_tokens // PAGE_SIZE
        self.width = attention.head_dim
        self.kv = None
        self.device = None

    def alloc(self, device: torch.device):
        self.device = device
        self.kv = torch.zeros((self.num_pages, PAGE_SIZE, self.width), dtype = torch.half,
                              device = device)

    def free(self):
        self.device = None
        self.kv = None

    def get_kv(self, cache_seqlens, block_table, sliding_window = -1):
        return self.kv, None

    def update_kv(self, cache_seqlens, block_table, k, v, length):
        pass

    def update_kv_direct(self, cache_seqlens, block_table, k, v, length):
        pass

    def write_rows(self, rows: torch.Tensor, cache_seqlens, block_table):
        """rows (bsz, s, width) fp16; write at positions cache_seqlens[r] .. + s per row.
        cache_seqlens and block_table are device tensors (paged scatter kernel)."""
        from ..ext import exllamav3_ext as ext
        ext.dspark_write_rows(rows.contiguous(), self.kv, block_table, cache_seqlens)

    def copy_page(self, source, from_page: int, to_page: int, num_tokens: int):
        self.kv[to_page, :num_tokens].copy_(source.kv[from_page, :num_tokens], non_blocking = True)

    def get_tensors(self):
        return [self.kv]

    def storage_size(self):
        return self.num_pages * PAGE_SIZE * self.width * torch.half.itemsize

    def overhead_size(self):
        return 0

    def tp_export(self, plan):
        raise NotImplementedError("Tensor-parallel loading is not supported for DSpark layers")
