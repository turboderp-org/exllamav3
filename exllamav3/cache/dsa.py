from __future__ import annotations
import torch
from ..constants import PAGE_SIZE
from .cache import Cache

"""
Per-job cache state for DSA (DeepSeek-V4-style hybrid sparse attention) layers.

Design principle: every piece of compressor bookkeeping is derivable from absolute position:
 
 - entry_count = position // m
 - buffer fill = position % m
 - overlap exists iff entry_count >= 1
 
The layer state holds only fixed-shape per-slot tensors and rewind is pure cursor arithmetic:

 - SWA ring: shifting linear buffer of raw roped K=V rows, page-aligned window_beg carried
   on the job state (identical mechanism to SWALayerState; window + overprovision slack
   gives guaranteed_rollback = PAGE_SIZE).
 - Compressed pool / indexer-key pool: append-only per-slot tensors indexed by entry number.
   Rewind is free.
 - Compressor sub-window buffers: rings of the last (PAGE_SIZE + m) PROJECTED (kv, gate)
   rows, indexed by absolute token position % ring size. Rewind = cursor move; the rows for
   the new partial window are still present for any rollback <= PAGE_SIZE.
 - CSA overlap (previous window's Ca slice): ring of the last few window-boundary snapshots
   indexed by (entry number - 1) % depth, depth sized for PAGE_SIZE worth of windows.

Pools are sized by the Cache's max_num_tokens budget: capacity = max_num_tokens // m entries
per slot. Memory example, V4-Flash at max_num_tokens 131072, one slot: CSA layer pool 32768 x
1KB = 32 MiB + indexer pool 8 MiB + rings ~2 MiB; HCA layer ~1.5 MiB; x active slots.
"""


class DSV4State:
    """
    Job-level state (recurrent_state_cls): position bookkeeping shared by every DSA layer
    of the model. Mirrors SWAState's contract; all tensor work lives in the layer states.
    """

    exported = False
    guaranteed_rollback = PAGE_SIZE

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
        self.position = position
        self.last_history = 0
        self.window_beg = position // PAGE_SIZE * PAGE_SIZE
        self.wshift = 0
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
        for k, l in self.cache.get_all_recurrent_layers().items():
            stashed[k] = l.stash(self.slot, self.position)
        return stashed

    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        self.window_beg = stashed["window_beg"]
        for k, l in self.cache.get_all_recurrent_layers().items():
            l.unstash(self.slot, stashed[k], self.position)

    def reset(self):
        self.position = 0
        self.window_beg = 0
        self.wshift = 0
        self.last_history = 0


class DSV4LayerState:
    """Per-layer, per-cache state tensors for one DSAttention module. Allocated on meta at
    construction, materialized by the module's load. Component set depends on layer type:
    every layer has the SWA ring; CSA/HCA add pools and compressor rings; CSA adds the
    indexer pool/rings and overlap snapshots."""

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

        self.pool_capacity = 0
        self.pool_c = self.pool_r = self.pool_idx = None
        self.comp_buf_kv = self.comp_buf_gate = None
        self.idx_buf_kv = self.idx_buf_gate = None
        self.comp_ovl = self.idx_ovl = None
        self._identity_bt = None

        if self.layer_type in ("csa", "hca"):
            m = module.compress_rate
            cap = -(-module.config.max_dsa_tokens // m)
            cap = -(-cap // PAGE_SIZE) * PAGE_SIZE
            self.pool_capacity = cap
            self.pool_c = mk(B, cap, self.D_c)
            self.pool_r = mk(B, cap, D_r)
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
                self.pool_idx = mk(B, cap, D_i)
                wi = module.indexer.wkv.out_features_unpadded
                self.idx_buf_kv = mk(B, self.buf_rows, wi)
                self.idx_buf_gate = mk(B, self.buf_rows, wi)
                self.idx_ovl = mk(B, self.ovl_depth, 2, m, D_i, dtype = torch.float)


    def _tensors(self):
        return [t for t in [
            self.ring, self.pool_c, self.pool_r, self.pool_idx,
            self.comp_buf_kv, self.comp_buf_gate, self.idx_buf_kv, self.idx_buf_gate,
            self.comp_ovl, self.idx_ovl,
        ] if t is not None]


    def _set_tensors(self, ts):
        names = ["ring", "pool_c", "pool_r", "pool_idx", "comp_buf_kv", "comp_buf_gate",
                 "idx_buf_kv", "idx_buf_gate", "comp_ovl", "idx_ovl"]
        it = iter(ts)
        for n in names:
            if getattr(self, n) is not None:
                setattr(self, n, next(it))


    def get_checkpoint_size(self):
        # Window slice + buffers + overlaps + pools at capacity (upper bound; actual stash
        # sizes shrink with position)
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
        m = self.module.compress_rate or 1
        out = [self.ring[slot, :min(self.ring_rows, position)].cpu()]
        if self.pool_c is not None:
            ec = position // m
            out.append(self.pool_c[slot, :ec].cpu())
            out.append(self.pool_r[slot, :ec].cpu())
            out.append(self.comp_buf_kv[slot].cpu())
            out.append(self.comp_buf_gate[slot].cpu())
            if self.pool_idx is not None:
                out.append(self.pool_idx[slot, :ec].cpu())
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
        if self.pool_c is not None:
            m = self.module.compress_rate
            ec = position // m
            self.pool_c[slot, :ec].copy_(next(it))
            self.pool_r[slot, :ec].copy_(next(it))
            self.comp_buf_kv[slot].copy_(next(it))
            self.comp_buf_gate[slot].copy_(next(it))
            if self.pool_idx is not None:
                self.pool_idx[slot, :ec].copy_(next(it))
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
