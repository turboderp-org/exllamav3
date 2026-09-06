from __future__ import annotations
import os
from typing_extensions import override
import math
import torch
import torch.nn.functional as F
from .module import Module
from .linear import Linear
from .rmsnorm import RMSNorm
from ..model.config import Config

"""
QSA (Qwen sparse attention) indexer: selects which tokens each query may attend to, at 4-token
block granularity (Qwen3.8-Flash-Next full-attention layers).

Per token, the indexer projects a raw 128-D key (cached unnormed and unroped) and index_n_heads
query heads (RMS-normed, partially roped at the query position). Every COMPLETE compress_ratio
block of raw keys is mean-pooled in fp32, RMS-normed, and roped at the block's START position.
Deterministic once the block completes, so pooled keys are cacheable/incremental. Block scores are
relu(q dot k) summed over the index heads / sqrt(head_dim); each query keeps the top
token_budget / compress_ratio blocks plus, always, the incomplete tail block. The selection is
ANDed into the causal mask of the main attention.

This module implements the eager batch path (no padding, contiguous positions). The rope tables
(cos/sin over the full history, matching the main attention's partial-rotary width) are supplied
by the caller.
"""


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Partial rotary on the first cos.shape[-1] dims of x. cos/sin broadcast over head dims.
    """
    r = cos.shape[-1]
    x_rope, x_pass = x[..., :r], x[..., r:]
    h = r // 2
    rot = torch.cat((-x_rope[..., h:], x_rope[..., :h]), dim = -1)
    return torch.cat((x_rope * cos + rot * sin, x_pass), dim = -1)


class QSAIndexer(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        n_heads: int,
        kv_heads: int,
        head_dim: int,
        token_budget: int,
        compress_ratio: int,
        rms_norm_eps: float,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        qbits_key: str = "bits",
    ):
        super().__init__(config = config, key = key, qmap = None)
        assert kv_heads == 1, "QSAIndexer assumes a single raw key head"
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.token_budget = token_budget
        self.compress_ratio = compress_ratio
        self.block_topk = token_budget // compress_ratio
        self.scale = 1.0 / math.sqrt(head_dim)

        self.index_qk_proj = Linear(
            config = config,
            key = f"{key}.index_qk_proj",
            in_features = hidden_size,
            out_features = (n_heads + kv_heads) * head_dim,
            qmap = qmap + ".input" if qmap is not None else None,
            out_dtype = torch.half,
            qbits_key = qbits_key,
        )
        self.q_layernorm = RMSNorm(config, f"{key}.q_layernorm", rms_norm_eps, constant_bias = 1.0)
        self.k_layernorm = RMSNorm(config, f"{key}.k_layernorm", rms_norm_eps, constant_bias = 1.0)
        self.register_submodule(self.index_qk_proj)
        self.register_submodule(self.q_layernorm)
        self.register_submodule(self.k_layernorm)

    @override
    def optimizer_targets(self):
        return self.index_qk_proj.optimizer_targets()

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        raise RuntimeError("QSAIndexer is not a standalone graph module; use project()/select()")

    def project(self, x: torch.Tensor, rope, params: dict, position: int = 0):
        """
        x: (bsz, seq, hidden); queries roped in-kernel at position + token index (sin/cos
        computed on the fly from inv_freq -- no cached tables). Returns (q (bsz, seq, n_heads,
        head_dim) normed + roped, raw_k (bsz, seq, head_dim) unnormed/unroped -- this is what
        the indexer key cache stores).
        """
        bsz, seq, _ = x.shape
        qk = self.index_qk_proj.forward(x.contiguous(), params)
        q, raw_k = qk.split([self.n_heads * self.head_dim, self.head_dim], dim = -1)
        q = self.q_layernorm.forward(q.reshape(bsz, seq, self.n_heads, self.head_dim).contiguous(), params)
        rope.apply(q, None, position, None, None, True)
        return q, raw_k

    def pool_keys(self, raw_k: torch.Tensor, rope, params: dict):
        """
        raw_k: (bsz, total_len, head_dim) full raw key history. Returns pooled block keys
        (bsz, num_blocks, head_dim): fp32 mean over each complete block, normed, roped in-kernel
        at the block START position (a stride-cr position_ids ramp).
        """
        bsz, total, dk = raw_k.shape
        cr = self.compress_ratio
        nb = total // cr
        if nb == 0:
            return raw_k.new_zeros((bsz, 0, dk))
        pooled = raw_k[:, : nb * cr].view(bsz, nb, cr, dk).float().mean(dim = 2).to(raw_k.dtype)
        pooled = self.k_layernorm.forward(pooled, params)
        starts = (torch.arange(nb, dtype = torch.int32, device = raw_k.device) * cr) \
            .unsqueeze(0).expand(bsz, nb).contiguous()
        rope.apply(pooled.view(bsz, nb, 1, dk), None, 0, None, starts, True)
        return pooled

    def project_ref(self, x: torch.Tensor, cos_q: torch.Tensor, sin_q: torch.Tensor, params: dict):
        """
        x: (bsz, seq, hidden); cos_q/sin_q: rope tables for the query positions, broadcastable to
        (bsz, seq, 1, rope_dim).
        Returns:
            q (bsz, seq, n_heads, head_dim) normed + roped
            raw_k (bsz, seq, head_dim) unnormed/unroped (this is what the indexer key cache stores).
        """
        bsz, seq, _ = x.shape
        qk = self.index_qk_proj.forward(x.contiguous(), params)
        q, raw_k = qk.split([self.n_heads * self.head_dim, self.head_dim], dim = -1)
        q = self.q_layernorm.forward(q.reshape(bsz, seq, self.n_heads, self.head_dim).contiguous(), params)
        q = _rope(q, cos_q.unsqueeze(-2), sin_q.unsqueeze(-2))
        return q, raw_k

    def pool_keys_ref(self, raw_k: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor, params: dict):
        """
        raw_k: (bsz, total_len, head_dim) full raw key history; cos_full/sin_full: rope tables for
        positions 0..total_len-1, broadcastable to (bsz, total_len, rope_dim).
        Returns:
             pooled block keys (bsz, num_blocks, head_dim): fp32 mean over each complete block, normed,
            roped at the block start position.
        """
        bsz, total, dk = raw_k.shape
        cr = self.compress_ratio
        nb = total // cr
        if nb == 0:
            return raw_k.new_zeros((bsz, 0, dk))
        pooled = raw_k[:, : nb * cr].view(bsz, nb, cr, dk).float().mean(dim = 2).to(raw_k.dtype)
        pooled = self.k_layernorm.forward(pooled, params)
        starts = torch.arange(0, nb * cr, cr, device = raw_k.device)
        return _rope(pooled, cos_full[..., starts, :], sin_full[..., starts, :])

    def block_scores(self, q: torch.Tensor, pooled: torch.Tensor):
        """
        q (bsz, seq, H, dk), pooled (bsz, nb, dk) -> (bsz, seq, nb) fp32
        """
        s = torch.einsum("bshd,bnd->bshn", q.float(), pooled.float())
        return F.relu(s).sum(dim = 2) * self.scale

    def token_mask(self, q: torch.Tensor, pooled: torch.Tensor, past_len: int, total_len: int):
        """
        Selection mask for causal, unpadded attention: (bsz, seq, total_len) bool, True = may
        attend (already includes causality). past_len = absolute position of the first query
        """
        bsz, seq = q.shape[:2]
        cr = self.compress_ratio
        dev = q.device
        abs_pos = past_len + torch.arange(seq, device = dev)                     # (seq,)
        kv_pos = torch.arange(total_len, device = dev)                           # (total,)
        nb_q = (abs_pos + 1) // cr                                               # complete blocks per query
        nb = pooled.shape[1]

        # tail (incomplete visible block) and causality
        mask = (kv_pos.unsqueeze(0) >= nb_q.unsqueeze(1) * cr) & \
               (kv_pos.unsqueeze(0) <= abs_pos.unsqueeze(1))                     # (seq, total)
        mask = mask.unsqueeze(0).expand(bsz, -1, -1).contiguous()

        if nb > 0:
            scores = self.block_scores(q, pooled)                                # (bsz, seq, nb)
            block_j = torch.arange(nb, device = dev)
            scores = scores.masked_fill(block_j.unsqueeze(0) >= nb_q.unsqueeze(1), -torch.inf)
            k = min(self.block_topk, nb)
            sel = scores.topk(k, dim = -1).indices                               # (bsz, seq, k)
            block_mask = torch.zeros((bsz, seq, nb), dtype = torch.bool, device = dev)
            block_mask.scatter_(-1, sel, True)
            # blocks selected past a query's visible range only cover future tokens (or the
            # force-included tail block), so causality below keeps the mask correct
            token_sel = block_mask.repeat_interleave(cr, dim = -1)
            if token_sel.shape[-1] < total_len:
                token_sel = F.pad(token_sel, (0, total_len - token_sel.shape[-1]))
            else:
                token_sel = token_sel[..., :total_len]
            mask |= token_sel & (kv_pos.view(1, 1, -1) <= abs_pos.view(1, -1, 1))
        return mask

    # ---- selection kernels -------------------------------------------------------------------
    # Selection = score -> top-k -> expand, three launches per (sequence, row slab), the same
    # kernels the BC sparse-decode graph runs (row-tiled scorer for many queries). Every scalar
    # comes from the host-side cache lengths: no per-layer H2D copy, no sync, and none of the
    # arange/where/cat elementwise chains of the torch references kept below for the tests

    # Rows per launch (also the expand kernel's fixed SEQ constexpr) and pools per score tile.
    # Selection scores are computed tile by tile into a fixed SEL_SLAB x SEL_TILE fp16 slab
    # (16 MiB, sharing the DSA indexer's backing) with a running top-k candidate set, so the
    # workspace never scales with the context (a single rows x T buffer would be 128 MiB per
    # slab at 256K). The slab stays wide: the scorer's pooled-key reuse is per row slab, so
    # narrower slabs cost more than the extra top-k passes save
    SEL_SLAB = 1024
    SEL_TILE = int(os.environ.get("EXL3_QSA_SCORE_TILE", 8192))
    # Row count up to which the plane-update workspaces come from g_tensor_cache (decode-class
    # calls: MTP verify, bsz > 1 fallbacks, where allocation latency matters); prefill chunks
    # allocate per call, the static cache being meant for small buffers only
    STATIC_ROWS = 32

    @staticmethod
    def _workspace(rows: int, numel: int, dtype, tag: str, device):
        from ..util.tensor import g_tensor_cache
        if rows <= QSAIndexer.STATIC_ROWS:
            return g_tensor_cache.get_bucketed(device, numel, dtype, tag)
        # Power-of-two size so the caching allocator reuses the freed block across chunks: the
        # score width grows with the context, and exact sizes would cache one segment per chunk
        nb = 1 << max(numel - 1, 0).bit_length()
        return torch.empty((nb,), dtype = dtype, device = device)[:numel]

    def k_pad(self):
        cr = self.compress_ratio
        return -(-(self.block_topk * cr + cr - 1) // 32) * 32

    def _sel_weights(self, rows: int, device):
        w = getattr(self, "_sel_w", None)
        if w is None or w.device != device:
            w = self._sel_w = torch.ones((self.SEL_SLAB, self.n_heads), dtype = torch.half, device = device)
        return w[:rows]

    def _select_rows(self, q_rows, pool_flat, pos0, T, out_rows, block_table = None, epp = 0):
        """
        One sequence's selection for consecutive query rows starting at absolute position pos0:
        q_rows (R, H, dk) roped, pool_flat (rows, dk) pooled keys (contiguous, or the paged plane
        with block_table/epp), T visible complete pools for the last row. Writes out_rows
        (R, K_pad) int32: selected pools expanded to tokens plus each row's tail, -1 padded.

        Scores are computed per SEL_TILE pools into a fixed slab. Beyond one tile the per-tile
        top-k candidates are merged under dsa_topk's own total order (score desc, index asc), so
        the result is identical to a single pass over the whole row.
        """
        import triton
        from .attention_fn.dsa_triton import dsa_indexer_scores, _dsa_pool_expand_kernel
        from ..ext import exllamav3_ext as ext
        from ..util.tensor import g_tensor_cache
        R = q_rows.shape[0]
        cr = self.compress_ratio
        dev = q_rows.device
        k_pad = out_rows.shape[1]
        k_sel = self.block_topk
        kp = -(-k_sel // 32) * 32
        t_tile = self.SEL_TILE
        if block_table is not None:
            t_tile = max(epp, t_tile // epp * epp)   # tiles must start on a pool page
        s_backing = g_tensor_cache.get(dev, (self.SEL_SLAB * self.SEL_TILE,), torch.half, "dsa_stile")

        def tile_scores(q_slab, rows, t0, t1):
            # Tile [t0, t1) of the pooled plane scored as if it started at pool 0: the row-0
            # position shifts by t0 * cr so the causal bounds shift by t0. Contiguous pools:
            # the launcher scans k_idx.shape[0] rows, so hand it exactly the visible ones
            s_stride = triton.cdiv(t1 - t0, 128) * 128
            sc = s_backing[: rows * s_stride].view(rows, s_stride)
            if block_table is None:
                return dsa_indexer_scores(
                    q_slab, self._sel_weights(rows, dev), pool_flat[t0 : t1], pos0 + r0 - t0 * cr,
                    cr, t1 - t0, scores = sc, scale = self.scale,
                )
            bt = block_table[t0 // epp : -(-t1 // epp)] if t0 else block_table
            return dsa_indexer_scores(
                q_slab, self._sel_weights(rows, dev), pool_flat, pos0 + r0 - t0 * cr, cr, t1 - t0,
                scores = sc, block_table = bt, epp = epp, scale = self.scale,
            )

        for r0 in range(0, R, self.SEL_SLAB):
            r1 = min(r0 + self.SEL_SLAB, R)
            rows = r1 - r0
            T_slab = min(T, (pos0 + r1) // cr)
            pool_idx = g_tensor_cache.get_bucketed(dev, rows * kp, torch.int32, "qsa_sel_pool") \
                .view(rows, kp)
            q_slab = q_rows[r0 : r1]
            if T_slab <= 0:
                pool_idx.fill_(-1)
            elif T_slab <= t_tile:
                sc = tile_scores(q_slab, rows, 0, T_slab)
                ext.dsa_topk(sc, pool_idx, min(k_sel, T_slab), None, 0)
            else:
                # Tiled: each tile's local top-k becomes candidate slot 1 next to the running set
                # in slot 0, and the native merge reduces the pair under the single-pass kernel's
                # own total order (score descending, index ascending), emitting the merged set --
                # with scores -- into slot 0 of the other workspace for the next tile. Workspaces
                # are fixed-size per call, so the allocator reuses them across chunks
                ws = [(torch.empty((rows, 2, kp), dtype = torch.int32, device = dev),
                       torch.empty((rows, 2, kp), dtype = torch.half, device = dev),
                       torch.zeros((rows, 2), dtype = torch.int32, device = dev)) for _ in range(2)]
                cur = 0
                tiles = list(range(0, T_slab, t_tile))
                for n, t0 in enumerate(tiles):
                    t1 = min(t0 + t_tile, T_slab)
                    sc = tile_scores(q_slab, rows, t0, t1)
                    w_idx, w_scr, w_cnt = ws[cur]
                    ext.dsa_topk_tile(sc, w_idx, w_scr, w_cnt, 1, min(k_sel, t1 - t0), t0)
                    if n == len(tiles) - 1:
                        ext.dsa_topk_merge_tiles(w_idx, w_scr, w_cnt, pool_idx, None, None, k_sel)
                    else:
                        n_idx, n_scr, n_cnt = ws[cur ^ 1]
                        ext.dsa_topk_merge_tiles(
                            w_idx, w_scr, w_cnt, n_idx[:, 0], n_scr[:, 0], n_cnt[:, 0], k_sel)
                        cur ^= 1
            with torch.cuda.device(dev):
                _dsa_pool_expand_kernel[(rows, triton.cdiv(k_pad, 256))](
                    pool_idx, out_rows[r0 : r1], pos0 + r0,
                    P = cr, SEL = self.block_topk, K_pad = k_pad, KP_pool = kp, TAIL = 1,
                    SEQ = self.SEL_SLAB, MULTIROW = 0, BLOCK = 256,
                )

    def select_indices(
        self,
        q_idx: torch.Tensor,
        pooled: torch.Tensor,
        past_len: int,
        batch_stride: int,
    ) -> torch.Tensor:
        """
        Selection as index lists instead of a mask: per query row, the top block_topk complete
        visible blocks expanded to tokens plus the tail block, as FLAT row indices (batch b's
        token t at b * batch_stride + t), -1 padded. q_idx (bsz, seq, H, dk) roped;
        pooled (bsz, nb, dk). Returns (bsz * seq, K_pad) int32 for the gathered attention
        kernel (K_pad matching the BC selection width).
        """
        bsz, seq = q_idx.shape[:2]
        nb = pooled.shape[1]
        out = torch.empty((bsz * seq, self.k_pad()), dtype = torch.int32, device = q_idx.device)
        pooled = pooled.contiguous()
        for b in range(bsz):
            rows = out[b * seq : (b + 1) * seq]
            self._select_rows(q_idx[b].contiguous(), pooled[b], past_len, nb, rows)
            if b:
                # flat row offset of this sequence's tokens (the nc gather is unpaged)
                rows.copy_(torch.where(rows >= 0, rows + b * batch_stride, rows))
        return out

    def select_indices_ref(
        self,
        q_idx: torch.Tensor,
        pooled: torch.Tensor,
        past_len: int,
        batch_stride: int,
        q_chunk: int = 1024,
    ) -> torch.Tensor:
        """Torch reference of select_indices (tests)."""
        bsz, seq = q_idx.shape[:2]
        cr = self.compress_ratio
        dev = q_idx.device
        nb = pooled.shape[1]
        k_pad = -(-(self.block_topk * cr + cr - 1) // 32) * 32
        out = torch.full((bsz, seq, k_pad), -1, dtype = torch.int32, device = dev)
        boffs = (torch.arange(bsz, device = dev) * batch_stride).view(bsz, 1, 1)

        for c0 in range(0, seq, q_chunk):
            c1 = min(c0 + q_chunk, seq)
            qpos = past_len + torch.arange(c0, c1, device = dev)                 # (C,)
            nbq = (qpos + 1) // cr
            scores = self.block_scores(q_idx[:, c0 : c1], pooled)                # (bsz, C, nb)
            scores = scores.masked_fill(
                torch.arange(nb, device = dev).view(1, 1, -1) >= nbq.view(1, -1, 1), -torch.inf)
            ksel = min(self.block_topk, nb)
            top = scores.topk(ksel, dim = -1)
            sel_ok = top.values > -torch.inf                                     # (bsz, C, ksel)
            sel_tok = (top.indices * cr).unsqueeze(-1) + torch.arange(cr, device = dev)
            sel_tok = sel_tok.flatten(2)                                         # (bsz, C, ksel*cr)
            sel_ok = sel_ok.unsqueeze(-1).expand(-1, -1, -1, cr).flatten(2)
            tail_tok = (nbq * cr).view(1, -1, 1) + torch.arange(cr, device = dev)
            tail_tok = tail_tok.expand(bsz, -1, -1)
            tok = torch.cat((sel_tok, tail_tok), dim = 2)                        # (bsz, C, L)
            ok = torch.cat((sel_ok, torch.ones_like(tail_tok, dtype = torch.bool)), dim = 2)
            ok &= tok <= qpos.view(1, -1, 1)
            out[:, c0 : c1, : tok.shape[2]] = \
                torch.where(ok, tok + boffs, tok.new_full((), -1)).int()
        return out.view(bsz * seq, k_pad)

    def sparse_attend_nc(self, attn, q, k, v, q_idx, pooled):
        """
        Non-cached sparse attention: gathered-GQA kernel over the per-row selection instead of
        a materialized (bsz, seq, seq) mask (whose SDPA fallback also expands K/V per q head).
        q (bsz, seq, qh, hd) roped, k/v (bsz, seq, kvh, hd); returns (bsz, seq, qh, hd) fp16.
        """
        from .attention_fn.qsa_triton import qsa_sparse_attend_rows
        bsz, seq = q.shape[:2]
        indices = self.select_indices(q_idx, pooled, past_len = 0, batch_stride = seq)
        o = qsa_sparse_attend_rows(
            q.reshape(bsz * seq, attn.num_q_heads, attn.head_dim).contiguous(),
            k.reshape(bsz * seq, attn.num_kv_heads, attn.head_dim).contiguous(),
            v.reshape(bsz * seq, attn.num_kv_heads, attn.head_dim).contiguous(),
            indices, attn.sm_scale,
        )
        return o.view(bsz, seq, attn.num_q_heads, attn.head_dim)

    def build_mask(
        self,
        x: torch.Tensor,
        raw_k_past: torch.Tensor | None,
        rope,
        params: dict,
    ):
        """
        Eager reference form of the selection (mask over all positions), kept for the parity
        tests: project current tokens, append to the raw key history, pool, select. Returns
        (mask (bsz, seq, total_len) bool, raw_k_full for the caller's cache).
        """
        past_len = 0 if raw_k_past is None else raw_k_past.shape[1]
        q, raw_k = self.project(x, rope, params, position = past_len)
        raw_k_full = raw_k if raw_k_past is None else torch.cat([raw_k_past, raw_k], dim = 1)
        pooled = self.pool_keys(raw_k_full, rope, params)
        mask = self.token_mask(q, pooled, past_len, raw_k_full.shape[1])
        return mask, raw_k_full

    # ---- cached (paged) path -------------------------------------------------------------------
    #
    # The side planes (raw per-token keys, pooled per-block keys) live in the attention layer's
    # CacheLayer_qsa and are maintained on EVERY cached forward; the sparse attention itself only
    # engages once some query position exceeds sparse_threshold()

    def sparse_threshold(self) -> int:
        # Highest query position for which dense attention is still exact + 1
        return 4 * self.block_topk + 3

    def _norm_w_half(self, which: str, device):
        """fp16 copy of a layernorm weight for the plane kernels (they add the +1 bias)."""
        attr = f"_{which}_norm_w_h"
        w = getattr(self, attr, None)
        if w is None or w.device != device:
            src = self.q_layernorm if which == "q" else self.k_layernorm
            w = src.weight.data.to(device = device, dtype = torch.half).contiguous()
            setattr(self, attr, w)
        return w

    def update_planes(
        self,
        layer,
        x: torch.Tensor,
        rope,
        block_table: torch.Tensor,
        cache_seqlens_cpu: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        """
        Project the current tokens, write raw keys into the paged plane, (re)pool the blocks the
        chunk touches. Returns the roped indexer queries (bsz, seq, n_heads, head_dim) for
        selection (a reused static, valid until the next call). The BC graph's plane kernels,
        JIT-launched: fused qk GEMM -> split + q RMS norm -> in-place partial rope with the
        device cache lengths as positions -> raw append -> pool update. No rope-table gathers,
        no per-layer H2D copy of the positions, no index arithmetic in torch. The pool kernel
        also writes the chunk's trailing incomplete pool (never selected; rebuilt once complete).
        """
        from .attention_fn.qsa_triton import _qsa_stage_kernel, _qsa_pool_update_kernel
        from .attention_fn.mla_triton import _mla_plane_update_kernel
        from ..util.tensor import g_tensor_cache, get_for_device
        bsz, seqlen, _ = x.shape
        dev = x.device
        cr, H, dk = self.compress_ratio, self.n_heads, self.head_dim
        R = bsz * seqlen
        seqlens = get_for_device(params, "cache_seqlens", dev, None)
        if seqlens is None or seqlens.dtype != torch.int32:
            seqlens = cache_seqlens_cpu.to(device = dev, dtype = torch.int32)
        bt = block_table if block_table.dtype == torch.int32 else block_table.int()
        bt = bt.contiguous()
        npr = bt.shape[1]
        page_size = layer.raw_k.shape[1]

        qk = self.index_qk_proj.forward(x.contiguous(), params).view(R, (H + 1) * dk)
        q = self._workspace(R, R * H * dk, torch.half, "qsa_up_q", dev) \
            .view(bsz, seqlen, H, dk)
        kraw = self._workspace(R, R * dk, torch.half, "qsa_up_k", dev) \
            .view(bsz, seqlen, dk)
        with torch.cuda.device(dev):
            _qsa_stage_kernel[(R, H + 1)](
                qk, self._norm_w_half("q", dev), q, kraw, R,
                eps = float(self.q_layernorm.rms_norm_eps), H_i = H, D = dk,
            )
            rope.apply(q, None, 0, seqlens, None, True)
            _mla_plane_update_kernel[(R,)](
                kraw, layer.raw_k, bt, seqlens, npr, seqlen,
                page_size = page_size, D = dk, DST_D = 0, DST_OFF = 0,
            )
            # MAXPOOLS is documentation of the grid height only (unused in the body); a fixed
            # value keeps one compile across chunk lengths
            _qsa_pool_update_kernel[(bsz, seqlen // cr + 1)](
                layer.raw_k.view(-1, dk), layer.pooled.view(-1, dk), self._norm_w_half("k", dev),
                rope.inv_freq, bt, seqlens, npr, seqlen,
                page_size = page_size, P = cr, D = dk, ROPE_R = 2 * rope.inv_freq.numel(),
                attn_factor = float(rope.attn_factor),
                eps = float(self.k_layernorm.rms_norm_eps), MAXPOOLS = 1,
            )
        return q

    def update_planes_ref(
        self,
        layer,
        x: torch.Tensor,
        rope,
        block_table: torch.Tensor,
        cache_seqlens_cpu: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        """Torch reference of update_planes (tests): writes only the completed pools."""
        bsz, seqlen, _ = x.shape
        dev = x.device
        cr = self.compress_ratio
        dk = self.head_dim
        max_pos = int(cache_seqlens_cpu.max().item()) + seqlen
        rope.expand_cache(max_pos)
        cos_all = rope.cached_cos
        sin_all = rope.cached_sin

        pos0 = cache_seqlens_cpu.long().to(dev)                                    # (bsz,)
        pos_ids = pos0.unsqueeze(1) + torch.arange(seqlen, device = dev)           # (bsz, seq)
        q, raw_k = self.project_ref(x, cos_all[pos_ids].half(), sin_all[pos_ids].half(), params)

        page_sz = layer.raw_k.shape[1]
        blocks_per_page = layer.pooled.shape[1]
        bt = block_table.long()
        raw_flat = layer.raw_k.view(-1, dk)
        pooled_flat = layer.pooled.view(-1, dk)

        # raw key write (row pages are disjoint, scatter is race-free)
        flat = bt.gather(1, pos_ids // page_sz) * page_sz + pos_ids % page_sz
        raw_flat[flat.flatten()] = raw_k.reshape(-1, dk).half()

        # pool blocks completed by this chunk: rows have different (b0, b1) ranges; pad to the max
        b0 = pos0 // cr
        b1 = (pos0 + seqlen) // cr
        nb_max = int((b1 - b0).max().item())
        if nb_max > 0:
            blocks = b0.unsqueeze(1) + torch.arange(nb_max, device = dev)          # (bsz, nbm)
            valid = blocks < b1.unsqueeze(1)
            blocks_c = torch.where(valid, blocks, b0.unsqueeze(1))                 # safe indices
            tok = (blocks_c * cr).unsqueeze(-1) + torch.arange(cr, device = dev)   # (bsz, nbm, cr)
            tflat = bt.unsqueeze(1).expand(-1, nb_max, -1) \
                .gather(2, tok // page_sz) * page_sz + tok % page_sz
            pooled = raw_flat[tflat].float().mean(dim = 2).to(torch.half)          # (bsz, nbm, dk)
            pooled = self.k_layernorm.forward(pooled, params)
            starts = blocks_c * cr
            pooled = _rope(pooled, cos_all[starts].half(), sin_all[starts].half())
            pflat = bt.gather(1, starts // page_sz) * blocks_per_page + blocks_c % blocks_per_page
            vm = valid.flatten()
            pooled_flat[pflat.flatten()[vm]] = pooled.reshape(-1, dk)[vm]
        return q

    def select_indices_paged(
        self,
        layer,
        q_idx: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cache-resident form of select_indices: pooled block keys are read from the paged plane
        through the block table and each row's positions are offset by its sequence's cache
        position. Emits per-SEQUENCE cache positions, (bsz * seq, K_pad) int32 -1-padded. The
        paged attention kernel maps them through the block table.
        """
        bsz, seq = q_idx.shape[:2]
        cr = self.compress_ratio
        epp = layer.pooled.shape[1]
        pool_flat = layer.pooled.view(-1, self.head_dim)
        bt = block_table.int()
        out = torch.empty((bsz * seq, self.k_pad()), dtype = torch.int32, device = q_idx.device)
        for b in range(bsz):
            pos0 = int(cache_seqlens_cpu[b])
            self._select_rows(
                q_idx[b].contiguous(), pool_flat, pos0, (pos0 + seq) // cr,
                out[b * seq : (b + 1) * seq], block_table = bt[b], epp = epp,
            )
        return out

    def select_indices_paged_ref(
        self,
        layer,
        q_idx: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens_cpu: torch.Tensor,
        q_chunk: int = 1024,
    ) -> torch.Tensor:
        """Torch reference of select_indices_paged (tests)."""
        bsz, seq = q_idx.shape[:2]
        cr = self.compress_ratio
        dk = self.head_dim
        dev = q_idx.device
        page_sz = layer.raw_k.shape[1]
        bpp = layer.pooled.shape[1]
        pooled_flat = layer.pooled.view(-1, dk)
        bt = block_table.long()
        pos0 = cache_seqlens_cpu.long().to(dev)
        nbf = (pos0 + seq) // cr                                                 # (bsz,)
        nbm = int(nbf.max().item())
        k_pad = -(-(self.block_topk * cr + cr - 1) // 32) * 32
        out = torch.full((bsz, seq, k_pad), -1, dtype = torch.int32, device = dev)

        blocks = torch.arange(nbm, device = dev).unsqueeze(0).expand(bsz, -1)    # (bsz, nbm)
        blk_c = torch.where(blocks < nbf.unsqueeze(1), blocks, torch.zeros_like(blocks))
        pflat = bt.gather(1, (blk_c * cr) // page_sz) * bpp + blk_c % bpp
        pk = pooled_flat[pflat]                                                  # (bsz, nbm, dk)

        for c0 in range(0, seq, q_chunk):
            c1 = min(c0 + q_chunk, seq)
            qpos = pos0.unsqueeze(1) + torch.arange(c0, c1, device = dev)        # (bsz, C)
            nbq = (qpos + 1) // cr
            scores = self.block_scores(q_idx[:, c0 : c1], pk)                    # (bsz, C, nbm)
            scores = scores.masked_fill(blocks.unsqueeze(1) >= nbq.unsqueeze(-1), -torch.inf)
            ksel = min(self.block_topk, nbm)
            top = scores.topk(ksel, dim = -1)
            sel_ok = top.values > -torch.inf                                     # (bsz, C, ksel)
            sel_tok = (top.indices * cr).unsqueeze(-1) + torch.arange(cr, device = dev)
            sel_tok = sel_tok.flatten(2)                                         # (bsz, C, ksel*cr)
            sel_ok = sel_ok.unsqueeze(-1).expand(-1, -1, -1, cr).flatten(2)
            tail_tok = (nbq * cr).unsqueeze(-1) + torch.arange(cr, device = dev)
            tok = torch.cat((sel_tok, tail_tok), dim = 2)                        # (bsz, C, L)
            ok = torch.cat((sel_ok, torch.ones_like(tail_tok, dtype = torch.bool)), dim = 2)
            ok &= tok <= qpos.unsqueeze(-1)
            out[:, c0 : c1, : tok.shape[2]] = \
                torch.where(ok, tok, tok.new_full((), -1)).int()
        return out.view(bsz * seq, k_pad)

    def sparse_attend(
        self,
        layer,
        attn,
        q: torch.Tensor,
        q_idx: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sparse paged attention through the gathered-GQA kernel: per-row selection over the
        pooled plane, then one split + combine launch over the cache pages -- nothing S x L is
        ever materialized (the einsum/flash predecessors gathered K/V per query chunk, which
        OOMed on long prefills). Serves cached prefill chunks and every eager sparse fallback
        (bsz > 1, MTP verify). K/V for the current tokens must already be written to the cache.
        q: (bsz, seq, num_q_heads, head_dim) roped; q_idx: roped indexer queries from
        update_planes. Returns (bsz, seq, num_q_heads, head_dim) fp16.
        """
        from .attention_fn.qsa_triton import qsa_sparse_attend_rows
        bsz, seq = q.shape[:2]
        indices = self.select_indices_paged(layer, q_idx, block_table, cache_seqlens_cpu)
        bt_rows = block_table.int().unsqueeze(1).expand(bsz, seq, -1) \
            .reshape(bsz * seq, -1).contiguous()
        o = qsa_sparse_attend_rows(
            q.reshape(bsz * seq, attn.num_q_heads, attn.head_dim).contiguous(),
            layer.k.view(-1, attn.num_kv_heads, attn.head_dim),
            layer.v.view(-1, attn.num_kv_heads, attn.head_dim),
            indices, attn.sm_scale,
            block_table = bt_rows, page_size = layer.k.shape[1],
        )
        return o.view(bsz, seq, attn.num_q_heads, attn.head_dim)
