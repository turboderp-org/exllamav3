"""
Batched reconstruct path for hot MoE experts at prefill.

Experts with more assigned rows than the fused kernel's temp-row capacity run through the
reconstruct path: dequantize the expert's matrices to fp16, run cuBLAS GEMMs. This module groups
experts (up to EXL3_MOE_RECON_BATCH per group, sorted by row count so the padding is small) and
runs each group with a fixed number of launches:

  - one pointer-table reconstruct per projection over all B matrices
  - the rows of each expert are gathered into a [B, cmax, k] slab (padding rows read a zero row
    and target a sink row of the output, so no masking is needed) and multiplied with one
    strided-batched fp16 GEMM per projection (fp32 accumulation, same as hgemm)
  - the Hadamard transforms run once per slab (had_r_128_batch: per-expert suh / svh rows
    picked from stacked [E, dim] tables, fp32 output for the down projection), which is the
    same arithmetic as the per-expert path: the rotated-basis weights are exact in fp16, only
    the activations round. EXL3_MOE_RECON_FOLDED=1 instead folds both transforms into the weights
    (reconstruct_had, the dense Linear prefill formulation): fewer launches, but the folded
    weights round to fp16, roughly doubling the relative error of the expert output
  - the activation runs over the whole slab, and one index_add_ folds the weighted results into
    the accumulator

Weights that would not fit the reconstruct scratch budget (EXL3_MOE_RECON_MB) shrink the group.
"""

from __future__ import annotations
import os
import numpy as np
import torch
from ..ext import exllamav3_ext as ext

RECON_BATCH = max(1, int(os.environ.get("EXL3_MOE_RECON_BATCH", 16)))
# Padded rows per group (B * cmax): bounds the gathered input / intermediate slabs
RECON_ROWS = max(1, int(os.environ.get("EXL3_MOE_RECON_ROWS", 16384)))
# Dequantized weight scratch budget (two live projections at a time)
RECON_MB = max(1, int(os.environ.get("EXL3_MOE_RECON_MB", 256)))
RECON_FOLDED = os.environ.get("EXL3_MOE_RECON_FOLDED", "0") != "0"
# Batching pays while a single expert's GEMM cannot fill the GPU: with 128x128 output tiles an
# m x n GEMM launches ceil(m/128) * ceil(n/128) blocks, and below roughly the SM count the
# strided-batched kernel wins by a wide margin (PRO 6000, 188 SMs, k = 2048: n = 768 is 1.45x
# faster batched at m = 1024 and 2.9x at m = 256; n = 1792 breaks even at m = 1024). Above
# that the batched cuBLAS kernels are ~10-15% slower than the single-GEMM ones, and the slab
# padding / gather traffic grows with the rows, so experts whose narrowest projection already
# spans more than RECON_TILES tiles stay per-expert. EXL3_MOE_RECON_TILES sets the budget
# (0 = batch everything); EXL3_MOE_RECON_MAX_ROWS caps rows directly
RECON_TILES = int(os.environ.get("EXL3_MOE_RECON_TILES", 64))
RECON_MAX_ROWS = int(os.environ.get("EXL3_MOE_RECON_MAX_ROWS", 0))
# Give up on adding a smaller expert to a group once padding would exceed this fraction of the
# real rows: a group of very uneven experts wastes GEMM work on zeros
PAD_MAX = 1.5

# act(g, u) -> a kernels; gateless relu2 rides relu_mul(u, u, a) = relu2(u)
_ACT_CALLS_GATED = {
    "silu": ext.silu_mul,
    "gelu": ext.gelu_mul,
    "swiglu_oai": ext.silu_oai_mul,
    "relu2": ext.relu2_mul,
}
_ACT_CALLS_GATELESS = {"relu2": ext.relu_mul}
# MoeCpuHost spec["activation"] indices
_ACT_IDX = {0: "silu", 1: "gelu", 2: "relu2", 3: "swiglu_oai"}


def plan_groups(experts, count_of, batch_max = None):
    """Split `experts` (ids) into groups for run_group, largest first, so each group's padding
    stays small. count_of: id -> row count."""
    batch_max = batch_max or RECON_BATCH
    order = sorted(experts, key = lambda e: -count_of(e))
    groups, cur, cmax, real = [], [], 0, 0
    for e in order:
        c = count_of(e)
        if cur:
            nb = len(cur) + 1
            if nb > batch_max or nb * cmax > RECON_ROWS or nb * cmax > PAD_MAX * (real + c):
                groups.append(cur)
                cur = []
        if not cur:
            cmax, real = c, 0
        cur.append(e)
        real += c
    if cur:
        groups.append(cur)
    return groups


def batch_cap(dims_g, dims_u, dims_d):
    """Largest group the weight scratch budget allows for these projection shapes (gate and up
    are live together, down reuses the gate scratch)"""
    kg, ng = (dims_g[0], dims_g[1]) if dims_g else (0, 0)
    per = 2 * (max(kg * ng, dims_d[0] * dims_d[1]) + dims_u[0] * dims_u[1])
    return max(1, min(RECON_BATCH, (RECON_MB << 20) // per))


class BatchReconLayer:
    """Per-layer constants for the batched path: projection dims, codebook flags, activation,
    and the stacked per-expert sign vectors. Trellis pointer tables are supplied per call (they
    differ between GPU-resident and streamed experts)."""

    def __init__(self, dims_g, dims_u, dims_d, cb_g, cb_u, cb_d, activation, act_limit, device,
                 scales, folded = None):
        # dims: (k, n, K); cb: (mcg, mul1); scales: per projection ("g", "u", "d") a pair of
        # lists of per-expert suh (k) and svh (n) half tensors, stacked here into [E, dim]
        # tables
        self.gated = dims_g is not None
        self.dims_g, self.dims_u, self.dims_d = dims_g, dims_u, dims_d
        self.cb_g, self.cb_u, self.cb_d = cb_g, cb_u, cb_d
        act = _ACT_IDX[activation] if isinstance(activation, int) else activation
        self.act_call = (_ACT_CALLS_GATED if self.gated else _ACT_CALLS_GATELESS)[act]
        self.act_limit = float(act_limit or 0.0)
        self.device = device
        self.cap = batch_cap(dims_g, dims_u, dims_d)
        if RECON_MAX_ROWS > 0:
            self.max_rows = RECON_MAX_ROWS
        elif RECON_TILES > 0:
            n_min = min(dims_u[1], dims_d[1])
            self.max_rows = max(128, (RECON_TILES * 128 * 128) // n_min)
        else:
            self.max_rows = 1 << 30
        # Static per-expert trellis pointer tables ([E] int64, device) when the experts are
        # resident: the per-group tables are then a device-side gather by expert id instead of
        # a host-built list. Streamed experts (VRAM slot addresses) pass pointers per call.
        self.ptr_tables = None
        self.folded = RECON_FOLDED if folded is None else folded
        # Stacked [E, dim] per-expert scale tables: had_r_128_batch reads rows by id in the
        # unfolded path; the folded path derives per-matrix suh/svh addresses from them
        self.scales = {}
        for p in ("g", "u", "d") if self.gated else ("u", "d"):
            suh, svh = scales[p]
            self.scales[p] = (torch.stack(list(suh)).contiguous(), torch.stack(list(svh)).contiguous())

    def _linear(self, x, W, p, ids, out_n, out_dtype = torch.half):
        """x: [B, cmax, k] input slab, W: [B, k, n] -> [B, cmax, n]. Unfolded: the same
        arithmetic as the per-expert path (had_r_128 with fp16 pre-scale in, fp16/fp32 post-
        scale out), with the per-expert scale rows picked from the stacked tables"""
        B, cmax, k = x.shape
        if not self.folded:
            suh, svh = self.scales[p]
            xh = torch.empty_like(x)
            ext.had_r_128_batch(x.view(B * cmax, k), xh.view(B * cmax, k), suh, None, ids, cmax, 1.0)
            x = xh
        y = torch.empty((B, cmax, out_n), dtype = out_dtype, device = x.device)
        ext.hgemm_batched(x, W, y)
        if not self.folded:
            y2 = y.view(B * cmax, out_n)
            ext.had_r_128_batch(y2, y2, None, svh, ids, cmax, 1.0)
        return y

    def _recon(self, W, ptrs, p, ids_d, K, cb):
        ptrs = ptrs.contiguous()
        if self.folded:
            # Per-matrix suh/svh addresses: row `id` of the stacked tables
            suh, svh = self.scales[p]
            ext.reconstruct_had_batch(
                W, ptrs,
                (suh.data_ptr() + ids_d * (suh.shape[1] * 2)).contiguous(),
                (svh.data_ptr() + ids_d * (svh.shape[1] * 2)).contiguous(),
                K, *cb)
        else:
            ext.reconstruct_batch(W, ptrs, K, *cb)

    def set_static_pointers(self, ptrs_g, ptrs_u, ptrs_d):
        """Resident experts: [E] int64 device tensors of trellis addresses per projection"""
        self.ptr_tables = (ptrs_g, ptrs_u, ptrs_d)

    def run_group(
        self,
        y_ext: torch.Tensor,       # (rows + 1, k_in) half; last row zero, k_in == dims_u[0]
        out_ext: torch.Tensor,     # (rows + 1, ho) float; last row is the padding sink
        tok_ext: torch.Tensor,     # (A + 1,) long: expert-sorted token ids, trailing sentinel = rows
        w_ext: torch.Tensor,       # (A + 1,) half: matching routing weights, trailing 0
        ids: list[int],            # expert ids of the group (index into scale / pointer tables)
        starts: list[int],         # per expert: segment start in tok_ext
        counts: list[int],         # per expert: rows
        ptrs: tuple | None = None, # streamed experts: per projection (g, u, d) lists of B
                                   # trellis addresses; None = gather from set_static_pointers
    ):
        B = len(counts)
        assert B <= self.cap
        cmax = max(counts)
        A = tok_ext.shape[0] - 1
        dev = y_ext.device
        ku, nu, Ku = self.dims_u
        kd, nd, Kd = self.dims_d

        # One host->device copy per group: [ids | starts | counts | ptr_g | ptr_u | ptr_d]. The
        # source is a fresh pageable tensor: CUDA stages those immediately without waiting on
        # the stream (measured: the host returns in ~0.2 ms with 300 ms of work queued), and
        # nothing here is read back. Everything else is derived on the device.
        meta_h = np.zeros((6, B), dtype = np.int64)
        meta_h[0] = ids
        meta_h[1] = starts
        meta_h[2] = counts
        if ptrs is not None:
            for i, pl in enumerate(ptrs):
                if pl is not None:
                    meta_h[3 + i] = pl
        meta = torch.from_numpy(meta_h).to(dev, non_blocking = True)
        ids_d = meta[0]
        if ptrs is not None:
            ptr_g, ptr_u, ptr_d = meta[3], meta[4], meta[5]
        else:
            tg, tu, td = self.ptr_tables
            ptr_g = tg.index_select(0, ids_d) if tg is not None else None
            ptr_u = tu.index_select(0, ids_d)
            ptr_d = td.index_select(0, ids_d)

        # Gather index into the sorted assignment list, [B, cmax]; pads read the sentinel
        ar = torch.arange(cmax, device = dev)
        src = meta[1].unsqueeze(1) + ar.unsqueeze(0)
        src = torch.where(ar.unsqueeze(0) < meta[2].unsqueeze(1), src, A).view(-1)
        tok = tok_ext.index_select(0, src)
        w = w_ext.index_select(0, src)
        x = y_ext.index_select(0, tok).view(B, cmax, ku)

        # Gate / up. Weight scratch is prefill-shaped and comes from the caching allocator per
        # call (the down projection reuses the gate slab, which is dead after the activation)
        kg, ng = (self.dims_g[0], self.dims_g[1]) if self.gated else (0, 0)
        scratch1 = torch.empty(B * max(kg * ng, kd * nd), dtype = torch.half, device = dev)
        Wu = torch.empty((B, ku, nu), dtype = torch.half, device = dev)
        self._recon(Wu, ptr_u, "u", ids_d, Ku, self.cb_u)
        u = self._linear(x, Wu, "u", ids_d, nu)
        del Wu
        if self.gated:
            Kg = self.dims_g[2]
            Wg = scratch1[:B * kg * ng].view(B, kg, ng)
            self._recon(Wg, ptr_g, "g", ids_d, Kg, self.cb_g)
            g = self._linear(x, Wg, "g", ids_d, ng)
            g2, u2 = g.view(B * cmax, ng), u.view(B * cmax, nu)
            self.act_call(g2, u2, u2, self.act_limit)      # in place into u
            del g
        else:
            u2 = u.view(B * cmax, nu)
            self.act_call(u2, u2, u2, self.act_limit)

        # Down: fp32 output, as the per-expert DQ path (hgemm into fp32, fp32 output Hadamard)
        Wd = scratch1[:B * kd * nd].view(B, kd, nd)
        self._recon(Wd, ptr_d, "d", ids_d, Kd, self.cb_d)
        d = self._linear(u, Wd, "d", ids_d, nd, out_dtype = torch.float)
        d2 = d.view(B * cmax, nd)
        ho = out_ext.shape[1]
        out_ext.index_add_(0, tok, d2[:, :ho].mul_(w.float().unsqueeze(1)))
