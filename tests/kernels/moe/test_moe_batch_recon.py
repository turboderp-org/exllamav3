"""
Batched expert reconstruct kernels (modules/moe_batch_recon.py): reconstruct_had_batch must equal
reconstruct_had_slice per matrix, reconstruct_batch must equal reconstruct per matrix, had_r_128_batch must equal
had_r_128 per block, hgemm_batched must equal hgemm_recon per matrix to fp16 rounding, plan_groups must follow its
padding rule, and the grouped expert MLP (BatchReconLayer, loop and slot-gather output modes) must match the
per-expert reconstruct path within fp16 rounding.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.moe_batch_recon import BatchReconLayer, plan_groups
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import ptr_table, routing_layout


@pytest.mark.parametrize("K", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("cb", [(False, True), (False, False), (True, False)])
def test_reconstruct_had_batch(device, K, cb):
    gen = torch.Generator().manual_seed(K * 10 + int(cb[0]) * 2 + int(cb[1]))
    k, n, B = 256, 384, 5
    mats = [(rand_trellis(k, n, K, gen, device), rand_scale(k, gen, device), rand_scale(n, gen, device))
            for _ in range(B)]
    ref = torch.empty((B, k, n), dtype = torch.half, device = device)
    for b, (t, suh, svh) in enumerate(mats):
        ext.reconstruct_had_slice(ref[b], t, suh, svh, K, cb[0], cb[1], 0)
    out = torch.empty((B, k, n), dtype = torch.half, device = device)
    ptrs = [ptr_table([m[i] for m in mats], device) for i in range(3)]
    ext.reconstruct_had_batch(out, ptrs[0], ptrs[1], ptrs[2], K, cb[0], cb[1])
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_hgemm_batched(device):
    gen = torch.Generator().manual_seed(1)
    B, m, k, n = 4, 37, 256, 384
    a = torch.randn((B, m, k), generator = gen).half().to(device)
    w = torch.randn((B, k, n), generator = gen).half().to(device) * 0.05
    c = torch.empty((B, m, n), dtype = torch.half, device = device)
    ext.hgemm_batched(a, w, c)
    for b in range(B):
        r = torch.empty((m, n), dtype = torch.half, device = device)
        ext.hgemm_recon(a[b], w[b], r)
        # cuBLAS may pick different algorithms for the batched and single calls, so compare to fp16 rounding
        # rather than bit-exactly
        assert ((c[b].float() - r.float()).abs().max() <= 2e-3 * r.float().abs().max()).item(), \
            f"batch {b} differs from hgemm_recon"


@pytest.mark.nogpu
def test_plan_groups():
    counts = {0: 600, 1: 590, 2: 300, 3: 2000, 4: 100, 5: 95, 6: 90}
    groups = plan_groups(list(counts), lambda e: counts[e], batch_max = 3)
    # default PAD_MAX 1.1: 2000 alone; 600+590 (1200 <= 1.1 * 1190) but not 300 (1800 > 1639);
    # 300 alone (600 > 440 with 100); 100+95+90 (300 <= 313)
    assert groups == [[3], [0, 1], [2], [4, 5, 6]]
    # every expert appears exactly once, groups keep the descending order
    flat = [e for g in groups for e in g]
    assert sorted(flat) == sorted(counts)
    assert all(counts[g[0]] >= counts[g[-1]] for g in groups)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
def test_had_r_128_batch(device, dtype):
    gen = torch.Generator().manual_seed(3)
    E, B, m, d = 7, 4, 6, 256
    table = (torch.rand((E, d), generator = gen) * 2 + 0.1).half().to(device)
    ids = torch.tensor([5, 0, 6, 2], dtype = torch.long, device = device)
    x = torch.randn((B * m, d), generator = gen).to(dtype).to(device)
    for pre in (True, False):
        out = torch.empty_like(x)
        ext.had_r_128_batch(x, out, table if pre else None, None if pre else table, ids, m, 1.0)
        ref = torch.empty_like(x)
        for b in range(B):
            sc = table[ids[b]]
            ext.had_r_128(x[b * m : (b + 1) * m], ref[b * m : (b + 1) * m], sc if pre else None, None if pre else sc, 1.0)
        torch.cuda.synchronize()
        assert torch.equal(out, ref), f"pre={pre} dtype={dtype}"


@pytest.mark.parametrize("K", [2, 4, 8])
def test_reconstruct_batch(device, K):
    gen = torch.Generator().manual_seed(K)
    k, n, B = 256, 384, 5
    mats = [rand_trellis(k, n, K, gen, device) for _ in range(B)]
    ref = torch.empty((B, k, n), dtype = torch.half, device = device)
    for b, t in enumerate(mats):
        ext.reconstruct(ref[b], t, K, False, True)
    out = torch.empty((B, k, n), dtype = torch.half, device = device)
    ext.reconstruct_batch(out, ptr_table(mats, device), K, False, True)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("interm_fp32", [False, True])
@pytest.mark.parametrize("static", [True, False])
@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize("gated", [True, False])
def test_batch_recon_layer_matches_per_expert(device, gated, folded, static, interm_fp32):
    gen = torch.Generator().manual_seed(7)
    k, n, K = 256, 384, 4
    E, rows = 6, 200
    trell = {p: [rand_trellis(k if p != "d" else n, n if p != "d" else k, K, gen, device) for _ in range(E)]
             for p in (("g", "u", "d") if gated else ("u", "d"))}
    suh = {p: [rand_scale(k if p != "d" else n, gen, device) for _ in range(E)] for p in trell}
    svh = {p: [rand_scale(n if p != "d" else k, gen, device) for _ in range(E)] for p in trell}
    y = torch.randn((rows, k), generator = gen).half().to(device)
    # Real routing layout: each token picks TOPK distinct experts with a skewed preference, the assignments are
    # sorted by expert (testlib.moe.routing_layout), tok / w are the sorted token ids and weights, and counts /
    # starts come from the histogram
    TOPK = 2
    pref = torch.tensor([0.35, 0.3, 0.15, 0.1, 0.06, 0.04])
    sel = torch.stack([torch.multinomial(pref, TOPK, replacement = False, generator = gen) for _ in range(rows)])
    lay = routing_layout(sel, torch.rand((rows, TOPK), generator = gen).half(), E)
    tok = lay.token_sorted.to(device)
    w = lay.weight_sorted.to(device)
    counts = lay.expert_count[:E].tolist()
    starts = lay.expert_start[:E].tolist()
    flat_expert = lay.flat_e.to(device)
    inv_order = lay.inv.to(device)
    expert_start = lay.expert_start.to(device)

    # The batched tier pads every expert of a group to the group's largest row count and
    # hgemm_batched picks the fp16-accumulator kernel or cuBLAS from that padded shape; the
    # per-expert reference pads the same way so hgemm_recon makes the same choice per expert
    # (the grouping is deterministic, computed here exactly as the loop below does)
    groups = plan_groups(list(range(E)), lambda e: counts[e], batch_max = 4)
    cmax_of = {e: max(counts[g] for g in grp) for grp in groups for e in grp}

    def lin(p, e, x, out_dtype = torch.half):
        # The per-expert reconstruct path (BC_BlockSparseMLP::run_single_expert_dq):
        # had(x * suh) @ W_hat -> had -> * svh, fp32 output for the down projection
        W = torch.empty((trell[p][e].shape[0] * 16, trell[p][e].shape[1] * 16), dtype = torch.half, device = device)
        ext.reconstruct(W, trell[p][e], K, False, True)
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh[p][e], None, 1.0)
        rows = cmax_of[e]
        xp = torch.zeros((rows, xh.shape[1]), dtype = torch.half, device = device); xp[:xh.shape[0]] = xh
        yp = torch.empty((rows, W.shape[1]), dtype = out_dtype, device = device)
        ext.hgemm_recon(xp, W, yp)
        yy = yp[:xh.shape[0]].contiguous()
        ext.had_r_128(yy, yy, None, svh[p][e], 1.0)
        return yy

    # fp32 intermediates (gemma4): gate / up outputs in fp32, the activation kernel writes fp16
    idt = torch.float if interm_fp32 else torch.half
    ref = torch.zeros((rows, k), dtype = torch.float, device = device)
    for e in range(E):
        idx = tok[starts[e] : starts[e] + counts[e]]
        x = y.index_select(0, idx)
        u = lin("u", e, x, idt)
        a = torch.empty((u.shape[0], u.shape[1]), dtype = torch.half, device = device) if interm_fp32 else u
        if gated:
            g = lin("g", e, x, idt)
            ext.silu_mul(g, u, a, 0.0)
        else:
            ext.relu_mul(u, u, a, 0.0)
        d = lin("d", e, a, torch.float)
        ref.index_add_(0, idx, d * w[starts[e] : starts[e] + counts[e]].float().unsqueeze(1))

    scales = {p: (suh[p], svh[p]) for p in trell}
    layer = BatchReconLayer((k, n, K) if gated else None, (k, n, K), (n, k, K),
                            (False, True), (False, True), (False, True),
                            "silu" if gated else "relu2", 0.0, device, scales, folded = folded, interm_fp32 = interm_fp32)
    y_ext = torch.cat([y, torch.zeros((1, k), dtype = torch.half, device = device)])
    out_ext = torch.zeros((rows + 1, k), dtype = torch.float, device = device)
    tok_ext = torch.cat([tok, torch.full((1,), rows, dtype = tok.dtype, device = device)])
    w_ext = torch.cat([w, torch.zeros((1,), dtype = torch.half, device = device)])
    if static:
        layer.set_static_pointers(*[ptr_table(trell[p], device) if p in trell else None for p in ("g", "u", "d")])
    outs = {}
    for slot_mode in (False, True):
        out_ext.zero_()
        if slot_mode:
            # Slot layout as BlockSparseMLP builds it: each group's experts get cmax rows each,
            # groups back to back; kind 2 = unweighted, the gather applies the routing weight
            n_slots = sum(len(grp) * max(counts[e] for e in grp) for grp in groups)
            scratch = torch.full((n_slots, k), float("nan"), dtype = torch.float, device = device)
            base = torch.zeros(E, dtype = torch.long); kind = torch.zeros(E, dtype = torch.long)
            slot = 0
            for grp in groups:
                cmax = max(counts[e] for e in grp)
                for b, e in enumerate(grp):
                    base[e] = slot + b * cmax; kind[e] = 2
                slot += len(grp) * cmax
            base, kind = base.to(device), kind.to(device)
        slot = 0
        for grp in groups:
            ptrs = None if static else tuple(
                [trell[p][e].data_ptr() for e in grp] if p in trell else None for p in ("g", "u", "d"))
            out_slab = None
            if slot_mode:
                cmax = max(counts[e] for e in grp); out_slab = scratch[slot : slot + len(grp) * cmax]; slot += len(grp) * cmax
            layer.run_group(y_ext, out_ext, tok_ext, w_ext, grp, [starts[e] for e in grp], [counts[e] for e in grp],
                            ptrs = ptrs, out_slab = out_slab)
        if slot_mode:
            ext.exl3_moe_gather(out_ext[:rows], scratch, flat_expert, inv_order, expert_start, base, kind, w)
        torch.cuda.synchronize()
        outs[slot_mode] = out_ext[:rows].clone()
    # Slot mode sums each token's contributions in k order, the loop in expert order: same fp32
    # terms, different order, so equal to fp32 rounding but not bit-exact
    d = (outs[True] - outs[False]).abs().max().item()
    assert d <= 1e-5 * ref.abs().max().item(), f"slot mode vs loop: {d}"
    err = (outs[True] - ref).abs().max().item()
    scale = ref.abs().max().item()
    # Unfolded: same arithmetic as the reference up to GEMM algorithm choice (cuBLAS batched vs single can differ
    # in accumulation order on some devices); folded weights round to fp16 after the Hadamard. Mutations (swapped
    # scale tables, scrambled ids) fail these by orders of magnitude
    tol = (4e-3 if folded else 5e-4) * scale + 1e-3
    assert err <= tol, f"max abs err {err} vs scale {scale} (folded {folded})"
