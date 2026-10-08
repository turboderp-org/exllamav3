"""
Deterministic output accumulation of the fused MoE kernel (ext.exl3_moe + ext.exl3_moe_gather): the slot-and-gather
path must be bit-reproducible across runs, agree with the atomic path to fp32 rounding, and match a torch reference
of the expert MLP (testlib.moe.expert_mlp_ref); the 32 / 64-row tile instances launched per expert range must agree
with a single 16-row launch; foreign (expert-parallel) picks in the sentinel bucket must never be gathered.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_experts
from testlib.moe import contiguous_layout, expert_mlp_ref, expert_ptr_tables, experts_to, moe_buffers, routing_layout

H, I, K, E, TOPK, T, CAP = 256, 256, 4, 8, 2, 64, 128


def routed_ref(ex, y, sel, w):
    """Weighted sum over each token's local picks (experts 0..E-1) of the expert MLP"""
    ref = torch.zeros((y.shape[0], H), dtype = torch.float, device = y.device)
    for e in range(E):
        m = sel == e
        toks = m.any(1).nonzero(as_tuple = True)[0]
        if toks.numel() == 0:
            continue
        d = expert_mlp_ref(ex, e, y.index_select(0, toks), K)
        ref.index_add_(0, toks, d * w[m].float().unsqueeze(1))
    return ref


def launch(y, out, layout_args, bufs, tabs, num_active = -1, scratch = None, base = None, lo = 1, hi = CAP,
           m_tile = 16):
    expert_count, token_sorted, weight_sorted = layout_args
    ext.exl3_moe(y, out, expert_count, token_sorted, weight_sorted, *bufs, 0, K, K, K,
                 *tabs["g"], *tabs["u"], *tabs["d"], False, True, False, True, False, True, 0.0, num_active,
                 scratch, base, lo, hi, m_tile, 0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("seed", [0, 1])
def test_fused_det_matches_atomic_and_reference(device, seed):
    gen = torch.Generator().manual_seed(seed)
    ex = experts_to(rand_experts(E, H, I, K, gen), device)
    y = (torch.randn((T, H), generator = gen) * 0.05).half().to(device)
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(T)]).to(device)   # distinct per token
    w = torch.rand((T, TOPK), generator = gen).half().to(device)
    ref = routed_ref(ex, y, sel, w)

    lay = routing_layout(sel, w, E)
    bufs = moe_buffers(H, I, CAP, device)
    tabs = expert_ptr_tables(ex, device)

    def run(det):
        out = torch.zeros((T, H), dtype = torch.float, device = device)
        if det:
            kind = ((lay.expert_count > 0) & (lay.expert_count <= CAP)).long()
            kind[E:] = 0
            fc = lay.expert_count * kind
            base = torch.cumsum(fc, 0) - fc
            scratch = torch.full((int(fc.sum()), H), float("nan"), dtype = torch.float, device = device)
        else:
            scratch = base = None
        launch(y, out, (lay.expert_count, lay.token_sorted, lay.weight_sorted), bufs, tabs,
               scratch = scratch, base = base)
        if det:
            ext.exl3_moe_gather(out, scratch, lay.flat_e, lay.inv, lay.expert_start, base, kind, lay.weight_sorted)
        torch.cuda.synchronize()
        return out

    d1, d2, a = run(True), run(True), run(False)
    assert torch.equal(d1, d2), "deterministic path is not bit-reproducible"
    scale = ref.abs().max().item()
    assert (d1 - a).abs().max().item() <= 1e-5 * scale, "det vs atomic"
    assert (d1 - ref).abs().max().item() <= 3e-3 * scale, "det vs reference"
    assert (a - ref).abs().max().item() <= 3e-3 * scale, "atomic vs reference"


@pytest.mark.parametrize("dims", [(256, 128), (512, 256)])   # N = 128 and N = 256 shapes
def test_fused_row_tiles_match_single_launch(device, dims):
    """The 32 / 64-row tile instances, launched per expert range, produce the same per-slot outputs as one 16-row
    launch over every expert: the arithmetic per row is unchanged, only the B dequant is shared across row
    fragments"""
    Hd, Id = dims
    counts = [3, 16, 17, 24, 32, 33, 40, 48, 64, 96, 128, 1]
    E2 = len(counts)
    gen = torch.Generator().manual_seed(7)
    ex = experts_to(rand_experts(E2, Hd, Id, K, gen), device)
    A = sum(counts)
    y = (torch.randn((A, Hd), generator = gen) * 0.05).half().to(device)
    weight_sorted = (torch.rand((A,), generator = gen) * 0.5 + 0.5).half().to(device)
    token_sorted, expert_count, starts = contiguous_layout(counts, device)
    base = starts.clone()
    bufs = moe_buffers(Hd, Id, CAP, device)
    tabs = expert_ptr_tables(ex, device)
    lay = (expert_count, token_sorted, weight_sorted)

    def run(lo, hi, m_tile, n_active, scratch):
        out = torch.zeros((A, Hd), dtype = torch.float, device = device)
        launch(y, out, lay, bufs, tabs, n_active, scratch, base, lo, hi, m_tile)

    # Same group geometry for every launch (num_active -1: fixed group width): the k-slice split, and so the fp32
    # reduction order, is then identical and the tiles must match bit for bit
    single = torch.full((A, Hd), float("nan"), dtype = torch.float, device = device)
    run(1, CAP, 16, -1, single)
    tiered = torch.full((A, Hd), float("nan"), dtype = torch.float, device = device)
    run(33, CAP, 64, -1, tiered)
    run(17, 32, 32, -1, tiered)
    run(1, 16, 16, -1, tiered)
    torch.cuda.synchronize()
    assert not torch.isnan(single).any() and not torch.isnan(tiered).any(), "an expert was skipped"
    if Hd % 256 == 0 and Id % 256 == 0:
        # The single launch takes the N = 256 instance here while the wide tiles are N = 128 kernels: a different
        # column tiling reduces k in a different order, so rounding-level agreement is the bound
        assert (tiered - single).abs().max().item() <= 2e-3 * single.abs().max().item(), \
            "N = 128 tiles vs N = 256 launch beyond fp32 rounding"
    else:
        assert torch.equal(single, tiered), "row-tile launches differ from the single 16-row launch"
    # Production geometry (per-launch active counts widen the groups): a different k-slice split changes the
    # reduction order, so only agreement to fp32 rounding is expected
    prod = torch.full((A, Hd), float("nan"), dtype = torch.float, device = device)
    run(33, CAP, 64, sum(1 for c in counts if c > 32), prod)
    run(17, 32, 32, sum(1 for c in counts if 16 < c <= 32), prod)
    run(1, 16, 16, sum(1 for c in counts if c <= 16), prod)
    torch.cuda.synchronize()
    assert not torch.isnan(prod).any()
    assert (prod - single).abs().max().item() <= 2e-3 * single.abs().max().item(), \
        "tiered launches vs single beyond fp32 rounding"


def test_fused_det_foreign_sentinel(device):
    """Expert-parallel shard: picks outside the local slice map to the sentinel bucket E of expert_count. The
    all-fused fast path builds its slot tables on the device from that count vector; the sentinel row must never be
    gathered (nothing writes its slots). This is the module's fast-path construction with the gather limited to the
    E real rows"""
    gen = torch.Generator().manual_seed(3)
    ex = experts_to(rand_experts(E, H, I, K, gen), device)
    y = (torch.randn((T, H), generator = gen) * 0.05).half().to(device)
    n_global = E + 4          # 4 foreign experts (ids E..E+3) live on another rank
    sel_g = torch.stack([torch.randperm(n_global, generator = gen)[:TOPK] for _ in range(T)])
    sel_g[0] = torch.tensor([E, E + 1])           # a token with no local expert at all
    sel_g = sel_g.to(device)
    w = torch.rand((T, TOPK), generator = gen).half().to(device)
    ref = routed_ref(ex, y, sel_g, w)

    lay = routing_layout(sel_g, w, E)
    bufs = moe_buffers(H, I, CAP, device)
    tabs = expert_ptr_tables(ex, device)
    A = lay.flat_e.shape[0]
    tables = torch.stack([lay.expert_start, lay.expert_start, (lay.expert_count > 0).long()])
    scratch = torch.full((A, H), float("nan"), dtype = torch.float, device = device)    # sentinel slots stay NaN
    out = torch.zeros((T, H), dtype = torch.float, device = device)
    launch(y, out, (lay.expert_count, lay.token_sorted, lay.weight_sorted), bufs, tabs,
           scratch = scratch, base = tables[0])
    ext.exl3_moe_gather(out, scratch, lay.flat_e, lay.inv, tables[1, :E], tables[0, :E], tables[2, :E],
                        lay.weight_sorted)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all(), "sentinel (foreign) slots were gathered"
    scale = ref.abs().max().item()
    assert (out - ref).abs().max().item() <= 3e-3 * scale
    assert out[0].abs().max().item() == 0.0     # token with no local expert contributes nothing


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("case", ["no_tokens", "no_active", "no_cap", "no_experts", "no_hidden"])
def test_empty_fused(device, case):
    """No tokens, no active experts, no rows per expert in this tier, no experts or no output columns: a no-op
    (output untouched, shape checks still applied)"""
    gen = torch.Generator().manual_seed(0)
    E_ = 0 if case == "no_experts" else E
    ex = experts_to(rand_experts(E_, H, I, K, gen), device)
    tabs = expert_ptr_tables(ex, device) if E_ else {p: [torch.empty(0, dtype = torch.long, device = device)] * 3
                                                    for p in "gud"}
    T_ = 0 if case == "no_tokens" else 4
    H_ = 0 if case == "no_hidden" else H
    y = torch.randn((T_, H_), device = device).half()
    sel = torch.randint(0, max(E_, 1), (T_, TOPK), device = device)
    if E_ == 0:
        sel.fill_(-1)
    lay = routing_layout(sel, torch.ones((T_, TOPK), dtype = torch.half, device = device), E_)
    bufs = moe_buffers(H_, I, 0 if case == "no_cap" else CAP, device)
    out = torch.full((T_, H_), 5.0, dtype = torch.float, device = device)
    args = (lay.expert_count, lay.token_sorted, lay.weight_sorted)
    launch(y, out, args, bufs, tabs, num_active = 0 if case == "no_active" else -1)
    torch.cuda.synchronize()
    assert (out == 5.0).all()
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        launch(y, out.half(), args, bufs, tabs, num_active = 0 if case == "no_active" else -1)
    _device_still_works(device)


def test_empty_fused_rejections(device):
    """An empty intermediate dim is rejected (the fused tier's output slots would stay unwritten); temp buffers
    without a group, or an expert_count without the sentinel bin, are rejected"""
    gen = torch.Generator().manual_seed(0)
    ex = experts_to(rand_experts(E, H, I, K, gen), device)
    tabs = expert_ptr_tables(ex, device)
    y = torch.randn((4, H), device = device).half()
    sel = torch.randint(0, E, (4, TOPK), device = device)
    lay = routing_layout(sel, torch.ones((4, TOPK), dtype = torch.half, device = device), E)
    args = (lay.expert_count, lay.token_sorted, lay.weight_sorted)
    out = torch.full((4, H), 5.0, dtype = torch.float, device = device)
    with pytest.raises(RuntimeError, match = "exl3_moe: empty intermediate dimension"):
        launch(y, out, args, moe_buffers(H, 0, CAP, device), tabs)
    with pytest.raises(RuntimeError, match = "exl3_moe: temp buffers hold no expert group"):
        launch(y, out, args, [b[:0] for b in moe_buffers(H, I, CAP, device)], tabs)
    with pytest.raises(RuntimeError, match = "exl3_moe: expert_count must hold"):
        launch(y, out, (lay.expert_count[:0], lay.token_sorted, lay.weight_sorted), moe_buffers(H, I, CAP, device),
               tabs)
    assert (out == 5.0).all()
    _device_still_works(device)


@pytest.mark.parametrize("tokens, topk", [(0, 2), (3, 0), (0, 0)])
def test_empty_gather(device, tokens, topk):
    """No tokens: no-op. No assignments per token: each row gains the empty sum (unchanged)"""
    out = torch.full((tokens, H), 5.0, dtype = torch.float, device = device)
    scratch = torch.empty((0, H), dtype = torch.float, device = device)
    flat = torch.empty(tokens * topk, dtype = torch.long, device = device)
    tables = torch.zeros(E + 1, dtype = torch.long, device = device)
    ext.exl3_moe_gather(out, scratch, flat, flat, tables, tables, tables, torch.empty(tokens * topk, dtype = torch.half,
                                                                                      device = device))
    torch.cuda.synchronize()
    assert (out == 5.0).all()
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.exl3_moe_gather(out.half(), scratch, flat, flat, tables, tables, tables,
                            torch.empty(tokens * topk, dtype = torch.half, device = device))
    _device_still_works(device)
