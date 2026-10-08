"""
Fused MoE kernel (ext.exl3_moe) reading packed CPU trellis tiles: the streamed tier feeds
arena bytes (tile group 2/8 order plus the planar dword order, exl3_moe_cpu_swizzle_group /
exl3_moe_cpu_planar_layout) straight into the kernel, which gathers whole tiles in its
cp.async B-stage and remaps the dequant's shared-word indices. The inputs are identical up to
byte order, so the output must be bit-identical to the native-layout run, for every layout the
rules can produce and (as generality) their cross-combinations. Half rates ride group 8 only.
Also covered: the per-projection layout args are validated, and the CPU-side rules only ever
emit combinations the GPU guards accept.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_experts
from testlib.moe import contiguous_layout, expert_ptr_tables, experts_to, moe_buffers, repack_trellis

H, I, E = 256, 512, 4
COUNTS = [(20, 5, 16, 1), (16, 16, 16, 16)]
LAYOUTS = [pytest.param((g, p), id = f"g{g}{'p' if p else ''}") for g in (2, 8) for p in (0, 1)]


def packed_experts(ex, group, planar):
    """Expert set with every trellis repacked to (group, planar); tensors stay on device:
    their data_ptr()s go straight into a kernel"""
    out = {p: [(repack_trellis(t, group, bool(planar)), su, sv) for t, su, sv in ex[p]] for p in ex}
    assert all(e[0].is_cuda for p in out for e in out[p]), "packed weights must be on the device"
    return out


def run_fused(ex, K, counts, groups = (0, 0, 0), planars = (0, 0, 0)):
    """Gated silu MLP over experts with per-expert token counts (topk 1: assignment i is token
    i, already grouped by expert), at the given per-projection packed layouts"""
    bsz = sum(counts)
    gen = torch.Generator().manual_seed(1234)
    y = (torch.randn((bsz, H), generator = gen) * 0.05).half().to(ex["u"][0][1].device)
    out = torch.zeros((bsz, H), dtype = torch.float, device = y.device)
    token_sorted, expert_count, _ = contiguous_layout(list(counts), y.device)
    wts = torch.ones(bsz, dtype = torch.half, device = y.device)
    bufs = moe_buffers(H, I, max(counts), y.device)
    tabs = expert_ptr_tables(ex, y.device)
    ext.exl3_moe(y, out, expert_count, token_sorted, wts, *bufs, 0,
                 float(K), float(K), float(K),
                 *tabs["g"], *tabs["u"], *tabs["d"], False, True, False, True, False, True,
                 0.0, E, None, None, 1, max(counts), 16,
                 *groups, *planars)
    # The kernel reads the trellis addresses from the tables asynchronously: the tensors behind
    # them must outlive the launch, so sync before returning (callers must keep them alive too)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("counts", COUNTS)
def test_fused_moe_packed_tiles_are_bit_exact(device, layout, K, counts):
    group, planar = layout
    ex = experts_to(rand_experts(E, H, I, K, torch.Generator().manual_seed(7)), device)
    ref = run_fused(ex, K, counts)
    pex = packed_experts(ex, group, planar)
    got = run_fused(pex, K, counts, (group,) * 3, (planar,) * 3)
    assert torch.equal(ref, got), f"packed layout g{group}p{planar} K{K} changed the fused output"

    # negative control: packed addressing over native bytes must NOT reproduce the reference
    wrong = run_fused(ex, K, counts, (group,) * 3, (planar,) * 3)
    assert not torch.equal(ref, wrong), "the group/planar arguments are being ignored"


@pytest.mark.parametrize("K", (1.5, 2.5, 3.5))
def test_fused_moe_packed_half_rates_are_bit_exact(device, K):
    """Half rates ride the group-8 tile order on the AVX-512 tiers (never planar); the B-stage
    gather is rate-agnostic, so the fused output must match the native bytes exactly"""
    counts = COUNTS[1]
    ex = experts_to(rand_experts(E, H, I, K, torch.Generator().manual_seed(11)), device)
    ref = run_fused(ex, K, counts)
    pex = packed_experts(ex, 8, 0)
    got = run_fused(pex, K, counts, (8, 8, 8))
    assert torch.equal(ref, got), f"group8 at K{K} changed the fused output"


def test_fused_moe_rejects_bad_layouts(device):
    counts = (16, 16)
    ex = experts_to(rand_experts(2, H, I, 4, torch.Generator().manual_seed(9)), device)
    with pytest.raises(RuntimeError):
        run_fused(ex, 4, counts, (4, 0, 0))
    with pytest.raises(RuntimeError):
        run_fused(ex, 4, counts, (2, 2, 2), (0, 2, 0))


def test_rules_only_produce_supported_combinations():
    """Whatever the tier rules hand out must be a combination the kernels accept: planar only
    with a swizzle, never on a half-integer rate, group in {0, 2, 8}."""
    for K in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8):
        g = ext.exl3_moe_cpu_swizzle_group(K)
        p = ext.exl3_moe_cpu_planar_layout(K)
        assert g in (0, 2, 8), f"K{K}: group {g}"
        assert p in (0, 1), f"K{K}: planar {p}"
        assert not p or g, f"K{K}: planar without a swizzle"
        assert not p or K == int(K), f"K{K}: planar on a half-integer rate"