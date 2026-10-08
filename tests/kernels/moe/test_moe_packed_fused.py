"""Packed and native trellises represent identical weights, so fused MoE outputs must agree
bit for bit, including mixed projection layouts and fractional rates. HIP fused consumers
require native weights; malformed descriptors must fail before launching on either backend.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.moe import contiguous_layout, expert_ptr_tables, moe_buffers, swizzle_trellis


def build_experts(E, h, inter, K, seed, device, K_down = None):
    """Per-expert (trellis, suh, svh) for gate/up ([h, inter]) and down ([inter, h])"""
    g = torch.Generator(device = "cpu").manual_seed(seed)
    def mk(k, n, rate):
        tk, tn = k // 16, n // 16
        tr = torch.randint(0, 256, (tk, tn, int(16 * rate) * 2), dtype = torch.uint8,
                           generator = g).view(torch.int16)
        suh = torch.rand(k, generator = g).half().abs() + 0.5
        svh = torch.rand(n, generator = g).half().abs() + 0.5
        return tr.to(device).contiguous(), suh.to(device), svh.to(device)
    return ([mk(h, inter, K) for _ in range(E)], [mk(h, inter, K) for _ in range(E)],
            [mk(inter, h, K if K_down is None else K_down) for _ in range(E)])


def run_fused(counts, gate, up, down, h, inter, K, groups, planars = (0, 0, 0), K_down = None):
    device = up[0][0].device
    E = len(counts)
    bsz = sum(counts)
    cap = max(counts)
    torch.manual_seed(1234)
    hs = torch.randn(bsz, h, device = device).half()
    out = torch.zeros(bsz, h, dtype = torch.float32, device = device)
    # topk = 1: assignments are already grouped by expert
    tok, expert_count, _ = contiguous_layout(counts, device)
    wts = torch.ones(bsz, dtype = torch.half, device = device)
    ts = moe_buffers(h, inter, cap, device)
    tabs = expert_ptr_tables(dict(g = gate, u = up, d = down), device)
    ext.exl3_moe(
        hs, out, expert_count, tok, wts, ts[0], ts[1], ts[2], ts[3],
        0, float(K), float(K), float(K if K_down is None else K_down),
        *tabs["g"], *tabs["u"], *tabs["d"],
        False, True, False, True, False, True,
        0.0, E, None, None, 1, cap, 16,
        groups[0], groups[1], groups[2],
        planars[0], planars[1], planars[2],
    )
    # The kernel reads the trellis addresses from the table asynchronously; the tensors behind
    # them must outlive the launch, so sync before returning (callers must also keep them alive)
    torch.cuda.synchronize(device)
    return out


LAYOUTS = [pytest.param((g, p), id = f"g{g}{'p' if p else ''}") for g in (2, 8) for p in (0, 1)]


@pytest.mark.cuda_only
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("counts", [(20, 5, 16, 1), (16, 16, 16, 16)])
def test_fused_moe_packed_tiles_are_bit_exact(device, layout, K, counts):
    group, planar = layout
    E, h, inter = len(counts), 256, 512
    gate, up, down = build_experts(E, h, inter, K, 7, device)
    ref = run_fused(counts, gate, up, down, h, inter, K, (0, 0, 0))

    def swapped(proj):
        return [(swizzle_trellis(tr, group, planar), su, sv) for tr, su, sv in proj]
    # bound, not inline: the repacked tensors must stay alive while the kernel runs
    pg, pu, pd = swapped(gate), swapped(up), swapped(down)
    got = run_fused(counts, pg, pu, pd, h, inter, K,
                    (group, group, group), (planar, planar, planar))
    assert torch.equal(ref, got), f"packed layout g{group}p{planar} K{K} changed the fused output"

    # negative control: packed addressing over native bytes must NOT reproduce the reference
    wrong = run_fused(counts, gate, up, down, h, inter, K, (group, group, group),
                      (planar, planar, planar))
    assert not torch.equal(ref, wrong), "the group/planar arguments are being ignored"
    del pg, pu, pd


@pytest.mark.parametrize("projection", [0, 1, 2], ids = ["gate", "up", "down"])
@pytest.mark.parametrize("K, group, planar", [
    (4, 4, 0), (4, 8, -1), (4, 8, 2), (4, 0, 1), (2.5, 2, 1), (2.5, 8, 1),
])
def test_fused_moe_rejects_bad_layouts(device, projection, K, group, planar):
    counts = (16, 16)
    E, h, inter = 2, 256, 512
    gate, up, down = build_experts(E, h, inter, K, 9, device)
    groups, planars = [0, 0, 0], [0, 0, 0]
    groups[projection], planars[projection] = group, planar
    with pytest.raises(RuntimeError):
        run_fused(counts, gate, up, down, h, inter, K, groups, planars)


@pytest.mark.rocm_only
@pytest.mark.parametrize("projection", [0, 1, 2], ids = ["gate", "up", "down"])
@pytest.mark.parametrize("group, planar", [(8, 0), (2, 1)])
def test_hip_fused_moe_requires_native_layout(device, projection, group, planar):
    counts = (16, 16)
    gate, up, down = build_experts(2, 256, 512, 4, 9, device)
    groups, planars = [0, 0, 0], [0, 0, 0]
    groups[projection], planars[projection] = group, planar
    with pytest.raises(RuntimeError):
        run_fused(counts, gate, up, down, 256, 512, 4, groups, planars)


@pytest.mark.cuda_only
@pytest.mark.parametrize("K_down", [1.5, 2.5, 3.5])
def test_fused_moe_mixed_down_rate_is_bit_exact(device, K_down):
    counts, K = (20, 5, 16, 1), 3
    gate, up, down = build_experts(4, 256, 512, K, 19, device, K_down)
    ref = run_fused(counts, gate, up, down, 256, 512, K, (0, 0, 0), K_down = K_down)
    layouts = ((2, 1), (8, 0), (8, 0))
    packed = [[(swizzle_trellis(tr, group, planar), su, sv) for tr, su, sv in proj]
              for proj, (group, planar) in zip((gate, up, down), layouts)]
    got = run_fused(counts, *packed, 256, 512, K, (2, 8, 8), (1, 0, 0), K_down)
    assert torch.equal(got, ref)


@pytest.mark.cuda_only
@pytest.mark.parametrize("K", [1.5, 2.5, 3.5])
def test_fused_moe_fractional_group8_is_bit_exact(device, K):
    counts = (16, 16, 16, 16)
    gate, up, down = build_experts(4, 256, 512, K, 23, device)
    ref = run_fused(counts, gate, up, down, 256, 512, K, (0, 0, 0))
    packed = [[(swizzle_trellis(tr, 8, 0), su, sv) for tr, su, sv in proj]
              for proj in (gate, up, down)]
    got = run_fused(counts, *packed, 256, 512, K, (8, 8, 8))
    assert torch.equal(got, ref)