"""
ext.moe_unswizzle_trellis: the GPU kernel that restores native (k/16, n/16, 16K) tile order (and native dword
order) for expert weights staged packed from the CPU arena must invert exactly the repack the child applies at
rehome (testlib.moe.repack_trellis), for every integer and half-integer K, for every tile group the rules can
produce (8 on the AVX-512 tiers, 2 + planar dwords on AVX2), across a batch laid out expert-major with g/u/d at
fixed byte offsets and a different K for down, and act as a plain copy for matrices flagged native (a K8 matrix
flagged group 8 is inverted like any other).
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_trellis
from testlib.moe import repack_trellis

ALL_RATES = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)


def down_rate(K):
    """A different rate for the down projection: integer K -> another integer, half K -> another half K"""
    if K != int(K):
        return {1.5: 3.5, 2.5: 1.5, 3.5: 2.5}[K]
    return max(1, (K + 2) % 8 + 1)


# K/group cases: band-8 (the AVX-512 rule) at every rate, band-2 + planar (the AVX2 rule) at
# integer rates; planar follows group 2, the only pairing the rules produce
CASES = [(K, 8, False) for K in ALL_RATES] + [(8, 8, True), (5, 8, True)]   # K 5: down at K 8
CASES += [(K, 2, True) for K in (1, 2, 4, 8)]


@pytest.mark.parametrize("K,group,swizzle_k8", CASES, ids = [f"K{K}-g{group}{'-k8swz' if s else ''}" for K, group, s in CASES])
def test_unswizzle_expert_major(device, K, group, swizzle_k8):
    gen = torch.Generator().manual_seed(int(K * 10) + group + swizzle_k8)
    E = 3
    dims = {"g": (512, 384, K), "u": (512, 384, K), "d": (384, 512, down_rate(K))}
    natives = {name: [rand_trellis(k, n, Kp, gen) for _ in range(E)] for name, (k, n, Kp) in dims.items()}
    nbytes = {name: (k // 16) * (n // 16) * int(16 * Kp) * 2 for name, (k, n, Kp) in dims.items()}
    offsets = {"g": 0, "u": nbytes["g"], "d": nbytes["g"] + nbytes["u"]}
    exp_b = sum(nbytes.values())
    # group 0 stays native per projection (the rule exempts K8 on the band-8 tier; half rates
    # have no group-2 layout); planar follows the group-2 rule, identity on half rates
    def layout(Kp):
        g = group if (Kp != 8 or swizzle_k8) else 0
        return g, (g == 2 and Kp == int(Kp))
    layouts = {name: layout(Kp) for name, (_, _, Kp) in dims.items()}

    staged = torch.zeros(E * exp_b // 2, dtype = torch.int16)
    for e in range(E):
        for name, off in offsets.items():
            t = natives[name][e]
            g, pl = layouts[name]
            t = repack_trellis(t, g, pl) if g else t
            a = (e * exp_b + off) // 2
            staged[a : a + t.numel()] = t.reshape(-1)
    src = staged.to(device)
    dst = torch.full_like(src, -1)
    for name, off in offsets.items():
        k, n, Kp = dims[name]
        g, pl = layouts[name]
        ext.moe_unswizzle_trellis(src, dst, E, exp_b, off, k // 16, n // 16, Kp, g, int(pl))
    torch.cuda.synchronize()
    for e in range(E):
        for name, off in offsets.items():
            t = natives[name][e]
            a = (e * exp_b + off) // 2
            got = dst[a : a + t.numel()].cpu().view_as(t)
            assert torch.equal(got, t), f"expert {e} {name} (K {dims[name][2]} g{layouts[name][0]} p{layouts[name][1]}) not restored"


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("E, tiles_k, tiles_n", [(0, 4, 8), (2, 0, 8), (2, 4, 0), (0, 0, 0)])
def test_empty_unswizzle(device, E, tiles_k, tiles_n):
    """No experts or no tiles: nothing to copy; alignment and sign checks still apply"""
    src = torch.zeros(4096, dtype = torch.int16, device = device)
    dst = torch.full_like(src, -1)
    ext.moe_unswizzle_trellis(src, dst, E, 4096, 0, tiles_k, tiles_n, 4, 8)
    torch.cuda.synchronize()
    assert (dst == -1).all()
    with pytest.raises(RuntimeError, match = "16-byte aligned"):
        ext.moe_unswizzle_trellis(src, dst, E, 4096, 8, tiles_k, tiles_n, 4, 8)
    with pytest.raises(RuntimeError, match = "negative size"):
        ext.moe_unswizzle_trellis(src, dst, -1, 4096, 0, tiles_k, tiles_n, 4, 8)
    with pytest.raises(RuntimeError, match = "group must be"):
        ext.moe_unswizzle_trellis(src, dst, max(E, 1), 4096, 0, max(tiles_k, 1), max(tiles_n, 8), 4, 4)
    _device_still_works(device)
