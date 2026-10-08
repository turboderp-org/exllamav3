"""
ext.moe_unswizzle_trellis: the GPU kernel that restores native (k/16, n/16, 16K) tile order for expert weights
staged band-swizzled from the CPU arena must invert exactly the permutation the child applies at rehome (physical
order (n/128 group, k-tile, member, tile), testlib.moe.swizzle_trellis), for every integer and half-integer K,
across a batch laid out expert-major with g/u/d at fixed byte offsets and a different K for down, and act as a plain
copy for matrices flagged unswizzled (K8 in production; a K8 matrix flagged swizzled is inverted like any other).
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_trellis
from testlib.moe import swizzle_trellis

ALL_RATES = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)


def down_rate(K):
    """A different rate for the down projection: integer K -> another integer, half K -> another half K"""
    if K != int(K):
        return {1.5: 3.5, 2.5: 1.5, 3.5: 2.5}[K]
    return max(1, (K + 2) % 8 + 1)


CASES = [(K, False) for K in ALL_RATES] + [(8, True), (5, True)]   # K 5: down at K 8


@pytest.mark.parametrize("K,swizzle_k8", CASES, ids = [f"K{K}-k8{'swz' if s else 'native'}" for K, s in CASES])
def test_unswizzle_expert_major(device, K, swizzle_k8):
    gen = torch.Generator().manual_seed(int(K * 10) + swizzle_k8)
    E = 3
    dims = {"g": (512, 384, K), "u": (512, 384, K), "d": (384, 512, down_rate(K))}
    natives = {name: [rand_trellis(k, n, Kp, gen) for _ in range(E)] for name, (k, n, Kp) in dims.items()}
    nbytes = {name: (k // 16) * (n // 16) * int(16 * Kp) * 2 for name, (k, n, Kp) in dims.items()}
    offsets = {"g": 0, "u": nbytes["g"], "d": nbytes["g"] + nbytes["u"]}
    exp_b = sum(nbytes.values())
    swizzled = {name: Kp != 8 or swizzle_k8 for name, (_, _, Kp) in dims.items()}

    staged = torch.zeros(E * exp_b // 2, dtype = torch.int16)
    for e in range(E):
        for name, off in offsets.items():
            t = natives[name][e]
            t = swizzle_trellis(t) if swizzled[name] else t
            a = (e * exp_b + off) // 2
            staged[a : a + t.numel()] = t.reshape(-1)
    src = staged.to(device)
    dst = torch.full_like(src, -1)
    for name, off in offsets.items():
        k, n, Kp = dims[name]
        ext.moe_unswizzle_trellis(src, dst, E, exp_b, off, k // 16, n // 16, Kp, swizzled[name])
    torch.cuda.synchronize()
    for e in range(E):
        for name, off in offsets.items():
            t = natives[name][e]
            a = (e * exp_b + off) // 2
            got = dst[a : a + t.numel()].cpu().view_as(t)
            assert torch.equal(got, t), f"expert {e} {name} (K {dims[name][2]}) not restored"
