"""
Packed CPU trellis layouts in the reconstruct entries (ext.reconstruct / _slice / _batch /
_had_slice / _had_batch): the streamed MoE prefill path feeds arena bytes (tile group 2/8 order
plus the planar dword order, exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout) straight
into these kernels instead of restoring native order first. Every packed combination must
reconstruct bit-identically to the same bytes in native order, and the guards must reject the
combinations the rules never produce (planar without a swizzle, planar on a half-integer rate,
groups outside {0, 2, 8}).
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_trellis, rand_scale
from testlib.moe import repack_trellis

GROUPS = [pytest.param(g, id = f"g{g}") for g in (0, 2, 8)]
INTEGRAL_K = (1, 2, 3, 4, 5, 6, 7, 8)


def reconstruct(out_shape, packed, K, device, group = 0, planar = 0):
    out = torch.empty(out_shape, dtype = torch.float16, device = device)
    ext.reconstruct(out, packed, float(K), False, True, group, planar)
    return out


@pytest.mark.parametrize("K", INTEGRAL_K)
@pytest.mark.parametrize("group", GROUPS)
def test_reconstruct_packed_is_bit_exact(device, K, group):
    for planar in ((0, 1) if group else (0,)):
        base = rand_trellis(256, 256, K, torch.Generator().manual_seed(1000 + K), device = device)
        ref = reconstruct((256, 256), base, K, device)
        got = reconstruct((256, 256), repack_trellis(base, group, bool(planar)), K, device, group, planar)
        assert torch.equal(ref, got), f"K{K} group{group} planar{planar} differs"


@pytest.mark.parametrize("K", (1.5, 2.5, 3.5))
def test_reconstruct_packed_half_rates_are_bit_exact(device, K):
    """The AVX-512 rule packs half rates into group 8 (never planar); the tile gather is
    rate-agnostic, so those bytes must reconstruct exactly like the native layout"""
    base = rand_trellis(256, 256, K, torch.Generator().manual_seed(1100 + int(2 * K)), device = device)
    ref = reconstruct((256, 256), base, K, device)
    got = reconstruct((256, 256), repack_trellis(base, 8), K, device, 8, 0)
    assert torch.equal(ref, got), f"K{K} group8 differs"


@pytest.mark.parametrize("K", (2, 4))
@pytest.mark.parametrize("group,planar", [(2, 1), (8, 0)])
def test_reconstruct_slice_with_offset(device, K, group, planar):
    base = rand_trellis(256, 512, K, torch.Generator().manual_seed(3000 + K), device = device)
    packed = repack_trellis(base, group, bool(planar))
    full = reconstruct((256, 512), base, K, device)
    sliced = torch.empty((256, 512 - 128), dtype = torch.float16, device = device)
    ext.reconstruct_slice(sliced, packed, float(K), False, True, 128, group, planar)
    assert torch.equal(sliced, full[:, 128:])


@pytest.mark.parametrize("K", (2, 4, 8))
@pytest.mark.parametrize("group,planar", [(2, 1), (8, 0)])
def test_batched_variants(device, K, group, planar):
    B = 4
    bases = [rand_trellis(256, 256, K, torch.Generator().manual_seed(4000 + i), device = device)
             for i in range(B)]
    packed = [repack_trellis(b, group, bool(planar)) for b in bases]
    ptrs = torch.tensor([p.data_ptr() for p in packed], dtype = torch.long, device = device)
    got = torch.empty((B, 256, 256), dtype = torch.float16, device = device)
    ext.reconstruct_batch(got, ptrs, float(K), False, True, group, planar)
    assert torch.equal(got, torch.stack([reconstruct((256, 256), b, K, device) for b in bases]))

    gen = torch.Generator().manual_seed(K)
    suh = [rand_scale(256, gen, device = device) for _ in range(B)]
    svh = [rand_scale(256, gen, device = device) for _ in range(B)]
    sp = torch.tensor([t.data_ptr() for t in suh], dtype = torch.long, device = device)
    vp = torch.tensor([t.data_ptr() for t in svh], dtype = torch.long, device = device)
    got_h = torch.empty((B, 256, 256), dtype = torch.float16, device = device)
    ext.reconstruct_had_batch(got_h, ptrs, sp, vp, float(K), False, True, group, planar)
    ref_h = torch.empty_like(got_h)
    for i, _ in enumerate(bases):
        ext.reconstruct_had_slice(ref_h[i], packed[i], suh[i], svh[i], float(K), False, True,
                                  0, group, planar)
    assert torch.equal(got_h, ref_h)


@pytest.mark.parametrize("args", [
    pytest.param((2.5, 8, 1), id = "planar_half_rate"),
    pytest.param((2.0, 4, 0), id = "bad_group"),
    pytest.param((2.0, 0, 1), id = "planar_alone"),
])
def test_invalid_layouts_rejected(device, args):
    K, group, planar = args
    base = rand_trellis(256, 256, 2, torch.Generator().manual_seed(5000), device = device)
    out = torch.empty((256, 256), dtype = torch.float16, device = device)
    with pytest.raises(RuntimeError):
        ext.reconstruct(out, base, K, False, True, group, planar)