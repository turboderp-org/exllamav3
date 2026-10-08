"""Packed CPU trellis layouts must reconstruct and restore bit-identically to native bytes."""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.moe import swizzle_trellis

GROUPS = [
    pytest.param((2, 0), id = "group2"),
    pytest.param((2, 1), id = "group2+planar"),
    pytest.param((8, 0), id = "group8"),
    pytest.param((8, 1), id = "group8+planar"),
]
CONSUMERS = ("reconstruct", "reconstruct_slice", "reconstruct_batch",
             "reconstruct_had_slice", "reconstruct_had_batch", "unswizzle")


def make_packed(tk, tn, K, seed = 0):
    """Random [k/16, n/16, 16K] trellis bytes in the native tile order"""
    g = torch.Generator(device = "cpu").manual_seed(seed)
    ps = int(16 * K)
    return torch.randint(0, 256, (tk, tn, ps * 2), dtype = torch.uint8, generator = g).view(torch.int16)


def reconstruct(packed, K, group = 0, planar = 0):
    tk, tn, _ = packed.shape
    out = torch.empty((tk * 16, tn * 16), dtype = torch.float16, device = packed.device)
    ext.reconstruct(out, packed, float(K), False, True, group, planar)
    return out


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("layout", GROUPS)
def test_reconstruct_packed_is_bit_exact(device, K, layout):
    group, planar = layout
    base = make_packed(16, 16, K, 1000 + K).to(device)
    ref = reconstruct(base, K)
    got = reconstruct(swizzle_trellis(base, group, planar), K, group, planar)
    assert torch.equal(ref, got), f"K{K} group{group} planar{planar} differs"


@pytest.mark.parametrize("K", [2, 4, 8])
@pytest.mark.parametrize("layout", GROUPS)
def test_restore_pass_and_direct_read_agree(device, K, layout):
    group, planar = layout
    base = make_packed(16, 16, K, 2000 + K).to(device)
    packed = swizzle_trellis(base, group, planar)
    src = packed.reshape(-1).contiguous()
    dst = torch.zeros(packed.numel(), dtype = torch.int16, device = device)
    ext.moe_unswizzle_trellis(src, dst, 1, packed.numel() * 2, 0,
                              base.shape[0], base.shape[1], float(K), group, planar)
    assert torch.equal(dst.view(base.shape), base), "restore kernel disagrees"
    assert torch.equal(reconstruct(base, K), reconstruct(packed, K, group, planar))


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("layout", [pytest.param((2, 1), id = "group2+planar"),
                                    pytest.param((8, 0), id = "group8")])
def test_reconstruct_slice_with_offset(device, K, layout):
    group, planar = layout
    base = make_packed(16, 32, K, 3000 + K).to(device)
    packed = swizzle_trellis(base, group, planar)
    k, n = base.shape[0] * 16, base.shape[1] * 16
    full = torch.empty((k, n), dtype = torch.float16, device = device)
    ext.reconstruct_slice(full, base, float(K), False, True, 0, 0, 0)
    sliced = torch.empty((k, n - 128), dtype = torch.float16, device = device)
    ext.reconstruct_slice(sliced, packed, float(K), False, True, 128, group, planar)
    assert torch.equal(sliced, full[:, 128:])


@pytest.mark.parametrize("K", [2, 4, 8])
@pytest.mark.parametrize("layout", [pytest.param((2, 1), id = "group2+planar"),
                                    pytest.param((8, 0), id = "group8")])
def test_batched_variants(device, K, layout):
    group, planar = layout
    B, tk, tn = 4, 16, 16
    bases = [make_packed(tk, tn, K, 4000 + i).to(device) for i in range(B)]
    packed = [swizzle_trellis(b, group, planar) for b in bases]
    k, n = tk * 16, tn * 16
    ptrs = torch.tensor([p.data_ptr() for p in packed], dtype = torch.long, device = device)
    got = torch.empty((B, k, n), dtype = torch.float16, device = device)
    ext.reconstruct_batch(got, ptrs, float(K), False, True, group, planar)
    assert torch.equal(got, torch.stack([reconstruct(b, K) for b in bases]))

    torch.manual_seed(K)
    suh = [torch.randn(k, dtype = torch.float16, device = device).abs() + 0.5 for _ in range(B)]
    svh = [torch.randn(n, dtype = torch.float16, device = device).abs() + 0.5 for _ in range(B)]
    sp = torch.tensor([t.data_ptr() for t in suh], dtype = torch.long, device = device)
    vp = torch.tensor([t.data_ptr() for t in svh], dtype = torch.long, device = device)
    got_h = torch.empty((B, k, n), dtype = torch.float16, device = device)
    ext.reconstruct_had_batch(got_h, ptrs, sp, vp, float(K), False, True, group, planar)
    ref_h = torch.empty_like(got_h)
    for i, _ in enumerate(bases):
        ext.reconstruct_had_slice(ref_h[i], bases[i], suh[i], svh[i], float(K), False, True,
                                  0, 0, 0)
    assert torch.equal(got_h, ref_h)


def consume_layout(device, K, group, planar, consumer, packed = False):
    """Exercise one public consumer with live tensors and identical logical weights/scales."""
    bases = [make_packed(16, 32, K, 5000 + i).to(device) for i in range(2)]
    tensors = [swizzle_trellis(t, group, planar) for t in bases] if packed else bases
    k, n = 256, 512
    width = n - 128 if consumer in ("reconstruct_slice", "reconstruct_had_slice") else n
    out = torch.empty((2, k, width), dtype = torch.float16, device = device)
    ptrs = torch.tensor([t.data_ptr() for t in tensors], dtype = torch.long, device = device)
    suh = [torch.linspace(0.5, 1.5, k, device = device).half() for _ in bases]
    svh = [torch.linspace(0.75, 1.75, n, device = device).half() for _ in bases]
    sp = torch.tensor([t.data_ptr() for t in suh], dtype = torch.long, device = device)
    vp = torch.tensor([t.data_ptr() for t in svh], dtype = torch.long, device = device)
    if consumer == "reconstruct_batch":
        ext.reconstruct_batch(out, ptrs, K, False, True, group, planar)
    elif consumer == "reconstruct_had_batch":
        ext.reconstruct_had_batch(out, ptrs, sp, vp, K, False, True, group, planar)
    elif consumer == "unswizzle":
        restored = torch.empty_like(tensors[0])
        ext.moe_unswizzle_trellis(tensors[0].view(-1), restored.view(-1), 1,
                                  tensors[0].numel() * 2, 0, 16, 32, K, group, planar)
        return restored
    else:
        for i, t in enumerate(tensors):
            if consumer == "reconstruct":
                ext.reconstruct(out[i], t, K, False, True, group, planar)
            elif consumer == "reconstruct_slice":
                ext.reconstruct_slice(out[i], t, K, False, True, 128, group, planar)
            else:
                ext.reconstruct_had_slice(out[i], t, suh[i], svh[i][128:], K,
                                          False, True, 128, group, planar)
    return out


@pytest.mark.parametrize("K", [1.5, 2.5, 3.5])
@pytest.mark.parametrize("group", [2, 8])
@pytest.mark.parametrize("consumer", CONSUMERS)
def test_fractional_nonplanar_is_bit_exact(device, K, group, consumer):
    ref = consume_layout(device, K, 0, 0, consumer)
    got = consume_layout(device, K, group, 0, consumer, packed = True)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("K, group, planar", [
    pytest.param(2.0, 4, 0, id = "bad_group"),
    pytest.param(2.0, 0, 1, id = "planar_alone"),
    pytest.param(2.5, 2, 1, id = "half_planar_group2"),
    pytest.param(2.5, 8, 1, id = "half_planar_group8"),
    pytest.param(2.0, 8, -1, id = "negative_planar"),
    pytest.param(2.0, 8, 2, id = "nonboolean_planar"),
])
@pytest.mark.parametrize("consumer", CONSUMERS)
def test_invalid_layouts_rejected(device, K, group, planar, consumer):
    with pytest.raises(RuntimeError):
        consume_layout(device, K, group, planar, consumer)


def test_rules_only_produce_supported_combinations(device):
    """Every layout emitted by the CPU tier rules must actually reconstruct native weights."""
    for K in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8):
        group = ext.exl3_moe_cpu_swizzle_group(K)
        planar = ext.exl3_moe_cpu_planar_layout(K)
        base = make_packed(16, 16, K, int(6000 + K * 10)).to(device)
        got = reconstruct(swizzle_trellis(base, group, planar), K, group, planar)
        assert torch.equal(got, reconstruct(base, K))
