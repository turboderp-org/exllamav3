# The multi-row-block GEMM shapes (TILESIZE_M > 16) must be a pure scheduling change: a row
# block's accumulation order over k is whatever the 16-row shape with the same TILESIZE_K and
# TILESIZE_N already does, so where such a partner exists the outputs have to match BIT FOR
# BIT, not within a tolerance. A tolerance-based check would pass straight through a
# reduction-order change, which is the failure this guards.
#
# The reduction staging area is per row block for the same reason, and getting that wrong
# showed up only at some bit widths and only above 24 rows, so both axes are swept.

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

# (TILESIZE_M, TILESIZE_K, TILESIZE_N) per shape index, mirroring EXL3_GEMM_SHAPE_n
SHAPES = {
    1: (16, 16, 128),
    2: (16, 32, 128),
    3: (16, 32, 256),
    4: (16, 16, 512),
    5: (32, 32, 128),
    6: (32, 16, 256),
    7: (32, 16, 128),
    8: (48, 16, 128),
    9: (64, 16, 128),
    10: (32, 16, 512),
}

# A multi-row-block shape and the 16-row shape it is a pure restriping of: same TILESIZE_K
# (same sub-k reduction) and same TILESIZE_N (same warp-to-column assignment)
IDENTICAL_PAIRS = [(5, 2), (7, 1), (8, 1), (9, 1), (10, 4)]

# Shape 6 (32, 16, 256) has no 16-row partner: the table holds no (16, 16, 256), and the
# natural M=32 twin of shape 3 -- (32, 32, 256) -- spills past the 128-register cap that
# blockDim 512 imposes and cannot be built. It is compared against shape 3, which computes
# the same output tile at a different TILESIZE_K, on a tolerance instead
NO_IDENTICAL_PARTNER = {6: 3}

MAX_M = max(m for m, _, _ in SHAPES.values())

# Spans a first row block that is full and partial, the last row block full and partial for
# every TILESIZE_M in the table, and strip boundaries above each of them
ROWS = [17, 20, 24, 25, 26, 31, 32, 33, 48, 49, 64, 65, 80, 96, 112, 127, 128, 144, 160]
SIZES = [(5120, 5120), (5120, 13824), (2048, 7168)]

_NUM_SHAPES = ext.exl3_gemm_num_kernel_shapes()
_needs_shapes = pytest.mark.skipif(
    _NUM_SHAPES < max(SHAPES), reason="no multi-row-block shapes built"
)
_needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _operands(k, n, bits, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    trellis = torch.randint(
        -32768, 32767, (k // 16, n // 16, 16 * bits), dtype=torch.int16, generator=g
    ).to(device)
    suh = (torch.randn(k, generator=g) * 0.1 + 1.0).half().to(device)
    svh = (torch.randn(n, generator=g) * 0.1 + 1.0).half().to(device)
    a = (torch.randn((max(ROWS), k), generator=g) * 0.05).half().to(device)
    return trellis, suh, svh, a


def _gemm(a, trellis, suh, svh, shape_idx, c_dtype, n):
    a = a.contiguous()
    a_had = torch.empty_like(a)
    c = torch.empty((a.shape[0], n), dtype=c_dtype, device=a.device)
    ext.exl3_gemm(a, trellis, c, suh, a_had, svh, shape_idx, False, False, 0)
    return c


@_needs_cuda
@_needs_shapes
@pytest.mark.parametrize("new_shape,ref_shape", IDENTICAL_PAIRS)
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("size", SIZES)
@torch.inference_mode()
def test_multi_row_block_matches_its_16_row_partner_bitwise(
    new_shape, ref_shape, bits, size, device="cuda:0"
):
    k, n = size
    trellis, suh, svh, a_all = _operands(k, n, bits, 11 + bits, device)
    for rows in ROWS:
        a = a_all[:rows]
        ref = _gemm(a, trellis, suh, svh, ref_shape, torch.float16, n)
        got = _gemm(a, trellis, suh, svh, new_shape, torch.float16, n)
        torch.cuda.synchronize()
        assert torch.equal(ref, got), (
            f"shape {new_shape} vs {ref_shape} bits={bits} {k}x{n} rows={rows}: "
            "not bit-identical"
        )
        # A race in the staging area shows up as run-to-run drift, which a single
        # comparison against a same-run reference can miss
        again = _gemm(a, trellis, suh, svh, new_shape, torch.float16, n)
        torch.cuda.synchronize()
        assert torch.equal(got, again), (
            f"shape {new_shape} bits={bits} {k}x{n} rows={rows}: not reproducible"
        )


@_needs_cuda
@_needs_shapes
@pytest.mark.parametrize("new_shape,ref_shape", sorted(NO_IDENTICAL_PARTNER.items()))
@pytest.mark.parametrize("bits", [2, 4, 6, 8])
@torch.inference_mode()
def test_shape_without_a_partner_agrees_within_tolerance_and_repeats(
    new_shape, ref_shape, bits, device="cuda:0"
):
    k, n = 5120, 5120
    trellis, suh, svh, a_all = _operands(k, n, bits, 11 + bits, device)
    for rows in ROWS:
        a = a_all[:rows]
        ref = _gemm(a, trellis, suh, svh, ref_shape, torch.float16, n).float()
        got = _gemm(a, trellis, suh, svh, new_shape, torch.float16, n).float()
        again = _gemm(a, trellis, suh, svh, new_shape, torch.float16, n).float()
        torch.cuda.synchronize()
        assert torch.equal(got, again), (
            f"shape {new_shape} bits={bits} rows={rows}: not reproducible"
        )
        rms = ref.pow(2).mean().sqrt().item()
        err = (got - ref).abs().max().item()
        # The two shapes reduce over k in a different order, so they differ by the same
        # fp16 accumulation slack any two TILESIZE_K values already differ by
        assert err < 0.05 * rms, (
            f"shape {new_shape} vs {ref_shape} bits={bits} rows={rows}: "
            f"max abs error {err} on output RMS {rms}"
        )


@_needs_cuda
@_needs_shapes
@pytest.mark.parametrize("new_shape,ref_shape", IDENTICAL_PAIRS)
@torch.inference_mode()
def test_multi_row_block_matches_partner_bitwise_fp32_out(
    new_shape, ref_shape, device="cuda:0"
):
    k, n, bits = 5120, 5120, 4
    trellis, suh, svh, a_all = _operands(k, n, bits, 7, device)
    for rows in [17, 32, 33, 64, 65, 144]:
        a = a_all[:rows]
        ref = _gemm(a, trellis, suh, svh, ref_shape, torch.float32, n)
        got = _gemm(a, trellis, suh, svh, new_shape, torch.float32, n)
        torch.cuda.synchronize()
        assert torch.equal(ref, got), (
            f"shape {new_shape} vs {ref_shape} rows={rows}: fp32 output not bit-identical"
        )


@_needs_shapes
def test_multi_row_block_shapes_declined_at_or_below_16_rows():
    # The <=16-row path is the hottest in the engine and must keep its existing shape set:
    # a 32-row shape there doubles the MMA count for no decode saving
    for shape_idx, (tilesize_m, _, _) in SHAPES.items():
        if tilesize_m == 16:
            continue
        for rows in (1, 5, 8, 15, 16):
            assert not ext.exl3_gemm_shape_compat(shape_idx, rows, 5120, 5120, 4), (
                f"shape {shape_idx} offered at {rows} rows"
            )
        assert ext.exl3_gemm_shape_compat(shape_idx, 17, 5120, 5120, 4)


@_needs_shapes
def test_every_multi_row_block_shape_is_covered_by_a_comparison():
    # A shape added to the table without a bit-identity partner or an explicit entry in
    # NO_IDENTICAL_PARTNER would ship with no numerics check at all
    compared = {s for s, _ in IDENTICAL_PAIRS} | set(NO_IDENTICAL_PARTNER)
    for shape_idx, (tilesize_m, _, _) in SHAPES.items():
        if tilesize_m > 16:
            assert shape_idx in compared, f"shape {shape_idx} has no numerics comparison"
    assert len(SHAPES) == _NUM_SHAPES, "SHAPES is out of step with the built shape table"
