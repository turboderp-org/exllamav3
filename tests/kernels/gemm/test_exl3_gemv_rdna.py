"""
The extended RDNA envelope of ext.exl3_gemv: integer K = 1..8 with every codebook (plain, mcg,
mul1), which the RDNA GEMV (rocm/quant/exl3_gemv_rdna.cu) accepts but the CUDA kernel does not
(kernels/gemm/test_exl3_gemv.py covers the CUDA contract: K 2..4 with mcg/mul1, K 4 with 3INST,
half-integer bitrates with mul1). Same interface, same contract otherwise: C (..., n) = A (..., k)
@ W for W = diag(suh) H W_hat H diag(svh), m == 1 for the direct binding, C fp16 or fp32, A
unmodified, repeat launches bit-identical; the multi-row route through ext.exl3_gemm covers
m 2..8 (route tag 92) and m > 8 must fall through to the cooperative GEMM.

Reference: the fp64 product with the weight from ext.reconstruct (the shared decode path, whose
codebook decode is separately tested), with a distinct packed stream per (K-slice, N-tile) so
tile-aliasing bugs cannot pass. Tolerance: the kernel accumulates in fp32 fdot2 and rounds the
rotations through fp16, so the reference sits a few fp16 ulps away; a wrong extraction, codebook
or lane index is an O(1) error.
"""
import math

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.rocm_only

CODEBOOKS = [
    ("plain", False, False),
    ("mcg", True, False),
    ("mul1", False, True),
]

# Packed words that decode to finite fp16 values for their codebook families: the mcg and
# mul1 codebooks always emit halves with exponent bits 0b11xx (finite) regardless of state,
# and the plain codebook is a fixed finite table, but the fork-validated palette is kept for
# familiarity. The reference asserts finiteness anyway so a bad word fails loudly, not subtly.
PALETTE = [0x1111, 0x2222, 0x5555, 0x2492]

_H128_CPU = None


def _hadamard_128(device):
    """Unnormalized Sylvester Hadamard matrix of order 128, as the kernels apply it
    (had_hf_r_128_inner: butterfly with sign -(lane & i), scaled by 1/sqrt(128))."""
    h = torch.ones((1, 1), dtype = torch.float64)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], dim = 1), torch.cat([h, -h], dim = 1)], dim = 0)
    return (h / math.sqrt(128.0)).to(device)


def _signs(n, device):
    return (torch.randint(0, 2, (n,), device = device) * 2 - 1).half()


def _make_trellis(K, size_k, size_n, device):
    """A distinct, finite packed stream for every (K-slice, N-tile), so aliased tile
    addressing cannot pass by accident."""
    kslices = size_k // 16
    ntiles = size_n // 16
    tile_id = torch.arange(kslices * ntiles, dtype = torch.int64).view(kslices, ntiles, 1)
    word = torch.arange(K * 16, dtype = torch.int64).view(1, 1, K * 16)
    mixed = tile_id * 1103515245 + word * 12345 + tile_id * word * 2654435761
    choices = ((mixed ^ (mixed >> 16)) >> 8) & 3
    # The first eight base-4 digits encode tile_id, guaranteeing distinct streams for
    # every shape in this suite while retaining only known-finite packed words.
    choices = torch.where(word < 8, (tile_id >> (2 * word)) & 3, choices)
    palette = torch.tensor(PALETTE, dtype = torch.int16)
    return palette[choices].to(device)


def _fp64_reference(A, trellis, suh, svh, mcg, mul1, device):
    """((A . suh) @ H) @ W in float64, then (@ H) . svh, with W from the shared dequantizer."""
    K = trellis.size(2) // 16
    size_k = A.size(-1)
    size_n = trellis.size(1) * 16
    weight = torch.empty((size_k, size_n), dtype = torch.half, device = device)
    ext.reconstruct(weight, trellis, float(K), mcg, mul1)
    assert torch.isfinite(weight).all(), (K, mcg, mul1)
    h = _hadamard_128(device)
    a = A.reshape(-1, size_k).double() * suh.double()
    a = (a.reshape(-1, 128) @ h).reshape(-1, size_k)
    y = a @ weight.double()
    y = (y.reshape(-1, 128) @ h).reshape(-1, size_n) * svh.double()
    return y.reshape(A.shape[:-1] + (size_n,))


def _run_direct_gemv(A, trellis, suh, svh, mcg, mul1, device, out_dtype = torch.float16):
    size_n = trellis.size(1) * 16
    c = torch.empty(A.shape[:-1] + (size_n,), dtype = out_dtype, device = device)
    a_had = torch.empty_like(A)
    ext.exl3_gemv(A, trellis, c, suh, a_had, svh, mcg, mul1)
    return c


def _assert_matches_oracle(actual, expected, atol = 0.03):
    assert torch.isfinite(actual).all()
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol = 0, atol = atol)


# ---------------------------------------------------------------------------------------------
# Binding presence: the GEMV route is the point of the suite, so a missing binding is a
# failure, not a skip
# ---------------------------------------------------------------------------------------------

def test_gemv_bindings_present(device):
    for name in ("exl3_gemv", "exl3_gemm", "reconstruct", "had_r_128"):
        assert hasattr(ext, name), f"missing binding: {name}"


# ---------------------------------------------------------------------------------------------
# Full (K, codebook) matrix through the direct binding: every integer bitrate 1..8 against
# every codebook, the contract dev's RDNA kernel actually supports (the fork's GEMV covered
# a gfx12-gated subset; dev has no gating to port)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cb_name, mcg, mul1", CODEBOOKS, ids = [c[0] for c in CODEBOOKS])
@pytest.mark.parametrize("K", range(1, 9), ids = lambda k: f"k{k}")
@torch.inference_mode()
def test_gemv_direct_full_codebook_matrix(device, K, cb_name, mcg, mul1):
    torch.manual_seed(1234 + K + 10 * int(mcg) + 100 * int(mul1))
    size_k = size_n = 128
    trellis = _make_trellis(K, size_k, size_n, device)
    suh, svh = _signs(size_k, device), _signs(size_n, device)
    a = torch.randn((1, size_k), dtype = torch.half, device = device) * 1e-3

    actual = _run_direct_gemv(a, trellis, suh, svh, mcg, mul1, device)
    expected = _fp64_reference(a, trellis, suh, svh, mcg, mul1, device)
    _assert_matches_oracle(actual, expected)


# Distinct tiles must survive dequantization: if every 16x16 tile decoded identically, an
# N-tile addressing bug would be invisible to the matrix above
@torch.inference_mode()
def test_reconstructed_tiles_differ(device):
    trellis = _make_trellis(8, 256, 256, device)
    weight = torch.empty((256, 256), dtype = torch.half, device = device)
    ext.reconstruct(weight, trellis, 8.0, True, False)
    tiles = [weight[:16, :16], weight[16:32, :16], weight[:16, 16:32]]
    assert all(not torch.equal(x, y) for i, x in enumerate(tiles) for y in tiles[i + 1:])


# ---------------------------------------------------------------------------------------------
# Output dtype: C may be fp16 or fp32 (c_fp32 selects the epilogue store)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32], ids = ["fp16", "fp32"])
@pytest.mark.parametrize("cb_name, mcg, mul1", CODEBOOKS, ids = [c[0] for c in CODEBOOKS])
@pytest.mark.parametrize("K", [2, 5, 8], ids = lambda k: f"k{k}")
@torch.inference_mode()
def test_gemv_direct_output_dtypes(device, K, cb_name, mcg, mul1, out_dtype):
    torch.manual_seed(3000 + K + 10 * int(mcg) + 100 * int(mul1))
    size_k = size_n = 256
    trellis = _make_trellis(K, size_k, size_n, device)
    suh, svh = _signs(size_k, device), _signs(size_n, device)
    a = torch.randn((1, size_k), dtype = torch.half, device = device) * 1e-3

    actual = _run_direct_gemv(a, trellis, suh, svh, mcg, mul1, device, out_dtype)
    assert actual.dtype == out_dtype
    expected = _fp64_reference(a, trellis, suh, svh, mcg, mul1, device)
    _assert_matches_oracle(actual, expected)


# ---------------------------------------------------------------------------------------------
# Launch-shape sweep over the direct binding: the wave selection is 16 warps up to 32 N-tiles,
# 8 warps up to 96 tiles (k <= 4096), else 4; the split-K form runs while N-tiles fit its
# window. All three wave counts and both single/split-K forms must agree with the oracle.
# ---------------------------------------------------------------------------------------------

GEMV_SHAPES = [
    (128, 128),     # 8 tiles: 16-warp blocks, split-K (4 warps)
    (4096, 1536),   # 96 tiles, k_blocks = 256: 8-warp blocks, split-K (16 warps)
    (2048, 4096),   # 256 tiles: 4-warp blocks, split-K (4 warps)
]

@pytest.mark.parametrize("cb_name, mcg, mul1", CODEBOOKS, ids = [c[0] for c in CODEBOOKS])
@pytest.mark.parametrize("size_k, size_n", GEMV_SHAPES, ids = ["narrow", "wide-k", "wide-n"])
@torch.inference_mode()
def test_gemv_direct_launch_shapes(device, size_k, size_n, cb_name, mcg, mul1):
    torch.manual_seed(4000 + size_k + size_n + 10 * int(mcg) + 100 * int(mul1))
    K = 4 if cb_name == "plain" else 3
    trellis = _make_trellis(K, size_k, size_n, device)
    suh, svh = _signs(size_k, device), _signs(size_n, device)
    a = torch.randn((1, size_k), dtype = torch.half, device = device) * 1e-3

    actual = _run_direct_gemv(a, trellis, suh, svh, mcg, mul1, device)
    expected = _fp64_reference(a, trellis, suh, svh, mcg, mul1, device)
    _assert_matches_oracle(actual, expected)


# ---------------------------------------------------------------------------------------------
# The direct binding accepts arbitrary-rank leading dims as long as their product is 1
# ---------------------------------------------------------------------------------------------

@torch.inference_mode()
def test_gemv_direct_accepts_rank3_inputs(device):
    torch.manual_seed(5)
    size_k = size_n = 128
    trellis = _make_trellis(2, size_k, size_n, device)
    suh, svh = _signs(size_k, device), _signs(size_n, device)
    a = torch.randn((1, 1, size_k), dtype = torch.half, device = device) * 1e-3

    actual = _run_direct_gemv(a, trellis, suh, svh, False, True, device)
    expected = _fp64_reference(a, trellis, suh, svh, False, True, device)
    _assert_matches_oracle(actual, expected)


# ---------------------------------------------------------------------------------------------
# Row-count modes: m = 1..8 all route to the RDNA multi-row GEMV through exl3_gemm (tag 92)
# and must match the fp64 oracle row for row
# ---------------------------------------------------------------------------------------------

def _run_gemm_route(A, trellis, suh, svh, mcg, mul1, device, out_dtype = torch.float16):
    size_k, size_n = A.size(-1), trellis.size(1) * 16
    c = torch.empty(A.shape[:-1] + (size_n,), dtype = out_dtype, device = device)
    a_had = torch.empty_like(A)
    tag = ext.exl3_gemm(A, trellis, c, suh, a_had, svh, 0, mcg, mul1, 0)
    return tag, c


@pytest.mark.parametrize("cb_name, mcg, mul1", CODEBOOKS, ids = [c[0] for c in CODEBOOKS])
@pytest.mark.parametrize("rows", range(1, 9), ids = lambda r: f"m{r}")
@torch.inference_mode()
def test_gemv_multirow_routes_and_matches(device, rows, cb_name, mcg, mul1):
    torch.manual_seed(6000 + rows + 10 * int(mcg) + 100 * int(mul1))
    K = {"plain": 4, "mcg": 2, "mul1": 6}[cb_name]
    size_k = size_n = 128
    trellis = _make_trellis(K, size_k, size_n, device)
    suh, svh = _signs(size_k, device), _signs(size_n, device)
    a = torch.randn((1, rows, size_k), dtype = torch.half, device = device) * 1e-3

    tag, actual = _run_gemm_route(a, trellis, suh, svh, mcg, mul1, device)
    assert tag == 92, f"m={rows}: expected the multi-row GEMV route (92), got {tag}"
    expected = _fp64_reference(a, trellis, suh, svh, mcg, mul1, device)
    _assert_matches_oracle(actual, expected)


# The wider shapes pick the 16- and 8-warp split-K variants of the multi-row dispatch
@pytest.mark.parametrize("size_k, size_n", [(4096, 512), (4096, 2048)], ids = ["warps16", "warps8"])
@pytest.mark.parametrize("rows", [1, 3, 8], ids = ["m1", "m3", "m8"])
@torch.inference_mode()
def test_gemv_multirow_warp_variants(device, rows, size_k, size_n):
    torch.manual_seed(7000 + rows + size_k + size_n)
    trellis = _make_trellis(3, size_k, size_n, device)
    suh, svh = _signs(size_k, device), _signs(size_n, device)
    a = torch.randn((1, rows, size_k), dtype = torch.half, device = device) * 1e-3

    tag, actual = _run_gemm_route(a, trellis, suh, svh, False, True, device)
    assert tag == 92, f"m={rows} ({size_k}x{size_n}): expected route 92, got {tag}"
    expected = _fp64_reference(a, trellis, suh, svh, False, True, device)
    _assert_matches_oracle(actual, expected)


# Beyond the multi-row cap (m > 8) the call must fall through to the cooperative GEMM and
# still be numerically right
@torch.inference_mode()
def test_gemv_m9_bypasses_gemv_routes(device):
    torch.manual_seed(9)
    size_k = size_n = 512
    trellis = _make_trellis(4, size_k, size_n, device)
    suh, svh = _signs(size_k, device), _signs(size_n, device)
    a = torch.randn((1, 9, size_k), dtype = torch.half, device = device) * 1e-3

    tag, actual = _run_gemm_route(a, trellis, suh, svh, False, False, device)
    assert tag not in (90, 91, 92), f"m=9 must not take a GEMV route, got {tag}"
    expected = _fp64_reference(a, trellis, suh, svh, False, False, device)
    _assert_matches_oracle(actual, expected, atol = 0.05)


# ---------------------------------------------------------------------------------------------
# Contract violations the binding rejects host-side
# ---------------------------------------------------------------------------------------------

@torch.inference_mode()
def test_gemv_rejects_half_integer_bitrate(device):
    a = torch.randn((1, 128), dtype = torch.half, device = device) * 1e-3
    trellis = torch.full((8, 8, 24), 0x1111, dtype = torch.int16, device = device)  # K = 1.5
    suh, svh = _signs(128, device), _signs(128, device)
    a_had = torch.empty_like(a)
    c = torch.empty((1, 128), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "half-integer bitrates"):
        ext.exl3_gemv(a, trellis, c, suh, a_had, svh, False, True)


@torch.inference_mode()
def test_gemv_rejects_k9(device):
    a = torch.randn((1, 128), dtype = torch.half, device = device) * 1e-3
    trellis = torch.full((8, 8, 144), 0x1111, dtype = torch.int16, device = device)  # K = 9
    suh, svh = _signs(128, device), _signs(128, device)
    a_had = torch.empty_like(a)
    c = torch.empty((1, 128), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "not eligible"):
        ext.exl3_gemv(a, trellis, c, suh, a_had, svh, False, False)


@torch.inference_mode()
def test_gemv_rejects_unaligned_k(device):
    a = torch.randn((1, 192), dtype = torch.half, device = device) * 1e-3  # 192 % 128 != 0
    trellis = torch.full((12, 8, 64), 0x1111, dtype = torch.int16, device = device)
    suh, svh = _signs(192, device), _signs(128, device)
    a_had = torch.empty_like(a)
    c = torch.empty((1, 128), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "not eligible"):
        ext.exl3_gemv(a, trellis, c, suh, a_had, svh, False, False)


@torch.inference_mode()
def test_gemv_rejects_both_codebooks(device):
    a = torch.randn((1, 128), dtype = torch.half, device = device) * 1e-3
    trellis = torch.full((8, 8, 64), 0x1111, dtype = torch.int16, device = device)
    suh, svh = _signs(128, device), _signs(128, device)
    a_had = torch.empty_like(a)
    c = torch.empty((1, 128), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "Specified both mcg and mul1"):
        ext.exl3_gemv(a, trellis, c, suh, a_had, svh, True, True)


@torch.inference_mode()
def test_gemv_requires_all_transforms(device):
    a = torch.randn((1, 128), dtype = torch.half, device = device) * 1e-3
    trellis = torch.full((8, 8, 32), 0x1111, dtype = torch.int16, device = device)
    suh, svh = _signs(128, device), _signs(128, device)
    a_had = torch.empty_like(a)
    c = torch.empty((1, 128), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "requires suh, A_had and svh"):
        ext.exl3_gemv(a, trellis, c, suh, a_had, None, False, False)

