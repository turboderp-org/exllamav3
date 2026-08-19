"""Triton implementation of the EXL3 linear layer (fused dequant + GEMM).

An opt-in alternative to the C++ BC/reconstruct paths, selected per layer with
the EXL3_PREFER_TRITON_LINEAR=1 environment variable (see LinearEXL3.forward).
Where the BC_* classes are unavailable (e.g. ROCm builds) this is the fast
path; on CUDA it is an alternative for comparison and benchmarking.

Three entry points, called directly like every other Triton path in this
project (no torch.library registration — the dispatcher overhead is not wanted
on the decode path):

    had_r_128_triton(x, y, suh, None, 1.0)   # row Hadamard transform
    exl3_gemm_triton(xh, t, y, ...)          # fused dequant + GEMM
    linear_exl3_triton(...)                  # full linear forward: hadamard
                                             # -> fused dequant-gemm -> hadamard
                                             # (+ optional bias)

The fused kernel decodes the EXL3 trellis tile-by-tile inside the K-loop
without materializing the weight matrix. Every bit width K = 1..8 has a
dedicated M==1 fast decode whose per-element word/shift lookup is realized
without data-dependent gathers (linear/affine u32 row loads + static-reshape
permutations, or per-(row, c>>3) constexpr window offsets for the odd
widths), so packed rows load as linear u32 vectors. Non-divisible shapes
fall back to a generic staged-row tl.gather decode that covers all widths.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

    # Triton dummy shims so importing this module doesn't fail when Triton is
    # unavailable (mirrors attention_fn/triton_paged.py)
    class _DummyTritonLanguage:
        constexpr = object()

    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn

    triton = _DummyTriton()
    tl = _DummyTritonLanguage()

# ---------------------------------------------------------------------------
# Triton Hadamard transform (128-element rows, radix-2 butterfly)
#
# Mirrors the C++ had_hf/had_ff_r_128 kernels exactly: the transform is
# evaluated in fp32 regardless of I/O dtype (with a single round to half at
# the very end for half output), r_scale = scale / sqrt(128) is applied in
# fp32 after the transform, and pre/post scales are applied in the I/O
# dtype. The butterfly runs sequentially over masks 1..64, which reproduces
# the C++ expression tree (4-point transform in registers, then 32-lane
# xor-shuffles) term-for-term, so results are bit-identical.
# ---------------------------------------------------------------------------

@triton.jit
def _had_stage(v, BLOCK_R: tl.constexpr, SPAN: tl.constexpr):
    """One radix-2 butterfly stage over a [BLOCK_R, 128] fp32 tile.

    Elements whose bit log2(SPAN) is 0 receive a+b, those with the bit set
    receive a-b, where b is the partner element at distance SPAN.
    """
    G: tl.constexpr = 128 // (2 * SPAN)
    pair = tl.permute(v.reshape(BLOCK_R, G, 2, SPAN), (0, 1, 3, 2))
    lo, hi = tl.split(pair)
    pair = tl.join(lo + hi, lo - hi)
    return tl.permute(pair, (0, 1, 3, 2)).reshape(BLOCK_R, 128)


@triton.jit
def _had_r_128_kernel(
    x_ptr, y_ptr, scale_ptr,
    n_rows,
    stride_xr, stride_yr,
    r_scale,
    IO_FP32: tl.constexpr,
    PRE_SCALED: tl.constexpr,
    POST_SCALED: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # One program transforms a [BLOCK_R, 128] tile: pid_m rows, pid_c the
    # 128-column block within each row. Scales are indexed by flat column
    # position (row-independent), matching the C++ kernel.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_m * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = rows < n_rows
    col = tl.arange(0, 128)

    x = tl.load(
        x_ptr + rows[:, None] * stride_xr + (pid_c * 128 + col)[None, :],
        mask=mask_r[:, None], other=0.0,
    )

    # Pre-scale, applied in the I/O dtype (half multiply for the half path,
    # exactly like the C++ __hmul2 version)
    if PRE_SCALED:
        pre = tl.load(scale_ptr + pid_c * 128 + col)
        if IO_FP32:
            x = x * pre.to(tl.float32)
        else:
            x = x * pre

    # fp32 radix-2 butterfly over bit masks 1..64, unrolled via a constexpr
    # helper. Each stage pairs element j with j^span: the bit-0 element gets
    # a+b, the bit-1 element a-b — the same expression tree as the C++
    # register H4 + xor-shuffle network, so the result is bit-identical.
    v = x.to(tl.float32)
    v = _had_stage(v, BLOCK_R, 1)
    v = _had_stage(v, BLOCK_R, 2)
    v = _had_stage(v, BLOCK_R, 4)
    v = _had_stage(v, BLOCK_R, 8)
    v = _had_stage(v, BLOCK_R, 16)
    v = _had_stage(v, BLOCK_R, 32)
    v = _had_stage(v, BLOCK_R, 64)
    v = v * r_scale

    # Post-scale. The C++ half kernel rounds the scaled transform to half
    # first and then multiplies in half; reproduce that ordering exactly.
    if POST_SCALED:
        post = tl.load(scale_ptr + pid_c * 128 + col)
        if IO_FP32:
            out = v * post.to(tl.float32)
        else:
            out = v.to(x_ptr.dtype.element_ty) * post
    else:
        out = v

    tl.store(
        y_ptr + rows[:, None] * stride_yr + (pid_c * 128 + col)[None, :],
        out.to(y_ptr.dtype.element_ty),
        mask=mask_r[:, None],
    )

# ---------------------------------------------------------------------------
# had_r_128_triton: Triton row Hadamard transform
#
# A Triton twin of the C++ pybind ``ext.had_r_128`` (quant/hadamard.cu),
# bit-identical by construction. Used ONLY inside the Triton EXL3 linear
# path so that path depends on Triton alone; every other caller keeps
# using the C++ kernel through pybind.
#
# Scope note: its kernel time matches or beats the C++ kernel at every shape
# (measured via CUDA-graph replay), but an EAGER call from Python pays the
# Python launch path. It exists for the graph-captured path, where launch
# overhead does not apply; don't adopt it on uncaptured hot paths.
# ---------------------------------------------------------------------------

_RSCALE_128 = 0.088388347648  # 1/sqrt(128), matches the C++ literal


def had_r_128_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    pre_scale: torch.Tensor | None,
    post_scale: torch.Tensor | None,
    scale: float,
) -> None:
    """y = (x.view(-1, 128) @ H128) * (pre|post)_scale, scaled by scale/sqrt(128).

    Matches the C++ ``had_r_128`` contract: input/output must be 2D, the same
    dtype (half or float), contiguous in the last dim, with last dim a
    multiple of 128; scales are half tensors with one element per column
    (flat, row-independent). Pre-scale multiplies before the transform in the
    I/O dtype; post-scale multiplies after it (for half output, after the
    round to half), like the C++ kernels.
    """
    assert input.dtype == output.dtype, "had_r_128_triton: input/output dtype mismatch"
    assert input.dtype in (torch.half, torch.float), \
        f"had_r_128_triton: unsupported dtype {input.dtype}"
    assert input.dim() == 2 and input.shape[-1] % 128 == 0
    # The kernel indexes the last dim with an implicit unit stride
    assert input.stride(-1) == 1, \
        f"had_r_128_triton: input last dim must be contiguous, got stride {input.stride(-1)}"
    assert output.stride(-1) == 1, \
        f"had_r_128_triton: output last dim must be contiguous, got stride {output.stride(-1)}"
    assert (pre_scale is None) or (post_scale is None)
    rows, cols = input.shape

    # Tiling (swept on RDNA3 via graph-of-64 replay timing): BLOCK_R=4 with a
    # single warp is optimal or tied-for-optimal at every shape from rows==1
    # (decode) through rows==512+ (prefill). The 128-wide tile leaves extra
    # warps idle; wider row tiles only help shapes too small to matter.
    BLOCK_R = 4
    num_warps = 1

    grid = (triton.cdiv(rows, BLOCK_R), cols // 128)
    _had_r_128_kernel[grid](
        input, output,
        pre_scale if pre_scale is not None else post_scale,
        rows,
        input.stride(0), output.stride(0),
        scale * _RSCALE_128,
        IO_FP32=input.dtype == torch.float,
        PRE_SCALED=pre_scale is not None,
        POST_SCALED=post_scale is not None,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# EXL3 dequantization in pure PyTorch
# ---------------------------------------------------------------------------

_TENSOR_CORE_PERM = None
_TENSOR_CORE_PERM_I = None

def _get_perm(device):
    global _TENSOR_CORE_PERM, _TENSOR_CORE_PERM_I
    if _TENSOR_CORE_PERM is None or _TENSOR_CORE_PERM.device != device:
        perm = [0] * 256
        for t in range(32):
            r0 = (t % 4) * 2; r1 = r0 + 1; r2 = r0 + 8; r3 = r0 + 9
            c0 = t // 4; c1 = c0 + 8
            perm[t*8+0] = r0*16+c0; perm[t*8+1] = r1*16+c0
            perm[t*8+2] = r2*16+c0; perm[t*8+3] = r3*16+c0
            perm[t*8+4] = r0*16+c1; perm[t*8+5] = r1*16+c1
            perm[t*8+6] = r2*16+c1; perm[t*8+7] = r3*16+c1
        perm_i = [0]*256
        for i, p in enumerate(perm):
            perm_i[p] = i
        _TENSOR_CORE_PERM = torch.tensor(perm, device=device, dtype=torch.long)
        _TENSOR_CORE_PERM_I = torch.tensor(perm_i, device=device, dtype=torch.long)
    return _TENSOR_CORE_PERM_I


_DQ_CACHE = {}
_LUT_CACHE = {}

# Per-row window offsets for the odd bit widths (K = 3, 5, 7).
#
# For those widths the per-element (word, shift) lookup does not factor into
# independent per-axis bit fields (the window end falls inside a code, so the
# word index and the funnel shift carry into each other). Instead each of the
# 32 (r, c//8) combinations of a 16x16 sub-tile has one fixed window offset
# D(r, c3) = 32*f + sh: element (r, c) reads the 16-bit code window starting
# at stream bit 32*(K_BITS*(c%8) + f) + sh, i.e. word (K_BITS*(c%8) + f) of
# the subtile at funnel shift sh (neighbor word -1 when sh > 16). These
# tables were recovered element-exactly from the C++ reconstruct kernel by
# differential probing and verified to reproduce its output bit-for-bit.
_M_ROW_OFFSETS = {
    3: [29, 17, 26, 14, 5, 57, 2, 54, 45, 33, 42, 94, 85, 73, 82, 70,
        23, 11, 20, 8, 63, 51, 60, 48, 39, 91, 36, 88, 79, 67, 76, 64],
    5: [27, 7, 22, 2, 51, 95, 46, 90, 75, 119, 70, 114, 99, 143, 158, 138,
        17, 61, 12, 56, 41, 85, 36, 80, 65, 109, 124, 104, 153, 133, 148, 128],
    7: [25, 61, 18, 54, 33, 69, 90, 126, 105, 141, 98, 134, 177, 213, 170, 206,
        11, 47, 4, 40, 83, 119, 76, 112, 155, 191, 148, 184, 163, 199, 220, 192],
}
_M_ROW_CACHE = {}


def _get_m_row_offsets(K_bits: int, device) -> torch.Tensor:
    key = (K_bits, str(device))
    if key not in _M_ROW_CACHE:
        _M_ROW_CACHE[key] = torch.tensor(
            _M_ROW_OFFSETS[K_bits], device=device, dtype=torch.int32
        )
    return _M_ROW_CACHE[key]


def _decode_lut(cb: int, device) -> torch.Tensor:
    key = (cb, str(device))
    if key not in _LUT_CACHE:
        x = torch.arange(65536, device=device, dtype=torch.int64)
        M = 0xFFFFFFFF
        if cb == 0:
            x = (x * 89226354) & M; x = (x + 64248484) & M
            x = 0x3b603b60 ^ (x & 0x8fff8fff)
            lo = (x & 0xFFFF).to(torch.int16).view(torch.float16)
            hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
            lut = lo + hi
        elif cb == 1:
            x = (x * 0xCBAC1FED) & M
            x = 0x3b603b60 ^ (x & 0x8fff8fff)
            lo = (x & 0xFFFF).to(torch.int16).view(torch.float16)
            hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
            lut = lo + hi
        elif cb == 2:
            x = (x * 0x83DCD12D) & M
            acc = torch.full_like(x, 0x6400)
            s = (acc + (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)) & 0xFFFF
            sum_h = s.to(torch.int16).view(torch.float16)
            k_inv = torch.tensor([0x1eee], dtype=torch.int16, device=device).view(torch.float16)
            k_bias_data = torch.tensor([0xc931], dtype=torch.int32, device=device).to(torch.int16).view(torch.float16)
            lut = sum_h * k_inv + k_bias_data
        _LUT_CACHE[key] = lut
    return _LUT_CACHE[key]


# ---------------------------------------------------------------------------
# M == 1 split-K GEMV plan (starved-N shapes)
#
# The decode GEMV's only lever on a CTA-starved shape (N/BLOCK_N under one
# wave, e.g. MLP down_proj N=4096-5120) is memory-level parallelism: CTAs x
# staged bytes in flight. Splitting the K loop across SPLITS CTAs per N tile
# multiplies the CTA count without shrinking the staged window, at the cost
# of SPLITS x N fp32 partials (tens of KB) and one tiny reduce kernel that
# also performs the output Hadamard transform (fusing away the separate
# had_r_128 launch for these linears).
#
# Numerics: the partial sums are fp32 and the reduce runs the butterfly in
# fp32, so the split path skips the one intermediate round-to-half the
# classic path takes between GEMV and Hadamard (slightly more accurate, same
# 2e-2 relative tolerance regime; accumulation order within fp32 changes).
# ---------------------------------------------------------------------------

_SPLITK_BUFS: dict = {}


def _m1_splitk_plan(M: int, N: int, K_dim: int, K_bits: int) -> int:
    """Number of K-splits for an M == 1 invocation, or 1 (classic path).

    Split-K applies only where the bits=4 fast path is guaranteed for the
    whole autotune pool (N and K divisible by 256 covers BLOCK_K up to 256),
    the shape is CTA-starved at any pool tile (small N), and each split
    still gets a meaningful K slice. EXL3_SPLITK=off (or =n) overrides for
    experiments; default behavior needs no environment variable.

    Splits scale with the K depth (measured on the MLP down_proj shapes,
    L2-cold layer sweeps, composite GB/s incl. hadamards + reduce):
    9B  N=4096  K=12288: 242 -> 445 (S=4) / 455 (S=8, BN32/BK256)
    27B N=5120  K=17408: 239 -> 353 (S=4) / 385 (S=8, BN32/BK256)
    """
    import os
    if M != 1 or K_bits != 4 or N > 8192 or N % 256 or K_dim % 256:
        return 1
    k_tiles = K_dim // 16
    splits = 8 if k_tiles >= 512 else (4 if k_tiles >= 256 else 1)
    env = os.environ.get("EXL3_SPLITK")
    if env is not None:
        splits = 0 if env.lower() in ("off", "0", "none") else int(env)
    # Every split needs at least ~4 outer iterations of a BK256 tile.
    if k_tiles < splits * 64:
        return 1
    return max(splits, 1)


def _get_splitk_buf(N: int, splits: int, device) -> torch.Tensor:
    key = (N, splits, str(device))
    buf = _SPLITK_BUFS.get(key)
    if buf is None:
        buf = torch.empty((splits, N), dtype=torch.float, device=device)
        _SPLITK_BUFS[key] = buf
    return buf


@triton.jit
def _m1_split_reduce_had_kernel(
    partials_ptr, y_ptr, scale_ptr,
    N, stride_ps, stride_yn,
    r_scale,
    SPLITS: tl.constexpr,
    IO_FP32: tl.constexpr,
):
    """Sum the split-K partials for one 128-column block and apply the output
    Hadamard transform + post-scale, reproducing _had_r_128_kernel's fp32
    butterfly and rounding order exactly (scale 1.0)."""
    pid = tl.program_id(0)
    col = pid * 128 + tl.arange(0, 128)
    acc = tl.zeros((128,), dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        acc += tl.load(partials_ptr + s * stride_ps + col)
    v = tl.reshape(acc, (1, 128))
    v = _had_stage(v, 1, 1)
    v = _had_stage(v, 1, 2)
    v = _had_stage(v, 1, 4)
    v = _had_stage(v, 1, 8)
    v = _had_stage(v, 1, 16)
    v = _had_stage(v, 1, 32)
    v = _had_stage(v, 1, 64)
    v = tl.reshape(v, (128,)) * r_scale
    post = tl.load(scale_ptr + col)
    if IO_FP32:
        out = v * post.to(tl.float32)
    else:
        out = v.to(y_ptr.dtype.element_ty) * post
    tl.store(y_ptr + col * stride_yn, out.to(y_ptr.dtype.element_ty))


def _m1_split_reduce_had(
    partials: torch.Tensor, y: torch.Tensor, post_scale: torch.Tensor, splits: int,
) -> None:
    N = partials.shape[1]
    assert y.stride(-1) == 1, "split reduce: output must be contiguous in the last dim"
    _m1_split_reduce_had_kernel[(N // 128,)](
        partials, y, post_scale,
        N, partials.stride(0), y.stride(-1),
        _RSCALE_128,
        SPLITS=splits,
        IO_FP32=y.dtype == torch.float,
        num_warps=1,
    )


# ---------------------------------------------------------------------------
# Triton matmul kernel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fused dequant + GEMM Triton kernel
#
# Each block computes a [BLOCK_M, BLOCK_N] output tile. The weight tile
# [BLOCK_K, BLOCK_N] is decoded on-the-fly from the compressed trellis every
# K-iteration (no full weight matrix is ever materialized).
#
# Memory strategy (mirrors the C++ kernel's sh_b staging): for every 16x16
# trellis sub-tile, the packed words of all covered n-sub-tile columns are
# fetched with linear, dword-vectorized u32 loads (128 B per sub-tile row,
# contiguous in the trellis), never scattered scalar gathers.
#
# Fast paths (full tiles): every bit width decodes the full [shift, word]
# code table with NO data-dependent gather, so each sub-tile weight is
# computed exactly once:
# - K = 1/2/4/8: the 16x16 sub-tile permutation factors into per-axis index
#   bits (see the kernel branches), realized by static reshape/permute
#   (tensor-core path) or folded into a broadcast pattern of the x vector
#   (M==1 GEMV path).
# - K = 3/5/7: per-(row, c>>3) window-offset tables (_M_ROW_OFFSETS) with
#   affine word-slice loads.
# - K = 6: four word-slice loads + the _funnel6 word-pair decode.
# For M == 1 the decoded tile is reduced in fp32 with the product tile
# accumulated elementwise over the whole K loop, so the k-loop issues no
# cross-lane reductions; the result is summed once per block at the end.
#
# Generic path (non-divisible tiles): staged row load + tl.gather decode,
# which lowers to LDS (shared-memory) reads.
# ---------------------------------------------------------------------------


def _exl3_gemm_early_prune(configs, named_args, **kwargs):
    """Restrict the config set per invocation:
    - Other bit widths run the generic tl.gather path, where this Triton
      build's LLVM aborts on large gather tiles.
    - Shapes whose N/K are not divisible by a config's tile fall back to the
      generic path too, so apply the same cap there.
    - M == 1 never benefits from BLOCK_M > 16 (the grid is a single block and
      the GEMV branch ignores BLOCK_M); pruning them also works around Triton
      compile failures for some narrow-BLOCK_N decode tiles at high warp
      counts.
    - M == 1 full-tile shapes are bucketed by N (RDNA3 starved-N rule): with
      N/BLOCK_N CTAs under ~one wave (48 CUs), only small-BLOCK_N tiles have
      enough parallelism, and at large N the wide tiles amortize the staged
      decode better. Every pool member was measured at or above the previous
      default pick's rate on its bucket's shapes, so a cold-clock autotune
      pass cannot lock in a regression.
    Every bit width has a gather-free fast path that handles the large tiles."""
    bits = kwargs.get("K_BITS", named_args.get("K_BITS"))
    n = kwargs.get("N", named_args.get("N"))
    k = kwargs.get("K_dim", named_args.get("K_dim"))
    m = kwargs.get("M", named_args.get("M"))
    fast_ok = n % 128 == 0 and k % 128 == 0
    if bits == 1:
        # This Triton build's TTGIR pass crashes on the K_BITS=1 decode tiles
        # ([*, 8] per sub-tile) with 8 warps at BLOCK_N=16.
        configs = [c for c in configs if not (c.num_warps >= 8 and c.kwargs["BLOCK_N"] <= 16)]
    if m == 1:
        out = [c for c in configs if c.kwargs["BLOCK_M"] == 16]
        if bits != 3:
            # The BLOCK_N=32/K=32 pair exists only for the bits=3 M1 run-funnel
            # decode; other widths never see it (identical autotune behaviour
            # to before it was added).
            out = [c for c in out if not (c.kwargs["BLOCK_N"] == 32 and c.kwargs["BLOCK_K"] == 32)]
            if fast_ok:
                # N-bucketed decode pools (bits != 3). Small N (CTA-starved,
                # e.g. MLP down_proj N=4096-5120 at BN128 = 32-40 CTAs): the
                # BN32 tiles restore CTA count at equal staged bytes and were
                # measured 251-280 GB/s where BN128/BK128 gives 239-244. Large
                # N (gate/up, lm_head): BN128/BK128 plus the BN64 tiles
                # (BN64/BK256 329-425 GB/s vs 321-410 for BN128 alone; the
                # bits=6 lm_head stream prefers BN64/BK128, 685 vs 617 GB/s).
                # BN64/BK128 stays in both pools: it is the weakest b4 member
                # for starved-N shapes (235-239 GB/s, at/below the old pick)
                # but the best tile for the bits=6 M1 stream and a pre-existing
                # option for the rarer widths, so autotune keeps today's
                # expected selections there.
                if (bits == 4 and n <= 8192 and n % 256 == 0
                        and k % 256 == 0 and k // 16 >= 256):
                    # Split-K-eligible bits=4 shape (see _m1_splitk_plan): the
                    # CTA count comes from the K splits, so the widest windows
                    # win outright (measured at S=8: BN32/BK256 383-455 GB/s,
                    # BN64/BK256 386-452, vs 367-375 for the BK128 tiles).
                    pool = ((32, 256), (64, 256))
                else:
                    pool = (((32, 128), (32, 256), (64, 128)) if n <= 8192
                            else ((128, 128), (64, 256), (64, 128)))
                out = [c for c in out
                       if (c.kwargs["BLOCK_N"], c.kwargs["BLOCK_K"]) in pool]
                return out if out else configs
        else:
            # bits=3 run-funnel decode: BLOCK_N=128 tiles collapse to the old
            # per-code path's rate (measured 229 GB/s at 1x5120x248320 vs 477+
            # for the narrow tiles), so keep the M1 list tight enough that a
            # cold-clock autotune pass cannot lock one in.
            out = [c for c in out if c.kwargs["BLOCK_N"] != 128]
        if not fast_ok:
            out = [c for c in out if c.kwargs["BLOCK_N"] <= 64 and c.kwargs["BLOCK_K"] <= 64]
        return out if out else configs
    if not fast_ok:
        small = [c for c in configs if c.kwargs["BLOCK_N"] <= 64 and c.kwargs["BLOCK_K"] <= 64]
        return small if small else configs
    return configs


def _exl3_gemm_configs():
    import os
    cfg_spec = os.environ.get("EXL3_GEMM_CONFIGS")
    if cfg_spec:
        # Format: "BM,BN,BK,GM:nw:ns;..." for quick manual sweeps.
        configs = []
        for part in cfg_spec.split(";"):
            part = part.strip()
            if not part:
                continue
            dims, _, rest = part.partition(":")
            bm, bn, bk, gm = (int(x) for x in dims.split(","))
            nw = int(rest.split(":")[0]) if rest else 4
            ns = int(rest.split(":")[1]) if ":" in rest else 3
            configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm}, num_warps=nw, num_stages=ns))
        return configs
    # Default autotune set. The decode path (M=1, bits=4) is pure bandwidth:
    # wide N/K tiles amortize the staged decode, and the no-dot reduction
    # path removes the tensor-core shape constraint. All M==1 configs are
    # within a few percent of each other at operating clocks, so a cold-clock
    # autotune pass cannot lock in a slow one.
    #
    # M == 1 (decode GEMV). The per-N pools are enforced by
    # _exl3_gemm_early_prune: starved-N shapes (down_proj-class, N/BLOCK_N
    # under one wave on 48 CUs) get the BN32 tiles; large-N shapes (gate/up,
    # lm_head) keep BN128/BK128 and gain the BN64/BK256 deep-K tile. Measured
    # (RX 7900 XTX, 4 bpw, L2-cold layer sweeps incl. both hadamards):
    #   down N=4096:  BN32/BK128 265, BN32/BK256 280 GB/s (was 242 at BN128)
    #   down N=5120:  BN32/BK128 252, BN32/BK256 251 GB/s (was 239-244)
    #   g/u   N=12288: BN128/BK128 406-410, BN64/BK256 425 GB/s
    #   g/u   N=17408: BN128/BK128 321-322, BN64/BK256 334 GB/s
    return [
        # M == 1 (decode GEMV), small-N pool (CTA-starved shapes)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256, "GROUP_M": 1}, num_warps=4, num_stages=3),
        # M == 1, large-N pool
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=8, num_stages=3),
        # M == 1 fallback for shapes outside both pools (never autotuned away:
        # kept so the pruned list is never empty on unusual shapes)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=4, num_stages=3),
        # M == 1, bits=3 run-funnel decode: the shared-funnel tile is narrow
        # (8 x BLOCK_N/2), so single-warp narrow blocks win the wide-N stream
        # (measured 484 vs 431 GB/s at 1x5120x248320). Pruned out for every
        # other invocation by _exl3_gemm_early_prune.
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 1}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 1}, num_warps=2, num_stages=3),
        # Generic path (other bit widths) and small shapes
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=3),
        # M > 1 (prefill)
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=8, num_stages=2),
    ]


_PRUNE = {"early_config_prune": _exl3_gemm_early_prune}


@triton.jit
def _decode_word_pair(
    low_u32, high_u32, shift,
    SHIFT_FITS_32: tl.constexpr,
    CB: tl.constexpr,
):
    """Funnel-shift a (low, high) u32 word pair into the 16-bit codebook index
    (generic-path helper) and decode it via _decode_u16."""
    if SHIFT_FITS_32:
        # 32-bit funnel: shift is guaranteed in [0,31] for K_BITS in {1,2,4}.
        neg_shift = tl.minimum(32 - shift, 31)
        windows = ((low_u32 >> shift) | (high_u32 << neg_shift)) & 0xFFFF
    else:
        low64 = (low_u32.to(tl.int64) & 0xFFFFFFFF) | ((high_u32.to(tl.int64) & 0xFFFFFFFF) << 32)
        windows = ((low64 >> shift) & 0xFFFF).to(tl.uint32)
    return _decode_u16(windows.to(tl.uint32), CB)


@triton.jit
def _funnel6(lo, hi, s):
    """bits=6 funnel: 16-bit code window from a (lo, hi) u32 word pair where
    hi is the word *preceding* lo in the tile's virtual bit stream, so the
    window can start past bit 31 of lo and the base word flips. lo, hi are
    [NN, 16] u32; s is an [S] shift vector; returns [S, NN, 16] u32 codes."""
    sel = s >= 32
    s32 = s & 31
    ns = tl.minimum(32 - s32, 31)
    base = tl.where(sel[:, None, None], hi[None, :, :], lo[None, :, :])
    second = tl.where(sel[:, None, None], lo[None, :, :], hi[None, :, :])
    return ((base >> s32[:, None, None]) | (second << ns[:, None, None])) & 0xFFFF


@triton.jit
def _decode_u16(w_u32, CB: tl.constexpr):
    """Inline arithmetic decode of 16-bit codebook indices (matches
    decode_3inst in the C++ reference): ~3 ALU ops instead of a 65536-entry
    LUT gather. Elementwise over u32 codes; returns fp16 weights."""
    if CB == 0:
        w_u32 = (w_u32 * 89226354 + 64248484) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    elif CB == 1:
        w_u32 = (w_u32 * 0xCBAC1FED) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    else:  # CB == 2 (mul1)
        w_u32 = (w_u32 * 0x83DCD12D) & 0xFFFFFFFF
        # byte sum: dp4a(x, 0x01010101, 0x6400) emulated
        db0 = w_u32 & 0xFF
        db1 = (w_u32 >> 8) & 0xFF
        db2 = (w_u32 >> 16) & 0xFF
        db3 = (w_u32 >> 24) & 0xFF
        w_u32 = (db0 + db1 + db2 + db3 + 0x6400) & 0xFFFF

    # bitcast low/high 16 bits to fp16 then add (cb 0/1), or fma (cb 2)
    if CB == 0 or CB == 1:
        lo = w_u32 & 0xFFFF
        hi = (w_u32 >> 16) & 0xFFFF
        lo_h = tl.cast(lo.to(tl.int16), tl.float16, bitcast=True)
        hi_h = tl.cast(hi.to(tl.int16), tl.float16, bitcast=True)
        return lo_h + hi_h
    else:
        sum16 = w_u32 & 0xFFFF
        h = tl.cast(sum16.to(tl.int16), tl.float16, bitcast=True)
        k_inv_h = tl.full((1,), 0x1eee, dtype=tl.int16)
        k_inv_h = tl.cast(k_inv_h, tl.float16, bitcast=True)
        k_bias_h = tl.full((1,), 0xc931, dtype=tl.int16)
        k_bias_h = tl.cast(k_bias_h, tl.float16, bitcast=True)
        return h * k_inv_h + k_bias_h


@triton.autotune(configs=_exl3_gemm_configs(), key=["M", "N", "K_dim", "K_BITS", "N_PACKED", "CB"], prune_configs_by=_PRUNE)
@triton.jit
def _fused_dequant_gemm_kernel(
    x_ptr, y_ptr,
    trellis_ptr,
    perm_i_ptr,
    mrow_ptr,
    M, N, K_dim,
    stride_xm, stride_xk,
    stride_tk, stride_tn,
    stride_ym, stride_yn,
    stride_ys,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_BITS: tl.constexpr,
    N_PACKED: tl.constexpr,
    CB: tl.constexpr,
    M1: tl.constexpr,
    SPLITS: tl.constexpr,
):
    NK: tl.constexpr = BLOCK_K // 16   # k-sub-tiles per weight tile
    NN: tl.constexpr = BLOCK_N // 16   # n-sub-tiles per weight tile
    N_U32: tl.constexpr = K_BITS * 256 // 32
    # For K_BITS in {1,2,4} the funnel shift never exceeds 31, so the 64-bit
    # funnel (high<<32 | low) >> shift can be computed with 32-bit ops only,
    # avoiding expensive emulated 64-bit arithmetic on RDNA3.
    SHIFT_FITS_32: tl.constexpr = (K_BITS == 1) | (K_BITS == 2) | (K_BITS == 4)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if SPLITS == 1:
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        pid_split = 0
    else:
        # M == 1 split-K GEMV: axis 0 tiles N, axis 1 slices the K loop. The
        # generic pid math above is unused (M == 1, GROUP_M irrelevant).
        pid_m = 0
        pid_n = pid % num_pid_n
        pid_split = pid // num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    tu32_ptr = trellis_ptr.to(tl.pointer_type(tl.uint32))
    stride_tk_u32 = stride_tk // 2
    stride_tn_u32 = stride_tn // 2
    base_n = (pid_n * NN) * stride_tn_u32

    n_k_tiles_total = K_dim // 16
    if SPLITS == 1:
        k_base = 0
        n_outer = tl.cdiv(n_k_tiles_total, NK)
    else:
        # Contiguous K slice for this split; splits beyond the remainder get
        # an empty range (their partial stays unwritten only if fully empty —
        # avoided by requiring K to have at least SPLITS * NK k-sub-tiles).
        tiles_per_split = tl.cdiv(n_k_tiles_total, SPLITS * NK) * NK
        k_base = pid_split * tiles_per_split
        n_outer = tl.cdiv(min(tiles_per_split, n_k_tiles_total - k_base), NK)

    if K_BITS == 4 and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # bits=4 fast path (full tiles only): coalesced staging + gather-free
        # algebraic decode.
        #
        # Staging: the packed words of all NN sub-tile columns of one k-tile
        # are contiguous, so two linear u32 loads (the row and the same row
        # shifted one word back) fetch everything dword-vectorized. The m1
        # row is wrapped within each sub-tile in registers (word -1 == word
        # 31), so no rotated/global scattered loads are ever issued.
        #
        # Decode: for sub-tile element (r, c) the codebook index comes from
        # trellis word pair (t-1, t) at shift s where
        #   t(r, c) = 4*(c%8) + (r%8)//2,   s(r, c) = 28 - 4*j(r, c),
        #   j(r, c) = 4*(c//8) + 2*(r//8) + (r%2),
        # a bijection (r, c) <-> (j, t) verified against _get_perm /
        # _dq_indices. Decoding the [8j, NN*32t] table of every (shift, word)
        # pair computes each weight exactly once; the permutation back to
        # (r, c) order is pure axis algebra.
        # ------------------------------------------------------------------
        j8 = tl.arange(0, 8)
        sh = 28 - 4 * j8                       # funnel shift per j row
        neg_sh = tl.minimum(32 - sh, 31)       # neighbor shift, masked to 0
        wc = tl.arange(0, NN * 32)             # staged word row
        nj8 = tl.arange(0, NN)

        if M1:
            # Decode path: pure GEMV reduction in fp32. The permuted weight
            # tile is never materialized: because (r, c) -> (j, t) is a
            # bijection, sum_r x[r] * W[r, c] == sum_{(j,t): c(j,t)=c}
            # Q[j,t] * X[j,t] with X[j, t] = x[r(j, t)] built from the 16 x
            # values by pure reshape/broadcast over the axis algebra
            #   j = 4*ch + 2*rh + p,  t = 4*cl + q,  r = 8*rh + 2*q + p,
            #   c = 16*nj + 8*ch + cl,
            # so the whole permutation lives in X's layout — free.
            #
            # The [ch, rh, p, nj, cl, q]-shaped product tile is accumulated
            # elementwise across the whole K loop (no cross-lane traffic per
            # iteration); the reduction over (rh, p, q) happens once at the
            # end. The m1 wrap word (t == 0 needs word 31) is a tiny [NN]
            # load instead of a per-subtile reduction.
            r16 = tl.arange(0, 16)
            acc6 = tl.zeros((2, 2, 2, NN, 8, 4), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_base + k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words = tl.load(row + wc)                          # [NN*32]
                    safe = (ktb > 0) | (base_n > 0)
                    m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                    w31 = tl.load(row + nj8 * 32 + 31)                # [NN]
                    w31_bcast = tl.reshape(
                        tl.broadcast_to(w31[:, None], (NN, 32)), (NN * 32,)
                    )
                    m1 = tl.where((wc % 32) == 0, w31_bcast, m1_lin)
                    q = ((words[None, :] >> sh[:, None]) |
                         (m1[None, :] << neg_sh[:, None])) & 0xFFFF    # [8, NN*32]
                    w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                    xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                    # X over (rh, p, q): r = 8*rh + 2*q + p
                    xpat = tl.permute(tl.reshape(xk, (2, 4, 2)), (0, 2, 1))
                    xb6 = tl.broadcast_to(
                        tl.reshape(xpat, (1, 2, 2, 1, 1, 4)), (2, 2, 2, NN, 8, 4)
                    )
                    acc6 += tl.reshape(w_dec, (2, 2, 2, NN, 8, 4)) * xb6
            s = tl.sum(acc6, 5)      # q    -> (ch, rh, p, nj, cl)
            s = tl.sum(s, 2)         # p    -> (ch, rh, nj, cl)
            s = tl.sum(s, 1)         # rh   -> (ch, nj, cl)
            acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))
            if SPLITS == 1:
                tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
            else:
                # Split-K partial: row pid_split of the [SPLITS, N] fp32 buffer.
                # stride_yn is the (unit) column stride of the partials buffer.
                tl.store(y_ptr + pid_split * stride_ys + offs_n * stride_yn, acc, mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words = tl.load(row + wc)
                    safe = (ktb > 0) | (base_n > 0)
                    m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                    w31 = tl.load(row + nj8 * 32 + 31)                # [NN]
                    w31_bcast = tl.reshape(
                        tl.broadcast_to(w31[:, None], (NN, 32)), (NN * 32,)
                    )
                    m1 = tl.where((wc % 32) == 0, w31_bcast, m1_lin)
                    q = ((words[None, :] >> sh[:, None]) |
                         (m1[None, :] << neg_sh[:, None])) & 0xFFFF
                    w = _decode_u16(q.to(tl.uint32), CB)
                    # reorder (ch, rh, p, nj, cl, q_) -> (r, c) statically
                    w = tl.reshape(w, (2, 2, 2, NN, 8, 4))
                    w = tl.permute(w, (1, 5, 2, 3, 0, 4))   # (rh, q_, p, nj, ch, cl)
                    w = tl.reshape(w, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif K_BITS == 6 and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # bits=6 fast path (full tiles only): gather-free algebraic decode,
        # twin of the bits=4 path. Verified against _dq_indices/_get_perm:
        #
        # e = 4*tg + jj,  tg = 4a + b      (jj = e%4, b = (e//4)%4, a = e//16)
        # code(e) = funnel(word(u), word(u-1), s) with
        #   u(e) = 3a + f(b),  f = [0,1,2,2]   (word index within the 48-word
        #                                      tile; u-1 wraps mod 48)
        #   s(e) = C_b - 6*jj,  C = [26, 34, 42, 18]
        # target position of e under the _get_perm permutation (verified
        # bijective bit-field assignment; e's bits are 32*cl + 16*(a&1) +
        # 8*(b>>1) + 4*(b&1) + 2*j1 + j0):
        #   r = 8*j1 + 4*(a&1) + 2*(b>>1) + j0
        #   c = 8*(b&1) + (a>>1)
        #
        # Only four linear word-slice loads are needed (all contiguous over
        # (nj, a), so everything stays coalesced, no tl.gather):
        #   b=0: (word 3a,   word 3a-1)  shift 26-6jj
        #   b=1: (word 3a+1, word 3a)    shift 34-6jj
        #   b=2: (word 3a+2, word 3a+1)  shift 42-6jj
        #   b=3: (word 3a+2, word 3a+1)  shift 18-6jj   (same words as b=2)
        # ------------------------------------------------------------------
        a16 = tl.arange(0, 16)
        nj8 = tl.arange(0, NN)
        j8 = tl.arange(0, 4)
        # word-slice addresses relative to the subtile base (mod 48 in-tile)
        wbase = tl.reshape(nj8[:, None] * 48 + 3 * a16[None, :], (NN * 16,))       # word 3a
        wone = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 1) % 48, (NN * 16,))  # 3a+1
        wtwo = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 2) % 48, (NN * 16,))  # 3a+2
        wneg = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 47) % 48, (NN * 16,)) # 3a-1
        # per-b constant shifts for the 4 jj rows
        C0 = tl.full((4,), 26, tl.int32); C1 = tl.full((4,), 34, tl.int32)
        C2 = tl.full((4,), 42, tl.int32); C3 = tl.full((4,), 18, tl.int32)
        sh6 = 6 * j8

        if M1:
            # GEMV: fold the permutation into the x broadcast. With
            # r = 8*j1 + 4*a0 + 2*b1 + j0, the (j1,j0,a0,b1)-indexed x
            # pattern comes from a reshape + permute + split of the 16
            # values; the b=0/1 decodes multiply x[..., b1=0], b=2/3 the
            # b1=1 half. Decodes reshape to (j1, j0, nj, cA, a0) since the
            # a axis factors as a = 8*cA + a0 (a0 = a&1, cA = a>>1 = c%8).
            r16 = tl.arange(0, 16)
            acc0 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
            acc1 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
            acc2 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
            acc3 = tl.zeros((2, 2, NN, 8, 2), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words2 = tl.reshape(tl.load(row + wbase), (NN, 16))
                    wone2 = tl.reshape(tl.load(row + wone), (NN, 16))
                    wtwo2 = tl.reshape(tl.load(row + wtwo), (NN, 16))
                    wneg2 = tl.reshape(tl.load(row + wneg), (NN, 16))
                    d0 = _decode_u16(_funnel6(words2, wneg2, C0 - sh6), CB).to(tl.float32)
                    d1 = _decode_u16(_funnel6(wone2, words2, C1 - sh6), CB).to(tl.float32)
                    d2 = _decode_u16(_funnel6(wtwo2, wone2, C2 - sh6), CB).to(tl.float32)
                    d3 = _decode_u16(_funnel6(wtwo2, wone2, C3 - sh6), CB).to(tl.float32)
                    xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                    # r = 8*j1 + 4*a0 + 2*b1 + j0  =>  (r3,r2,r1,r0)=(j1,a0,b1,j0)
                    xr = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (0, 3, 1, 2))
                    x_lo, x_hi = tl.split(xr)
                    x_lo = tl.broadcast_to(tl.reshape(x_lo, (2, 2, 1, 1, 2)), (2, 2, NN, 8, 2))
                    x_hi = tl.broadcast_to(tl.reshape(x_hi, (2, 2, 1, 1, 2)), (2, 2, NN, 8, 2))
                    acc0 += tl.reshape(d0, (2, 2, NN, 8, 2)) * x_lo
                    acc1 += tl.reshape(d1, (2, 2, NN, 8, 2)) * x_lo
                    acc2 += tl.reshape(d2, (2, 2, NN, 8, 2)) * x_hi
                    acc3 += tl.reshape(d3, (2, 2, NN, 8, 2)) * x_hi
            # reduce over (j1, j0, a0); leaves (nj, cA) per b; output
            # n = 16*nj + 8*b0 + cA with b0 = b&1 (b=0,2 -> 0; b=1,3 -> 1)
            s0 = tl.sum(tl.sum(tl.sum(acc0, 0), 0), 2)
            s1 = tl.sum(tl.sum(tl.sum(acc1, 0), 0), 2)
            s2v = tl.sum(tl.sum(tl.sum(acc2, 0), 0), 2)
            s3 = tl.sum(tl.sum(tl.sum(acc3, 0), 0), 2)
            h0 = s0 + s2v
            h1 = s1 + s3
            out = tl.permute(tl.join(h0, h1), (0, 2, 1))
            tl.store(y_ptr + offs_n * stride_yn, tl.reshape(out, (BLOCK_N,)).to(y_ptr.dtype.element_ty), mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words2 = tl.reshape(tl.load(row + wbase), (NN, 16))
                    wone2 = tl.reshape(tl.load(row + wone), (NN, 16))
                    wtwo2 = tl.reshape(tl.load(row + wtwo), (NN, 16))
                    wneg2 = tl.reshape(tl.load(row + wneg), (NN, 16))
                    d0 = _decode_u16(_funnel6(words2, wneg2, C0 - sh6), CB)
                    d1 = _decode_u16(_funnel6(wone2, words2, C1 - sh6), CB)
                    d2 = _decode_u16(_funnel6(wtwo2, wone2, C2 - sh6), CB)
                    d3 = _decode_u16(_funnel6(wtwo2, wone2, C3 - sh6), CB)
                    # reorder to (r, n): decode is [jj, nj, a]; reshape to
                    # (j1, j0, nj, cA, a0) — a = 8*cA + a0 with a0 = a&1 and
                    # cA = a>>1 = c%8 — then permute to
                    # (j1, a0, b1, j0, nj, b0, cA) and fold
                    # r = 8*j1 + 4*a0 + 2*b1 + j0, n = 16*nj + 8*b0 + cA.
                    P0 = tl.permute(tl.reshape(d0, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    P1 = tl.permute(tl.reshape(d1, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    P2 = tl.permute(tl.reshape(d2, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    P3 = tl.permute(tl.reshape(d3, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
                    J0 = tl.join(P0, P2)      # (j1, j0, a0, nj, cA, b1)
                    J1 = tl.join(P1, P3)
                    Wt = tl.join(J0, J1)      # (j1, j0, a0, nj, cA, b1, b0)
                    Wt = tl.permute(Wt, (0, 2, 5, 1, 3, 6, 4))
                    w = tl.reshape(Wt, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif (K_BITS == 1 or K_BITS == 2 or K_BITS == 8) and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # Power-of-two widths (K = 1, 2, 8): same gather-free structure as
        # the bits=4 path, generalized. The (r, c) -> (word, shift) lookup is
        # a pure bit-field map: with r = 8*r3 + 4*r2 + 2*r1 + r0 and
        # c = 8*c3 + cl (cl = c%8), the 5 bits (r2, r1, c3, r3, r0) split —
        # the first log2(K_BITS) of them pack into the sub-tile word index
        #   word = K_BITS*cl + g,      g = (r2, r1, c3)[:log2(K)] packed MSB-first
        # and the remaining bits form the shift row
        #   row  = the (5 - log2(K)) remaining bits, MSB-first
        #   shift(row) = 32 - K_BITS - K_BITS*row
        # (verified element-exactly against the C++ reconstruct kernel; the
        # K_BITS == 4 case is the branch above). Every (row, word) pair is one
        # sub-tile element exactly once, so like bits=4 the packed row loads
        # linear and the permutation is realized by static reshapes; the m1
        # neighbor word (needed when shift > 16) is the row shifted one word
        # back, wrapped inside the sub-tile in registers.
        # ------------------------------------------------------------------
        ROWS: tl.constexpr = 32 // K_BITS
        rows = tl.arange(0, ROWS)
        sh = (32 - K_BITS) - K_BITS * rows
        neg_sh = tl.minimum(32 - sh, 31)
        wc = tl.arange(0, NN * N_U32)

        if M1:
            # GEMV: fold the permutation into the x broadcast (see bits=4).
            r16 = tl.arange(0, 16)
            if K_BITS == 1:
                acc7 = tl.zeros((2, 2, 2, 2, 2, NN, 8), dtype=tl.float32)  # (r2,r1,c3,r3,r0,nj,cl)
            elif K_BITS == 2:
                acc7 = tl.zeros((2, 2, 2, 2, NN, 8, 2), dtype=tl.float32)  # (r1,c3,r3,r0,nj,cl,r2)
            else:
                acc7 = tl.zeros((2, 2, NN, 8, 2, 2, 2), dtype=tl.float32)  # (r3,r0,nj,cl,r2,r1,c3)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words = tl.load(row + wc)
                    safe = (ktb > 0) | (base_n > 0)
                    m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                    wlast = tl.load(row + (wc // N_U32) * N_U32 + (N_U32 - 1))
                    m1 = tl.where((wc % N_U32) == 0, wlast, m1_lin)
                    q = ((words[None, :] >> sh[:, None]) |
                         (m1[None, :] << neg_sh[:, None])) & 0xFFFF     # [ROWS, NN*N]
                    w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                    xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                    if K_BITS == 1:
                        # r = 8*r3 + 4*r2 + 2*r1 + r0
                        xpat = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (1, 2, 0, 3))
                        xb = tl.broadcast_to(
                            tl.reshape(xpat, (2, 2, 1, 2, 2, 1, 1)), (2, 2, 2, 2, 2, NN, 8)
                        )
                        acc7 += tl.reshape(w_dec, (2, 2, 2, 2, 2, NN, 8)) * xb
                    elif K_BITS == 2:
                        xpat = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (2, 0, 3, 1))
                        xb = tl.broadcast_to(
                            tl.reshape(xpat, (2, 1, 2, 2, 1, 1, 2)), (2, 2, 2, 2, NN, 8, 2)
                        )
                        acc7 += tl.reshape(w_dec, (2, 2, 2, 2, NN, 8, 2)) * xb
                    else:
                        xpat = tl.permute(tl.reshape(xk, (2, 2, 2, 2)), (0, 3, 1, 2))
                        xb = tl.broadcast_to(
                            tl.reshape(xpat, (2, 2, 1, 1, 2, 2, 1)), (2, 2, NN, 8, 2, 2, 2)
                        )
                        acc7 += tl.reshape(w_dec, (2, 2, NN, 8, 2, 2, 2)) * xb
            if K_BITS == 1:
                s = tl.sum(tl.sum(tl.sum(tl.sum(acc7, 0), 0), 1), 1)     # -> (c3, nj, cl)
            elif K_BITS == 2:
                s = tl.sum(tl.sum(tl.sum(tl.sum(acc7, 0), 1), 1), 3)     # -> (c3, nj, cl)
            else:
                s = tl.sum(tl.sum(tl.sum(tl.sum(acc7, 0), 0), 2), 2)     # -> (nj, cl, c3)
                s = tl.permute(s, (0, 2, 1))
            acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))       # n = 16*nj + 8*c3 + cl
            tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    words = tl.load(row + wc)
                    safe = (ktb > 0) | (base_n > 0)
                    m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
                    wlast = tl.load(row + (wc // N_U32) * N_U32 + (N_U32 - 1))
                    m1 = tl.where((wc % N_U32) == 0, wlast, m1_lin)
                    q = ((words[None, :] >> sh[:, None]) |
                         (m1[None, :] << neg_sh[:, None])) & 0xFFFF
                    w = _decode_u16(q.to(tl.uint32), CB)
                    # static reorder (row bits, word bits) -> (r, c)
                    if K_BITS == 1:
                        w = tl.reshape(w, (2, 2, 2, 2, 2, NN, 8))        # (r2,r1,c3,r3,r0,nj,cl)
                        w = tl.permute(w, (3, 0, 1, 4, 5, 2, 6))         # (r3,r2,r1,r0,nj,c3,cl)
                    elif K_BITS == 2:
                        w = tl.reshape(w, (2, 2, 2, 2, NN, 8, 2))        # (r1,c3,r3,r0,nj,cl,r2)
                        w = tl.permute(w, (2, 6, 0, 3, 4, 1, 5))         # (r3,r2,r1,r0,nj,c3,cl)
                    else:
                        w = tl.reshape(w, (2, 2, NN, 8, 2, 2, 2))        # (r3,r0,nj,cl,r2,r1,c3)
                        w = tl.permute(w, (0, 4, 5, 1, 2, 6, 3))         # (r3,r2,r1,r0,nj,c3,cl)
                    w = tl.reshape(w, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif K_BITS == 3 and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # bits=3 fast path (full tiles only). The D-table rows regroup into 8
        # run-groups g = 2v + c3 (rows r = 2v + 8q + p, m = 2q + p) whose four
        # 16-bit windows are consecutive 3-bit steps of ONE 32-bit funnel
        # Q_g = (W[a_g] >> b_g) | (W[a_g - 1] << (32 - b_g)) of subtile words
        # (word indices mod 24; b_g = (84 - 12*g) % 32, a_g = g // 4),
        # window(g, m) = (Q_g >> (9 - 3*m)) & 0xFFFF. Verified bit-exact
        # against the D-table decode. Two strided u32 slice loads (the g/4
        # word and its -1 neighbor, stride 3 over the 24-word subtile) feed
        # all eight funnels, so each weight costs one load-lane instead of
        # two and one funnel instead of four.
        # ------------------------------------------------------------------
        r16 = tl.arange(0, 16)
        g8 = tl.arange(0, 8)
        col = tl.arange(0, NN * 8)
        njc = col // 8
        clc = col % 8
        base_n3 = (pid_n * NN) * stride_tn_u32

        base_g = (84 - 12 * g8) % 32
        neg_g = tl.minimum(32 - base_g, 31)
        a_g = 2 - (84 - 12 * g8) // 32
        w_a = njc[None, :] * N_U32 + 3 * clc[None, :] + a_g[:, None]
        w_b = njc[None, :] * N_U32 + (3 * clc[None, :] + a_g[:, None] + N_U32 - 1) % N_U32

        if M1:
            # acc[g, nj*8 + clc]; each of the 4 rows of a group accumulates
            # into the shared (g, column) slot with its own x element.
            acc8 = tl.zeros((8, NN * 8), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n3
                    A = tl.load(row + w_a)
                    B = tl.load(row + w_b)
                    # no 32-bit mask on Q: the extraction masks drop every bit
                    # above 24, including the bit-31 pollution B<<31 at base 0
                    Q = (A >> base_g[:, None]) | (B << neg_g[:, None])
                    xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                    # X_m[g] = xk[2*(g//2) + 8*(m//2) + m%2]: pairs (e, o) of
                    # the rows xk[2i+p], then halves v<4 / v>=4, interleaved
                    # over c3 by the final (4, 2) -> 8 broadcast.
                    e, o = tl.split(tl.reshape(xk, (8, 2)))
                    e_lo, e_hi = tl.split(tl.permute(tl.reshape(e, (2, 4)), (1, 0)))
                    o_lo, o_hi = tl.split(tl.permute(tl.reshape(o, (2, 4)), (1, 0)))
                    x_m0 = tl.broadcast_to(tl.reshape(tl.broadcast_to(e_lo[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                    x_m1 = tl.broadcast_to(tl.reshape(tl.broadcast_to(o_lo[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                    x_m2 = tl.broadcast_to(tl.reshape(tl.broadcast_to(e_hi[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                    x_m3 = tl.broadcast_to(tl.reshape(tl.broadcast_to(o_hi[:, None], (4, 2)), (8,))[:, None], (8, NN * 8))
                    acc8 += (
                        _decode_u16(((Q >> 9) & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m0
                        + _decode_u16(((Q >> 6) & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m1
                        + _decode_u16(((Q >> 3) & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m2
                        + _decode_u16((Q & 0xFFFF).to(tl.uint32), CB).to(tl.float32) * x_m3
                    )
            # (v, c3, nj, clc) -> sum v -> n = 16*nj + 8*c3 + clc
            s = tl.sum(tl.reshape(acc8, (4, 2, NN, 8)), 0)   # (c3, nj, clc)
            acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))
            tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n3
                    A = tl.load(row + w_a)
                    B = tl.load(row + w_b)
                    Q = (A >> base_g[:, None]) | (B << neg_g[:, None])
                    # (v, c3, nj, clc, p, q) -> (r = 8q + 2v + p, n = 16*nj +
                    # 8*c3 + clc) with m = 2q + p
                    q0 = tl.reshape((Q >> 9) & 0xFFFF, (4, 2, NN, 8))
                    q1 = tl.reshape((Q >> 6) & 0xFFFF, (4, 2, NN, 8))
                    q2 = tl.reshape((Q >> 3) & 0xFFFF, (4, 2, NN, 8))
                    q3 = tl.reshape(Q & 0xFFFF, (4, 2, NN, 8))
                    wq = tl.join(tl.join(q0, q1), tl.join(q2, q3))
                    wq = tl.permute(wq, (5, 0, 4, 2, 1, 3)).reshape(16, BLOCK_N)
                    w = _decode_u16(wq.to(tl.uint32), CB)
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    elif (K_BITS == 5 or K_BITS == 7) and (N % BLOCK_N == 0) and (K_dim % BLOCK_K == 0):
        # ------------------------------------------------------------------
        # Odd widths (K = 3, 5, 7): the (word, shift) lookup does not factor
        # into independent per-axis bit fields (the 16-bit decode window ends
        # inside a code, so word index and shift carry into each other).
        # Instead, each of the 32 (r, c3) rows of a sub-tile has ONE fixed
        # window offset D = 32*f + sh (see _M_ROW_OFFSETS): element (r, c)
        # reads word K_BITS*(c%8) + f(r, c//8) at funnel shift sh(r, c//8),
        # neighbor word -1 when sh > 16. The decode tile is [32 rows,
        # NN*8 (c%8)] with affine word addresses (stride K_BITS in the
        # column axis, constant row offset), so there is still no
        # data-dependent gather; every word of the sub-tile is used exactly
        # once per f-slice.
        # ------------------------------------------------------------------
        r16 = tl.arange(0, 16)
        mrow = tl.arange(0, 32)                     # row = 2*r + c3
        D_vec = tl.load(mrow_ptr + mrow)
        f_vec = D_vec // 32
        sh_vec = D_vec % 32
        neg_vec = tl.minimum(32 - sh_vec, 31)
        col = tl.arange(0, NN * 8)
        njc = col // 8
        clc = col % 8
        w_lo = njc[None, :] * N_U32 + K_BITS * clc[None, :] + f_vec[:, None]
        t_hi = K_BITS * clc[None, :] + f_vec[:, None] - 1
        w_hi = njc[None, :] * N_U32 + (t_hi + N_U32) % N_U32

        if M1:
            accm = tl.zeros((32, NN * 8), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    lo = tl.load(row + w_lo)                       # [32, NN*8]
                    hi = tl.load(row + w_hi)
                    q = ((lo >> sh_vec[:, None]) |
                         (hi << neg_vec[:, None])) & 0xFFFF
                    w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
                    xk = tl.load(x_ptr + (ktb * 16 + r16) * stride_xk).to(tl.float32)
                    xb = tl.reshape(
                        tl.broadcast_to(tl.reshape(xk, (16, 1, 1, 1)), (16, 2, NN, 8)),
                        (32, NN * 8),
                    )
                    accm += w_dec * xb
            # (r, c3, nj, cl) -> sum over r -> (c3, nj, cl) -> n = 16*nj + 8*c3 + cl
            s = tl.sum(tl.reshape(accm, (16, 2, NN, 8)), 0)
            acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))
            tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    row = tu32_ptr + ktb * stride_tk_u32 + base_n
                    lo = tl.load(row + w_lo)
                    hi = tl.load(row + w_hi)
                    q = ((lo >> sh_vec[:, None]) |
                         (hi << neg_vec[:, None])) & 0xFFFF
                    w = _decode_u16(q.to(tl.uint32), CB)
                    # (r, c3, nj, cl) -> (r, nj, c3, cl) -> [16, BLOCK_N]
                    w = tl.permute(tl.reshape(w, (16, 2, NN, 8)), (0, 2, 1, 3))
                    w = tl.reshape(w, (16, BLOCK_N))
                    k_off = ktb * 16 + tl.arange(0, 16)
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )
    else:
        # ------------------------------------------------------------------
        # Generic path (other bit widths / non-full tiles): staged row load +
        # tl.gather decode. The packed words for all NN sub-tile columns of
        # one k-sub-tile are contiguous, so one u32 load fetches them
        # coalesced; per-element decode words are then gathered from the
        # staged row via shared memory instead of scattered global scalars.
        # ------------------------------------------------------------------
        r16 = tl.arange(0, 16)
        n_idx = tl.arange(0, BLOCK_N)
        elem_flat = r16[:, None] * 16 + (n_idx % 16)[None, :]  # [16, BLOCK_N]
        elem_idx = tl.load(perm_i_ptr + elem_flat).to(tl.int32)

        if K_BITS == 4:
            lane = elem_idx // 8
            r = elem_idx % 8
            word_low_idx = lane
            word_high_idx = (lane + 31) % 32
            shift = (7 - r) * 4
        elif K_BITS == 2:
            q16 = elem_idx // 16
            i1 = q16
            i0 = (i1 + 15) % 16
            r = elem_idx % 8
            shift0 = ((~(elem_idx // 8 * 8)) & 8) * 2
            word_low_idx = i1
            word_high_idx = i0
            shift = shift0 + (7 - r) * 2
        elif K_BITS == 1:
            q32 = elem_idx // 32
            i1 = q32
            i0 = (i1 + 7) % 8
            r = elem_idx % 8
            shift0 = (~(elem_idx // 8 * 8)) & 24
            word_low_idx = i1
            word_high_idx = i0
            shift = shift0 + (7 - r)
        elif K_BITS == 3:
            t_offset = elem_idx // 8 * 8
            r = elem_idx % 8
            b1 = (t_offset + 257) * K_BITS
            b0 = b1 - 16
            b2 = b1 + K_BITS * 7
            i0 = b0 // 32
            i2 = (b2 - 1) // 32
            s2 = (i2 + 1) * 32 - b2
            word_low_idx = i2 % N_U32
            word_high_idx = i0 % N_U32
            shift = s2 + (7 - r) * K_BITS
        elif K_BITS == 7:
            # dq2x2 widths: the C++ decode pairs consecutive codes across the
            # word boundary, so the per-element window does not follow the
            # t_offset/j algebra of the dq4 widths. Use the verified per-row
            # window offsets (same tables as the odd-width fast path).
            row = r16[:, None] * 2 + ((n_idx[None, :] % 16) // 8)
            d = tl.load(mrow_ptr + row)
            word_low_idx = K_BITS * (n_idx[None, :] % 8) + d // 32
            word_high_idx = (word_low_idx - 1 + N_U32) % N_U32
            shift = d % 32
        else:
            t = (elem_idx // 4) * 4
            j = elem_idx % 4
            b0 = (t + 257) * K_BITS - 16
            b2 = (t + 260) * K_BITS
            i0 = b0 // 32
            i2 = (b2 - 1) // 32
            s2 = (i2 + 1) * 32 - b2
            word_low_idx = i2 % N_U32
            word_high_idx = i0 % N_U32
            shift = s2 + (3 - j) * K_BITS

        # Gather indices into the staged row: sub-tile nj occupies words
        # [nj*N_U32, (nj+1)*N_U32).
        tile_off = (n_idx // 16) * N_U32
        idx_low = word_low_idx + tile_off[None, :]
        idx_high = word_high_idx + tile_off[None, :]

        WCOLS: tl.constexpr = NN * N_U32
        WCOLS_P2: tl.constexpr = triton.next_power_of_2(WCOLS)
        wcols = tl.arange(0, WCOLS_P2)
        tiles_n = N // 16
        n_words_valid = min(WCOLS, max(tiles_n - pid_n * NN, 0) * N_U32)
        wmask = wcols < n_words_valid

        if M1:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    k_ok = ktb < n_k_tiles_total
                    words = tl.load(
                        tu32_ptr + ktb * stride_tk_u32 + base_n + wcols,
                        mask=wmask & k_ok, other=0,
                    )
                    src = tl.broadcast_to(words[None, :], (16, WCOLS_P2))
                    low_u32 = tl.gather(src, idx_low, 1)
                    high_u32 = tl.gather(src, idx_high, 1)
                    w = _decode_word_pair(low_u32, high_u32, shift, SHIFT_FITS_32, CB)
                    xk = tl.load(
                        x_ptr + (ktb * 16 + r16) * stride_xk,
                        mask=k_ok & (r16 < 16), other=0.0,
                    )
                    acc += tl.sum(w.to(tl.float32) * xk.to(tl.float32)[:, None], 0)
            tl.store(y_ptr + offs_n * stride_yn, acc.to(y_ptr.dtype.element_ty), mask=mask_n)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_outer in range(n_outer):
                for ki in tl.static_range(NK):
                    ktb = k_outer * NK + ki
                    k_ok = ktb < n_k_tiles_total
                    words = tl.load(
                        tu32_ptr + ktb * stride_tk_u32 + base_n + wcols,
                        mask=wmask & k_ok, other=0,
                    )
                    src = tl.broadcast_to(words[None, :], (16, WCOLS_P2))
                    low_u32 = tl.gather(src, idx_low, 1)
                    high_u32 = tl.gather(src, idx_high, 1)
                    w = _decode_word_pair(low_u32, high_u32, shift, SHIFT_FITS_32, CB)
                    k_off = ktb * 16 + r16
                    x_block = tl.load(
                        x_ptr + offs_m[:, None] * stride_xm + k_off[None, :] * stride_xk,
                        mask=mask_m[:, None] & (k_off < K_dim)[None, :],
                        other=0.0,
                    )
                    acc = tl.dot(x_block, w, acc)
            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                acc.to(y_ptr.dtype.element_ty),
                mask=mask_m[:, None] & mask_n[None, :],
            )


def exl3_gemm_triton(
    x: torch.Tensor,
    trellis: torch.Tensor,
    y: torch.Tensor,
    lut: torch.Tensor,
    perm_i: torch.Tensor,
    K_bits: int,
    tiles_n: int,
    cb: int = 0,
    splits: int = 1,
) -> None:
    if not has_triton:
        raise RuntimeError("exl3_gemm_triton requires Triton")
    """Fused EXL3 dequant + fp16 matmul. Does NOT materialize the weight matrix.

    ``splits > 1`` (M == 1 bits=4 full-tile shapes only) runs the split-K GEMV:
    ``y`` must then be the [splits, N] fp32 partials buffer from
    _get_splitk_buf, to be summed by _m1_split_reduce_had afterwards.
    """
    M, K_dim = x.shape
    N = y.shape[1]

    # The split store exists only in the bits=4 M==1 fast branch; never let a
    # split request reach any other path.
    splits = splits if (M == 1 and K_bits == 4) else 1

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]) * splits,
    )

    _fused_dequant_gemm_kernel[grid](
        x, y,
        trellis,
        perm_i,
        _get_m_row_offsets(K_bits, x.device) if K_bits in _M_ROW_OFFSETS else perm_i,
        M, N, K_dim,
        x.stride(0), x.stride(1),
        trellis.stride(0), trellis.stride(1),
        y.stride(0), y.stride(1),
        y.stride(0),
        K_BITS=K_bits,
        N_PACKED=trellis.shape[-1],
        CB=cb,
        M1=(M == 1),
        SPLITS=splits,
    )


# ---------------------------------------------------------------------------
# LinearEXL3 composition: had_r_128_triton -> gemm -> had_r_128_triton
# ---------------------------------------------------------------------------

def _linear_exl3_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    xh: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    bias: torch.Tensor | None,
    in_features: int,
    out_features: int,
) -> None:
    """Complete EXL3 linear forward into pre-allocated buffers.

    Writes the Hadamard-transformed input to ``xh`` and the result to ``y``.
    All tensors must be pre-allocated with stable addresses for CUDA graph
    capture; nothing is allocated inside this call.
    """
    # A cast here would allocate and silently break a capturing graph; callers
    # that need dtype conversion must cast before calling.
    assert x.dtype == torch.half, f"_linear_exl3_triton: expected half input, got {x.dtype}"

    # Phase 1: input Hadamard transform -> xh
    had_r_128_triton(x, xh, suh, None, 1.0)

    # Phase 2 + 3: fused dequant + Triton GEMM -> y
    cb = 1 if mcg else (2 if mul1 else 0)
    splits = _m1_splitk_plan(x.shape[0], out_features, in_features, K)
    if splits > 1:
        # Split-K GEMV into fp32 partials, then a fused reduce + output
        # Hadamard (replaces the separate had_r_128_triton launch).
        partials = _get_splitk_buf(out_features, splits, x.device)
        exl3_gemm_triton(
            xh, trellis, partials,
            _decode_lut(cb, x.device), _get_perm(x.device),
            K, trellis.shape[1], cb, splits,
        )
        _m1_split_reduce_had(partials, y, svh, splits)
    else:
        exl3_gemm_triton(
            xh, trellis, y,
            _decode_lut(cb, x.device), _get_perm(x.device),
            K, trellis.shape[1], cb,
        )
        # Phase 4: output Hadamard transform (in place)
        had_r_128_triton(y, y, None, svh, 1.0)

    if bias is not None:
        y.add_(bias)


def linear_exl3_triton(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    in_features: int,
    out_features: int,
    device: torch.device,
    out_dtype: torch.dtype = torch.half,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused EXL3 dequant + GEMM, allocating and returning the output.

    Convenience wrapper over ``_linear_exl3_triton`` for uncaptured callers;
    graph-capturing paths (BC) call the preallocated-buffer function directly
    to keep tensor addresses stable across replays.

    All bit widths K = 1..8 are decoded in-kernel (fast gather-free paths for
    full tiles, a staged-row gather path for the rest).

    The per-call ``xh`` workspace allocation is deliberate and cheap: the
    caching allocator serves it from its free list (no cudaMalloc, no sync)
    after the first few calls, and this wrapper is off the decode hot path —
    prefill is dominated by the GEMM itself.
    """
    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    x_half = x_flat if x_flat.dtype == torch.half else x_flat.to(torch.half)

    y = torch.empty((rows, out_features), dtype=out_dtype, device=device)
    xh = torch.empty_like(x_half)

    _linear_exl3_triton(
        x_half, y, xh, trellis, suh, svh, K, mcg, mul1, bias,
        in_features, out_features,
    )

    return y.view(original_shape[:-1] + (out_features,))
