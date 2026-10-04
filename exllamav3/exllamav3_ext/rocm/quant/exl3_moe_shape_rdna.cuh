#pragma once

// =============================================================================
// MoE tile-K override for RDNA
// =============================================================================
//
// quant/exl3_moe_common.cuh sets MOE_TILESIZE_K to 32 with a bare
// #define (not #ifndef), so -D cannot override it -- the header would simply
// redefine it back. This file is included AFTER that header by both the MoE
// kernel sibling and the MoE host sibling, so the two agree on one value.
// They must: the host derives blockDim from it
// (EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16), so a mismatch between host and
// kernel is a silently wrong launch, not a compile error.
//
// The value is the common header's 32. This header exists anyway, for two
// reasons: the override mechanism is the only way to change it at all (that
// #define is bare, so -D alone cannot win), and the hazard below is worth recording at the
// point where the constant is set.
//
// MOE_TILESIZE_K = 32 makes TILEBLOCKS_K = 2 in exl3_gemm_kernel_inner, which
// switches on the split-K machinery -- the sub_k operand split, the per-warp B
// dequant staging, and threadblock_reduce(). *Every* shape in the RDNA shape
// table uses TILESIZE_K = 16, so that path compiles out of every validated GEMM
// configuration and the MoE kernel is its only live caller. It held two real
// defects, both invisible at TILEBLOCKS_K == 1:
//
//   - threadblock_reduce() read a different sh_c address than it wrote, past the
//     end of the LDS block. Made exl3_moe emit all-NaN.
//   - sh_b_dq was sized NUM_WARPS (warps per sub_k group) and indexed by a
//     warp_id that wraps at NUM_WARPS, so the sub_k 0 and sub_k 1 warps sharing
//     a warp_id staged different B fragments into the same buffer with only a
//     __syncwarp between them. Left a residual error after the first fix.
//
// Both are fixed: fused MoE matches an fp32 reference as closely as the
// per-expert path does, and the split-K path is faster than forcing 16, so
// forcing 16 costs throughput and buys nothing now.
//
// EXL3_RDNA_MOE_TILESIZE_K=16 forces the single-K path -- the geometry every RDNA
// GEMM shape is validated on. Keep it as the first bisect step if MoE output ever
// looks wrong again: if 16 fixes it, the fault is in the split-K machinery.
// =============================================================================

#ifndef EXL3_RDNA_MOE_TILESIZE_K
#define EXL3_RDNA_MOE_TILESIZE_K 32
#endif

#ifdef MOE_TILESIZE_K
#undef MOE_TILESIZE_K
#endif
#define MOE_TILESIZE_K EXL3_RDNA_MOE_TILESIZE_K

static_assert(EXL3_GEMM_BASE_THREADS * EXL3_RDNA_MOE_TILESIZE_K / 16 <= 1024,
    "MoE block width exceeds the maximum workgroup size");

// -----------------------------------------------------------------------------
// Pipelined MoE mainloop (exl3_moe_inner_rdna.cuh) tuning
// -----------------------------------------------------------------------------
// EXL3_MOE_PIPE_DB: B register-ring depth, k-tiles in flight per wave (2, 4 or 8).
#ifndef EXL3_MOE_PIPE_DB
#define EXL3_MOE_PIPE_DB 4
#endif
static_assert(EXL3_MOE_PIPE_DB == 2 || EXL3_MOE_PIPE_DB == 4 || EXL3_MOE_PIPE_DB == 8,
    "EXL3_MOE_PIPE_DB must be 2, 4 or 8");

// EXL3_MOE_PIPE_WPE: waves per SIMD the pipelined kernel is register-budgeted for.
// 8 (= 1536 / 8 = 192 VGPRs) lets two 512-thread blocks share a WGP; the host still
// asks the runtime (hipOccupancyMaxActiveBlocksPerMultiprocessor) before using it.
#ifndef EXL3_MOE_PIPE_WPE
#define EXL3_MOE_PIPE_WPE 8
#endif
