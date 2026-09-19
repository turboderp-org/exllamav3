#pragma once

// sm_70 (Volta) support primitives for the EXL3 kernels.
//
// mma.m8n8k4 fragment layouts (PTX ISA 9.7.13.4.1, .f16 A/B, .f32 C/D,
// .row.col). One m8n8k4 instruction runs 4 MMA computations; computation c
// (c = lane>>2 for the low group) uses lanes {4c..4c+3} (low) and
// {16+4c..16+4c+3} (high). Per MMA computation, the 8 participating lanes
// hold:
//
//   A (row-major, 2 .f16x2 regs = 4 elems/lane):
//     row = lane%4 (+4 if lane >= 16), col = i, i = 0..3
//   B (col-major, 2 .f16x2 regs = 4 elems/lane):
//     col = lane%4 (+4 if lane >= 16), row = i, i = 0..3
//   D/C (.f32, 8 regs/lane):
//     row = X (+4 if lane >= 16), X = (lane&1) + (i&2)
//     col = (i&4) + (lane&2) + (i&1),  i = 0..7
//
// Verified against the PTX ISA tables and 1Cat SM70_MMA_884
// thread_offset_C by tests/test_sm70_lane_map.py (tilings perfect, D = A@B
// through the maps, per-MMA agreement with 1Cat's composed-tile map).
//
// The GEMV decode lane-remap (Step 1 of the port plan) is expressed as:
// given the sm80 m16n8k16 GEMV fragment assignment (lane L decodes weight
// elements (row = L >> 2, k-pair = L & 3) of a 16x16 tile), produce the
// m8n8k4 operand values either by re-indexing the decode (preferred) or by
// lane exchange.

#include <cstdint>

namespace exl3_sm70 {

// ---- Host/device lane-map tables ----------------------------------------
// For a 16x16 weight tile processed as 2 (m) x 2 (k4) m8n8k4 MMAs:
//   MMA(mh, kh): rows [8*mh .. 8*mh+7], k-cols [4*kh .. 4*kh+3]
// Lane L of the owning warp must hold:
//   A operand: elements W[8*mh + (L&3) + 4*r][4*kh + (L>>2)*2 + (r&1)]  r=0..1
//   B operand: elements W[8*mh + (L&3) + 4*r][4*kh + (L>>2)*2 + (r&1)]  r=0..1
// (A and B pick the same element value in this GEMV because the "activation"
// is the x-fragment broadcast; see mma_ab_sm70 below.)

// Element (row, k) of the 16x16 tile that lane L must decode for
// MMA(mh, kh), operand half r (0 or 1). constexpr-friendly.
constexpr int elem_row(int L, int mh, int r) { return 8 * mh + (L & 3) + 4 * r; }
constexpr int elem_k(int L, int kh, int r) { return 4 * kh + ((L >> 2) * 2 + (r & 1)); }


// ---- Shuffle-exchange pattern (decode → m8n8k4 ownership) ---------------
// The trellis decode yields, per 16x16 tile, 8 values per lane in sm80
// m16n8k16 B-fragment order: lane Ls holds codes {8*Ls .. 8*Ls+7}; code c
// ↔ cell (k = 2*(Ls&3) + (i&1) + 8*(i>>1), col = (Ls>>2) + 8*(j>>2)) with
// j = c&7, i = j&3 (tiling verified: 256 codes ↔ 256 cells, bijective).
//
// The m8n8k4 consumer (lane Lp) needs cells
//   W[(Lp&3) + 4*mh][4*((Lp>>2)&3) + i]   mh = 0..1, i = 0..3
//   x[4*((Lp>>2)&3) + i]                  (activation, same k)
// The owning decode lane and slot are, in closed form (validated over all
// 256 cells):
//   Ls   = 16*mh + (i>>1) + 4*(Lp&3) + 2*((Lp>>2)&1)
//   slot = (i&1) + 2*(((Lp>>2)>>1)&1)
// → 8 __shfl per lane per tile re-home the decoded values; the same
// values serve both the A (activation) and B (weight) operands.

// Source decode lane for consumer lane Lp, MMA half mh, k-index i.
constexpr int shfl_src(int Lp, int mh, int i)
{
    return 16 * mh + (i >> 1) + 4 * (Lp & 3) + 2 * ((Lp >> 2) & 1);
}
// Source slot (0..7) within the source lane's 8 decoded values.
constexpr int shfl_slot(int Lp, int i)
{
    return (i & 1) + 2 * (((Lp >> 2) >> 1) & 1);
}
// C fragment: lane L, reg 0..7 → (row, col) within the 8x8 C tile.
// (Per MMA computation; the warp's 4 computations each tile the 8x8.)
constexpr int c_row(int L, int reg)
{
    return (L & 1) + ((reg >> 1) & 1) * 2 + (L >= 16 ? 4 : 0);
}
constexpr int c_col(int L, int reg)
{
    return ((reg >> 2) & 1) * 4 + (L & 2) + (reg & 1);
}

// ---- mma.m8n8k4 primitive (sm_70 native) --------------------------------
// D (8 f32) += A (8x4 .f16) @ B (4x8 .f16), .row.col. Native on sm_70.
// A/B fragments: 2 .f16x2 regs per lane (4 halves); D: 8 f32 regs.
__device__ __forceinline__ void mma_m8n8k4_rc_f32(
    const uint32_t a0, const uint32_t a1,   // A fragment (2 .f16x2)
    const uint32_t b0, const uint32_t b1,   // B fragment (2 .f16x2)
    float* d)                               // 8 f32 accumulators
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 800)
    asm volatile
    (
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1)
    );
#endif
}

}  // namespace exl3_sm70