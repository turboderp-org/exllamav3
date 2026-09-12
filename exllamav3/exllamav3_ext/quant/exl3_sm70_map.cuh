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

}  // namespace exl3_sm70