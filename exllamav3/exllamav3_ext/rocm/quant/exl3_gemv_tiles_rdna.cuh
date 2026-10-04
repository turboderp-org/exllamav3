#pragma once

// =============================================================================
// EXL3 GEMV "tiles" core for RDNA: fewer VALU cycles per 16x16 weight tile
// =============================================================================
//
// Drop-in replacement for the barrier-free dot core (exl3_gemv_dot_tile_direct
// and its multi-row form exl3_gemv_dot_tile_direct_mr): same lane layout, same
// per-lane fdot2 chains in the same order, same quad reduction -- so every
// output is BIT-IDENTICAL to the core it replaces. Only the instruction
// sequence that produces the decoded weights and the addresses changes.
//
// Why: the decode GEMVs had the VALU as the binding unit, not DRAM. Per 16x16
// tile per lane the old K=2 loop spent:
//   8 x v_mul_lo_u32     (hash multiply)         QUARTER rate
//   8 x v_dot4_u32_u8    (byte sum)              HALF rate
//   v_lshrrev_b64        (fshift)
//   64-bit per-lane address math + a divergent (exec-mask) loop
//   bfe / and / lshl_or packing / pk_fma / dot2
// (add, bfe, perm, alignbit, sad_u8, sad_hi_u8, mad_u32_u16, pk_mul_lo_u16 and
// dot2_f32_f16 issue at full rate.)
//
// The rewrite, each step exact integer arithmetic:
//
//   hash multiply  P = x * C mod 2^32 for a 16-bit x (x < 2^16):
//                  P = x * C_lo16 + ((x * C_hi16) mod 2^16) << 16
//                  r = v_pk_mul_lo_u16 src, [0 | C_hi16]   (op_sel picks x's
//                      half of src for the high lane; the low lane is x*0 = 0)
//                  P = v_mad_u32_u16 src, C_lo16, r        (op_sel picks x)
//                  two full-rate ops for one quarter-rate op. x may sit in
//                  either 16-bit half of a register, so a window that starts
//                  at bit 0 or 16 of an aligned word needs no extraction op
//                  and the rest need one shift (no mask: the u16 operand
//                  ignores the upper bits). The cb 0 additive constant rides in
//                  the same two ops via v_pk_mad_u16.
//   byte sum       v_sad_u8 (P, 0, acc) is sum|P_i - 0| + acc, the dp4a-by-ones
//                  sum at full rate; v_sad_hi_u8 adds the second sum << 16, so
//                  a decoded PAIR comes out already packed as [0x6400 + s0 |
//                  0x6400 + s1] -- no and / lshl_or packing (mul1, cb 2).
//   fshift         one v_alignbit_b32 (the shift is < 32 for the aligned
//                  layouts) instead of a 64-bit shift.
//   addressing     the tile base is wave-uniform (readfirstlane'd warp index,
//                  block indices, pointer from the parameter tables), so the
//                  loop runs on SGPR base pointers with a constant 32-bit
//                  per-lane offset: global_load vN, vOff, s[base:base+1]
//                  offset:imm, a scalar loop counter and s_cbranch_scc, zero
//                  VALU address math per tile.
//
// K = 1, 2, 4 take the hand-laid aligned decoders; K = 3, 5..8 run the generic
// dq8 / dq4 / dq2x2 bit arithmetic of quant/exl3_dq.cuh (unchanged) with the
// fast pair decoder and the uniform addressing.
//
// Selected at runtime (exl3_gemv_core_mode(), EXL3_ROCM_GEMV_TILES=0 selects
// the direct core; EXL3_GEMV_LDS=1 still pins the LDS core).
//
// Portability: v_pk_mul_lo_u16 / v_pk_mad_u16 (VOP3P), v_mad_u32_u16 with
// op_sel, v_sad_u8 / v_sad_hi_u8 exist on gfx10.3+, gfx11 and gfx12 with the
// same syntax.
// =============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../../ptx.cuh"
#include "../../quant/exl3_dq.cuh"

#define EXL3_GEMV_CORE_DIRECT 0     // barrier-free direct core
#define EXL3_GEMV_CORE_LDS    1     // LDS unswizzle core (EXL3_GEMV_LDS=1)
#define EXL3_GEMV_CORE_TILES  2     // this file (default)

typedef __attribute__((address_space(1))) const uint32_t exl3_g_u32;
typedef __attribute__((address_space(3))) const uint32_t exl3_l_u32;

// -----------------------------------------------------------------------------
// Wave-uniform helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ int exl3_uni(int x)
{
    return __builtin_amdgcn_readfirstlane(x);
}

template <typename T>
__device__ __forceinline__ T* exl3_uni_ptr(T* p)
{
    uint64_t v = (uint64_t) p;
    uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t) v);
    uint32_t hi = __builtin_amdgcn_readfirstlane((uint32_t) (v >> 32));
    return (T*) (((uint64_t) hi << 32) | (uint64_t) lo);
}

// -----------------------------------------------------------------------------
// Hash multiply: x * C (+ D) mod 2^32 for the 16-bit x in the low (HI = false)
// or high (HI = true) half of src
// -----------------------------------------------------------------------------

template <uint32_t C, uint32_t D, bool HI>
__device__ __forceinline__ uint32_t exl3_hmul16(uint32_t src)
{
    constexpr uint32_t k_hi = C & 0xffff0000u;           // [0 | C_hi16]
    constexpr uint32_t c_lo = C & 0xffffu;
    uint32_t r, p;
    if constexpr (D == 0)
    {
        // r = [src.x * 0 | src.x * C_hi16]
        if constexpr (HI)
            asm ("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(r) : "v"(src), "s"(k_hi));
        else
            asm ("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(r) : "v"(src), "s"(k_hi));
    }
    else
    {
        // r = [D_lo16 | src.x * C_hi16 + D_hi16]
        constexpr uint32_t d = D;
        if constexpr (HI)
            asm ("v_pk_mad_u16 %0, %1, %2, %3" : "=v"(r) : "v"(src), "s"(k_hi), "s"(d));
        else
            asm ("v_pk_mad_u16 %0, %1, %2, %3 op_sel_hi:[0,1,1]" : "=v"(r) : "v"(src), "s"(k_hi), "s"(d));
    }
    // p = src.x * C_lo16 + r  (32-bit, carries included)
    if constexpr (HI)
        asm ("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
    else
        asm ("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(p) : "v"(src), "s"(c_lo), "v"(r));
    return p;
}

template <int cb, bool HI>
__device__ __forceinline__ uint32_t exl3_hash16(uint32_t src)
{
    if constexpr (cb == 0) return exl3_hmul16<89226354u, 64248484u, HI>(src);
    if constexpr (cb == 1) return exl3_hmul16<0xCBAC1FEDu, 0u, HI>(src);
    if constexpr (cb == 2) return exl3_hmul16<0x83DCD12Du, 0u, HI>(src);
    return 0;
}

// Two hashed words -> the decoded pair (lo = word 0, hi = word 1), bit for bit
// decode_3inst_2<cb> of the same two 16-bit indices.
template <int cb>
__device__ __forceinline__ half2 exl3_decode_pair(uint32_t p0, uint32_t p1)
{
    if constexpr (cb == 2)
    {
        uint32_t s = __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u);
        s = __builtin_amdgcn_sad_hi_u8(p1, 0u, s);
        half2_uint32 u(s);
        const half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
        const half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
        return __hfma2(u.as_half2, k_inv_h2, k_bias_h2);
    }
    else
    {
        p0 = exl3_lop3_6a(p0, 0x8fff8fffu, 0x3b603b60u);
        p1 = exl3_lop3_6a(p1, 0x8fff8fffu, 0x3b603b60u);
        half2_uint32 xu0(p0);
        half2_uint32 xu1(p1);
        half2 d0 = __lows2half2(xu0.as_half2, xu1.as_half2);
        half2 d1 = __highs2half2(xu0.as_half2, xu1.as_half2);
        return __hadd2(d0, d1);
    }
}

// Pair decode from two sources, each 16-bit index in the low half (or high,
// per the flags) of its register
template <int cb, bool HI0, bool HI1>
__device__ __forceinline__ half2 exl3_dq_pair(uint32_t src0, uint32_t src1)
{
    return exl3_decode_pair<cb>(exl3_hash16<cb, HI0>(src0), exl3_hash16<cb, HI1>(src1));
}

// Generic form for the dq8/dq4/dq2x2 paths: indices in the low halves
template <int cb>
__device__ __forceinline__ half2 exl3_decode_3inst_2_fast(uint32_t w0, uint32_t w1)
{
    return exl3_dq_pair<cb, false, false>(w0, w1);
}


// -----------------------------------------------------------------------------
// Half-integer bitrates (K + 0.5 bpw, K = 1..3, mul1 codebook only)
// -----------------------------------------------------------------------------
// They ride the integer `bits` template slot as a pseudo width EXL3_HALF_BITS(K)
// = 16 + K (17 / 18 / 19 = 1.5 / 2.5 / 3.5 bpw), so every GEMV body that is
// templated on bits takes them without a new template parameter -- the integer
// instantiations keep their names and their code (every half branch below is
// an `if constexpr` on Exl3Width<bits>::half, false for bits 1..8). Tile layout
// (quant/exl3_dq.cuh dq8_half): positions alternate K and K + 1 bits (odd
// positions carry the extra bit), 4 * (2K + 1) dwords per 16x16 tile.
// Selected only for half-rate weights, behind EXL3_ROCM_HALF_GEMV (default on;
// =0 sends half rates back to the cooperative GEMM / mgemm).

#ifndef EXL3_HALF_BITS
#define EXL3_HALF_BITS(ka) (16 + (ka))   // also in exl3_gemv_multirow_rdna.cuh
#endif

template <int bits>
struct Exl3Width
{
    static constexpr bool half = bits > 16;
    static constexpr int ka = half ? bits - 16 : bits;                      // integer part
    static constexpr int tile_bytes = half ? 16 * (2 * ka + 1) : 32 * bits;  // bytes per 16x16 tile
    static constexpr int tile_u16 = tile_bytes / 2;
};

// -----------------------------------------------------------------------------
// Per-lane load plan: which dwords of a tile lane L reads, fixed for the loop
// -----------------------------------------------------------------------------
// A tile is 8 * bits dwords. Lane L (t_offset = 8 L) reads R dwords of it and
// decodes its 8 weights from those alone. The plan holds byte offsets from the
// tile base (constant over the loop, so the loads are SGPR base + one VGPR
// offset) and the funnel shifts. Load and decode are split so the loop can put
// several k-tiles' loads in flight before the first decode.
//
//   bits 1, 2, 4  aligned decoders: 2 dwords, one shift (< 32)
//   bits 3        dq8<3, cb, 4>:    2 dwords
//   bits 5, 6, 8  dq4 x 2:          4 dwords
//   bits 7        dq2x2 x 2:        8 dwords
//   half K + 0.5  dq8_half:         4 dwords (two window groups of two), shifts < 32

template <int bits> struct Exl3RawCount { static constexpr int R = Exl3Width<bits>::half ? 4 : ((bits <= 4) ? 2 : (bits == 7 ? 8 : 4)); };

template <int bits>
struct Exl3LanePlan
{
    static constexpr int R = Exl3RawCount<bits>::R;
    uint32_t off[R];     // byte offsets within the tile; off[2i] = "a" (hi), off[2i+1] = "b" (lo)
    int sh[R / 2];       // funnel shift (aligned forms) or s2 (generic forms)
};

// Index helpers shared by plan and decode for the generic forms (the
// arithmetic of quant/exl3_dq.cuh's dq8 / dq4 / dq2x2)
template <int bits>
__device__ __forceinline__ void exl3_dq4_idx(int t_offset, int& i0, int& i2, int& s2)
{
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 3 * bits;
    int b2 = b1 + 16;
    i0 = (b0 / 32) % (bits * 256 / 32);
    s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    i2 = ((b2 - 1) / 32) % (bits * 256 / 32);
}

template <int bits>
__device__ __forceinline__ void exl3_dq2_idx(int t_offset, int& i0, int& i2, int& s2)
{
    int b0 = (t_offset + 257) * bits - 16;
    int b1 = b0 + 1 * bits;
    int b2 = b1 + 16;
    i0 = (b0 / 32) % (bits * 256 / 32);
    s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    i2 = ((b2 - 1) / 32) % (bits * 256 / 32);
}

template <int bits>
__device__ __forceinline__ void exl3_dq8_idx(int t_offset, int& i0, int& i2, int& s2)
{
    int b1 = (t_offset + 257) * bits;
    int b0 = b1 - 16;
    int b2 = b1 + bits * 7;
    i0 = (b0 / 32) % (bits * 256 / 32);
    s2 = ((b2 - 1) / 32 + 1) * 32 - b2;
    i2 = ((b2 - 1) / 32) % (bits * 256 / 32);
}

template <int bits>
__device__ __forceinline__ Exl3LanePlan<bits> exl3_lane_plan(int lane)
{
    Exl3LanePlan<bits> p;
    const int t_offset = lane << 3;
    if constexpr (Exl3Width<bits>::half)
    {
        // dq8_half's window arithmetic (quant/exl3_dq.cuh), per lane: group 7 (windows
        // 4..7) and group 3 (windows 0..3), each two dwords and one funnel shift
        constexpr int KA = Exl3Width<bits>::ka;
        constexpr int bits2 = 2 * KA + 1;
        constexpr int words = 4 * bits2;
        constexpr int gspan = 18 + 3 * KA;
        const int e7 = ((t_offset >> 1) + 4) * bits2 + 128 * bits2;
        const int e3 = e7 - 2 * bits2;
        const int hi7 = (e7 - 1) / 32, lo7 = (e7 - gspan) / 32;
        const int hi3 = (e3 - 1) / 32, lo3 = (e3 - gspan) / 32;
        p.off[0] = (lo7 % words) * 4; p.off[1] = (hi7 % words) * 4; p.sh[0] = (hi7 + 1) * 32 - e7;
        p.off[2] = (lo3 % words) * 4; p.off[3] = (hi3 % words) * 4; p.sh[1] = (hi3 + 1) * 32 - e3;
    }
    else if constexpr (bits == 1)
    {
        int i1 = t_offset >> 5;
        p.off[1] = i1 * 4;
        p.off[0] = ((i1 + 7) & 7) * 4;
        p.sh[0] = (~t_offset) & 24;
    }
    else if constexpr (bits == 2)
    {
        int i1 = t_offset >> 4;
        p.off[1] = i1 * 4;
        p.off[0] = ((i1 + 15) & 15) * 4;
        p.sh[0] = ((~t_offset) & 8) << 1;
    }
    else if constexpr (bits == 4)
    {
        int i1 = t_offset >> 3;
        p.off[1] = i1 * 4;
        p.off[0] = ((i1 + 31) & 31) * 4;
        p.sh[0] = 20;
    }
    else if constexpr (bits == 3)
    {
        int i0, i2, s2;
        exl3_dq8_idx<bits>(t_offset, i0, i2, s2);
        p.off[0] = i0 * 4; p.off[1] = i2 * 4; p.sh[0] = s2;
    }
    else if constexpr (bits == 7)
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                int i0, i2, s2;
                exl3_dq2_idx<bits>(t_offset + 4 * h + 2 * i, i0, i2, s2);
                p.off[(h * 2 + i) * 2] = i0 * 4; p.off[(h * 2 + i) * 2 + 1] = i2 * 4; p.sh[h * 2 + i] = s2;
            }
    }
    else
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int i0, i2, s2;
            exl3_dq4_idx<bits>(t_offset + 4 * h, i0, i2, s2);
            p.off[h * 2] = i0 * 4; p.off[h * 2 + 1] = i2 * 4; p.sh[h] = s2;
        }
    }
    return p;
}

// Aligned decoders, from the two loaded dwords. Word k is decoded exactly as
// dq8_aligned_{1bit,2bits,4bits}: w7 at bit 0 of the funnel result, w(7-i) at
// bit i*bits; frag0 = (w0,w1),(w2,w3), frag1 = (w4,w5),(w6,w7).

template <int cb>
__device__ __forceinline__ void exl3_dq8_2bits_fast(uint32_t a, uint32_t b, int shift, FragB& frag0, FragB& frag1)
{
    b = __funnelshift_r(b, a, shift);            // shift is 0 or 16: one v_alignbit
    frag0[0] = exl3_dq_pair<cb, false, false>(b >> 14, b >> 12);
    frag0[1] = exl3_dq_pair<cb, false, false>(b >> 10, b >> 8);
    frag1[0] = exl3_dq_pair<cb, false, false>(b >> 6,  b >> 4);
    frag1[1] = exl3_dq_pair<cb, false, false>(b >> 2,  b);
}

template <int cb>
__device__ __forceinline__ void exl3_dq8_1bit_fast(uint32_t a, uint32_t b, int shift, FragB& frag0, FragB& frag1)
{
    b = __funnelshift_r(b, a, shift);            // shift in {0, 8, 16, 24}
    frag0[0] = exl3_dq_pair<cb, false, false>(b >> 7, b >> 6);
    frag0[1] = exl3_dq_pair<cb, false, false>(b >> 5, b >> 4);
    frag1[0] = exl3_dq_pair<cb, false, false>(b >> 3, b >> 2);
    frag1[1] = exl3_dq_pair<cb, false, false>(b >> 1, b);
}

template <int cb>
__device__ __forceinline__ void exl3_dq8_4bits_fast(uint32_t a, uint32_t b, FragB& frag0, FragB& frag1)
{
    uint32_t s = __funnelshift_r(b, a, 20);
    // w0..w2 from s (bits 8, 4, 0), w3..w7 from b (bits 16, 12, 8, 4, 0);
    // the bit-16 window is b's high half (no extraction op)
    frag0[0] = exl3_dq_pair<cb, false, false>(s >> 8, s >> 4);
    frag0[1] = exl3_dq_pair<cb, false, true >(s,      b);
    frag1[0] = exl3_dq_pair<cb, false, false>(b >> 12, b >> 8);
    frag1[1] = exl3_dq_pair<cb, false, false>(b >> 4,  b);
}

// Generic widths: dq8<3, cb, 4>, dq4, dq2 bit arithmetic verbatim (fshift is
// the 64-bit funnel there: s2 + bits * k can exceed 31), fast pair decoder.

template <int bits, int cb>
__device__ __forceinline__ void exl3_dq8_3bits_fast(uint32_t a, uint32_t b, int s2, FragB& frag0, FragB& frag1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> bits;
    w5 = w6 >> bits;
    w4 = w5 >> bits;
    w3 = fshift(b, a, s2 + bits * 4);
    w2 = w3 >> bits;
    w1 = w2 >> bits;
    w0 = w1 >> bits;
    frag0[0] = exl3_decode_3inst_2_fast<cb>(w0, w1);
    frag0[1] = exl3_decode_3inst_2_fast<cb>(w2, w3);
    frag1[0] = exl3_decode_3inst_2_fast<cb>(w4, w5);
    frag1[1] = exl3_decode_3inst_2_fast<cb>(w6, w7);
}

template <int bits, int cb>
__device__ __forceinline__ void exl3_dq4_fast(uint32_t a, uint32_t b, int s2, FragB& frag)
{
    uint32_t w3 = fshift(b, a, s2);
    uint32_t w2 = fshift(b, a, s2 + bits);
    uint32_t w1 = fshift(b, a, s2 + bits * 2);
    uint32_t w0 = fshift(b, a, s2 + bits * 3);
    frag[0] = exl3_decode_3inst_2_fast<cb>(w0, w1);
    frag[1] = exl3_decode_3inst_2_fast<cb>(w2, w3);
}

template <int bits, int cb>
__device__ __forceinline__ half2 exl3_dq2_fast(uint32_t a, uint32_t b, int s2)
{
    uint32_t w1 = fshift(b, a, s2);
    uint32_t w0 = fshift(b, a, s2 + bits);
    return exl3_decode_3inst_2_fast<cb>(w0, w1);
}

// Half-integer K + 0.5: dq8_half verbatim (the funnel shift is < 32, so one
// v_alignbit per group), fast pair decoder on the low 16 bits of each window.
// raw = { a7, b7, a3, b3 } (a = the lower-index word, the high half of the funnel)
template <int KA, int cb>
__device__ __forceinline__ void exl3_dq8_half_fast(const uint32_t* raw, int s7, int s3, FragB& frag0, FragB& frag1)
{
    uint32_t w7 = __funnelshift_r(raw[1], raw[0], s7);
    uint32_t w6 = w7 >> (KA + 1);
    uint32_t w5 = w6 >> KA;
    uint32_t w4 = w5 >> (KA + 1);
    uint32_t w3 = __funnelshift_r(raw[3], raw[2], s3);
    uint32_t w2 = w3 >> (KA + 1);
    uint32_t w1 = w2 >> KA;
    uint32_t w0 = w1 >> (KA + 1);
    frag0[0] = exl3_decode_3inst_2_fast<cb>(w0, w1);
    frag0[1] = exl3_decode_3inst_2_fast<cb>(w2, w3);
    frag1[0] = exl3_decode_3inst_2_fast<cb>(w4, w5);
    frag1[1] = exl3_decode_3inst_2_fast<cb>(w6, w7);
}

template <int bits, int cb>
__device__ __forceinline__ void exl3_dq_tile_decode
(
    const uint32_t* raw,                 // R dwords, plan order
    const Exl3LanePlan<bits>& lp,
    FragB& frag0,
    FragB& frag1
)
{
    if constexpr (Exl3Width<bits>::half) exl3_dq8_half_fast<Exl3Width<bits>::ka, cb>(raw, lp.sh[0], lp.sh[1], frag0, frag1);
    else if constexpr (bits == 1) exl3_dq8_1bit_fast<cb>(raw[0], raw[1], lp.sh[0], frag0, frag1);
    else if constexpr (bits == 2) exl3_dq8_2bits_fast<cb>(raw[0], raw[1], lp.sh[0], frag0, frag1);
    else if constexpr (bits == 4) exl3_dq8_4bits_fast<cb>(raw[0], raw[1], frag0, frag1);
    else if constexpr (bits == 3) exl3_dq8_3bits_fast<bits, cb>(raw[0], raw[1], lp.sh[0], frag0, frag1);
    else if constexpr (bits == 7)
    {
        frag0[0] = exl3_dq2_fast<bits, cb>(raw[0], raw[1], lp.sh[0]);
        frag0[1] = exl3_dq2_fast<bits, cb>(raw[2], raw[3], lp.sh[1]);
        frag1[0] = exl3_dq2_fast<bits, cb>(raw[4], raw[5], lp.sh[2]);
        frag1[1] = exl3_dq2_fast<bits, cb>(raw[6], raw[7], lp.sh[3]);
    }
    else
    {
        exl3_dq4_fast<bits, cb>(raw[0], raw[1], lp.sh[0], frag0);
        exl3_dq4_fast<bits, cb>(raw[2], raw[3], lp.sh[1], frag1);
    }
}

// Loads through a wave-uniform base + per-lane 32-bit byte offset
typedef __attribute__((address_space(1))) const char exl3_g_char;
typedef __attribute__((address_space(3))) const char exl3_l_char;

__device__ __forceinline__ uint32_t exl3_ld_g(exl3_g_char* base, uint32_t off)
{
    return *(exl3_g_u32*) (base + off);
}

__device__ __forceinline__ uint32_t exl3_ld_l(const char* base, uint32_t off)
{
    return *(exl3_l_u32*) ((exl3_l_char*) base + off);
}

// -----------------------------------------------------------------------------
// The core: M rows, one N-tile, k16-tiles [kb_begin, kb_end)
// -----------------------------------------------------------------------------
// A: this slab's rotated rows (lda halves apart), in global memory (A_LDS =
// false) or LDS (true). B: the matrix's trellis. All pointer and range
// arguments must be wave-uniform (the callers pass block indices, kernel
// arguments, and exl3_uni()'d warp indices). out[r]: row r's column-lane value
// (lanes 0-15), exactly exl3_gemv_dot_tile_direct_mr's contract.
//
// U k-tiles are loaded before the first is decoded (their loads in flight
// together), then decoded and accumulated in k order, so the fdot2 chains are
// the same instruction sequence as U = 1. The remainder runs at U = 1.

// Keeps a loop-invariant lane offset opaque inside the loop. Without it LICM
// hoists the zero-extension of the 32-bit offset out of the loop and
// instruction selection, seeing a bare 64-bit VGPR add, emits two VALU ops of
// address arithmetic per load instead of the global_load saddr form (SGPR
// base + 32-bit VGPR offset + immediate). The asm is empty; the register is
// carried through it unchanged, so it costs nothing.
__device__ __forceinline__ void exl3_opaque(uint32_t& v)
{
    asm volatile ("" : "+v"(v));
}

template <int bits, int cb, int M, bool A_LDS, int U, int T>
__device__ __forceinline__ void exl3_tiles_step
(
    const char* A_c,         // byte pointer, this k-group's first slice (uniform)
    const int lda_b,         // row stride, bytes
    exl3_g_char* b_c,        // byte pointer, this k-group's first tile (uniform)
    const uint32_t b_stride, // bytes between consecutive k-tiles of this N-tile
    uint32_t* off,           // lane plan byte offsets (R), loop-invariant
    uint32_t& a_off,         // lane's A byte offset within a slice
    const Exl3LanePlan<bits>& lp,
    float* accA,             // [T][M]
    float* accB
)
{
    constexpr int R = Exl3LanePlan<bits>::R;
    constexpr int tile_bytes = Exl3Width<bits>::tile_bytes;
    #pragma unroll
    for (int i = 0; i < R; ++i) exl3_opaque(off[i]);
    exl3_opaque(a_off);

    uint32_t raw[U][T][R];
    uint32_t w01[U][M], w89[U][M];
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int t = 0; t < T; ++t)
            #pragma unroll
            for (int i = 0; i < R; ++i)
                raw[u][t][i] = exl3_ld_g(b_c + (size_t) u * b_stride + t * tile_bytes, off[i]);
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const char* a = A_c + r * lda_b + u * 32;
            if constexpr (A_LDS) { w01[u][r] = exl3_ld_l(a, a_off); w89[u][r] = exl3_ld_l(a + 16, a_off); }
            else                 { w01[u][r] = exl3_ld_g((exl3_g_char*) a, a_off); w89[u][r] = exl3_ld_g((exl3_g_char*) a + 16, a_off); }
        }
    #pragma unroll
    for (int u = 0; u < U; ++u)
        #pragma unroll
        for (int t = 0; t < T; ++t)
        {
            FragB frag0, frag1;
            exl3_dq_tile_decode<bits, cb>(raw[u][t], lp, frag0, frag1);
            #pragma unroll
            for (int r = 0; r < M; ++r)
            {
                half2_uint32 a01(w01[u][r]);
                half2_uint32 a89(w89[u][r]);
                float& aA = accA[t * M + r];
                float& aB = accB[t * M + r];
                aA = __builtin_amdgcn_fdot2(a01.as_half2, frag0[0], aA, false);
                aA = __builtin_amdgcn_fdot2(a89.as_half2, frag0[1], aA, false);
                aB = __builtin_amdgcn_fdot2(a01.as_half2, frag1[0], aB, false);
                aB = __builtin_amdgcn_fdot2(a89.as_half2, frag1[1], aB, false);
            }
        }
}

// T adjacent N-tiles [tile_n, tile_n + T) per wave, sharing the A loads and
// reading T * 32 * bits contiguous bytes per k-tile. out[t * M + r].
template <int bits, int cb, int M, bool A_LDS, int U = 1, int T = 1>
__device__ __forceinline__ void exl3_gemv_dot_tile_tiles
(
    const half* __restrict__ A,
    const int lda,
    const uint16_t* __restrict__ B,
    const int n_tiles,
    const int tile_n,
    const int lane,
    const int kb_begin,
    const int kb_end,
    float* out
)
{
    // Every argument but lane is wave-uniform by contract; say so, so the
    // tile base and the loop live in SGPRs (the callers' warp index is a
    // VGPR value to the compiler).
    A = exl3_uni_ptr(A);
    B = exl3_uni_ptr(B);
    const int n_tiles_u = exl3_uni(n_tiles);
    const int tile_n_u = exl3_uni(tile_n);
    const int kb_begin_u = exl3_uni(kb_begin);
    const int kb_end_u = exl3_uni(kb_end);
    const int lda_u = exl3_uni(lda);

    constexpr int tile_bytes = Exl3Width<bits>::tile_bytes;
    const Exl3LanePlan<bits> lp = exl3_lane_plan<bits>(lane);
    constexpr int R = Exl3LanePlan<bits>::R;
    uint32_t off[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) off[i] = lp.off[i];
    uint32_t a_off = (uint32_t) (lane & 3) * 4;             // (A[r0], A[r0+1]) within a k16 slice

    float accA[T * M], accB[T * M];
    #pragma unroll
    for (int r = 0; r < T * M; ++r) { accA[r] = 0.0f; accB[r] = 0.0f; }

    const uint32_t b_stride = (uint32_t) n_tiles_u * tile_bytes;
    exl3_g_char* b_c = (exl3_g_char*) (const char*) B + ((size_t) kb_begin_u * n_tiles_u + tile_n_u) * tile_bytes;
    const char* a_c = (const char*) (A + kb_begin_u * 16);
    const int lda_b = lda_u * 2;

    int k = kb_begin_u;
    if constexpr (U > 1)
    {
        for (; k + U <= kb_end_u; k += U)
        {
            exl3_tiles_step<bits, cb, M, A_LDS, U, T>(a_c, lda_b, b_c, b_stride, off, a_off, lp, accA, accB);
            b_c += (size_t) U * b_stride;
            a_c += U * 32;
        }
    }
    for (; k < kb_end_u; k++)
    {
        exl3_tiles_step<bits, cb, M, A_LDS, 1, T>(a_c, lda_b, b_c, b_stride, off, a_off, lp, accA, accB);
        b_c += b_stride;
        a_c += 32;
    }

    #pragma unroll
    for (int r = 0; r < T * M; ++r)
    {
        accA[r] += __shfl_xor(accA[r], 1, 32);
        accA[r] += __shfl_xor(accA[r], 2, 32);
        accB[r] += __shfl_xor(accB[r], 1, 32);
        accB[r] += __shfl_xor(accB[r], 2, 32);
        float vA = __shfl(accA[r], (lane & 7) * 4, 32);
        float vB = __shfl(accB[r], (lane & 7) * 4, 32);
        out[r] = (lane & 8) ? vB : vA;
    }
}

// -----------------------------------------------------------------------------
// Per-architecture tile shape
// -----------------------------------------------------------------------------
// U: k-tiles whose loads are issued together (device, compile time). T: adjacent
// N-tiles per wave (host picks the launch grid, exl3_gemv_tiles_tpb in
// exl3_gemv_rdna.cu; the kernels are built for EXL3_GEMV_TILES_TMAX):
//
//                       K=1-3   K=4   K=5-6   K=7-8
//   split-K, T = 2        U2     U1     U1      U1
//   split-K, T = 1        U4     U2     U2      U1     (EXL3_GEMV_TILES_T=1)
//   single-warp (head)    U4     U2     U4      U1
//   rows M = 8: U1 (register budget)
//
// All RDNA3/4 parts take the same values -- they share the wave32 VMEM/VALU
// structure; only the DRAM latency/bandwidth ratio that sets the best U
// differs. The VGPR count stays <= 64 for every entry, so occupancy is not
// what the table trades.

#define EXL3_GEMV_TILES_TMAX 2

// Half-integer rates (4 raw dwords per tile at every K): own table, with
// EXL3_HALF_U_* compile-time overrides (-D).
#ifndef EXL3_HALF_U_T2
#define EXL3_HALF_U_T2 2
#endif
#ifndef EXL3_HALF_U_T1
#define EXL3_HALF_U_T1 4
#endif
template <int KA, int M, int T>
__device__ __forceinline__ constexpr int exl3_tiles_u_half_splitk()
{
    if constexpr (M >= 8) return 1;
    if constexpr (M >= 4) return 2 < EXL3_HALF_U_T2 ? 2 : EXL3_HALF_U_T2;
    if constexpr (T >= 2) return EXL3_HALF_U_T2;
    return EXL3_HALF_U_T1;
}

template <int bits, int M, int T>
__device__ __forceinline__ constexpr int exl3_tiles_u_splitk()
{
#if defined(__gfx1151__) || defined(__gfx1150__) || 1   /* all RDNA */
    if constexpr (Exl3Width<bits>::half) return exl3_tiles_u_half_splitk<Exl3Width<bits>::ka, M, T>();
    if constexpr (M >= 8) return 1;
    if constexpr (T >= 2) return (bits <= 3) ? 2 : 1;
    return (bits <= 3) ? 4 : (bits <= 6 ? 2 : 1);
#endif
}

template <int bits>
__device__ __forceinline__ constexpr int exl3_tiles_u_single()
{
    if constexpr (Exl3Width<bits>::half) return 4;
    return (bits <= 3) ? 4 : (bits == 4 ? 2 : (bits <= 6 ? 4 : 1));
}
