#pragma once

#include <cuda_fp16.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../ptx.cuh"
#include "../../quant/exl3_dq.cuh"
#include "exl3_rdna3.cuh"
#include "exl3_rdna3_had.cuh"

#ifdef __HIP_PLATFORM_AMD__

namespace exl3_rdna3_ns {

constexpr int NT = EXL3_RDNA3_NT;

// idx * 0x83DCD12D (mod 2^32) for a 16-bit trellis window. On RDNA3 the 32-bit v_mul_lo_u32 is quarter
// rate; two full-rate 24-bit multiplies give the same product (only the low 8 bits of idx * 0x83 reach
// the top byte). Inline asm: LLVM otherwise folds the pair back into one v_mul_lo_u32
__device__ __forceinline__ uint32_t mul1_mul(uint32_t x)
{
#if defined(__HIP_DEVICE_COMPILE__)
    uint32_t lo, hi;
    asm ("v_mul_u32_u24 %0, 0xdcd12d, %1" : "=v"(lo) : "v"(x));
    asm ("v_mul_u32_u24 %0, 0x83, %1" : "=v"(hi) : "v"(x));
    return lo + (hi << 24);
#else
    return x * 0x83DCD12Du;
#endif
}
constexpr int THREADS = EXL3_RDNA3_THREADS;
constexpr int XS_BYTES = 16384;   // LDS budget for the staged x chunk

typedef _Float16 hv2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ uint32_t h2u(half2 h)
{
    return *reinterpret_cast<uint32_t*>(&h);
}

// v_dot2_f32_f16: c + a.x * b.x + a.y * b.y, fp32 accumulate
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false);
}

__device__ __forceinline__ void wave_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

using exl3_rdna3_had::had128;

// 64-bit funnel: (a:b) >> s, low 32 bits, 0 <= s < 64
__device__ __forceinline__ uint32_t fsh(uint32_t b, uint32_t a, int s)
{
    return (uint32_t) ((((uint64_t) a << 32) | (uint64_t) b) >> s);
}

// (a:b) >> (s + T) for a runtime s < 32 and a constant T, without the quarter-rate 64-bit shift:
// y = low word of (a:b) >> s (v_alignbit), hi = a >> s, then constant shifts
struct Funnel
{
    uint32_t y, hi;
    __device__ __forceinline__ Funnel(uint32_t b, uint32_t a, int s) :
        y(__builtin_amdgcn_alignbit(a, b, s)), hi(a >> s) {}
    template <int T> __device__ __forceinline__ uint32_t at() const
    {
        if constexpr (T == 0) return y;
        else if constexpr (T >= 32) return hi >> (T - 32);
        else return (y >> T) | (hi << (32 - T));
    }
};

// mul1 codebook, raw form: fp16 bit pattern of 1024 + bytesum(idx * 0x83DCD12D) for two 16-bit windows,
// packed as half2. The affine map to the codebook value (k_inv * h + k_bias) is applied to the dot products
// instead (see the kernel epilogue), saving the per-weight fma
__device__ __forceinline__ uint32_t mul1_raw_pair(uint32_t x0, uint32_t x1)
{
    // v_sad_u8 against zero is a full-rate byte sum (v_dot4_u32_u8 is half rate on RDNA3); v_sad_hi_u8
    // adds the second sum into the high half, so the pair packs without a separate shift/or
    const uint32_t s = __builtin_amdgcn_sad_u8(mul1_mul(x0), 0u, 0x64006400u);
    return __builtin_amdgcn_sad_hi_u8(mul1_mul(x1), 0u, s);
}

// Eight 16-bit windows (positions t_offset .. +7 of the tile, t_offset = 8 * lane) for integer bitrates,
// from register-resident tile words. gather(j) returns tile word j. Window order matches dq_dispatch:
// frag0 = (w0, w1), (w2, w3); frag1 = (w4, w5), (w6, w7)
template <int bits, typename G>
__device__ __forceinline__ void windows8(G gather, int lane, uint32_t* w)
{
    constexpr int W = bits * 256 / 32;
    const int t = lane << 3;
    if constexpr (bits == 2)
    {
        const int i1 = t >> 4;
        uint32_t a = gather((i1 + 15) & 15);
        uint32_t b = gather(i1);
        b = __builtin_amdgcn_alignbit(a, b, ((~t) & 8) << 1);
        #pragma unroll
        for (int j = 0; j < 8; ++j) w[7 - j] = (b >> (2 * j)) & 0xffff;
    }
    else if constexpr (bits == 3)
    {
        const int b1 = (t + 257) * 3;
        const int b2 = b1 + 21;
        const int i0 = (b1 - 16) / 32;
        const int i2 = (b2 - 1) / 32;
        const int s2 = (i2 + 1) * 32 - b2;
        uint32_t a = gather(i0 % W);
        uint32_t b = gather(i2 % W);
        const Funnel f(b, a, s2);
        uint32_t w7 = f.at<0>();
        uint32_t w3 = f.at<12>();
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            w[7 - j] = (w7 >> (3 * j)) & 0xffff;
            w[3 - j] = (w3 >> (3 * j)) & 0xffff;
        }
    }
    else if constexpr (bits == 4)
    {
        const int i1 = t >> 3;
        uint32_t a = gather((i1 + 31) & 31);
        uint32_t b = gather(i1);
        uint32_t s = fsh(b, a, 20);
        #pragma unroll
        for (int j = 0; j < 5; ++j) w[7 - j] = (b >> (4 * j)) & 0xffff;
        #pragma unroll
        for (int j = 0; j < 3; ++j) w[2 - j] = (s >> (4 * j)) & 0xffff;
    }
    else  // 5, 6: two groups of four windows (dq4)
    {
        #pragma unroll
        for (int g = 0; g < 2; ++g)
        {
            const int b0 = (t + 4 * g + 257) * bits - 16;
            const int b2 = b0 + 3 * bits + 16;
            const int i0 = b0 / 32;
            const int i2 = (b2 - 1) / 32;
            const int s2 = (i2 + 1) * 32 - b2;
            uint32_t a = gather(i0 % W);
            uint32_t b = gather(i2 % W);
            const Funnel f(b, a, s2);
            w[4 * g + 3] = f.at<0>() & 0xffff;
            w[4 * g + 2] = f.at<bits>() & 0xffff;
            w[4 * g + 1] = f.at<2 * bits>() & 0xffff;
            w[4 * g + 0] = f.at<3 * bits>() & 0xffff;
        }
    }
}

}  // namespace exl3_rdna3_ns

// One block: output columns group * 128 .. +127, rows rc * MR .. +MR-1, k-slices of split
template <int bits, bool half_k, int cb, int MR, bool c_fp32>
__global__ __launch_bounds__(EXL3_RDNA3_THREADS)
void exl3_rdna3_kernel
(
    const uint2* __restrict__ xh,       // input after suh / Hadamard, pair layout (exl3_rdna3_had_kernel)
    const uint32_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ counters,
    const float* __restrict__ xcs,      // per-128-block sums of xh
    float* __restrict__ ws,
    const half* __restrict__ svh,
    int splits,
    int ks_per_split
)
{
    using namespace exl3_rdna3_ns;

    const int groups = size_n / 128;
    const int rc = blockIdx.y;
    const int group = blockIdx.x % groups;
    const int split = blockIdx.x / groups;

    constexpr int TWORDS = half_k ? 4 * (2 * bits + 1) : 8 * bits;   // uint32 per 16x16 tile
    // Raw integer path: each lane loads exactly the tile words its eight windows need (2, or 4 for 5 / 6
    // bits) instead of one slice of the tile plus cross-lane gathers (ds_bpermute, and a select for tiles
    // over 32 words). The wave still reads the same tile bytes (coalesced), the arithmetic is unchanged
    // (4-bit single-row matmuls stay on the gather path, measured ~2% faster there)
    constexpr bool DIRECT = cb == 2 && !half_k && bits >= 2 && bits <= 6 && (bits != 4 || MR >= 4);
    constexpr int LPT = DIRECT ? (bits >= 5 ? 4 : 2) : (TWORDS + 31) / 32;   // words per lane per tile
    constexpr int KCS = XS_BYTES / (MR * 2) / 16 / 8 * 8;              // k-slices per staged x chunk
    static_assert(KCS % 8 == 0, "x chunk must cover whole 128-element Hadamard blocks");
    // Register-resident tile words and raw mul1 decode (codebook affine map folded into the epilogue)
    constexpr bool RAW = cb == 2 && !half_k && bits >= 2 && bits <= 6;
    constexpr int PF = 4;   // prefetch ring depth, k-slices (4 beat 8 at 1-8 rows with warm clocks)

    // x chunk, [MR][KCS][4] pairs: .x = x[16 ks + 2q .. +1], .y = x[16 ks + 8 + 2q .. +1]
    __shared__ uint2 xs[MR * KCS * 4];
    __shared__ uint32_t stage[RAW ? 1 : NT][2][TWORDS];
    __shared__ float red[MR][128];
    __shared__ float xsum[MR];
    __shared__ int last_flag;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q = lane & 3;

    const int row0 = rc * MR;
    const int rows = min(MR, size_m - row0);
    const int kslices = size_k / 16;
    const int ks_begin = split * ks_per_split;
    const int ks_end = min(kslices, ks_begin + ks_per_split);
    const int nt = group * NT + warp;                // this wave's n-tile
    const uint32_t sstride_b = (uint32_t) (size_n / 16) * TWORDS * 4u;   // bytes per k-slice row of tiles

    if constexpr (RAW)
    {
        // Codebook bias term: sum of the transformed input over this block's k-range, per row
        if (threadIdx.x < MR)
        {
            float s = 0.0f;
            if ((int) threadIdx.x < rows)
                for (int c = ks_begin / 8; c < ks_end / 8; ++c)
                    s += xcs[(size_t) (row0 + threadIdx.x) * (kslices / 8) + c];
            xsum[threadIdx.x] = s;
        }
    }

    // Raw buffer over the trellis (byte offsets; EXL3 tensors are well below 2 GB)
    const __amdgpu_buffer_rsrc_t brsrc = __builtin_amdgcn_make_buffer_rsrc((void*) B, (short) 0, 0x7fffffff, 0x31004000);

    float acc[MR][2];
    #pragma unroll
    for (int m = 0; m < MR; ++m) acc[m][0] = acc[m][1] = 0.0f;

    for (int kc0 = ks_begin; kc0 < ks_end; kc0 += KCS)
    {
        const int nk = min(ks_end - kc0, KCS);   // multiple of 8

        // Weight prefetch first, so the stream is in flight while x is staged. nk is a multiple of PF (chunks
        // are whole 128-element blocks), so the unrolled body below is branch-free; the prefetch past the
        // end re-reads the last slice instead of branching, which keeps vmcnt waits incremental
        uint32_t pf[PF][LPT];
        int lofs[LPT];
        #pragma unroll
        for (int l = 0; l < LPT; ++l) lofs[l] = min(l * 32 + lane, TWORDS - 1);
        if constexpr (DIRECT)
        {
            // Word indices in windows8's gather order, from windows8 itself (so the two can't disagree)
            int c = 0;
            uint32_t unused[8];
            auto record = [&] (int jw) -> uint32_t { lofs[c++] = jw; return 0u; };
            windows8<bits>(record, lane, unused);
        }

        // Per-lane byte offset within a k-slice row (vector) + k-slice offset (uniform, scalar), 32-bit
        uint32_t voff[LPT];
        #pragma unroll
        for (int l = 0; l < LPT; ++l) voff[l] = (uint32_t) (nt * TWORDS + lofs[l]) * 4u;
        auto ldw = [&] (int slice, int l) -> uint32_t
        {
            return __builtin_amdgcn_raw_buffer_load_b32(brsrc, voff[l], (uint32_t) slice * sstride_b, 6);   // slc | dlc: streaming
        };

        #pragma unroll
        for (int d = 0; d < PF; ++d)
            #pragma unroll
            for (int l = 0; l < LPT; ++l)
                pf[d][l] = ldw(kc0 + d, l);

        // Stage this chunk of the transformed input: nk * 32 contiguous bytes per row, 16-byte copies.
        // Rows past size_m are zero so the unrolled MAC loop needs no row guards
        __syncthreads();
        {
            const int per_row = nk * 2;   // uint4 per row
            for (int idx = threadIdx.x; idx < MR * per_row; idx += EXL3_RDNA3_THREADS)
            {
                const int m = idx / per_row;
                const int j = idx % per_row;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m < rows)
                    v = ((const uint4*) (xh + ((size_t) (row0 + m) * kslices + kc0) * 4))[j];
                ((uint4*) (xs + (size_t) m * KCS * 4))[j] = v;
            }
        }
        __syncthreads();

        for (int ib = 0; ib < nk; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                const int i = ib + d;
                uint32_t w[LPT];
                #pragma unroll
                for (int l = 0; l < LPT; ++l) w[l] = pf[d][l];

                // Refill slot d once its words are consumed (after the gathers below), so the load can target
                // the same registers: loading earlier makes the compiler rename the ring and copy it back at the
                // loop edge, which waits for every outstanding load (vmcnt(0)). Buffer-load intrinsic: with
                // plain loads InstCombine folds phi(load, load) into load(phi(addr)), moving the prefetch to the
                // point of use and serializing the stream
                const int inext = min(i + PF, nk - 1);
                auto refill = [&] ()
                {
                    #pragma unroll
                    for (int l = 0; l < LPT; ++l) pf[d][l] = ldw(kc0 + inext, l);
                    // Pin the load here: the scheduler otherwise sinks the refills to the loop end in reverse
                    // order, and the next iteration's first use then waits for all of them
                    __builtin_amdgcn_sched_barrier(0);
                };

                uint32_t w00, w01, w10, w11;   // this tile's decoded weights (packed half2)
                if constexpr (RAW)
                {
                    // Tile words resolved in-wave (ds_bpermute), no LDS round trip
                    int wc = 0;
                    auto gather = [&] (int j) -> uint32_t
                    {
                        if constexpr (DIRECT) return w[wc++];
                        else if constexpr (LPT == 1) return __shfl(w[0], j);
                        else
                        {
                            uint32_t lo = __shfl(w[0], j & 31);
                            uint32_t hi = __shfl(w[1], j & 31);
                            return j < 32 ? lo : hi;
                        }
                    };
                    uint32_t win[8];
                    windows8<bits>(gather, lane, win);
                    refill();
                    w00 = mul1_raw_pair(win[0], win[1]);
                    w01 = mul1_raw_pair(win[2], win[3]);
                    w10 = mul1_raw_pair(win[4], win[5]);
                    w11 = mul1_raw_pair(win[6], win[7]);
                }
                else
                {
                    uint32_t* st = stage[warp][d & 1];
                    #pragma unroll
                    for (int l = 0; l < LPT; ++l)
                        if (l * 32 + lane < TWORDS)
                            st[l * 32 + lane] = w[l];
                    refill();
                    wave_sync();
                    FragB f0, f1;
                    dq_dispatch<bits, cb, half_k>(st, lane << 3, f0, f1);
                    w00 = h2u(f0[0]); w01 = h2u(f0[1]);
                    w10 = h2u(f1[0]); w11 = h2u(f1[1]);
                }

                #pragma unroll
                for (int m = 0; m < MR; ++m)
                {
                    const uint2 xv = xs[(m * KCS + i) * 4 + q];
                    acc[m][0] = dot2(w00, xv.x, acc[m][0]);
                    acc[m][0] = dot2(w01, xv.y, acc[m][0]);
                    acc[m][1] = dot2(w10, xv.x, acc[m][1]);
                    acc[m][1] = dot2(w11, xv.y, acc[m][1]);
                }
            }
        }
    }

    // Reduce over the four lanes sharing a column; lane l holds columns l / 4 and 8 + l / 4 of its tile
    #pragma unroll
    for (int m = 0; m < MR; ++m)
    {
        #pragma unroll
        for (int f = 0; f < 2; ++f)
        {
            float v = acc[m][f];
            v += __shfl_xor(v, 1);
            v += __shfl_xor(v, 2);
            if constexpr (RAW)
            {
                // Codebook value = k_inv * h + k_bias: sum(w * x) = k_inv * sum(h * x) + k_bias * sum(x)
                const float k_inv = __half2float(__ushort_as_half(0x1eee));
                const float k_bias = __half2float(__ushort_as_half(0xc931));
                v = fmaf(v, k_inv, k_bias * xsum[m]);
            }
            acc[m][f] = v;
        }
        if (q == 0)
        {
            red[m][warp * 16 + (lane >> 2)] = acc[m][0];
            red[m][warp * 16 + 8 + (lane >> 2)] = acc[m][1];
        }
    }
    __syncthreads();

    // Cross-split reduction: last block to arrive for this (row chunk, column group) sums the partials
    if (splits > 1)
    {
        const int gidx = rc * groups + group;
        float* wp = ws + ((size_t) split * size_m + row0) * size_n + group * 128;
        for (int idx = threadIdx.x; idx < rows * 128; idx += EXL3_RDNA3_THREADS)
        {
            const int m = idx >> 7, c = idx & 127;
            wp[(size_t) m * size_n + c] = red[m][c];
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int prev = atomicAdd(&counters[gidx], 1);
            last_flag = prev == splits - 1;
        }
        __syncthreads();
        if (!last_flag) return;
        __threadfence();

        for (int idx = threadIdx.x; idx < rows * 128; idx += EXL3_RDNA3_THREADS)
        {
            const int m = idx >> 7, c = idx & 127;
            const float* rp = ws + (size_t) (row0 + m) * size_n + group * 128 + c;
            float s = 0.0f;
            for (int sp = 0; sp < splits; ++sp)
                s += __builtin_nontemporal_load(rp + (size_t) sp * size_m * size_n);
            red[m][c] = s;
        }
        if (threadIdx.x == 0) counters[gidx] = 0;
        __syncthreads();
    }

    // Output Hadamard, 1/sqrt(128), svh post-scale
    for (int m = warp; m < rows; m += NT)
    {
        float h0 = red[m][lane * 4 + 0];
        float h1 = red[m][lane * 4 + 1];
        float h2 = red[m][lane * 4 + 2];
        float h3 = red[m][lane * 4 + 3];
        had128(h0, h1, h2, h3, lane);
        const int col = group * 128 + lane * 4;
        const half2* sp = (const half2*) (svh + col);
        const half2 s01 = sp[0], s23 = sp[1];
        const float r = 0.088388347648f;
        h0 *= r * __low2float(s01);
        h1 *= r * __high2float(s01);
        h2 *= r * __low2float(s23);
        h3 *= r * __high2float(s23);
        const size_t o = (size_t) (row0 + m) * size_n + col;
        if constexpr (c_fp32)
        {
            *((float4*) (((float*) C) + o)) = make_float4(h0, h1, h2, h3);
        }
        else
        {
            half2* cp = (half2*) (((half*) C) + o);
            cp[0] = __floats2half2_rn(h0, h1);
            cp[1] = __floats2half2_rn(h2, h3);
        }
    }
}

// Kernel table for one integer bitrate. Half-integer rates (bits + 0.5) exist for bits 1..3 with mul1 only
#define EXL3_RDNA3_MR_SWITCH(bits, hk, cb, fp32)                                           \
    switch (mr_idx)                                                                         \
    {                                                                                       \
        case 0: return exl3_rdna3_kernel<bits, hk, cb, 1, fp32>;                            \
        case 1: return exl3_rdna3_kernel<bits, hk, cb, 2, fp32>;                            \
        case 2: return exl3_rdna3_kernel<bits, hk, cb, 4, fp32>;                            \
        case 3: return exl3_rdna3_kernel<bits, hk, cb, 8, fp32>;                            \
        case 4: return exl3_rdna3_kernel<bits, hk, cb, 16, fp32>;                           \
        case 5: return exl3_rdna3_kernel<bits, hk, cb, 3, fp32>;                            \
        case 6: return exl3_rdna3_kernel<bits, hk, cb, 5, fp32>;                            \
        case 7: return exl3_rdna3_kernel<bits, hk, cb, 6, fp32>;                            \
        case 8: return exl3_rdna3_kernel<bits, hk, cb, 12, fp32>;                           \
    }                                                                                       \
    return nullptr;

#define EXL3_RDNA3_INT_BODY(bits)                                                           \
    if (cb == 0) { if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, false, 0, true) } else { EXL3_RDNA3_MR_SWITCH(bits, false, 0, false) } } \
    if (cb == 1) { if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, false, 1, true) } else { EXL3_RDNA3_MR_SWITCH(bits, false, 1, false) } } \
    if (cb == 2) { if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, false, 2, true) } else { EXL3_RDNA3_MR_SWITCH(bits, false, 2, false) } } \
    return nullptr;

#define EXL3_RDNA3_GETTER(bits)                                                             \
fp_exl3_rdna3_kernel exl3_rdna3_get_k##bits(bool half_k, int cb, int mr_idx, bool c_fp32)   \
{                                                                                           \
    if (half_k) return nullptr;                                                             \
    EXL3_RDNA3_INT_BODY(bits)                                                               \
}

#define EXL3_RDNA3_GETTER_HALF(bits)                                                        \
fp_exl3_rdna3_kernel exl3_rdna3_get_k##bits(bool half_k, int cb, int mr_idx, bool c_fp32)   \
{                                                                                           \
    if (half_k)                                                                             \
    {                                                                                       \
        if (cb != 2) return nullptr;                                                        \
        if (c_fp32) { EXL3_RDNA3_MR_SWITCH(bits, true, 2, true) }                           \
        else        { EXL3_RDNA3_MR_SWITCH(bits, true, 2, false) }                          \
    }                                                                                       \
    EXL3_RDNA3_INT_BODY(bits)                                                               \
}

#else

#define EXL3_RDNA3_GETTER(bits) \
fp_exl3_rdna3_kernel exl3_rdna3_get_k##bits(bool, int, int, bool) { return nullptr; }
#define EXL3_RDNA3_GETTER_HALF(bits) EXL3_RDNA3_GETTER(bits)

#endif
