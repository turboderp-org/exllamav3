#pragma once

// ROCm side of det_gemm.cuh (included at its end on ROCm only): the int8 GEMM blocks as RDNA WMMA warp tiles
// (v_wmma_i32_16x16x16_iu8) reading the kernels' swizzled LDS tiles directly, plus the copy helpers the
// kernels stage those tiles with.
//
// The per-chunk sums are exact int32 (14-bit operands split into int8 hi / lo, at most 128 terms), so every
// correct implementation produces the same values whatever its instruction or summation order; the fp32 flush
// (det_flush) runs in the same fixed order as on CUDA. Ranks of a tensor-parallel group therefore agree on any
// mix of RDNA generations, which is the property the deterministic kernels exist for.

// Shared-window addresses are LDS byte offsets (address space 3), the counterpart of PTX's 32-bit shared
// addresses
typedef unsigned det_u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) det_u32x4 det_lds_u32x4;

__device__ __forceinline__ unsigned det_smem_u32(const void* p)
{
    return (unsigned) (uintptr_t) (const __attribute__((address_space(3))) void*) p;
}

// No asynchronous copy: a synchronous 16-byte load/store. As with cp.async's source size, only the first
// src_bytes bytes come from global memory and the rest of the 16 are zero (K tails). Every consumer of a
// staged tile passes a __syncthreads() after the wait, and a synchronous store has completed by then, so
// commit and wait have nothing left to do
__device__ __forceinline__ void det_cp_async16(unsigned dst, const void* src, int src_bytes)
{
    det_u32x4 v = { 0, 0, 0, 0 };
    if (src_bytes == 16)
        v = *(const det_u32x4*) src;
    else
        for (int i = 0; i < src_bytes; ++i)
            ((unsigned char*) &v)[i] = ((const unsigned char*) src)[i];
    *(det_lds_u32x4*) (uintptr_t) dst = v;
}
__device__ __forceinline__ void det_cp_async_commit() {}
template <int N> __device__ __forceinline__ void det_cp_async_wait() {}

// int8 WMMA, 16 x 16 x 16, D = A B^T + C with A and B^T both row-major in the LDS tiles (rows of K bytes).
// Every lane feeds row (lane % 16) of A and of B^T. RDNA3 (gfx11) takes all 16 K values of that row in each
// lane (both half-waves carry the same data); RDNA4 (gfx12) splits them, lane L taking K = 8 (L / 16) .. + 7.
// Element v of the int32 result belongs to column (lane % 16) and to row 2 v + L / 16 on gfx11,
// v + 8 (L / 16) on gfx12
typedef int det_v8i __attribute__((ext_vector_type(8)));
#if defined(__GFX12__)
    typedef int det_wmma_op __attribute__((ext_vector_type(2)));
#else
    typedef int det_wmma_op __attribute__((ext_vector_type(4)));
#endif
typedef __attribute__((address_space(3))) const det_wmma_op det_lds_wmma_op;

__device__ __forceinline__ det_v8i det_wmma_iu8(det_wmma_op a, det_wmma_op b, det_v8i c)
{
#if defined(__GFX12__)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
#else
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#endif
}

// Operand of one k16 step from a swizzled tile (ROWB-byte rows): row (lane % 16) of the 16 starting at row0,
// 16-byte piece kpiece (gfx12: this lane's 8-byte half of it)
template <int ROWB>
__device__ __forceinline__ det_wmma_op det_wmma_load(unsigned tile, int row0, int kpiece, int lane)
{
    unsigned addr = tile + det_swz<ROWB>(row0 + (lane & 15), kpiece);
#if defined(__GFX12__)
    addr += 8 * ((lane >> 4) & 1);
#endif
    return *(det_lds_wmma_op*) (uintptr_t) addr;
}

// Row within the 16-row block of accumulator element v
__device__ __forceinline__ int det_wmma_row(int v, int lane)
{
#if defined(__GFX12__)
    return v + 8 * ((lane >> 4) & 1);
#else
    return 2 * v + ((lane >> 4) & 1);
#endif
}

// A warp's (MT * 16) x (NT * 16) output tile, accumulated in fp32 over K chunks. Element (i, j, v) is output
// row 16 i + det_wmma_row(v), column 16 j + lane % 16 of the tile
template <int MT, int NT>
struct DetWmmaTile
{
    float acc[MT][NT][8];

    __device__ __forceinline__ void clear()
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j)
                #pragma unroll
                for (int v = 0; v < 8; ++v) acc[i][j][v] = 0.0f;
    }

    // One K chunk of KCH from the hi / lo planes of A (rows a_row0 ..) and B^T (rows b_row0 ..): the exact sums
    // hh = A_hi B_hi^T and x = A_hi B_lo^T + A_lo B_hi^T, flushed with the chunk scale of each A row
    // (sa[a_row0 + row])
    template <int ROWB, int KCH>
    __device__ __forceinline__ void chunk(unsigned a_hi, unsigned a_lo, int a_row0, unsigned b_hi, unsigned b_lo, int b_row0,
                                          const float* sa, int lane)
    {
        det_v8i hh[MT][NT], x[MT][NT];
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j) { hh[i][j] = det_v8i(0); x[i][j] = det_v8i(0); }

        #pragma unroll
        for (int kp = 0; kp < KCH / 16; ++kp)
        {
            det_wmma_op ah[MT], al[MT], bh[NT], bl[NT];
            #pragma unroll
            for (int i = 0; i < MT; ++i)
            {
                ah[i] = det_wmma_load<ROWB>(a_hi, a_row0 + 16 * i, kp, lane);
                al[i] = det_wmma_load<ROWB>(a_lo, a_row0 + 16 * i, kp, lane);
            }
            #pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                bh[j] = det_wmma_load<ROWB>(b_hi, b_row0 + 16 * j, kp, lane);
                bl[j] = det_wmma_load<ROWB>(b_lo, b_row0 + 16 * j, kp, lane);
            }
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j)
                {
                    hh[i][j] = det_wmma_iu8(ah[i], bh[j], hh[i][j]);
                    x[i][j] = det_wmma_iu8(ah[i], bl[j], x[i][j]);
                    x[i][j] = det_wmma_iu8(al[i], bh[j], x[i][j]);
                }
        }

        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                const float s = sa[a_row0 + 16 * i + det_wmma_row(v, lane)];
                #pragma unroll
                for (int j = 0; j < NT; ++j) acc[i][j][v] = det_flush(hh[i][j][v], x[i][j][v], s, acc[i][j][v]);
            }
    }
};

// AMDGPU folds an fp32 multiply or FMA whose result is converted to half into one mixed-precision instruction
// (v_fma_mix), which rounds once, straight to half, where CUDA rounds to fp32 first and then to half. An empty
// asm on the fp32 value keeps the two roundings. Only the deterministic translation units include this header,
// so the override is limited to them
__host__ __device__ __forceinline__ __half det_float2half_rn(float x)
{
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(x));
#endif
    return __float2half_rn(x);
}
__host__ __device__ __forceinline__ __half2 det_floats2half2_rn(float a, float b)
{
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(a), "+v"(b));
#endif
    return __floats2half2_rn(a, b);
}
#define __float2half_rn det_float2half_rn
#define __floats2half2_rn det_floats2half2_rn
