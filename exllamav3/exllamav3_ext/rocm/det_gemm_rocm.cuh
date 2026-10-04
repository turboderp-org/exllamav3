#pragma once

// ROCm implementations of det_gemm.cuh's PTX primitives (int8 mma.sync m16n8k32, ldmatrix.x4, cp.async and
// the shared-window addresses they take). Included by det_gemm.cuh on ROCm only.
//
// The MMA keeps NVIDIA's per-lane fragment layout, so the deterministic kernels run unchanged. It is
// emulated with lane shuffles and signed 4-way byte dot products; integer sums are exact, so the results are
// bit-identical to the tensor-core instruction, which is what makes the deterministic path worth having here.
//
// Fragment layouts, per the PTX ISA (m16n8k32, .s8), with g = lane / 4 and t = lane % 4:
//   A (16x32, row-major): a[0] = A[g][4t .. 4t+3],   a[1] = A[g+8][4t ..],   a[2] = A[g][16+4t ..],   a[3] = A[g+8][16+4t ..]
//   B (32x8, col-major):  b[0] = B[4t .. 4t+3][g],   b[1] = B[16+4t ..][g]
//   C (16x8, s32):        c[0] = D[g][2t], c[1] = D[g][2t+1], c[2] = D[g+8][2t], c[3] = D[g+8][2t+1]

// Shared-window addresses are LDS byte offsets (address space 3), the counterpart of PTX's 32-bit shared
// addresses
typedef __attribute__((address_space(3))) unsigned det_lds_u32;
typedef unsigned det_u32x4 __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) det_u32x4 det_lds_u32x4;

__device__ __forceinline__ unsigned det_smem_u32(const void* p)
{
    return (unsigned) (uintptr_t) (const __attribute__((address_space(3))) void*) p;
}

__device__ __forceinline__ int det_sdot4(unsigned a, unsigned b, int c)
{
    return __builtin_amdgcn_sudot4(true, (int) a, true, (int) b, c, false);
}

__device__ __forceinline__ void det_mma_s8(int* c, const unsigned* a, const unsigned* b)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    // Rows g and g + 8 of A live in lanes 4g .. 4g + 3; columns 2t and 2t + 1 of B in lanes 8t .. 8t + 3
    // and 8t + 4 .. 8t + 7. Lane 4r + j holds k = 4j .. 4j + 3 in its first word and 16 + 4j .. in its second
    int s00 = c[0], s01 = c[1], s10 = c[2], s11 = c[3];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int la = 4 * g + j;
        const unsigned r0_lo = (unsigned) __shfl((int) a[0], la);
        const unsigned r1_lo = (unsigned) __shfl((int) a[1], la);
        const unsigned r0_hi = (unsigned) __shfl((int) a[2], la);
        const unsigned r1_hi = (unsigned) __shfl((int) a[3], la);
        const int lb0 = 8 * t + j;
        const int lb1 = 8 * t + 4 + j;
        const unsigned c0_lo = (unsigned) __shfl((int) b[0], lb0);
        const unsigned c0_hi = (unsigned) __shfl((int) b[1], lb0);
        const unsigned c1_lo = (unsigned) __shfl((int) b[0], lb1);
        const unsigned c1_hi = (unsigned) __shfl((int) b[1], lb1);
        s00 = det_sdot4(r0_hi, c0_hi, det_sdot4(r0_lo, c0_lo, s00));
        s01 = det_sdot4(r0_hi, c1_hi, det_sdot4(r0_lo, c1_lo, s01));
        s10 = det_sdot4(r1_hi, c0_hi, det_sdot4(r1_lo, c0_lo, s10));
        s11 = det_sdot4(r1_hi, c1_hi, det_sdot4(r1_lo, c1_lo, s11));
    }
    c[0] = s00; c[1] = s01; c[2] = s10; c[3] = s11;
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

// ldmatrix.sync.aligned.m8n8.x4.b16: lanes 8i .. 8i + 7 supply the row addresses of matrix i, and lane l
// receives, for each matrix i, the 32-bit word at row l / 4, word l % 4 of that matrix
__device__ __forceinline__ void det_ldmatrix_x4(unsigned* r, unsigned addr)
{
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const unsigned row = (unsigned) __shfl((int) addr, 8 * i + (lane >> 2));
        r[i] = *(det_lds_u32*) (uintptr_t) (row + 4 * (lane & 3));
    }
}

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
