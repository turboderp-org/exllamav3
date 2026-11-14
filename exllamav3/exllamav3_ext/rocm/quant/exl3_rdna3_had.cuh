#pragma once

// Input transform of the RDNA3 EXL3 matmul (exl3_rdna3.cu): x * suh -> 128-point Hadamard -> 1/sqrt(128) -> fp16 in
// the matmul's LDS pair layout ([m][k/16][4] uint2), plus the per-128-block sums of the rounded values (codebook
// bias term)

#include <cuda_fp16.h>

namespace exl3_rdna3_had {

// Natural-order 128-point Walsh-Hadamard across one wave, lane holds elements 4 * lane .. + 3
__device__ __forceinline__ void had128(float& h0, float& h1, float& h2, float& h3, int lane)
{
    float s0 = h0 + h1, d0 = h0 - h1, s1 = h2 + h3, d1 = h2 - h3;
    h0 = s0 + s1; h1 = d0 + d1; h2 = s0 - s1; h3 = d0 - d1;
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        float p0 = __shfl_xor(h0, i);
        float p1 = __shfl_xor(h1, i);
        float p2 = __shfl_xor(h2, i);
        float p3 = __shfl_xor(h3, i);
        bool hi = lane & i;
        h0 = hi ? p0 - h0 : h0 + p0;
        h1 = hi ? p1 - h1 : h1 + p1;
        h2 = hi ? p2 - h2 : h2 + p2;
        h3 = hi ? p3 - h3 : h3 + p3;
    }
}

// One 128-element block c of row m, lane holding elements 4 * lane .. +3 (fp16)
__device__ __forceinline__ void transform_block
(
    half2 x01,
    half2 x23,
    const half* __restrict__ suh,
    uint2* __restrict__ xh,
    float* __restrict__ xcs,
    int m,
    int c,
    int size_k,
    int lane
)
{
    const int kbase = c * 128;
    const int kblocks = size_k / 128;
    const half2* sp = (const half2*) (suh + kbase + lane * 4);
    half2 a01 = __hmul2(x01, sp[0]);
    half2 a23 = __hmul2(x23, sp[1]);
    float h0 = __low2float(a01), h1 = __high2float(a01), h2 = __low2float(a23), h3 = __high2float(a23);
    had128(h0, h1, h2, h3, lane);
    const float r = 0.088388347648f;
    const half2 p0 = __floats2half2_rn(h0 * r, h1 * r);
    const half2 p1 = __floats2half2_rn(h2 * r, h3 * r);

    float s = __low2float(p0) + __high2float(p0) + __low2float(p1) + __high2float(p1);
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) s += __shfl_xor(s, i);
    if (lane == 0) xcs[(size_t) m * kblocks + c] = s;

    // Elements 4 * lane .. +3 of the block: k-slice c * 8 + lane / 4, offset j0 = 4 * (lane % 4)
    uint32_t* xrow = (uint32_t*) &xh[((size_t) m * (size_k / 16) + c * 8 + (lane >> 2)) * 4];
    const int j0 = (lane & 3) * 4;
    const int j1 = j0 + 2;
    xrow[((j0 & 7) >> 1) * 2 + (j0 >> 3)] = *reinterpret_cast<const uint32_t*>(&p0);
    xrow[((j1 & 7) >> 1) * 2 + (j1 >> 3)] = *reinterpret_cast<const uint32_t*>(&p1);
}

}  // namespace exl3_rdna3_had
