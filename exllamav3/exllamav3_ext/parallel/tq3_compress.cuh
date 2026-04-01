#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

// TQ3: 3-level ternary quantization for fp16 tensors.
//
// Each block of TQ3_BLOCK_SIZE (32) fp16 values is compressed to 10 bytes:
//   - 1x fp16 scale   (2 bytes)  — max absolute value in the block
//   - 2x uint32_t     (8 bytes)  — two bits per element: 00=-1, 01=0, 10=+1
//                                  (packed MSB→LSB for bp0 at bit 31, bp1 at bit 31)
//
// Encoding: for each element v / scale
//   |x| < TQ3_BOUNDARY  → ternary 0 (bp bit = 0, sign bit = 0)
//   x >= +TQ3_BOUNDARY  → ternary +1 (bp bit = 1, sign bit = 0)
//   x <= -TQ3_BOUNDARY  → ternary +1 magnitude, negative (bp bit = 1, sign bit = 1)
//
// bp0 holds the sign bits (1 = negative non-zero)
// bp1 holds the magnitude bits (1 = non-zero)
//
// Decompression: dst[i] = scale * (bp1_i ? (bp0_i ? -1.0h : +1.0h) : 0.0h)

#define TQ3_BLOCK_SIZE  32
#define TQ3_BOUNDARY    0.5f

// ---------------------------------------------------------------------------
// tq3_compress_block
//
// Compress 32 fp16 values from src[] into a (bp0, bp1, scale) triplet.
// All three output pointers must be writable by the calling thread.
// This is a pure device function — call once per block of 32 elements.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tq3_compress_block
(
    const half* __restrict__ src,
    uint32_t*   __restrict__ bp0,      // sign-bit plane (1 = negative non-zero)
    uint32_t*   __restrict__ bp1,      // magnitude-bit plane (1 = non-zero)
    half*       __restrict__ scale     // max abs value of the block
)
{
    // Pass 1: find max absolute value (in float for precision)
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < TQ3_BLOCK_SIZE; ++i)
    {
        float v = __half2float(src[i]);
        float av = (v < 0.0f) ? -v : v;
        if (av > max_abs) max_abs = av;
    }

    // Store scale (fp16); guard against zero denominator
    *scale = __float2half(max_abs);
    float inv_scale = (max_abs > 0.0f) ? (1.0f / max_abs) : 0.0f;

    // Pass 2: quantize → pack into two uint32 bit planes (bit 0 = element 0)
    uint32_t b0 = 0u;
    uint32_t b1 = 0u;
    #pragma unroll
    for (int i = 0; i < TQ3_BLOCK_SIZE; ++i)
    {
        float v = __half2float(src[i]) * inv_scale;
        float av = (v < 0.0f) ? -v : v;
        if (av >= TQ3_BOUNDARY)
        {
            b1 |= (1u << i);                     // non-zero
            if (v < 0.0f) b0 |= (1u << i);       // negative
        }
    }

    *bp0 = b0;
    *bp1 = b1;
}

// ---------------------------------------------------------------------------
// tq3_decompress_block
//
// Reconstruct 32 fp16 values from (bp0, bp1, scale) into dst[].
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tq3_decompress_block
(
    uint32_t    bp0,
    uint32_t    bp1,
    half        scale,
    half* __restrict__ dst
)
{
    float fscale = __half2float(scale);
    #pragma unroll
    for (int i = 0; i < TQ3_BLOCK_SIZE; ++i)
    {
        uint32_t mag  = (bp1 >> i) & 1u;
        uint32_t sign = (bp0 >> i) & 1u;
        float v = 0.0f;
        if (mag) v = sign ? -fscale : fscale;
        dst[i] = __float2half(v);
    }
}

// ---------------------------------------------------------------------------
// tq3_decompress_add_block
//
// Fused decompress + accumulate: dst[i] += decompressed[i]
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tq3_decompress_add_block
(
    uint32_t    bp0,
    uint32_t    bp1,
    half        scale,
    half* __restrict__ dst
)
{
    float fscale = __half2float(scale);
    #pragma unroll
    for (int i = 0; i < TQ3_BLOCK_SIZE; ++i)
    {
        uint32_t mag  = (bp1 >> i) & 1u;
        uint32_t sign = (bp0 >> i) & 1u;
        if (mag)
        {
            float contrib = sign ? -fscale : fscale;
            dst[i] = __float2half(__half2float(dst[i]) + contrib);
        }
    }
}
