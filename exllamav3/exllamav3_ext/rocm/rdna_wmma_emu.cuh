#pragma once

// WMMA emulation for RDNA1/2 device passes (no v_wmma instructions before gfx11). The kernels keep the gfx11
// fragment layouts; what changes is how one 16x16x16 step is computed from them.
//
// The builtins' semantics in those layouts, D = X Y + C: operand X's lane L holds row (L % 16) of X, operand
// Y's lane L holds column (L % 16) of Y, and element v of the result in lane L is D[2 v + L / 16][L % 16].
// Each lane therefore has its own Y column and fetches the X row it needs from lane (2 v + L / 16) with a
// shuffle, then forms the 16-term dot product in fp32 (or int32). Slow, but functional; the fast paths on
// these parts are the GEMV kernels, which do not use WMMA.

#include <hip/hip_runtime.h>
#include <stdint.h>

#if defined(EXL3_FORCE_WMMA_EMULATION) || (defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX11__) && !defined(__GFX12__))
    #define EXL3_WMMA_EMULATED 1
#endif

namespace rdna_wmma_emu
{

typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));
typedef int      int32x4_t __attribute__((ext_vector_type(4)));
typedef int      int32x8_t __attribute__((ext_vector_type(8)));

// Operand X of lane `lane`, as NDW dwords
template <int NDW>
__device__ __forceinline__ void fetch_row(const void* x, int lane, int* dst)
{
    const int* src = (const int*) x;
    #pragma unroll
    for (int j = 0; j < NDW; ++j) dst[j] = __shfl(src[j], lane);
}

__device__ __forceinline__ int result_row(int v)
{
    return 2 * v + ((threadIdx.x >> 4) & 1);
}

// fp32 += fp16 x fp16
__device__ __forceinline__ float8_t mma_f32_f16(half16_t x, half16_t y, float8_t c)
{
    const _Float16* yp = (const _Float16*) &y;
    #pragma unroll
    for (int v = 0; v < 8; ++v)
    {
        int xr[8];
        fetch_row<8>(&x, result_row(v), xr);
        const _Float16* xp = (const _Float16*) xr;
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) s = fmaf((float) xp[k], (float) yp[k], s);
        c[v] += s;
    }
    return c;
}

// fp32 += bf16 x bf16
__device__ __forceinline__ float8_t mma_f32_bf16(const void* x, const void* y, float8_t c)
{
    const unsigned short* yp = (const unsigned short*) y;
    #pragma unroll
    for (int v = 0; v < 8; ++v)
    {
        int xr[8];
        fetch_row<8>(x, result_row(v), xr);
        const unsigned short* xp = (const unsigned short*) xr;
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k)
            s = fmaf(__uint_as_float((unsigned) xp[k] << 16), __uint_as_float((unsigned) yp[k] << 16), s);
        c[v] += s;
    }
    return c;
}

// fp16 += fp16 x fp16, accumulator in the half-slots i * 2 + opsel of a 16-wide fragment, the others untouched
template <bool opsel>
__device__ __forceinline__ half16_t mma_f16_f16(half16_t x, half16_t y, half16_t c)
{
    const _Float16* yp = (const _Float16*) &y;
    _Float16* cp = (_Float16*) &c;
    #pragma unroll
    for (int v = 0; v < 8; ++v)
    {
        int xr[8];
        fetch_row<8>(&x, result_row(v), xr);
        const _Float16* xp = (const _Float16*) xr;
        float s = (float) cp[v * 2 + (opsel ? 1 : 0)];
        #pragma unroll
        for (int k = 0; k < 16; ++k) s = fmaf((float) xp[k], (float) yp[k], s);
        cp[v * 2 + (opsel ? 1 : 0)] = (_Float16) s;
    }
    return c;
}

// int32 += int8 x int8, each operand signed or unsigned, optionally saturating
template <bool signed_x, bool signed_y, bool clamp>
__device__ __forceinline__ int32x8_t mma_i32_i8(int32x4_t x, int32x4_t y, int32x8_t c)
{
    const unsigned char* yp = (const unsigned char*) &y;
    #pragma unroll
    for (int v = 0; v < 8; ++v)
    {
        int xr[4];
        fetch_row<4>(&x, result_row(v), xr);
        const unsigned char* xp = (const unsigned char*) xr;
        int s = 0;
        #pragma unroll
        for (int k = 0; k < 16; ++k)
        {
            int a = signed_x ? (int) (signed char) xp[k] : (int) xp[k];
            int b = signed_y ? (int) (signed char) yp[k] : (int) yp[k];
            s += a * b;
        }
        if (clamp)
        {
            long long t = (long long) c[v] + s;
            c[v] = t > 2147483647LL ? 2147483647 : (t < -2147483648LL ? (int) -2147483648LL : (int) t);
        }
        else c[v] += s;
    }
    return c;
}

}  // namespace rdna_wmma_emu
