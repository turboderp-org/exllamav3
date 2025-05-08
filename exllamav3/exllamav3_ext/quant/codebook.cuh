#pragma once

// "3INST" procedural codebook

__device__ inline half decode_3inst(uint32_t x)
{
    x *= 89226354u;
    x += 64248484u;
    // x &= 0b10001111111111111000111111111111u;
    // x ^= 0b00111011011000000011101101100000u;
    // Compiler doesn't automatically generate LOP3
    asm volatile ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
    half2_uint32 xu(x);
    return __hadd(__low2half(xu.as_half2), __high2half(xu.as_half2));
}

__device__ inline half2 decode_3inst_2(uint32_t x0, uint32_t x1)
{
    x0 *= 89226354u;
    x1 *= 89226354u;
    x0 += 64248484u;
    x1 += 64248484u;
    // x0 &= 0b10001111111111111000111111111111u;
    // x1 &= 0b10001111111111111000111111111111u;
    // x0 ^= 0b00111011011000000011101101100000u;
    // x1 ^= 0b00111011011000000011101101100000u;
    // Compiler doesn't automatically generate LOP3
    asm volatile ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm volatile ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    half2_uint32 xu0(x0);
    half2_uint32 xu1(x1);
    half2 d0 = __halves2half2(__low2half(xu0.as_half2), __low2half(xu1.as_half2));
    half2 d1 = __halves2half2(__high2half(xu0.as_half2), __high2half(xu1.as_half2));
    return __hadd2(d0, d1);
}

__device__ inline float decode_3inst_f(uint64_t x)
{
    return __half2float(decode_3inst(x));
}

__device__ inline float decode_3inst_f_diff(uint64_t x, float d)
{
    return __half2float(decode_3inst(x)) - d;
}

// "2MAD" procedural codebook, much more overhead than 3INST, slightly better distribution at 2bpw

__device__ inline half decode_2mad(uint64_t x)
{
    x = x * 264435761u + 1013904223u;
    x = ((x * 1664525u) >> 32) + x;
    int32_t c = (int32_t) __dp4a((uint32_t) x, 0x01010101u, 0xFFFFFE02u);
    half y = __hmul(__int2half_rn(c), __float2half_rn(0.008415));
    return y;
}

__device__ inline float decode_2mad_f(uint64_t x)
{
    x = x * 264435761u + 1013904223u;
    x = ((x * 1664525u) >> 32) + x;
    int32_t c = (int32_t) __dp4a((uint32_t) x, 0x01010101u, 0xFFFFFE02u);
    float y = __int2float_rn(c) * 0.008415f;
    return y;
}

__device__ inline float decode_2mad_f_diff(uint64_t x, float d)
{
    x = x * 264435761u + 1013904223u;
    x = ((x * 1664525u) >> 32) + x;
    int32_t c = (int32_t) __dp4a((uint32_t) x, 0x01010101u, 0xFFFFFE02u);
    float y = fma(__int2float_rn(c), 0.008415f, -d);
    return y;
}

//

__device__ inline half decode_pcb(uint64_t x)
{
//    return decode_2mad(x);
    return decode_3inst(x);
}

__device__ inline float decode_pcb_f(uint64_t x)
{
//    return decode_2mad_f(x);
    return decode_3inst_f(x);
}

__device__ inline float decode_pcb_f_diff(uint64_t x, float d)
{
//    return decode_2mad_f_diff(x, d);
    return decode_3inst_f_diff(x, d);
}
