#pragma once

// Plain C++ equivalents of the integer and bit-field PTX instructions used in the kernels. The CUDA
// build keeps its inline PTX (ptx.cuh, codebook.cuh, exl3_gemv_int8_kernel.cuh); these replace it on
// ROCm, where inline PTX does not exist. They are always defined so tests/test_ptx_portable.py can check
// each one against the PTX instruction it stands in for on an NVIDIA device.

#include <cstdint>

// lop3.b32 d, a, b, c, 0x6a: the lookup table 0x6a is (a & b) ^ c (with a = 0xf0, b = 0xcc, c = 0xaa,
// (0xf0 & 0xcc) ^ 0xaa = 0x6a)
__device__ __forceinline__ uint32_t exl3_lop3_6a(uint32_t a, uint32_t b, uint32_t c)
{
    return (a & b) ^ c;
}

// shf.r.wrap.b32 d, lo, hi, n: the low 32 bits of the 64-bit value hi:lo shifted right by n mod 32
__device__ __forceinline__ uint32_t exl3_shf_r_wrap(uint32_t lo, uint32_t hi, uint32_t n)
{
    uint64_t v = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    return static_cast<uint32_t>(v >> (n & 31));
}

// bfe.u32 d, a, pos, 16, for pos < 32: bits beyond the top of a read as zero
__device__ __forceinline__ uint32_t exl3_bfe_u32_16(uint32_t a, uint32_t pos)
{
    return (a >> pos) & 0xffffu;
}

// bfe.u64 d, a, pos, len, returning the low 32 bits. A zero length, or a start position at or past the
// top of the value, yields zero; bits beyond the top read as zero
__device__ __forceinline__ uint32_t exl3_bfe_u64(uint64_t a, int pos, int len)
{
    if (len <= 0 || pos >= 64) return 0;
    uint64_t v = a >> pos;
    if (len < 64) v &= (1ull << len) - 1ull;
    return static_cast<uint32_t>(v);
}

// mul.lo.u32 / mul.hi.u32
__device__ __forceinline__ uint32_t exl3_mul_lo_u32(uint32_t x, uint32_t y)
{
    return x * y;
}

__device__ __forceinline__ uint32_t exl3_mul_hi_u32(uint32_t x, uint32_t y)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(x) * static_cast<uint64_t>(y)) >> 32);
}

// dp4a.u32.s32 d, a, b, c: the four unsigned bytes of a times the four signed bytes of b, summed onto c
__device__ __forceinline__ int exl3_dp4a_us(uint32_t a, uint32_t b, int c)
{
    // v_dot4_u32_i8 is a gfx11+ instruction (dot8-insts); the invocability test is per target, where
    // __has_builtin would be true for every AMD device pass
    #if defined(USE_ROCM) && defined(__HIP_DEVICE_COMPILE__) && !defined(EXL3_FORCE_SCALAR_DOT)
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_sudot4))
            return __builtin_amdgcn_sudot4(false, static_cast<int>(a), true, static_cast<int>(b), c, false);
    #endif
    int d = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        d += static_cast<int>((a >> (8 * i)) & 0xffu) * static_cast<int>(static_cast<int8_t>(b >> (8 * i)));
    return d;
}
