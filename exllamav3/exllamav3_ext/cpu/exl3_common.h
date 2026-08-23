#pragma once

// Host-side EXL3 format helpers shared between the CPU MoE GEMM path (moe_mul1.cpp) and the
// dense CPU reconstructor (reconstruct.cpp): the tensor-core tile permutation, packed-state
// extraction, the procedural codebook decoders, and the 128-point Sylvester transform.
//
// One implementation of the packing convention lives here; neither consumer may maintain a
// divergent copy.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include <c10/util/Half.h>

#if defined(__GNUC__) && defined(__linux__)
#define EXL3_CPU_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define EXL3_CPU_TARGET_AVX2
#endif

namespace exl3_common {

constexpr uint32_t MUL1_MULT = 0x83DCD12Du;
constexpr float HAD_SCALE = 0.08838834764831845f;  // 1/sqrt(128)

inline float half_to_float(uint16_t bits)
{
    return static_cast<float>(c10::Half(bits, c10::Half::from_bits()));
}

inline float half_add_bits(uint16_t a, uint16_t b)
{
    // Single-rounding half add, identical to __hadd on device
    c10::Half ha(a, c10::Half::from_bits());
    c10::Half hb(b, c10::Half::from_bits());
    return static_cast<float>(ha + hb);
}

// Matches tensor-core permutation baked into the EXL3 tile storage format: state index ->
// (row * 16 + col) within one 16x16 weight tile
constexpr std::array<uint16_t, 256> make_tc_perm()
{
    std::array<uint16_t, 256> p{};
    for (int t = 0; t < 32; ++t)
    {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        p[t * 8 + 0] = r0 * 16 + c0; p[t * 8 + 1] = r1 * 16 + c0;
        p[t * 8 + 2] = r2 * 16 + c0; p[t * 8 + 3] = r3 * 16 + c0;
        p[t * 8 + 4] = r0 * 16 + c1; p[t * 8 + 5] = r1 * 16 + c1;
        p[t * 8 + 6] = r2 * 16 + c1; p[t * 8 + 7] = r3 * 16 + c1;
    }
    return p;
}

constexpr std::array<uint16_t, 256> make_tc_perm_inv()
{
    std::array<uint16_t, 256> inv{};
    constexpr auto perm = make_tc_perm();
    for (int i = 0; i < 256; ++i) inv[perm[i]] = i;
    return inv;
}

inline uint32_t load_u32_(const uint16_t* ptr, int index)
{
    uint32_t v;
    std::memcpy(&v, ptr + index * 2, sizeof(v));
    return v;
}

// Extract state `t_offset`'s bits-bit code as a 16-bit window from one 16x16 tile's packed
// words (a uint16 array of length bits * 16)
template <int bits>
inline uint16_t decode_state_scalar(const uint16_t* packed, int t_offset)
{
    constexpr int words32 = bits * 256 / 32;
    const int b0 = t_offset * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const uint64_t merged = (static_cast<uint64_t>(load_u32_(packed, (b0 / 32) % words32)) << 32) |
                            load_u32_(packed, ((b1 - 1) / 32) % words32);
    return static_cast<uint16_t>(merged >> shift);
}

// LOP3 immediate 0x6a used by the 3inst/mcg codebook bit construction, extracted empirically
// against the GPU instruction: with PTX's LUT indexing (bit i = f(4a+2b+c)) the immediate
// encodes f(a, b, c) = (a & b) ^ c
inline uint32_t lop3_6a(uint32_t x)
{
    return (x & 0x8fff8fffu) ^ 0x3b603b60u;
}

// Procedural codebook decoders, bit-exact with decode_3inst in quant/codebook.cuh:
// cb 0 = default/3inst, cb 1 = mcg, cb 2 = mul1
template <int cb>
inline float decode_3inst(uint32_t x)
{
    if constexpr (cb == 0)
    {
        x = x * 89226354u + 64248484u;
    }
    else if constexpr (cb == 1)
    {
        x *= 0xCBAC1FEDu;
    }
    else
    {
        x *= MUL1_MULT;
    }

    if constexpr (cb == 2)
    {
        // acc = 0x6400 puts the integer into the fp16 bit range whose reinterpretation equals
        // (byte_sum + 1024.0)
        const uint32_t sum =
            (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + ((x >> 24) & 0xff) + 0x6400u;
        const float k_inv = half_to_float(0x1eee);   // 0.00677 = 1/147.7
        const float k_bias = half_to_float(0xc931);  // (-1024.0 - 510.0) * k_inv
        const float h = half_to_float(static_cast<uint16_t>(sum));
        return std::fma(h, k_inv, k_bias);
    }
    else
    {
        const uint32_t y = lop3_6a(x);
        return half_add_bits(static_cast<uint16_t>(y & 0xffff), static_cast<uint16_t>(y >> 16));
    }
}

// -------------------------------------------------------------------------------------------
//  Transforms
// -------------------------------------------------------------------------------------------

inline void hadamard_128_scalar(float* v)
{
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += 2 * width)
            for (int i = 0; i < width; ++i)
            {
                const float a = v[base + i];
                const float b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
}

EXL3_CPU_TARGET_AVX2
inline void hadamard_128_avx2(float* v)
{
    __m256 r[16];
    for (int i = 0; i < 16; ++i) r[i] = _mm256_loadu_ps(v + i * 8);

    // width 1: butterfly within adjacent pairs
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute_ps(r[i], 0b10110001);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b10101010);
    }

    // width 2: butterfly between 64-bit pairs
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute_ps(r[i], 0b01001110);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11001100);
    }

    // width 4: butterfly between 128-bit halves
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute2f128_ps(r[i], r[i], 0x01);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11110000);
    }

    // widths 8..64: whole-register butterflies
    for (int width = 1; width < 16; width *= 2)
        for (int base = 0; base < 16; base += 2 * width)
            for (int i = 0; i < width; ++i)
            {
                const __m256 a = r[base + i];
                const __m256 b = r[base + width + i];
                r[base + i] = _mm256_add_ps(a, b);
                r[base + width + i] = _mm256_sub_ps(a, b);
            }

    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(v + i * 8, r[i]);
}

} // namespace exl3_common
