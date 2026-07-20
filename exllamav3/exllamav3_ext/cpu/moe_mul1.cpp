#include "moe_mul1.h"
#include <c10/util/Half.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifndef __linux__
#include <intrin.h>
#endif

// CPU MoE expert GEMM for mul1 EXL3 tensors.
//
// The mul1 codebook is affine in a byte-sum: w(s) = (bytesum(s * 0x83DCD12D) - 510) * k_inv with
// k_inv = fp16(0x1eee). With int8 activations x8 and one scale q per input row:
//   sum_k w_kn x_k = k_inv * q * ( sum_k bytesum(s_kn * M) * x8_k  -  510 * sum_k x8_k )
// and bytesum(s*M) * x8 is exactly one AVX-512 VNNI vpdpbusd per 16 weights (unsigned operand =
// product bytes, signed operand = the int8 activation replicated x4; u8*s8 word products stay
// below 2^15, which makes the operand order load-bearing). Accuracy matches the GPU int8-GEMV
// mode-2 class (~0.9% per-call output RMS). i32 accumulators are safe for k up to ~8192.
//
// State extraction uses compile-time (bits, row) index tables for vpermt2var plus immediate
// funnel shifts, following benchmarks/exl3_cpu_gemm. The GEMV streams k-major with a contiguous
// band of output tiles held in register accumulators per worker, so cold expert weights are read
// near-sequentially from DRAM (measured 3.4x over per-output-column traversal on cold stacks).
//
// No generic lambdas inside target-attributed functions (GCC does not let lambdas inherit the
// target), hence the recursive-template row unrolling.

namespace { std::atomic<bool> g_prof_enabled { false }; }

void exl3_moe_cpu_set_prof(bool enabled)
{
    g_prof_enabled.store(enabled, std::memory_order_relaxed);
}

namespace {

constexpr uint32_t MUL1_MULT = 0x83DCD12Du;
constexpr float HAD_SCALE = 0.088388347648f;
constexpr int MAX_M = 4;

#if defined(__GNUC__) && defined(__linux__)
#define M1_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define M1_TARGET_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
#else
#define M1_TARGET_AVX2
#define M1_TARGET_VNNI
#endif

inline void cpu_pause()
{
#ifdef __linux__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}

inline float half_to_float(at::Half h) { return static_cast<float>(h); }

inline float mul1_k_inv()
{
    // fp16 0x1eee
    static const float v = half_to_float(c10::Half(uint16_t(0x1eee), c10::Half::from_bits()));
    return v;
}

// -------------------------------------------------------------------------------------------
//   Format tables
// -------------------------------------------------------------------------------------------

// Matches tensor-core permutation baked into by EXL3 tile storage format
constexpr std::array<uint16_t, 256> make_tc_perm()
{
    std::array<uint16_t, 256> p{};
    #pragma unroll
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
    const auto perm = make_tc_perm();
    for (int i = 0; i < 256; ++i) inv[perm[i]] = i;
    return inv;
}

template <int bits, int row, bool second_word>
constexpr std::array<int32_t, 16> make_row_indices()
{
    std::array<int32_t, 16> idx{};
    const auto inv = make_tc_perm_inv();
    constexpr int words32 = bits * 256 / 32;
    for (int col = 0; col < 16; ++col) {
        const int t = inv[row * 16 + col];
        const int b0 = t * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        idx[col] = (second_word ? (b1 - 1) / 32 : b0 / 32) % words32;
    }
    return idx;
}

template <int bits, int row, bool second_word>
constexpr uint16_t make_row_himask()
{
    uint16_t mask = 0;
    const auto idx = make_row_indices<bits, row, second_word>();
    for (int col = 0; col < 16; ++col)
        if (idx[col] >= 32) mask |= uint16_t(1) << col;
    return mask;
}

template <int bits, int row>
constexpr int row_shift(int col)
{
    const auto inv = make_tc_perm_inv();
    const int t = inv[row * 16 + col];
    const int b1 = t * bits + bits + 256 * bits;
    return ((b1 - 1) / 32 + 1) * 32 - b1;
}

inline uint32_t load_u32_(const uint16_t* ptr, int index)
{
    uint32_t v;
    std::memcpy(&v, ptr + index * 2, sizeof(v));
    return v;
}

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

inline float decode_mul1_scalar(uint16_t state)
{
    const uint32_t x = static_cast<uint32_t>(state) * MUL1_MULT;
    const int sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    return (static_cast<float>(sum) - 510.0f) * mul1_k_inv();
}

// -------------------------------------------------------------------------------------------
//   ISA dispatch
// -------------------------------------------------------------------------------------------

// Declared early: the transforms below select on it
enum class Isa { Scalar, Avx2, Vnni };
extern const Isa g_isa;

// -------------------------------------------------------------------------------------------
//   Transforms
// -------------------------------------------------------------------------------------------

void hadamard_128_scalar(float* v)
{
    #pragma unroll
    for (int width = 1; width < 128; width *= 2)
        #pragma unroll
        for (int base = 0; base < 128; base += 2 * width)
            #pragma unroll
            for (int i = 0; i < width; ++i) {
                const float a = v[base + i];
                const float b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
}

M1_TARGET_AVX2
void hadamard_128_avx2(float* v)
{
    __m256 r[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = _mm256_loadu_ps(v + i * 8);

    // width 1: butterfly within adjacent pairs
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute_ps(r[i], 0b10110001);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b10101010);
    }

    // width 2: butterfly between 64-bit pairs
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute_ps(r[i], 0b01001110);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11001100);
    }

    // width 4: butterfly between 128-bit halves
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute2f128_ps(r[i], r[i], 0x01);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11110000);
    }

    // widths 8..64: whole-register butterflies
    #pragma unroll
    for (int width = 1; width < 16; width *= 2)
        #pragma unroll
        for (int base = 0; base < 16; base += 2 * width)
            #pragma unroll
            for (int i = 0; i < width; ++i)
            {
                const __m256 a = r[base + i];
                const __m256 b = r[base + width + i];
                r[base + i] = _mm256_add_ps(a, b);
                r[base + width + i] = _mm256_sub_ps(a, b);
            }

    #pragma unroll
    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(v + i * 8, r[i]);
}

inline void hadamard_128(float* v)
{
    if (g_isa != Isa::Scalar) hadamard_128_avx2(v);
    else                      hadamard_128_scalar(v);
}

// -------------------------------------------------------------------------------------------
//   Prepared input (per GEMV, per chunk)
// -------------------------------------------------------------------------------------------

struct PreparedIn
{
    float* tin;         // m x k, transformed fp32 (scalar kernel)
    int32_t* splat32;   // m x k, int8 activation replicated x4
    float q[MAX_M];
    int32_t sum_x8[MAX_M];
};

M1_TARGET_AVX2
void quantize_row_avx2(const float* dst, int32_t* splat, int k, float& q_out, int32_t& s_out)
{
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < k; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(dst + i)));
    alignas(32) float mx[8];
    _mm256_store_ps(mx, vmax);
    float amax = 0.0f;
    for (int i = 0; i < 8; ++i) amax = std::max(amax, mx[i]);
    const float q = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m256 rq = _mm256_set1_ps(1.0f / q);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    const __m256i rep = _mm256_set1_epi32(0x01010101);
    const __m256i mask8 = _mm256_set1_epi32(0xff);
    __m256i vsum = _mm256_setzero_si256();
    #pragma unroll
    for (int i = 0; i < k; i += 8)
    {
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(dst + i), rq));
        v = _mm256_min_epi32(hi, _mm256_max_epi32(lo, v));
        vsum = _mm256_add_epi32(vsum, v);
        const __m256i b = _mm256_and_si256(v, mask8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(splat + i), _mm256_mullo_epi32(b, rep));
    }
    alignas(32) int32_t sm[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(sm), vsum);
    s_out = sm[0] + sm[1] + sm[2] + sm[3] + sm[4] + sm[5] + sm[6] + sm[7];
    q_out = q;
}

M1_TARGET_AVX2
void prepare_block_avx2(const void* srcv, bool f16, const at::Half* suh, float* dst, int k)
{
    const __m256 hs = _mm256_set1_ps(HAD_SCALE);
    for (int block = 0; block < k; block += 128)
    {
        #pragma unroll
        for (int i = 0; i < 128; i += 8)
        {
            __m256 x;
            if (f16)
                x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    static_cast<const uint16_t*>(srcv) + block + i)));
            else
                x = _mm256_loadu_ps(static_cast<const float*>(srcv) + block + i);
            const __m256 s = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                reinterpret_cast<const uint16_t*>(suh) + block + i)));
            _mm256_storeu_ps(dst + block + i, _mm256_mul_ps(x, s));
        }
        hadamard_128_avx2(dst + block);

        #pragma unroll
        for (int i = 0; i < 128; i += 8)
            _mm256_storeu_ps(dst + block + i,
                             _mm256_mul_ps(_mm256_loadu_ps(dst + block + i), hs));
    }
}

// src_f16 / src_f32: one of them non-null; rows gathered by token index
void prepare_rows
(
    const MoeCpuMatrix& mat,
    const at::Half* src_f16, const float* src_f32, int src_stride,
    const int* token_idx, int m,
    PreparedIn& p
)
{
    const int k = mat.k;
    for (int r = 0; r < m; ++r)
    {
        float* dst = p.tin + static_cast<size_t>(r) * k;
        const size_t src_off = static_cast<size_t>(token_idx[r]) * src_stride;
        if (g_isa != Isa::Scalar)
        {
            prepare_block_avx2(src_f16 ? reinterpret_cast<const void*>(src_f16 + src_off)
                                       : reinterpret_cast<const void*>(src_f32 + src_off),
                               src_f16 != nullptr, mat.suh, dst, k);
        }
        else
        {
            for (int block = 0; block < k; block += 128)
            {
                float vals[128];
                for (int i = 0; i < 128; ++i)
                {
                    const float xv = src_f16 ? half_to_float(src_f16[src_off + block + i])
                                             : src_f32[src_off + block + i];
                    vals[i] = xv * half_to_float(mat.suh[block + i]);
                }
                hadamard_128(vals);
                for (int i = 0; i < 128; ++i)
                    dst[block + i] = vals[i] * HAD_SCALE;
            }
        }

        // int8 quantization, one scale per row
        int32_t* splat = p.splat32 + static_cast<size_t>(r) * k;
        float q;
        int32_t s;
        if (g_isa != Isa::Scalar)
        {
            quantize_row_avx2(dst, splat, k, q, s);
        }
        else
        {
            float amax = 0.0f;
            for (int i = 0; i < k; ++i) amax = std::max(amax, std::fabs(dst[i]));
            q = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float rq = 1.0f / q;
            s = 0;
            for (int i = 0; i < k; ++i)
            {
                int v = static_cast<int>(std::lround(dst[i] * rq));
                v = std::clamp(v, -127, 127);
                s += v;
                splat[i] = static_cast<int32_t>(static_cast<uint8_t>(static_cast<int8_t>(v))) * 0x01010101;
            }
        }
        p.q[r] = q;
        p.sum_x8[r] = s;
    }
}

// -------------------------------------------------------------------------------------------
//   AVX-512 VNNI banded kernel
// -------------------------------------------------------------------------------------------

template <int bits, int row>
M1_TARGET_VNNI
inline __m512i extract_row(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    alignas(64) static constexpr auto i0d = make_row_indices<bits, row, false>();
    alignas(64) static constexpr auto i1d = make_row_indices<bits, row, true>();
    constexpr int s0 = row_shift<bits, row>(0);
    constexpr int s1 = row_shift<bits, row>(8);
    const __m512i i0 = _mm512_load_si512(i0d.data());
    const __m512i i1 = _mm512_load_si512(i1d.data());
    __m512i a = _mm512_permutex2var_epi32(p0, i0, p1);
    __m512i b = _mm512_permutex2var_epi32(p0, i1, p1);
    if constexpr (bits > 4)
    {
        // Up to 64 packed words: indices >= 32 select from the second register pair. vpermt2var
        // uses index bits [4:0], so the same index vectors address both pairs; constexpr masks
        // choose per lane
        constexpr __mmask16 hm0 = make_row_himask<bits, row, false>();
        constexpr __mmask16 hm1 = make_row_himask<bits, row, true>();
        if constexpr (hm0 != 0)
            a = _mm512_mask_blend_epi32(hm0, a, _mm512_permutex2var_epi32(p2, i0, p3));
        if constexpr (hm1 != 0)
            b = _mm512_mask_blend_epi32(hm1, b, _mm512_permutex2var_epi32(p2, i1, p3));
    }
    const __m512i c0 = _mm512_or_si512(_mm512_srli_epi32(b, s0), _mm512_slli_epi32(a, 32 - s0));
    const __m512i c1 = _mm512_or_si512(_mm512_srli_epi32(b, s1), _mm512_slli_epi32(a, 32 - s1));
    return _mm512_and_si512(_mm512_mask_blend_epi32(0xff00, c0, c1), _mm512_set1_epi32(0xffff));
}

template <int bits, int rows, int band, int R>
M1_TARGET_VNNI
inline void vnni_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (R < 16) {
        const __m512i code = extract_row<bits, R>(p0, p1, p2, p3);
        const __m512i prod = _mm512_mullo_epi32(code, _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT)));
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod,
                _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
        vnni_band_rows<bits, rows, band, R + 1>(p0, p1, p2, p3, b, splat, k, acc);
    }
}

template <int bits, int rows, int band>
M1_TARGET_VNNI
void vnni_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    constexpr auto ld_mask = [](int lo) -> __mmask16
    {
        const int n = words32 - lo;
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(0);
    constexpr __mmask16 mask1 = ld_mask(16);
    constexpr __mmask16 mask2 = ld_mask(32);
    constexpr __mmask16 mask3 = ld_mask(48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat = in.splat32 + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = packed_row + b * packed_size;
            _mm_prefetch(reinterpret_cast<const char*>(packed + row_stride), _MM_HINT_T1);
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            vnni_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + static_cast<size_t>(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <int bits, int rows>
M1_TARGET_VNNI
void vnni_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    // m = 1 supports band widths up to 16 (16 zmm accumulators). Measured on the 7960X: 16 is
    // not better than 8 for decode-shape jobs (medians 1.01 vs 0.98 ms, interleaved A/B) -- the
    // prefetcher already covers 8-tile bursts and the extra accumulators cost load-scheduling
    // registers -- so 8 is fixed (was a runtime switch via EXL3_MOE_CPU_BAND during that
    // investigation; no case remained for deviating from 8, so removed ahead of further
    // microoptimization work that wants less runtime branching in this path)
    constexpr int band_cap = 8;

    const int max_band = rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8);
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: vnni_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: vnni_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: vnni_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: vnni_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: vnni_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: vnni_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: vnni_band<bits, rows, 7>(mat, in, tout, n0); break;
            case 8: vnni_band<bits, rows, 8>(mat, in, tout, n0); break;
            default:
                if constexpr (rows == 1)
                {
                    switch (band)
                    {
                        case 9: vnni_band<bits, 1, 9>(mat, in, tout, n0); break;
                        case 10: vnni_band<bits, 1, 10>(mat, in, tout, n0); break;
                        case 11: vnni_band<bits, 1, 11>(mat, in, tout, n0); break;
                        case 12: vnni_band<bits, 1, 12>(mat, in, tout, n0); break;
                        case 13: vnni_band<bits, 1, 13>(mat, in, tout, n0); break;
                        case 14: vnni_band<bits, 1, 14>(mat, in, tout, n0); break;
                        case 15: vnni_band<bits, 1, 15>(mat, in, tout, n0); break;
                        default: vnni_band<bits, 1, 16>(mat, in, tout, n0); break;
                    }
                }
                break;
        }
        n0 += band;
    }
}

// -------------------------------------------------------------------------------------------
//   AVX2
// -------------------------------------------------------------------------------------------

// maddubs saturates its i16 pair sums (2 * 255 * 127 > 32767), so the product bytes are split
// even/odd; each masked pair then holds a single u8 x s8 product

// State decode: AVX2 has no cross-lane permute wider than one 256-bit (8-lane) register, unlike
// AVX-512's vpermt2var (16-wide, spanning 2 registers = 32 lanes). A row's 8-column half can draw
// from any of the `bits` registers a k-tile occupies (packed_size = 16*bits u16 = bits registers
// of 8 u32 each), so instead of the VNNI path's fixed two-register-pair span, this walks every
// candidate register and blends in only the ones that actually contribute for a given (bits,
// row, half) -- resolved entirely at compile time via row/half/word-selector being template
// parameters, exactly mirroring how VNNI's hi/lo masks are compile-time per row. Requires the
// row loop itself to be compile-time-unrolled (below), not the runtime loop the gather-based
// first cut of this used: a runtime row made the blend masks runtime values too, which needs a
// variable blend (or a gather) instead of a free compile-time-immediate blend, and benchmarking
// showed AVX2 gather is not a win on this hardware (a modest 1.7x over the scalar-decode
// baseline, vs the several-x this register-permute version gets).

template <int bits, int row, bool second_word, int half, int Reg>
constexpr uint8_t avx2_reg_mask()
{
    constexpr auto idx16 = make_row_indices<bits, row, second_word>();
    uint8_t mask = 0;
    for (int i = 0; i < 8; ++i)
        if (idx16[half * 8 + i] / 8 == Reg) mask |= uint8_t(1) << i;
    return mask;
}

template <int bits, int row, bool second_word, int half>
constexpr std::array<int32_t, 8> avx2_lane_idx()
{
    constexpr auto idx16 = make_row_indices<bits, row, second_word>();
    std::array<int32_t, 8> out{};
    for (int i = 0; i < 8; ++i) out[i] = idx16[half * 8 + i] % 8;
    return out;
}

// Permutes+blends together only the registers that actually contribute a lane to this half, in
// increasing Reg order (skipped candidates cost nothing -- if constexpr eliminates them, so low
// bitrates collapse to a single unconditional permute, same as VNNI's cheapest case)
template <int bits, int row, bool second_word, int half, int Reg = 0>
M1_TARGET_AVX2
inline __m256i avx2_gather_half(const __m256i (&preg)[bits])
{
    // All-compile-time-constant arguments: the compiler folds this to a single constant load,
    // same as a hand-written lookup table. No lambda (this function is target-attributed, and
    // GCC does not propagate the target to a lambda's closure -- see file header note).
    constexpr auto li = avx2_lane_idx<bits, row, second_word, half>();
    const __m256i lane_idx_v = _mm256_setr_epi32(li[0], li[1], li[2], li[3], li[4], li[5], li[6], li[7]);
    if constexpr (Reg + 1 >= bits)
    {
        // last candidate: every column not already claimed must come from here
        return _mm256_permutevar8x32_epi32(preg[Reg], lane_idx_v);
    }
    else
    {
        constexpr uint8_t mask = avx2_reg_mask<bits, row, second_word, half, Reg>();
        if constexpr (mask == 0)
        {
            return avx2_gather_half<bits, row, second_word, half, Reg + 1>(preg);
        }
        else
        {
            const __m256i cur = _mm256_permutevar8x32_epi32(preg[Reg], lane_idx_v);
            const __m256i rest = avx2_gather_half<bits, row, second_word, half, Reg + 1>(preg);
            return _mm256_blend_epi32(rest, cur, mask);
        }
    }
}

// Decodes row `row`'s 16 states into two 8-wide registers (cols 0-7, cols 8-15) from the k-tile's
// pre-loaded registers; no scalar decode_state_scalar calls. b0/b1 and the funnel shift follow
// the same bit layout as decode_state_scalar; the shift is shared across each 8-column half (by
// construction of the tile's tensor-core permutation, the same invariant the VNNI path relies on
// for its single per-half s0/s1).
template <int bits, int row>
M1_TARGET_AVX2
inline void avx2_row_codes(const __m256i (&preg)[bits], __m256i& codes_lo, __m256i& codes_hi)
{
    const __m256i a_lo = avx2_gather_half<bits, row, false, 0>(preg);
    const __m256i b_lo = avx2_gather_half<bits, row, true, 0>(preg);
    const __m256i a_hi = avx2_gather_half<bits, row, false, 1>(preg);
    const __m256i b_hi = avx2_gather_half<bits, row, true, 1>(preg);
    constexpr int s0 = row_shift<bits, row>(0);
    constexpr int s1 = row_shift<bits, row>(8);
    const __m256i mask16 = _mm256_set1_epi32(0xffff);
    codes_lo = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_lo, s0), _mm256_slli_epi32(a_lo, 32 - s0)), mask16);
    codes_hi = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_hi, s1), _mm256_slli_epi32(a_hi, 32 - s1)), mask16);
}

template <int bits, int row = 0>
M1_TARGET_AVX2
inline void avx2_rows_accum(
    const __m256i (&preg)[bits], const int32_t* splat, int k, int m, __m256i (&acc)[MAX_M][2],
    const __m256i& mult, const __m256i& ones16, const __m256i& even_mask)
{
    if constexpr (row < 16)
    {
        __m256i codes_lo, codes_hi;
        avx2_row_codes<bits, row>(preg, codes_lo, codes_hi);

        const __m256i prod0 = _mm256_mullo_epi32(codes_lo, mult);
        const __m256i prod1 = _mm256_mullo_epi32(codes_hi, mult);
        const __m256i p0e = _mm256_and_si256(prod0, even_mask);
        const __m256i p0o = _mm256_andnot_si256(even_mask, prod0);
        const __m256i p1e = _mm256_and_si256(prod1, even_mask);
        const __m256i p1o = _mm256_andnot_si256(even_mask, prod1);

        for (int i = 0; i < m; ++i)
        {
            const __m256i xs = _mm256_set1_epi32(splat[static_cast<size_t>(i) * k + row]);
            acc[i][0] = _mm256_add_epi32(acc[i][0], _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_maddubs_epi16(p0e, xs), ones16),
                _mm256_madd_epi16(_mm256_maddubs_epi16(p0o, xs), ones16)));
            acc[i][1] = _mm256_add_epi32(acc[i][1], _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_maddubs_epi16(p1e, xs), ones16),
                _mm256_madd_epi16(_mm256_maddubs_epi16(p1o, xs), ones16)));
        }
        avx2_rows_accum<bits, row + 1>(preg, splat, k, m, acc, mult, ones16, even_mask);
    }
}

template <int bits>
M1_TARGET_AVX2
void avx2_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    const __m256i mult = _mm256_set1_epi32(static_cast<int32_t>(MUL1_MULT));
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i even_mask = _mm256_set1_epi16(0x00ff);

    for (int tile_n = tn0; tile_n < tn1; ++tile_n)
    {
        __m256i acc[MAX_M][2];
        for (int i = 0; i < m; ++i)
        {
            acc[i][0] = _mm256_setzero_si256();
            acc[i][1] = _mm256_setzero_si256();
        }

        const uint16_t* packed = mat.trellis + static_cast<size_t>(tile_n) * packed_size;
        const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed += row_stride)
        {
            const int32_t* splat = in.splat32 + tile_k * 16;
            // One 256-bit (8xu32) register per bits: covers packed_size = 16*bits u16 = bits*8
            // u32 words exactly, the whole k-tile's row of packed states
            __m256i preg[bits];
            for (int i = 0; i < bits; ++i)
                preg[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i * 16));
            avx2_rows_accum<bits>(preg, splat, mat.k, m, acc, mult, ones16, even_mask);
        }

        for (int i = 0; i < m; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m256 corr = _mm256_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            float* out = tout + static_cast<size_t>(i) * mat.n + tile_n * 16;
            _mm256_storeu_ps(out, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][0]), _mm256_set1_ps(scale), corr));
            _mm256_storeu_ps(out + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][1]), _mm256_set1_ps(scale), corr));
        }
    }
}

// -------------------------------------------------------------------------------------------
//   Scalar fallback
// -------------------------------------------------------------------------------------------

template <int bits>
void scalar_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr auto perm = make_tc_perm();

    for (int tile_n = tn0; tile_n < tn1; ++tile_n)
    {
        float acc[MAX_M][16] = {};
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k)
        {
            const uint16_t* packed = mat.trellis + (static_cast<size_t>(tile_k) * tiles_n + tile_n) * packed_size;
            float tile[256];

            for (int t = 0; t < 256; ++t)
                tile[perm[t]] = decode_mul1_scalar(decode_state_scalar<bits>(packed, t));

            for (int i = 0; i < m; ++i)
            {
                const float* x = in.tin + static_cast<size_t>(i) * mat.k + tile_k * 16;
                for (int r = 0; r < 16; ++r)
                    for (int c = 0; c < 16; ++c)
                        acc[i][c] += x[r] * tile[r * 16 + c];
            }
        }
        for (int i = 0; i < m; ++i)
            std::memcpy(tout + static_cast<size_t>(i) * mat.n + tile_n * 16, acc[i], 16 * sizeof(float));
    }
}

// -------------------------------------------------------------------------------------------
//   Dispatch
// -------------------------------------------------------------------------------------------

Isa detect_isa()
{
    Isa hw;
#if defined(__GNUC__) && defined(__linux__)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni"))
        hw = Isa::Vnni;
    else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        hw = Isa::Avx2;
    else
        hw = Isa::Scalar;
#else
    int info[4];
    __cpuidex(info, 7, 0);
    const bool avx512 = (info[1] & (1 << 16)) && (info[1] & (1 << 30)) && (info[1] & (1 << 31));
    const bool vnni = (info[2] & (1 << 11)) != 0;
    const bool avx2 = (info[1] & (1 << 5)) != 0;
    hw = (avx512 && vnni) ? Isa::Vnni : (avx2 ? Isa::Avx2 : Isa::Scalar);
#endif

    // EXL3_MOE_CPU_MAX_ISA=scalar|avx2|vnni: cap detection at a lower tier for testing (e.g.
    // exercising the AVX2 path on AVX512-VNNI hardware). Never upgrades past what the CPU
    // actually supports; an unrecognized value is ignored.
    if (const char* e = std::getenv("EXL3_MOE_CPU_MAX_ISA"))
    {
        std::string s(e);
        for (char& c : s) c = (char) std::tolower((unsigned char) c);
        Isa cap;
        if (s == "scalar") cap = Isa::Scalar;
        else if (s == "avx2") cap = Isa::Avx2;
        else if (s == "vnni" || s == "avx512") cap = Isa::Vnni;
        else return hw;
        if (cap < hw) hw = cap;
    }
    return hw;
}

const Isa g_isa = []{ return detect_isa(); }();

void run_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    switch (g_isa) {
        case Isa::Vnni:
        {
            switch (mat.bits * 4 + m - 1)
            {
                case 1 * 4 + 0: vnni_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
                case 1 * 4 + 1: vnni_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
                case 1 * 4 + 2: vnni_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
                case 1 * 4 + 3: vnni_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 0: vnni_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 1: vnni_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 2: vnni_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 3: vnni_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 0: vnni_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 1: vnni_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 2: vnni_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 3: vnni_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 0: vnni_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 1: vnni_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 2: vnni_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 3: vnni_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 0: vnni_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 1: vnni_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 2: vnni_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 3: vnni_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 0: vnni_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 1: vnni_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 2: vnni_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 3: vnni_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 0: vnni_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 1: vnni_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 2: vnni_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 3: vnni_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
                case 8 * 4 + 0: vnni_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
                case 8 * 4 + 1: vnni_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
                case 8 * 4 + 2: vnni_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
                case 8 * 4 + 3: vnni_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
            }
            return;
        }
        case Isa::Avx2:
        {
            switch (mat.bits)
            {
                case 1: avx2_tiles<1>(mat, in, tout, m, tn0, tn1); return;
                case 2: avx2_tiles<2>(mat, in, tout, m, tn0, tn1); return;
                case 3: avx2_tiles<3>(mat, in, tout, m, tn0, tn1); return;
                case 4: avx2_tiles<4>(mat, in, tout, m, tn0, tn1); return;
                case 5: avx2_tiles<5>(mat, in, tout, m, tn0, tn1); return;
                case 6: avx2_tiles<6>(mat, in, tout, m, tn0, tn1); return;
                case 7: avx2_tiles<7>(mat, in, tout, m, tn0, tn1); return;
                default: avx2_tiles<8>(mat, in, tout, m, tn0, tn1); return;
            }
        }
        case Isa::Scalar:
        {
            switch (mat.bits)
            {
                case 1: scalar_tiles<1>(mat, in, tout, m, tn0, tn1); return;
                case 2: scalar_tiles<2>(mat, in, tout, m, tn0, tn1); return;
                case 3: scalar_tiles<3>(mat, in, tout, m, tn0, tn1); return;
                case 4: scalar_tiles<4>(mat, in, tout, m, tn0, tn1); return;
                case 5: scalar_tiles<5>(mat, in, tout, m, tn0, tn1); return;
                case 6: scalar_tiles<6>(mat, in, tout, m, tn0, tn1); return;
                case 7: scalar_tiles<7>(mat, in, tout, m, tn0, tn1); return;
                default: scalar_tiles<8>(mat, in, tout, m, tn0, tn1); return;
            }
        }
    }
}

// -------------------------------------------------------------------------------------------
//   Thread pool (persistent, spin-parked; master participates as worker 0)
// -------------------------------------------------------------------------------------------

typedef void (*PoolFn)(void* ctx, int worker, int num_workers);

struct Pool
{
    int spawned = 0;
    std::atomic<uint64_t> gen{0};
    std::atomic<uint64_t> done{0};
    std::atomic<PoolFn> fn{nullptr};
    void* ctx = nullptr;
    int num_workers = 1;

    void worker_loop(int idx)
    {
        uint64_t seen = 0;
        int idle = 0;
        while (true) {
            const uint64_t g = gen.load(std::memory_order_acquire);
            if (g == seen)
            {
                if (++idle < 8192) { cpu_pause(); continue; }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            idle = 0;
            seen = g;

            // num_workers may have shrunk below this worker's index: surplus workers must not
            // run the function (their (idx, num_workers) pair indexes out of range) and must not
            // ack, or run() returns before the participating workers have finished
            const int nw = num_workers;
            if (idx < nw)
            {
                fn.load(std::memory_order_relaxed)(ctx, idx, nw);
                done.fetch_add(1, std::memory_order_release);
            }
        }
    }

    void ensure(int n)
    {
        while (spawned < n - 1)
        {
            std::thread(&Pool::worker_loop, this, spawned + 1).detach();
            ++spawned;
        }
        num_workers = n;
    }

    // Run fn on workers 0..n-1; returns when all are done (implicit barrier)
    void run(PoolFn f, void* c)
    {
        const int n = num_workers;
        if (n <= 1) { f(c, 0, 1); return; }
        ctx = c;
        fn.store(f, std::memory_order_relaxed);
        const uint64_t d0 = done.load(std::memory_order_acquire);
        gen.fetch_add(1, std::memory_order_release);
        f(c, 0, n);
        while (static_cast<int64_t>(done.load(std::memory_order_acquire) - d0) < n - 1)
            cpu_pause();
    }
};

Pool g_pool;
std::mutex g_pool_mutex;

// -------------------------------------------------------------------------------------------
//   MoE Layer registry
// -------------------------------------------------------------------------------------------

std::vector<MoeCpuLayer*> g_layers;
std::mutex g_layers_mutex;

// -------------------------------------------------------------------------------------------
//   Forward driver
// -------------------------------------------------------------------------------------------

struct Chunk
{
    int expert;
    int m;
    int token[MAX_M];
    float weight[MAX_M];
};

struct ForwardCtx
{
    const MoeCpuLayer* layer;
    const at::Half* x;
    float* out;
    int m_total;
    std::vector<Chunk> chunks;

    // workspace, per chunk (pointers into the persistent per-thread arena below: fresh
    // allocations per call cost more in first-touch page faults than the small phases do work)
    float* tout_g;       // chunks x m x I (quant space, then transformed in place)
    float* tout_u;
    float* tout_d;       // chunks x m x H
    std::vector<PreparedIn> prep_g, prep_u, prep_d;

    int phase = 0;
};

struct ForwardArena
{
    std::vector<float> tin_g, tin_u, tin_d;
    std::vector<int32_t> splat_g, splat_u, splat_d;
    std::vector<float> tout_g, tout_u, tout_d;
    std::vector<PreparedIn> prep_g, prep_u, prep_d;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

M1_TARGET_AVX2
void transform_out_avx2(const MoeCpuMatrix& mat, float* tout, int m)
{
    const __m256 hs = _mm256_set1_ps(HAD_SCALE);
    for (int r = 0; r < m; ++r)
        for (int block = 0; block < mat.n; block += 128)
        {
            float* v = tout + static_cast<size_t>(r) * mat.n + block;
            hadamard_128_avx2(v);
            for (int i = 0; i < 128; i += 8)
            {
                const __m256 s = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    reinterpret_cast<const uint16_t*>(mat.svh) + block + i)));
                __m256 x = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i), hs), s);
                if (mat.bias)
                    x = _mm256_add_ps(x, _mm256_cvtph_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(
                            reinterpret_cast<const uint16_t*>(mat.bias) + block + i))));
                _mm256_storeu_ps(v + i, x);
            }
        }
}

void transform_out(const MoeCpuMatrix& mat, float* tout, int m)
{
    if (g_isa != Isa::Scalar) { transform_out_avx2(mat, tout, m); return; }

    for (int r = 0; r < m; ++r)
        for (int block = 0; block < mat.n; block += 128)
        {
            float* v = tout + static_cast<size_t>(r) * mat.n + block;
            hadamard_128(v);
            if (mat.bias)
                for (int i = 0; i < 128; ++i)
                    v[i] = v[i] * HAD_SCALE * half_to_float(mat.svh[block + i])
                           + half_to_float(mat.bias[block + i]);
            else
                for (int i = 0; i < 128; ++i)
                    v[i] *= HAD_SCALE * half_to_float(mat.svh[block + i]);
        }
}


// Assign workers to GEMVs: with more GEMVs than workers, each worker strides over whole GEMVs;
// otherwise GEMV j gets the contiguous worker group [j*nw/total, (j+1)*nw/total). Returns false
// when this worker has no assignment. Missing either regime silently drops GEMVs.
inline bool gemv_assignment(int worker, int num_workers, int total, int& j0, int& j_step, int& sub, int& per)
{
    if (total <= 0) return false;
    if (total >= num_workers)
    {
        j0 = worker; j_step = num_workers; sub = 0; per = 1;
        return j0 < total;
    }

    for (int j = 0; j < total; ++j)
    {
        const int w0 = j * num_workers / total;
        const int w1 = (j + 1) * num_workers / total;
        if (worker >= w0 && worker < w1) {
            j0 = j; j_step = total; sub = worker - w0; per = w1 - w0;
            return true;
        }
    }
    return false;
}

void forward_phase(void* vctx, int worker, int num_workers)
{
    ForwardCtx& c = *static_cast<ForwardCtx*>(vctx);
    const MoeCpuLayer& L = *c.layer;
    const int nc = static_cast<int>(c.chunks.size());
    const int H = L.hidden_size;
    const int I = L.interm_size;

    switch (c.phase) {

        case 0:
        {
            // Prepare gate and up inputs, distributed over (chunk, gate/up)
            const int gu = L.gates.empty() ? 1 : 2;
            for (int j = worker; j < nc * gu; j += num_workers)
            {
                const Chunk& ch = c.chunks[j / gu];
                const bool up = gu == 1 || (j % gu);
                const MoeCpuMatrix& mat = up ? L.ups[ch.expert] : L.gates[ch.expert];
                PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                prepare_rows(mat, c.x, nullptr, H, ch.token, ch.m, p);
            }
            break;
        }

        case 1:
        {
            // Gate + up GEMVs: workers spread over the GEMVs, contiguous tile ranges within each
            const int gu = L.gates.empty() ? 1 : 2;
            const int total = nc * gu;
            int j0, j_step, sub, per;
            if (gemv_assignment(worker, num_workers, total, j0, j_step, sub, per))
                for (int j = j0; j < total; j += j_step)
                {
                    const Chunk& ch = c.chunks[j / gu];
                    const bool up = gu == 1 || (j % gu);
                    const MoeCpuMatrix& mat = up ? L.ups[ch.expert] : L.gates[ch.expert];
                    const PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                    float* tout = (up ? c.tout_u : c.tout_g) + static_cast<size_t>(j / gu) * MAX_M * I;
                    const int tiles_n = mat.n / 16;
                    const int t0 = tiles_n * sub / per;
                    const int t1 = tiles_n * (sub + 1) / per;
                    run_tiles(mat, p, tout, ch.m, t0, t1);
                }
            break;
        }

        case 2:
        {
            // Output transform for gate/up, activation, prepare down input; per chunk. Gated: act(g)
            // * u accumulated into g; gateless: relu2 applied to u in place
            const bool gated = !L.gates.empty();
            for (int j = worker; j < nc; j += num_workers) {
                const Chunk& ch = c.chunks[j];
                float* g = c.tout_g + static_cast<size_t>(j) * MAX_M * I;
                float* u = c.tout_u + static_cast<size_t>(j) * MAX_M * I;
                if (gated) transform_out(L.gates[ch.expert], g, ch.m);
                transform_out(L.ups[ch.expert], u, ch.m);
                const size_t count = static_cast<size_t>(ch.m) * I;
                float* a = gated ? g : u;
                switch (L.activation) {
                    case 0:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            g[i] = gv / (1.0f + std::exp(-gv)) * u[i];
                        }
                        break;
                    case 1:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float cdf = 0.5f * (1.0f + std::erf(gv * 0.70710678f));
                            g[i] = gv * cdf * u[i];
                        }
                        break;
                    case 3: {
                        // gpt-oss clamped swiglu: g = min(g, limit); a = (clamp(u, -l, l) + 1) * g *
                        // sigmoid(1.702 * g)
                        const float lim = L.act_limit;
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = std::min(g[i], lim);
                            const float uv = std::clamp(u[i], -lim, lim);
                            g[i] = (uv + 1.0f) * gv / (1.0f + std::exp(-1.702f * gv));
                        }
                        break;
                    }
                    default:
                        for (size_t i = 0; i < count; ++i) {
                            const float uv = u[i] > 0.0f ? u[i] : 0.0f;
                            u[i] = uv * uv;
                        }
                        break;
                }
                static const int idx4[MAX_M] = {0, 1, 2, 3};
                prepare_rows(L.downs[ch.expert], nullptr, a, I, idx4, ch.m, c.prep_d[j]);
            }
            break;
        }

        case 3:
        {
            // Down GEMVs
            int j0, j_step, sub, per;
            if (gemv_assignment(worker, num_workers, nc, j0, j_step, sub, per))
                for (int j = j0; j < nc; j += j_step) {
                    const Chunk& ch = c.chunks[j];
                    const MoeCpuMatrix& mat = L.downs[ch.expert];
                    float* tout = c.tout_d + static_cast<size_t>(j) * MAX_M * H;
                    const int tiles_n = mat.n / 16;
                    const int t0 = tiles_n * sub / per;
                    const int t1 = tiles_n * (sub + 1) / per;
                    run_tiles(mat, c.prep_d[j], tout, ch.m, t0, t1);
                }
            break;
        }

        case 4:
        {
            // Down output transform (per chunk), then weighted accumulate into out, partitioned
            // over hidden columns so overlapping token rows are race-free
            for (int j = worker; j < nc; j += num_workers) {
                const Chunk& ch = c.chunks[j];
                transform_out(L.downs[ch.expert], c.tout_d + static_cast<size_t>(j) * MAX_M * H, ch.m);
            }
            break;
        }

        case 5:
        {
            const int c0 = H * worker / num_workers;
            const int c1 = H * (worker + 1) / num_workers;
            for (int j = 0; j < nc; ++j) {
                const Chunk& ch = c.chunks[j];
                const float* d = c.tout_d + static_cast<size_t>(j) * MAX_M * H;
                for (int r = 0; r < ch.m; ++r) {
                    float* dst = c.out + static_cast<size_t>(ch.token[r]) * H;
                    const float* src = d + static_cast<size_t>(r) * H;
                    const float w = ch.weight[r];
                    for (int col = c0; col < c1; ++col)
                        dst[col] += w * src[col];
                }
            }
            break;
        }
    }
}

} // namespace

static const MoeCpuLayer* get_layer(int64_t handle);

namespace {

struct StageCtx
{
    const MoeCpuLayer* layer;
    const uint32_t* ids;
    int count;
    uint8_t* dst;
};

inline size_t trellis_bytes(const MoeCpuMatrix& m)
{
    return static_cast<size_t>(m.k / 16) * (m.n / 16) * 16 * m.bits * 2;
}

void stage_phase(void* vctx, int worker, int num_workers)
{
    StageCtx& c = *static_cast<StageCtx*>(vctx);
    const bool gated = !c.layer->gates.empty();
    const int nmat = gated ? 3 : 2;
    // Each (expert, matrix) is one unit; offsets accumulate expert-major in g, u, d order
    const size_t gb = gated ? trellis_bytes(c.layer->gates[0]) : 0;
    const size_t ub = trellis_bytes(c.layer->ups[0]);
    const size_t db = trellis_bytes(c.layer->downs[0]);
    const size_t per_expert = gb + ub + db;
    for (int u = worker; u < c.count * nmat; u += num_workers)
    {
        const int e = c.ids[u / nmat];
        const int mi = u % nmat;
        size_t off = static_cast<size_t>(u / nmat) * per_expert;
        const uint16_t* srcp;
        size_t bytes;
        if (gated && mi == 0)      { srcp = c.layer->gates[e].trellis; bytes = gb; }
        else if (mi == (gated ? 1 : 0)) { srcp = c.layer->ups[e].trellis; bytes = ub; off += gb; }
        else                       { srcp = c.layer->downs[e].trellis; bytes = db; off += gb + ub; }
        std::memcpy(c.dst + off, srcp, bytes);
    }
}

} // namespace

void exl3_moe_cpu_stage_experts
(
    int64_t handle,
    const uint32_t* expert_ids,
    int count,
    uint8_t* dst,
    int threads
)
{
    // Runs on the worker's stager thread, concurrently with compute jobs on the pool: use
    // scratch threads, never the shared pool. A few threads saturate memcpy DRAM bandwidth.
    StageCtx ctx { get_layer(handle), expert_ids, count, dst };
    const bool gated = !ctx.layer->gates.empty();
    int units = count * (gated ? 3 : 2);
    int nt = std::min(threads > 0 ? threads : 1, units);
    if (nt <= 1)
    {
        stage_phase(&ctx, 0, 1);
        return;
    }
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (int i = 0; i < nt; ++i)
        ts.emplace_back(stage_phase, &ctx, i, nt);
    for (auto& t : ts)
        t.join();
}

bool exl3_moe_cpu_has_avx2() { return g_isa != Isa::Scalar; }
bool exl3_moe_cpu_has_avx512_vnni() { return g_isa == Isa::Vnni; }

static MoeCpuMatrix make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias
)
{
    TORCH_CHECK(trellis.device().is_cpu() && trellis.is_contiguous(), "trellis must be contiguous CPU");
    TORCH_CHECK(trellis.dim() == 3, "trellis must be [k/16, n/16, 16K]");
    MoeCpuMatrix m;
    m.trellis = reinterpret_cast<const uint16_t*>(trellis.data_ptr());
    m.suh = reinterpret_cast<const at::Half*>(suh.data_ptr());
    m.svh = reinterpret_cast<const at::Half*>(svh.data_ptr());
    m.bias = bias ? reinterpret_cast<const at::Half*>(bias->data_ptr()) : nullptr;
    m.k = static_cast<int>(trellis.size(0)) * 16;
    m.n = static_cast<int>(trellis.size(1)) * 16;
    m.bits = static_cast<int>(trellis.size(2)) / 16;
    TORCH_CHECK(m.bits >= 1 && m.bits <= 8, "CPU MoE requires K in [1, 8]");
    TORCH_CHECK(m.k % 128 == 0 && m.n % 128 == 0, "dims must be divisible by 128");
    TORCH_CHECK(m.k <= 8192, "k too large for i32 accumulation");
    return m;
}

int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit
)
{
    auto* layer = new MoeCpuLayer;
    const size_t E = up_trellis.size();
    const bool gated = !gate_trellis.empty();
    TORCH_CHECK(down_trellis.size() == E && (!gated || gate_trellis.size() == E), "expert count mismatch");
    TORCH_CHECK(gated ? (activation == 0 || activation == 1 || activation == 3) : activation == 2, "gated experts take silu/gelu/swiglu_oai, gateless take relu2");
    TORCH_CHECK(gate_bias.empty() || gate_bias.size() == E, "gate bias count mismatch");
    TORCH_CHECK(up_bias.empty() || up_bias.size() == E, "up bias count mismatch");
    TORCH_CHECK(down_bias.empty() || down_bias.size() == E, "down bias count mismatch");
    layer->num_experts = static_cast<int>(E);
    layer->activation = static_cast<int>(activation);
    layer->act_limit = static_cast<float>(act_limit);
    for (size_t e = 0; e < E; ++e) {
        if (gated) {
            layer->gates.push_back(make_matrix(gate_trellis[e], gate_suh[e], gate_svh[e],
                                               gate_bias.empty() ? nullptr : &gate_bias[e]));
            for (auto& t : {gate_trellis[e], gate_suh[e], gate_svh[e]})
                layer->refs.push_back(t);
            if (!gate_bias.empty()) layer->refs.push_back(gate_bias[e]);
        }
        layer->ups.push_back(make_matrix(up_trellis[e], up_suh[e], up_svh[e],
                                         up_bias.empty() ? nullptr : &up_bias[e]));
        layer->downs.push_back(make_matrix(down_trellis[e], down_suh[e], down_svh[e],
                                           down_bias.empty() ? nullptr : &down_bias[e]));
        for (auto& t : {up_trellis[e], up_suh[e], up_svh[e], down_trellis[e], down_suh[e], down_svh[e]})
            layer->refs.push_back(t);
        if (!up_bias.empty()) layer->refs.push_back(up_bias[e]);
        if (!down_bias.empty()) layer->refs.push_back(down_bias[e]);
    }
    layer->hidden_size = layer->ups[0].k;
    layer->interm_size = layer->ups[0].n;
    TORCH_CHECK(layer->downs[0].k == layer->interm_size && layer->downs[0].n == layer->hidden_size,
                "expert shape mismatch");

    std::lock_guard<std::mutex> lock(g_layers_mutex);
    g_layers.push_back(layer);
    return static_cast<int64_t>(g_layers.size() - 1);
}

void exl3_moe_cpu_free_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    if (handle >= 0 && handle < static_cast<int64_t>(g_layers.size()))
    {
        delete g_layers[handle];
        g_layers[handle] = nullptr;
    }
}

static const MoeCpuLayer* get_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(g_layers.size()) && g_layers[handle], "invalid CPU MoE layer handle");
    return g_layers[handle];
}

void exl3_moe_cpu_forward_raw(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* wts,
    float* out,
    int rows,
    int topk,
    int threads
)
{
    const MoeCpuLayer* layer = get_layer(handle);
    const int m_total = rows;
    const int top_k = topk;

    ForwardCtx ctx;
    ctx.layer = layer;
    ctx.x = x;
    ctx.out = out;
    ctx.m_total = m_total;
    std::memset(ctx.out, 0, static_cast<size_t>(m_total) * layer->hidden_size * sizeof(float));

    // Group token assignments by expert, then split into chunks of MAX_M rows
    std::vector<std::vector<std::pair<int, float>>> per_expert(layer->num_experts);
    for (int t = 0; t < m_total; ++t)
        for (int j = 0; j < top_k; ++j)
        {
            const int32_t e = sel[static_cast<size_t>(t) * top_k + j];
            if (e >= 0 && e < layer->num_experts)
                per_expert[e].emplace_back(t, half_to_float(wts[static_cast<size_t>(t) * top_k + j]));
        }
    for (int e = 0; e < layer->num_experts; ++e)
    {
        auto& lst = per_expert[e];
        for (size_t i = 0; i < lst.size(); i += MAX_M)
        {
            Chunk ch;
            ch.expert = e;
            ch.m = static_cast<int>(std::min<size_t>(MAX_M, lst.size() - i));
            for (int r = 0; r < ch.m; ++r)
            {
                ch.token[r] = lst[i + r].first;
                ch.weight[r] = lst[i + r].second;
            }
            ctx.chunks.push_back(ch);
        }
    }
    const int nc = static_cast<int>(ctx.chunks.size());
    if (!nc) return;

    // Workspace: persistent per-thread arena, grown but never shrunk
    const int H = layer->hidden_size;
    const int I = layer->interm_size;
    ForwardArena& ar = ForwardArena::get();
    auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(ar.tin_g, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.tin_u, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.tin_d, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.splat_g, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.splat_u, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.splat_d, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.tout_g, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.tout_u, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.tout_d, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.prep_g, nc); grow(ar.prep_u, nc); grow(ar.prep_d, nc);
    ctx.tout_g = ar.tout_g.data();
    ctx.tout_u = ar.tout_u.data();
    ctx.tout_d = ar.tout_d.data();
    ctx.prep_g = ar.prep_g; ctx.prep_u = ar.prep_u; ctx.prep_d = ar.prep_d;
    for (int j = 0; j < nc; ++j)
    {
        ctx.prep_g[j] = { ar.tin_g.data() + static_cast<size_t>(j) * MAX_M * H,
                          ar.splat_g.data() + static_cast<size_t>(j) * MAX_M * H, {}, {} };
        ctx.prep_u[j] = { ar.tin_u.data() + static_cast<size_t>(j) * MAX_M * H,
                          ar.splat_u.data() + static_cast<size_t>(j) * MAX_M * H, {}, {} };
        ctx.prep_d[j] = { ar.tin_d.data() + static_cast<size_t>(j) * MAX_M * I,
                          ar.splat_d.data() + static_cast<size_t>(j) * MAX_M * I, {}, {} };
    }

    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.ensure(threads > 0 ? threads : 1);

    // Per-phase wall time, reported every 512 jobs; enabled once at worker startup via
    // exl3_moe_cpu_set_prof (MoeCpuTuning.cpu_prof in moe_cpu_host.py, EXL3_MOE_CPU_PROF env)
    const bool prof = g_prof_enabled.load(std::memory_order_relaxed);
    static double phase_us[6] = {};
    static long prof_jobs = 0;

    for (int phase = 0; phase <= 5; ++phase) {
        ctx.phase = phase;
        if (prof)
        {
            const auto t0 = std::chrono::steady_clock::now();
            g_pool.run(&forward_phase, &ctx);
            phase_us[phase] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
        else
        {
            g_pool.run(&forward_phase, &ctx);
        }
    }
    if (prof && ++prof_jobs % 512 == 0)
    {
        printf(" -- moe_cpu prof (%ld jobs, us/job): prep_gu %.1f | gemv_gu %.1f | act+prep_d %.1f"
               " | gemv_d %.1f | tf_d %.1f | accum %.1f\n",
               prof_jobs, phase_us[0] / prof_jobs, phase_us[1] / prof_jobs, phase_us[2] / prof_jobs,
               phase_us[3] / prof_jobs, phase_us[4] / prof_jobs, phase_us[5] / prof_jobs);
        fflush(stdout);
    }
}

void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
)
{
    TORCH_CHECK(x.device().is_cpu() && selected.device().is_cpu() && weights.device().is_cpu() && out.device().is_cpu(), "CPU MoE tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kFloat, "dtype mismatch");

    const int m_total = static_cast<int>(x.size(0));
    const int top_k = static_cast<int>(selected.size(-1));

    // Raw path takes int32 selection
    std::vector<int32_t> sel32(static_cast<size_t>(m_total) * top_k);
    if (selected.scalar_type() == at::kLong)
    {
        const int64_t* s = selected.data_ptr<int64_t>();
        for (size_t i = 0; i < sel32.size(); ++i) sel32[i] = static_cast<int32_t>(s[i]);
    }
    else
    {
        TORCH_CHECK(selected.scalar_type() == at::kInt, "selected must be int32 or int64");
        std::memcpy(sel32.data(), selected.data_ptr<int32_t>(), sel32.size() * 4);
    }

    exl3_moe_cpu_forward_raw
    (
        handle,
        reinterpret_cast<const at::Half*>(x.data_ptr()),
        sel32.data(),
        reinterpret_cast<const at::Half*>(weights.data_ptr()),
        out.data_ptr<float>(),
        m_total, top_k,
        static_cast<int>(num_threads)
    );
}
