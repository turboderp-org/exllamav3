#include "moe_mul1.h"
#include <c10/util/Half.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <chrono>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#else
// min/max macro suppression is handled globally (-DNOMINMAX in setup.py): this TU uses
// std::min/max/clamp throughout
#include <intrin.h>
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")   // WaitOnAddress / WakeByAddressAll (Pool)
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
// On the AVX2 tier (no VNNI) the byte-sum must be emulated with vpmaddubsw, whose i16 pair sums
// saturate once an activation is inside the pair. The accumulate therefore keeps the pair x-free:
// sum the product bytes once per k-row into i16 pair-sums <= 510, then multiply by x with one
// vpmaddwd per token row. +-127 activations, bit-exact vs the masked accumulate it replaces,
// ~2x its throughput (measured on Zen 3 + Zen 5).
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
#define M1_TARGET_VBMI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))
#else
#define M1_TARGET_AVX2
#define M1_TARGET_VNNI
#define M1_TARGET_VBMI
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

// Declared early: the transforms below select on it. Vbmi = Vnni + AVX512-VBMI (Zen4+, Ice
// Lake+); kept as a separate tier because Cascade/Cooper Lake have VNNI without VBMI.
enum class Isa { Scalar, Avx2, Vnni, Vbmi };
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
    // m x k, x8 (two's-complement, read as i16) in BOTH 16-bit slots of the dword. AVX2
    // bytesum-first accumulate: the product-byte pair sums (b0+b1, b2+b3, each <= 510) are
    // formed once per k-row with x OUTSIDE the pair (so no saturation at full +-127), then a
    // single vpmaddwd per token row against this pattern returns bytesum*x as one i32.
    // Bit-exact reassociation of the same integer sum: 4 ops/row + 4 shared, vs 16 (masked)
    // or 12 (activation-split). VNNI/VBMI ignore.
    int32_t* splat_dup;
    float q[MAX_M];
    int32_t sum_x8[MAX_M];
};

M1_TARGET_AVX2
void quantize_row_avx2(const float* dst, int32_t* splat, int32_t* splat_dup,
    int k, float& q_out, int32_t& s_out)
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
        // x in both 16-bit slots (see PreparedIn::splat_dup): the low 16 bits of v are already
        // the two's-complement i16 activation; replicate into the high slot
        if (splat_dup)
        {
            const __m256i low16 = _mm256_and_si256(v, _mm256_set1_epi32(0xffff));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(splat_dup + i),
                _mm256_or_si256(low16, _mm256_slli_epi32(low16, 16)));
        }
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
        // dup is only read by the AVX2 maddubs kernels; skip the stores on the VNNI/VBMI tiers
        int32_t* splat_dup = (p.splat_dup && g_isa == Isa::Avx2)
            ? p.splat_dup + static_cast<size_t>(r) * k : nullptr;
        float q;
        int32_t s;
        if (g_isa != Isa::Scalar)
        {
            quantize_row_avx2(dst, splat, splat_dup, k, q, s);
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

// Gather the two 32-bit word vectors covering row `row`'s 16-bit states (the permute stage of
// the extraction, split out so a row pair can share it -- see vnni_band_rows)
template <int bits, int row>
M1_TARGET_VNNI
inline void dword_gather(__m512i p0, __m512i p1, __m512i p2, __m512i p3, __m512i& a, __m512i& b)
{
    alignas(64) static constexpr auto i0d = make_row_indices<bits, row, false>();
    alignas(64) static constexpr auto i1d = make_row_indices<bits, row, true>();
    const __m512i i0 = _mm512_load_si512(i0d.data());
    const __m512i i1 = _mm512_load_si512(i1d.data());
    a = _mm512_permutex2var_epi32(p0, i0, p1);
    b = _mm512_permutex2var_epi32(p0, i1, p1);
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
}

// Shift-merge codes for `row` out of its gathered word vectors; delta = bits extracts row+1
// from row's own gather (valid when word_pair_ok). Vector shifts by >= 32 are well-defined
// zero, so the s' == 0 case needs no special path.
template <int bits, int row, int delta>
M1_TARGET_VNNI
inline __m512i dword_codes(__m512i a, __m512i b)
{
    constexpr int s0 = row_shift<bits, row>(0) - delta;
    constexpr int s1 = row_shift<bits, row>(8) - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds shift headroom");
    const __m512i c0 = _mm512_or_si512(_mm512_srli_epi32(b, s0), _mm512_slli_epi32(a, 32 - s0));
    const __m512i c1 = _mm512_or_si512(_mm512_srli_epi32(b, s1), _mm512_slli_epi32(a, 32 - s1));
    return _mm512_and_si512(_mm512_mask_blend_epi32(0xff00, c0, c1), _mm512_set1_epi32(0xffff));
}

template <int bits, int row>
M1_TARGET_VNNI
inline __m512i extract_row(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    __m512i a, b;
    dword_gather<bits, row>(p0, p1, p2, p3, a, b);
    return dword_codes<bits, row, 0>(a, b);
}

// Word-level row pairing: rows 2p/2p+1 differ by exactly `bits` in bit position, so when both
// half-row shifts of the even row have >= bits of headroom, one word gather serves both rows
// (the odd row is the same gather shifted by an extra `bits`). AVX-512's 32 registers absorb
// the extra live values; the identical restructure measured a net LOSS on 16-register AVX2
// (spills), and on the dword path it only pays where the saved permutes beat the unchanged
// shift-merge epilogue: measured +7% K1/K2, +6% K5, -2..-5% K4/K6/K7 (7960X). Even the
// gated-off pair-step body costs ~2% on K4 (deferred dpbusd changes the dependency chains), so
// non-winning K keep the original row-by-row body verbatim.
template <int bits, int row>
constexpr bool word_pair_ok()
{
    return row_shift<bits, row>(0) >= bits && row_shift<bits, row>(8) >= bits;
}

template <int bits>
constexpr bool dword_pair_wins() { return bits == 1 || bits == 2 || bits == 5; }

template <int bits, int rows, int band, int R>
M1_TARGET_VNNI
inline void vnni_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (dword_pair_wins<bits>())
    {
        if constexpr (R < 16) {
            const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
            __m512i a, wb;
            dword_gather<bits, R>(p0, p1, p2, p3, a, wb);
            const __m512i code0 = dword_codes<bits, R, 0>(a, wb);
            __m512i code1;
            if constexpr (word_pair_ok<bits, R>())
            {
                code1 = dword_codes<bits, R, bits>(a, wb);
            }
            else
            {
                dword_gather<bits, R + 1>(p0, p1, p2, p3, a, wb);
                code1 = dword_codes<bits, R + 1, 0>(a, wb);
            }
            const __m512i prod0 = _mm512_mullo_epi32(code0, mult);
            const __m512i prod1 = _mm512_mullo_epi32(code1, mult);
            for (int i = 0; i < rows; ++i)
            {
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R + 1]));
            }
            vnni_band_rows<bits, rows, band, R + 2>(p0, p1, p2, p3, b, splat, k, acc);
        }
    }
    else
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
}

template <int bits, int rows, int band>
M1_TARGET_VNNI
void vnni_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    // The remaining-word count is computed at the call sites: MSVC rejects reading even a
    // constexpr local inside a capture-less lambda (C3493), unlike GCC/clang
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    // Swizzled (band-contiguous) trellis layout: tile (kt, nt) lives at group nt/8, then kt,
    // then member nt%8, so a band's k-stream is (near-)sequential instead of packed_size-sized
    // runs strided by the full row. Requires band-aligned tile ranges (divisors of 8 within one
    // group; enforced by the band tables in *_tiles and the group-aligned splits in
    // forward_phase). Prefetch step differs accordingly.
    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat = in.splat32 + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * tiles_k * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                // Whole-group band on the swizzled layout: the k-stream is sequential, one line
                // one step ahead is enough and the HW prefetcher follows the run (wider/farther
                // measured neutral on K4, 7960X)
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                // Strided stream: the native layout (K8, or VNNI-only CPUs where nothing is
                // swizzled) strides row_stride per step, and a partial-group band on the
                // swizzled layout (rows 3-4 at band 4) reads half a group then skips half; both
                // outrun the HW prefetcher, so touch every line of the tile row a few steps
                // ahead, as the AVX2 tier does (PR #331). +11% decode on K8 (7960X, four
                // order-alternated pairs, 60.9 -> 67.9 tok/s), prefill unchanged; bits == 6 keeps
                // the shorter distance that tier found necessary for its 96-byte rows
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
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

    // Swizzled layout requires bands that are divisors of 8 (whole or partial groups); the
    // VBMI tier widens these (see vbmi_tiles) but the dword scheme's extra live temporaries
    // don't leave the register headroom for that here
    const int max_band = mat.swz ? (rows == 1 ? 8 : rows <= 3 ? 4 : 2)
                                 : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
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
//   AVX-512 VBMI banded kernel
//
//   Replaces the dword extraction's two 32-bit cross permutes + shift-merge with a single
//   byte-level permute (vpermb / vpermt2b): the 3 bytes covering each column's 16-bit state
//   are gathered directly into the dword lane, then one (even K) or two blended (odd K)
//   sub-byte right-shifts + mask produce the same codes bit-exactly.
//
//   Layout facts this relies on (from make_tc_perm; validated bit-exact against the dword
//   scheme for K1-8 x m1-4 in benchmarks/moe_mul1_bench):
//   - within a half-row (cols 0-7 / 8-15) the inverse-permutation index steps by 32 per
//     column, so bit offsets step by 32*bits and shift % 8 is uniform per half-row at every K
//   - the two half-rows differ by 4*bits bits: same shift % 8 for even K, +/-4 for odd K
//   - rows (2p, 2p+1) differ by exactly `bits` bits, so when shift % 8 >= bits for every
//     column of the even row, one gather serves both rows ("byte pairing": K1/K2/K4 all 8
//     pairs, K3/K6 rows 8-15 only, K5/K7/K8 never)
// -------------------------------------------------------------------------------------------

// For each column, byte indices (into the tile's 32*bits packed bytes) of the 3 bytes covering
// bits [shift, shift+16) of the (w0:w1) combined word window; 4th lane byte unused (index 0,
// never observed: shift%8 + 16 <= 23 keeps the value inside the low 3 bytes)
template <int bits, int row>
constexpr std::array<uint8_t, 64> make_row_byte_indices()
{
    std::array<uint8_t, 64> idx{};
    const auto inv = make_tc_perm_inv();
    constexpr int words32 = bits * 256 / 32;
    for (int col = 0; col < 16; ++col)
    {
        const int t = inv[row * 16 + col];
        const int b0 = t * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        const int w0 = (b0 / 32) % words32;          // high (earlier) word
        const int w1 = ((b1 - 1) / 32) % words32;    // low (later) word
        const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
        const int fb = shift / 8;
        for (int byte = 0; byte < 3; ++byte)
        {
            const int mb = fb + byte;
            const int src = mb < 4 ? w1 * 4 + mb : w0 * 4 + (mb - 4);
            idx[col * 4 + byte] = static_cast<uint8_t>(src);
        }
        idx[col * 4 + 3] = 0;
    }
    return idx;
}

// For bits > 4 (tile spans 4 zmms): which gathered bytes come from the (p2,p3) pair.
// vpermt2b consumes idx bits [6:0], so raw indices >= 128 address the high pair directly.
template <int bits, int row>
constexpr uint64_t make_row_byte_himask()
{
    const auto idx = make_row_byte_indices<bits, row>();
    uint64_t m = 0;
    for (int i = 0; i < 64; ++i)
        if (idx[i] >= 128) m |= uint64_t(1) << i;
    return m;
}

// Byte-level pairing is valid iff the odd row's value stays inside the even row's gathered
// byte window for every column, i.e. shift % 8 >= bits everywhere (stricter than the dword
// path's word_pair_ok, which only needs the full shift's headroom)
template <int bits, int row>
constexpr bool byte_pair_ok()
{
    for (int col = 0; col < 16; ++col)
        if (row_shift<bits, row>(col) % 8 < bits) return false;
    return true;
}

template <int bits, int row>
M1_TARGET_VBMI
inline __m512i gather_row_bytes(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    alignas(64) static constexpr auto bidx = make_row_byte_indices<bits, row>();
    const __m512i idx = _mm512_load_si512(bidx.data());
    if constexpr (bits <= 2)
    {
        (void) p1; (void) p2; (void) p3;
        return _mm512_permutexvar_epi8(idx, p0);
    }
    else if constexpr (bits <= 4)
    {
        (void) p2; (void) p3;
        return _mm512_permutex2var_epi8(p0, idx, p1);
    }
    else
    {
        constexpr uint64_t hm = make_row_byte_himask<bits, row>();
        if constexpr (hm == 0)
            return _mm512_permutex2var_epi8(p0, idx, p1);
        else if constexpr (hm == ~uint64_t(0))
            return _mm512_permutex2var_epi8(p2, idx, p3);
        else
            return _mm512_mask_blend_epi8(static_cast<__mmask64>(hm),
                _mm512_permutex2var_epi8(p0, idx, p1),
                _mm512_permutex2var_epi8(p2, idx, p3));
    }
}

// delta = 0 extracts `row` itself; delta = bits extracts row+1 from row's gathered bytes
template <int bits, int row, int delta>
M1_TARGET_VBMI
inline __m512i shift_mask_row(__m512i g)
{
    constexpr int s0 = row_shift<bits, row>(0) % 8 - delta;
    constexpr int s1 = row_shift<bits, row>(8) % 8 - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds sub-byte shift headroom");
    if constexpr (s0 == s1)
        return _mm512_and_si512(_mm512_srli_epi32(g, s0), _mm512_set1_epi32(0xffff));
    else
        return _mm512_and_si512(_mm512_mask_blend_epi32(0xff00,
            _mm512_srli_epi32(g, s0), _mm512_srli_epi32(g, s1)), _mm512_set1_epi32(0xffff));
}

template <int bits, int rows, int band, int P>
M1_TARGET_VBMI
inline void vbmi_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (P < 8)
    {
        constexpr int R = P * 2;
        const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
        __m512i c0, c1;
        if constexpr (byte_pair_ok<bits, R>())
        {
            const __m512i g = gather_row_bytes<bits, R>(p0, p1, p2, p3);
            c0 = shift_mask_row<bits, R, 0>(g);
            c1 = shift_mask_row<bits, R, bits>(g);
        }
        else
        {
            c0 = shift_mask_row<bits, R, 0>(gather_row_bytes<bits, R>(p0, p1, p2, p3));
            c1 = shift_mask_row<bits, R + 1, 0>(gather_row_bytes<bits, R + 1>(p0, p1, p2, p3));
        }
        const __m512i prod0 = _mm512_mullo_epi32(c0, mult);
        const __m512i prod1 = _mm512_mullo_epi32(c1, mult);
        for (int i = 0; i < rows; ++i)
        {
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0,
                _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1,
                _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R + 1]));
        }
        vbmi_band_rows<bits, rows, band, P + 1>(p0, p1, p2, p3, b, splat, k, acc);
    }
}

template <int bits, int rows, int band>
M1_TARGET_VBMI
void vbmi_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    // Same MSVC-safe form as vnni_band (C3493: no constexpr locals read inside the lambda)
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat = in.splat32 + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * tiles_k * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                // Whole-group band on the swizzled layout: the k-stream is sequential, one line
                // one step ahead is enough and the HW prefetcher follows the run (wider/farther
                // measured neutral on K4, 7960X)
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                // Strided stream: the native layout (K8, or VNNI-only CPUs where nothing is
                // swizzled) strides row_stride per step, and a partial-group band on the
                // swizzled layout (rows 3-4 at band 4) reads half a group then skips half; both
                // outrun the HW prefetcher, so touch every line of the tile row a few steps
                // ahead, as the AVX2 tier does (PR #331). +11% decode on K8 (7960X, four
                // order-alternated pairs, 60.9 -> 67.9 tok/s), prefill unchanged; bits == 6 keeps
                // the shorter distance that tier found necessary for its 96-byte rows
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            vbmi_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat, mat.k, acc);
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
M1_TARGET_VBMI
void vbmi_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    constexpr int band_cap = 8;

    // Swizzled layout: bands must be divisors of 8 (whole or partial groups). Unlike the
    // dword scheme, byte-gather extraction needs few temporaries, so wider bands (up to 16
    // zmm accumulators: rows2 x band8, rows3/4 x band4) fit the register budget and keep the
    // swizzled stream at full duty. Narrow divisor bands (read-N-skip-N) measured BELOW
    // native layout at m>1. K2 rows4 prefers band 2 (measured 216 vs 197 Gw/s at band 4).
    const int max_band = mat.swz
        ? (rows <= 2 ? 8 : (rows == 4 && bits == 2 ? 2 : 4))
        : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: vbmi_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: vbmi_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: vbmi_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: vbmi_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: vbmi_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: vbmi_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: vbmi_band<bits, rows, 7>(mat, in, tout, n0); break;
            default: vbmi_band<bits, rows, 8>(mat, in, tout, n0); break;
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

M1_TARGET_AVX2
inline void avx2_accum_row(__m256i codes_lo, __m256i codes_hi, const int32_t* splat_dup, int k,
    int m, __m256i (&acc)[MAX_M][2], const __m256i& mult, const __m256i& ones32, int row)
{
    // Bytesum-first accumulate: maddubs(prod, 0x01010101) sums each product-byte pair into an
    // i16 lane ((b0+b1), (b2+b3), <= 510). x is OUTSIDE the pair so vpmaddubsw cannot saturate
    // at full +-127 activations; one vpmaddwd per token row against splat_dup (x8 in both 16-bit
    // slots) then folds (b0+b1)*x+(b2+b3)*x into a single i32. 4 shared + 4 per-row ops,
    // bit-exact vs the 16-op masked accumulate it replaces (verified K1-K8 x m1-4 against an
    // exact scalar reference). The token loop is unrolled by hand: with a runtime-bounded loop
    // GCC spills the pair sums and pays per-iteration overhead (~1.4x on Zen 3).
    const __m256i p_lo = _mm256_maddubs_epi16(_mm256_mullo_epi32(codes_lo, mult), ones32);
    const __m256i p_hi = _mm256_maddubs_epi16(_mm256_mullo_epi32(codes_hi, mult), ones32);
    #define ACC_ROW(i) \
        if ((i) < m) { \
            const __m256i xs = _mm256_set1_epi32(splat_dup[static_cast<size_t>(i) * k + row]); \
            acc[i][0] = _mm256_add_epi32(acc[i][0], _mm256_madd_epi16(p_lo, xs)); \
            acc[i][1] = _mm256_add_epi32(acc[i][1], _mm256_madd_epi16(p_hi, xs)); \
        }
    ACC_ROW(0) ACC_ROW(1) ACC_ROW(2) ACC_ROW(3)
    #undef ACC_ROW
}

// Word-level row pairing on AVX2 is gated to bits == 8 ONLY: measured +11% there (each gather
// walks all 8 candidate registers, so halving gathers wins even though the 4 shared word
// registers staying live across the even row's accumulate spills -- AVX2 has 16 architectural
// ymm registers). At every other K the same restructure measured neutral-to-negative
// (K3 -5% 1T / -14% 24T-cold on the 7960X); do not widen the gate without re-measuring.
template <int bits, int row = 0>
M1_TARGET_AVX2
inline void avx2_rows_accum(
    const __m256i (&preg)[bits], const int32_t* splat_dup, int k, int m, __m256i (&acc)[MAX_M][2],
    const __m256i& mult, const __m256i& ones32)
{
    if constexpr (bits == 8)
    {
        if constexpr (row < 16)
        {
            static_assert(word_pair_ok<bits, row>(), "K8 pairs are fully eligible by layout");
            const __m256i a_lo = avx2_gather_half<bits, row, false, 0>(preg);
            const __m256i b_lo = avx2_gather_half<bits, row, true, 0>(preg);
            const __m256i a_hi = avx2_gather_half<bits, row, false, 1>(preg);
            const __m256i b_hi = avx2_gather_half<bits, row, true, 1>(preg);
            constexpr int s0 = row_shift<bits, row>(0);
            constexpr int s1 = row_shift<bits, row>(8);
            const __m256i mask16 = _mm256_set1_epi32(0xffff);
            __m256i codes_lo = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_lo, s0), _mm256_slli_epi32(a_lo, 32 - s0)), mask16);
            __m256i codes_hi = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_hi, s1), _mm256_slli_epi32(a_hi, 32 - s1)), mask16);
            avx2_accum_row(codes_lo, codes_hi, splat_dup, k, m, acc, mult, ones32, row);
            // Odd row: same gathered words, shifted by an extra `bits` (>= 32 shifts are
            // well-defined zero, so the slli term drops out cleanly when s - bits == 0)
            codes_lo = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_lo, s0 - bits), _mm256_slli_epi32(a_lo, 32 - (s0 - bits))), mask16);
            codes_hi = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_hi, s1 - bits), _mm256_slli_epi32(a_hi, 32 - (s1 - bits))), mask16);
            avx2_accum_row(codes_lo, codes_hi, splat_dup, k, m, acc, mult, ones32, row + 1);
            avx2_rows_accum<bits, row + 2>(preg, splat_dup, k, m, acc, mult, ones32);
        }
    }
    else if constexpr (row < 16)
    {
        __m256i codes_lo, codes_hi;
        avx2_row_codes<bits, row>(preg, codes_lo, codes_hi);
        avx2_accum_row(codes_lo, codes_hi, splat_dup, k, m, acc, mult, ones32, row);
        avx2_rows_accum<bits, row + 1>(preg, splat_dup, k, m, acc, mult, ones32);
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
    const __m256i ones32 = _mm256_set1_epi32(0x01010101);
    const int32_t* splat_dup = in.splat_dup;

    // The k-major stream strides row_stride (>= 8 KB) per step, beyond what the HW prefetcher
    // tracks (and the swizzled layout the VNNI path uses is not applied for AVX2). Cold-stack
    // (offloaded expert) microbench: +20..70% at K>=4, largest at K8, warm-neutral, so always
    // on. Distance 4 measured best cold (>= 2 everywhere within noise, 4 adds another +10..35%
    // at K5-K8 cold); prefetching past the allocation end is architecturally safe.
    constexpr int pf_lines = (32 * bits + 63) / 64;   // cache lines per tile row
    // bits==6 (96B rows) collapses at distance 4 when cold (reproducibly ~2x slower on both
    // the 7960X and this Zen5 box; the 3-line window from 4 rows out interacts badly with the
    // 96B stride). Distance 2 measures >= everywhere else for K6 while costing <2% warm.
    constexpr int pf_dist = (bits == 6) ? 2 : 4;

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
            const uint16_t* pf = packed + row_stride * pf_dist;
            #pragma unroll
            for (int l = 0; l < pf_lines; ++l)
                _mm_prefetch(reinterpret_cast<const char*>(pf) + l * 64, _MM_HINT_T0);

            const int32_t* splat_k = splat_dup + tile_k * 16;
            // One 256-bit (8xu32) register per bits: covers packed_size = 16*bits u16 = bits*8
            // u32 words exactly, the whole k-tile's row of packed states
            __m256i preg[bits];
            for (int i = 0; i < bits; ++i)
                preg[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i * 16));
            avx2_rows_accum<bits>(preg, splat_k, mat.k, m, acc, mult, ones32);
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
        hw = __builtin_cpu_supports("avx512vbmi") ? Isa::Vbmi : Isa::Vnni;
    else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        hw = Isa::Avx2;
    else
        hw = Isa::Scalar;
#else
    // __builtin_cpu_supports checks OS state-saving internally; this branch must do it by hand:
    // CPUID feature bits report hardware capability only, so without OSXSAVE + XCR0 checks a
    // hypervisor/OS that doesn't context-switch YMM/ZMM state would pass detection and fault at
    // the first vector instruction. FMA (leaf 1) mirrors the Linux branch's avx2+fma gate.
    int l0[4];
    __cpuid(l0, 0);
    if (l0[0] < 7)
    {
        hw = Isa::Scalar;
    }
    else
    {
        int l1[4];
        __cpuid(l1, 1);
        const bool osxsave = (l1[2] & (1u << 27)) != 0;
        const bool fma = (l1[2] & (1u << 12)) != 0;
        const uint64_t xcr0 = osxsave ? _xgetbv(0) : 0;
        const bool ymm_os = (xcr0 & 0x06) == 0x06;          // XMM + YMM state
        const bool zmm_os = (xcr0 & 0xe6) == 0xe6;          // + opmask, ZMM_Hi256, Hi16_ZMM
        int info[4];
        __cpuidex(info, 7, 0);
        const bool avx512 = (info[1] & (1u << 16)) && (info[1] & (1u << 30)) && (info[1] & (1u << 31));
        const bool vnni = (info[2] & (1u << 11)) != 0;
        const bool vbmi = (info[2] & (1u << 1)) != 0;
        const bool avx2 = (info[1] & (1u << 5)) != 0;
        hw = (avx512 && vnni && zmm_os) ? (vbmi ? Isa::Vbmi : Isa::Vnni)
           : (avx2 && fma && ymm_os)    ? Isa::Avx2
           :                              Isa::Scalar;
    }
#endif

    // EXL3_MOE_CPU_MAX_ISA=scalar|avx2|vnni|vbmi: cap detection at a lower tier for testing
    // (e.g. exercising the AVX2 path on AVX512-VNNI hardware, or the dword scheme on VBMI
    // hardware). Never upgrades past what the CPU actually supports; an unrecognized value is
    // ignored.
    if (const char* e = std::getenv("EXL3_MOE_CPU_MAX_ISA"))
    {
        std::string s(e);
        for (char& c : s) c = (char) std::tolower((unsigned char) c);
        Isa cap;
        if (s == "scalar") cap = Isa::Scalar;
        else if (s == "avx2") cap = Isa::Avx2;
        else if (s == "vnni" || s == "avx512") cap = Isa::Vnni;
        else if (s == "vbmi") cap = Isa::Vbmi;
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
        case Isa::Vbmi:
        {
            switch (mat.bits * 4 + m - 1)
            {
                case 1 * 4 + 0: vbmi_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
                case 1 * 4 + 1: vbmi_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
                case 1 * 4 + 2: vbmi_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
                case 1 * 4 + 3: vbmi_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 0: vbmi_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 1: vbmi_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 2: vbmi_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
                case 2 * 4 + 3: vbmi_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 0: vbmi_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 1: vbmi_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 2: vbmi_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
                case 3 * 4 + 3: vbmi_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 0: vbmi_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 1: vbmi_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 2: vbmi_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
                case 4 * 4 + 3: vbmi_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 0: vbmi_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 1: vbmi_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 2: vbmi_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
                case 5 * 4 + 3: vbmi_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 0: vbmi_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 1: vbmi_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 2: vbmi_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
                case 6 * 4 + 3: vbmi_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 0: vbmi_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 1: vbmi_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 2: vbmi_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
                case 7 * 4 + 3: vbmi_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
                // K8: byte pairing impossible (shift % 8 == 0) and the byte windows straddle
                // the register pairs -- measured slower than the dword scheme, so route there
                case 8 * 4 + 0: vnni_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
                case 8 * 4 + 1: vnni_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
                case 8 * 4 + 2: vnni_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
                case 8 * 4 + 3: vnni_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
            }
            return;
        }
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

// Physical-core-first CPU ordering: one logical CPU per distinct physical core, SMT siblings
// appended after. Without this, spawned std::thread workers are placed wherever the scheduler
// puts them, which on an SMT host can silently collide two workers onto one physical core.
// EXL3_MOE_CPU_PIN=0 disables.
//
// Linux: entries are plain logical CPU indices (as taken by CPU_SET). Windows: entries encode
// (processor group << 16) | bit-within-group, decoded by Pool::pin_self -- SetThreadAffinityMask
// only addresses the calling thread's current group, so systems with more than 64 logical
// processors (multiple processor groups) need the group-aware SetThreadGroupAffinity instead.
// UNVERIFIED: no Windows toolchain was available to compile-test this branch; check it (e.g. via
// EXL3_MOE_CPU_PROF timing before/after, or Task Manager's per-core view during a CPU-offloaded
// pass) before relying on it on a real system, particularly one with multiple processor groups.
#ifdef __linux__
inline std::vector<int> physical_core_order()
{
    std::vector<int> order;
    std::map<std::pair<int, int>, int> seen;
    std::vector<int> smt_siblings;
    const int ncpu = static_cast<int>(std::thread::hardware_concurrency());
    for (int cpu = 0; cpu < ncpu; ++cpu)
    {
        auto read_int = [&](const char* file) -> int {
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + file);
            int v = -1;
            f >> v;
            return v;
        };
        const int core_id = read_int("topology/core_id");
        const int pkg_id = read_int("topology/physical_package_id");
        if (core_id < 0) { order.push_back(cpu); continue; }   // topology unreadable: fall back
        auto key = std::make_pair(pkg_id, core_id);
        if (seen.find(key) == seen.end()) { seen[key] = cpu; order.push_back(cpu); }
        else smt_siblings.push_back(cpu);
    }
    order.insert(order.end(), smt_siblings.begin(), smt_siblings.end());
    return order;
}
#else
inline std::vector<int> physical_core_order()
{
    std::vector<int> order, smt_siblings;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return order;
    std::vector<char> buf(len);
    auto* first_rec = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first_rec, &len)) return order;
    size_t off = 0;
    while (off < len)
    {
        auto* rec = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data() + off);
        if (rec->Relationship == RelationProcessorCore)
        {
            bool first = true;
            for (WORD g = 0; g < rec->Processor.GroupCount; ++g)
            {
                const GROUP_AFFINITY& ga = rec->Processor.GroupMask[g];
                for (int bit = 0; bit < 64; ++bit)
                {
                    if (ga.Mask & (KAFFINITY(1) << bit))
                    {
                        const int enc = (static_cast<int>(ga.Group) << 16) | bit;
                        if (first) { order.push_back(enc); first = false; }
                        else smt_siblings.push_back(enc);
                    }
                }
            }
        }
        off += rec->Size;
    }
    order.insert(order.end(), smt_siblings.begin(), smt_siblings.end());
    return order;
}
#endif

inline bool pin_threads_enabled()
{
    static const bool v = [] {
        const char* e = std::getenv("EXL3_MOE_CPU_PIN");
        return !(e && *e == '0');
    }();
    return v;
}

struct Pool
{
    int spawned = 0;
    // One word per dispatch: generation in the high bits, participant count in the low 16. A
    // worker reads both in a single load, so a worker that was preempted between observing a
    // new generation and reading its participant count can never pair a stale generation with
    // the next dispatch's count
    std::atomic<uint64_t> dispatch{0};
    uint64_t generation = 0;
    std::atomic<uint64_t> done{0};
    std::atomic<PoolFn> fn{nullptr};
    void* ctx = nullptr;
    int num_workers = 1;
    static constexpr int DISPATCH_NW_BITS = 16;
    static int dispatch_nw(uint64_t d) { return (int) (d & ((1ull << DISPATCH_NW_BITS) - 1)); }
    std::vector<int> core_order;

    void pin_self(int idx)
    {
        if (core_order.empty()) return;
        const int enc = core_order[idx % core_order.size()];
#ifdef __linux__
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(enc, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
        GROUP_AFFINITY ga{};
        ga.Group = static_cast<WORD>(enc >> 16);
        ga.Mask = KAFFINITY(1) << (enc & 0xffff);
        SetThreadGroupAffinity(GetCurrentThread(), &ga, nullptr);
#endif
    }

    void worker_loop(int idx)
    {
        pin_self(idx);
        uint64_t seen = 0;
        int idle = 0;
        while (true) {
            const uint64_t g = dispatch.load(std::memory_order_acquire);
            if (g == seen)
            {
                // Matches the outer job-ring poll's threshold (moe_handoff.cu)
                if (++idle < 65536) { cpu_pause(); continue; }
#ifdef __linux__
                std::this_thread::sleep_for(std::chrono::microseconds(50));
#else
                // Never a timed nap here: Windows rounds short sleeps up to the timer quantum
                // (default 15.6 ms), and the run() barrier turns one late waker into everyone
                // oversleeping the next phase dispatc
                uint64_t cmp = seen;
                WaitOnAddress(&dispatch, &cmp, sizeof(uint64_t), INFINITE);
#endif
                continue;
            }
            idle = 0;
            seen = g;

            // The participant count may sit below this worker's index (pool shrink, or a
            // small-job dispatch cap): surplus workers must not run the function (their
            // (idx, nw) pair indexes out of range) and must not ack, or run() returns before
            // the participating workers have finished. The count travels in the dispatch word
            // itself, so it always belongs to the generation just observed
            const int nw = dispatch_nw(g);
            if (idx < nw)
            {
                fn.load(std::memory_order_relaxed)(ctx, idx, nw);
                done.fetch_add(1, std::memory_order_release);
            }
        }
    }

    void ensure(int n)
    {
        if (pin_threads_enabled() && core_order.empty()) core_order = physical_core_order();
        while (spawned < n - 1)
        {
            std::thread(&Pool::worker_loop, this, spawned + 1).detach();
            ++spawned;
        }
        num_workers = n;
        pin_self(0);   // worker 0 is the calling thread itself, never goes through worker_loop
    }

    // Run fn on workers 0..n-1; returns when all are done (implicit barrier). n_req > 0
    // caps the worker count for this run: small jobs (one or two experts) saturate RAM
    // bandwidth on a fraction of the pool, and every surplus worker is another straggler
    // candidate at the six per-phase barriers
    void run(PoolFn f, void* c, int n_req = 0)
    {
        int n = num_workers;
        if (n_req > 0 && n_req < n) n = n_req;
        if (n <= 1) { f(c, 0, 1); return; }
        ctx = c;
        fn.store(f, std::memory_order_relaxed);
        const uint64_t d0 = done.load(std::memory_order_acquire);
        ++generation;
        dispatch.store((generation << DISPATCH_NW_BITS) | (uint64_t) n, std::memory_order_release);
#ifndef __linux__
        WakeByAddressAll(&dispatch);
#endif
        f(c, 0, n);
        while (static_cast<int64_t>(done.load(std::memory_order_acquire) - d0) < n - 1)
            cpu_pause();
    }
};

Pool g_pool;
std::mutex g_pool_mutex;

// -------------------------------------------------------------------------------------------
//   Pool self-test hook (tests/test_moe_cpu_pool_.py)
// -------------------------------------------------------------------------------------------

// Drives the pool with a participant count alternating between `small` and the full pool, with
// `threads` workers (oversubscribe to force preemption inside the dispatch handshake). Every
// dispatch must run each participant exactly once, no non-participant at all, and run() must not
// return while a participant is still inside the function. Returns the number of anomalies.
struct PoolStressCtx
{
    std::atomic<int>* runs;
    std::atomic<int>* active;
    int spin;
};

static void pool_stress_fn(void* c, int idx, int nw)
{
    auto* s = (PoolStressCtx*) c;
    s->active->fetch_add(1, std::memory_order_acq_rel);
    s->runs[idx].fetch_add(1, std::memory_order_acq_rel);
    volatile uint64_t x = (uint64_t) idx;
    for (int i = 0; i < s->spin * (1 + (idx % 7)); ++i)
        x = x * 6364136223846793005ull + 1442695040888963407ull;
    s->active->fetch_sub(1, std::memory_order_acq_rel);
}


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
    std::vector<int32_t> splat_dup_g, splat_dup_u, splat_dup_d;
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

// Contiguous tile range for split sub/per of one GEMV. Swizzled matrices hand out whole
// 8-tile groups per worker: the band kernels' swizzled addressing assumes n0's group is not
// split mid-band (band tables only produce divisors of 8, which stay inside a group only
// when the range starts group-aligned).
inline void tile_split(const MoeCpuMatrix& mat, int sub, int per, int& t0, int& t1)
{
    const int tiles_n = mat.n / 16;
    if (mat.swz)
    {
        const int groups = tiles_n / 8;
        t0 = groups * sub / per * 8;
        t1 = groups * (sub + 1) / per * 8;
    }
    else
    {
        t0 = tiles_n * sub / per;
        t1 = tiles_n * (sub + 1) / per;
    }
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
                    int t0, t1;
                    tile_split(mat, sub, per, t0, t1);
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
                // Nonzero act_limit clamps the up path symmetrically and the activated gate
                // from above, BEFORE the multiply (matching the GPU act_mul kernels). DS4
                // ships swiglu_limit = 10 with plain silu: hidden states deep into a long
                // context push |u| into the thousands, and skipping the clamp here made
                // offloaded experts diverge arbitrarily far from their GPU-resident twins
                const float lim = L.act_limit != 0.0f
                    ? L.act_limit : std::numeric_limits<float>::infinity();
                switch (L.activation) {
                    case 0:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float av = std::min(gv / (1.0f + std::exp(-gv)), lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 1:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float cdf = 0.5f * (1.0f + std::erf(gv * 0.70710678f));
                            const float av = std::min(gv * cdf, lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
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
                    int t0, t1;
                    tile_split(mat, sub, per, t0, t1);
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

// Staged bytes are copied verbatim, swizzled or not: the GPU restores the native tile order
// after the DMA (moe_unswizzle_trellis), which is one read + one write at VRAM bandwidth instead
// of tiles_k * groups scattered memcpys here. Un-swizzling on the stager thread measured as the
// whole VBMI-vs-VNNI prefill gap on a fully streamed 119B model (~17% at 32K).
inline void stage_copy_trellis(uint8_t* dst, const MoeCpuMatrix& m)
{
    std::memcpy(dst, m.trellis, trellis_bytes(m));
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
        const MoeCpuMatrix* m;
        if (gated && mi == 0)      { m = &c.layer->gates[e]; }
        else if (mi == (gated ? 1 : 0)) { m = &c.layer->ups[e]; off += gb; }
        else                       { m = &c.layer->downs[e]; off += gb + ub; }
        stage_copy_trellis(c.dst + off, *m);
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
bool exl3_moe_cpu_has_avx512_vnni() { return g_isa >= Isa::Vnni; }
bool exl3_moe_cpu_has_avx512_vbmi() { return g_isa == Isa::Vbmi; }

static MoeCpuMatrix make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias,
    bool swizzled
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
    // K8 tensors are exempt from swizzling (routed to the dword kernel, which would gain
    // nothing) -- the child loader applies the same bits != 8 rule when repacking, so the two
    // sides agree per tensor
    m.swz = swizzled && m.bits != 8 ? 1 : 0;
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
    double act_limit,
    int64_t swizzled
)
{
    auto* layer = new MoeCpuLayer;
    const bool swz = swizzled != 0;
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
                                               gate_bias.empty() ? nullptr : &gate_bias[e], swz));
            for (auto& t : {gate_trellis[e], gate_suh[e], gate_svh[e]})
                layer->refs.push_back(t);
            if (!gate_bias.empty()) layer->refs.push_back(gate_bias[e]);
        }
        layer->ups.push_back(make_matrix(up_trellis[e], up_suh[e], up_svh[e],
                                         up_bias.empty() ? nullptr : &up_bias[e], swz));
        layer->downs.push_back(make_matrix(down_trellis[e], down_suh[e], down_svh[e],
                                           down_bias.empty() ? nullptr : &down_bias[e], swz));
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

// Exported pool self-test (helpers above live in the anonymous namespace of this TU)
int64_t exl3_moe_cpu_pool_stress(int threads, int iters, int small, int spin)
{
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.ensure(threads);
    std::vector<std::atomic<int>> runs(threads);
    std::atomic<int> active{0};
    int64_t anomalies = 0;
    for (int it = 0; it < iters; ++it)
    {
        const int n_req = (it & 1) ? small : 0;
        const int n = (n_req > 0 && n_req < threads) ? n_req : threads;
        for (auto& r : runs) r.store(0, std::memory_order_relaxed);
        PoolStressCtx c{runs.data(), &active, spin};
        g_pool.run(&pool_stress_fn, &c, n_req);
        if (active.load(std::memory_order_acquire) != 0) ++anomalies;          // returned early
        for (int i = 0; i < threads; ++i)
        {
            const int r = runs[i].load(std::memory_order_acquire);
            if (i < n ? r != 1 : r != 0) ++anomalies;                           // double / missing / surplus run
        }
    }
    return anomalies;
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
    grow(ar.splat_dup_g, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.splat_dup_u, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.splat_dup_d, static_cast<size_t>(nc) * MAX_M * I);
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
                          ar.splat_g.data() + static_cast<size_t>(j) * MAX_M * H,
                          ar.splat_dup_g.data() + static_cast<size_t>(j) * MAX_M * H, {}, {} };
        ctx.prep_u[j] = { ar.tin_u.data() + static_cast<size_t>(j) * MAX_M * H,
                          ar.splat_u.data() + static_cast<size_t>(j) * MAX_M * H,
                          ar.splat_dup_u.data() + static_cast<size_t>(j) * MAX_M * H, {}, {} };
        ctx.prep_d[j] = { ar.tin_d.data() + static_cast<size_t>(j) * MAX_M * I,
                          ar.splat_d.data() + static_cast<size_t>(j) * MAX_M * I,
                          ar.splat_dup_d.data() + static_cast<size_t>(j) * MAX_M * I, {}, {} };
    }

    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.ensure(threads > 0 ? threads : 1);

    // Small-job worker cap (see Pool::run): enough cores to saturate RAM bandwidth on a
    // couple of expert GEMVs, few enough to keep the phase barriers tight
    static const int small_cap = [](){
        const char* e = getenv("EXL3_MOE_CPU_SMALL_WORKERS");
        return e ? atoi(e) : 0;
    }();
    const int n_run = nc <= 2 ? small_cap : 0;

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
            g_pool.run(&forward_phase, &ctx, n_run);
            phase_us[phase] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
        else
        {
            g_pool.run(&forward_phase, &ctx, n_run);
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
