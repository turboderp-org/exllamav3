// Native vision-tower CPU GEMM: C = A @ B (+ bias), fp32 accumulation, fused epilogue.
//
// Replaces the torch oneDNN matmul on the vision CPU offload path (see cpu_matmul): the
// oneDNN call pays a full activation upcast temporary plus separate bias-add and output-
// cast passes. This kernel streams A with on-the-fly conversion, reads B from a repacked
// panel-major cache, accumulates in registers, and applies bias + output cast while C is
// still hot.
//
// Two compute routes, selected per call in C++ (never by the caller):
//   dpbf16: A (half/bf16) x B (bf16) via AVX512-BF16 vpdpbusd-class dot product
//           (_mm512_dpbf16_ps), fp32 accumulation. This is the Qwen vision hot path
//           (bf16 weights) and the only way to match oneDNN's dpbf16 throughput.
//   FMA:    fp32 accumulate for all other covered combos (fp32 B, fp32 A, ...).
//   AVX2/scalar fallbacks preserve correctness where AVX-512 is unavailable.
//
// B arrives repacked as panels [NB, K, 64] (NB = ceil(N / 64), last panel zero-padded;
// see repack_64 in util/tensor.py): each k-step then walks B sequentially. A strided walk
// over plain [K, N] stalls ~5x on L3 latency with this kernel's reuse pattern (measured:
// 200 vs 1000 GFLOP/s), so the Python layer repacks resident weights once and caches the
// panels; these entries never see plain B.
//
// Blocking (per worker tile of MC rows x NC cols, driven by the shared MoE pool):
//   dpbf16/FMA-512: 8x64 / 4x64 register blocks (16 explicitly-named zmm accumulators).
//   AVX2:           4x16 register blocks (8 explicitly-named ymm accumulators).
// N/M tails use a scalar helper (never masked vector stores); K needs no tail handling.
//
// Codegen rules learned the hard way (verified in the emitted assembly):
//   1. Accumulators must be individually-named variables, never an indexed array:
//      GCC keeps vector arrays in memory and spills them every k-iteration (~10x loss).
//   2. Helpers called inside a target-attributed kernel must carry the same target
//      attribute: GCC will not inline a baseline-ISA callee into an AVX-512 caller,
//      turning every A element load into a real call (another ~10x loss).
// Concepts follow the llamafile tinyBLAS structure (register-blocked GEMM, fused epilogue)
// reimplemented for the panel-major weight layout used here; no vendored code.

// dpbf16 micro-kernels need the BF16 ISA tier (strict superset of AVX-512)
#define VG_DP_TARGET EXL3_CPU_TARGET_BF16

#include "vision_gemm.h"
#include "exl3_common.h"
#include "moe_mul1.h"

#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cstring>

namespace {

constexpr int VG_MC = 256;
// Multiple of 64 so tile origins are panel-aligned
constexpr int VG_NC = 512;
// Reduction panels for the dpbf16 A-pack (8 rows x KC bf16 = 4 KB, L1-resident)
constexpr int VG_KC = 256;
// Panel width; every vision N (768/1024/2304/3072/...) is a multiple of 64
constexpr int VG_R = 64;

enum VgDtype : uint8_t { VG_F32, VG_F16, VG_BF16 };
enum VgIsa : uint8_t { VG_SCALAR, VG_AVX2, VG_AVX512 };

// Global pair-interleave indices for the dpbf16 B operand: dst[2i]=a[i], dst[2i+1]=b[i]
// (16-bit lanes). The unpack instructions interleave per 128-bit lane, but dpbf16 pairs
// lanes globally, so a permutex2var with these indices is required (verified by one-hot
// column probing: unpack-only permutes columns in 8-wide groups).
static const uint16_t VG_IDX_LO[32] = {
    0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39,
    8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47
};
static const uint16_t VG_IDX_HI[32] = {
    16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
    24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63
};
VgDtype vg_dtype(at::ScalarType t)
{
    if (t == at::kFloat) return VG_F32;
    if (t == at::kHalf) return VG_F16;
    return VG_BF16;
}

inline int vg_esz(VgDtype t) { return t == VG_F32 ? 4 : 2; }

VgIsa vg_pick_isa()
{
    static const VgIsa v = [] {
        if (const char* e = std::getenv("EXL3_VISION_GEMM_ISA"))
        {
            if (std::strcmp(e, "scalar") == 0) return VG_SCALAR;
            if (std::strcmp(e, "avx2") == 0) return VG_AVX2;
            if (std::strcmp(e, "avx512") == 0) return VG_AVX512;
        }
        return exl3_moe_cpu_has_avx512() ? VG_AVX512 : VG_AVX2;
    }();
    return v;
}

bool vg_has_bf16()
{
#if defined(__GNUC__) && defined(__linux__)
    static const bool v = __builtin_cpu_supports("avx512bf16");
    return v;
#else
    return false;
#endif
}

// Scalar lane conversions (tails and the scalar fallback; vector paths convert inline)
inline float vg_load_scalar(const char* p, VgDtype dt)
{
    if (dt == VG_F32)
    {
        float f;
        std::memcpy(&f, p, 4);
        return f;
    }
    uint16_t b;
    std::memcpy(&b, p, 2);
    if (dt == VG_F16) return exl3_common::half_to_float(b);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline void vg_store_scalar(char* p, VgDtype dt, float f)
{
    if (dt == VG_F32)
    {
        std::memcpy(p, &f, 4);
        return;
    }
    if (dt == VG_F16)
    {
        const c10::Half h(f);
        uint16_t b = h.x;
        std::memcpy(p, &b, 2);
        return;
    }
    // round-to-nearest-even bf16 (matches torch cpu conversion for finite magnitudes)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    const uint16_t b = static_cast<uint16_t>(u >> 16);
    std::memcpy(p, &b, 2);
}

struct VgCtx
{
    const char* A;
    const char* B;
    const float* bias;   // nullable, fp32 [N]
    char* C;
    int64_t M, K, N;
    int aesz, besz, cesz;
    VgDtype adt, bdt, cdt;
    VgIsa isa;
    bool use_dp;         // dpbf16 route (bf16 B, half/bf16 A, AVX512-BF16 present)
};

// Full scalar fallback, also used for N/M tails of the vector paths. B is panel-major
// [NB, K, 64], so element (k, n) lives at panel (n / 64), row k, lane (n % 64).
void vg_scalar_tile(const VgCtx& c, int64_t m0, int64_t m1, int64_t n0, int64_t n1)
{
    for (int64_t m = m0; m < m1; ++m)
    {
        const char* arow = c.A + static_cast<size_t>(m) * c.K * c.aesz;
        char* crow = c.C + static_cast<size_t>(m) * c.N * c.cesz;
        for (int64_t n = n0; n < n1; ++n)
        {
            float acc = 0.0f;
            const size_t boff = (static_cast<size_t>(n / VG_R) * c.K) * VG_R + (n % VG_R);
            for (int64_t k = 0; k < c.K; ++k)
            {
                const float a = vg_load_scalar(arow + static_cast<size_t>(k) * c.aesz, c.adt);
                const float b = vg_load_scalar(
                    c.B + (boff + static_cast<size_t>(k) * VG_R) * c.besz, c.bdt);
                acc += a * b;
            }
            if (c.bias) acc += c.bias[n];
            vg_store_scalar(crow + static_cast<size_t>(n) * c.cesz, c.cdt, acc);
        }
    }
}

// -------------------------------------------------------------------------------------------
// AVX-512 FMA path: 4 rows x 64 cols (fp32 accumulation, any covered A/B)
// -------------------------------------------------------------------------------------------

template <VgDtype ADT>
EXL3_CPU_TARGET_AVX512
inline float vg_ld_a512(const char*& ap)
{
    float f;
    if constexpr (ADT == VG_F32)
    {
        std::memcpy(&f, ap, 4);
        ap += 4;
        return f;
    }
    uint16_t b;
    std::memcpy(&b, ap, 2);
    ap += 2;
    if constexpr (ADT == VG_F16) return exl3_common::half_to_float(b);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    std::memcpy(&f, &u, 4);
    return f;
}
// Vector A-pack: one row of half/bf16 -> fp32 (both widenings exact, so the pack is
// bit-transparent). Runs once per tile row; the FMA blocks then reuse it across all
// 32-wide n-blocks instead of scalar-converting per (row, k, n-block).
template <VgDtype ADT>
EXL3_CPU_TARGET_AVX512
inline void vg_pack_a_f32(const char* arow, int64_t K, float* dst)
{
    static_assert(ADT == VG_F16 || ADT == VG_BF16);
    int64_t k = 0;
    if constexpr (ADT == VG_F16)
    {
        for (; k + 16 <= K; k += 16)
        {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(arow + static_cast<size_t>(k) * 2));
            _mm512_storeu_ps(dst + k, _mm512_cvtph_ps(v));
        }
    }
    else
    {
        for (; k + 16 <= K; k += 16)
        {
            const __m256i v = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(arow + static_cast<size_t>(k) * 2));
            _mm512_storeu_ps(dst + k, _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu16_epi32(v), 16)));
        }
    }
    for (; k < K; ++k)
    {
        uint16_t b;
        std::memcpy(&b, arow + static_cast<size_t>(k) * 2, 2);
        if constexpr (ADT == VG_F16)
            dst[k] = exl3_common::half_to_float(b);
        else
        {
            uint32_t u = static_cast<uint32_t>(b) << 16;
            std::memcpy(dst + k, &u, 4);
        }
    }
}

template <VgDtype BDT>
EXL3_CPU_TARGET_AVX512
inline void vg_ld_b512(const char* bp, __m512& b0, __m512& b1)
{
    if constexpr (BDT == VG_F32)
    {
        const float* p = reinterpret_cast<const float*>(bp);
        b0 = _mm512_loadu_ps(p);
        b1 = _mm512_loadu_ps(p + 16);
    }
    else if constexpr (BDT == VG_F16)
    {
        const __m256i* p = reinterpret_cast<const __m256i*>(bp);
        b0 = _mm512_cvtph_ps(_mm256_loadu_si256(p));
        b1 = _mm512_cvtph_ps(_mm256_loadu_si256(p + 1));
    }
    else
    {
        const __m256i* p = reinterpret_cast<const __m256i*>(bp);
        b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
            _mm256_loadu_si256(p)), 16));
        b1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
            _mm256_loadu_si256(p + 1)), 16));
    }
}

template <VgDtype ADT>
EXL3_CPU_TARGET_AVX512
inline void vg_row512(const char*& ap, const __m512& b0, const __m512& b1,
                      __m512& c0, __m512& c1)
{
    const __m512 a = _mm512_set1_ps(vg_ld_a512<ADT>(ap));
    c0 = _mm512_fmadd_ps(a, b0, c0);
    c1 = _mm512_fmadd_ps(a, b1, c1);
}

template <VgDtype CDT>
EXL3_CPU_TARGET_AVX512
inline void vg_epi512(char* crow, const float* bias, int64_t n0,
                      const __m512& c0, const __m512& c1)
{
    if constexpr (CDT == VG_F32)
    {
        float* out = reinterpret_cast<float*>(crow);
        __m512 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + n0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + n0 + 16));
        }
        _mm512_storeu_ps(out, v0);
        _mm512_storeu_ps(out + 16, v1);
    }
    else if constexpr (CDT == VG_F16)
    {
        __m512 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + n0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + n0 + 16));
        }
        __m128i* out = reinterpret_cast<__m128i*>(crow);
        const __m512 vs[2] = {v0, v1};
        for (int j = 0; j < 2; ++j)
        {
            _mm_storeu_si128(out + j * 2, _mm256_cvtps_ph(
                _mm512_extractf32x8_ps(vs[j], 0), _MM_FROUND_TO_NEAREST_INT));
            _mm_storeu_si128(out + j * 2 + 1, _mm256_cvtps_ph(
                _mm512_extractf32x8_ps(vs[j], 1), _MM_FROUND_TO_NEAREST_INT));
        }
    }
    else
    {
        alignas(64) float tmp[32];
        __m512 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + n0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + n0 + 16));
        }
        _mm512_storeu_ps(tmp, v0);
        _mm512_storeu_ps(tmp + 16, v1);
        for (int j = 0; j < 32; ++j)
            vg_store_scalar(crow + static_cast<size_t>(j) * 2, VG_BF16, tmp[j]);
    }
}

template <VgDtype ADT, VgDtype BDT, VgDtype CDT>
EXL3_CPU_TARGET_AVX512
void vg_block512(const VgCtx& c, int64_t m0, int mr, int64_t n0)
{
    // 8 rows x 32 cols. Halving the width (vs 4x64) halves B-panel re-reads per
    // output element; B traffic dominates once panels spill out of L3 between the
    // tower's back-to-back GEMMs. The k-step still walks full 64-wide panel rows
    // (bnz) from the 32-wide in-panel offset; A re-reads double but stream.
    constexpr int ASZ = ADT == VG_F32 ? 4 : 2;
    constexpr int BSZ = BDT == VG_F32 ? 4 : 2;
    constexpr int CSZ = CDT == VG_F32 ? 4 : 2;
    // A row pointers, resolved once per block; dead M-tail lanes reuse the last valid
    // row (their accumulators are discarded, so values only need to be in-bounds)
    const char* ap[8];
    for (int r = 0; r < 8; ++r)
    {
        const int64_t rr = (r < mr) ? m0 + r : c.M - 1;
        ap[r] = c.A + static_cast<size_t>(rr) * c.K * ASZ;
    }
    // Blocked B panel (n0 / 64) at in-panel offset (n0 % 64); the k-step walks it
    // sequentially with a full-panel-row stride
    const char* bp = c.B + (static_cast<size_t>(n0 / VG_R) * c.K * VG_R +
                            static_cast<size_t>(n0 % VG_R)) * BSZ;
    const size_t bnz = VG_R * BSZ;

    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
    __m512 c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
    __m512 c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();

    for (int64_t k = 0; k < c.K; ++k)
    {
        __m512 b0, b1;
        vg_ld_b512<BDT>(bp, b0, b1);
        bp += bnz;
        vg_row512<ADT>(ap[0], b0, b1, c00, c01);
        vg_row512<ADT>(ap[1], b0, b1, c10, c11);
        vg_row512<ADT>(ap[2], b0, b1, c20, c21);
        vg_row512<ADT>(ap[3], b0, b1, c30, c31);
        vg_row512<ADT>(ap[4], b0, b1, c40, c41);
        vg_row512<ADT>(ap[5], b0, b1, c50, c51);
        vg_row512<ADT>(ap[6], b0, b1, c60, c61);
        vg_row512<ADT>(ap[7], b0, b1, c70, c71);
    }

    if (mr > 0) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0) * c.N + n0) * CSZ,
                               c.bias, n0, c00, c01);
    if (mr > 1) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0 + 1) * c.N + n0) * CSZ,
                               c.bias, n0, c10, c11);
    if (mr > 2) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0 + 2) * c.N + n0) * CSZ,
                               c.bias, n0, c20, c21);
    if (mr > 3) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0 + 3) * c.N + n0) * CSZ,
                               c.bias, n0, c30, c31);
    if (mr > 4) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0 + 4) * c.N + n0) * CSZ,
                               c.bias, n0, c40, c41);
    if (mr > 5) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0 + 5) * c.N + n0) * CSZ,
                               c.bias, n0, c50, c51);
    if (mr > 6) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0 + 6) * c.N + n0) * CSZ,
                               c.bias, n0, c60, c61);
    if (mr > 7) vg_epi512<CDT>(c.C + (static_cast<size_t>(m0 + 7) * c.N + n0) * CSZ,
                               c.bias, n0, c70, c71);
}

// -------------------------------------------------------------------------------------------
// AVX-512 dpbf16 path: 8 rows x 64 cols, bf16 x bf16 -> fp32
// -------------------------------------------------------------------------------------------

template <VgDtype ADT>
VG_DP_TARGET
inline void vg_pack_a(const char* arow, int64_t k0, int64_t k1, char* panel)
{
    // Pack one A row segment [k0, k1) as dense bf16; runs once per (block, kc-panel)
    // and is reused across all 64 columns, so plain vector steady state + scalar tail
    if constexpr (ADT == VG_BF16)
    {
        std::memcpy(panel, arow + static_cast<size_t>(k0) * 2,
                    static_cast<size_t>(k1 - k0) * 2);
    }
    else
    {
        static_assert(ADT == VG_F16);
        int64_t k = k0;
        for (; k + 32 <= k1; k += 32)
        {
            const __m256i* src =
                reinterpret_cast<const __m256i*>(arow + static_cast<size_t>(k) * 2);
            __m256i* dst = reinterpret_cast<__m256i*>(
                panel + static_cast<size_t>(k - k0) * 2);
            const __m512 f0 = _mm512_cvtph_ps(_mm256_loadu_si256(src));
            const __m512 f1 = _mm512_cvtph_ps(_mm256_loadu_si256(src + 1));
            const __m256bh o0 = _mm512_cvtneps_pbh(f0);
            const __m256bh o1 = _mm512_cvtneps_pbh(f1);
            std::memcpy(dst, &o0, 32);
            std::memcpy(dst + 1, &o1, 32);
        }
        for (; k < k1; ++k)
        {
            uint16_t b;
            std::memcpy(&b, arow + static_cast<size_t>(k) * 2, 2);
            const float f = exl3_common::half_to_float(b);
            uint32_t u;
            std::memcpy(&u, &f, 4);
            u += 0x7FFFu + ((u >> 16) & 1u);
            const uint16_t o = static_cast<uint16_t>(u >> 16);
            std::memcpy(panel + static_cast<size_t>(k - k0) * 2, &o, 2);
        }
    }
}

// Bit-cast helper: GCC 11 provides __m512bh and _mm512_dpbf16_ps but no si512->pbh
// cast intrinsic; memcpy is exactly equivalent and optimizes away
VG_DP_TARGET
inline __m512bh vg_to_bh(__m512i v)
{
    __m512bh r;
    std::memcpy(&r, &v, 64);
    return r;
}
template <VgDtype CDT>
VG_DP_TARGET
inline void vg_epi_dp32(char* crow, const float* bias, int64_t n0,
                        const __m512& c0, const __m512& c1)
{
    if constexpr (CDT == VG_F32)
    {
        float* out = reinterpret_cast<float*>(crow);
        __m512 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + n0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + n0 + 16));
        }
        _mm512_storeu_ps(out, v0);
        _mm512_storeu_ps(out + 16, v1);
    }
    else if constexpr (CDT == VG_F16)
    {
        __m512 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + n0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + n0 + 16));
        }
        __m128i* out = reinterpret_cast<__m128i*>(crow);
        _mm_storeu_si128(out, _mm256_cvtps_ph(
            _mm512_extractf32x8_ps(v0, 0), _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(out + 1, _mm256_cvtps_ph(
            _mm512_extractf32x8_ps(v0, 1), _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(out + 2, _mm256_cvtps_ph(
            _mm512_extractf32x8_ps(v1, 0), _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(out + 3, _mm256_cvtps_ph(
            _mm512_extractf32x8_ps(v1, 1), _MM_FROUND_TO_NEAREST_INT));
    }
    else
    {
        alignas(64) float tmp[32];
        __m512 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm512_add_ps(v0, _mm512_loadu_ps(bias + n0));
            v1 = _mm512_add_ps(v1, _mm512_loadu_ps(bias + n0 + 16));
        }
        _mm512_storeu_ps(tmp, v0);
        _mm512_storeu_ps(tmp + 16, v1);
        for (int j = 0; j < 32; ++j)
            vg_store_scalar(crow + static_cast<size_t>(j) * 2, VG_BF16, tmp[j]);
    }
}

// dpbf16 micro-kernel: k advances in pairs. Lane i of _mm512_dpbf16_ps computes
//   dst[i] = s1[2i]*s2[2i] + s1[2i+1]*s2[2i+1] + dst[i],
// so with the A pair (A[m][k], A[m][k+1]) broadcast into every dword of s1 and the
// B pair (B[k][n+i], B[k+1][n+i]) interleaved into s2, one instruction performs the
// length-2 partial dot for 16 columns. Two slabs of 32 columns cover the 64-wide block.
template <VgDtype ADT, VgDtype CDT>
VG_DP_TARGET
void vg_block_dp(const VgCtx& c, int64_t m0, int mr, int64_t n0)
{
    static_assert(ADT == VG_F16 || ADT == VG_BF16);
    constexpr int CSZ = CDT == VG_F32 ? 4 : 2;
    const char* arow[8];
    for (int r = 0; r < 8; ++r)
    {
        const int64_t rr = (r < mr) ? m0 + r : c.M - 1;
        arow[r] = c.A + static_cast<size_t>(rr) * c.K * 2;
    }
    const char* bbase = c.B + static_cast<size_t>(n0 / VG_R) * c.K * VG_R * 2;

    // A-pack: 8 rows x KC bf16, L1-resident, shared by both slabs
    alignas(64) char apack[8 * VG_KC * 2];
    const __m512i idx_lo = _mm512_loadu_si512(VG_IDX_LO);
    const __m512i idx_hi = _mm512_loadu_si512(VG_IDX_HI);
    for (int s = 0; s < 2; ++s)
    {
        __m512 e00 = _mm512_setzero_ps(), e01 = _mm512_setzero_ps();
        __m512 e10 = _mm512_setzero_ps(), e11 = _mm512_setzero_ps();
        __m512 e20 = _mm512_setzero_ps(), e21 = _mm512_setzero_ps();
        __m512 e30 = _mm512_setzero_ps(), e31 = _mm512_setzero_ps();
        __m512 e40 = _mm512_setzero_ps(), e41 = _mm512_setzero_ps();
        __m512 e50 = _mm512_setzero_ps(), e51 = _mm512_setzero_ps();
        __m512 e60 = _mm512_setzero_ps(), e61 = _mm512_setzero_ps();
        __m512 e70 = _mm512_setzero_ps(), e71 = _mm512_setzero_ps();
        for (int64_t k0 = 0; k0 < c.K; k0 += VG_KC)
        {
            const int64_t k1 = std::min<int64_t>(c.K, k0 + VG_KC);
            for (int r = 0; r < 8; ++r)
                vg_pack_a<ADT>(arow[r], k0, k1, apack + static_cast<size_t>(r) * VG_KC * 2);
            const char* bb = bbase + static_cast<size_t>(k0) * VG_R * 2 +
                             static_cast<size_t>(s) * 32 * 2;
            for (int64_t k = k0; k < k1; k += 2)
            {
                const __m512i bk = _mm512_loadu_si512(bb);
                const __m512i bk1 = _mm512_loadu_si512(
                    bb + static_cast<size_t>(VG_R) * 2);
                const __m512bh blo = vg_to_bh(
                    _mm512_permutex2var_epi16(bk, idx_lo, bk1));
                const __m512bh bhi = vg_to_bh(
                    _mm512_permutex2var_epi16(bk, idx_hi, bk1));
                bb += static_cast<size_t>(2) * VG_R * 2;
#define VG_DP_ROW(r, e0, e1) \
    do { \
        int32_t pair; \
        std::memcpy(&pair, apack + static_cast<size_t>(r) * VG_KC * 2 + \
                    static_cast<size_t>(k - k0) * 2, 4); \
        const __m512bh a = vg_to_bh(_mm512_set1_epi32(pair)); \
        e0 = _mm512_dpbf16_ps(e0, a, blo); \
        e1 = _mm512_dpbf16_ps(e1, a, bhi); \
    } while (0)
                VG_DP_ROW(0, e00, e01); VG_DP_ROW(1, e10, e11);
                VG_DP_ROW(2, e20, e21); VG_DP_ROW(3, e30, e31);
                VG_DP_ROW(4, e40, e41); VG_DP_ROW(5, e50, e51);
                VG_DP_ROW(6, e60, e61); VG_DP_ROW(7, e70, e71);
#undef VG_DP_ROW
            }
        }
        const int64_t ns = n0 + s * 32;
        if (mr > 0) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0) * c.N + ns) * CSZ,
                                     c.bias, ns, e00, e01);
        if (mr > 1) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0 + 1) * c.N + ns) * CSZ,
                                     c.bias, ns, e10, e11);
        if (mr > 2) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0 + 2) * c.N + ns) * CSZ,
                                     c.bias, ns, e20, e21);
        if (mr > 3) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0 + 3) * c.N + ns) * CSZ,
                                     c.bias, ns, e30, e31);
        if (mr > 4) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0 + 4) * c.N + ns) * CSZ,
                                     c.bias, ns, e40, e41);
        if (mr > 5) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0 + 5) * c.N + ns) * CSZ,
                                     c.bias, ns, e50, e51);
        if (mr > 6) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0 + 6) * c.N + ns) * CSZ,
                                     c.bias, ns, e60, e61);
        if (mr > 7) vg_epi_dp32<CDT>(c.C + (static_cast<size_t>(m0 + 7) * c.N + ns) * CSZ,
                                     c.bias, ns, e70, e71);
    }
}

// -------------------------------------------------------------------------------------------
// AVX2 path: 4 rows x 16 cols
// -------------------------------------------------------------------------------------------

template <VgDtype ADT>
EXL3_CPU_TARGET_AVX2
inline float vg_ld_a256(const char*& ap)
{
    float f;
    if constexpr (ADT == VG_F32)
    {
        std::memcpy(&f, ap, 4);
        ap += 4;
        return f;
    }
    uint16_t b;
    std::memcpy(&b, ap, 2);
    ap += 2;
    if constexpr (ADT == VG_F16) return exl3_common::half_to_float(b);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    std::memcpy(&f, &u, 4);
    return f;
}

template <VgDtype BDT>
EXL3_CPU_TARGET_AVX2
inline void vg_ld_b256(const char* bp, __m256& b0, __m256& b1)
{
    if constexpr (BDT == VG_F32)
    {
        const float* p = reinterpret_cast<const float*>(bp);
        b0 = _mm256_loadu_ps(p);
        b1 = _mm256_loadu_ps(p + 8);
    }
    else if constexpr (BDT == VG_F16)
    {
        const __m128i* p = reinterpret_cast<const __m128i*>(bp);
        b0 = _mm256_cvtph_ps(_mm_loadu_si128(p));
        b1 = _mm256_cvtph_ps(_mm_loadu_si128(p + 1));
    }
    else
    {
        // 8x bf16 -> 8x fp32 per 16-byte lane (exact shift, no BF16 ext needed)
        const __m128i* p = reinterpret_cast<const __m128i*>(bp);
        const __m128i v0 = _mm_loadu_si128(p);
        const __m128i v1 = _mm_loadu_si128(p + 1);
        b0 = _mm256_castsi256_ps(_mm256_set_m128i(
            _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(v0, 8)), 16),
            _mm_slli_epi32(_mm_cvtepu16_epi32(v0), 16)));
        b1 = _mm256_castsi256_ps(_mm256_set_m128i(
            _mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(v1, 8)), 16),
            _mm_slli_epi32(_mm_cvtepu16_epi32(v1), 16)));
    }
}

template <VgDtype ADT>
EXL3_CPU_TARGET_AVX2
inline void vg_row256(const char*& ap, const __m256& b0, const __m256& b1,
                      __m256& c0, __m256& c1)
{
    const __m256 a = _mm256_set1_ps(vg_ld_a256<ADT>(ap));
    c0 = _mm256_fmadd_ps(a, b0, c0);
    c1 = _mm256_fmadd_ps(a, b1, c1);
}

template <VgDtype CDT>
EXL3_CPU_TARGET_AVX2
inline void vg_epi256(char* crow, const float* bias, int64_t n0,
                      const __m256& c0, const __m256& c1)
{
    if constexpr (CDT == VG_F32)
    {
        float* out = reinterpret_cast<float*>(crow);
        __m256 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(bias + n0));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(bias + n0 + 8));
        }
        _mm256_storeu_ps(out, v0);
        _mm256_storeu_ps(out + 8, v1);
    }
    else if constexpr (CDT == VG_F16)
    {
        __m256 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(bias + n0));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(bias + n0 + 8));
        }
        __m128i* out = reinterpret_cast<__m128i*>(crow);
        _mm_storeu_si128(out, _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT));
        _mm_storeu_si128(out + 1, _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT));
    }
    else
    {
        alignas(32) float tmp[16];
        __m256 v0 = c0, v1 = c1;
        if (bias)
        {
            v0 = _mm256_add_ps(v0, _mm256_loadu_ps(bias + n0));
            v1 = _mm256_add_ps(v1, _mm256_loadu_ps(bias + n0 + 8));
        }
        _mm256_storeu_ps(tmp, v0);
        _mm256_storeu_ps(tmp + 8, v1);
        for (int j = 0; j < 16; ++j)
            vg_store_scalar(crow + static_cast<size_t>(j) * 2, VG_BF16, tmp[j]);
    }
}

template <VgDtype ADT, VgDtype BDT, VgDtype CDT>
EXL3_CPU_TARGET_AVX2
void vg_block256(const VgCtx& c, int64_t m0, int mr, int64_t n0)
{
    constexpr int ASZ = ADT == VG_F32 ? 4 : 2;
    constexpr int BSZ = BDT == VG_F32 ? 4 : 2;
    constexpr int CSZ = CDT == VG_F32 ? 4 : 2;
    const char* ap[4];
    for (int r = 0; r < 4; ++r)
    {
        const int64_t rr = (r < mr) ? m0 + r : c.M - 1;
        ap[r] = c.A + static_cast<size_t>(rr) * c.K * ASZ;
    }
    // Blocked B, 16-wide quarter of panel (n0 / 64) at in-panel offset (n0 % 64)
    const char* bp = c.B + (static_cast<size_t>(n0 / VG_R) * c.K * VG_R + (n0 % VG_R)) * BSZ;
    const size_t bnz = 16 * BSZ;
    (void) bnz;

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

    // NOTE: bnz stepping walks panel quarters sequentially only while the 16-wide
    // window stays inside one K row of the panel; the panel row is 64 wide and this
    // block covers 16 of it, so advance per k by the full panel row instead and
    // re-derive the quarter base (keeps every access in-bounds by construction)
    const size_t brow = static_cast<size_t>(VG_R) * BSZ;
    for (int64_t k = 0; k < c.K; ++k)
    {
        __m256 b0, b1;
        vg_ld_b256<BDT>(bp, b0, b1);
        bp += brow;
        vg_row256<ADT>(ap[0], b0, b1, c00, c01);
        vg_row256<ADT>(ap[1], b0, b1, c10, c11);
        vg_row256<ADT>(ap[2], b0, b1, c20, c21);
        vg_row256<ADT>(ap[3], b0, b1, c30, c31);
    }

    if (mr > 0) vg_epi256<CDT>(c.C + (static_cast<size_t>(m0) * c.N + n0) * CSZ,
                               c.bias, n0, c00, c01);
    if (mr > 1) vg_epi256<CDT>(c.C + (static_cast<size_t>(m0 + 1) * c.N + n0) * CSZ,
                               c.bias, n0, c10, c11);
    if (mr > 2) vg_epi256<CDT>(c.C + (static_cast<size_t>(m0 + 2) * c.N + n0) * CSZ,
                               c.bias, n0, c20, c21);
    if (mr > 3) vg_epi256<CDT>(c.C + (static_cast<size_t>(m0 + 3) * c.N + n0) * CSZ,
                               c.bias, n0, c30, c31);
}

// Per-ISA runners: dtype dispatch resolves once per call site (vg_tile_fn), so the
// block loops below stay monomorphic. N-outer keeps each B panel L2-resident across
// all M blocks of the tile (M-outer re-read it from L3 per block, ~10x slower).
template <VgDtype ADT, VgDtype BDT, VgDtype CDT>
EXL3_CPU_TARGET_AVX512
void vg_run512(const VgCtx& c, int64_t m0, int64_t m1, int64_t n0, int64_t n1)
{
    // 8x32 blocks; 32-wide halves complete in-panel or stop at the 64 panel edge
    // (all vision N are multiples of 64, so the scalar tail only guards odd tiles).
    // Half/bf16 A is widened to fp32 once per tile row (vector pack, bit-exact)
    // and reused across all n-blocks; fp32 A runs directly from the input.
    if constexpr (ADT == VG_F32)
    {
        int64_t n = n0;
        for (; n + 32 <= n1; n += 32)
            for (int64_t m = m0; m < m1; m += 8)
                vg_block512<ADT, BDT, CDT>(c, m,
                    static_cast<int>(std::min<int64_t>(8, m1 - m)), n);
        if (n < n1)
            for (int64_t m = m0; m < m1; m += 8)
                vg_scalar_tile(c, m, std::min(m + 8, m1), n, n1);
        return;
    }
    else
    {
    thread_local std::vector<char> apack;
    const size_t need = static_cast<size_t>(m1 - m0) * static_cast<size_t>(c.K) * 4;
    if (apack.size() < need) apack.reserve(need), apack.resize(need);
    float* P = reinterpret_cast<float*>(apack.data());
    for (int64_t m = m0; m < m1; ++m)
        vg_pack_a_f32<ADT>(c.A + static_cast<size_t>(m) * c.K * c.aesz,
                           c.K, P + (m - m0) * c.K);
    // Shallow copy retargeted at the pack; rows go relative, C stays absolute via
    // the row base shift, everything else (B, bias, K, N) is unchanged.
    VgCtx cp = c;
    cp.A = reinterpret_cast<const char*>(P);
    cp.C = c.C + static_cast<size_t>(m0) * c.N * c.cesz;
    cp.M = m1 - m0;
    cp.adt = VG_F32;
    cp.aesz = 4;
    const int64_t r1 = m1 - m0;
    int64_t n = n0;
    for (; n + 32 <= n1; n += 32)
        for (int64_t m = 0; m < r1; m += 8)
            vg_block512<VG_F32, BDT, CDT>(cp, m,
                static_cast<int>(std::min<int64_t>(8, r1 - m)), n);
    if (n < n1)
        for (int64_t m = 0; m < r1; m += 8)
            vg_scalar_tile(cp, m, std::min(m + 8, r1), n, n1);
    }
}

template <VgDtype ADT, VgDtype CDT>
EXL3_CPU_TARGET_AVX512
void vg_run_dp(const VgCtx& c, int64_t m0, int64_t m1, int64_t n0, int64_t n1)
{
    int64_t n = n0;
    for (; n + VG_R <= n1; n += VG_R)
        for (int64_t m = m0; m < m1; m += 8)
            vg_block_dp<ADT, CDT>(c, m,
                static_cast<int>(std::min<int64_t>(8, m1 - m)), n);
    if (n < n1)
        for (int64_t m = m0; m < m1; m += 8)
            vg_scalar_tile(c, m, std::min(m + 8, m1), n, n1);
}

template <VgDtype ADT, VgDtype BDT, VgDtype CDT>
EXL3_CPU_TARGET_AVX2
void vg_run256(const VgCtx& c, int64_t m0, int64_t m1, int64_t n0, int64_t n1)
{
    int64_t n = n0;
    for (; n + 16 <= n1; n += 16)
        for (int64_t m = m0; m < m1; m += 4)
            vg_block256<ADT, BDT, CDT>(c, m,
                static_cast<int>(std::min<int64_t>(4, m1 - m)), n);
    if (n < n1)
        for (int64_t m = m0; m < m1; m += 4)
            vg_scalar_tile(c, m, std::min(m + 4, m1), n, n1);
}

void vg_run_scalar(const VgCtx& c, int64_t m0, int64_t m1, int64_t n0, int64_t n1)
{
    vg_scalar_tile(c, m0, m1, n0, n1);
}

// Route + dtype dispatch per tile (tiles are large; the switch cost is negligible).
// The dpbf16 route needs bf16 B, half/bf16 A, and AVX512-BF16 hardware; everything else
// covered goes through the FMA path on AVX-512.
template <VgIsa ISA>
void vg_tile_typed(const VgCtx& c, int64_t m0, int64_t m1, int64_t n0, int64_t n1)
{
    if constexpr (ISA == VG_AVX512)
    {
        if (c.use_dp)
        {
            if (c.adt == VG_F16)
            {
                switch (c.cdt)
                {
                    case VG_F32: vg_run_dp<VG_F16, VG_F32>(c, m0, m1, n0, n1); break;
                    case VG_F16: vg_run_dp<VG_F16, VG_F16>(c, m0, m1, n0, n1); break;
                    default: vg_run_dp<VG_F16, VG_BF16>(c, m0, m1, n0, n1); break;
                }
            }
            else
            {
                switch (c.cdt)
                {
                    case VG_F32: vg_run_dp<VG_BF16, VG_F32>(c, m0, m1, n0, n1); break;
                    case VG_F16: vg_run_dp<VG_BF16, VG_F16>(c, m0, m1, n0, n1); break;
                    default: vg_run_dp<VG_BF16, VG_BF16>(c, m0, m1, n0, n1); break;
                }
            }
            return;
        }
#define VG_CALL_C(ADT, BDT, CDT) vg_run512<ADT, BDT, CDT>(c, m0, m1, n0, n1);
#define VG_SWITCH_C(ADT, BDT) \
    switch (c.cdt) \
    { \
        case VG_F32: VG_CALL_C(ADT, BDT, VG_F32); break; \
        case VG_F16: VG_CALL_C(ADT, BDT, VG_F16); break; \
        default: VG_CALL_C(ADT, BDT, VG_BF16); break; \
    }
#define VG_SWITCH_B(ADT) \
    switch (c.bdt) \
    { \
        case VG_F32: VG_SWITCH_C(ADT, VG_F32); break; \
        case VG_F16: VG_SWITCH_C(ADT, VG_F16); break; \
        default: VG_SWITCH_C(ADT, VG_BF16); break; \
    }
        switch (c.adt)
        {
            case VG_F32: VG_SWITCH_B(VG_F32); break;
            case VG_F16: VG_SWITCH_B(VG_F16); break;
            default: VG_SWITCH_B(VG_BF16); break;
        }
#undef VG_CALL_C
#undef VG_SWITCH_C
#undef VG_SWITCH_B
    }
    else
    {
#define VG_CALL_C(ADT, BDT, CDT) vg_run256<ADT, BDT, CDT>(c, m0, m1, n0, n1);
#define VG_SWITCH_C(ADT, BDT) \
    switch (c.cdt) \
    { \
        case VG_F32: VG_CALL_C(ADT, BDT, VG_F32); break; \
        case VG_F16: VG_CALL_C(ADT, BDT, VG_F16); break; \
        default: VG_CALL_C(ADT, BDT, VG_BF16); break; \
    }
#define VG_SWITCH_B(ADT) \
    switch (c.bdt) \
    { \
        case VG_F32: VG_SWITCH_C(ADT, VG_F32); break; \
        case VG_F16: VG_SWITCH_C(ADT, VG_F16); break; \
        default: VG_SWITCH_C(ADT, VG_BF16); break; \
    }
        switch (c.adt)
        {
            case VG_F32: VG_SWITCH_B(VG_F32); break;
            case VG_F16: VG_SWITCH_B(VG_F16); break;
            default: VG_SWITCH_B(VG_BF16); break;
        }
#undef VG_CALL_C
#undef VG_SWITCH_C
#undef VG_SWITCH_B
    }
}

void vg_tile_fn(void* ctx, int64_t t, int64_t t_end)
{
    (void) t_end;
    auto* c = static_cast<VgCtx*>(ctx);
    const int64_t n_mc = (c->M + VG_MC - 1) / VG_MC;
    const int64_t n_nc = (c->N + VG_NC - 1) / VG_NC;
    const int64_t mc = t / n_nc;
    const int64_t nc = t % n_nc;
    const int64_t m0 = mc * VG_MC;
    const int64_t m1 = std::min(c->M, m0 + VG_MC);
    const int64_t n0 = nc * VG_NC;
    const int64_t n1 = std::min(c->N, n0 + VG_NC);
    if (c->isa == VG_AVX512) vg_tile_typed<VG_AVX512>(*c, m0, m1, n0, n1);
    else if (c->isa == VG_AVX2) vg_tile_typed<VG_AVX2>(*c, m0, m1, n0, n1);
    else vg_run_scalar(*c, m0, m1, n0, n1);
}

} // namespace

void vision_gemm_blocked_cpu(
    at::Tensor x,
    at::Tensor wb,
    c10::optional<at::Tensor> bias,
    at::Tensor out)
{
    TORCH_CHECK(x.device().is_cpu() && wb.device().is_cpu() && out.device().is_cpu(),
                "vision_gemm_blocked_cpu: CPU tensors only");
    TORCH_CHECK(x.dim() == 2 && wb.dim() == 3 && out.dim() == 2,
                "vision_gemm_blocked_cpu: expected [M, K] x [NB, K, 64] -> [M, N]");
    TORCH_CHECK(x.is_contiguous() && wb.is_contiguous() && out.is_contiguous(),
                "vision_gemm_blocked_cpu: contiguous only");
    TORCH_CHECK(x.is_floating_point() && wb.is_floating_point() && out.is_floating_point(),
                "vision_gemm_blocked_cpu: fp only");
    const int64_t M = x.size(0), K = x.size(1), N = out.size(1);
    TORCH_CHECK(wb.size(1) == K && wb.size(2) == VG_R, "vision_gemm_blocked_cpu: panel shape");
    TORCH_CHECK(out.size(0) == M, "vision_gemm_blocked_cpu: out shape");
    TORCH_CHECK(wb.size(0) * VG_R >= N, "vision_gemm_blocked_cpu: panels cover N");
    TORCH_CHECK(K > 0 && N > 0, "vision_gemm_blocked_cpu: empty reduction/output dim");
    TORCH_CHECK(out.scalar_type() == at::kFloat || out.scalar_type() == at::kHalf ||
                out.scalar_type() == at::kBFloat16, "vision_gemm_blocked_cpu: out fp32/fp16/bf16");
    if (M == 0) return;

    at::Tensor bias32;
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined())
    {
        TORCH_CHECK(bias->device().is_cpu() && bias->is_floating_point(), "vision_gemm_blocked_cpu: fp CPU bias");
        TORCH_CHECK(bias->numel() == N, "vision_gemm_blocked_cpu: bias must be [N]");
        bias32 = bias->to(at::kFloat).contiguous();
        bias_ptr = bias32.data_ptr<float>();
    }

    VgCtx c;
    c.A = static_cast<const char*>(x.const_data_ptr());
    c.B = static_cast<const char*>(wb.const_data_ptr());
    c.bias = bias_ptr;
    c.C = static_cast<char*>(out.data_ptr());
    c.M = M;
    c.K = K;
    c.N = N;
    c.adt = vg_dtype(x.scalar_type());
    c.bdt = vg_dtype(wb.scalar_type());
    c.cdt = vg_dtype(out.scalar_type());
    c.aesz = vg_esz(c.adt);
    c.besz = vg_esz(c.bdt);
    c.cesz = vg_esz(c.cdt);
    c.isa = vg_pick_isa();
    // dpbf16 consumes k in broadcast pairs; odd K takes the FMA path (all vision K even)
    c.use_dp = c.isa == VG_AVX512 && c.bdt == VG_BF16 &&
               (c.adt == VG_F16 || c.adt == VG_BF16) && (c.K % 2 == 0) && vg_has_bf16();

    const int64_t total = ((M + VG_MC - 1) / VG_MC) * ((N + VG_NC - 1) / VG_NC);
    cpu_pool_run_tiles(total, &vg_tile_fn, &c);
}
