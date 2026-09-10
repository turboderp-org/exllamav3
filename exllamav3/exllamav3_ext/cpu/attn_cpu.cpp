// Fused attention for CPU tensors: dense/varlen QK^T + softmax + AV in fp32 with a
// single round to the output dtype.
//
// Vector structure follows the repo's CPU idiom (act_mul_cpu.cpp, rms_norm_cpu.cpp,
// rope_cpu.cpp): target-attributed kernels per width, helpers called inside a kernel
// carry the same target attribute (GCC will not inline a baseline-ISA callee into an
// AVX-512 caller), f16c half conversion, integer-shift bf16 widening with RNE pack
// stores. Algorithm ordering mirrors ggml's ggml_compute_forward_flash_attn_ext
// one-chunk path (per-query online softmax with sinks/softcap/mask, ggml/src/ggml-cpu/
// ops.cpp): scores in fp32, max + exp-sum normalization, probability-weighted V
// accumulation. Reimplemented here for the repo's conventions (two-pass scores buffer
// instead of online V rescaling, which keeps the hot AV loop streaming and lets the
// softmax use the same vector exp as act_mul); no vendored code. The vector exp is
// the same range-reduction polynomial family as ggml's ggml_v_expf (~1.5 ulps).

#include "attn_cpu.h"
#include "exl3_common.h"
#include "moe_mul1.h"

#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

enum AcDt : uint8_t { AC_F32, AC_F16, AC_BF16 };
enum AcIsa : uint8_t { AC_SCALAR, AC_AVX2, AC_AVX512 };

constexpr int64_t AC_TOKEN_CHUNK = 16;
constexpr int AC_RND = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

AcDt ac_dtype(at::ScalarType t)
{
    if (t == at::kFloat) return AC_F32;
    if (t == at::kHalf) return AC_F16;
    return AC_BF16;
}

AcIsa ac_pick_isa()
{
    static const AcIsa v = [] {
        if (const char* e = std::getenv("EXL3_ATTN_CPU_ISA"))
        {
            if (std::strcmp(e, "scalar") == 0) return AC_SCALAR;
            if (std::strcmp(e, "avx2") == 0) return AC_AVX2;
            if (std::strcmp(e, "avx512") == 0) return AC_AVX512;
        }
        if (exl3_moe_cpu_has_avx512()) return AC_AVX512;
        if (exl3_moe_cpu_has_avx2()) return AC_AVX2;
        return AC_SCALAR;
    }();
    return v;
}

// -------------------------------------------------------------------------------------------
// Scalar lane conversion (tails, bf16 outputs, scalar fallback)
// -------------------------------------------------------------------------------------------

inline float ac_load_s(const char* p, AcDt dt)
{
    if (dt == AC_F32)
    {
        float f;
        std::memcpy(&f, p, 4);
        return f;
    }
    uint16_t b;
    std::memcpy(&b, p, 2);
    if (dt == AC_F16) return exl3_common::half_to_float(b);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline void ac_store_s(char* p, AcDt dt, float f)
{
    if (dt == AC_F32)
    {
        std::memcpy(p, &f, 4);
        return;
    }
    if (dt == AC_F16)
    {
        const c10::Half h(f);
        const uint16_t b = h.x;
        std::memcpy(p, &b, 2);
        return;
    }
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    const uint16_t b = static_cast<uint16_t>(u >> 16);
    std::memcpy(p, &b, 2);
}

// -------------------------------------------------------------------------------------------
// Vector loads (f32/f16/bf16 -> f32 vector)
// -------------------------------------------------------------------------------------------

template <AcDt DT>
EXL3_CPU_TARGET_AVX2
inline __m256 ac_ld8(const char* p)
{
    if constexpr (DT == AC_F32) return _mm256_loadu_ps((const float*)p);
    if constexpr (DT == AC_F16) return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)p));
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16));
}

template <AcDt DT>
EXL3_CPU_TARGET_AVX512
inline __m512 ac_ld16(const char* p)
{
    if constexpr (DT == AC_F32) return _mm512_loadu_ps(p);
    if constexpr (DT == AC_F16) return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)p));
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}

EXL3_CPU_TARGET_AVX2
inline void ac_st8_f16(char* p, __m256 v)
{
    _mm_storeu_si128((__m128i*)p, _mm256_cvtps_ph(v, AC_RND));
}

EXL3_CPU_TARGET_AVX512
inline void ac_st16_f16(char* p, __m512 v)
{
    _mm256_storeu_si256((__m256i*)p, _mm512_cvtps_ph(v, AC_RND));
}

// -------------------------------------------------------------------------------------------
// Vector exp, same polynomial family as act_mul (ggml_v_expf class).
// -------------------------------------------------------------------------------------------

EXL3_CPU_TARGET_AVX2
inline __m256 ac_exp8(__m256 x)
{
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0f))));
    const __m256 ax = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), n);
    const __m256i big = _mm256_castps_si256(
        _mm256_cmp_ps(ax, _mm256_set1_ps(126.0f), _CMP_GT_OQ));
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                        _mm256_set1_ps(0x1.573e2ep-5f)), u,
                        _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                        _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm256_movemask_ps(_mm256_castsi256_ps(big)))
        return _mm256_fmadd_ps(j, k, k);
    const __m256i neg = _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ));
    const __m256i gi = _mm256_and_si256(neg, _mm256_set1_epi32(0x82000000));
    const __m256 s1 = _mm256_castsi256_ps(
        _mm256_add_epi32(gi, _mm256_set1_epi32(0x7f000000)));
    const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, gi));
    const __m256i huge = _mm256_castps_si256(
        _mm256_cmp_ps(ax, _mm256_set1_ps(192.0f), _CMP_GT_OQ));
    const __m256 scaled = _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1);
    const __m256 normal = _mm256_fmadd_ps(k, j, k);
    const __m256 infzero = _mm256_mul_ps(s1, s1);
    return _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(huge), infzero),
                        _mm256_andnot_ps(_mm256_castsi256_ps(huge),
                                         _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(big), scaled),
                                                      _mm256_andnot_ps(_mm256_castsi256_ps(big), normal))));
}

EXL3_CPU_TARGET_AVX512
inline __m512 ac_exp16(__m512 x)
{
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                                      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __m512i e = _mm512_slli_epi32(_mm512_castps_si512(z), 23);
    const __m512 k = _mm512_castsi512_ps(
        _mm512_add_epi32(e, _mm512_castps_si512(_mm512_set1_ps(1.0f))));
    const __m512 ax = _mm512_castsi512_ps(
        _mm512_andnot_epi32(_mm512_castps_si512(_mm512_set1_ps(-0.0f)),
                            _mm512_castps_si512(n)));
    const __mmask16 big = _mm512_cmp_ps_mask(ax, _mm512_set1_ps(126.0f), _CMP_GT_OQ);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                        _mm512_set1_ps(0x1.573e2ep-5f)), u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                        _mm512_set1_ps(0x1.fffdb6p-2f))),
        u, _mm512_mul_ps(_mm512_set1_ps(0x1.ffffecp-1f), b));
    const __mmask16 negm = _mm512_cmp_ps_mask(n, _mm512_setzero_ps(), _CMP_LE_OQ);
    const __m512i gi = _mm512_maskz_set1_epi32(negm, 0x82000000);
    const __m512 s1 = _mm512_castsi512_ps(
        _mm512_add_epi32(gi, _mm512_set1_epi32(0x7f000000)));
    const __m512 s2 = _mm512_castsi512_ps(_mm512_sub_epi32(e, gi));
    const __mmask16 huge = _mm512_cmp_ps_mask(ax, _mm512_set1_ps(192.0f), _CMP_GT_OQ);
    const __m512 scaled = _mm512_mul_ps(_mm512_fmadd_ps(s2, j, s2), s1);
    const __m512 normal = _mm512_fmadd_ps(k, j, k);
    const __m512 infzero = _mm512_mul_ps(s1, s1);
    return _mm512_mask_blend_ps(huge, _mm512_mask_blend_ps(big, normal, scaled), infzero);
}

// -------------------------------------------------------------------------------------------
// QK dot: q is f32 buffer, k is packed row in KD.
// -------------------------------------------------------------------------------------------

template <AcDt KD>
EXL3_CPU_TARGET_AVX2
inline float ac_dot8(const float* q, const char* k, int D)
{
    constexpr int esz = (KD == AC_F32) ? 4 : 2;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    const int vec_end = D / 8 * 8;
    const int vec_end2 = D / 16 * 16;
    for (; i < vec_end2; i += 16)
    {
        const __m256 qv0 = _mm256_loadu_ps(q + i);
        const __m256 kv0 = ac_ld8<KD>(k + (size_t)i * esz);
        acc0 = _mm256_fmadd_ps(qv0, kv0, acc0);
        const __m256 qv1 = _mm256_loadu_ps(q + i + 8);
        const __m256 kv1 = ac_ld8<KD>(k + (size_t)(i + 8) * esz);
        acc1 = _mm256_fmadd_ps(qv1, kv1, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    for (; i < vec_end; i += 8)
    {
        const __m256 qv = _mm256_loadu_ps(q + i);
        const __m256 kv = ac_ld8<KD>(k + (size_t)i * esz);
        acc = _mm256_fmadd_ps(qv, kv, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < D; ++i) s += q[i] * ac_load_s(k + (size_t)i * esz, KD);
    return s;
}

template <AcDt KD>
EXL3_CPU_TARGET_AVX512
inline float ac_dot16(const float* q, const char* k, int D)
{
    constexpr int esz = (KD == AC_F32) ? 4 : 2;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    const int vec_end = D / 16 * 16;
    const int vec_end2 = D / 32 * 32;
    for (; i < vec_end2; i += 32)
    {
        const __m512 qv0 = _mm512_loadu_ps(q + i);
        const __m512 kv0 = ac_ld16<KD>(k + (size_t)i * esz);
        acc0 = _mm512_fmadd_ps(qv0, kv0, acc0);
        const __m512 qv1 = _mm512_loadu_ps(q + i + 16);
        const __m512 kv1 = ac_ld16<KD>(k + (size_t)(i + 16) * esz);
        acc1 = _mm512_fmadd_ps(qv1, kv1, acc1);
    }
    __m512 acc = _mm512_add_ps(acc0, acc1);
    for (; i < vec_end; i += 16)
    {
        const __m512 qv = _mm512_loadu_ps(q + i);
        const __m512 kv = ac_ld16<KD>(k + (size_t)i * esz);
        acc = _mm512_fmadd_ps(qv, kv, acc);
    }
    float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    float s = 0.0f;
    for (int j = 0; j < 16; ++j) s += tmp[j];
    for (; i < D; ++i) s += q[i] * ac_load_s(k + (size_t)i * esz, KD);
    return s;
}
// 4-query register-shared dot (f32 hot path): K chunks loaded once into regs,
// FMAs with 4 Q rows (4 independent chains, better ILP + 4x less K L1 traffic
// vs 4 separate ac_dot16 calls). Tail scalar for D%16. Used by the blocked job
// when all 4 queries cover k (dense non-causal; causal/windowed use masked
// single-dot path to stay exact).
EXL3_CPU_TARGET_AVX512
inline void ac_dot4_16_f32(const float* q0, const float* q1, const float* q2,
                           const float* q3, const float* k, int D, float* out)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    int i = 0;
    const int ve = D / 16 * 16;
    for (; i < ve; i += 16)
    {
        const __m512 kv = _mm512_loadu_ps(k + i);
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(q0 + i), kv, a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(q1 + i), kv, a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(q2 + i), kv, a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(q3 + i), kv, a3);
    }
    float t0[16], t1[16], t2[16], t3[16];
    _mm512_storeu_ps(t0, a0); _mm512_storeu_ps(t1, a1);
    _mm512_storeu_ps(t2, a2); _mm512_storeu_ps(t3, a3);
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int j = 0; j < 16; ++j) { s0 += t0[j]; s1 += t1[j]; s2 += t2[j]; s3 += t3[j]; }
    for (; i < D; ++i) { s0 += q0[i] * k[i]; s1 += q1[i] * k[i]; s2 += q2[i] * k[i]; s3 += q3[i] * k[i]; }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

// 4-query register-shared AV (f32): V chunk loaded once, FMAs into 4 acc rows.
EXL3_CPU_TARGET_AVX512
inline void ac_mad4_16_f32(float* a0, float* a1, float* a2, float* a3,
                           const float* v, float p0, float p1, float p2, float p3, int D)
{
    const __m512 b0 = _mm512_set1_ps(p0), b1 = _mm512_set1_ps(p1);
    const __m512 b2 = _mm512_set1_ps(p2), b3 = _mm512_set1_ps(p3);
    int i = 0;
    const int ve = D / 16 * 16;
    for (; i < ve; i += 16)
    {
        const __m512 vv = _mm512_loadu_ps(v + i);
        _mm512_storeu_ps(a0 + i, _mm512_fmadd_ps(vv, b0, _mm512_loadu_ps(a0 + i)));
        _mm512_storeu_ps(a1 + i, _mm512_fmadd_ps(vv, b1, _mm512_loadu_ps(a1 + i)));
        _mm512_storeu_ps(a2 + i, _mm512_fmadd_ps(vv, b2, _mm512_loadu_ps(a2 + i)));
        _mm512_storeu_ps(a3 + i, _mm512_fmadd_ps(vv, b3, _mm512_loadu_ps(a3 + i)));
    }
    for (; i < D; ++i)
    {
        const float vv = v[i];
        a0[i] += vv * p0; a1[i] += vv * p1; a2[i] += vv * p2; a3[i] += vv * p3;
    }
}

inline float ac_dot_scalar(const float* q, const char* k, int D, AcDt kdt, int kesz)
{
    float s = 0.0f;
    for (int i = 0; i < D; ++i) s += q[i] * ac_load_s(k + (size_t)i * kesz, kdt);
    return s;
}
template <AcDt VD>
EXL3_CPU_TARGET_AVX2
inline void ac_mad8(float* acc, const char* v, float prob, int D)
{
    constexpr int esz = (VD == AC_F32) ? 4 : 2;
    const __m256 pb = _mm256_set1_ps(prob);
    int i = 0;
    const int vec_end = D / 8 * 8;
    const int vec_end2 = D / 16 * 16;
    for (; i < vec_end2; i += 16)
    {
        const __m256 vv0 = ac_ld8<VD>(v + (size_t)i * esz);
        __m256 av0 = _mm256_loadu_ps(acc + i);
        av0 = _mm256_fmadd_ps(vv0, pb, av0);
        _mm256_storeu_ps(acc + i, av0);
        const __m256 vv1 = ac_ld8<VD>(v + (size_t)(i + 8) * esz);
        __m256 av1 = _mm256_loadu_ps(acc + i + 8);
        av1 = _mm256_fmadd_ps(vv1, pb, av1);
        _mm256_storeu_ps(acc + i + 8, av1);
    }
    for (; i < vec_end; i += 8)
    {
        const __m256 vv = ac_ld8<VD>(v + (size_t)i * esz);
        __m256 av = _mm256_loadu_ps(acc + i);
        av = _mm256_fmadd_ps(vv, pb, av);
        _mm256_storeu_ps(acc + i, av);
    }
    for (; i < D; ++i) acc[i] += prob * ac_load_s(v + (size_t)i * esz, VD);
}

template <AcDt VD>
EXL3_CPU_TARGET_AVX512
inline void ac_mad16(float* acc, const char* v, float prob, int D)
{
    constexpr int esz = (VD == AC_F32) ? 4 : 2;
    const __m512 pb = _mm512_set1_ps(prob);
    int i = 0;
    const int vec_end = D / 16 * 16;
    const int vec_end2 = D / 32 * 32;
    for (; i < vec_end2; i += 32)
    {
        const __m512 vv0 = ac_ld16<VD>(v + (size_t)i * esz);
        __m512 av0 = _mm512_loadu_ps(acc + i);
        av0 = _mm512_fmadd_ps(vv0, pb, av0);
        _mm512_storeu_ps(acc + i, av0);
        const __m512 vv1 = ac_ld16<VD>(v + (size_t)(i + 16) * esz);
        __m512 av1 = _mm512_loadu_ps(acc + i + 16);
        av1 = _mm512_fmadd_ps(vv1, pb, av1);
        _mm512_storeu_ps(acc + i + 16, av1);
    }
    for (; i < vec_end; i += 16)
    {
        const __m512 vv = ac_ld16<VD>(v + (size_t)i * esz);
        __m512 av = _mm512_loadu_ps(acc + i);
        av = _mm512_fmadd_ps(vv, pb, av);
        _mm512_storeu_ps(acc + i, av);
    }
    for (; i < D; ++i) acc[i] += prob * ac_load_s(v + (size_t)i * esz, VD);
}

inline void ac_mad_scalar(float* acc, const char* v, float prob, int D, AcDt vdt, int vesz)
{
    for (int i = 0; i < D; ++i) acc[i] += prob * ac_load_s(v + (size_t)i * vesz, vdt);
}

// -------------------------------------------------------------------------------------------
// Scores post-processing (f32 only): scale + optional softcap, max, exp-sum.
// Softcap matches the Triton/CUDA path: s = C * tanh(s * scale / C).
// tanh via 1 - 2/(exp(2y)+1) with the same -Ofast reassociation guard as act_mul.
// -------------------------------------------------------------------------------------------

EXL3_CPU_TARGET_AVX2
inline void ac_proc_256(float* s, int n, float scale, float softcap)
{
    if (softcap == 0.0f)
    {
        const __m256 sc = _mm256_set1_ps(scale);
        int i = 0;
        const int ve = n / 8 * 8;
        for (; i < ve; i += 8)
        {
            __m256 v = _mm256_loadu_ps(s + i);
            v = _mm256_mul_ps(v, sc);
            _mm256_storeu_ps(s + i, v);
        }
        for (; i < n; ++i) s[i] *= scale;
        return;
    }
    const float inv = scale / softcap;
    const __m256 vinv = _mm256_set1_ps(inv);
    const __m256 vtwo = _mm256_set1_ps(2.0f);
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vcap = _mm256_set1_ps(softcap);
    const __m256 vmax = _mm256_set1_ps(FLT_MAX);
    int i = 0;
    const int ve = n / 8 * 8;
    for (; i < ve; i += 8)
    {
        __m256 v = _mm256_loadu_ps(s + i);
        __m256 y = _mm256_mul_ps(v, vinv);
        __m256 e = ac_exp8(_mm256_mul_ps(y, vtwo));
        __m256 ec = _mm256_min_ps(e, vmax);
        __m256 th = _mm256_sub_ps(vone, _mm256_div_ps(_mm256_set1_ps(2.0f),
                                                     _mm256_add_ps(ec, vone)));
        _mm256_storeu_ps(s + i, _mm256_mul_ps(th, vcap));
    }
    for (; i < n; ++i)
    {
        const float y = s[i] * inv;
        const float e = std::exp(2.0f * y);
        const float ec = e > FLT_MAX ? FLT_MAX : e;
        s[i] = (1.0f - 2.0f / (ec + 1.0f)) * softcap;
    }
}

EXL3_CPU_TARGET_AVX512
inline void ac_proc_512(float* s, int n, float scale, float softcap)
{
    if (softcap == 0.0f)
    {
        const __m512 sc = _mm512_set1_ps(scale);
        int i = 0;
        const int ve = n / 16 * 16;
        for (; i < ve; i += 16)
        {
            __m512 v = _mm512_loadu_ps(s + i);
            v = _mm512_mul_ps(v, sc);
            _mm512_storeu_ps(s + i, v);
        }
        for (; i < n; ++i) s[i] *= scale;
        return;
    }
    const float inv = scale / softcap;
    const __m512 vinv = _mm512_set1_ps(inv);
    const __m512 vtwo = _mm512_set1_ps(2.0f);
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 vcap = _mm512_set1_ps(softcap);
    const __m512 vmax = _mm512_set1_ps(FLT_MAX);
    int i = 0;
    const int ve = n / 16 * 16;
    for (; i < ve; i += 16)
    {
        __m512 v = _mm512_loadu_ps(s + i);
        __m512 y = _mm512_mul_ps(v, vinv);
        __m512 e = ac_exp16(_mm512_mul_ps(y, vtwo));
        __m512 ec = _mm512_min_ps(e, vmax);
        __m512 th = _mm512_sub_ps(vone, _mm512_div_ps(_mm512_set1_ps(2.0f),
                                                     _mm512_add_ps(ec, vone)));
        _mm512_storeu_ps(s + i, _mm512_mul_ps(th, vcap));
    }
    for (; i < n; ++i)
    {
        const float y = s[i] * inv;
        const float e = std::exp(2.0f * y);
        const float ec = e > FLT_MAX ? FLT_MAX : e;
        s[i] = (1.0f - 2.0f / (ec + 1.0f)) * softcap;
    }
}

inline void ac_proc_scalar(float* s, int n, float scale, float softcap)
{
    if (softcap == 0.0f)
    {
        for (int i = 0; i < n; ++i) s[i] *= scale;
        return;
    }
    const float inv = scale / softcap;
    for (int i = 0; i < n; ++i)
    {
        const float y = s[i] * inv;
        s[i] = std::tanh(y) * softcap;
    }
}

EXL3_CPU_TARGET_AVX2
inline float ac_max_256(const float* s, int n)
{
    __m256 m = _mm256_set1_ps(-INFINITY);
    int i = 0;
    const int ve = n / 8 * 8;
    for (; i < ve; i += 8) m = _mm256_max_ps(m, _mm256_loadu_ps(s + i));
    float tmp[8];
    _mm256_storeu_ps(tmp, m);
    float r = tmp[0];
    for (int j = 1; j < 8; ++j) r = r > tmp[j] ? r : tmp[j];
    for (; i < n; ++i) r = r > s[i] ? r : s[i];
    return r;
}

EXL3_CPU_TARGET_AVX512
inline float ac_max_512(const float* s, int n)
{
    __m512 m = _mm512_set1_ps(-INFINITY);
    int i = 0;
    const int ve = n / 16 * 16;
    for (; i < ve; i += 16) m = _mm512_max_ps(m, _mm512_loadu_ps(s + i));
    float tmp[16];
    _mm512_storeu_ps(tmp, m);
    float r = tmp[0];
    for (int j = 1; j < 16; ++j) r = r > tmp[j] ? r : tmp[j];
    for (; i < n; ++i) r = r > s[i] ? r : s[i];
    return r;
}

inline float ac_max_scalar(const float* s, int n)
{
    float r = -INFINITY;
    for (int i = 0; i < n; ++i) r = r > s[i] ? r : s[i];
    return r;
}

EXL3_CPU_TARGET_AVX2
inline float ac_exp_sum_256(float* s, int n, float m_top)
{
    const __m256 vm = _mm256_set1_ps(m_top);
    float sum = 0.0f;
    int i = 0;
    const int ve = n / 8 * 8;
    for (; i < ve; i += 8)
    {
        __m256 v = _mm256_loadu_ps(s + i);
        v = ac_exp8(_mm256_sub_ps(v, vm));
        _mm256_storeu_ps(s + i, v);
        float tmp[8];
        _mm256_storeu_ps(tmp, v);
        for (int j = 0; j < 8; ++j) sum += tmp[j];
    }
    for (; i < n; ++i)
    {
        const float e = std::exp(s[i] - m_top);
        s[i] = e;
        sum += e;
    }
    return sum;
}

EXL3_CPU_TARGET_AVX512
inline float ac_exp_sum_512(float* s, int n, float m_top)
{
    const __m512 vm = _mm512_set1_ps(m_top);
    float sum = 0.0f;
    int i = 0;
    const int ve = n / 16 * 16;
    for (; i < ve; i += 16)
    {
        __m512 v = _mm512_loadu_ps(s + i);
        v = ac_exp16(_mm512_sub_ps(v, vm));
        _mm512_storeu_ps(s + i, v);
    float tmp[16];
    _mm512_storeu_ps(tmp, v);
    for (int j = 0; j < 16; ++j) sum += tmp[j];
    }
    for (; i < n; ++i)
    {
        const float e = std::exp(s[i] - m_top);
        s[i] = e;
        sum += e;
    }
    return sum;
}

inline float ac_exp_sum_scalar(float* s, int n, float m_top)
{
    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        const float e = std::exp(s[i] - m_top);
        s[i] = e;
        sum += e;
    }
    return sum;
}
template <AcIsa ISA>
inline void ac_store_row(char* o_ptr, const float* acc, int D, AcDt odt);

template <>
EXL3_CPU_TARGET_AVX512
inline void ac_store_row<AC_AVX512>(char* o_ptr, const float* acc, int D, AcDt odt)
{
    if (odt == AC_F32)
    {
        std::memcpy(o_ptr, acc, (size_t)D * 4);
        return;
    }
    if (odt == AC_F16)
    {
        const int esz = 2;
        int i = 0;
        const int ve = D / 16 * 16;
        for (; i < ve; i += 16) ac_st16_f16(o_ptr + (size_t)i * esz, _mm512_loadu_ps(acc + i));
        for (; i < D; ++i) ac_store_s(o_ptr + (size_t)i * esz, odt, acc[i]);
        return;
    }
    const int esz = 2;
    for (int i = 0; i < D; ++i) ac_store_s(o_ptr + (size_t)i * esz, odt, acc[i]);
}

template <>
EXL3_CPU_TARGET_AVX2
inline void ac_store_row<AC_AVX2>(char* o_ptr, const float* acc, int D, AcDt odt)
{
    if (odt == AC_F32)
    {
        std::memcpy(o_ptr, acc, (size_t)D * 4);
        return;
    }
    if (odt == AC_F16)
    {
        const int esz = 2;
        int i = 0;
        const int ve = D / 8 * 8;
        for (; i < ve; i += 8) ac_st8_f16(o_ptr + (size_t)i * esz, _mm256_loadu_ps(acc + i));
        for (; i < D; ++i) ac_store_s(o_ptr + (size_t)i * esz, odt, acc[i]);
        return;
    }
    const int esz = 2;
    for (int i = 0; i < D; ++i) ac_store_s(o_ptr + (size_t)i * esz, odt, acc[i]);
}

template <>
inline void ac_store_row<AC_SCALAR>(char* o_ptr, const float* acc, int D, AcDt odt)
{
    if (odt == AC_F32)
    {
        std::memcpy(o_ptr, acc, (size_t)D * 4);
        return;
    }
    const int esz = 2;
    for (int i = 0; i < D; ++i) ac_store_s(o_ptr + (size_t)i * esz, odt, acc[i]);
}

struct AcCtx
{
    const char* q;
    const char* k;
    const char* v;
    char* o;
    int qesz, kesz, vesz, oesz;
    AcDt qdt, kdt, vdt, odt;
    int64_t B, Lq, Lk, Hq, Hkv, D;
    int64_t group;
    float scale;
    float softcap;
    bool causal;
    int64_t wl, wr;   // < 0 disables that side
    int64_t q_start;
    const float* sinks;   // nullable, [Hq]
    const int64_t* segs;  // nullable, 2*Lq: [start,end) per lq (varlen); null = dense
    AcIsa isa;
};

template <AcIsa ISA>
inline void ac_job_one(const AcCtx& c, int64_t b, int64_t lq, int64_t hq,
                       float* q_buf, float* acc, float* scores)
{
    const int D = (int)c.D;
    const int64_t hkv = hq / c.group;

    int64_t k_lo, k_hi;
    if (c.segs)
    {
        const int64_t s = c.segs[(size_t)lq * 2];
        const int64_t e = c.segs[(size_t)lq * 2 + 1];
        const int64_t q_rel = lq - s;
        const int64_t seg_len = e - s;
        int64_t lo_rel = 0, hi_rel = seg_len;
        if (c.causal && q_rel + 1 < hi_rel) hi_rel = q_rel + 1;
        if (c.wl >= 0)
        {
            const int64_t wlo = q_rel - c.wl;
            if (wlo > lo_rel) lo_rel = wlo;
        }
        if (c.wr >= 0)
        {
            const int64_t whi = q_rel + c.wr + 1;
            if (whi < hi_rel) hi_rel = whi;
        }
        if (lo_rel < 0) lo_rel = 0;
        if (hi_rel > seg_len) hi_rel = seg_len;
        k_lo = s + lo_rel;
        k_hi = s + hi_rel;
    }
    else
    {
        const int64_t q_abs = c.q_start + lq;
        int64_t lo = 0, hi = c.Lk;
        if (c.causal && q_abs + 1 < hi) hi = q_abs + 1;
        if (c.wl >= 0)
        {
            const int64_t wlo = q_abs - c.wl;
            if (wlo > lo) lo = wlo;
        }
        if (c.wr >= 0)
        {
            const int64_t whi = q_abs + c.wr + 1;
            if (whi < hi) hi = whi;
        }
        if (lo < 0) lo = 0;
        if (hi > c.Lk) hi = c.Lk;
        k_lo = lo;
        k_hi = hi;
    }

    const size_t q_row = ((size_t)b * c.Hq + hq) * c.Lq + lq;
    char* o_ptr = c.o + q_row * (size_t)D * c.oesz;

    if (k_lo >= k_hi)
    {
        for (int i = 0; i < D; ++i) ac_store_s(o_ptr + (size_t)i * c.oesz, c.odt, 0.0f);
        return;
    }
    const int k_len = (int)(k_hi - k_lo);

    const char* q_ptr = c.q + q_row * (size_t)D * c.qesz;
    for (int i = 0; i < D; ++i) q_buf[i] = ac_load_s(q_ptr + (size_t)i * c.qesz, c.qdt);

    // Scores = raw QK dots.
    if constexpr (ISA == AC_AVX512)
    {
        switch (c.kdt)
        {
            case AC_F32:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    scores[j] = ac_dot16<AC_F32>(q_buf, c.k + k_row * (size_t)D * c.kesz, D);
                }
                break;
            case AC_F16:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    scores[j] = ac_dot16<AC_F16>(q_buf, c.k + k_row * (size_t)D * c.kesz, D);
                }
                break;
            default:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    scores[j] = ac_dot16<AC_BF16>(q_buf, c.k + k_row * (size_t)D * c.kesz, D);
                }
                break;
        }
        ac_proc_512(scores, k_len, c.scale, c.softcap);
    }
    else if constexpr (ISA == AC_AVX2)
    {
        switch (c.kdt)
        {
            case AC_F32:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    scores[j] = ac_dot8<AC_F32>(q_buf, c.k + k_row * (size_t)D * c.kesz, D);
                }
                break;
            case AC_F16:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    scores[j] = ac_dot8<AC_F16>(q_buf, c.k + k_row * (size_t)D * c.kesz, D);
                }
                break;
            default:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    scores[j] = ac_dot8<AC_BF16>(q_buf, c.k + k_row * (size_t)D * c.kesz, D);
                }
                break;
        }
        ac_proc_256(scores, k_len, c.scale, c.softcap);
    }
    else
    {
        for (int j = 0; j < k_len; ++j)
        {
            const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
            scores[j] = ac_dot_scalar(q_buf, c.k + k_row * (size_t)D * c.kesz, D, c.kdt, c.kesz);
        }
        ac_proc_scalar(scores, k_len, c.scale, c.softcap);
    }

    float m;
    if constexpr (ISA == AC_AVX512) m = ac_max_512(scores, k_len);
    else if constexpr (ISA == AC_AVX2) m = ac_max_256(scores, k_len);
    else m = ac_max_scalar(scores, k_len);

    float m_top = m;
    float sink = 0.0f;
    const bool has_sink = c.sinks != nullptr;
    if (has_sink)
    {
        sink = c.sinks[hq];
        if (sink > m_top) m_top = sink;
    }

    float l;
    if constexpr (ISA == AC_AVX512) l = ac_exp_sum_512(scores, k_len, m_top);
    else if constexpr (ISA == AC_AVX2) l = ac_exp_sum_256(scores, k_len, m_top);
    else l = ac_exp_sum_scalar(scores, k_len, m_top);
    if (has_sink) l += std::exp(sink - m_top);

    if (!(l > 0.0f) || m_top == -INFINITY)
    {
        for (int i = 0; i < D; ++i) ac_store_s(o_ptr + (size_t)i * c.oesz, c.odt, 0.0f);
        return;
    }
    const float inv = 1.0f / l;

    for (int i = 0; i < D; ++i) acc[i] = 0.0f;
    if constexpr (ISA == AC_AVX512)
    {
        switch (c.vdt)
        {
            case AC_F32:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    ac_mad16<AC_F32>(acc, c.v + v_row * (size_t)D * c.vesz, scores[j] * inv, D);
                }
                break;
            case AC_F16:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    ac_mad16<AC_F16>(acc, c.v + v_row * (size_t)D * c.vesz, scores[j] * inv, D);
                }
                break;
            default:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    ac_mad16<AC_BF16>(acc, c.v + v_row * (size_t)D * c.vesz, scores[j] * inv, D);
                }
                break;
        }
    }
    else if constexpr (ISA == AC_AVX2)
    {
        switch (c.vdt)
        {
            case AC_F32:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    ac_mad8<AC_F32>(acc, c.v + v_row * (size_t)D * c.vesz, scores[j] * inv, D);
                }
                break;
            case AC_F16:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    ac_mad8<AC_F16>(acc, c.v + v_row * (size_t)D * c.vesz, scores[j] * inv, D);
                }
                break;
            default:
                for (int j = 0; j < k_len; ++j)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
                    ac_mad8<AC_BF16>(acc, c.v + v_row * (size_t)D * c.vesz, scores[j] * inv, D);
                }
                break;
        }
    }
    else
    {
        for (int j = 0; j < k_len; ++j)
        {
            const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + (k_lo + j);
            ac_mad_scalar(acc, c.v + v_row * (size_t)D * c.vesz, scores[j] * inv, D, c.vdt, c.vesz);
        }
    }

    ac_store_row<ISA>(o_ptr, acc, D, c.odt);
}
// Online dense non-causal flash path (AVX512 f32, Mb=32, Nb=512): single pass over
// K/V per Q-block (vs two passes in two-pass path), no N^2 scores (only Nb tiles
// in cache), exact same math (scale/softcap per tile, running m/l rescale, sinks
// init m=sink/l=1). Covers vision dense 1k/2k/4k non-causal (the torch gap shapes).
// Returns false when not applicable so caller falls back to generic block/two-pass.
EXL3_CPU_TARGET_AVX512
inline bool ac_block_online_dense_512(const AcCtx& c, int64_t b, int64_t hq,
                                      const int64_t* lqs, int nq,
                                      float* q_bufs, float* accs, float* scores_base, int64_t max_klen)
{
    if (c.segs || c.causal || c.wl >= 0 || c.wr >= 0) return false;
    if (c.kdt != AC_F32 || c.qdt != AC_F32 || c.vdt != AC_F32) return false;
    if (nq != 32) return false;
    const int D = (int)c.D;
    const int64_t hkv = hq / c.group;
    const int64_t Lk = c.Lk;
    // Q rows must be consecutive for the dense fast path (vision dense always is;
    // varlen/spans use generic path). Check lqs[i] == lqs[0]+i.
    for (int i = 1; i < 32; ++i) if (lqs[i] != lqs[0] + i) return false;
    for (int i = 0; i < 32; ++i)
    {
        const size_t q_row = ((size_t)b * c.Hq + hq) * c.Lq + (lqs[0] + i);
        std::memcpy(q_bufs + (size_t)i * D, c.q + q_row * (size_t)D * c.qesz, (size_t)D * 4);
        float* acb = accs + (size_t)i * D;
        for (int j = 0; j < D; ++j) acb[j] = 0.0f;
    }
    float m[32], l[32];
    const bool has_sink = c.sinks != nullptr;
    const float sk = has_sink ? c.sinks[hq] : 0.0f;
    for (int i = 0; i < 32; ++i) { m[i] = has_sink ? sk : -INFINITY; l[i] = has_sink ? 1.0f : 0.0f; }
    constexpr int64_t NB = 512;
    // Tile buffer reuses scores_base rows (32 x max_klen) with stride max_klen;
    // only first Nt per row used per K tile.
    for (int64_t k0 = 0; k0 < Lk; k0 += NB)
    {
        const int64_t k1 = k0 + NB < Lk ? k0 + NB : Lk;
        const int nt = (int)(k1 - k0);
        // Scores tile 32 x nt via 8x dot4 groups per k (K in regs once per group).
        for (int64_t k = k0; k < k1; ++k)
        {
            const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
            const float* kp = (const float*)(c.k + k_row * (size_t)D * c.kesz);
            for (int g = 0; g < 8; ++g)
            {
                const float* q0 = q_bufs + (size_t)(g * 4) * D;
                float o4[4];
                ac_dot4_16_f32(q0, q0 + D, q0 + 2 * D, q0 + 3 * D, kp, D, o4);
                for (int t = 0; t < 4; ++t)
                    scores_base[(size_t)(g * 4 + t) * max_klen + (k - k0)] = o4[t];
            }
        }
        for (int i = 0; i < 32; ++i)
            ac_proc_512(scores_base + (size_t)i * max_klen, nt, c.scale, c.softcap);
        for (int i = 0; i < 32; ++i)
        {
            float* sc = scores_base + (size_t)i * max_klen;
            const float m_old = m[i];
            const float tm = ac_max_512(sc, nt);
            const float m_new = m_old > tm ? m_old : tm;
            const float alpha = std::exp(m_old - m_new);
            l[i] *= alpha;
            float* acb = accs + (size_t)i * D;
            const __m512 va = _mm512_set1_ps(alpha);
            int j = 0;
            const int ve = D / 16 * 16;
            for (; j < ve; j += 16)
            {
                __m512 av = _mm512_loadu_ps(acb + j);
                _mm512_storeu_ps(acb + j, _mm512_mul_ps(av, va));
            }
            for (; j < D; ++j) acb[j] *= alpha;
            const float s = ac_exp_sum_512(sc, nt, m_new);
            l[i] += s;
            m[i] = m_new;
        }
        // AV tile with freshly exp'd scores (no inv yet, online numerator).
        for (int64_t k = k0; k < k1; ++k)
        {
            const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
            const float* vp = (const float*)(c.v + v_row * (size_t)D * c.vesz);
            const int64_t j = k - k0;
            for (int g = 0; g < 8; ++g)
            {
                float* a0 = accs + (size_t)(g * 4) * D;
                ac_mad4_16_f32(a0, a0 + D, a0 + 2 * D, a0 + 3 * D, vp,
                               scores_base[(size_t)(g * 4) * max_klen + j],
                               scores_base[(size_t)(g * 4 + 1) * max_klen + j],
                               scores_base[(size_t)(g * 4 + 2) * max_klen + j],
                               scores_base[(size_t)(g * 4 + 3) * max_klen + j], D);
            }
        }
    }
    for (int i = 0; i < 32; ++i)
    {
        if (!(l[i] > 0.0f) || m[i] == -INFINITY) return false;
        const float inv = 1.0f / l[i];
        float* acb = accs + (size_t)i * D;
        const __m512 vi = _mm512_set1_ps(inv);
        int j = 0;
        const int ve = D / 16 * 16;
        for (; j < ve; j += 16)
            _mm512_storeu_ps(acb + j, _mm512_mul_ps(_mm512_loadu_ps(acb + j), vi));
        for (; j < D; ++j) acb[j] *= inv;
        const int64_t lq = lqs[i];
        const size_t o_row = ((size_t)b * c.Hq + hq) * c.Lq + lq;
        ac_store_row<AC_AVX512>(c.o + o_row * (size_t)D * c.oesz, acb, D, c.odt);
    }
    return true;
}
// Blocked 16-query job sharing K/V loads across queries of the same (batch, head).
// Same math as ac_job_one, but K/V rows are loaded once per k and reused for up to
// 16 queries (L3 traffic /16, L1 hit for the others, regs-shared dot4/av4 fast paths
// for f32 in groups of 4). Cross-segment boundary blocks share with per-query masking
// (waste one block per boundary, negligible). Returns false only on empty ranges so
// the caller falls back to per-query ac_job_one. q_bufs/accs are 16xD, scores 16xmax.
template <AcIsa ISA>
inline bool ac_job_block4(const AcCtx& c, int64_t b, int64_t hq,
                          const int64_t* lqs, int nq,
                          float* q_bufs, float* accs, float* scores_base, int64_t max_klen)
{
    const int D = (int)c.D;
    const int64_t hkv = hq / c.group;
    // NOTE: online dense flash path disabled (measured 286ms vs 234ms two-pass at
    // 4k dense; rescaling + 8x max/exp calls outweigh single-pass saving at Mb=32).
    // Keep until Mb>=64 online with true GEMM microkernels lands.
    int64_t k_lo[16], k_hi[16];
    for (int i = 0; i < nq; ++i)
    {
        const int64_t lq = lqs[i];
        if (c.segs)
        {
            const int64_t s = c.segs[(size_t)lq * 2];
            const int64_t e = c.segs[(size_t)lq * 2 + 1];
            // No same-segment early-out: cross-boundary blocks share K/V loads
            // with per-query masking (waste is one block per boundary, negligible).
            const int64_t q_rel = lq - s;
            const int64_t seg_len = e - s;
            int64_t lo_rel = 0, hi_rel = seg_len;
            if (c.causal && q_rel + 1 < hi_rel) hi_rel = q_rel + 1;
            if (c.wl >= 0)
            {
                const int64_t wlo = q_rel - c.wl;
                if (wlo > lo_rel) lo_rel = wlo;
            }
            if (c.wr >= 0)
            {
                const int64_t whi = q_rel + c.wr + 1;
                if (whi < hi_rel) hi_rel = whi;
            }
            if (lo_rel < 0) lo_rel = 0;
            if (hi_rel > seg_len) hi_rel = seg_len;
            k_lo[i] = s + lo_rel;
            k_hi[i] = s + hi_rel;
        }
        else
        {
            const int64_t q_abs = c.q_start + lq;
            int64_t lo = 0, hi = c.Lk;
            if (c.causal && q_abs + 1 < hi) hi = q_abs + 1;
            if (c.wl >= 0)
            {
                const int64_t wlo = q_abs - c.wl;
                if (wlo > lo) lo = wlo;
            }
            if (c.wr >= 0)
            {
                const int64_t whi = q_abs + c.wr + 1;
                if (whi < hi) hi = whi;
            }
            if (lo < 0) lo = 0;
            if (hi > c.Lk) hi = c.Lk;
            k_lo[i] = lo;
            k_hi[i] = hi;
        }
        if (k_lo[i] >= k_hi[i]) return false;
        const size_t q_row = ((size_t)b * c.Hq + hq) * c.Lq + lq;
        const char* q_ptr = c.q + q_row * (size_t)D * c.qesz;
        float* qb = q_bufs + (size_t)i * D;
        for (int j = 0; j < D; ++j) qb[j] = ac_load_s(q_ptr + (size_t)j * c.qesz, c.qdt);
    }
    int64_t klo_min = k_lo[0], khi_max = k_hi[0];
    for (int i = 1; i < nq; ++i)
    {
        if (k_lo[i] < klo_min) klo_min = k_lo[i];
        if (k_hi[i] > khi_max) khi_max = k_hi[i];
    }
    // Dots: K row loaded once per k, reused for all queries covering it.
    if constexpr (ISA == AC_AVX512)
    {
        // Fast path: f32, nq multiple of 4 as dot4 groups (dense/varlen,
        // causal/windowed via per-query masking on store). Register-shared dot4
        // (K in regs once per group); L3 traffic /nq via blocking (nq=32 here).
        if ((nq % 4) == 0 && c.kdt == AC_F32 && c.qdt == AC_F32)
        {
            const int ng = nq / 4;
            for (int64_t k = klo_min; k < khi_max; ++k)
            {
                const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                const float* kp = (const float*)(c.k + k_row * (size_t)D * c.kesz);
                for (int g = 0; g < ng; ++g)
                {
                    const float* q0 = q_bufs + (size_t)(g * 4) * D;
                    const float* q1 = q0 + D, *q2 = q0 + 2 * D, *q3 = q0 + 3 * D;
                    float o4[4];
                    ac_dot4_16_f32(q0, q1, q2, q3, kp, D, o4);
                    for (int t = 0; t < 4; ++t)
                    {
                        const int i = g * 4 + t;
                        if (k >= k_lo[i] && k < k_hi[i])
                            scores_base[(size_t)i * max_klen + (k - k_lo[i])] = o4[t];
                    }
                }
            }
        }
        else switch (c.kdt)
        {
            case AC_F32:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* kp = c.k + k_row * (size_t)D * c.kesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            scores_base[(size_t)i * max_klen + (k - k_lo[i])] = ac_dot16<AC_F32>(q_bufs + (size_t)i * D, kp, D);
                }
                break;
            case AC_F16:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* kp = c.k + k_row * (size_t)D * c.kesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            scores_base[(size_t)i * max_klen + (k - k_lo[i])] = ac_dot16<AC_F16>(q_bufs + (size_t)i * D, kp, D);
                }
                break;
            default:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* kp = c.k + k_row * (size_t)D * c.kesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            scores_base[(size_t)i * max_klen + (k - k_lo[i])] = ac_dot16<AC_BF16>(q_bufs + (size_t)i * D, kp, D);
                }
                break;
        }
        for (int i = 0; i < nq; ++i) ac_proc_512(scores_base + (size_t)i * max_klen, (int)(k_hi[i] - k_lo[i]), c.scale, c.softcap);
    }
    else if constexpr (ISA == AC_AVX2)
    {
        switch (c.kdt)
        {
            case AC_F32:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* kp = c.k + k_row * (size_t)D * c.kesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            scores_base[(size_t)i * max_klen + (k - k_lo[i])] = ac_dot8<AC_F32>(q_bufs + (size_t)i * D, kp, D);
                }
                break;
            case AC_F16:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* kp = c.k + k_row * (size_t)D * c.kesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            scores_base[(size_t)i * max_klen + (k - k_lo[i])] = ac_dot8<AC_F16>(q_bufs + (size_t)i * D, kp, D);
                }
                break;
            default:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* kp = c.k + k_row * (size_t)D * c.kesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            scores_base[(size_t)i * max_klen + (k - k_lo[i])] = ac_dot8<AC_BF16>(q_bufs + (size_t)i * D, kp, D);
                }
                break;
        }
        for (int i = 0; i < nq; ++i) ac_proc_256(scores_base + (size_t)i * max_klen, (int)(k_hi[i] - k_lo[i]), c.scale, c.softcap);
    }
    else
    {
        for (int64_t k = klo_min; k < khi_max; ++k)
        {
            const size_t k_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
            const char* kp = c.k + k_row * (size_t)D * c.kesz;
            for (int i = 0; i < nq; ++i)
                if (k >= k_lo[i] && k < k_hi[i])
                    scores_base[(size_t)i * max_klen + (k - k_lo[i])] = ac_dot_scalar(q_bufs + (size_t)i * D, kp, D, c.kdt, c.kesz);
        }
        for (int i = 0; i < nq; ++i) ac_proc_scalar(scores_base + (size_t)i * max_klen, (int)(k_hi[i] - k_lo[i]), c.scale, c.softcap);
    }
    float inv[16];
    for (int i = 0; i < nq; ++i)
    {
        float* sc = scores_base + (size_t)i * max_klen;
        const int k_len = (int)(k_hi[i] - k_lo[i]);
        float m;
        if constexpr (ISA == AC_AVX512) m = ac_max_512(sc, k_len);
        else if constexpr (ISA == AC_AVX2) m = ac_max_256(sc, k_len);
        else m = ac_max_scalar(sc, k_len);
        float m_top = m;
        if (c.sinks)
        {
            const float sk = c.sinks[hq];
            if (sk > m_top) m_top = sk;
        }
        float l;
        if constexpr (ISA == AC_AVX512) l = ac_exp_sum_512(sc, k_len, m_top);
        else if constexpr (ISA == AC_AVX2) l = ac_exp_sum_256(sc, k_len, m_top);
        else l = ac_exp_sum_scalar(sc, k_len, m_top);
        if (c.sinks) l += std::exp(c.sinks[hq] - m_top);
        if (!(l > 0.0f) || m_top == -INFINITY) return false;
        inv[i] = 1.0f / l;
        float* acb = accs + (size_t)i * D;
        for (int j = 0; j < D; ++j) acb[j] = 0.0f;
    }
    if constexpr (ISA == AC_AVX512)
    {
        if ((nq % 4) == 0 && c.vdt == AC_F32)
        {
            // ng av4 groups; masked via zero probs. V in regs once per group,
            // L3 /nq via blocking (nq=32 here).
            const int ng = nq / 4;
            for (int64_t k = klo_min; k < khi_max; ++k)
            {
                const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                const float* vp = (const float*)(c.v + v_row * (size_t)D * c.vesz);
                for (int g = 0; g < ng; ++g)
                {
                    float* a0 = accs + (size_t)(g * 4) * D;
                    float* a1 = a0 + D, *a2 = a0 + 2 * D, *a3 = a0 + 3 * D;
                    float q0 = 0, q1 = 0, q2 = 0, q3 = 0;
                    for (int t = 0; t < 4; ++t)
                    {
                        const int i = g * 4 + t;
                        float pp = 0.0f;
                        if (k >= k_lo[i] && k < k_hi[i])
                            pp = scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i];
                        if (t == 0) q0 = pp; else if (t == 1) q1 = pp;
                        else if (t == 2) q2 = pp; else q3 = pp;
                    }
                    ac_mad4_16_f32(a0, a1, a2, a3, vp, q0, q1, q2, q3, D);
                }
            }
        }
        else switch (c.vdt)
        {
            case AC_F32:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* vp = c.v + v_row * (size_t)D * c.vesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            ac_mad16<AC_F32>(accs + (size_t)i * D, vp, scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i], D);
                }
                break;
            case AC_F16:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* vp = c.v + v_row * (size_t)D * c.vesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            ac_mad16<AC_F16>(accs + (size_t)i * D, vp, scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i], D);
                }
                break;
            default:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* vp = c.v + v_row * (size_t)D * c.vesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            ac_mad16<AC_BF16>(accs + (size_t)i * D, vp, scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i], D);
                }
                break;
        }
    }
    else if constexpr (ISA == AC_AVX2)
    {
        switch (c.vdt)
        {
            case AC_F32:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* vp = c.v + v_row * (size_t)D * c.vesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            ac_mad8<AC_F32>(accs + (size_t)i * D, vp, scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i], D);
                }
                break;
            case AC_F16:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* vp = c.v + v_row * (size_t)D * c.vesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            ac_mad8<AC_F16>(accs + (size_t)i * D, vp, scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i], D);
                }
                break;
            default:
                for (int64_t k = klo_min; k < khi_max; ++k)
                {
                    const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
                    const char* vp = c.v + v_row * (size_t)D * c.vesz;
                    for (int i = 0; i < nq; ++i)
                        if (k >= k_lo[i] && k < k_hi[i])
                            ac_mad8<AC_BF16>(accs + (size_t)i * D, vp, scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i], D);
                }
                break;
        }
    }
    else
    {
        for (int64_t k = klo_min; k < khi_max; ++k)
        {
            const size_t v_row = ((size_t)b * c.Hkv + hkv) * c.Lk + k;
            const char* vp = c.v + v_row * (size_t)D * c.vesz;
            for (int i = 0; i < nq; ++i)
                if (k >= k_lo[i] && k < k_hi[i])
                    ac_mad_scalar(accs + (size_t)i * D, vp, scores_base[(size_t)i * max_klen + (k - k_lo[i])] * inv[i], D, c.vdt, c.vesz);
        }
    }
    for (int i = 0; i < nq; ++i)
    {
        const int64_t lq = lqs[i];
        const size_t q_row = ((size_t)b * c.Hq + hq) * c.Lq + lq;
        ac_store_row<ISA>(c.o + q_row * (size_t)D * c.oesz, accs + (size_t)i * D, D, c.odt);
    }
    return true;
}

template <AcIsa ISA>
inline void ac_run(const AcCtx& c, int64_t max_klen)
{
    const int64_t total_q = c.B * c.Lq;
    if (total_q <= 0) return;
    const int D = (int)c.D;
    // 16 queries x (Q + acc + scores) shared per tile; reused per head.
    const size_t scratch = (size_t)16 * ((size_t)2 * D + (size_t)max_klen);
    auto process = [&](int64_t b, int64_t e)
    {
        const int64_t b0 = b * AC_TOKEN_CHUNK;
        int64_t e0 = e * AC_TOKEN_CHUNK;
        if (e0 > total_q) e0 = total_q;
        std::vector<float> buf(scratch);
        float* q_bufs = buf.data();
        float* accs = q_bufs + (size_t)16 * D;
        float* scores_base = accs + (size_t)16 * D;
        // Single-query fallback partitions (first rows of the block buffers).
        float* q1 = q_bufs;
        float* a1 = accs;
        float* s1 = scores_base;
        for (int64_t t = b0; t < e0;)
        {
            int64_t bb0 = t / c.Lq;
            int64_t lq0 = t % c.Lq;
            int nq = 0;
            int64_t lqs[16];
            // Gather up to 16 consecutive queries of the same batch only; blocks
            // never cross batches so K/V bases stay shared.
            while (nq < 16 && t + nq < e0)
            {
                const int64_t tt = t + nq;
                const int64_t bb = tt / c.Lq;
                if (bb != bb0) break;
                lqs[nq++] = tt % c.Lq;
            }
            // lq0 var unused beyond gather; keep symmetric with old driver.
            (void)lq0;
            for (int64_t hq = 0; hq < c.Hq; ++hq)
            {
                if (nq > 1 && ac_job_block4<ISA>(c, bb0, hq, lqs, nq, q_bufs, accs, scores_base, max_klen))
                    continue;
                for (int i = 0; i < nq; ++i)
                    ac_job_one<ISA>(c, bb0, lqs[i], hq, q1, a1, s1);
            }
            t += nq;
        }
    };
    struct Tile
    {
        static void tile(void* ctx, int64_t b, int64_t e)
        {
            (*static_cast<decltype(&process)>(ctx))(b, e);
        }
    };
    const int64_t total = (total_q + AC_TOKEN_CHUNK - 1) / AC_TOKEN_CHUNK;
    cpu_pool_run_tiles(total, &Tile::tile, (void*)&process);
}

inline int64_t ac_read_idx(const char* p, int sz)
{
    if (sz == 4)
    {
        int32_t v;
        std::memcpy(&v, p, 4);
        return (int64_t)v;
    }
    int64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

} // namespace

void attn_cpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& o,
    double sm_scale,
    bool causal,
    int64_t window_left,
    int64_t window_right,
    double softcap,
    const c10::optional<at::Tensor>& sinks,
    const c10::optional<at::Tensor>& cu_seqlens,
    int64_t q_start)
{
    TORCH_CHECK(q.device().is_cpu() && k.device().is_cpu() &&
                v.device().is_cpu() && o.device().is_cpu(),
                "attn_cpu: CPU tensors only");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() &&
                v.is_contiguous() && o.is_contiguous(),
                "attn_cpu: q, k, v and o must be contiguous");
    for (const at::Tensor* t : {(const at::Tensor*)&q, (const at::Tensor*)&k, (const at::Tensor*)&v, (const at::Tensor*)&o})
        TORCH_CHECK(t->is_floating_point() && (t->scalar_type() == at::kFloat ||
                    t->scalar_type() == at::kHalf || t->scalar_type() == at::kBFloat16),
                    "attn_cpu: q/k/v/o dtype must be f32/f16/bf16");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && o.dim() == 4,
                "attn_cpu: q/k/v/o must be 4-D [B, L, H, D]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) && q.size(0) == o.size(0),
                "attn_cpu: batch mismatch");
    TORCH_CHECK(q.size(1) == o.size(1) && q.size(2) == o.size(2) && q.size(3) == o.size(3),
                "attn_cpu: o must share q's shape");
    TORCH_CHECK(k.size(1) == v.size(1) && k.size(2) == v.size(2) && k.size(3) == q.size(3),
                "attn_cpu: k/v shape mismatch");
    TORCH_CHECK(q.size(2) % k.size(2) == 0, "attn_cpu: GQA heads must divide evenly");
    const int64_t D = q.size(3);
    TORCH_CHECK(D >= 1 && D <= 1024, "attn_cpu: head_dim out of range");

    if (q.numel() == 0) return;

    const float* sinks_ptr = nullptr;
    if (sinks.has_value())
    {
        const at::Tensor& s = sinks.value();
        TORCH_CHECK(s.device().is_cpu() && s.is_contiguous(), "attn_cpu: sinks must be contiguous CPU");
        TORCH_CHECK(s.dim() == 1 && s.size(0) == q.size(2), "attn_cpu: sinks must be [Hq]");
        TORCH_CHECK(s.scalar_type() == at::kFloat, "attn_cpu: sinks must be float32");
        sinks_ptr = static_cast<const float*>(s.const_data_ptr());
    }

    std::vector<int64_t> seg_pairs;
    const int64_t* segs_ptr = nullptr;
    int64_t max_klen = k.size(1);
    if (cu_seqlens.has_value())
    {
        const at::Tensor& cu = cu_seqlens.value();
        TORCH_CHECK(cu.device().is_cpu() && cu.is_contiguous(), "attn_cpu: cu_seqlens must be contiguous CPU");
        TORCH_CHECK(cu.dim() == 1 && cu.size(0) >= 2, "attn_cpu: cu_seqlens must be 1-D [S+1]");
        TORCH_CHECK(cu.scalar_type() == at::kInt || cu.scalar_type() == at::kLong,
                    "attn_cpu: cu_seqlens must be int32/int64");
        const int64_t S = cu.size(0) - 1;
        const char* base = static_cast<const char*>(cu.const_data_ptr());
        const int sz = (int)cu.element_size();
        std::vector<int64_t> bounds(S + 1);
        for (int64_t i = 0; i <= S; ++i) bounds[i] = ac_read_idx(base + (size_t)i * sz, sz);
        TORCH_CHECK(bounds[0] == 0, "attn_cpu: cu_seqlens must start at 0");
        TORCH_CHECK(bounds[S] == q.size(1) && bounds[S] == k.size(1),
                    "attn_cpu: cu_seqlens must end at Lq == Lk");
        for (int64_t i = 0; i < S; ++i) TORCH_CHECK(bounds[i] <= bounds[i + 1], "attn_cpu: cu_seqlens decreasing");
        seg_pairs.resize((size_t)q.size(1) * 2);
        max_klen = 0;
        for (int64_t i = 0; i < S; ++i)
        {
            const int64_t len = bounds[i + 1] - bounds[i];
            if (len > max_klen) max_klen = len;
            for (int64_t p = bounds[i]; p < bounds[i + 1]; ++p)
            {
                seg_pairs[(size_t)p * 2] = bounds[i];
                seg_pairs[(size_t)p * 2 + 1] = bounds[i + 1];
            }
        }
        if (!seg_pairs.empty()) segs_ptr = seg_pairs.data();
        if (max_klen <= 0) max_klen = 1;
    }

    // Head-major repack [B,L,H,D] -> [B,H,L,D] so K/V stream sequentially (was
    // strided by H*D, 4KB jumps thrashing TLB/prefetcher; vision_gemm measured 5x
    // for the same pattern). Cost O(N*D) vs O(N^2*D) attention, negligible.
    // Q/O share q's shape; K/V share theirs (GQA: Hkv may differ from Hq).
    at::Tensor q2 = q.permute({0, 2, 1, 3}).contiguous();
    at::Tensor k2 = k.permute({0, 2, 1, 3}).contiguous();
    at::Tensor v2 = v.permute({0, 2, 1, 3}).contiguous();
    at::Tensor o2 = at::empty({o.size(0), o.size(2), o.size(1), o.size(3)}, o.options());
    AcCtx c{};
    c.q = static_cast<const char*>(q2.const_data_ptr());
    c.k = static_cast<const char*>(k2.const_data_ptr());
    c.v = static_cast<const char*>(v2.const_data_ptr());
    c.o = static_cast<char*>(o2.mutable_data_ptr());
    c.qesz = (int)q2.element_size();
    c.kesz = (int)k2.element_size();
    c.vesz = (int)v2.element_size();
    c.oesz = (int)o2.element_size();
    c.qdt = ac_dtype(q2.scalar_type());
    c.kdt = ac_dtype(k2.scalar_type());
    c.vdt = ac_dtype(v2.scalar_type());
    c.odt = ac_dtype(o2.scalar_type());
    c.B = q.size(0);
    c.Lq = q.size(1);
    c.Lk = k.size(1);
    c.Hq = q.size(2);
    c.Hkv = k.size(2);
    c.D = D;
    c.group = c.Hq / c.Hkv;
    c.scale = (float)sm_scale;
    c.softcap = (float)softcap;
    c.causal = causal;
    c.wl = window_left;
    c.wr = window_right;
    c.q_start = q_start;
    c.sinks = sinks_ptr;
    c.segs = segs_ptr;
    c.isa = ac_pick_isa();

    if (c.isa == AC_AVX512) ac_run<AC_AVX512>(c, max_klen);
    else if (c.isa == AC_AVX2) ac_run<AC_AVX2>(c, max_klen);
    else ac_run<AC_SCALAR>(c, max_klen);
    o.copy_(o2.permute({0, 2, 1, 3}));
}
