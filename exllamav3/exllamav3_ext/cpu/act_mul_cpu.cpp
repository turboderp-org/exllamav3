// Native fused gate*up activation for CPU tensors: z = act(g) * u in fp32, single round
// to the output dtype, with the CUDA act_limit clamp folded in (no temporaries, one pass).
//
// Vector structure follows the ggml CPU idiom (ggml/src/ggml-cpu/vec.h ggml_v_silu and the
// ARM limited-optimized-routine vector exp it builds on): range-reduction polynomial exp,
// silu as g / (1 + exp(-g)), gelu's tanh through the same exp. Reimplemented here for the
// repo's conventions; no vendored code. Reference for the algorithm family only.
//
// Codegen rules from vision_gemm.cpp apply: helpers called inside a target-attributed
// kernel carry the same target attribute (GCC will not inline a baseline-ISA callee into
// an AVX-512 caller, turning every exp into a real call). Loads/stores mirror
// rms_norm_cpu.cpp (f16c half conversion, integer-shift bf16 widening, RNE stores).

#include "act_mul_cpu.h"
#include "exl3_common.h"
#include "moe_mul1.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfloat>

namespace {

enum AmDt : uint8_t { AM_F32, AM_F16, AM_BF16 };
enum AmIsa : uint8_t { AM_SCALAR, AM_AVX2, AM_AVX512 };
enum AmAct : uint8_t { AM_SILU, AM_GELU, AM_RELU2 };

constexpr int64_t AM_CHUNK = 4096;
constexpr int AM_RND = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
constexpr float AM_GELU_C = 0.797884560803f;  // sqrt(2/pi), same as CUDA _gelu

AmDt am_dtype(at::ScalarType t)
{
    if (t == at::kFloat) return AM_F32;
    if (t == at::kHalf) return AM_F16;
    return AM_BF16;
}

AmIsa am_pick_isa()
{
    static const AmIsa v = [] {
        if (const char* e = std::getenv("EXL3_ACT_MUL_CPU_ISA"))
        {
            if (std::strcmp(e, "scalar") == 0) return AM_SCALAR;
            if (std::strcmp(e, "avx2") == 0) return AM_AVX2;
            if (std::strcmp(e, "avx512") == 0) return AM_AVX512;
        }
        if (exl3_moe_cpu_has_avx512()) return AM_AVX512;
        if (exl3_moe_cpu_has_avx2()) return AM_AVX2;
        return AM_SCALAR;
    }();
    return v;
}

// -------------------------------------------------------------------------------------------
// Scalar lane conversion (tails, bf16 outputs, scalar fallback)
// -------------------------------------------------------------------------------------------

inline float am_load_s(const char* p, AmDt dt)
{
    if (dt == AM_F32)
    {
        float f;
        std::memcpy(&f, p, 4);
        return f;
    }
    uint16_t b;
    std::memcpy(&b, p, 2);
    if (dt == AM_F16) return exl3_common::half_to_float(b);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline void am_store_s(char* p, AmDt dt, float f)
{
    if (dt == AM_F32)
    {
        std::memcpy(p, &f, 4);
        return;
    }
    if (dt == AM_F16)
    {
        const c10::Half h(f);
        const uint16_t b = h.x;
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

inline float am_silu_s(float g) { return g / (1.0f + std::exp(-g)); }

inline float am_gelu_s(float g)
{
    const float t = AM_GELU_C * (g + 0.044715f * g * g * g);
    const float e = std::exp(-2.0f * std::fabs(t));
    const float th = std::copysign((1.0f - e) / (1.0f + e), t);
    return 0.5f * g * (1.0f + th);
}

inline float am_relu2_s(float g)
{
    const float t = g > 0.0f ? g : 0.0f;
    return t * t;
}

inline float am_act_s(AmAct act, float g)
{
    if (act == AM_SILU) return am_silu_s(g);
    if (act == AM_GELU) return am_gelu_s(g);
    return am_relu2_s(g);
}

// -------------------------------------------------------------------------------------------
// Vector loads/stores (f32/f16 in, f32/f16 out; bf16 outputs are scalar-routed)
// -------------------------------------------------------------------------------------------

template <AmDt DT>
EXL3_CPU_TARGET_AVX2
inline __m256 am_ld8(const char* p)
{
    if constexpr (DT == AM_F32) return _mm256_loadu_ps((const float*)p);
    if constexpr (DT == AM_F16) return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)p));
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16));
}

EXL3_CPU_TARGET_AVX2
inline void am_st8(char* p, __m256 v, AmDt dt)
{
    if (dt == AM_F32) { _mm256_storeu_ps((float*)p, v); return; }
    _mm_storeu_si128((__m128i*)p, _mm256_cvtps_ph(v, AM_RND));
}

template <AmDt DT>
EXL3_CPU_TARGET_AVX512
inline __m512 am_ld16(const char* p)
{
    if constexpr (DT == AM_F32) return _mm512_loadu_ps(p);
    if constexpr (DT == AM_F16) return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)p));
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}

EXL3_CPU_TARGET_AVX512
inline void am_st16(char* p, __m512 v, AmDt dt)
{
    if (dt == AM_F32) { _mm512_storeu_ps(p, v); return; }
    _mm256_storeu_si256((__m256i*)p, _mm512_cvtps_ph(v, AM_RND));
}

// -------------------------------------------------------------------------------------------
// Vector exp, range-reduction polynomial family (same class as ggml's ggml_v_expf):
// ~1.5 + 0.5 ulps, >88.38 flushes to inf, <-103.97 flushes to 0. Plenty for a sigmoid/
// tanh gate inside a 3e-3 end-to-end envelope.
// -------------------------------------------------------------------------------------------

EXL3_CPU_TARGET_AVX2
inline __m256 am_exp8(__m256 x)
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
inline __m512 am_exp16(__m512 x)
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
// Vector act bodies (one per width; act selected at compile time, clamp folded in)
// -------------------------------------------------------------------------------------------

template <AmAct ACT>
EXL3_CPU_TARGET_AVX2
inline __m256 am_act8(__m256 g, __m256 u, float limit)
{
    __m256 s;
    if constexpr (ACT == AM_SILU)
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        s = _mm256_div_ps(g, _mm256_add_ps(one, am_exp8(_mm256_sub_ps(_mm256_setzero_ps(), g))));
    }
    else if constexpr (ACT == AM_GELU)
    {
        const __m256 g2 = _mm256_mul_ps(g, g);
        const __m256 t = _mm256_mul_ps(_mm256_set1_ps(AM_GELU_C),
                                       _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), _mm256_mul_ps(g2, g), g));
        const __m256 e2 = am_exp8(_mm256_add_ps(t, t));
        // Clamp before the division: with -Ofast the compiler reassociates 1 - 2/(e2+1)
        // across the quotient, so an infinite e2 must never reach it (inf/inf = NaN).
        // min is a single instruction, invisible to reassociation; the clamped value
        // still rounds th to exactly 1.
        const __m256 e2c = _mm256_min_ps(e2, _mm256_set1_ps(FLT_MAX));
        const __m256 th = _mm256_sub_ps(_mm256_set1_ps(1.0f),
                                        _mm256_div_ps(_mm256_set1_ps(2.0f),
                                                      _mm256_add_ps(e2c, _mm256_set1_ps(1.0f))));
        s = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                          _mm256_mul_ps(g, _mm256_add_ps(_mm256_set1_ps(1.0f), th)));
    }
    else
    {
        const __m256 t = _mm256_max_ps(g, _mm256_setzero_ps());
        s = _mm256_mul_ps(t, t);
    }
    // Asymmetric act_limit (CUDA order): u symmetric, activated gate from above only.
    // silu_oai (gpt-oss) intentionally not here: only the MoE path uses it.
    if (limit != 0.0f)
    {
        const __m256 lo = _mm256_set1_ps(-limit);
        const __m256 hi = _mm256_set1_ps(limit);
        u = _mm256_min_ps(_mm256_max_ps(u, lo), hi);
        s = _mm256_min_ps(s, hi);
    }
    return _mm256_mul_ps(s, u);
}

template <AmAct ACT>
EXL3_CPU_TARGET_AVX512
inline __m512 am_act16(__m512 g, __m512 u, float limit)
{
    __m512 s;
    if constexpr (ACT == AM_SILU)
    {
        const __m512 one = _mm512_set1_ps(1.0f);
        s = _mm512_div_ps(g, _mm512_add_ps(one, am_exp16(_mm512_sub_ps(_mm512_setzero_ps(), g))));
    }
    else if constexpr (ACT == AM_GELU)
    {
        const __m512 g2 = _mm512_mul_ps(g, g);
        const __m512 t = _mm512_mul_ps(_mm512_set1_ps(AM_GELU_C),
                                       _mm512_fmadd_ps(_mm512_set1_ps(0.044715f), _mm512_mul_ps(g2, g), g));
        const __m512 e2 = am_exp16(_mm512_add_ps(t, t));
        // Same -Ofast reassociation guard as the AVX2 branch above: e2 must be finite
        // before 1 - 2/(e2+1), else the combined quotient evaluates inf/inf = NaN.
        const __m512 e2c = _mm512_min_ps(e2, _mm512_set1_ps(FLT_MAX));
        const __m512 th = _mm512_sub_ps(_mm512_set1_ps(1.0f),
                                        _mm512_div_ps(_mm512_set1_ps(2.0f),
                                                      _mm512_add_ps(e2c, _mm512_set1_ps(1.0f))));
        s = _mm512_mul_ps(_mm512_set1_ps(0.5f),
                          _mm512_mul_ps(g, _mm512_add_ps(_mm512_set1_ps(1.0f), th)));
    }
    else
    {
        const __m512 t = _mm512_max_ps(g, _mm512_setzero_ps());
        s = _mm512_mul_ps(t, t);
    }
    if (limit != 0.0f)
    {
        const __m512 lo = _mm512_set1_ps(-limit);
        const __m512 hi = _mm512_set1_ps(limit);
        u = _mm512_min_ps(_mm512_max_ps(u, lo), hi);
        s = _mm512_min_ps(s, hi);
    }
    return _mm512_mul_ps(s, u);
}

// -------------------------------------------------------------------------------------------
// Tiling driver
// -------------------------------------------------------------------------------------------

struct AmCtx
{
    const char* g;
    const char* u;
    char* a;
    int64_t n;
    int gesz, uesz, aesz;
    AmDt gdt, udt, adt;
    float limit;
    AmAct act;
    AmIsa isa;
};

inline void am_elem_scalar(const AmCtx& c, int64_t i)
{
    // Block reads before the write: a may alias g or u.
    const float g = am_load_s(c.g + (size_t)i * c.gesz, c.gdt);
    const float u = am_load_s(c.u + (size_t)i * c.uesz, c.udt);
    float s = am_act_s(c.act, g);
    float uu = u;
    if (c.limit != 0.0f)
    {
        if (uu < -c.limit) uu = -c.limit;
        if (uu > c.limit) uu = c.limit;
        if (s > c.limit) s = c.limit;
    }
    am_store_s(c.a + (size_t)i * c.aesz, c.adt, s * uu);
}

template <AmAct ACT, AmDt GD, AmDt UD, AmDt AD>
EXL3_CPU_TARGET_AVX512
void am_range_512(const AmCtx& c, int64_t b, int64_t e)
{
    constexpr int64_t V = 16;
    const int64_t vec_end = b + (e - b) / V * V;
    int64_t i = b;
    for (; i < vec_end; i += V)
    {
        // Both inputs fully loaded before the (possibly aliasing) store.
        const __m512 g = am_ld16<GD>(c.g + (size_t)i * c.gesz);
        const __m512 u = am_ld16<UD>(c.u + (size_t)i * c.uesz);
        am_st16(c.a + (size_t)i * c.aesz, am_act16<ACT>(g, u, c.limit), AD);
    }
    for (; i < e; ++i) am_elem_scalar(c, i);
}

template <AmAct ACT, AmDt GD, AmDt UD, AmDt AD>
EXL3_CPU_TARGET_AVX2
void am_range_256(const AmCtx& c, int64_t b, int64_t e)
{
    constexpr int64_t V = 8;
    const int64_t vec_end = b + (e - b) / V * V;
    int64_t i = b;
    for (; i < vec_end; i += V)
    {
        const __m256 g = am_ld8<GD>(c.g + (size_t)i * c.gesz);
        const __m256 u = am_ld8<UD>(c.u + (size_t)i * c.uesz);
        am_st8(c.a + (size_t)i * c.aesz, am_act8<ACT>(g, u, c.limit), c.adt);
    }
    for (; i < e; ++i) am_elem_scalar(c, i);
}

template <AmAct ACT>
void am_call_ad(const AmCtx& c, int64_t b, int64_t e)
{
#define AM_CALL_C(GD, UD, AD) \
    (c.isa == AM_AVX512 ? am_range_512<ACT, GD, UD, AD>(c, b, e) \
                        : am_range_256<ACT, GD, UD, AD>(c, b, e))
#define AM_SWITCH_AD(GD, UD) \
    switch (c.adt) \
    { \
        case AM_F32: AM_CALL_C(GD, UD, AM_F32); break; \
        default: AM_CALL_C(GD, UD, AM_F16); break; \
    }
#define AM_SWITCH_UD(GD) \
    switch (c.udt) \
    { \
        case AM_F32: AM_SWITCH_AD(GD, AM_F32); break; \
        case AM_F16: AM_SWITCH_AD(GD, AM_F16); break; \
        default: AM_SWITCH_AD(GD, AM_BF16); break; \
    }
    switch (c.gdt)
    {
        case AM_F32: AM_SWITCH_UD(AM_F32); break;
        case AM_F16: AM_SWITCH_UD(AM_F16); break;
        default: AM_SWITCH_UD(AM_BF16); break;
    }
#undef AM_CALL_C
#undef AM_SWITCH_AD
#undef AM_SWITCH_UD
}

inline void am_range(const AmCtx& c, int64_t b, int64_t e)
{
    // bf16 outputs run the threaded scalar kernel (same rule as rms_norm); vector stores
    // only widen to f32/f16. Scalar-only hosts take the scalar kernel for everything.
    if (c.isa == AM_SCALAR || c.adt == AM_BF16)
    {
        for (int64_t i = b; i < e; ++i) am_elem_scalar(c, i);
        return;
    }
    if (c.act == AM_SILU) am_call_ad<AM_SILU>(c, b, e);
    else if (c.act == AM_GELU) am_call_ad<AM_GELU>(c, b, e);
    else am_call_ad<AM_RELU2>(c, b, e);
}

inline void am_run(const AmCtx& c)
{
    if (c.n <= 0) return;
    auto process = [&](int64_t b, int64_t e)
    {
        const int64_t b0 = b * AM_CHUNK;
        int64_t e0 = e * AM_CHUNK;
        if (e0 > c.n) e0 = c.n;
        am_range(c, b0, e0);
    };
    struct Tile
    {
        static void tile(void* ctx, int64_t b, int64_t e)
        {
            (*static_cast<decltype(&process)>(ctx))(b, e);
        }
    };
    const int64_t total = (c.n + AM_CHUNK - 1) / AM_CHUNK;
    cpu_pool_run_tiles(total, &Tile::tile, (void*)&process);
}

inline void am_check(const at::Tensor& g, const at::Tensor& u, const at::Tensor& a, const char* name)
{
    TORCH_CHECK(g.device().is_cpu() && u.device().is_cpu() && a.device().is_cpu(),
                name, ": g, u and a must be on CPU");
    TORCH_CHECK(g.is_contiguous() && u.is_contiguous() && a.is_contiguous(),
                name, ": g, u and a must be contiguous");
    TORCH_CHECK(g.is_floating_point() && u.is_floating_point() && a.is_floating_point(),
                name, ": g, u and a must be floating point");
    for (const at::Tensor* t : {&g, &u, &a})
        TORCH_CHECK(t->scalar_type() == at::kFloat || t->scalar_type() == at::kHalf ||
                    t->scalar_type() == at::kBFloat16, name, ": g, u and a dtype must be f32/f16/bf16");
}

inline void am_dispatch(const at::Tensor& g, const at::Tensor& u, at::Tensor& a,
                        float act_limit, AmAct act, const char* name)
{
    am_check(g, u, a, name);
    const int64_t n = g.numel();
    if (n == 0) return;
    AmCtx c{};
    c.g = static_cast<const char*>(g.const_data_ptr());
    c.u = static_cast<const char*>(u.const_data_ptr());
    c.a = static_cast<char*>(a.mutable_data_ptr());
    c.n = n;
    c.gesz = g.element_size();
    c.uesz = u.element_size();
    c.aesz = a.element_size();
    c.gdt = am_dtype(g.scalar_type());
    c.udt = am_dtype(u.scalar_type());
    c.adt = am_dtype(a.scalar_type());
    c.limit = act_limit;
    c.act = act;
    c.isa = am_pick_isa();
    am_run(c);
}

} // namespace

void silu_mul_cpu(const at::Tensor& g, const at::Tensor& u, at::Tensor& a, float act_limit)
{
    am_dispatch(g, u, a, act_limit, AM_SILU, "silu_mul_cpu");
}

void gelu_mul_cpu(const at::Tensor& g, const at::Tensor& u, at::Tensor& a, float act_limit)
{
    am_dispatch(g, u, a, act_limit, AM_GELU, "gelu_mul_cpu");
}

void relu2_mul_cpu(const at::Tensor& g, const at::Tensor& u, at::Tensor& a, float act_limit)
{
    am_dispatch(g, u, a, act_limit, AM_RELU2, "relu2_mul_cpu");
}
