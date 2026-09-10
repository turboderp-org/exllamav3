// Native fused RMSNorm for CPU tensors: two streaming passes per row (variance,
// then scale x weight) with fp32 accumulation, matching norm.cu semantics including
// the RES_IN half-rounding rule (the r writeback is re-read, never the float sum)
// and bias-on-weights folding.
//
// Vector paths (AVX-512 x16 / AVX2 x8, f16c half conversion, integer-shift bf16
// widening) cover f32/f16 inputs, weights and outputs; bf16 outputs and scalar-only
// hosts take the threaded scalar kernel, which is also the tail handler. Helpers
// called inside a target-attributed kernel carry the same attribute (GCC will not
// inline a baseline-ISA callee into an AVX-512 caller).

#include "rms_norm_cpu.h"
#include "moe_mul1.h"
#include "exl3_common.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

enum RnDt : uint8_t { RN_F32, RN_F16, RN_BF16 };
enum RnIsa : uint8_t { RN_SCALAR, RN_AVX2, RN_AVX512 };

constexpr int RN_RES_NONE = 0;
constexpr int RN_RES_POST = 1;
constexpr int RN_RES_IN = 2;

constexpr int RN_RND = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

RnDt rn_dtype(at::ScalarType t)
{
    if (t == at::kFloat) return RN_F32;
    if (t == at::kHalf) return RN_F16;
    return RN_BF16;
}

RnIsa rn_pick_isa()
{
    static const RnIsa v = [] {
        if (const char* e = std::getenv("EXL3_RMSNORM_CPU_ISA"))
        {
            if (std::strcmp(e, "scalar") == 0) return RN_SCALAR;
            if (std::strcmp(e, "avx2") == 0) return RN_AVX2;
            if (std::strcmp(e, "avx512") == 0) return RN_AVX512;
        }
        if (exl3_moe_cpu_has_avx512()) return RN_AVX512;
        if (exl3_moe_cpu_has_avx2()) return RN_AVX2;
        return RN_SCALAR;
    }();
    return v;
}

inline float rn_load_s(const char* p, RnDt dt)
{
    if (dt == RN_F32)
    {
        float f;
        std::memcpy(&f, p, 4);
        return f;
    }
    uint16_t b;
    std::memcpy(&b, p, 2);
    if (dt == RN_F16) return exl3_common::half_to_float(b);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline void rn_store_s(char* p, RnDt dt, float f)
{
    if (dt == RN_F32)
    {
        std::memcpy(p, &f, 4);
        return;
    }
    if (dt == RN_F16)
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

// -------------------------------------------------------------------------------------------
// Vector loads/stores (f32/f16; bf16 outputs are scalar-routed by the dispatcher)
// -------------------------------------------------------------------------------------------

template <RnDt DT>
EXL3_CPU_TARGET_AVX2
inline __m256 rn_ld8(const char* p)
{
    if constexpr (DT == RN_F32) return _mm256_loadu_ps((const float*)p);
    const __m128i v = _mm_loadu_si128((const __m128i*)p);
    if constexpr (DT == RN_F16) return _mm256_cvtph_ps(v);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(v), 16));
}

template <RnDt DT>
EXL3_CPU_TARGET_AVX2
inline void rn_st8(char* p, __m256 v)
{
    if constexpr (DT == RN_F32) { _mm256_storeu_ps((float*)p, v); return; }
    _mm_storeu_si128((__m128i*)p, _mm256_cvtps_ph(v, RN_RND));
}

template <RnDt DT>
EXL3_CPU_TARGET_AVX512
inline __m512 rn_ld16(const char* p)
{
    if constexpr (DT == RN_F32) return _mm512_loadu_ps(p);
    if constexpr (DT == RN_F16) return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)p));
    const __m256i v = _mm256_loadu_si256((const __m256i*)p);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(v), 16));
}

template <RnDt DT>
EXL3_CPU_TARGET_AVX512
inline void rn_st16(char* p, __m512 v)
{
    if constexpr (DT == RN_F32) { _mm512_storeu_ps(p, v); return; }
    _mm256_storeu_si256((__m256i*)p, _mm512_cvtps_ph(v, RN_RND));
}

EXL3_CPU_TARGET_AVX2
inline float rn_hsum8(__m256 v)
{
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

struct RnCtx
{
    const char* x;
    const float* wtab;  // folded (w + bias) * scale, null when unweighted
    char* y;
    const char* r;      // RES_IN source (also written back); null otherwise
    int64_t dim;
    float eps;
    float scale_mul;    // 1 when weighted (folded into wtab), else constant_scale
    int xesz, yesz, resz;
    RnDt xdt, ydt, rdt;
    int mode;
    int64_t groups;
    RnIsa isa;
};

void rn_row_scalar(const RnCtx& c, int64_t row)
{
    const char* xr = c.x + (size_t)row * c.dim * c.xesz;
    char* yr = c.y + (size_t)row * c.dim * c.yesz;
    char* rw = c.mode == RN_RES_IN ? const_cast<char*>(c.r) + (size_t)row * c.dim * c.resz : nullptr;
    const char* rr = rw;
    const float* wr = c.wtab ? c.wtab + (size_t)(row % c.groups) * c.dim : nullptr;

    float acc = 0.0f;
    for (int64_t col = 0; col < c.dim; ++col)
    {
        float s = rn_load_s(xr + (size_t)col * c.xesz, c.xdt);
        if (c.mode == RN_RES_IN)
        {
            s += rn_load_s(rr + (size_t)col * c.resz, c.rdt);
            rn_store_s(rw + (size_t)col * c.resz, c.rdt, s);
            s = rn_load_s(rw + (size_t)col * c.resz, c.rdt);
        }
        acc += s * s;
    }
    const float rmf = c.scale_mul / std::sqrt(acc / (float)c.dim + c.eps);
    for (int64_t col = 0; col < c.dim; ++col)
    {
        float v = c.mode == RN_RES_IN
            ? rn_load_s(rr + (size_t)col * c.resz, c.rdt)
            : rn_load_s(xr + (size_t)col * c.xesz, c.xdt);
        v *= rmf;
        if (wr) v *= wr[col];
        if (c.mode == RN_RES_POST)
        {
            // CUDA rounds the normed value to the output dtype before accumulating
            if (c.ydt != RN_F32) { uint8_t tmp[4]; rn_store_s((char*)tmp, c.ydt, v); v = rn_load_s((char*)tmp, c.ydt); }
            v += rn_load_s(yr + (size_t)col * c.yesz, c.ydt);
        }
        rn_store_s(yr + (size_t)col * c.yesz, c.ydt, v);
    }
}

// RD is only read in RES_IN mode; the dispatcher only instantiates F32/F16 there
// (bf16 r/y rows run scalar), so the round-trip/store below never sees bf16.
template <RnDt XD, RnDt YD, RnDt RD>
EXL3_CPU_TARGET_AVX512
void rn_rows_512(const RnCtx& c, int64_t r0, int64_t r1)
{
    constexpr int64_t V = 16;
    const int64_t dim = c.dim;
    const int64_t vec_end = dim / V * V;
    const int mode = c.mode;
    for (int64_t row = r0; row < r1; ++row)
    {
        const char* xr = c.x + (size_t)row * dim * c.xesz;
        char* yr = c.y + (size_t)row * dim * c.yesz;
        const char* rr = mode == RN_RES_IN ? c.r + (size_t)row * dim * c.resz : nullptr;
        char* rw = const_cast<char*>(rr);
        const float* wr = c.wtab ? c.wtab + (size_t)(row % c.groups) * dim : nullptr;

        __m512 acc = _mm512_setzero_ps();
        for (int64_t col = 0; col < vec_end; col += V)
        {
            __m512 v = rn_ld16<XD>(xr + (size_t)col * c.xesz);
            if (mode == RN_RES_IN)
            {
                v = _mm512_add_ps(v, rn_ld16<RD>(rr + (size_t)col * c.resz));
                if constexpr (RD == RN_F16)
                {
                    const __m256i h = _mm512_cvtps_ph(v, RN_RND);
                    _mm256_storeu_si256((__m256i*)(rw + (size_t)col * c.resz), h);
                    v = _mm512_cvtph_ps(h);
                }
                else
                {
                    _mm512_storeu_ps((float*)(rw + (size_t)col * c.resz), v);
                }
            }
            acc = _mm512_fmadd_ps(v, v, acc);
        }
        float tail = 0.0f;
        for (int64_t col = vec_end; col < dim; ++col)
        {
            float s = rn_load_s(xr + (size_t)col * c.xesz, c.xdt);
            if (mode == RN_RES_IN)
            {
                s += rn_load_s(rr + (size_t)col * c.resz, c.rdt);
                rn_store_s(rw + (size_t)col * c.resz, c.rdt, s);
                s = rn_load_s(rw + (size_t)col * c.resz, c.rdt);
            }
            tail += s * s;
        }
        const float rmf = c.scale_mul / std::sqrt((_mm512_reduce_add_ps(acc) + tail) / (float)dim + c.eps);
        const __m512 vrmf = _mm512_set1_ps(rmf);
        for (int64_t col = 0; col < vec_end; col += V)
        {
            __m512 v = mode == RN_RES_IN
                ? rn_ld16<RD>(rr + (size_t)col * c.resz)
                : rn_ld16<XD>(xr + (size_t)col * c.xesz);
            v = _mm512_mul_ps(v, vrmf);
            if (wr) v = _mm512_mul_ps(v, _mm512_loadu_ps(wr + col));
            if (mode == RN_RES_POST)
            {
                if constexpr (YD == RN_F16) v = _mm512_cvtph_ps(_mm512_cvtps_ph(v, RN_RND));
                v = _mm512_add_ps(v, rn_ld16<YD>(yr + (size_t)col * c.yesz));
            }
            rn_st16<YD>(yr + (size_t)col * c.yesz, v);
        }
        for (int64_t col = vec_end; col < dim; ++col)
        {
            float v = mode == RN_RES_IN
                ? rn_load_s(rr + (size_t)col * c.resz, c.rdt)
                : rn_load_s(xr + (size_t)col * c.xesz, c.xdt);
            v *= rmf;
            if (wr) v *= wr[col];
            if (mode == RN_RES_POST)
            {
                if (c.ydt != RN_F32) { uint8_t tmp[4]; rn_store_s((char*)tmp, c.ydt, v); v = rn_load_s((char*)tmp, c.ydt); }
                v += rn_load_s(yr + (size_t)col * c.yesz, c.ydt);
            }
            rn_store_s(yr + (size_t)col * c.yesz, c.ydt, v);
        }
    }
}

template <RnDt XD, RnDt YD, RnDt RD>
EXL3_CPU_TARGET_AVX2
void rn_rows_256(const RnCtx& c, int64_t r0, int64_t r1)
{
    constexpr int64_t V = 8;
    const int64_t dim = c.dim;
    const int64_t vec_end = dim / V * V;
    const int mode = c.mode;
    for (int64_t row = r0; row < r1; ++row)
    {
        const char* xr = c.x + (size_t)row * dim * c.xesz;
        char* yr = c.y + (size_t)row * dim * c.yesz;
        const char* rr = mode == RN_RES_IN ? c.r + (size_t)row * dim * c.resz : nullptr;
        char* rw = const_cast<char*>(rr);
        const float* wr = c.wtab ? c.wtab + (size_t)(row % c.groups) * dim : nullptr;

        __m256 acc = _mm256_setzero_ps();
        for (int64_t col = 0; col < vec_end; col += V)
        {
            __m256 v = rn_ld8<XD>(xr + (size_t)col * c.xesz);
            if (mode == RN_RES_IN)
            {
                v = _mm256_add_ps(v, rn_ld8<RD>(rr + (size_t)col * c.resz));
                if constexpr (RD == RN_F16)
                {
                    const __m128i h = _mm256_cvtps_ph(v, RN_RND);
                    _mm_storeu_si128((__m128i*)(rw + (size_t)col * c.resz), h);
                    v = _mm256_cvtph_ps(h);
                }
                else
                {
                    _mm256_storeu_ps((float*)(rw + (size_t)col * c.resz), v);
                }
            }
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        float tail = 0.0f;
        for (int64_t col = vec_end; col < dim; ++col)
        {
            float s = rn_load_s(xr + (size_t)col * c.xesz, c.xdt);
            if (mode == RN_RES_IN)
            {
                s += rn_load_s(rr + (size_t)col * c.resz, c.rdt);
                rn_store_s(rw + (size_t)col * c.resz, c.rdt, s);
                s = rn_load_s(rw + (size_t)col * c.resz, c.rdt);
            }
            tail += s * s;
        }
        const float rmf = c.scale_mul / std::sqrt((rn_hsum8(acc) + tail) / (float)dim + c.eps);
        const __m256 vrmf = _mm256_set1_ps(rmf);
        for (int64_t col = 0; col < vec_end; col += V)
        {
            __m256 v = mode == RN_RES_IN
                ? rn_ld8<RD>(rr + (size_t)col * c.resz)
                : rn_ld8<XD>(xr + (size_t)col * c.xesz);
            v = _mm256_mul_ps(v, vrmf);
            if (wr) v = _mm256_mul_ps(v, _mm256_loadu_ps(wr + col));
            if (mode == RN_RES_POST)
            {
                if constexpr (YD == RN_F16) v = _mm256_cvtph_ps(_mm256_cvtps_ph(v, RN_RND));
                v = _mm256_add_ps(v, rn_ld8<YD>(yr + (size_t)col * c.yesz));
            }
            rn_st8<YD>(yr + (size_t)col * c.yesz, v);
        }
        for (int64_t col = vec_end; col < dim; ++col)
        {
            float v = mode == RN_RES_IN
                ? rn_load_s(rr + (size_t)col * c.resz, c.rdt)
                : rn_load_s(xr + (size_t)col * c.xesz, c.xdt);
            v *= rmf;
            if (wr) v *= wr[col];
            if (mode == RN_RES_POST)
            {
                if (c.ydt != RN_F32) { uint8_t tmp[4]; rn_store_s((char*)tmp, c.ydt, v); v = rn_load_s((char*)tmp, c.ydt); }
                v += rn_load_s(yr + (size_t)col * c.yesz, c.ydt);
            }
            rn_store_s(yr + (size_t)col * c.yesz, c.ydt, v);
        }
    }
}

template <RnDt XD, RnDt YD, RnDt RD>
void rn_call_rows(const RnCtx& c, int64_t r0, int64_t r1)
{
    if (c.isa == RN_AVX512) rn_rows_512<XD, YD, RD>(c, r0, r1);
    else rn_rows_256<XD, YD, RD>(c, r0, r1);
}

template <RnDt XD>
void rn_d_yd(const RnCtx& c, int64_t r0, int64_t r1)
{
    const RnDt yd = c.ydt;
    const RnDt rd = c.mode == RN_RES_IN ? c.rdt : RN_F32;
    if (yd == RN_F32 && rd == RN_F32) rn_call_rows<XD, RN_F32, RN_F32>(c, r0, r1);
    else if (yd == RN_F32) rn_call_rows<XD, RN_F32, RN_F16>(c, r0, r1);
    else if (rd == RN_F32) rn_call_rows<XD, RN_F16, RN_F32>(c, r0, r1);
    else rn_call_rows<XD, RN_F16, RN_F16>(c, r0, r1);
}

void rn_range(const RnCtx& c, int64_t r0, int64_t r1)
{
    // bf16 outputs (and bf16 res_in writeback) run the threaded scalar kernel;
    // everything else runs vectorized at the detected ISA.
    if (c.isa == RN_SCALAR || c.ydt == RN_BF16 ||
        (c.mode == RN_RES_IN && c.rdt == RN_BF16))
    {
        for (int64_t row = r0; row < r1; ++row) rn_row_scalar(c, row);
        return;
    }
    if (c.xdt == RN_F32) rn_d_yd<RN_F32>(c, r0, r1);
    else if (c.xdt == RN_F16) rn_d_yd<RN_F16>(c, r0, r1);
    else rn_d_yd<RN_BF16>(c, r0, r1);
}
void rn_run(const RnCtx& c, int64_t rows)
{
    if (rows <= 0) return;
    auto process = [&](int64_t b, int64_t e) { rn_range(c, b, e); };
    struct Tile
    {
        static void tile(void* ctx, int64_t b, int64_t e)
        {
            (*static_cast<decltype(&process)>(ctx))(b, e);
        }
    };
    cpu_pool_run_tiles(rows, &Tile::tile, (void*)&process);
}

void rn_fill_ctx(
    RnCtx& c,
    const at::Tensor& x,
    const at::Tensor& y,
    const c10::optional<at::Tensor>& r,
    int mode)
{
    c.x = static_cast<const char*>(x.const_data_ptr());
    c.y = static_cast<char*>(y.mutable_data_ptr());
    c.r = r.has_value() ? static_cast<const char*>(r.value().const_data_ptr()) : nullptr;
    c.dim = x.size(-1);
    c.xesz = x.element_size();
    c.yesz = y.element_size();
    c.resz = r.has_value() ? r.value().element_size() : 4;
    c.xdt = rn_dtype(x.scalar_type());
    c.ydt = rn_dtype(y.scalar_type());
    c.rdt = r.has_value() ? rn_dtype(r.value().scalar_type()) : RN_F32;
    c.mode = mode;
    c.isa = rn_pick_isa();
}

void rn_check_pair(const at::Tensor& x, const at::Tensor& y, const char* name)
{
    TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu(),
                name, ": x and y must be on CPU");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                name, ": x and y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), name, ": x and y shape mismatch");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                x.scalar_type() == at::kBFloat16, name, ": x dtype must be f32/f16/bf16");
    TORCH_CHECK(y.scalar_type() == at::kFloat || y.scalar_type() == at::kHalf ||
                y.scalar_type() == at::kBFloat16, name, ": y dtype must be f32/f16/bf16");
    TORCH_CHECK(x.size(-1) > 0, name, ": last dim must be non-empty");
}

const float* rn_fold_table(
    std::vector<float>& wtab,
    const c10::optional<at::Tensor>& w,
    int64_t dim,
    int64_t groups,
    float constant_bias,
    float constant_scale,
    float& scale_mul,
    const char* name)
{
    if (!w.has_value()) return nullptr;
    const at::Tensor& wt = w.value();
    TORCH_CHECK(wt.device().is_cpu() && wt.is_contiguous(), name, ": w must be CPU contiguous");
    TORCH_CHECK(wt.scalar_type() == at::kFloat || wt.scalar_type() == at::kHalf ||
                wt.scalar_type() == at::kBFloat16, name, ": w dtype must be f32/f16/bf16");
    const int64_t wn = wt.numel();
    TORCH_CHECK(wn == dim || (groups > 1 && wn == groups * dim),
                name, ": w numel must be dim (or groups * dim)");
    const RnDt wdt = rn_dtype(wt.scalar_type());
    const char* wp = static_cast<const char*>(wt.const_data_ptr());
    const int wesz = wt.element_size();
    wtab.resize((size_t)wn);
    for (int64_t i = 0; i < wn; ++i)
        wtab[(size_t)i] = (rn_load_s(wp + (size_t)i * wesz, wdt) + constant_bias) * constant_scale;
    scale_mul = 1.0f;
    return wtab.data();
}

} // namespace

void rms_norm_cpu(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool add_residual,
    int64_t w_groups)
{
    rn_check_pair(x, y, "rms_norm_cpu");
    TORCH_CHECK(w_groups >= 1, "rms_norm_cpu: w_groups must be >= 1");
    const int64_t dim = x.size(-1);
    const int64_t rows = x.numel() / dim;
    if (rows == 0) return;

    RnCtx c{};
    rn_fill_ctx(c, x, y, c10::nullopt, add_residual ? RN_RES_POST : RN_RES_NONE);
    c.eps = epsilon;
    c.groups = w_groups;
    std::vector<float> wtab;
    float scale_mul = constant_scale;
    c.wtab = rn_fold_table(wtab, w, dim, w_groups, constant_bias, constant_scale, scale_mul,
                           "rms_norm_cpu");
    c.scale_mul = scale_mul;
    rn_run(c, rows);
}

void rms_norm_res_in_cpu(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    at::Tensor r,
    float epsilon,
    float constant_bias,
    float constant_scale)
{
    rn_check_pair(x, y, "rms_norm_res_in_cpu");
    TORCH_CHECK(r.device().is_cpu() && r.is_contiguous(), "rms_norm_res_in_cpu: r must be CPU contiguous");
    TORCH_CHECK(r.sizes() == x.sizes(), "rms_norm_res_in_cpu: r and x shape mismatch");
    TORCH_CHECK(r.scalar_type() == at::kFloat || r.scalar_type() == at::kHalf ||
                r.scalar_type() == at::kBFloat16, "rms_norm_res_in_cpu: r dtype must be f32/f16/bf16");
    const int64_t dim = x.size(-1);
    const int64_t rows = x.numel() / dim;
    if (rows == 0) return;

    RnCtx c{};
    rn_fill_ctx(c, x, y, r, RN_RES_IN);
    c.eps = epsilon;
    c.groups = 1;
    std::vector<float> wtab;
    float scale_mul = constant_scale;
    c.wtab = rn_fold_table(wtab, w, dim, 1, constant_bias, constant_scale, scale_mul,
                           "rms_norm_res_in_cpu");
    c.scale_mul = scale_mul;
    rn_run(c, rows);
}
