// Fused rotary position embedding for CPU tensors: q/k rotation in fp32 with a single
// round to the output dtype, sin/cos hoisted per (token, section) like the CUDA
// rope_kernel sin_cache/cos_cache (O(seq*P) transcendentals, O(seq*heads*P) pure
// mul/add rotates).
//
// Vector structure follows the repo's CPU idiom (act_mul_cpu.cpp, rms_norm_cpu.cpp):
// target-attributed kernels per width, helpers called inside a kernel carry the same
// target attribute (GCC will not inline a baseline-ISA callee into an AVX-512 caller),
// f16c half conversion, integer-shift bf16 widening with RNE pack stores. Algorithm
// ordering mirrors ggml's ggml_compute_forward_rope_flt (per-row sin/cos cache, then
// rotate pairs, untouched tails memcopied); reimplemented here for the repo's
// conventions, no vendored code. GPT-J pairs rotate in place without deinterleaving:
// out = v * c_dup + swap_adjacent(v) * w, w = [-s, +s, ...] (verified per lane:
// out[2j] = a*c - b*s, out[2j+1] = b*c + a*s).

#include "rope_cpu.h"
#include "exl3_common.h"
#include "moe_mul1.h"

#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int ROPESTYLE_GPTJ = 1;
constexpr int ROPESTYLE_NEOX = 2;
constexpr int ROPE_CPU_MAX_ROTATE_DIMS = 4;
constexpr int64_t RP_TOKEN_CHUNK = 4;
constexpr int RP_RND = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

enum RpDt : uint8_t { RP_F32, RP_F16, RP_BF16 };
enum RpIsa : uint8_t { RP_SCALAR, RP_AVX2, RP_AVX512 };

RpDt rp_dtype(at::ScalarType t)
{
    if (t == at::kFloat) return RP_F32;
    if (t == at::kHalf) return RP_F16;
    return RP_BF16;
}

RpIsa rp_pick_isa()
{
    static const RpIsa v = [] {
        if (const char* e = std::getenv("EXL3_ROPE_CPU_ISA"))
        {
            if (std::strcmp(e, "scalar") == 0) return RP_SCALAR;
            if (std::strcmp(e, "avx2") == 0) return RP_AVX2;
            if (std::strcmp(e, "avx512") == 0) return RP_AVX512;
        }
        if (exl3_moe_cpu_has_avx512()) return RP_AVX512;
        if (exl3_moe_cpu_has_avx2()) return RP_AVX2;
        return RP_SCALAR;
    }();
    return v;
}

// -------------------------------------------------------------------------------------------
// Scalar lane conversion (tails, bf16 outputs, scalar fallback)
// -------------------------------------------------------------------------------------------

inline float rp_load_s(const char* p, RpDt dt)
{
    if (dt == RP_F32)
    {
        float f;
        std::memcpy(&f, p, 4);
        return f;
    }
    uint16_t b;
    std::memcpy(&b, p, 2);
    if (dt == RP_F16) return exl3_common::half_to_float(b);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline void rp_store_s(char* p, RpDt dt, float f)
{
    if (dt == RP_F32)
    {
        std::memcpy(p, &f, 4);
        return;
    }
    if (dt == RP_F16)
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

inline int64_t rp_read_idx(const char* p, int sz)
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

// -------------------------------------------------------------------------------------------
// Vector loads/stores (f32/f16 in, f32/f16 out; bf16 outputs are scalar-routed, same rule
// as act_mul/rms_norm: vector stores only narrow to f32/f16)
// -------------------------------------------------------------------------------------------

template <RpDt DT>
EXL3_CPU_TARGET_AVX2
inline __m256 rp_ld8(const char* p)
{
    if constexpr (DT == RP_F32) return _mm256_loadu_ps((const float*)p);
    if constexpr (DT == RP_F16) return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)p));
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16));
}

EXL3_CPU_TARGET_AVX2
inline void rp_st8(char* p, __m256 v, RpDt dt)
{
    if (dt == RP_F32) { _mm256_storeu_ps((float*)p, v); return; }
    _mm_storeu_si128((__m128i*)p, _mm256_cvtps_ph(v, RP_RND));
}

template <RpDt DT>
EXL3_CPU_TARGET_AVX512
inline __m512 rp_ld16(const char* p)
{
    if constexpr (DT == RP_F32) return _mm512_loadu_ps(p);
    if constexpr (DT == RP_F16) return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)p));
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}

EXL3_CPU_TARGET_AVX512
inline void rp_st16(char* p, __m512 v, RpDt dt)
{
    if (dt == RP_F32) { _mm512_storeu_ps(p, v); return; }
    _mm256_storeu_si256((__m256i*)p, _mm512_cvtps_ph(v, RP_RND));
}

// -------------------------------------------------------------------------------------------
// Context
// -------------------------------------------------------------------------------------------

struct RpCtx
{
    const char* q;
    char* oq;
    const char* k = nullptr;   // null when the call has no k
    char* ok = nullptr;
    int q_iesz = 0, q_oesz = 0, k_iesz = 0, k_oesz = 0;
    RpDt q_idt = RP_F16, q_odt = RP_F16, k_idt = RP_F16, k_odt = RP_F16;
    int64_t bsz = 0, seq = 0;
    int Hq = 0, Hk = 0, hd = 0;
    int P = 0;                 // rotary width (partial_head_dim)
    int Phalf = 0;             // P / 2
    int R = 1;                 // rotate_dims (section count)
    int style = ROPESTYLE_NEOX;
    float attn = 1.0f;
    bool table = false;
    const float* freq = nullptr;
    int64_t table_seq = 0;     // 3-D table: size(-2); 2-D table: row count (size 0)
    int table_dim = 0;
    int64_t position = 0;
    const char* posp = nullptr;
    int posz = 0;
    const char* pidp = nullptr;
    int pidz = 0;
    int pid_rank = 0;
    RpIsa isa = RP_SCALAR;
};

inline int64_t rp_pos(const RpCtx& c, int64_t b, int64_t s, int rdim)
{
    if (c.pidp)
    {
        if (c.pid_rank == 3)
            return rp_read_idx(c.pidp + ((b * c.seq + s) * c.R + rdim) * c.pidz, c.pidz);
        return rp_read_idx(c.pidp + (b * c.seq + s) * c.pidz, c.pidz);
    }
    if (c.posp)
        return rp_read_idx(c.posp + b * c.posz, c.posz) + s;
    return c.position + s;
}

// -------------------------------------------------------------------------------------------
// Scalar rows (all dtype combinations, bf16 outputs, tails)
// -------------------------------------------------------------------------------------------

inline void rp_row_scalar(
    const char* in, char* out, int hd, int P, int R, int style,
    const float* sin, const float* cos,
    RpDt idt, RpDt odt, int iesz, int oesz)
{
    const int H = P / 2;
    for (int r = 0; r < R; ++r)
    {
        const float* sn = sin + (size_t)r * H;
        const float* cs = cos + (size_t)r * H;
        const char* si = in + (size_t)r * P * iesz;
        char* so = out + (size_t)r * P * oesz;
        if (style == ROPESTYLE_NEOX)
        {
            for (int j = 0; j < H; ++j)
            {
                // Block reads before the writes: out may alias in.
                const float v1 = rp_load_s(si + (size_t)j * iesz, idt);
                const float v2 = rp_load_s(si + (size_t)(j + H) * iesz, idt);
                rp_store_s(so + (size_t)j * oesz, odt, v1 * cs[j] - v2 * sn[j]);
                rp_store_s(so + (size_t)(j + H) * oesz, odt, v2 * cs[j] + v1 * sn[j]);
            }
        }
        else
        {
            for (int j = 0; j < H; ++j)
            {
                const float v1 = rp_load_s(si + (size_t)(2 * j) * iesz, idt);
                const float v2 = rp_load_s(si + (size_t)(2 * j + 1) * iesz, idt);
                rp_store_s(so + (size_t)(2 * j) * oesz, odt, v1 * cs[j] - v2 * sn[j]);
                rp_store_s(so + (size_t)(2 * j + 1) * oesz, odt, v2 * cs[j] + v1 * sn[j]);
            }
        }
    }
    if (R == 1 && P < hd)
    {
        for (int j = P; j < hd; ++j)
            rp_store_s(out + (size_t)j * oesz, odt, rp_load_s(in + (size_t)j * iesz, idt));
    }
}

// -------------------------------------------------------------------------------------------
// Vector rows: NEOX split-half (contiguous halves), GPT-J adjacent pairs (in-place trick)
// -------------------------------------------------------------------------------------------

template <RpDt ID, RpDt OD>
EXL3_CPU_TARGET_AVX512
void rp_row_neox_512(
    const char* in, char* out, int hd, int P, int R,
    const float* sin, const float* cos, int iesz, int oesz)
{
    constexpr int V = 16;
    const int H = P / 2;
    for (int r = 0; r < R; ++r)
    {
        const float* sn = sin + (size_t)r * H;
        const float* cs = cos + (size_t)r * H;
        const char* si = in + (size_t)r * P * iesz;
        char* so = out + (size_t)r * P * oesz;
        int j = 0;
        const int vend = H / V * V;
        for (; j < vend; j += V)
        {
            // Both halves fully loaded before the (possibly aliasing) stores.
            const __m512 v1 = rp_ld16<ID>(si + (size_t)j * iesz);
            const __m512 v2 = rp_ld16<ID>(si + (size_t)(j + H) * iesz);
            const __m512 c = _mm512_loadu_ps(cs + j);
            const __m512 s = _mm512_loadu_ps(sn + j);
            rp_st16(so + (size_t)j * oesz,
                    _mm512_fmsub_ps(v1, c, _mm512_mul_ps(v2, s)), OD);
            rp_st16(so + (size_t)(j + H) * oesz,
                    _mm512_fmadd_ps(v2, c, _mm512_mul_ps(v1, s)), OD);
        }
        for (; j < H; ++j)
        {
            const float v1 = rp_load_s(si + (size_t)j * iesz, ID);
            const float v2 = rp_load_s(si + (size_t)(j + H) * iesz, ID);
            rp_store_s(so + (size_t)j * oesz, OD, v1 * cs[j] - v2 * sn[j]);
            rp_store_s(so + (size_t)(j + H) * oesz, OD, v2 * cs[j] + v1 * sn[j]);
        }
    }
    if (R == 1 && P < hd)
    {
        for (int j = P; j < hd; ++j)
            rp_store_s(out + (size_t)j * oesz, OD, rp_load_s(in + (size_t)j * iesz, ID));
    }
}

template <RpDt ID, RpDt OD>
EXL3_CPU_TARGET_AVX2
void rp_row_neox_256(
    const char* in, char* out, int hd, int P, int R,
    const float* sin, const float* cos, int iesz, int oesz)
{
    constexpr int V = 8;
    const int H = P / 2;
    for (int r = 0; r < R; ++r)
    {
        const float* sn = sin + (size_t)r * H;
        const float* cs = cos + (size_t)r * H;
        const char* si = in + (size_t)r * P * iesz;
        char* so = out + (size_t)r * P * oesz;
        int j = 0;
        const int vend = H / V * V;
        for (; j < vend; j += V)
        {
            const __m256 v1 = rp_ld8<ID>(si + (size_t)j * iesz);
            const __m256 v2 = rp_ld8<ID>(si + (size_t)(j + H) * iesz);
            const __m256 c = _mm256_loadu_ps(cs + j);
            const __m256 s = _mm256_loadu_ps(sn + j);
            rp_st8(so + (size_t)j * oesz,
                   _mm256_fmsub_ps(v1, c, _mm256_mul_ps(v2, s)), OD);
            rp_st8(so + (size_t)(j + H) * oesz,
                   _mm256_fmadd_ps(v2, c, _mm256_mul_ps(v1, s)), OD);
        }
        for (; j < H; ++j)
        {
            const float v1 = rp_load_s(si + (size_t)j * iesz, ID);
            const float v2 = rp_load_s(si + (size_t)(j + H) * iesz, ID);
            rp_store_s(so + (size_t)j * oesz, OD, v1 * cs[j] - v2 * sn[j]);
            rp_store_s(so + (size_t)(j + H) * oesz, OD, v2 * cs[j] + v1 * sn[j]);
        }
    }
    if (R == 1 && P < hd)
    {
        for (int j = P; j < hd; ++j)
            rp_store_s(out + (size_t)j * oesz, OD, rp_load_s(in + (size_t)j * iesz, ID));
    }
}

template <RpDt ID, RpDt OD>
EXL3_CPU_TARGET_AVX512
void rp_row_gptj_512(
    const char* in, char* out, int hd, int P, int R,
    const float* sin, const float* cos, int iesz, int oesz)
{
    // v = [a0 b0 a1 b1 ...]; out = v * c_dup + swap_adjacent(v) * w,
    // c_dup = [c0 c0 c1 c1 ...], w = [-s0 +s0 -s1 +s1 ...].
    const __m512i dup_idx = _mm512_setr_epi32(
        0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512 sign = _mm512_castsi512_ps(_mm512_setr_epi32(
        (int)0x80000000, 0, (int)0x80000000, 0,
        (int)0x80000000, 0, (int)0x80000000, 0,
        (int)0x80000000, 0, (int)0x80000000, 0,
        (int)0x80000000, 0, (int)0x80000000, 0));
    constexpr int V = 16;
    for (int r = 0; r < R; ++r)
    {
        const float* sn = sin + (size_t)r * (P / 2);
        const float* cs = cos + (size_t)r * (P / 2);
        const char* si = in + (size_t)r * P * iesz;
        char* so = out + (size_t)r * P * oesz;
        int j = 0;
        const int vend = P / V * V;
        for (; j < vend; j += V)
        {
            const __m512 v = rp_ld16<ID>(si + (size_t)j * iesz);
            const __m512 cd = _mm512_permutexvar_ps(
                dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(cs + j / 2)));
            const __m512 sd = _mm512_permutexvar_ps(
                dup_idx, _mm512_castps256_ps512(_mm256_loadu_ps(sn + j / 2)));
            const __m512 sw = _mm512_shuffle_ps(v, v, 0xB1);
            rp_st16(so + (size_t)j * oesz,
                    _mm512_fmadd_ps(sw, _mm512_xor_ps(sd, sign),
                                    _mm512_mul_ps(v, cd)), OD);
        }
        for (; j < P; j += 2)
        {
            const float v1 = rp_load_s(si + (size_t)j * iesz, ID);
            const float v2 = rp_load_s(si + (size_t)(j + 1) * iesz, ID);
            const float c = cs[j / 2], s = sn[j / 2];
            rp_store_s(so + (size_t)j * oesz, OD, v1 * c - v2 * s);
            rp_store_s(so + (size_t)(j + 1) * oesz, OD, v2 * c + v1 * s);
        }
    }
    if (R == 1 && P < hd)
    {
        for (int j = P; j < hd; ++j)
            rp_store_s(out + (size_t)j * oesz, OD, rp_load_s(in + (size_t)j * iesz, ID));
    }
}

template <RpDt ID, RpDt OD>
EXL3_CPU_TARGET_AVX2
void rp_row_gptj_256(
    const char* in, char* out, int hd, int P, int R,
    const float* sin, const float* cos, int iesz, int oesz)
{
    const __m256 sign = _mm256_setr_ps(
        -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f);
    constexpr int V = 8;
    for (int r = 0; r < R; ++r)
    {
        const float* sn = sin + (size_t)r * (P / 2);
        const float* cs = cos + (size_t)r * (P / 2);
        const char* si = in + (size_t)r * P * iesz;
        char* so = out + (size_t)r * P * oesz;
        int j = 0;
        const int vend = P / V * V;
        for (; j < vend; j += V)
        {
            const __m256 v = rp_ld8<ID>(si + (size_t)j * iesz);
            const __m128 c4 = _mm_loadu_ps(cs + j / 2);
            const __m128 s4 = _mm_loadu_ps(sn + j / 2);
            const __m256 cd = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_unpacklo_ps(c4, c4)),
                _mm_unpackhi_ps(c4, c4), 1);
            const __m256 sd = _mm256_insertf128_ps(
                _mm256_castps128_ps256(_mm_unpacklo_ps(s4, s4)),
                _mm_unpackhi_ps(s4, s4), 1);
            const __m256 sw = _mm256_shuffle_ps(v, v, 0xB1);
            rp_st8(so + (size_t)j * oesz,
                   _mm256_fmadd_ps(sw, _mm256_xor_ps(sd, sign),
                                   _mm256_mul_ps(v, cd)), OD);
        }
        for (; j < P; j += 2)
        {
            const float v1 = rp_load_s(si + (size_t)j * iesz, ID);
            const float v2 = rp_load_s(si + (size_t)(j + 1) * iesz, ID);
            const float c = cs[j / 2], s = sn[j / 2];
            rp_store_s(so + (size_t)j * oesz, OD, v1 * c - v2 * s);
            rp_store_s(so + (size_t)(j + 1) * oesz, OD, v2 * c + v1 * s);
        }
    }
    if (R == 1 && P < hd)
    {
        for (int j = P; j < hd; ++j)
            rp_store_s(out + (size_t)j * oesz, OD, rp_load_s(in + (size_t)j * iesz, ID));
    }
}

// One tensor's rows: vector widths only narrow to f32/f16 (bf16 outputs take the
// threaded scalar kernel, same rule as act_mul/rms_norm).
template <RpDt ID, RpDt OD>
inline void rp_row_any(
    const RpCtx& c, const char* in, char* out,
    const float* sin, const float* cos, int iesz, int oesz)
{
    if (c.isa == RP_AVX512)
    {
        if (c.style == ROPESTYLE_NEOX)
            rp_row_neox_512<ID, OD>(in, out, c.hd, c.P, c.R, sin, cos, iesz, oesz);
        else
            rp_row_gptj_512<ID, OD>(in, out, c.hd, c.P, c.R, sin, cos, iesz, oesz);
    }
    else
    {
        if (c.style == ROPESTYLE_NEOX)
            rp_row_neox_256<ID, OD>(in, out, c.hd, c.P, c.R, sin, cos, iesz, oesz);
        else
            rp_row_gptj_256<ID, OD>(in, out, c.hd, c.P, c.R, sin, cos, iesz, oesz);
    }
}

inline void rp_apply_row(
    const RpCtx& c, const char* in, char* out,
    const float* sin, const float* cos,
    RpDt idt, RpDt odt, int iesz, int oesz)
{
    if (c.isa == RP_SCALAR || odt == RP_BF16)
    {
        rp_row_scalar(in, out, c.hd, c.P, c.R, c.style, sin, cos, idt, odt, iesz, oesz);
        return;
    }
#define RP_CALL_C(ID, OD) rp_row_any<ID, OD>(c, in, out, sin, cos, iesz, oesz)
#define RP_SWITCH_OD(ID) \
    switch (odt) \
    { \
        case RP_F32: RP_CALL_C(ID, RP_F32); break; \
        default: RP_CALL_C(ID, RP_F16); break; \
    }
    switch (idt)
    {
        case RP_F32: RP_SWITCH_OD(RP_F32); break;
        case RP_F16: RP_SWITCH_OD(RP_F16); break;
        default: RP_SWITCH_OD(RP_BF16); break;
    }
#undef RP_CALL_C
#undef RP_SWITCH_OD
}

// -------------------------------------------------------------------------------------------
// Per-token driver: hoist sin/cos for every section, then rotate all heads (q and k)
// -------------------------------------------------------------------------------------------

inline void rp_token(const RpCtx& c, int64_t b, int64_t s, float* buf)
{
    float* sin = buf;
    float* cos = buf + (size_t)c.R * c.Phalf;
    for (int r = 0; r < c.R; ++r)
    {
        const int64_t p = rp_pos(c, b, s, r);
        float* sn = sin + (size_t)r * c.Phalf;
        float* cs = cos + (size_t)r * c.Phalf;
        if (c.table)
        {
            const float* row = c.freq +
                ((c.table_dim == 3 ? b * c.table_seq + p : p) * c.Phalf);
            for (int j = 0; j < c.Phalf; ++j)
            {
                sn[j] = ::sinf(row[j]) * c.attn;
                cs[j] = ::cosf(row[j]) * c.attn;
            }
        }
        else
        {
            const float pf = (float)p;
            for (int j = 0; j < c.Phalf; ++j)
            {
                const float a = pf * c.freq[j];
                sn[j] = ::sinf(a) * c.attn;
                cs[j] = ::cosf(a) * c.attn;
            }
        }
    }
    const size_t q_row_in = (size_t)c.hd * c.q_iesz;
    const size_t q_row_out = (size_t)c.hd * c.q_oesz;
    for (int h = 0; h < c.Hq; ++h)
    {
        const size_t row = ((size_t)b * c.seq + s) * c.Hq + h;
        rp_apply_row(c, c.q + row * q_row_in, c.oq + row * q_row_out,
                     sin, cos, c.q_idt, c.q_odt, c.q_iesz, c.q_oesz);
    }
    if (c.k)
    {
        const size_t k_row_in = (size_t)c.hd * c.k_iesz;
        const size_t k_row_out = (size_t)c.hd * c.k_oesz;
        for (int h = 0; h < c.Hk; ++h)
        {
            const size_t row = ((size_t)b * c.seq + s) * c.Hk + h;
            rp_apply_row(c, c.k + row * k_row_in, c.ok + row * k_row_out,
                         sin, cos, c.k_idt, c.k_odt, c.k_iesz, c.k_oesz);
        }
    }
}

inline void rp_run(const RpCtx& c)
{
    const int64_t tokens = c.bsz * c.seq;
    if (tokens <= 0) return;
    const size_t scratch = (size_t)2 * c.R * c.Phalf;
    auto process = [&](int64_t b, int64_t e)
    {
        const int64_t b0 = b * RP_TOKEN_CHUNK;
        int64_t e0 = e * RP_TOKEN_CHUNK;
        if (e0 > tokens) e0 = tokens;
        std::vector<float> buf(scratch);
        for (int64_t t = b0; t < e0; ++t)
            rp_token(c, t / c.seq, t % c.seq, buf.data());
    };
    struct Tile
    {
        static void tile(void* ctx, int64_t b, int64_t e)
        {
            (*static_cast<decltype(&process)>(ctx))(b, e);
        }
    };
    const int64_t total = (tokens + RP_TOKEN_CHUNK - 1) / RP_TOKEN_CHUNK;
    cpu_pool_run_tiles(total, &Tile::tile, (void*)&process);
}

inline void rp_check_4d(const at::Tensor& t, const at::Tensor& ref, const char* name)
{
    TORCH_CHECK(t.dim() == 4, name, ": q/k/out must be 4-D [bsz, seq, heads, hd]");
    TORCH_CHECK(t.stride(3) == 1, name, ": innermost dim must be dense");
    TORCH_CHECK(t.size(0) == ref.size(0) && t.size(1) == ref.size(1) &&
                t.size(3) == ref.size(3),
                name, ": bsz/seq/hd mismatch");
}

} // namespace

void rope_cpu(
    const at::Tensor& q,
    at::Tensor& out_q,
    const c10::optional<at::Tensor>& k,
    c10::optional<at::Tensor>& out_k,
    const at::Tensor& inv_freq,
    int64_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    int rope_style,
    float attn_factor,
    int rotate_dims)
{
    TORCH_CHECK(q.device().is_cpu() && out_q.device().is_cpu() &&
                inv_freq.device().is_cpu(), "rope_cpu: CPU tensors only");
    TORCH_CHECK(q.is_contiguous() && out_q.is_contiguous() && inv_freq.is_contiguous(),
                "rope_cpu: q, out_q and inv_freq must be contiguous");
    for (const at::Tensor* t : {(const at::Tensor*)&q, (const at::Tensor*)&out_q})
        TORCH_CHECK(t->is_floating_point() && (t->scalar_type() == at::kFloat ||
                    t->scalar_type() == at::kHalf || t->scalar_type() == at::kBFloat16),
                    "rope_cpu: q/out_q dtype must be f32/f16/bf16");
    TORCH_CHECK(out_q.strides() == q.strides(), "rope_cpu: out_q must share q's layout");
    TORCH_CHECK(inv_freq.scalar_type() == at::kFloat, "rope_cpu: inv_freq must be float32");
    TORCH_CHECK(rope_style == ROPESTYLE_GPTJ || rope_style == ROPESTYLE_NEOX,
                "rope_cpu: rope_style must be GPTJ or NEOX");
    TORCH_CHECK(rotate_dims >= 1 && rotate_dims <= ROPE_CPU_MAX_ROTATE_DIMS,
                "rope_cpu: rotate_dims out of range");
    TORCH_CHECK(!k.has_value() == !out_k.has_value(),
                "rope_cpu: k and out_k must be given together");
    TORCH_CHECK((positions.has_value() ? 1 : 0) + (position_ids.has_value() ? 1 : 0) <= 1,
                "rope_cpu: invalid arguments");

    const bool has_k = k.has_value();
    if (has_k)
    {
        TORCH_CHECK(k->device().is_cpu() && out_k->device().is_cpu(),
                    "rope_cpu: CPU tensors only");
        TORCH_CHECK(k->is_contiguous() && out_k->is_contiguous(),
                    "rope_cpu: k and out_k must be contiguous");
        for (const at::Tensor* t : {(const at::Tensor*)&k.value(), (const at::Tensor*)&out_k.value()})
            TORCH_CHECK(t->is_floating_point() && (t->scalar_type() == at::kFloat ||
                        t->scalar_type() == at::kHalf || t->scalar_type() == at::kBFloat16),
                        "rope_cpu: k/out_k dtype must be f32/f16/bf16");
        TORCH_CHECK(out_k->strides() == k->strides(), "rope_cpu: out_k must share k's layout");
        rp_check_4d(k.value(), q, "rope_cpu");
    }
    rp_check_4d(q, q, "rope_cpu");
    rp_check_4d(out_q, q, "rope_cpu");

    if (q.numel() == 0) return;

    const int hd = (int)q.size(3);
    const int P = (int)inv_freq.size(-1) * 2;
    TORCH_CHECK(P > 0 && P % 2 == 0, "rope_cpu: bad inv_freq width");
    if (rotate_dims == 1)
    {
        TORCH_CHECK(P <= hd, "rope_cpu: rotary width exceeds head_dim");
    }
    else
    {
        TORCH_CHECK(hd == P * rotate_dims,
                    "rope_cpu: rotate_dims is inconsistent with inv_freq and head_dim");
    }

    const bool table = inv_freq.dim() > 1;
    int64_t table_seq = 0;
    int table_dim = 0;
    if (table)
    {
        TORCH_CHECK(inv_freq.dim() == 2 || inv_freq.dim() == 3,
                    "rope_cpu: inv_freq table must be 2-D or 3-D");
        table_dim = (int)inv_freq.dim();
        table_seq = table_dim == 3 ? inv_freq.size(-2) : inv_freq.size(0);
    }

    RpCtx c{};
    c.q = static_cast<const char*>(q.const_data_ptr());
    c.oq = static_cast<char*>(out_q.mutable_data_ptr());
    c.q_iesz = (int)q.element_size();
    c.q_oesz = (int)out_q.element_size();
    c.q_idt = rp_dtype(q.scalar_type());
    c.q_odt = rp_dtype(out_q.scalar_type());
    if (has_k)
    {
        c.k = static_cast<const char*>(k->const_data_ptr());
        c.ok = static_cast<char*>(out_k->mutable_data_ptr());
        c.k_iesz = (int)k->element_size();
        c.k_oesz = (int)out_k->element_size();
        c.k_idt = rp_dtype(k->scalar_type());
        c.k_odt = rp_dtype(out_k->scalar_type());
        c.Hk = (int)k->size(2);
    }
    c.bsz = q.size(0);
    c.seq = q.size(1);
    c.Hq = (int)q.size(2);
    c.hd = hd;
    c.P = P;
    c.Phalf = P / 2;
    c.R = rotate_dims;
    c.style = rope_style;
    c.attn = attn_factor;
    c.table = table;
    c.freq = static_cast<const float*>(inv_freq.const_data_ptr());
    c.table_seq = table_seq;
    c.table_dim = table_dim;
    c.position = position;
    if (positions.has_value())
    {
        TORCH_CHECK(positions->device().is_cpu() && positions->is_contiguous(),
                    "rope_cpu: positions must be a contiguous CPU tensor");
        TORCH_CHECK(positions->scalar_type() == at::kInt ||
                    positions->scalar_type() == at::kLong,
                    "rope_cpu: positions must be int32/int64");
        TORCH_CHECK(positions->dim() == 1 && positions->size(0) == c.bsz,
                    "rope_cpu: positions is incorrect shape");
        c.posp = static_cast<const char*>(positions->const_data_ptr());
        c.posz = (int)positions->element_size();
    }
    if (position_ids.has_value())
    {
        TORCH_CHECK(position_ids->device().is_cpu() && position_ids->is_contiguous(),
                    "rope_cpu: position_ids must be a contiguous CPU tensor");
        TORCH_CHECK(position_ids->scalar_type() == at::kInt ||
                    position_ids->scalar_type() == at::kLong,
                    "rope_cpu: position_ids must be int32/int64");
        const int rd = (int)position_ids->dim();
        TORCH_CHECK(rd == 2 || (rd == 3 && position_ids->size(-1) == rotate_dims),
                    "rope_cpu: position_ids wrong number of dims");
        TORCH_CHECK(position_ids->size(0) == c.bsz && position_ids->size(1) == c.seq,
                    "rope_cpu: position_ids is incorrect shape");
        c.pidp = static_cast<const char*>(position_ids->const_data_ptr());
        c.pidz = (int)position_ids->element_size();
        c.pid_rank = rd;
    }
    c.isa = rp_pick_isa();

    rp_run(c);
}
