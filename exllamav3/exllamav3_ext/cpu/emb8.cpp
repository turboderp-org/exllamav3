#include "emb8.h"
#include "../util.h"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#pragma comment(lib, "Synchronization.lib")
#endif

// Same convention as moe_mul1.cpp: per-function target attributes on GCC so the
// AVX2 kernels build without global -march flags; MSVC compiles the intrinsics
// unconditionally and relies on the runtime CPUID gate below to never execute
// them on unsupported CPUs.
#if defined(__GNUC__) && defined(__linux__)
#define EMB8_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define EMB8_TARGET_AVX2
#endif

namespace EXL3
{

namespace
{

// Row kernels. Bit-exactness contract with the torch path in
// exllamav3/modules/embedding.py:
// int8 values are cast to fp32, multiplied by the fp32 scale, then rounded to
// the output type once (vcvtps2ph rounds to nearest even like c10::Half).

inline void dequant_row_f32_scalar(const int8_t* q, const at::Half* s, float* o, int64_t hidden)
{
    const int64_t nb = hidden / 32;
    for (int64_t b = 0; b < nb; b++)
    {
        const float sd = float(s[b]);
        const int8_t* qb = q + b * 32;
        float* ob = o + b * 32;
        for (int k = 0; k < 32; k++)
            ob[k] = float(qb[k]) * sd;
    }
}

inline void dequant_row_f16_scalar(const int8_t* q, const at::Half* s, at::Half* o, int64_t hidden)
{
    const int64_t nb = hidden / 32;
    for (int64_t b = 0; b < nb; b++)
    {
        const float sd = float(s[b]);
        const int8_t* qb = q + b * 32;
        at::Half* ob = o + b * 32;
        for (int k = 0; k < 32; k++)
            ob[k] = at::Half(float(qb[k]) * sd);
    }
}

EMB8_TARGET_AVX2
inline void dequant_row_f32_avx2(const int8_t* q, const at::Half* s, float* o, int64_t hidden)
{
    const int64_t nb = hidden / 32;
    for (int64_t b = 0; b < nb; b++)
    {
        const __m256 sd = _mm256_set1_ps(float(s[b]));
        const int8_t* qb = q + b * 32;
        float* ob = o + b * 32;
        for (int k = 0; k < 32; k += 16)
        {
            __m128i q8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qb + k));
            __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)), sd);
            __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q8, 8))), sd);
            _mm256_storeu_ps(ob + k, f0);
            _mm256_storeu_ps(ob + k + 8, f1);
        }
    }
}

EMB8_TARGET_AVX2
inline void dequant_row_f16_avx2(const int8_t* q, const at::Half* s, at::Half* o, int64_t hidden)
{
    const int64_t nb = hidden / 32;
    for (int64_t b = 0; b < nb; b++)
    {
        const __m256 sd = _mm256_set1_ps(float(s[b]));
        const int8_t* qb = q + b * 32;
        at::Half* ob = o + b * 32;
        for (int k = 0; k < 32; k += 8)
        {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qb + k));
            __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            __m128i h = _mm256_cvtps_ph(_mm256_mul_ps(f, sd), 0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(ob + k), h);
        }
    }
}

struct Job
{
    const int8_t* q;
    const at::Half* s;
    const int64_t* ids;
    void* out;
    int64_t hidden;
    bool f32;
    bool avx2;
};

EMB8_TARGET_AVX2
void run_range_avx2(const Job& j, int64_t r0, int64_t r1)
{
    const int64_t sb = j.hidden / 32;
    for (int64_t r = r0; r < r1; r++)
    {
        const int64_t id = j.ids[r];
        const int8_t* qr = j.q + id * j.hidden;
        const at::Half* sr = j.s + id * sb;
        if (j.f32)
            dequant_row_f32_avx2(qr, sr, reinterpret_cast<float*>(j.out) + r * j.hidden, j.hidden);
        else
            dequant_row_f16_avx2(qr, sr, reinterpret_cast<at::Half*>(j.out) + r * j.hidden, j.hidden);
    }
}

void run_range_scalar(const Job& j, int64_t r0, int64_t r1)
{
    const int64_t sb = j.hidden / 32;
    for (int64_t r = r0; r < r1; r++)
    {
        const int64_t id = j.ids[r];
        const int8_t* qr = j.q + id * j.hidden;
        const at::Half* sr = j.s + id * sb;
        if (j.f32)
            dequant_row_f32_scalar(qr, sr, reinterpret_cast<float*>(j.out) + r * j.hidden, j.hidden);
        else
            dequant_row_f16_scalar(qr, sr, reinterpret_cast<at::Half*>(j.out) + r * j.hidden, j.hidden);
    }
}

inline void run_range(const Job& j, int64_t r0, int64_t r1)
{
    if (j.avx2)
        run_range_avx2(j, r0, r1);
    else
        run_range_scalar(j, r0, r1);
}

// Runtime ISA gate, same shape as moe_mul1.cpp: __builtin_cpu_supports checks OS
// state-saving on GCC; the manual branch must check OSXSAVE + XCR0 because
// CPUID feature bits report hardware capability only.
bool cpu_has_avx2()
{
#if defined(__GNUC__) && defined(__linux__)
    static const bool ok =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#elif defined(__x86_64__) || defined(_M_X64)
    static const bool ok = []
    {
        int l0[4];
        __cpuid(l0, 0);
        if (l0[0] < 7)
            return false;
        int l1[4];
        __cpuid(l1, 1);
        const bool osxsave = (l1[2] & (1u << 27)) != 0;
        const bool fma = (l1[2] & (1u << 12)) != 0;
        const bool f16c = (l1[2] & (1u << 29)) != 0;
        const uint64_t xcr0 = osxsave ? _xgetbv(0) : 0;
        const bool ymm_os = (xcr0 & 0x06) == 0x06;   // XMM + YMM state
        int info[4];
        __cpuidex(info, 7, 0);
        const bool avx2 = (info[1] & (1u << 5)) != 0;
        return avx2 && fma && f16c && ymm_os;
    }();
#else
    static const bool ok = false;
#endif
    // EXL3_EMB8_MAX_ISA=scalar forces the scalar path for testing; never upgrades.
    static const bool cap_scalar = []
    {
        const char* e = std::getenv("EXL3_EMB8_MAX_ISA");
        return e && std::string(e) == "scalar";
    }();
    return ok && !cap_scalar;
}

} // namespace

at::Tensor emb8_dequant
(
    const at::Tensor& q_table,
    const at::Tensor& scale_table,
    const at::Tensor& ids,
    bool f32_out
)
{
    TORCH_CHECK(q_table.device().is_cpu() && scale_table.device().is_cpu() && ids.device().is_cpu(),
        "emb8_dequant: CPU tensors only");
    TORCH_CHECK(q_table.is_contiguous() && scale_table.is_contiguous(),
        "emb8_dequant: tables must be contiguous");
    TORCH_CHECK_DTYPE(q_table, kChar);
    TORCH_CHECK_DTYPE(scale_table, kHalf);
    TORCH_CHECK_DTYPE(ids, kLong);
    TORCH_CHECK_DIM(q_table, 2);
    TORCH_CHECK_DIV(q_table, 1, 32);
    const int64_t hidden = q_table.size(1);
    TORCH_CHECK_DIM(scale_table, 2);
    TORCH_CHECK_SHAPES(scale_table, 0, q_table, 0, 1);
    TORCH_CHECK_SIZE(scale_table, 1, hidden / 32);

    const at::Tensor ids_c = ids.contiguous();
    const int64_t n = ids_c.numel();
    const at::ScalarType dt = f32_out ? at::kFloat : at::kHalf;
    if (n == 0)
        return at::empty({0, hidden}, q_table.options().dtype(dt));

    const int64_t vocab = q_table.size(0);
    const int64_t* id_ptr = ids_c.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; i++)
        TORCH_CHECK(id_ptr[i] >= 0 && id_ptr[i] < vocab, "emb8_dequant: id out of range [0, vocab)");

    at::Tensor out = at::empty({n, hidden}, at::TensorOptions().dtype(dt).device(at::kCPU));

    Job job;
    job.q = q_table.data_ptr<int8_t>();
    job.s = scale_table.data_ptr<at::Half>();
    job.ids = ids_c.data_ptr<int64_t>();
    job.out = out.data_ptr();
    job.hidden = hidden;
    job.f32 = f32_out;
    job.avx2 = cpu_has_avx2();

    // Threads over contiguous row chunks; at::get_num_threads() honors the usual
    // torch/OMP_NUM_THREADS knobs. Measured: SMT siblings add nothing (the pass
    // is DRAM-bound), so never exceed the intra-op thread count. Below a few
    // hundred rows the spawn cost outweighs the parallel gain.
    const int64_t hw = std::max<int64_t>(1, at::get_num_threads());
    int nt = (int) std::min<int64_t>(hw, std::max<int64_t>(1, n / 128));
    if (n < 256)
        nt = 1;

    if (nt == 1)
    {
        run_range(job, 0, n);
        return out;
    }

    const int64_t chunk = (n + nt - 1) / nt;
    std::vector<std::thread> ts;
    ts.reserve(nt - 1);
    for (int i = 1; i < nt; i++)
    {
        const int64_t r0 = i * chunk;
        const int64_t r1 = std::min(n, r0 + chunk);
        if (r0 >= r1)
            break;
        ts.emplace_back([&job, r0, r1] { run_range(job, r0, r1); });
    }
    run_range(job, 0, std::min(n, chunk));
    for (std::thread& t : ts)
        t.join();

    return out;
}

} // namespace EXL3