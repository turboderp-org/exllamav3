// AVX2 "planar" layout microbench.
//
// Hypothesis (bench/swz_findings.md): the AVX2 tier's per-row register walk in
// avx2_gather_half (bits x vpermd+blend per half-row) is what (a) makes K3/K4/K7 warm 1T
// compute-bound and (b) locks them out of the band-2 swizzle kernel (spills). Under the
// tc-perm tile layout, a half-row's 8 source dwords advance by exactly `bits` per column,
// i.e. they form ONE residue class mod bits of the tile's 8*bits dwords. Repacking each
// tile's dwords so every residue class lives in one 8-dword register
//     new[q] = old[bits * (q % 8) + q / 8]      (q, w in dwords; integer rates)
// collapses every gather to a SINGLE vpermd (vpermd is full 256-bit cross-lane).
//
// Variants:
//   PROD    production avx2_tiles (native layout, pf dist 4 / K6 2)
//   PLAN    avx2_tiles body, planar gather, planar layout
//   SWZP    band-2 kernel (production avx2_swz_tiles body), planar gather, group-2+planar
// All must be BIT-EXACT vs PROD (same codes, same instruction sequence, same order).
//
// build: g++ -Ofast -std=c++17 -I bench/stub bench/bench_planar.cpp -o bench/bench_planar -lpthread

#include "../exllamav3/exllamav3_ext/cpu/moe_mul1.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_s() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

// --------------------------------------------------------------------------- flush
static std::vector<char> g_flush;
volatile long g_flush_sink = 0;
static void flush_caches()
{
    if (g_flush.empty()) g_flush.resize(256ull << 20);
    long s = 0;
    const volatile char* b = g_flush.data();
    for (size_t i = 0; i < g_flush.size(); i += 64) s += b[i];
    g_flush_sink = s;
}

// --------------------------------------------------------------------------- layout repacks
// group-2 tile-order swizzle: tile (kt, nt) -> (nt/2, kt, nt%2)
static std::vector<uint16_t> repack_group2(const std::vector<uint16_t>& src, int tiles_k,
    int tiles_n, int packed_size)
{
    std::vector<uint16_t> dst(src.size());
    for (int kt = 0; kt < tiles_k; ++kt)
        for (int nt = 0; nt < tiles_n; ++nt)
        {
            const uint16_t* s = src.data() + (static_cast<size_t>(kt) * tiles_n + nt) * packed_size;
            uint16_t* d = dst.data() + ((static_cast<size_t>(nt >> 1) * tiles_k + kt) * 2
                                        + (nt & 1)) * packed_size;
            std::memcpy(d, s, static_cast<size_t>(packed_size) * 2);
        }
    return dst;
}

// planar dword permute within each tile: new[q] = old[bits*(q%8) + q/8] (q in dwords)
static std::vector<uint16_t> repack_planar(const std::vector<uint16_t>& src, int bits)
{
    const int ps = 16 * bits;              // u16 per tile
    const int wd = 8 * bits;               // dwords per tile
    std::vector<uint16_t> dst(src.size());
    const size_t tiles = src.size() / ps;
    for (size_t t = 0; t < tiles; ++t)
        for (int q = 0; q < wd; ++q)
        {
            const int o = bits * (q & 7) + q / 8;
            dst[t * ps + 2 * q]     = src[t * ps + 2 * o];
            dst[t * ps + 2 * q + 1] = src[t * ps + 2 * o + 1];
        }
    return dst;
}

// --------------------------------------------------------------------------- planar gather
// Register / lane tables: old dword w of a half-row lives at planar position 8*(w%bits)+w/bits
// -> register w%bits, lane w/bits. All 8 columns of a half must agree on the register
// (the orbit property; static_assert enforces it at compile time).
template <int bits, int row, bool second_word, int half>
constexpr int bp_planar_reg()
{
    constexpr auto idx = make_row_indices<bits, false, row, second_word>();
    const int r = idx[half * 8] % bits;
    for (int c = 1; c < 8; ++c)
        if (idx[half * 8 + c] % bits != r) return -1;   // caught by the static_assert below
    return r;
}

template <int bits, int row, bool second_word, int half>
M1_TARGET_AVX2
inline __m256i bp_avx2_gather_half_planar(const __m256i (&preg)[bits])
{
    static_assert(bp_planar_reg<bits, row, second_word, half>() >= 0, "half-row words span registers");
    constexpr auto idx = make_row_indices<bits, false, row, second_word>();
    constexpr int REG = bp_planar_reg<bits, row, second_word, half>();
    // lane = planar position within the register = w / bits
    return _mm256_permutevar8x32_epi32(preg[REG], _mm256_setr_epi32(
        idx[half * 8 + 0] / bits, idx[half * 8 + 1] / bits, idx[half * 8 + 2] / bits,
        idx[half * 8 + 3] / bits, idx[half * 8 + 4] / bits, idx[half * 8 + 5] / bits,
        idx[half * 8 + 6] / bits, idx[half * 8 + 7] / bits));
}

template <int bits, int row>
M1_TARGET_AVX2
inline void bp_avx2_row_codes_planar(const __m256i (&preg)[bits], __m256i& codes_lo, __m256i& codes_hi)
{
    const __m256i a_lo = bp_avx2_gather_half_planar<bits, row, false, 0>(preg);
    const __m256i b_lo = bp_avx2_gather_half_planar<bits, row, true, 0>(preg);
    const __m256i a_hi = bp_avx2_gather_half_planar<bits, row, false, 1>(preg);
    const __m256i b_hi = bp_avx2_gather_half_planar<bits, row, true, 1>(preg);
    const __m256i mask16 = _mm256_set1_epi32(0xffff);
    constexpr int s0 = row_shift<bits, false, row>(0);
    constexpr int s1 = row_shift<bits, false, row>(8);
    codes_lo = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_lo, s0), _mm256_slli_epi32(a_lo, 32 - s0)), mask16);
    codes_hi = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_hi, s1), _mm256_slli_epi32(a_hi, 32 - s1)), mask16);
}

template <int bits, int row = 0>
M1_TARGET_AVX2
inline void bp_avx2_rows_accum_planar(
    const __m256i (&preg)[bits], const int32_t* splat_dup, int k, int m, __m256i (&acc)[MAX_M][2],
    const __m256i& mult, const __m256i& ones32)
{
    if constexpr (row < 16)
    {
        __m256i codes_lo, codes_hi;
        bp_avx2_row_codes_planar<bits, row>(preg, codes_lo, codes_hi);
        avx2_accum_row(codes_lo, codes_hi, splat_dup, k, m, acc, mult, ones32, row);
        bp_avx2_rows_accum_planar<bits, row + 1>(preg, splat_dup, k, m, acc, mult, ones32);
    }
}

// --------------------------------------------------------------------------- variant kernels
template <int bits>
M1_TARGET_AVX2
void avx2_tiles_planar(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    const __m256i mult = _mm256_set1_epi32(static_cast<int32_t>(MUL1_MULT));
    const __m256i ones32 = _mm256_set1_epi32(0x01010101);
    const int32_t* splat_dup = in.splat_dup;
    constexpr int pf_lines = (packed_size * 2 + 63) / 64;
    constexpr int pf_dist = (bits == 6) ? 2 : 4;

    for (int tile_n = tn0; tile_n < tn1; ++tile_n)
    {
        __m256i acc[MAX_M][2];
        for (int i = 0; i < m; ++i) { acc[i][0] = _mm256_setzero_si256(); acc[i][1] = _mm256_setzero_si256(); }
        const uint16_t* packed = mat.trellis + static_cast<size_t>(tile_n) * packed_size;
        const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed += row_stride)
        {
            const uint16_t* pf = packed + row_stride * pf_dist;
            #pragma unroll
            for (int l = 0; l < pf_lines; ++l)
                _mm_prefetch(reinterpret_cast<const char*>(pf) + l * 64, _MM_HINT_T0);
            const int32_t* splat_k = splat_dup + tile_k * 16;
            __m256i preg[bits];
            for (int i = 0; i < bits; ++i)
                preg[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i * 16));
            bp_avx2_rows_accum_planar<bits>(preg, splat_k, mat.k, m, acc, mult, ones32);
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

// band-2 + planar (group-2 tile order + planar dwords): the combination that should unlock
// K3/K4/K7 for the banded path (production avx2_swz_tiles body, planar gathers)
// SWZP now runs the PRODUCTION kernel (integrated); PLAN keeps a bench-local clone of the
// planar gather on the native tile layout (not shipped: band-2 measures above it everywhere)
template <int bits>
M1_TARGET_AVX2
void bp_avx2_swz_planar(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    avx2_swz2_planar<bits>(mat, in, tout, m, tn0, tn1);
}

// --------------------------------------------------------------------------- fixture
struct Fixture
{
    MoeCpuMatrix mat;
    std::vector<uint16_t> trellis;                 // already repacked per mode
    std::vector<at::Half> suh, svh;
    PreparedIn in;
    std::vector<int32_t> splat, splat_dup;
    std::vector<float> tout;
};

static std::vector<uint16_t> g_src_trellis;        // native layout, per (bits, k, n)

static std::vector<uint16_t>& src_trellis(int bits, int k, int n, unsigned seed)
{
    static int cbits = -1, ck = -1, cn = -1; static unsigned cseed = 0;
    if (cbits == bits && ck == k && cn == n && cseed == seed) return g_src_trellis;
    cbits = bits; ck = k; cn = n; cseed = seed;
    std::mt19937 rng(seed);
    g_src_trellis.resize((size_t)(k / 16) * (n / 16) * 16 * bits);
    for (auto& w : g_src_trellis) w = (uint16_t)rng();
    return g_src_trellis;
}

// mode: 0 = native, 1 = planar, 2 = group2+planar, 3 = group2 (native dwords)
static Fixture make_fixture(int bits, int k, int n, int mode, int mrows)
{
    Fixture f;
    std::mt19937 rng(1234 + bits * 31 + mrows);
    const int tiles_k = k / 16, tiles_n = n / 16, ps = 16 * bits;
    auto src = src_trellis(bits, k, n, 1234 + bits * 31 + mrows);
    if (mode == 1) f.trellis = repack_planar(src, bits);
    else if (mode == 2) f.trellis = repack_planar(repack_group2(src, tiles_k, tiles_n, ps), bits);
    else if (mode == 3) f.trellis = repack_group2(src, tiles_k, tiles_n, ps);
    else f.trellis = src;

    f.suh.resize(k); f.svh.resize(n);
    for (auto& h : f.suh) h = at::Half(uint16_t(0x3c00 + (rng() % 64)), at::Half::from_bits());
    for (auto& h : f.svh) h = at::Half(uint16_t(0x3c00 + (rng() % 64)), at::Half::from_bits());
    f.mat.trellis = f.trellis.data();
    f.mat.suh = f.suh.data(); f.mat.svh = f.svh.data(); f.mat.bias = nullptr;
    f.mat.k = k; f.mat.n = n; f.mat.bits = bits; f.mat.hb = 0; f.mat.swz = 0;

    f.splat.assign((size_t)mrows * k, 0);
    f.splat_dup.assign((size_t)mrows * k, 0);
    for (int r = 0; r < mrows; ++r)
        for (int i = 0; i < k; ++i)
        {
            int v = (int)(rng() % 255) - 127;
            const int32_t b = (int32_t)(uint8_t)(int8_t)v;
            f.splat[(size_t)r * k + i] = b * 0x01010101;
            f.splat_dup[(size_t)r * k + i] = (int32_t)(((uint32_t)(uint16_t)(int16_t)(int8_t)(uint8_t)b) * 0x00010001u);
        }
    f.in.tin = nullptr;
    f.in.splat32 = f.splat.data();
    f.in.splat_dup = f.splat_dup.data();
    for (int r = 0; r < mrows; ++r)
    {
        f.in.q[r] = 0.01f;
        long s = 0;
        for (int i = 0; i < k; ++i) s += (int8_t)(uint8_t)(f.splat[(size_t)r * k + i] & 0xff);
        f.in.sum_x8[r] = (int32_t)s;
    }
    f.in.rows = mrows;
    f.tout.assign((size_t)mrows * n, 0.0f);
    return f;
}

// --------------------------------------------------------------------------- drivers
typedef void (*RunFn)(const MoeCpuMatrix&, const PreparedIn&, float*, int, int, int);

template <int bits>
static RunFn prod_fn()  { return (RunFn)+[](const MoeCpuMatrix& m, const PreparedIn& i, float* t, int mm, int a, int b) { avx2_tiles<bits, false>(m, i, t, mm, a, b); }; }
template <int bits>
static RunFn plan_fn()  { return (RunFn)+[](const MoeCpuMatrix& m, const PreparedIn& i, float* t, int mm, int a, int b) { avx2_tiles_planar<bits>(m, i, t, mm, a, b); }; }
template <int bits>
static RunFn swzp_fn()  { return (RunFn)+[](const MoeCpuMatrix& m, const PreparedIn& i, float* t, int mm, int a, int b) { avx2_swz2_planar<bits>(m, i, t, mm, a, b); }; }
template <int bits>
static RunFn swz_fn()   { return (RunFn)+[](const MoeCpuMatrix& m, const PreparedIn& i, float* t, int mm, int a, int b) { avx2_swz_tiles<bits>(m, i, t, mm, a, b); }; }

struct Variant { const char* name; RunFn run; int mode; };

template <int bits>
static void bench_bits(int m, int k, int n, bool cold, int reps)
{
    const int tiles_n = n / 16;
    const double weights = (double)k * n;
    Variant variants[] = {
        { "PROD", prod_fn<bits>(), 0 },
        { "PLAN", plan_fn<bits>(), 1 },
        { "SWZP", swzp_fn<bits>(), 2 },
        { "SWZG", swz_fn<bits>(), 3 },   // production band-2 kernel, group-2 layout (reference)
    };
    std::vector<float> ref;
    printf("%s k=%d n=%d m=%d %-4s %10s %8s\n", cold ? "COLD" : "WARM", k, n, m, "", "us", "Gw/s");
    for (auto& v : variants)
    {
        Fixture f = make_fixture(bits, k, n, v.mode, m);
        // touch + capture reference from PROD
        v.run(f.mat, f.in, f.tout.data(), m, 0, tiles_n);
        if (v.name[0] == 'P' && v.name[1] == 'R') ref = f.tout;
        else if (ref.size() == f.tout.size() && std::memcmp(ref.data(), f.tout.data(), ref.size() * 4) != 0)
            printf("  !! %s NOT BIT-EXACT vs PROD (bits=%d)\n", v.name, bits);
        double best = 1e30;
        for (int r = 0; r < reps; ++r)
        {
            if (cold) flush_caches();
            double t0 = now_s();
            v.run(f.mat, f.in, f.tout.data(), m, 0, tiles_n);
            double dt = now_s() - t0;
            if (dt < best) best = dt;
        }
        printf("  %-5s %10.1f %8.2f\n", v.name, best * 1e6, weights / best / 1e9);
    }
}

int main(int argc, char** argv)
{
    int only_bits = argc > 1 ? atoi(argv[1]) : 0;
    int m = argc > 2 ? atoi(argv[2]) : 1;
    bool cold = argc > 3 && atoi(argv[3]);
    const int k = 2944;
    const int n = cold ? 32768 : 2944;
    const int reps = cold ? 5 : 15;
    auto run = [&](int b){
        switch (b) {
            case 1: bench_bits<1>(m, k, n, cold, reps); break;
            case 2: bench_bits<2>(m, k, n, cold, reps); break;
            case 3: bench_bits<3>(m, k, n, cold, reps); break;
            case 4: bench_bits<4>(m, k, n, cold, reps); break;
            case 5: bench_bits<5>(m, k, n, cold, reps); break;
            case 6: bench_bits<6>(m, k, n, cold, reps); break;
            case 7: bench_bits<7>(m, k, n, cold, reps); break;
            case 8: bench_bits<8>(m, k, n, cold, reps); break;
        }
    };
    if (only_bits) run(only_bits);
    else for (int b = 1; b <= 8; ++b) run(b);
    return 0;
}