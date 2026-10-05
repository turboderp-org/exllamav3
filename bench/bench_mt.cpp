// AVX2 planar layout MULTI-THREAD microbench: aggregate GEMV bandwidth vs the DRAM wall.
//
// Splits one GEMV's n-tiles across T pinned workers in 8-tile groups (the production
// assign_gemvs flat split), cold (256 MB read-flush before each rep), and reports aggregate
// Gw/s per variant next to a pure read-stream bandwidth reference (same thread count, same
// footprint). If a variant's bytes/s meets the read reference, the MT phase is bandwidth-
// bound and per-weight ALU wins (planar) will NOT move end-to-end decode -- they buy power
// and headroom instead.
//
// build: g++ -Ofast -std=c++17 -I bench/stub bench/bench_mt.cpp -o bench/bench_mt -lpthread
// run:   ./bench_mt [threads]   (default: all cores; bits looped 1..8)

#include "../exllamav3/exllamav3_ext/cpu/moe_mul1.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

using clk = std::chrono::steady_clock;
static double now_s() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

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

// --------------------------------------------------------------------- repacks (as loader)
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
static std::vector<uint16_t> repack_planar(const std::vector<uint16_t>& src, int bits)
{
    const int ps = 16 * bits;
    std::vector<uint16_t> dst(src.size());
    const size_t tiles = src.size() / ps;
    for (size_t t = 0; t < tiles; ++t)
        for (int q = 0; q < 8 * bits; ++q)
        {
            const int o = bits * (q & 7) + q / 8;
            dst[t * ps + 2 * q]     = src[t * ps + 2 * o];
            dst[t * ps + 2 * q + 1] = src[t * ps + 2 * o + 1];
        }
    return dst;
}

// --------------------------------------------------------------------- fixture (one expert)
struct Fixture
{
    MoeCpuMatrix mat;
    std::vector<uint16_t> trellis;
    std::vector<at::Half> suh, svh;
    PreparedIn in;
    std::vector<int32_t> splat, splat_dup;
    std::vector<float> tout;
};

static Fixture make_fixture(int bits, int k, int n, int mode)   // mode 0 native, 2 g2+planar, 3 g2
{
    Fixture f;
    std::mt19937 rng(1234 + bits * 31);
    const int tiles_k = k / 16, tiles_n = n / 16, ps = 16 * bits;
    std::vector<uint16_t> src((size_t)tiles_k * tiles_n * ps);
    for (auto& w : src) w = (uint16_t)rng();
    if (mode == 2) f.trellis = repack_planar(repack_group2(src, tiles_k, tiles_n, ps), bits);
    else if (mode == 3) f.trellis = repack_group2(src, tiles_k, tiles_n, ps);
    else f.trellis = src;
    f.suh.resize(k); f.svh.resize(n);
    for (auto& h : f.suh) h = at::Half(uint16_t(0x3c00), at::Half::from_bits());
    for (auto& h : f.svh) h = at::Half(uint16_t(0x3c00), at::Half::from_bits());
    f.mat.trellis = f.trellis.data();
    f.mat.suh = f.suh.data(); f.mat.svh = f.svh.data(); f.mat.bias = nullptr;
    f.mat.k = k; f.mat.n = n; f.mat.bits = bits; f.mat.hb = 0; f.mat.swz = mode ? 1 : 0;
    f.splat.assign(k, 0x01010101);
    f.splat_dup.assign(k, 0x00010001);
    f.in.tin = nullptr; f.in.splat32 = f.splat.data(); f.in.splat_dup = f.splat_dup.data();
    f.in.q[0] = 1.0f; f.in.sum_x8[0] = k; f.in.rows = 1;
    f.tout.assign(n, 0.0f);
    return f;
}

typedef void (*RunFn)(const MoeCpuMatrix&, const PreparedIn&, float*, int, int, int);
template <int bits> static RunFn prod_fn() { return +[](const MoeCpuMatrix& m, const PreparedIn& i, float* t, int mm, int a, int b){ avx2_tiles<bits,false>(m,i,t,mm,a,b); }; }
template <int bits> static RunFn swzg_fn() { return +[](const MoeCpuMatrix& m, const PreparedIn& i, float* t, int mm, int a, int b){ avx2_swz_tiles<bits>(m,i,t,mm,a,b); }; }
template <int bits> static RunFn swzp_fn() { return +[](const MoeCpuMatrix& m, const PreparedIn& i, float* t, int mm, int a, int b){ avx2_swz2_planar<bits>(m,i,t,mm,a,b); }; }

static void pin_to(int core)
{
#ifdef __linux__
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
#else
    (void)core;
#endif
}

// read-bandwidth reference: T threads sum disjoint slabs of a buffer larger than L3
static double read_bw_gbs(int threads, int core0)
{
    const size_t bytes = 192ull << 20;
    std::vector<char> buf(bytes, 1);
    double best = 0;
    for (int rep = 0; rep < 3; ++rep)
    {
        flush_caches();
        double t0 = now_s();
        std::vector<std::thread> ts;
        volatile long sink = 0;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t]{
                pin_to(core0 + t);
                const size_t per = bytes / threads & ~63ull;
                const volatile char* p = buf.data() + t * per;
                long s = 0;
                for (size_t i = 0; i < per; i += 64) s += p[i];
                sink = s;
            });
        for (auto& th : ts) th.join();
        double dt = now_s() - t0;
        best = best == 0 ? dt : (best < dt ? best : dt);
    }
    return bytes / best / 1e9;
}

template <int bits>
static void bench_bits(int threads, int k, int n, int core0)
{
    const int tiles_n = n / 16;
    const double weights = (double)k * n;
    struct V { const char* name; RunFn fn; int mode; };
    const V vs[] = { {"PROD", prod_fn<bits>(), 0}, {"SWZG", swzg_fn<bits>(), 3}, {"SWZP", swzp_fn<bits>(), 2} };
    printf("K%d T%-2d", bits, threads);
    for (auto& v : vs)
    {
        Fixture f = make_fixture(bits, k, n, v.mode);
        double best = 1e30;
        for (int rep = 0; rep < 5; ++rep)
        {
            flush_caches();
            double t0 = now_s();
            std::vector<std::thread> ts;
            for (int t = 0; t < threads; ++t)
                ts.emplace_back([&, t]{
                    pin_to(core0 + t);
                    // production flat split in 8-tile groups
                    const int a = (int)((long long)tiles_n * t / threads) & ~7;
                    const int b = t == threads - 1 ? tiles_n
                        : (((int)((long long)tiles_n * (t + 1) / threads) + 7) & ~7);
                    if (a < b) v.fn(f.mat, f.in, f.tout.data(), 1, a, b);
                });
            for (auto& th : ts) th.join();
            double dt = now_s() - t0;
            best = best < dt ? best : dt;
        }
        printf("   %s %6.2f Gw/s", v.name, weights / best / 1e9);
    }
    printf("\n");
}

int main(int argc, char** argv)
{
    int threads = argc > 1 ? atoi(argv[1]) : (int)std::thread::hardware_concurrency();
    int core0 = argc > 2 ? atoi(argv[2]) : 0;
    const int k = 2944, n = 32768;
    printf("read-bandwidth reference (T=%d): %.1f GB/s\n", threads, read_bw_gbs(threads, core0));
    printf("cold GEMV, k=%d n=%d, split in 8-tile groups, best of 5\n", k, n);
    for (int t = 1; t <= threads; t = t < 2 ? 2 : t * 2)
    {
        bench_bits<2>(t, k, n, core0);
        bench_bits<4>(t, k, n, core0);
        bench_bits<6>(t, k, n, core0);
    }
    return 0;
}