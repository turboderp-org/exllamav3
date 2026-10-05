// End-to-end production-path check for the AVX2 planar layout.
//
// Registers two layers through the real exl3_moe_cpu_make_layer / exl3_moe_cpu_forward_raw
// API (bench/stub torch): one with native trellis bytes (swizzled=0), one with the bytes the
// loader produces for the AVX2 tier (group-2 tile order + planar dwords, per
// exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout). Forcing EXL3_MOE_CPU_MAX_ISA=avx2
// the two forwards must be BIT-EXACT; without the cap this also exercises the AVX-512 tiers
// (their output is compared against the native layer with a tolerance only, since tier
// equivalence is bit-exact by design but this test's focus is the AVX2 dispatch).
//
// build: g++ -Ofast -std=c++17 -I bench/stub bench/bench_e2e.cpp -o bench/bench_e2e -lpthread

#include "../exllamav3/exllamav3_ext/cpu/moe_mul1.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <deque>
#include <vector>

namespace {

// exactly the loader repack (rehome/install in moe_cpu_host.py): group g tile order, then
// planar dwords within each tile (new[q] = old[bits*(q%8) + q/8])
std::vector<uint16_t> loader_repack(const std::vector<uint16_t>& src, int bits, int tiles_k,
    int tiles_n, int group, bool planar)
{
    const int ps = 16 * bits;   // u16 per tile
    std::vector<uint16_t> dst(src.size());
    for (int kt = 0; kt < tiles_k; ++kt)
        for (int nt = 0; nt < tiles_n; ++nt)
        {
            const uint16_t* s = src.data() + (static_cast<size_t>(kt) * tiles_n + nt) * ps;
            size_t dpos;
            if (group)
                dpos = ((static_cast<size_t>(nt / group) * tiles_k + kt) * group + (nt % group)) * ps;
            else
                dpos = (static_cast<size_t>(kt) * tiles_n + nt) * ps;
            uint16_t* d = dst.data() + dpos;
            if (planar)
                for (int q = 0; q < 8 * bits; ++q)
                {
                    const int o = bits * (q & 7) + q / 8;
                    d[2 * q] = s[2 * o]; d[2 * q + 1] = s[2 * o + 1];
                }
            else
                std::memcpy(d, s, static_cast<size_t>(ps) * 2);
        }
    return dst;
}

at::Tensor wrap(void* p, int64_t d0, int64_t d1, int64_t d2, at::ScalarType st)
{
    at::Tensor t; t.p = p; t.dims[0] = d0; t.dims[1] = d1; t.dims[2] = d2; t.nd = 3; t.st = st;
    return t;
}
at::Tensor wrap1(void* p, int64_t d0, at::ScalarType st)
{
    at::Tensor t; t.p = p; t.dims[0] = d0; t.nd = 1; t.st = st;
    return t;
}

struct Proj { std::vector<uint16_t> tr, tr_s; std::vector<uint16_t> suh, svh; };

int run_case(int bits, int E, int H, int I, int rows, int topk, unsigned seed)
{
    std::mt19937 rng(seed + bits);
    const int tiles_k_h = H / 16, tiles_n_i = I / 16, tiles_k_i = I / 16, tiles_n_h = H / 16;

    std::vector<Proj> g(E), u(E), d(E);
    for (int e = 0; e < E; ++e)
    {
        g[e].tr.resize((size_t)tiles_k_h * tiles_n_i * 16 * bits);
        u[e].tr.resize((size_t)tiles_k_h * tiles_n_i * 16 * bits);
        d[e].tr.resize((size_t)tiles_k_i * tiles_n_h * 16 * bits);
        for (auto* v : {&g[e].tr, &u[e].tr, &d[e].tr})
            for (auto& w : *v) w = (uint16_t)rng();
        g[e].suh.resize(H); g[e].svh.resize(I);
        u[e].suh.resize(H); u[e].svh.resize(I);
        d[e].suh.resize(I); d[e].svh.resize(H);
        for (auto* v : {&g[e].suh, &g[e].svh, &u[e].suh, &u[e].svh, &d[e].suh, &d[e].svh})
            for (auto& w : *v) w = (uint16_t)(0x3c00 + (rng() % 64));
    }

    // inputs
    std::vector<uint16_t> x((size_t)rows * H);
    for (auto& w : x) w = (uint16_t)(0x3800 + (rng() % 512));
    std::vector<int32_t> sel((size_t)rows * topk);
    std::vector<uint16_t> wts((size_t)rows * topk);
    for (auto& v : sel) v = (int32_t)(rng() % E);
    for (auto& w : wts) w = (uint16_t)(0x3800 + (rng() % 256));

    // stable storage for every tensor handed to a layer (deque: elements never move)
    static std::deque<std::vector<uint16_t>> keep;

    auto make_layer = [&](bool swz_mode) -> int64_t {
        // swz_mode: bytes are repacked the way the loader would for THIS process's tier
        std::vector<at::Tensor> gt, gs, gv, ut, us, uv, dt, ds, dv;
        for (int e = 0; e < E; ++e)
        {
            const int grp = swz_mode ? exl3_moe_cpu_swizzle_group(bits) : 0;
            const bool pl = swz_mode ? exl3_moe_cpu_planar_layout(bits) != 0 : false;
            auto rep = [&](std::vector<uint16_t>& nat, int tk, int tn) -> std::vector<uint16_t> {
                if (!grp) return nat;
                return loader_repack(nat, bits, tk, tn, grp, pl);
            };
            // one owned copy per (layer, expert, projection)
            keep.push_back(rep(g[e].tr, tiles_k_h, tiles_n_i)); auto& gtr = keep.back();
            keep.push_back(rep(u[e].tr, tiles_k_h, tiles_n_i)); auto& utr = keep.back();
            keep.push_back(rep(d[e].tr, tiles_k_i, tiles_n_h)); auto& dtr = keep.back();
            gt.push_back(wrap(gtr.data(), tiles_k_h, tiles_n_i, 16 * bits, at::kShort));
            gs.push_back(wrap1(g[e].suh.data(), H, at::kHalf));
            gv.push_back(wrap1(g[e].svh.data(), I, at::kHalf));
            ut.push_back(wrap(utr.data(), tiles_k_h, tiles_n_i, 16 * bits, at::kShort));
            us.push_back(wrap1(u[e].suh.data(), H, at::kHalf));
            uv.push_back(wrap1(u[e].svh.data(), I, at::kHalf));
            dt.push_back(wrap(dtr.data(), tiles_k_i, tiles_n_h, 16 * bits, at::kShort));
            ds.push_back(wrap1(d[e].suh.data(), I, at::kHalf));
            dv.push_back(wrap1(d[e].svh.data(), H, at::kHalf));
        }
        return exl3_moe_cpu_make_layer(gt, gs, gv, ut, us, uv, dt, ds, dv, {}, {}, {},
                                       0 /*silu*/, 0.0, swz_mode ? 1 : 0);
    };

    const int64_t h_native = make_layer(false);
    const int64_t h_swz = make_layer(true);

    std::vector<float> outA((size_t)rows * H), outB((size_t)rows * H);
    exl3_moe_cpu_forward_raw(h_native, reinterpret_cast<const at::Half*>(x.data()), sel.data(),
        reinterpret_cast<const at::Half*>(wts.data()), outA.data(), rows, topk, 4);
    exl3_moe_cpu_forward_raw(h_swz, reinterpret_cast<const at::Half*>(x.data()), sel.data(),
        reinterpret_cast<const at::Half*>(wts.data()), outB.data(), rows, topk, 4);

    int fails = 0;
    if (std::memcmp(outA.data(), outB.data(), outA.size() * 4) != 0)
    {
        double worst = 0; size_t wi = 0;
        for (size_t i = 0; i < outA.size(); ++i)
        {
            const double d = std::fabs((double)outA[i] - outB[i]);
            const double r = d / (std::fabs(outA[i]) + 1e-3);
            if (r > worst) { worst = r; wi = i; }
        }
        printf("K%d rows=%d: NOT identical (worst rel %.3g at %zu: %.6f vs %.6f)\n",
               bits, rows, worst, wi, outA[wi], outB[wi]);
        ++fails;
    }
    else printf("K%d rows=%d: bit-exact (%s, isa=%d)\n", bits, rows,
                exl3_moe_cpu_planar_layout(bits) ? "planar" : "swz/native", (int)g_isa);
    exl3_moe_cpu_free_layer(h_native);
    exl3_moe_cpu_free_layer(h_swz);
    return fails;
}

// CPU mirror of moe_unswizzle_kernel's planar branch (moe_unswizzle.cu): exact index
// transcription, verified to invert the loader repack for every shipped rate. The CUDA side
// cannot be executed on this dev box, so this is its only correctness gate -- keep in sync.
static int test_unswizzle_mirror()
{
    int fails = 0;
    const int tiles_k = 3, tiles_n = 8;   // group-2 legal
    std::mt19937 rng(77);
    for (int bits = 2; bits <= 8; ++bits)
    {
        const int tile_b = 32 * bits, ps = tile_b / 2;
        std::vector<uint16_t> native((size_t)tiles_k * tiles_n * ps);
        for (auto& w : native) w = (uint16_t)rng();
        auto staged = loader_repack(native, bits, tiles_k, tiles_n, 2, true);
        // native_out via the kernel's loop structure (one run = group tiles, dword copy)
        std::vector<uint16_t> native_out(staged.size());
        const int wd = tile_b / 4;
        for (int kt = 0; kt < tiles_k; ++kt)
            for (int g = 0; g < tiles_n / 8; ++g)
                for (int sub = 0; sub < 4; ++sub)     // 8/group = 4 sub-runs of group 2
                {
                    // mirrors the kernel's src/dst run addressing
                    const size_t sbase = ((size_t)g * 4 * tiles_k + kt + (size_t)sub * tiles_k) * 2 * ps;   // u16
                    const size_t dbase = ((size_t)kt * tiles_n + (size_t)g * 8 + sub * 2) * ps;
                    const uint32_t* s = reinterpret_cast<const uint32_t*>(staged.data() + sbase);
                    uint32_t* d = reinterpret_cast<uint32_t*>(native_out.data() + dbase);
                    for (int i = 0; i < 2 * wd; ++i)
                    {
                        const int q = i % wd;
                        d[(i / wd) * wd + bits * (q & 7) + (q >> 3)] = s[i];
                    }
                }
        if (native_out != native) { printf("unswizzle mirror FAIL bits=%d\n", bits); ++fails; }
    }
    printf(fails ? "unswizzle mirror: %d FAILURES\n" : "unswizzle mirror: PASS (K2-K8)\n", fails);
    return fails;
}

} // namespace

int main(int argc, char** argv)
{
    if (test_unswizzle_mirror()) return 1;
    int fails = 0;
    for (int bits : {1, 2, 3, 4, 5, 6, 7, 8})
        for (int rows : {1, 5, 9})
            fails += run_case(bits, 4, 512, 512, rows, 3, 999);
    printf(fails ? "FAILURES: %d\n" : "ALL PASS\n", fails);
    return fails != 0;
}