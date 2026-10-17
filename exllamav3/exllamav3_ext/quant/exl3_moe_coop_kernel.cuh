#pragma once

// Fused decode-shaped MoE kernels (bsz 1..MAX_BSZN), two ordinary launches per layer:
//
//   rot (bsz > 1): per (slot, chunk, projection) warp: x row (zero-padded to Hi) * suh -> Hadamard
//                 into the had_g / had_u scratch. At bsz 1 kernel A does this itself into shared
//                 memory and the launch is skipped.
//   A (gate/up):  one block per (expert run, 32-column group, projection). Slots are grouped by
//                 expert inside every block (identical redundant computation from sel/rw): a run
//                 is up to ROWS slots that picked the same expert, and the block runs the GEMV
//                 tile with those slots as the rows of one m16 MMA, so an expert's weights are read
//                 once per run rather than once per slot. Per row, the last of the 4 x nproj
//                 arrivals covering a 128-column chunk (completion counter) finishes the chunk:
//                 output rotation * svh, bias, activation, down-input rotation * down suh, stored
//                 as the down input.
//   B (down):     one block per (expert run, group), same multi-row tile on the rotated
//                 activation; the last arrival of a (token, chunk) (counter over active slots x 4
//                 groups) rotates every slot's result * svh, adds the down bias, sums the slots in
//                 a fixed order, merges the shared expert and stores the trimmed row.
//
// The grid is sized for the no-duplicates case (slots x groups x projections); blocks past the
// actual run count exit at once.

#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "../compat.cuh"
#include "exl3_gemv_kernel.cuh"
#include "hadamard_inner.cuh"
#include "exl3_moe_coop.cuh"

namespace exl3_moe_coop_ns {

constexpr int WK = 16;                          // k-split (warps per block)
constexpr int WNT = MOE_COOP_WNT;               // adjacent n-tiles per warp
constexpr int PF = 4;                           // prefetch ring depth
constexpr int FOLD = 4;                         // fp16 -> fp32 fold cadence
constexpr int THREADS = WK * 32;
constexpr int COLS = WNT * 16;                  // columns per warp
constexpr int ROWS = 8;                         // slots of one expert per block (MMA rows)
template <bool WIDE> constexpr int tile_wn() { return WIDE ? 4 : 1; }          // warps along n
template <bool WIDE> constexpr int tile_wk() { return WK / tile_wn<WIDE>(); }  // warps along k
template <bool WIDE> constexpr int tile_cols() { return tile_wn<WIDE>() * COLS; }
template <bool WIDE> constexpr int tile_gpc() { return tile_cols<WIDE>() >= 128 ? 1 : 128 / tile_cols<WIDE>(); }  // block groups per 128-chunk
template <bool WIDE> constexpr int tile_cpb() { return tile_cols<WIDE>() >= 128 ? tile_cols<WIDE>() / 128 : 1; }  // 128-chunks per block
constexpr int MAX_SLOTS = 256;                  // MAX_BSZN * top_k bound (top_k <= 32)
constexpr int STAGE_WORDS = WNT * 8 * 8;        // words per warp per k-slice at 8 bpw
constexpr float R_SCALE = 0.088388347648f;      // 1/sqrt(128)

static_assert(THREADS == MOE_COOP_THREADS, "block size mismatch");

// Register-dequant widths (shuffle extraction, no staging). 3 bpw stays on the staged generic
// path on Ampere where its register form loses to the block-pipelined kernel)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
template <int bits>
constexpr bool tile_reg() { return bits == 2 || bits == 4; }
#else
template <int bits>
constexpr bool tile_reg() { return bits == 2 || bits == 3 || bits == 4; }
#endif
// Shared-memory layout is sized on the host, which doesn't know the device arch: reserve the
// staging area for every width some arch stages (unused on the others)
// HALF: half-integer rate bits + 0.5 (mul1 only), always through the staged generic path
template <int bits, bool HALF = false>
constexpr bool tile_may_stage() { return HALF || !(bits == 2 || bits == 4); }

// Dynamic shared memory layout (bytes)
template <int bits, bool HALF = false>
__host__ __device__ constexpr int smem_stage_bytes() { return tile_may_stage<bits, HALF>() ? WK * STAGE_WORDS * 4 : 0; }
constexpr int smem_red_bytes() { return WK * ROWS * COLS * 4; }   // same for both geometries: wk x ROWS x cols
constexpr int smem_part_bytes() { return WK * 128 * 4; }
template <int bits, bool HALF = false>
__host__ __device__ constexpr int smem_a_bytes(int Hi) { return Hi * 2 + smem_red_bytes() + smem_stage_bytes<bits, HALF>(); }
template <int bits, bool HALF = false>
__host__ __device__ constexpr int smem_b_bytes() { return smem_red_bytes() + smem_stage_bytes<bits, HALF>() + smem_part_bytes(); }

// ---------------------------------------------------------------------------------------------
// Elementwise helpers (fp32 forms of activation_kernels.cuh, which can't be included twice)

__device__ __forceinline__ float act_silu(float x)
{
    float e = __expf(-x);
    return x * __fdividef(1.0f, 1.0f + e);
}

__device__ __forceinline__ float act_gelu(float x)
{
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float t = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh_opt(t));
}

__device__ __forceinline__ float act_relu2(float x)
{
    x = fmaxf(0.0f, x);
    return x * x;
}

// gpt-oss clamped swiglu: gate clamped from above only, up symmetrically, alpha = 1.702 in the
// sigmoid and +1 on the up path
__device__ __forceinline__ float act_oai_swiglu(float g, float u, float limit)
{
    if (limit != 0.0f)
    {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    float glu = g * (1.0f / (1.0f + __expf(-1.702f * g)));
    return (u + 1.0f) * glu;
}

// Gated activation with the act_limit semantics of act_mul_kernel_*: gate lane activated, up lane
// clamped symmetrically and the activated gate from above when a limit is set. Gateless experts
// pass g = u, giving relu(u) * u = relu2(u) exactly
__device__ __forceinline__ float act_gate(int act, bool gated, float g, float u, float limit)
{
    if (act == MOE_COOP_ACT_SILU_OAI) return act_oai_swiglu(g, u, limit);
    float x;
    if (!gated) x = fmaxf(0.0f, u);
    else if (act == MOE_COOP_ACT_SILU) x = act_silu(g);
    else if (act == MOE_COOP_ACT_GELU) x = act_gelu(g);
    else x = act_relu2(g);
    if (limit != 0.0f)
    {
        u = fminf(fmaxf(u, -limit), limit);
        x = fminf(x, limit);
    }
    return x * u;
}

// 128-element Hadamard across the warp on 4 values per lane, same butterfly as had_*_r_128_inner
__device__ __forceinline__ void had128(float& v0, float& v1, float& v2, float& v3, int lane)
{
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    v0 = h0 * R_SCALE;
    v1 = h1 * R_SCALE;
    v2 = h2 * R_SCALE;
    v3 = h3 * R_SCALE;
}

__device__ __forceinline__ void unpack_h4(uint2 u, float& v0, float& v1, float& v2, float& v3)
{
    half2 a = *reinterpret_cast<half2*>(&u.x);
    half2 b = *reinterpret_cast<half2*>(&u.y);
    v0 = __low2float(a);
    v1 = __high2float(a);
    v2 = __low2float(b);
    v3 = __high2float(b);
}

__device__ __forceinline__ void load_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    unpack_h4(*((const uint2*) p), v0, v1, v2, v3);
}

// L2-coherent loads for data written by other blocks of the same launch
__device__ __forceinline__ void load_h4_cg(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    unpack_h4(__ldcg((const uint2*) p), v0, v1, v2, v3);
}

__device__ __forceinline__ void load_f4_cg(const float* p, float& v0, float& v1, float& v2, float& v3)
{
    float4 f = __ldcg((const float4*) p);
    v0 = f.x; v1 = f.y; v2 = f.z; v3 = f.w;
}

__device__ __forceinline__ void store_h4(half* p, float v0, float v1, float v2, float v3)
{
    half2 a = __floats2half2_rn(v0, v1);
    half2 b = __floats2half2_rn(v2, v3);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *((uint2*) p) = u;
}

__device__ __forceinline__ void scale_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    float s0, s1, s2, s3;
    load_h4(p, s0, s1, s2, s3);
    v0 *= s0; v1 *= s1; v2 *= s2; v3 *= s3;
}

__device__ __forceinline__ void add_h4(const half* p, float& v0, float& v1, float& v2, float& v3)
{
    float s0, s1, s2, s3;
    load_h4(p, s0, s1, s2, s3);
    v0 += s0; v1 += s1; v2 += s2; v3 += s3;
}

struct SlotInfo
{
    int row;
    int local;
    float w;
    bool active;
};

__device__ __forceinline__ SlotInfo slot_info(const MoeCoopParams& p, int s)
{
    SlotInfo si;
    si.row = s / p.topk;
    int64_t e = p.sel[s];
    si.w = __half2float(p.rw[s]);
    bool ok = si.w != 0.0f;
    if (p.min_expert >= 0)
    {
        ok = ok && e >= p.min_expert && e < p.max_expert;
        e -= p.min_expert;
    }
    si.local = ok ? (int) e : 0;
    si.active = ok;
    return si;
}

// ---------------------------------------------------------------------------------------------
// Slot grouping (bsz > 1): active slots sorted by (local expert, slot) and cut into runs of at
// most ROWS slots of one expert. Built once per call by block 0 of the rotation kernel into the
// run table (p.runs: [n_runs, -, run_start[0..], order[0..]]), read by every block of A and B.
// At bsz 1 top-k picks are distinct, so runs are the slots themselves and no table is built

__device__ __forceinline__ void build_runs
(
    const MoeCoopParams& p, int16_t* sh_order, int16_t* sh_local, int* sh_scan, int* sh_res
)
{
    const int slots = p.bsz * p.topk;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    int* run_start = p.runs + 2;
    int* order = p.runs + 2 + p.slots_max + 1;

    // Routing of every slot: local expert or -1 when inactive
    if (t < slots)
    {
        const SlotInfo si = slot_info(p, t);
        sh_local[t] = si.active ? (int16_t) si.local : (int16_t) -1;
    }
    if (t == 0) sh_res[0] = 0;
    __syncthreads();

    // Rank of every active slot by (expert, slot)
    const int my_e = t < slots ? sh_local[t] : -1;
    if (my_e >= 0)
    {
        const int key = (my_e << 8) | t;
        int rank = 0;
        for (int u = 0; u < slots; ++u)
        {
            const int eu = sh_local[u];
            rank += (eu >= 0 && ((eu << 8) | u) < key) ? 1 : 0;
        }
        sh_order[rank] = (int16_t) t;
        atomicMax(sh_res, rank + 1);     // active slot count
    }
    __syncthreads();
    const int n_active = sh_res[0];

    // Run starts: a new expert, or ROWS slots into the same expert (first index of an expert =
    // number of active slots with a smaller expert)
    int is_start = 0;
    if (t < n_active)
    {
        const int e_t = sh_local[sh_order[t]];
        int first = 0;
        for (int u = 0; u < slots; ++u)
        {
            const int eu = sh_local[u];
            first += (eu >= 0 && eu < e_t) ? 1 : 0;
        }
        is_start = (t == 0 || sh_local[sh_order[t - 1]] != e_t || (t - first) % ROWS == 0) ? 1 : 0;
        order[t] = sh_order[t];
    }
    // Exclusive prefix count of run starts over the block (warp scan + warp totals)
    int v = is_start;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int n = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += n;
    }
    if (lane == 31) sh_scan[warp] = v;
    __syncthreads();
    int base = 0, total_runs = 0;
    for (int w = 0; w < WK; ++w)
    {
        if (w < warp) base += sh_scan[w];
        total_runs += sh_scan[w];
    }
    const int run_id = base + v - is_start;
    if (is_start) run_start[run_id] = t;
    if (t == 0)
    {
        run_start[total_runs] = n_active;
        p.runs[0] = total_runs;
    }
}

// The block's run: its slot indices into rows[] (shared) and the row count; false past the end
__device__ __forceinline__ bool read_run(const MoeCoopParams& p, int run_idx, int* rows, int& nrows, int* sh_res)
{
    if (!p.a_global)
    {
        // bsz 1: run = slot (an inactive slot returns false; kernel A checks for empty rows)
        if (run_idx >= p.bsz * p.topk) return false;
        const SlotInfo si = slot_info(p, run_idx);
        if (!si.active) return false;
        if (threadIdx.x == 0) rows[0] = run_idx;
        nrows = 1;
        __syncthreads();
        return true;
    }
    const int n_runs = p.runs[0];
    if (run_idx >= n_runs) return false;
    const int start = p.runs[2 + run_idx];
    const int end = p.runs[2 + run_idx + 1];
    nrows = end - start;
    if (threadIdx.x < nrows) rows[threadIdx.x] = p.runs[2 + p.slots_max + 1 + start + threadIdx.x];
    __syncthreads();
    return true;
}

// ---------------------------------------------------------------------------------------------
// GEMV tile: one 32-column group of an (nrows, k) x (k, n) product, the rows being the run's slots
// (row r reads A at A2 + rows[r] * a_stride2 half2 and writes C at rows[r] * c_stride), the whole
// block splitting k across its 16 warps. Body follows exl3_gemv_kernel (MMODE 1, CFG 0) for
// 2/3/4 bpw; other widths stage each k-slice's tile words in shared memory and decode with
// dq_dispatch

template <int bits, int cb, bool WIDE, bool HALF = false>
__device__ __forceinline__ void gemv_tile
(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    size_t a_stride2,
    const int* __restrict__ rows,
    int nrows,
    void* __restrict__ C,
    size_t c_stride,
    bool c_f32,
    int k_begin,                        // k-slice range of this block (split-k across blocks)
    int k_end,
    int ntiles,
    int group,                          // block column group of tile_cols<WIDE>() columns
    float* __restrict__ sh_red,         // [WKK][ROWS][TCOLS]
    uint32_t* __restrict__ sh_stage     // [WK][STAGE_WORDS], staged widths only
)
{
    constexpr bool REG = HALF ? false : tile_reg<bits>();
    constexpr int TWORDS = HALF ? 4 * (2 * bits + 1) : 8 * bits;              // uint32 per 16x16 tile
    constexpr int GWORDS = WNT * TWORDS;                                        // uint32 per warp per k-slice
    constexpr int LOADS = REG ? (bits == 2 ? WNT / 2 : WNT) : CEIL_DIVIDE(GWORDS, 32);
    constexpr int LSTRIDE = (REG && bits == 3) ? 24 : 32;
    constexpr int WN = tile_wn<WIDE>();
    constexpr int WKK = tile_wk<WIDE>();
    constexpr int TCOLS = tile_cols<WIDE>();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int wn = warp % WN;                   // adjacent warps read adjacent column pieces
    const int wk = warp / WN;

    const int kslices = k_end - k_begin;
    const int chunk = CEIL_DIVIDE(kslices, WKK);
    const int ks0 = k_begin + wk * chunk;
    const int myn = max(0, min(chunk, k_end - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    // A fragment row of this lane: rows[lane / 4] of the run
    const int r0 = lane >> 2;
    const bool r0_ok = r0 < nrows;
    const size_t a_row0 = r0_ok ? (size_t) rows[r0] * a_stride2 : 0;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + (group * WN + wn) * GWORDS + lane;
    [[maybe_unused]] uint32_t* stage = sh_stage + warp * STAGE_WORDS;

    auto ld_b = [&] (int i, int l) -> uint32_t
    {
        if constexpr (REG && bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else if constexpr (!REG)
            return (l * 32 + lane < GWORDS) ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF)
    {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (!REG)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (l * 32 + lane < GWORDS)
                        stage[l * 32 + lane] = bw[l];
                __syncwarp();
            }

            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (!REG)
                {
                    dq_dispatch<bits, cb, HALF>(stage + t * TWORDS, lane << 3, f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
    // tile*16 + frag*8 + 2*(l%4) (+1)
    if (r0_ok)
    {
        const int c0 = 2 * (lane & 3);
        float* red = sh_red + (wk * ROWS + r0) * TCOLS + wn * COLS;
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int col = t * 16 + f * 8 + c0;
                red[col + 0] = acc0[t][f].x;
                red[col + 1] = acc0[t][f].y;
            }
    }
    __syncthreads();

    for (int o = threadIdx.x; o < TCOLS * nrows; o += THREADS)     // up to 128 x 8 outputs per block
    {
        const int r = o / TCOLS;
        const int c = o % TCOLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WKK; ++j)
            sum += sh_red[(j * ROWS + r) * TCOLS + c];
        const size_t idx = (size_t) rows[r] * c_stride + group * TCOLS + c;
        if (c_f32) ((float*) C)[idx] = sum;
        else       ((half*) C)[idx] = __float2half_rn(sum);
    }
    __syncthreads();
}

// Completion counter: every thread fences its own stores, then one thread arrives. Returns true
// (block-uniform) for the last arriving block, which may then read the other blocks' results
// through L2 (load_*_cg)
__device__ __forceinline__ bool arrive_last(int* counter, int expected, int* sh_flag)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        *sh_flag = (atomicAdd(counter, 1) == expected - 1);
    __syncthreads();
    bool last = *sh_flag;
    if (last) __threadfence();
    return last;
}

// Number of active slots of a token
__device__ __forceinline__ int token_active_slots(const MoeCoopParams& p, int row)
{
    int n = 0;
    for (int k = 0; k < p.topk; ++k)
        n += slot_info(p, row * p.topk + k).active ? 1 : 0;
    return n;
}

// A token with no active slot on this rank (expert-parallel sharding: every pick lives elsewhere)
// gets no arrival in the down stage, so nobody would write its output row. Writes one 128-chunk of
// that row: the shared-expert term when there is one, else zeros. Warp-level
__device__ __forceinline__ void write_empty_row_chunk(const MoeCoopParams& p, int row, int chunk, int lane)
{
    const int col = chunk * 128 + lane * 4;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    if (p.sh_out)
    {
        float gv = 1.0f;
        if (p.sh_gate_w)
        {
            const half* xr = p.x + (size_t) row * p.x_stride;
            float dot = 0.0f;
            for (int i = lane * 2; i < p.H; i += 64)
            {
                half2 xv = *((const half2*) (xr + i));
                half2 wv = *((const half2*) (p.sh_gate_w + i));
                dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            gv = 1.0f / (1.0f + __expf(-dot));
        }
        const float* sh = p.sh_out + (size_t) row * p.H + col;
        if (col + 0 < p.H) o0 = gv * sh[0];
        if (col + 1 < p.H) o1 = gv * sh[1];
        if (col + 2 < p.H) o2 = gv * sh[2];
        if (col + 3 < p.H) o3 = gv * sh[3];
    }
    float* dst = p.out + (size_t) row * p.out_stride + col;
    if (col + 3 < p.H_out)
        *((float4*) dst) = make_float4(o0, o1, o2, o3);
    else
    {
        if (col + 0 < p.H_out) dst[0] = o0;
        if (col + 1 < p.H_out) dst[1] = o1;
        if (col + 2 < p.H_out) dst[2] = o2;
    }
}

// Input rotation of one 128-chunk of a slot's x row: zero-padded from H to Hi, * suh, Hadamard
__device__ __forceinline__ void rotate_chunk(const MoeCoopParams& p, int row, const half* suh, int c, half* dst, int lane)
{
    const int col = c * 128 + lane * 4;
    const half* xr = p.x + (size_t) row * p.x_stride;
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (col + 3 < p.H)
        load_h4(xr + col, v0, v1, v2, v3);
    else
    {
        if (col + 0 < p.H) v0 = __half2float(xr[col + 0]);
        if (col + 1 < p.H) v1 = __half2float(xr[col + 1]);
        if (col + 2 < p.H) v2 = __half2float(xr[col + 2]);
    }
    scale_h4(suh + col, v0, v1, v2, v3);
    had128(v0, v1, v2, v3, lane);
    store_h4(dst + col, v0, v1, v2, v3);
}

// ---------------------------------------------------------------------------------------------
// Rotation pre-kernel (bsz > 1): warp per (slot, chunk, projection) into had_g / had_u

#ifdef EXL3_MOE_COOP_DEFINE_ROT
__global__ __launch_bounds__(THREADS)
void exl3_moe_coop_rot_kernel(const MoeCoopParams p)
{
    __shared__ int16_t sh_order[MAX_SLOTS];
    __shared__ int16_t sh_local[MAX_SLOTS];
    __shared__ int sh_scan[WK];
    __shared__ int sh_res[4];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int chunks = p.Hi / 128;
    const int slots = p.bsz * p.topk;
    if (blockIdx.x == 0) build_runs(p, sh_order, sh_local, sh_scan, sh_res);
    const int item = blockIdx.x * (THREADS / 32) + warp;
    if (item >= slots * chunks * nproj) return;
    const int s = item / (chunks * nproj);
    const int rem = item % (chunks * nproj);
    const int c = rem / nproj;
    const bool is_gate = p.gated && (rem % nproj) == 0;
    const SlotInfo si = slot_info(p, s);
    if (!si.active)
    {
        // Empty token row (no active slot on this rank): its first slot's warps write the output
        // chunks c, c + chunks, ... (Hi chunks vs Ho chunks may differ)
        if (rem % nproj == 0 && s % p.topk == 0 && token_active_slots(p, si.row) == 0)
            for (int oc = c; oc < p.Ho / 128; oc += chunks)
                write_empty_row_chunk(p, si.row, oc, lane);
        return;
    }
    const half* suh = (const half*) (is_gate ? p.g_suh : p.u_suh)[si.local];
    half* dst = (is_gate ? p.had_g : p.had_u) + (size_t) s * p.Hi;
    rotate_chunk(p, si.row, suh, c, dst, lane);
}
#else
__global__ void exl3_moe_coop_rot_kernel(const MoeCoopParams p);
#endif

// ---------------------------------------------------------------------------------------------
// Kernel A: gate/up GEMV per (expert run, group, projection) with the chunk epilogue (rotation,
// bias, activation, down-input rotation) by the last arrival per (slot, chunk)

template <int bits, int cb, bool WIDE, bool HALF = false>
__global__ __launch_bounds__(THREADS)
void exl3_moe_coop_a_kernel(const MoeCoopParams p)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    extern __shared__ uint32_t smem_dyn[];
    half* sh_A = (half*) smem_dyn;
    float* sh_red = (float*) (smem_dyn + p.Hi / 2);
    uint32_t* sh_stage = smem_dyn + p.Hi / 2 + WK * ROWS * COLS;
    __shared__ int sh_flag;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int ng = p.I / TCOLS;

    // Reset the down stage's counters for this call (its previous use finished in stream order)
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS)
        p.ctr_b[i] = 0;

    const int item = blockIdx.x;
    const bool is_gate = p.gated && (item % nproj) == 0;
    const int rem = item / nproj;
    const int ks = rem % p.ksplit_a;                 // split-k index: partial output rows
    const int rem2 = rem / p.ksplit_a;
    const int run_idx = rem2 / ng;
    const int group = rem2 % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res))
    {
        // bsz 1: an inactive slot's blocks (up projection, split 0) write the output chunks of an
        // empty token row, one warp per chunk, strided by the number of gate/up groups
        if (!p.a_global && run_idx < p.bsz * p.topk && !is_gate && ks == 0 && run_idx % p.topk == 0)
        {
            const int row = run_idx / p.topk;
            if (token_active_slots(p, row) == 0)
                for (int oc = group + warp * ng; oc < p.Ho / 128; oc += ng * WK)
                    write_empty_row_chunk(p, row, oc, lane);
        }
        return;
    }
    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.Hi / 16;
    const int k_begin = kslices_all * ks / p.ksplit_a;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_a;

    const half* suh = (const half*) (is_gate ? p.g_suh : p.u_suh)[local];
    const half2* A2;
    size_t a_stride2;
    if (p.a_global)
    {
        A2 = (const half2*) (is_gate ? p.had_g : p.had_u);
        a_stride2 = (size_t) p.Hi / 2;
    }
    else
    {
        // bsz 1: every run is a single slot; rotate its input into shared memory here. A zero
        // row stride makes the tile read it whatever the slot index (which still selects the C row)
        const SlotInfo si = slot_info(p, sh_rows[0]);
        for (int c = warp; c < p.Hi / 128; c += WK)
            rotate_chunk(p, si.row, suh, c, sh_A, lane);
        __syncthreads();
        A2 = (const half2*) sh_A;
        a_stride2 = 0;
    }

    {
        const uint32_t* B32 = (const uint32_t*) (is_gate ? p.g_trellis : p.u_trellis)[local];
        // Partial output rows of split ks live at slot + ks * slots (ksplit * slots <= slots_max)
        const size_t part = (size_t) ks * (p.bsz * p.topk) * p.I * (p.gu_f32 ? 4 : 2);
        void* C = (void*) (((char*) (is_gate ? p.gu_g : p.gu_u)) + part);
        gemv_tile<bits, cb, WIDE, HALF>(B32, A2, a_stride2, sh_rows, nrows, C, p.I, p.gu_f32, k_begin, k_end, p.I / 16, group, sh_red, sh_stage);
    }

    for (int r = 0; r < nrows; ++r)
    for (int j = 0; j < CPB; ++j)
    {
        const int s = sh_rows[r];
        const int chunk = (group * CPB + j) / GPC;
        if (!arrive_last(p.ctr_a + (size_t) s * (p.I / 128) + chunk, GPC * nproj * p.ksplit_a, &sh_flag)) continue;
        if (p.dbg & 1) continue;
        if (warp == 0)
        {
            const int col = chunk * 128 + lane * 4;
            const size_t off = (size_t) s * p.I + col;
            const size_t pstride = (size_t) (p.bsz * p.topk) * p.I;

            // Sum of the split-k partials (fixed order)
            auto load_sum = [&] (const void* buf, float& v0, float& v1, float& v2, float& v3)
            {
                v0 = v1 = v2 = v3 = 0.0f;
                for (int q = 0; q < p.ksplit_a; ++q)
                {
                    float t0, t1, t2, t3;
                    if (p.gu_f32) load_f4_cg(((const float*) buf) + q * pstride + off, t0, t1, t2, t3);
                    else          load_h4_cg(((const half*) buf) + q * pstride + off, t0, t1, t2, t3);
                    v0 += t0; v1 += t1; v2 += t2; v3 += t3;
                }
            };
            float u0, u1, u2, u3;
            load_sum(p.gu_u, u0, u1, u2, u3);
            had128(u0, u1, u2, u3, lane);
            scale_h4(((const half*) p.u_svh[local]) + col, u0, u1, u2, u3);
            if (p.u_bias) add_h4(((const half*) p.u_bias[local]) + col, u0, u1, u2, u3);

            float g0 = u0, g1 = u1, g2 = u2, g3 = u3;
            if (p.gated)
            {
                load_sum(p.gu_g, g0, g1, g2, g3);
                had128(g0, g1, g2, g3, lane);
                scale_h4(((const half*) p.g_svh[local]) + col, g0, g1, g2, g3);
                if (p.g_bias) add_h4(((const half*) p.g_bias[local]) + col, g0, g1, g2, g3);
            }

            float a0 = act_gate(p.act, p.gated, g0, u0, p.act_limit);
            float a1 = act_gate(p.act, p.gated, g1, u1, p.act_limit);
            float a2 = act_gate(p.act, p.gated, g2, u2, p.act_limit);
            float a3 = act_gate(p.act, p.gated, g3, u3, p.act_limit);

            scale_h4(((const half*) p.d_suh[local]) + col, a0, a1, a2, a3);
            had128(a0, a1, a2, a3, lane);
            store_h4(p.act_out + off, a0, a1, a2, a3);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel B: down GEMV per (expert run, group); the last arrival of a (token, chunk) reduces the
// token's slots (rotation * svh, down bias, routing weight) in a fixed order, merges the shared
// expert and stores the trimmed row

template <int bits, int cb, bool WIDE, bool HALF = false>
__global__ __launch_bounds__(THREADS)
void exl3_moe_coop_b_kernel(const MoeCoopParams p)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    extern __shared__ uint32_t smem_dyn[];
    float* sh_red = (float*) smem_dyn;
    uint32_t* sh_stage = smem_dyn + WK * ROWS * COLS;
    float* sh_part = (float*) (smem_dyn + WK * ROWS * COLS + (tile_may_stage<bits, HALF>() ? WK * STAGE_WORDS : 0));
    __shared__ int sh_flag;
    __shared__ int sh_n_active;
    __shared__ float sh_gate;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ng = p.Ho / TCOLS;

    // Reset the gate/up stage's counters for the next call
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * THREADS)
        p.ctr_a[i] = 0;

    const int item = blockIdx.x;
    const int ks = item % p.ksplit_b;
    const int rem = item / p.ksplit_b;
    const int run_idx = rem / ng;
    const int group = rem % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res)) return;
    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.I / 16;
    const int k_begin = kslices_all * ks / p.ksplit_b;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_b;

    {
        const uint32_t* B32 = (const uint32_t*) p.d_trellis[local];
        float* C = p.d_out + (size_t) ks * (p.bsz * p.topk) * p.Ho;
        gemv_tile<bits, cb, WIDE, HALF>(B32, (const half2*) p.act_out, (size_t) p.I / 2, sh_rows, nrows, C, p.Ho, true,
                                  k_begin, k_end, p.Ho / 16, group, sh_red, sh_stage);
    }

    for (int r = 0; r < nrows; ++r)
    for (int j = 0; j < CPB; ++j)
    {
        const int s = sh_rows[r];
        const int row = s / p.topk;
        const int chunk = (group * CPB + j) / GPC;

        // Active slots of this token: the number of arrivals to expect for (row, chunk)
        if (threadIdx.x == 0)
        {
            int n = 0;
            for (int k = 0; k < p.topk; ++k)
                n += slot_info(p, row * p.topk + k).active ? 1 : 0;
            sh_n_active = n;
        }
        __syncthreads();
        if (!arrive_last(p.ctr_b + (size_t) row * (p.Ho / 128) + chunk, sh_n_active * GPC * p.ksplit_b, &sh_flag)) continue;
        if (p.dbg & 2) continue;

        const int col = chunk * 128 + lane * 4;

        // Shared-expert gate: sigmoid(x_row . w), fp32 across the block
        if (p.sh_gate_w)
        {
            const half* xr = p.x + (size_t) row * p.x_stride;
            float dot = 0.0f;
            for (int i = threadIdx.x * 2; i < p.H; i += THREADS * 2)
            {
                half2 xv = *((const half2*) (xr + i));
                half2 wv = *((const half2*) (p.sh_gate_w + i));
                dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) sh_part[warp] = dot;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                float tt = 0.0f;
                for (int w = 0; w < WK; ++w) tt += sh_part[w];
                sh_gate = 1.0f / (1.0f + __expf(-tt));
            }
            __syncthreads();
        }

        // Each warp takes slots warp, warp + WK, ...; per-warp partials are summed in warp order
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int k = warp; k < p.topk; k += WK)
        {
            const int sk = row * p.topk + k;
            const SlotInfo sj = slot_info(p, sk);
            if (!sj.active) continue;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            for (int q = 0; q < p.ksplit_b; ++q)     // split-k partials, fixed order
            {
                float t0, t1, t2, t3;
                load_f4_cg(p.d_out + ((size_t) q * (p.bsz * p.topk) + sk) * p.Ho + col, t0, t1, t2, t3);
                v0 += t0; v1 += t1; v2 += t2; v3 += t3;
            }
            had128(v0, v1, v2, v3, lane);
            scale_h4(((const half*) p.d_svh[sj.local]) + col, v0, v1, v2, v3);
            if (p.d_bias) add_h4(((const half*) p.d_bias[sj.local]) + col, v0, v1, v2, v3);
            o0 += sj.w * v0;
            o1 += sj.w * v1;
            o2 += sj.w * v2;
            o3 += sj.w * v3;
        }
        float* part = sh_part + warp * 128 + lane * 4;
        part[0] = o0; part[1] = o1; part[2] = o2; part[3] = o3;
        __syncthreads();
        if (warp == 0)
        {
            o0 = 0.0f; o1 = 0.0f; o2 = 0.0f; o3 = 0.0f;
            #pragma unroll
            for (int w = 0; w < WK; ++w)
            {
                const float* q = sh_part + w * 128 + lane * 4;
                o0 += q[0]; o1 += q[1]; o2 += q[2]; o3 += q[3];
            }

            if (p.sh_out)
            {
                const float gv = p.sh_gate_w ? sh_gate : 1.0f;
                const float* sh = p.sh_out + (size_t) row * p.H + col;
                if (col + 0 < p.H) o0 += gv * sh[0];
                if (col + 1 < p.H) o1 += gv * sh[1];
                if (col + 2 < p.H) o2 += gv * sh[2];
                if (col + 3 < p.H) o3 += gv * sh[3];
            }

            float* dst = p.out + (size_t) row * p.out_stride + col;
            if (col + 3 < p.H_out)
                *((float4*) dst) = make_float4(o0, o1, o2, o3);
            else
            {
                if (col + 0 < p.H_out) dst[0] = o0;
                if (col + 1 < p.H_out) dst[1] = o1;
                if (col + 2 < p.H_out) dst[2] = o2;
            }
        }
        __syncthreads();
    }
}

}  // namespace exl3_moe_coop_ns
