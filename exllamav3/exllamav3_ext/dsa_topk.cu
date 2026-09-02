#include <cuda_fp16.h>
#include <climits>
#include "dsa_topk.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"
#include <mutex>
#include <map>

/*
top-k index selection for the DSA lightning indexer.

Radix select on the 16-bit ordered key of the fp16 score (monotonic bit flip): two
256-bucket histogram passes pin down the exact k-th threshold value v*, then ordered
stream compaction emits all entries with score > v* followed by enough == v* ties.
*/

#define TOPK_THREADS 1024

__device__ __forceinline__ uint16_t topk_key_u(const uint16_t u)
{
    return (u & 0x8000) ? (uint16_t) ~u : (uint16_t) (u | 0x8000);
}

// Descending radix bucket search, warp 0 only: find the highest bucket b such that the
// count of entries in buckets ABOVE b is < target <= count including b. Writes res[0] = b
// (-1 if the total is below target) and res[1] = count strictly above b. Lane l owns the
// descending 8-bucket group [255 - 8l .. 248 - 8l]; one shfl scan replaces the serial
// 256-bucket walk (a ~3us latency chain at the head of both histogram passes)
__device__ __forceinline__ void topk_find_bucket(const int* hist, int target, int lane, int* res)
{
    int g0 = 255 - 8 * lane;
    int gs = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) gs += hist[g0 - j];
    int incl = gs;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int n = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += n;
    }
    int excl = incl - gs;
    bool cross = excl < target && incl >= target;
    unsigned bal = __ballot_sync(0xffffffffu, cross);
    if (!bal)
    {
        if (lane == 31) { res[0] = -1; res[1] = incl; }
        return;
    }
    if (lane == __ffs(bal) - 1)
    {
        int acc = excl;
        int b = g0;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            b = g0 - j;
            if (acc + hist[b] >= target) break;
            acc += hist[b];
        }
        res[0] = b;
        res[1] = acc;
    }
}

__device__ __forceinline__ uint16_t topk_key(const half h)
{
    return topk_key_u(__half_as_ushort(h));
}

template <bool VEC>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_kernel
(
    const half* __restrict__ scores,     // (R, s_stride), -inf past each row's causal bound
    int* __restrict__ out,               // (R, k_pad) int32, -1 padded
    int T,
    const int s_stride,
    const int k,
    const int k_pad,
    const int* __restrict__ t_ptr,       // device T override (graph modes)
    const int t_seq                      // > 0: t_ptr is per-JOB, T = t_ptr[row / t_seq]
                                         // (rows only scan their own causal region, so no
                                         // -inf backfill of the score buffer is needed)
)
{
    if (t_ptr) T = t_seq > 0 ? t_ptr[blockIdx.x / t_seq] : *t_ptr;
    constexpr uint16_t KEY_NEG_INF = 0x03ff;   // topk_key(0xfc00)

    const half* row_s = scores + (size_t) blockIdx.x * s_stride;
    int* row_o = out + (size_t) blockIdx.x * k_pad;
    int t = threadIdx.x;
    int lane = t % 32;
    int warp = t / 32;
    constexpr int NUM_WARPS = TOPK_THREADS / 32;

    __shared__ int hist[256];
    __shared__ int warp_cnt[NUM_WARPS];
    __shared__ int warp_off[NUM_WARPS];
    __shared__ int sh_res[2];
    __shared__ int sh_tile_total;

    // Pass 1: histogram of the high key byte. Vectorized: uint4 loads, 8 keys per LDG
    // (rows are 16-byte aligned: the host checks s_stride % 8 == 0), scalar tail
    const uint4* row_s8 = (const uint4*) row_s;
    const int T8 = VEC ? T >> 3 : 0;
    if (t < 256) hist[t] = 0;
    __syncthreads();
    for (int i = t; i < T8; i += TOPK_THREADS)
    {
        uint4 v = row_s8[i];
        uint32_t ws[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint16_t ka = topk_key_u((uint16_t) ws[j]);
            uint16_t kb = topk_key_u((uint16_t) (ws[j] >> 16));
            if (ka > KEY_NEG_INF) atomicAdd(&hist[ka >> 8], 1);
            if (kb > KEY_NEG_INF) atomicAdd(&hist[kb >> 8], 1);
        }
    }
    for (int i = T8 * 8 + t; i < T; i += TOPK_THREADS)
    {
        uint16_t key = topk_key(row_s[i]);
        if (key > KEY_NEG_INF)
            atomicAdd(&hist[key >> 8], 1);
    }
    __syncthreads();
    if (warp == 0)
        topk_find_bucket(hist, k, lane, sh_res);
    __syncthreads();
    int b1 = sh_res[0];          // -1: fewer than k finite entries -> take them all
    int cnt_hi = sh_res[1];      // entries with high byte strictly above b1

    uint16_t thresh;
    int n_eq_take;
    if (b1 < 0)
    {
        // Everything finite is selected: threshold just above -inf, no tie cap
        thresh = KEY_NEG_INF;
        n_eq_take = 0;
    }
    else
    {
        // Pass 2: histogram of the low key byte among entries with high byte == b1
        // (hist reads of pass 1 all happened before the last barrier)
        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int i = t; i < T8; i += TOPK_THREADS)
        {
            uint4 v = row_s8[i];
            uint32_t ws[4] = { v.x, v.y, v.z, v.w };
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                uint16_t ka = topk_key_u((uint16_t) ws[j]);
                uint16_t kb = topk_key_u((uint16_t) (ws[j] >> 16));
                if (ka > KEY_NEG_INF && (ka >> 8) == b1) atomicAdd(&hist[ka & 0xff], 1);
                if (kb > KEY_NEG_INF && (kb >> 8) == b1) atomicAdd(&hist[kb & 0xff], 1);
            }
        }
        for (int i = T8 * 8 + t; i < T; i += TOPK_THREADS)
        {
            uint16_t key = topk_key(row_s[i]);
            if (key > KEY_NEG_INF && (key >> 8) == b1)
                atomicAdd(&hist[key & 0xff], 1);
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k - cnt_hi, lane, sh_res);
        __syncthreads();
        int b0 = max(sh_res[0], 0);             // in-band search cannot miss; clamp defensively
        thresh = (uint16_t) ((b1 << 8) | b0);
        n_eq_take = (k - cnt_hi) - sh_res[1];   // ties at v* still needed after all > v*
    }

    // Passes 3a/3b: ordered compaction, 8 elements per thread per tile (uint4 loads).
    // Position of element i is (tile, warp, lane, intra-lane bit). Monotonic in i, so
    // the ascending-index emission order is preserved. Per tile: one intra-warp shfl scan
    // + a warp-0 scan over warp totals; the running output base stays in a REGISTER
    // (block-uniform: everyone adds the same tile total read between the two barriers),
    // so a tile costs 2 barriers for 8 * TOPK_THREADS elements instead of 3 per
    // TOPK_THREADS. The thread whose uint4 slot straddles T picks up the scalar tail
    const int n_slots = (T + 7) >> 3;   // slots above T8 take the scalar path
    const int n_tiles = (n_slots + TOPK_THREADS - 1) / TOPK_THREADS;
    int base = 0;

    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        // pass 0: keys > thresh (all emitted); pass 1: ties == thresh, capped
        const int eq_start = base;
        const int cap_end = pass == 0 ? INT_MAX : eq_start + n_eq_take;
        for (int t0 = 0; t0 < n_tiles && base < cap_end; ++t0)
        {
            int i8 = t0 * TOPK_THREADS + t;
            unsigned m = 0;
            if (VEC && i8 < T8)
            {
                uint4 v = row_s8[i8];
                uint32_t ws[4] = { v.x, v.y, v.z, v.w };
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    uint16_t ka = topk_key_u((uint16_t) ws[j]);
                    uint16_t kb = topk_key_u((uint16_t) (ws[j] >> 16));
                    bool pa = pass == 0 ? (ka > thresh) : (ka == thresh);
                    bool pb = pass == 0 ? (kb > thresh) : (kb == thresh);
                    if (pa && ka > KEY_NEG_INF) m |= 1u << (2 * j);
                    if (pb && kb > KEY_NEG_INF) m |= 1u << (2 * j + 1);
                }
            }
            else if (i8 >= T8 && i8 < n_slots)
            {
                // Scalar strip: the straddling slot's tail (VEC) or every slot (!VEC).
                // Same 8-element ownership, so the emission order is unchanged
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    int idx = i8 * 8 + j;
                    if (idx >= T) break;
                    uint16_t kx = topk_key(row_s[idx]);
                    bool px = pass == 0 ? (kx > thresh) : (kx == thresh);
                    if (px && kx > KEY_NEG_INF) m |= 1u << j;
                }
            }
            int c = __popc(m);
            int incl = c;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
                int n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            if (lane == 31) warp_cnt[warp] = incl;
            __syncthreads();
            if (warp == 0)
            {
                int wc = lane < NUM_WARPS ? warp_cnt[lane] : 0;
                int wincl = wc;
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1)
                {
                    int n = __shfl_up_sync(0xffffffffu, wincl, o);
                    if (lane >= o) wincl += n;
                }
                if (lane < NUM_WARPS) warp_off[lane] = wincl - wc;
                if (lane == 31) sh_tile_total = wincl;
            }
            __syncthreads();
            int obase = base + warp_off[warp] + (incl - c);
            int ibase = i8 * 8;
            while (m)
            {
                int j = __ffs((int) m) - 1;
                m &= m - 1;
                if (obase < cap_end)
                    row_o[obase] = ibase + j;
                ++obase;
            }
            base += sh_tile_total;      // uniform: read between the barriers, added by all
        }
        if (pass == 1)
            base = min(base, cap_end);
    }
    int emitted = base;

    // -1 padding
    for (int j = emitted + t; j < k_pad; j += TOPK_THREADS)
        row_o[j] = -1;
}


/*
Split/merge variant for few-row selections (decode): the single-block-per-row kernel is
parallelism-starved when R is small and T is large (three sequential sweeps of T by one
block). Phase 1 selects each of TOPK_SPLIT_G spans' local top-k in parallel (a guaranteed
superset of the row's global top-k) and phase 2 selects over the <= G * k candidates.
Candidates are stored span-major with in-span ascending order, so the merge's ordered
compaction resolves ties in ascending global index order, identical to the legacy kernel.
*/

#define TOPK_SPLIT_G 32
#define TOPK_SPLIT_MIN_T 32768
#define TOPK_SPLIT_MAX_R 16
#define TOPK_SPLIT_MAX_KPAD 4096

// Fixed-size per-device candidate workspace: allocated once at first use (warmup precedes
// graph capture) and never grown, so captured graphs can never hold a dangling pointer.
// Replays and eager calls on one device are stream-ordered, so sharing one buffer is safe
static at::Tensor& dsa_topk_ws(int device)
{
    static std::mutex mtx;
    static std::map<int, at::Tensor> ws;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ws.find(device);
    if (it == ws.end())
    {
        // Candidate indices + counts (int32) followed by candidate scores (half, same count
        // as indices; int-sized allocation keeps one flat buffer)
        at::Tensor t = at::empty
        (
            {(int64_t) TOPK_SPLIT_MAX_R * TOPK_SPLIT_G * TOPK_SPLIT_MAX_KPAD * 3 / 2 + TOPK_SPLIT_MAX_R * TOPK_SPLIT_G},
            at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, device))
        );
        it = ws.emplace(device, std::move(t)).first;
    }
    return it->second;
}

template <bool VEC>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_split_kernel
(
    const half* __restrict__ scores,
    int* __restrict__ ws_idx,            // (R, G, k_pad) candidate indices, span-major
    half* __restrict__ ws_scr,           // (R, G, k_pad) candidate scores (coalesced merge reads)
    int* __restrict__ ws_cnt,            // (R, G) candidate counts
    int T,
    const int s_stride,
    const int k,
    const int k_pad,
    const int* __restrict__ t_ptr,       // device T override (graph modes, t_seq == 0 only)
    const int ws_G,                      // tile mode (ws_slot >= 0): workspace slots per row,
    const int ws_slot,                   //   the slot this launch fills (grid (R, 1)),
    const int idx_offset                 //   and the global index of the tile's first entry
)
{
    if (t_ptr) T = *t_ptr;
    constexpr uint16_t KEY_NEG_INF = 0x03ff;

    const bool tile = ws_slot >= 0;
    const int g = tile ? ws_slot : blockIdx.y;
    const int G = tile ? ws_G : gridDim.y;
    const int span = tile ? T : (((T + G - 1) / G + 7) & ~7);           // 8-aligned spans
    const int beg = tile ? 0 : g * span;
    const int Tl = min(beg + span, T) - beg;                             // local length
    int* row_ws = ws_idx + ((size_t) blockIdx.x * G + g) * k_pad;
    half* row_scr = ws_scr + ((size_t) blockIdx.x * G + g) * k_pad;
    int* cnt_out = ws_cnt + (size_t) blockIdx.x * G + g;

    int t = threadIdx.x;
    if (Tl <= 0)
    {
        if (t == 0) *cnt_out = 0;
        return;
    }

    const half* row_s = scores + (size_t) blockIdx.x * s_stride + beg;   // 16B aligned (beg % 8 == 0)
    int lane = t % 32;
    int warp = t / 32;
    constexpr int NUM_WARPS = TOPK_THREADS / 32;

    __shared__ int hist[256];
    __shared__ int warp_cnt[NUM_WARPS];
    __shared__ int warp_off[NUM_WARPS];
    __shared__ int sh_res[2];
    __shared__ int sh_tile_total;

    const uint4* row_s8 = (const uint4*) row_s;
    const int T8 = VEC ? Tl >> 3 : 0;
    if (t < 256) hist[t] = 0;
    __syncthreads();
    for (int i = t; i < T8; i += TOPK_THREADS)
    {
        uint4 v = row_s8[i];
        uint32_t w[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            uint16_t ka = topk_key_u((uint16_t) w[j]);
            uint16_t kb = topk_key_u((uint16_t) (w[j] >> 16));
            if (ka > KEY_NEG_INF) atomicAdd(&hist[ka >> 8], 1);
            if (kb > KEY_NEG_INF) atomicAdd(&hist[kb >> 8], 1);
        }
    }
    for (int i = T8 * 8 + t; i < Tl; i += TOPK_THREADS)
    {
        uint16_t key = topk_key(row_s[i]);
        if (key > KEY_NEG_INF)
            atomicAdd(&hist[key >> 8], 1);
    }
    __syncthreads();
    if (warp == 0)
        topk_find_bucket(hist, k, lane, sh_res);
    __syncthreads();
    int b1 = sh_res[0];
    int cnt_hi = sh_res[1];

    uint16_t thresh;
    int n_eq_take;
    if (b1 < 0)
    {
        thresh = KEY_NEG_INF;
        n_eq_take = 0;
    }
    else
    {
        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int i = t; i < T8; i += TOPK_THREADS)
        {
            uint4 v = row_s8[i];
            uint32_t w[4] = { v.x, v.y, v.z, v.w };
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                uint16_t ka = topk_key_u((uint16_t) w[j]);
                uint16_t kb = topk_key_u((uint16_t) (w[j] >> 16));
                if (ka > KEY_NEG_INF && (ka >> 8) == b1) atomicAdd(&hist[ka & 0xff], 1);
                if (kb > KEY_NEG_INF && (kb >> 8) == b1) atomicAdd(&hist[kb & 0xff], 1);
            }
        }
        for (int i = T8 * 8 + t; i < Tl; i += TOPK_THREADS)
        {
            uint16_t key = topk_key(row_s[i]);
            if (key > KEY_NEG_INF && (key >> 8) == b1)
                atomicAdd(&hist[key & 0xff], 1);
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k - cnt_hi, lane, sh_res);
        __syncthreads();
        int b0 = max(sh_res[0], 0);
        thresh = (uint16_t) ((b1 << 8) | b0);
        n_eq_take = (k - cnt_hi) - sh_res[1];
    }

    const int n_slots = (Tl + 7) >> 3;
    const int n_tiles = (n_slots + TOPK_THREADS - 1) / TOPK_THREADS;
    int base = 0;

    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        const int eq_start = base;
        const int cap_end = pass == 0 ? INT_MAX : eq_start + n_eq_take;
        for (int t0 = 0; t0 < n_tiles && base < cap_end; ++t0)
        {
            int i8 = t0 * TOPK_THREADS + t;
            unsigned m = 0;
            if (VEC && i8 < T8)
            {
                uint4 v = row_s8[i8];
                uint32_t w[4] = { v.x, v.y, v.z, v.w };
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    uint16_t ka = topk_key_u((uint16_t) w[j]);
                    uint16_t kb = topk_key_u((uint16_t) (w[j] >> 16));
                    bool pa = pass == 0 ? (ka > thresh) : (ka == thresh);
                    bool pb = pass == 0 ? (kb > thresh) : (kb == thresh);
                    if (pa && ka > KEY_NEG_INF) m |= 1u << (2 * j);
                    if (pb && kb > KEY_NEG_INF) m |= 1u << (2 * j + 1);
                }
            }
            else if (i8 >= T8 && i8 < n_slots)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    int idx = i8 * 8 + j;
                    if (idx >= Tl) break;
                    uint16_t kx = topk_key(row_s[idx]);
                    bool px = pass == 0 ? (kx > thresh) : (kx == thresh);
                    if (px && kx > KEY_NEG_INF) m |= 1u << j;
                }
            }
            int c = __popc(m);
            int incl = c;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1)
            {
                int n = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += n;
            }
            if (lane == 31) warp_cnt[warp] = incl;
            __syncthreads();
            if (warp == 0)
            {
                int wc = lane < NUM_WARPS ? warp_cnt[lane] : 0;
                int wincl = wc;
                #pragma unroll
                for (int o = 1; o < 32; o <<= 1)
                {
                    int n = __shfl_up_sync(0xffffffffu, wincl, o);
                    if (lane >= o) wincl += n;
                }
                if (lane < NUM_WARPS) warp_off[lane] = wincl - wc;
                if (lane == 31) sh_tile_total = wincl;
            }
            __syncthreads();
            int obase = base + warp_off[warp] + (incl - c);
            int ibase = i8 * 8;
            while (m)
            {
                int j = __ffs((int) m) - 1;
                m &= m - 1;
                if (obase < cap_end)
                {
                    row_ws[obase] = idx_offset + beg + ibase + j;   // global index
                    row_scr[obase] = row_s[ibase + j];
                }
                ++obase;
            }
            base += sh_tile_total;
        }
        if (pass == 1)
            base = min(base, cap_end);
    }
    if (t == 0)
        *cnt_out = base;
}

__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_merge_kernel
(
    const int* __restrict__ ws_idx,      // (R, G, k_pad)
    const half* __restrict__ ws_scr,     // (R, G, k_pad)
    const int* __restrict__ ws_cnt,      // (R, G)
    int* __restrict__ out,               // (R, k_pad), -1 padded
    const int G,
    const int k,
    const int k_pad,
    half* __restrict__ out_scr,          // optional (R, k_pad): scores of the emitted indices
    int* __restrict__ out_cnt,           // optional (R,): number emitted (<= k)
    const int out_stride,                // row stride of out / out_scr (a workspace slot view)
    const int out_cnt_stride
)
{
    const int* row_ws = ws_idx + (size_t) blockIdx.x * G * k_pad;
    const half* row_scr = ws_scr + (size_t) blockIdx.x * G * k_pad;
    const int* row_cnt = ws_cnt + (size_t) blockIdx.x * G;
    int* row_o = out + (size_t) blockIdx.x * out_stride;
    half* row_os = out_scr ? out_scr + (size_t) blockIdx.x * out_stride : nullptr;
    int t = threadIdx.x;
    int lane = t % 32;
    int warp = t / 32;
    constexpr int NUM_WARPS = TOPK_THREADS / 32;

    __shared__ int hist[256];
    __shared__ int cnt_sh[TOPK_SPLIT_G];
    __shared__ int warp_cnt[NUM_WARPS];
    __shared__ int warp_off[NUM_WARPS];
    __shared__ int sh_res[2];
    __shared__ int sh_tile_total;

    if (t < G) cnt_sh[t] = row_cnt[t];
    if (t < 256) hist[t] = 0;
    __syncthreads();

    int total = 0;
    for (int g = 0; g < G; ++g) total += cnt_sh[g];

    if (total > k)
    {
        // Pass 1: high-byte histogram over the candidates (all finite by construction)
        for (int g = 0; g < G; ++g)
        {
            int cg = cnt_sh[g];
            for (int i = t; i < cg; i += TOPK_THREADS)
            {
                uint16_t key = topk_key(row_scr[g * k_pad + i]);
                atomicAdd(&hist[key >> 8], 1);
            }
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k, lane, sh_res);
        __syncthreads();
        int b1 = sh_res[0];
        int cnt_hi = sh_res[1];

        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int g = 0; g < G; ++g)
        {
            int cg = cnt_sh[g];
            for (int i = t; i < cg; i += TOPK_THREADS)
            {
                uint16_t key = topk_key(row_scr[g * k_pad + i]);
                if ((key >> 8) == b1) atomicAdd(&hist[key & 0xff], 1);
            }
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, k - cnt_hi, lane, sh_res);
        __syncthreads();
        int b0 = max(sh_res[0], 0);
        uint16_t thresh = (uint16_t) ((b1 << 8) | b0);
        int n_eq_take = (k - cnt_hi) - sh_res[1];

        // Ordered compaction over candidates, span-major = ascending global index. One
        // candidate per thread per tile; tie order matches the legacy kernel exactly
        int base = 0;
        #pragma unroll 1
        for (int pass = 0; pass < 2; ++pass)
        {
            const int eq_start = base;
            const int cap_end = pass == 0 ? INT_MAX : eq_start + n_eq_take;
            for (int g = 0; g < G && base < cap_end; ++g)
            {
                int cg = cnt_sh[g];
                for (int c0 = 0; c0 < cg && base < cap_end; c0 += TOPK_THREADS)
                {
                    int i = c0 + t;
                    int gidx = -1;
                    half gscr = __ushort_as_half(0xfc00);
                    bool sel = false;
                    if (i < cg)
                    {
                        gidx = row_ws[g * k_pad + i];
                        gscr = row_scr[g * k_pad + i];
                        uint16_t key = topk_key(gscr);
                        sel = pass == 0 ? (key > thresh) : (key == thresh);
                    }
                    int c = sel ? 1 : 0;
                    int incl = c;
                    #pragma unroll
                    for (int o = 1; o < 32; o <<= 1)
                    {
                        int n = __shfl_up_sync(0xffffffffu, incl, o);
                        if (lane >= o) incl += n;
                    }
                    if (lane == 31) warp_cnt[warp] = incl;
                    __syncthreads();
                    if (warp == 0)
                    {
                        int wc = lane < NUM_WARPS ? warp_cnt[lane] : 0;
                        int wincl = wc;
                        #pragma unroll
                        for (int o = 1; o < 32; o <<= 1)
                        {
                            int n = __shfl_up_sync(0xffffffffu, wincl, o);
                            if (lane >= o) wincl += n;
                        }
                        if (lane < NUM_WARPS) warp_off[lane] = wincl - wc;
                        if (lane == 31) sh_tile_total = wincl;
                    }
                    __syncthreads();
                    if (sel)
                    {
                        int obase = base + warp_off[warp] + (incl - c);
                        if (obase < cap_end)
                        {
                            row_o[obase] = gidx;
                            if (row_os) row_os[obase] = gscr;
                        }
                    }
                    base += sh_tile_total;
                }
            }
            if (pass == 1)
                base = min(base, cap_end);
        }
        for (int j = base + t; j < k_pad; j += TOPK_THREADS)
            row_o[j] = -1;
        if (out_cnt && t == 0) out_cnt[(size_t) blockIdx.x * out_cnt_stride] = base;
    }
    else
    {
        // Fewer candidates than k: emit them all in order, pad the rest
        int base = 0;
        for (int g = 0; g < G; ++g)
        {
            int cg = cnt_sh[g];
            for (int i = t; i < cg; i += TOPK_THREADS)
            {
                row_o[base + i] = row_ws[g * k_pad + i];
                if (row_os) row_os[base + i] = row_scr[g * k_pad + i];
            }
            base += cg;
        }
        for (int j = base + t; j < k_pad; j += TOPK_THREADS)
            row_o[j] = -1;
        if (out_cnt && t == 0) out_cnt[(size_t) blockIdx.x * out_cnt_stride] = base;
    }
}

void dsa_topk_gr
(
    const at::Tensor& scores,            // (R, T) half, possibly a view with row stride
    at::Tensor& indices,                 // (R, k_pad) int32 out
    int k,
    Graph* graph,
    const c10::optional<at::Tensor>& t_ptr,
    int t_seq
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(indices, kInt);
    TORCH_CHECK(scores.dim() == 2 && indices.dim() == 2, "dsa_topk: 2D tensors expected");
    TORCH_CHECK(scores.stride(1) == 1, "dsa_topk: scores innermost dim must be dense");
    TORCH_CHECK(indices.is_contiguous(), "dsa_topk: indices must be contiguous");
    int R = scores.size(0);
    int T = scores.size(1);
    int k_pad = indices.size(1);
    TORCH_CHECK(indices.size(0) == R && k <= k_pad, "dsa_topk: output shape mismatch");

    const int* t_ptr_ = t_ptr ? (const int*) t_ptr.value().data_ptr() : nullptr;
    bool vec = scores.stride(0) % 8 == 0 && ((uintptr_t) scores.data_ptr()) % 16 == 0;

    // Few-row, long-scan selections are parallelism-starved in the single-block kernel;
    // split each row into G span-local selections and merge. EXL3_DSA_TOPK_SPLIT=0 forces
    // the legacy path (A/B testing)
    static const bool allow_split = [](){ const char* e = getenv("EXL3_DSA_TOPK_SPLIT"); return !(e && *e == '0'); }();
    bool split = allow_split && vec && t_seq == 0 &&
        R <= TOPK_SPLIT_MAX_R && T >= TOPK_SPLIT_MIN_T && k_pad <= TOPK_SPLIT_MAX_KPAD;
    if (split)
    {
        // Enough spans to cut the candidate count well below T, few enough that the merge
        // stays cheap. Computed from the launch-time T: graphs capture at full static width,
        // so replays with a smaller patched T just leave trailing spans empty
        int G = min(TOPK_SPLIT_G, max(2, T / (8 * max(k, 1))));
        at::Tensor& ws = dsa_topk_ws(scores.device().index());
        int* ws_idx = (int*) ws.data_ptr();
        int* ws_cnt = ws_idx + (size_t) TOPK_SPLIT_MAX_R * TOPK_SPLIT_G * TOPK_SPLIT_MAX_KPAD;
        half* ws_scr = (half*) (ws_cnt + (size_t) TOPK_SPLIT_MAX_R * TOPK_SPLIT_G);
        dim3 grid_s(R, G);
        dsa_topk_split_kernel<true><<<grid_s, TOPK_THREADS, 0, stream>>>
        (
            (const half*) scores.data_ptr(), ws_idx, ws_scr, ws_cnt,
            T, (int) scores.stride(0), k, k_pad, t_ptr_, 0, -1, 0
        );
        cuda_check(cudaPeekAtLastError());
        dsa_topk_merge_kernel<<<R, TOPK_THREADS, 0, stream>>>
        (
            ws_idx, ws_scr, ws_cnt,
            (int*) indices.data_ptr(), G, k, k_pad, nullptr, nullptr, k_pad, 1
        );
        cuda_check(cudaPeekAtLastError());
        if (graph && !t_ptr_)
        {
            graph->record_param((void*) dsa_topk_split_kernel<true>, GP_dsa_T, 4, 4);
            graph->record_param((void*) dsa_topk_split_kernel<true>, GP_end, 0);
        }
        return;
    }

    void* kfn = vec ? (void*) dsa_topk_kernel<true> : (void*) dsa_topk_kernel<false>;
    if (vec)
        dsa_topk_kernel<true><<<R, TOPK_THREADS, 0, stream>>>
        (
            (const half*) scores.data_ptr(),
            (int*) indices.data_ptr(),
            T, (int) scores.stride(0), k, k_pad, t_ptr_, t_seq
        );
    else
        dsa_topk_kernel<false><<<R, TOPK_THREADS, 0, stream>>>
        (
            (const half*) scores.data_ptr(),
            (int*) indices.data_ptr(),
            T, (int) scores.stride(0), k, k_pad, t_ptr_, t_seq
        );
    cuda_check(cudaPeekAtLastError());

    // With a device-side T there is nothing to patch per replay
    if (graph && !t_ptr_)
    {
        graph->record_param(kfn, GP_dsa_T, 2, 4);
        graph->record_param(kfn, GP_end, 0);
    }
}

// Tiled selection without a full-width score row: each scored tile contributes its local
// top-k as a candidate slot (dsa_topk_tile), and dsa_topk_merge_tiles reduces the slots of a
// row to the exact top-k under the same total order as the single-pass kernel (score
// descending, index ascending), optionally emitting scores and counts so the result can seed
// slot 0 of the next merge. Workspaces: ws_idx (R, G, k_pad) i32, ws_scr (R, G, k_pad) half,
// ws_cnt (R, G) i32

void dsa_topk_tile
(
    const at::Tensor& scores,            // (R, T_tile) half, dense rows
    at::Tensor ws_idx,
    at::Tensor ws_scr,
    at::Tensor ws_cnt,
    int64_t slot,
    int64_t k,
    int64_t idx_offset
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(ws_idx, kInt);
    TORCH_CHECK_DTYPE(ws_scr, kHalf);
    TORCH_CHECK_DTYPE(ws_cnt, kInt);
    TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1, "dsa_topk_tile: scores must be (R, T) with dense rows");
    TORCH_CHECK(ws_idx.dim() == 3 && ws_idx.is_contiguous() && ws_scr.sizes() == ws_idx.sizes() &&
                ws_scr.is_contiguous() && ws_cnt.dim() == 2 && ws_cnt.is_contiguous(), "dsa_topk_tile: bad workspace");
    int R = scores.size(0);
    int T = scores.size(1);
    int G = ws_idx.size(1);
    int k_pad = ws_idx.size(2);
    TORCH_CHECK(ws_idx.size(0) == R && ws_cnt.size(0) == R && ws_cnt.size(1) == G, "dsa_topk_tile: workspace rows/slots mismatch");
    TORCH_CHECK(slot >= 0 && slot < G && k <= k_pad && G <= TOPK_SPLIT_G, "dsa_topk_tile: bad slot / k");
    bool vec = scores.stride(0) % 8 == 0 && ((uintptr_t) scores.data_ptr()) % 16 == 0;
    dim3 grid(R, 1);
    if (vec)
        dsa_topk_split_kernel<true><<<grid, TOPK_THREADS, 0, stream>>>
        (
            (const half*) scores.data_ptr(), (int*) ws_idx.data_ptr(), (half*) ws_scr.data_ptr(),
            (int*) ws_cnt.data_ptr(), T, (int) scores.stride(0), (int) k, k_pad, nullptr,
            G, (int) slot, (int) idx_offset
        );
    else
        dsa_topk_split_kernel<false><<<grid, TOPK_THREADS, 0, stream>>>
        (
            (const half*) scores.data_ptr(), (int*) ws_idx.data_ptr(), (half*) ws_scr.data_ptr(),
            (int*) ws_cnt.data_ptr(), T, (int) scores.stride(0), (int) k, k_pad, nullptr,
            G, (int) slot, (int) idx_offset
        );
    cuda_check(cudaPeekAtLastError());
}

void dsa_topk_merge_tiles
(
    const at::Tensor& ws_idx,
    const at::Tensor& ws_scr,
    const at::Tensor& ws_cnt,
    at::Tensor out_idx,                  // (R, k_pad) i32, -1 padded
    const c10::optional<at::Tensor>& out_scr,   // (R, k_pad) half
    const c10::optional<at::Tensor>& out_cnt,   // (R,) i32
    int64_t k
)
{
    const at::cuda::OptionalCUDAGuard device_guard(ws_idx.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(ws_idx, kInt);
    TORCH_CHECK_DTYPE(ws_scr, kHalf);
    TORCH_CHECK_DTYPE(ws_cnt, kInt);
    TORCH_CHECK_DTYPE(out_idx, kInt);
    TORCH_CHECK(ws_idx.dim() == 3 && ws_idx.is_contiguous() && ws_scr.is_contiguous() && ws_cnt.is_contiguous() &&
                out_idx.dim() == 2 && out_idx.stride(1) == 1, "dsa_topk_merge_tiles: bad tensors");
    int R = ws_idx.size(0);
    int G = ws_idx.size(1);
    int k_pad = ws_idx.size(2);
    TORCH_CHECK(out_idx.size(0) == R && out_idx.size(1) == k_pad && k <= k_pad && G <= TOPK_SPLIT_G,
                "dsa_topk_merge_tiles: shape mismatch");
    half* os = nullptr;
    int* oc = nullptr;
    int oc_stride = 1;
    if (out_scr)
    {
        TORCH_CHECK_DTYPE(out_scr.value(), kHalf);
        TORCH_CHECK(out_scr.value().sizes() == out_idx.sizes() && out_scr.value().strides() == out_idx.strides(),
                    "dsa_topk_merge_tiles: out_scr must match out_idx");
        os = (half*) out_scr.value().data_ptr();
    }
    if (out_cnt)
    {
        TORCH_CHECK_DTYPE(out_cnt.value(), kInt);
        TORCH_CHECK(out_cnt.value().dim() == 1 && out_cnt.value().size(0) == R, "dsa_topk_merge_tiles: bad out_cnt");
        oc = (int*) out_cnt.value().data_ptr();
        oc_stride = (int) out_cnt.value().stride(0);
    }
    dsa_topk_merge_kernel<<<R, TOPK_THREADS, 0, stream>>>
    (
        (const int*) ws_idx.data_ptr(), (const half*) ws_scr.data_ptr(), (const int*) ws_cnt.data_ptr(),
        (int*) out_idx.data_ptr(), G, (int) k, k_pad, os, oc, (int) out_idx.stride(0), oc_stride
    );
    cuda_check(cudaPeekAtLastError());
}

// Per-job sequence state for the batched sparse-DSA graph stages (BC_MLAttention
// MULTIROW slots): row 0 = q_pos0 (past length), row 1 = past + q_len, the scoring scan
// width / causal clamp and the per-job top-k bound. Derived on device from the (graph-
// patched) cache_seqlens tensor, so batched sparse replay needs no host-side state writes

__global__ void dsa_seq_state_kernel
(
    const int* __restrict__ seqlens,     // (bsz,) i32, past lengths (pre-append)
    int* __restrict__ arr,               // (2, arr_stride) i32 out
    const int bsz,
    const int q_len,
    const int arr_stride
)
{
    int b = threadIdx.x;
    if (b >= bsz) return;
    int sl = seqlens[b];
    arr[b] = sl;
    arr[arr_stride + b] = sl + q_len;
}

void dsa_seq_state_gr
(
    const at::Tensor& cache_seqlens,     // (bsz,) i32, device
    at::Tensor& arr,                     // (2, arr_stride) i32, device static
    int bsz,
    int q_len,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(cache_seqlens.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(cache_seqlens, kInt);
    TORCH_CHECK_DTYPE(arr, kInt);
    TORCH_CHECK(arr.dim() == 2 && arr.size(0) == 2 && arr.size(1) >= bsz && arr.is_contiguous(),
                "dsa_seq_state: bad state array");

    dsa_seq_state_kernel<<<1, 32, 0, stream>>>
    (
        (const int*) cache_seqlens.data_ptr(),
        (int*) arr.data_ptr(),
        bsz, q_len, (int) arr.size(1)
    );
    cuda_check(cudaPeekAtLastError());
    if (graph)
    {
        graph->record_param((void*) dsa_seq_state_kernel, GP_attn_seqlens, 0);
        graph->record_param((void*) dsa_seq_state_kernel, GP_end, 0);
    }
}

void dsa_topk
(
    const at::Tensor& scores,
    at::Tensor indices,
    int64_t k,
    const c10::optional<at::Tensor>& t_ptr,
    int64_t t_seq
)
{
    dsa_topk_gr(scores, indices, (int) k, nullptr, t_ptr, (int) t_seq);
}
