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

Radix select on the ordered key of the score (monotonic bit flip; 16 bits for fp16 scores,
32 for fp32): one 256-bucket histogram pass per key byte pins down the exact k-th threshold
value v*, then ordered stream compaction emits all entries with score > v* followed by
enough == v* ties.
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

// Score types: fp16 (16-bit ordered key, two radix passes) and fp32 (32-bit key, four). The
// selection logic is shared; only the key width, the pass count and the vector load differ
template <typename S> struct TopkScore;

template <> struct TopkScore<half>
{
    typedef uint16_t key_t;
    static constexpr int PASSES = 2;
    static constexpr key_t KEY_NEG_INF = 0x03ff;       // key of -inf (0xfc00)
    static __device__ __forceinline__ key_t key(const half h) { return topk_key_u(__half_as_ushort(h)); }
    // Keys of the 8 scores of slot i8 (one 16-byte load; rows are 16-byte aligned)
    static __device__ __forceinline__ void load8(const half* row, const int i8, key_t* ks)
    {
        uint4 v = ((const uint4*) row)[i8];
        uint32_t w[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            ks[2 * j] = topk_key_u((uint16_t) w[j]);
            ks[2 * j + 1] = topk_key_u((uint16_t) (w[j] >> 16));
        }
    }
};

template <> struct TopkScore<float>
{
    typedef uint32_t key_t;
    static constexpr int PASSES = 4;
    static constexpr key_t KEY_NEG_INF = 0x007fffffu;  // key of -inf (0xff800000)
    static __device__ __forceinline__ key_t key_u(const uint32_t u)
    {
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    static __device__ __forceinline__ key_t key(const float f) { return key_u(__float_as_uint(f)); }
    static __device__ __forceinline__ void load8(const float* row, const int i8, key_t* ks)
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            uint4 v = ((const uint4*) row)[2 * i8 + h];
            ks[4 * h] = key_u(v.x);
            ks[4 * h + 1] = key_u(v.y);
            ks[4 * h + 2] = key_u(v.z);
            ks[4 * h + 3] = key_u(v.w);
        }
    }
};

// Radix select over one score row of T entries, called by every thread of the block: one
// 256-bucket histogram pass per key byte, most significant first, each restricted to the
// entries matching the bytes found so far. Leaves thresh = key of the k-th largest finite
// score and n_eq_take = how many entries tied at it are still needed after everything above
// (fewer than k finite entries: thresh just above -inf and no tie cap, so all are taken).
// Vectorized: 16-byte loads, 8 keys per slot (rows are 16-byte aligned: the host checks
// s_stride % 8 == 0), scalar tail
template <bool VEC, typename S>
__device__ __forceinline__ void topk_radix_select
(
    const S* row_s,
    const int T,
    const int k,
    int* hist,
    int* sh_res,
    typename TopkScore<S>::key_t& thresh,
    int& n_eq_take
)
{
    typedef TopkScore<S> TS;
    typedef typename TS::key_t key_t;
    constexpr key_t KEY_NEG_INF = TS::KEY_NEG_INF;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int T8 = VEC ? T >> 3 : 0;

    key_t prefix = 0;
    int need = k;
    thresh = KEY_NEG_INF;
    n_eq_take = 0;

    #pragma unroll 1
    for (int p = 0; p < TS::PASSES; ++p)
    {
        const int shift = 8 * (TS::PASSES - 1 - p);
        if (t < 256) hist[t] = 0;
        __syncthreads();
        for (int i = t; i < T8; i += TOPK_THREADS)
        {
            key_t ks[8];
            TS::load8(row_s, i, ks);
            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                const key_t hi = p == 0 ? (key_t) 0 : (key_t) (ks[j] >> (shift + 8));
                if (ks[j] > KEY_NEG_INF && hi == prefix) atomicAdd(&hist[(ks[j] >> shift) & 0xff], 1);
            }
        }
        for (int i = T8 * 8 + t; i < T; i += TOPK_THREADS)
        {
            const key_t key = TS::key(row_s[i]);
            const key_t hi = p == 0 ? (key_t) 0 : (key_t) (key >> (shift + 8));
            if (key > KEY_NEG_INF && hi == prefix) atomicAdd(&hist[(key >> shift) & 0xff], 1);
        }
        __syncthreads();
        if (warp == 0)
            topk_find_bucket(hist, need, lane, sh_res);
        __syncthreads();
        const int b = sh_res[0];
        if (p == 0 && b < 0)
            return;                             // fewer than k finite entries: take them all
        need -= sh_res[1];                      // entries strictly above this byte's bucket
        prefix = (key_t) ((prefix << 8) | (key_t) max(b, 0));   // in-band search cannot miss; clamp defensively
    }
    thresh = prefix;
    n_eq_take = need;                           // ties at the threshold still needed
}

// Selection mask of the 8 entries of slot i8 for one compaction pass (bit j = entry
// i8 * 8 + j): pass 0 keys > thresh, pass 1 keys == thresh. Slots at or above T8 take the
// scalar strip: the straddling slot's tail (VEC) or every slot (!VEC), with the same
// 8-element ownership, so the emission order is unchanged
template <bool VEC, typename S>
__device__ __forceinline__ unsigned topk_slot_mask
(
    const S* row_s,
    const int i8,
    const int T,
    const int pass,
    const typename TopkScore<S>::key_t thresh
)
{
    typedef TopkScore<S> TS;
    typedef typename TS::key_t key_t;
    constexpr key_t KEY_NEG_INF = TS::KEY_NEG_INF;
    const int T8 = VEC ? T >> 3 : 0;
    const int n_slots = (T + 7) >> 3;
    unsigned m = 0;
    if (VEC && i8 < T8)
    {
        key_t ks[8];
        TS::load8(row_s, i8, ks);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            bool px = pass == 0 ? (ks[j] > thresh) : (ks[j] == thresh);
            if (px && ks[j] > KEY_NEG_INF) m |= 1u << j;
        }
    }
    else if (i8 >= T8 && i8 < n_slots)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int idx = i8 * 8 + j;
            if (idx >= T) break;
            key_t kx = TS::key(row_s[idx]);
            bool px = pass == 0 ? (kx > thresh) : (kx == thresh);
            if (px && kx > KEY_NEG_INF) m |= 1u << j;
        }
    }
    return m;
}

template <bool VEC, typename S>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_kernel
(
    const S* __restrict__ scores,        // (R, s_stride), -inf past each row's causal bound
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

    const S* row_s = scores + (size_t) blockIdx.x * s_stride;
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

    typename TopkScore<S>::key_t thresh;
    int n_eq_take;
    topk_radix_select<VEC, S>(row_s, T, k, hist, sh_res, thresh, n_eq_take);

    // Ordered compaction, 8 elements per thread per tile (16-byte loads).
    // Position of element i is (tile, warp, lane, intra-lane bit). Monotonic in i, so
    // the ascending-index emission order is preserved. Per tile: one intra-warp shfl scan
    // + a warp-0 scan over warp totals; the running output base stays in a REGISTER
    // (block-uniform: everyone adds the same tile total read between the two barriers),
    // so a tile costs 2 barriers for 8 * TOPK_THREADS elements instead of 3 per
    // TOPK_THREADS. The thread whose slot straddles T picks up the scalar tail
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
            unsigned m = topk_slot_mask<VEC, S>(row_s, i8, T, pass, thresh);
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
    // Leaked: a static map would free its tensors from a static destructor at exit, after the CUDA
    // allocator has gone (a cudaMallocAsync free at teardown terminates the process)
    static auto* ws = new std::map<int, at::Tensor>();
    std::lock_guard<std::mutex> lock(mtx);
    auto it = ws->find(device);
    if (it == ws->end())
    {
        // Candidate indices + counts (int32) followed by candidate scores (sized for fp32,
        // the wider of the two score types, one per index; one flat int-sized buffer)
        at::Tensor t = at::empty
        (
            {(int64_t) TOPK_SPLIT_MAX_R * TOPK_SPLIT_G * TOPK_SPLIT_MAX_KPAD * 2 + TOPK_SPLIT_MAX_R * TOPK_SPLIT_G},
            at::TensorOptions().dtype(at::kInt).device(at::Device(at::kCUDA, device))
        );
        it = ws->emplace(device, std::move(t)).first;
    }
    return it->second;
}

// Allocate the split workspace ahead of a graph capture over `scores_static` (the full-width score
// static a captured top-k scans). The eager pass that precedes a capture may scan only the live
// prefix of that static, below TOPK_SPLIT_MIN_T, and so never take the split path itself; without
// this the replay-side split kernel would allocate its workspace under capture (cudaMalloc is not
// permitted while a stream is capturing)
void dsa_topk_prepare(const at::Tensor& scores_static, int k_pad)
{
    static const bool allow_split = [](){ const char* e = getenv("EXL3_DSA_TOPK_SPLIT"); return !(e && *e == '0'); }();
    if (!allow_split || scores_static.dim() != 2) return;
    if (scores_static.size(1) < TOPK_SPLIT_MIN_T || k_pad > TOPK_SPLIT_MAX_KPAD) return;
    (void) dsa_topk_ws(scores_static.device().index());
}

template <bool VEC, typename S>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_split_kernel
(
    const S* __restrict__ scores,
    int* __restrict__ ws_idx,            // (R, G, k_pad) candidate indices, span-major
    S* __restrict__ ws_scr,              // (R, G, k_pad) candidate scores (coalesced merge reads)
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

    const bool tile = ws_slot >= 0;
    const int g = tile ? ws_slot : blockIdx.y;
    const int G = tile ? ws_G : gridDim.y;
    const int span = tile ? T : (((T + G - 1) / G + 7) & ~7);           // 8-aligned spans
    const int beg = tile ? 0 : g * span;
    const int Tl = min(beg + span, T) - beg;                             // local length
    int* row_ws = ws_idx + ((size_t) blockIdx.x * G + g) * k_pad;
    S* row_scr = ws_scr + ((size_t) blockIdx.x * G + g) * k_pad;
    int* cnt_out = ws_cnt + (size_t) blockIdx.x * G + g;

    int t = threadIdx.x;
    if (Tl <= 0)
    {
        if (t == 0) *cnt_out = 0;
        return;
    }

    const S* row_s = scores + (size_t) blockIdx.x * s_stride + beg;      // 16B aligned (beg % 8 == 0)
    int lane = t % 32;
    int warp = t / 32;
    constexpr int NUM_WARPS = TOPK_THREADS / 32;

    __shared__ int hist[256];
    __shared__ int warp_cnt[NUM_WARPS];
    __shared__ int warp_off[NUM_WARPS];
    __shared__ int sh_res[2];
    __shared__ int sh_tile_total;

    typename TopkScore<S>::key_t thresh;
    int n_eq_take;
    topk_radix_select<VEC, S>(row_s, Tl, k, hist, sh_res, thresh, n_eq_take);

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
            unsigned m = topk_slot_mask<VEC, S>(row_s, i8, Tl, pass, thresh);
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

template <typename S>
__global__ __launch_bounds__(TOPK_THREADS)
void dsa_topk_merge_kernel
(
    const int* __restrict__ ws_idx,      // (R, G, k_pad)
    const S* __restrict__ ws_scr,        // (R, G, k_pad)
    const int* __restrict__ ws_cnt,      // (R, G)
    int* __restrict__ out,               // (R, k_pad), -1 padded
    const int G,
    const int k,
    const int k_pad,
    S* __restrict__ out_scr,             // optional (R, k_pad): scores of the emitted indices
    int* __restrict__ out_cnt,           // optional (R,): number emitted (<= k)
    const int out_stride,                // row stride of out / out_scr (a workspace slot view)
    const int out_cnt_stride
)
{
    typedef TopkScore<S> TS;
    typedef typename TS::key_t key_t;

    const int* row_ws = ws_idx + (size_t) blockIdx.x * G * k_pad;
    const S* row_scr = ws_scr + (size_t) blockIdx.x * G * k_pad;
    const int* row_cnt = ws_cnt + (size_t) blockIdx.x * G;
    int* row_o = out + (size_t) blockIdx.x * out_stride;
    S* row_os = out_scr ? out_scr + (size_t) blockIdx.x * out_stride : nullptr;
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
    __syncthreads();

    int total = 0;
    for (int g = 0; g < G; ++g) total += cnt_sh[g];

    if (total >= k)
    {
        // Radix select over the candidates (all finite by construction), one histogram
        // pass per key byte as in topk_radix_select
        key_t prefix = 0;
        int need = k;
        #pragma unroll 1
        for (int p = 0; p < TS::PASSES; ++p)
        {
            const int shift = 8 * (TS::PASSES - 1 - p);
            if (t < 256) hist[t] = 0;
            __syncthreads();
            for (int g = 0; g < G; ++g)
            {
                int cg = cnt_sh[g];
                for (int i = t; i < cg; i += TOPK_THREADS)
                {
                    const key_t key = TS::key(row_scr[g * k_pad + i]);
                    const key_t hi = p == 0 ? (key_t) 0 : (key_t) (key >> (shift + 8));
                    if (hi == prefix) atomicAdd(&hist[(key >> shift) & 0xff], 1);
                }
            }
            __syncthreads();
            if (warp == 0)
                topk_find_bucket(hist, need, lane, sh_res);
            __syncthreads();
            need -= sh_res[1];
            prefix = (key_t) ((prefix << 8) | (key_t) max(sh_res[0], 0));
        }
        const key_t thresh = prefix;
        const int n_eq_take = need;

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
                    S gscr = S();
                    bool sel = false;
                    if (i < cg)
                    {
                        gidx = row_ws[g * k_pad + i];
                        gscr = row_scr[g * k_pad + i];
                        key_t key = TS::key(gscr);
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

template <typename S>
static void dsa_topk_launch
(
    const at::Tensor& scores,
    at::Tensor& indices,
    int k,
    Graph* graph,
    const int* t_ptr_,
    int t_seq,
    cudaStream_t stream
)
{
    int R = scores.size(0);
    int T = scores.size(1);
    int k_pad = indices.size(1);
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
        S* ws_scr = (S*) (ws_cnt + (size_t) TOPK_SPLIT_MAX_R * TOPK_SPLIT_G);
        dim3 grid_s(R, G);
        dsa_topk_split_kernel<true, S><<<grid_s, TOPK_THREADS, 0, stream>>>
        (
            (const S*) scores.data_ptr(), ws_idx, ws_scr, ws_cnt,
            T, (int) scores.stride(0), k, k_pad, t_ptr_, 0, -1, 0
        );
        cuda_check(cudaPeekAtLastError());
        dsa_topk_merge_kernel<S><<<R, TOPK_THREADS, 0, stream>>>
        (
            ws_idx, ws_scr, ws_cnt,
            (int*) indices.data_ptr(), G, k, k_pad, nullptr, nullptr, k_pad, 1
        );
        cuda_check(cudaPeekAtLastError());
        if (graph && !t_ptr_)
        {
            graph->record_param((void*) dsa_topk_split_kernel<true, S>, GP_dsa_T, 4, 4);
            graph->record_param((void*) dsa_topk_split_kernel<true, S>, GP_end, 0);
        }
        return;
    }

    void* kfn = vec ? (void*) dsa_topk_kernel<true, S> : (void*) dsa_topk_kernel<false, S>;
    if (vec)
        dsa_topk_kernel<true, S><<<R, TOPK_THREADS, 0, stream>>>
        (
            (const S*) scores.data_ptr(),
            (int*) indices.data_ptr(),
            T, (int) scores.stride(0), k, k_pad, t_ptr_, t_seq
        );
    else
        dsa_topk_kernel<false, S><<<R, TOPK_THREADS, 0, stream>>>
        (
            (const S*) scores.data_ptr(),
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

void dsa_topk_gr
(
    const at::Tensor& scores,            // (R, T) half or float, possibly a view with row stride
    at::Tensor& indices,                 // (R, k_pad) int32 out
    int k,
    Graph* graph,
    const c10::optional<at::Tensor>& t_ptr,
    int t_seq
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    const bool f32 = scores.scalar_type() == at::kFloat;
    TORCH_CHECK(f32 || scores.scalar_type() == at::kHalf, "dsa_topk: scores must be float16 or float32");
    TORCH_CHECK_DTYPE(indices, kInt);
    TORCH_CHECK(scores.dim() == 2 && indices.dim() == 2, "dsa_topk: 2D tensors expected");
    TORCH_CHECK(scores.stride(1) == 1, "dsa_topk: scores innermost dim must be dense");
    TORCH_CHECK(indices.is_contiguous(), "dsa_topk: indices must be contiguous");
    int R = scores.size(0);
    int T = scores.size(1);
    int k_pad = indices.size(1);
    TORCH_CHECK(indices.size(0) == R && k <= k_pad, "dsa_topk: output shape mismatch");
    TORCH_CHECK(k >= 0, "dsa_topk: negative k");
    // k == 0 selects nothing: every output slot is padding. Otherwise selecting from an empty
    // row is undefined (with a device-side T, the static width bounds every replay's T)
    TORCH_CHECK(k == 0 || T > 0, "dsa_topk: top-k over an empty score row");
    if (!R || !k_pad) return;
    if (!k)
    {
        cuda_check(cudaMemsetAsync(indices.data_ptr(), 0xff, (size_t) R * k_pad * sizeof(int), stream));
        return;
    }

    const int* t_ptr_ = t_ptr ? (const int*) t_ptr.value().data_ptr() : nullptr;
    if (f32)
        dsa_topk_launch<float>(scores, indices, k, graph, t_ptr_, t_seq, stream);
    else
        dsa_topk_launch<half>(scores, indices, k, graph, t_ptr_, t_seq, stream);
}

// Tiled selection without a full-width score row: each scored tile contributes its local
// top-k as a candidate slot (dsa_topk_tile), and dsa_topk_merge_tiles reduces the slots of a
// row to the exact top-k under the same total order as the single-pass kernel (score
// descending, index ascending), optionally emitting scores and counts so the result can seed
// slot 0 of the next merge. Workspaces: ws_idx (R, G, k_pad) i32, ws_scr (R, G, k_pad) in the
// score dtype (half or float), ws_cnt (R, G) i32

template <typename S>
static void dsa_topk_tile_launch
(
    const at::Tensor& scores, at::Tensor& ws_idx, at::Tensor& ws_scr, at::Tensor& ws_cnt,
    int slot, int k, int idx_offset, cudaStream_t stream
)
{
    int R = scores.size(0);
    int T = scores.size(1);
    int G = ws_idx.size(1);
    int k_pad = ws_idx.size(2);
    bool vec = scores.stride(0) % 8 == 0 && ((uintptr_t) scores.data_ptr()) % 16 == 0;
    dim3 grid(R, 1);
    if (vec)
        dsa_topk_split_kernel<true, S><<<grid, TOPK_THREADS, 0, stream>>>
        (
            (const S*) scores.data_ptr(), (int*) ws_idx.data_ptr(), (S*) ws_scr.data_ptr(),
            (int*) ws_cnt.data_ptr(), T, (int) scores.stride(0), k, k_pad, nullptr,
            G, slot, idx_offset
        );
    else
        dsa_topk_split_kernel<false, S><<<grid, TOPK_THREADS, 0, stream>>>
        (
            (const S*) scores.data_ptr(), (int*) ws_idx.data_ptr(), (S*) ws_scr.data_ptr(),
            (int*) ws_cnt.data_ptr(), T, (int) scores.stride(0), k, k_pad, nullptr,
            G, slot, idx_offset
        );
    cuda_check(cudaPeekAtLastError());
}

void dsa_topk_tile
(
    const at::Tensor& scores,            // (R, T_tile) half or float, dense rows
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
    const bool f32 = scores.scalar_type() == at::kFloat;
    TORCH_CHECK(f32 || scores.scalar_type() == at::kHalf, "dsa_topk_tile: scores must be float16 or float32");
    TORCH_CHECK_DTYPE(ws_idx, kInt);
    TORCH_CHECK(ws_scr.scalar_type() == scores.scalar_type(), "dsa_topk_tile: ws_scr must have the dtype of the scores");
    TORCH_CHECK_DTYPE(ws_cnt, kInt);
    TORCH_CHECK(scores.dim() == 2 && scores.stride(1) == 1, "dsa_topk_tile: scores must be (R, T) with dense rows");
    TORCH_CHECK(ws_idx.dim() == 3 && ws_idx.is_contiguous() && ws_scr.sizes() == ws_idx.sizes() &&
                ws_scr.is_contiguous() && ws_cnt.dim() == 2 && ws_cnt.is_contiguous(), "dsa_topk_tile: bad workspace");
    int R = scores.size(0);
    int G = ws_idx.size(1);
    int k_pad = ws_idx.size(2);
    TORCH_CHECK(ws_idx.size(0) == R && ws_cnt.size(0) == R && ws_cnt.size(1) == G, "dsa_topk_tile: workspace rows/slots mismatch");
    TORCH_CHECK(slot >= 0 && slot < G && k <= k_pad && G <= TOPK_SPLIT_G, "dsa_topk_tile: bad slot / k");
    TORCH_CHECK(k >= 0, "dsa_topk_tile: negative k");
    if (!R) return;
    // k == 0 (or an empty tile) contributes no candidates
    if (!k)
    {
        cuda_check(cudaMemset2DAsync((int*) ws_cnt.data_ptr() + slot, ws_cnt.stride(0) * sizeof(int), 0,
                                     sizeof(int), R, stream));
        return;
    }
    if (f32)
        dsa_topk_tile_launch<float>(scores, ws_idx, ws_scr, ws_cnt, (int) slot, (int) k, (int) idx_offset, stream);
    else
        dsa_topk_tile_launch<half>(scores, ws_idx, ws_scr, ws_cnt, (int) slot, (int) k, (int) idx_offset, stream);
}

void dsa_topk_merge_tiles
(
    const at::Tensor& ws_idx,
    const at::Tensor& ws_scr,
    const at::Tensor& ws_cnt,
    at::Tensor out_idx,                  // (R, k_pad) i32, -1 padded
    const c10::optional<at::Tensor>& out_scr,   // (R, k_pad) in the dtype of ws_scr
    const c10::optional<at::Tensor>& out_cnt,   // (R,) i32
    int64_t k
)
{
    const at::cuda::OptionalCUDAGuard device_guard(ws_idx.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const bool f32 = ws_scr.scalar_type() == at::kFloat;
    TORCH_CHECK_DTYPE(ws_idx, kInt);
    TORCH_CHECK(f32 || ws_scr.scalar_type() == at::kHalf, "dsa_topk_merge_tiles: ws_scr must be float16 or float32");
    TORCH_CHECK_DTYPE(ws_cnt, kInt);
    TORCH_CHECK_DTYPE(out_idx, kInt);
    TORCH_CHECK(ws_idx.dim() == 3 && ws_idx.is_contiguous() && ws_scr.is_contiguous() && ws_cnt.is_contiguous() &&
                out_idx.dim() == 2 && out_idx.stride(1) == 1, "dsa_topk_merge_tiles: bad tensors");
    int R = ws_idx.size(0);
    int G = ws_idx.size(1);
    int k_pad = ws_idx.size(2);
    TORCH_CHECK(out_idx.size(0) == R && out_idx.size(1) == k_pad && k <= k_pad && G <= TOPK_SPLIT_G,
                "dsa_topk_merge_tiles: shape mismatch");
    TORCH_CHECK(ws_scr.sizes() == ws_idx.sizes() && ws_cnt.dim() == 2 && ws_cnt.size(0) == R && ws_cnt.size(1) == G,
                "dsa_topk_merge_tiles: workspace shape mismatch");
    TORCH_CHECK(k >= 0, "dsa_topk_merge_tiles: negative k");
    TORCH_CHECK(k == 0 || G > 0, "dsa_topk_merge_tiles: top-k over an empty set of tiles");
    void* os = nullptr;
    int* oc = nullptr;
    int oc_stride = 1;
    if (out_scr)
    {
        TORCH_CHECK(out_scr.value().scalar_type() == ws_scr.scalar_type(),
                    "dsa_topk_merge_tiles: out_scr must have the dtype of ws_scr");
        TORCH_CHECK(out_scr.value().sizes() == out_idx.sizes() && out_scr.value().strides() == out_idx.strides(),
                    "dsa_topk_merge_tiles: out_scr must match out_idx");
        os = out_scr.value().data_ptr();
    }
    if (out_cnt)
    {
        TORCH_CHECK_DTYPE(out_cnt.value(), kInt);
        TORCH_CHECK(out_cnt.value().dim() == 1 && out_cnt.value().size(0) == R, "dsa_topk_merge_tiles: bad out_cnt");
        oc = (int*) out_cnt.value().data_ptr();
        oc_stride = (int) out_cnt.value().stride(0);
    }
    if (!R) return;
    // k == 0 selects nothing: every output slot is padding, no scores, zero counts
    if (!k)
    {
        if (k_pad)
            cuda_check(cudaMemset2DAsync(out_idx.data_ptr(), out_idx.stride(0) * sizeof(int), 0xff,
                                         k_pad * sizeof(int), R, stream));
        if (oc)
            cuda_check(cudaMemset2DAsync(oc, oc_stride * sizeof(int), 0, sizeof(int), R, stream));
        return;
    }
    if (f32)
        dsa_topk_merge_kernel<float><<<R, TOPK_THREADS, 0, stream>>>
        (
            (const int*) ws_idx.data_ptr(), (const float*) ws_scr.data_ptr(), (const int*) ws_cnt.data_ptr(),
            (int*) out_idx.data_ptr(), G, (int) k, k_pad, (float*) os, oc, (int) out_idx.stride(0), oc_stride
        );
    else
        dsa_topk_merge_kernel<half><<<R, TOPK_THREADS, 0, stream>>>
        (
            (const int*) ws_idx.data_ptr(), (const half*) ws_scr.data_ptr(), (const int*) ws_cnt.data_ptr(),
            (int*) out_idx.data_ptr(), G, (int) k, k_pad, (half*) os, oc, (int) out_idx.stride(0), oc_stride
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
