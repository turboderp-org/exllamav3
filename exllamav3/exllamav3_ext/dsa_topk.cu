#include <cuda_fp16.h>
#include <climits>
#include "dsa_topk.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"

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
