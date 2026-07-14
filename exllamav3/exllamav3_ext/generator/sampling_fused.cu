#include <cuda_fp16.h>
#include "sampling_fused.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include <limits>
#include <curand_kernel.h>
#include "../reduction.cuh"

// Fused terminal sampling step for collapsed sampler stacks. Computes, in one pass over the
// logits, the equivalent of the temperature/min-P/Gumbel pipelines without materializing a
// softmax:
//
//   p_i >= min_p * p_max                    <=>  l_i >= M + ln(min_p),  M = max(l)
//   softmax(l)^(1/T) renormalized           <=>  softmax(l / T)
//   categorical sample from softmax(l / T)  ==   argmax(l_i / T + g_i), g_i ~ Gumbel(0, 1)
//
// Modes:
//   0: greedy argmax (temperature/min-P have no effect on the winner)
//   1: Gumbel sample of softmax(l / T)
//   2: Gumbel sample of softmax(l / T) restricted to { i : l_i >= M + minp_log }, where
//      minp_log = ln(min_p) if min-P precedes temperature in the stack, T * ln(min_p) if it
//      follows it
//
// The Gumbel noise stream matches gumbel_noise_f16/f32/log (Philox keyed on (random, index)),
// so a collapsed stack picks the same token as the uncollapsed reference for the same rand_u32,
// up to float rounding at exact ties. The optional additive logit mask (half, one row or bsz
// rows) replaces the logits + logit_mask add on the Python side; indices at or beyond the mask
// width count as masked out, matching the -inf padding semantics of the eager path.

namespace
{

constexpr float FS_NEG_INF = -std::numeric_limits<float>::infinity();

#define FS_THREADS 256

inline __device__ float fs_gumbel(float x)
{
    return -__logf(fmaxf(-__logf(fmaxf(x, 1e-20)), 1e-20));
}

inline __device__ float fs_to_float(half x) { return __half2float(x); }
inline __device__ float fs_to_float(float x) { return x; }

// Argmax reduction preferring the lowest index among equal values, so the block/grid split
// cannot change which of two tied logits wins (and greedy mode matches a sequential scan)

inline __device__ void fs_amax_first(ValIdx& a, float val, int idx)
{
    if (val > a.val || (val == a.val && idx < a.idx))
    {
        a.val = val;
        a.idx = idx;
    }
}

inline __device__ float fs_block_reduce_max(float v)
{
    __shared__ float shared[FS_THREADS / 32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_max_f(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    if (warp_id == 0)
    {
        v = lane_id < FS_THREADS / 32 ? shared[lane_id] : FS_NEG_INF;
        v = warp_reduce_max_f(v);
    }
    return v;
}

inline __device__ ValIdx fs_warp_reduce_argmax_first(ValIdx v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        fs_amax_first(v, other_val, other_idx);
    }
    return v;
}

inline __device__ ValIdx fs_block_reduce_argmax_first(ValIdx v)
{
    __shared__ ValIdx shared[FS_THREADS / 32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = fs_warp_reduce_argmax_first(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    if (warp_id == 0)
    {
        v = lane_id < FS_THREADS / 32 ? shared[lane_id] : ValIdx{ FS_NEG_INF, INT_MAX };
        v = fs_warp_reduce_argmax_first(v);
    }
    return v;
}

// Pass 1 (mode 2 only): per-block partial max of the (masked) logits

template <typename T>
__global__ __launch_bounds__(FS_THREADS)
void fs_partial_max_kernel
(
    const T* __restrict__ logits,               // (bsz, dim)
    const half* __restrict__ mask,              // (1 or bsz, >= size) or nullptr
    float* __restrict__ ws_max,                 // (bsz, num_blocks)
    const int dim,
    const int size,
    const int num_blocks,
    const bool mask_per_row
)
{
    int row = blockIdx.y;
    const T* row_logits = logits + (size_t) row * dim;
    const half* row_mask = mask ? mask + (mask_per_row ? (size_t) row * dim : 0) : nullptr;

    float m = FS_NEG_INF;
    for (int i = blockIdx.x * FS_THREADS + threadIdx.x; i < size; i += num_blocks * FS_THREADS)
    {
        float x = fs_to_float(row_logits[i]);
        if (row_mask) x += fs_to_float(row_mask[i]);
        m = fmaxf(m, x);
    }
    m = fs_block_reduce_max(m);
    if (threadIdx.x == 0)
        ws_max[row * num_blocks + blockIdx.x] = m;
}

// Mode 3 (top-K/top-P): histogram select. The logits are bucketed by depth below the max in
// filter-temperature units over FS_HIST_RANGE nats (everything deeper has relative probability
// < e^-FS_HIST_RANGE and is clamped into the last bucket). Mass is accumulated as 2^40
// fixed-point exp((l - M) / T_filter) in u64, so atomic accumulation order cannot change the
// result (deterministic) and the summation error (<= 0.5 per element against a >= 2^40 total)
// is far below fp32 cumsum error.
//
// All the truncations keep top segments of the same ordering, so the composite kept set is
// { i : bin(l_i) <= bound } for one lexicographic (bucket, sub-bucket) bound. The select
// runs in rounds: the coarse histogram
// locates the boundary bucket for the binding criterion, a refinement pass re-histograms just
// that bucket into 1024 sub-buckets (FS_HIST_RANGE / 1024^2 = 1/32768 nat, finer than fp16
// ULP at practical logit magnitudes, so exact for fp16 logits), and the next select round
// resolves the exact threshold. Top-K refines first, because top-P's target is p times the
// mass of the top-K-truncated set; the exact target then locates the top-P crossing either
// inside the already-refined bucket or in an earlier bucket, which gets its own refinement
// round. The crossing sub-bucket is dropped, matching the eager rule that the token crossing
// the cumulative sum is dropped; tokens tied at the exact cutoff are all kept.

#define FS_HIST_RANGE 32.0f
#define FS_MASS_SCALE 1099511627776.0f  // 2^40
#define FS_NB FUSED_SAMPLER_HIST_BUCKETS

// Refinement status in the per-row control block
#define FS_ST_DONE 0        // ctrl->keep_b/keep_s are final
#define FS_ST_REFINE_K 1    // refinement of the top-K boundary bucket pending
#define FS_ST_REFINE_P 2    // refinement of the top-P crossing bucket pending

struct FSCtrl
{
    float m;                // global (masked) max logit
    unsigned int status;
    unsigned int ref_bucket;// coarse bucket to refine / being consumed
    unsigned int k_rem;     // top-K: kept slots remaining within the refined bucket
    unsigned int keep_b;    // kept-set bound: lexicographically (bucket, sub-bucket) <=
    unsigned int keep_s;    //   (keep_b, keep_s), evaluated with the exact binning expression
    unsigned long long target;  // top-P: fixed-point mass target over the kept set
};

inline __device__ unsigned long long* fs_hist_mass(uint8_t* hist, int row, int level)
{
    return (unsigned long long*) (hist + (size_t) row * FUSED_SAMPLER_HIST_STRIDE + level * FS_NB * 12);
}
inline __device__ unsigned int* fs_hist_count(uint8_t* hist, int row, int level)
{
    return (unsigned int*) (fs_hist_mass(hist, row, level) + FS_NB);
}
inline __device__ FSCtrl* fs_hist_ctrl(uint8_t* hist, int row)
{
    return (FSCtrl*) (hist + (size_t) row * FUSED_SAMPLER_HIST_STRIDE + 2 * FS_NB * 12);
}
// Element bin under the shared binning expression; every kernel (histogram, refinement,
// argmax) uses this same fp32 arithmetic, so the partition is bit-identical everywhere and no
// threshold is ever reconstructed from bucket-edge arithmetic
inline __device__ void fs_bin(float x, float m, float bucket_scale, int& b, int& s)
{
    float db = (m - x) * bucket_scale;
    b = min((int) db, FS_NB - 1);
    s = min(max((int) ((db - (float) b) * (float) FS_NB), 0), FS_NB - 1);
}

// Coarse pass: bin all (masked, min-P-filtered) elements by depth below the max

template <typename T>
__global__ __launch_bounds__(FS_THREADS)
void fs_histogram_kernel
(
    const T* __restrict__ logits,               // (bsz, dim)
    const half* __restrict__ mask,              // (1 or bsz, >= size) or nullptr
    const float* __restrict__ ws_max,           // (bsz, num_blocks) partial maxes
    uint8_t* __restrict__ hist,                 // FUSED_SAMPLER_HIST_STRIDE per row, zeroed
    const int dim,
    const int size,
    const int num_blocks,
    const bool mask_per_row,
    const int filters,
    const float minp_log,
    const float inv_temp_filter
)
{
    int row = blockIdx.y;
    const T* row_logits = logits + (size_t) row * dim;
    const half* row_mask = mask ? mask + (mask_per_row ? (size_t) row * dim : 0) : nullptr;

    float m = FS_NEG_INF;
    for (int j = threadIdx.x; j < num_blocks; j += FS_THREADS)
        m = fmaxf(m, ws_max[row * num_blocks + j]);
    m = fs_block_reduce_max(m);
    __shared__ float sh_m;
    if (threadIdx.x == 0)
    {
        sh_m = m;
        if (blockIdx.x == 0)
            fs_hist_ctrl(hist, row)->m = m;
    }

    __shared__ unsigned int sh_cnt[FS_NB];
    __shared__ unsigned long long sh_mass[FS_NB];
    for (int b = threadIdx.x; b < FS_NB; b += FS_THREADS)
    {
        sh_cnt[b] = 0;
        sh_mass[b] = 0;
    }
    __syncthreads();

    m = sh_m;
    float minp_threshold = (filters & FUSED_SAMPLER_F_MINP) ? m + minp_log : FS_NEG_INF;
    float bucket_scale = (float) FS_NB / FS_HIST_RANGE * inv_temp_filter;

    for (int i = blockIdx.x * FS_THREADS + threadIdx.x; i < size; i += num_blocks * FS_THREADS)
    {
        float x = fs_to_float(row_logits[i]);
        if (row_mask) x += fs_to_float(row_mask[i]);
        if (x < minp_threshold || x == FS_NEG_INF) continue;
        int b, s;
        fs_bin(x, m, bucket_scale, b, s);
        float w = __expf((x - m) * inv_temp_filter);
        atomicAdd(&sh_cnt[b], 1u);
        atomicAdd(&sh_mass[b], (unsigned long long) (w * FS_MASS_SCALE + 0.5f));
    }
    __syncthreads();

    unsigned long long* g_mass = fs_hist_mass(hist, row, 0);
    unsigned int* g_cnt = fs_hist_count(hist, row, 0);
    for (int b = threadIdx.x; b < FS_NB; b += FS_THREADS)
    {
        if (sh_cnt[b]) atomicAdd(&g_cnt[b], sh_cnt[b]);
        if (sh_mass[b]) atomicAdd(&g_mass[b], sh_mass[b]);
    }
}

// Refinement pass: re-bin the elements of ctrl->ref_bucket into 1024 sub-buckets. Elements are
// assigned to coarse buckets with the exact same expression as the coarse pass, so the
// partition is identical.

template <typename T>
__global__ __launch_bounds__(FS_THREADS)
void fs_histogram_refine_kernel
(
    const T* __restrict__ logits,
    const half* __restrict__ mask,
    uint8_t* __restrict__ hist,
    const int dim,
    const int size,
    const int num_blocks,
    const bool mask_per_row,
    const int filters,
    const float minp_log,
    const float inv_temp_filter
)
{
    int row = blockIdx.y;
    FSCtrl* ctrl = fs_hist_ctrl(hist, row);
    if (ctrl->status == FS_ST_DONE) return;
    int ref_bucket = ctrl->ref_bucket;
    float m = ctrl->m;

    const T* row_logits = logits + (size_t) row * dim;
    const half* row_mask = mask ? mask + (mask_per_row ? (size_t) row * dim : 0) : nullptr;

    __shared__ unsigned int sh_cnt[FS_NB];
    __shared__ unsigned long long sh_mass[FS_NB];
    for (int b = threadIdx.x; b < FS_NB; b += FS_THREADS)
    {
        sh_cnt[b] = 0;
        sh_mass[b] = 0;
    }
    __syncthreads();

    float minp_threshold = (filters & FUSED_SAMPLER_F_MINP) ? m + minp_log : FS_NEG_INF;
    float bucket_scale = (float) FS_NB / FS_HIST_RANGE * inv_temp_filter;

    for (int i = blockIdx.x * FS_THREADS + threadIdx.x; i < size; i += num_blocks * FS_THREADS)
    {
        float x = fs_to_float(row_logits[i]);
        if (row_mask) x += fs_to_float(row_mask[i]);
        if (x < minp_threshold || x == FS_NEG_INF) continue;
        int b, s;
        fs_bin(x, m, bucket_scale, b, s);
        if (b != ref_bucket) continue;
        float w = __expf((x - m) * inv_temp_filter);
        atomicAdd(&sh_cnt[s], 1u);
        atomicAdd(&sh_mass[s], (unsigned long long) (w * FS_MASS_SCALE + 0.5f));
    }
    __syncthreads();

    unsigned long long* g_mass = fs_hist_mass(hist, row, 1);
    unsigned int* g_cnt = fs_hist_count(hist, row, 1);
    for (int b = threadIdx.x; b < FS_NB; b += FS_THREADS)
    {
        if (sh_cnt[b]) atomicAdd(&g_cnt[b], sh_cnt[b]);
        if (sh_mass[b]) atomicAdd(&g_mass[b], sh_mass[b]);
    }
}

// Select rounds. One block per row; phase 0 consumes the coarse histogram, phase 1 consumes a
// refinement (and may schedule the top-P refinement round). Inclusive scans use a sequential
// per-thread slice plus a Hillis-Steele scan of the per-thread totals.

inline __device__ void fs_scan_1024
(
    const unsigned int* __restrict__ g_cnt,
    const unsigned long long* __restrict__ g_mass,
    unsigned int* sh_c,                         // FS_THREADS
    unsigned long long* sh_ms,                  // FS_THREADS
    unsigned int* cum_c,                        // FS_NB
    unsigned long long* cum_m                   // FS_NB
)
{
    constexpr int PER_THREAD = FS_NB / FS_THREADS;
    int t = threadIdx.x;

    unsigned int lc[PER_THREAD];
    unsigned long long lm[PER_THREAD];
    unsigned int csum = 0;
    unsigned long long msum = 0;
    #pragma unroll
    for (int j = 0; j < PER_THREAD; ++j)
    {
        csum += g_cnt[t * PER_THREAD + j];
        msum += g_mass[t * PER_THREAD + j];
        lc[j] = csum;
        lm[j] = msum;
    }
    sh_c[t] = csum;
    sh_ms[t] = msum;
    __syncthreads();
    for (int offset = 1; offset < FS_THREADS; offset <<= 1)
    {
        unsigned int c = t >= offset ? sh_c[t - offset] : 0;
        unsigned long long ms = t >= offset ? sh_ms[t - offset] : 0;
        __syncthreads();
        sh_c[t] += c;
        sh_ms[t] += ms;
        __syncthreads();
    }
    unsigned int c_prefix = t > 0 ? sh_c[t - 1] : 0;
    unsigned long long m_prefix = t > 0 ? sh_ms[t - 1] : 0;
    #pragma unroll
    for (int j = 0; j < PER_THREAD; ++j)
    {
        cum_c[t * PER_THREAD + j] = lc[j] + c_prefix;
        cum_m[t * PER_THREAD + j] = lm[j] + m_prefix;
    }
    __syncthreads();
}

// First cumulative-mass position strictly exceeding target (FS_NB if none): the crossing
// bucket under the drop-the-crossing rule
inline __device__ int fs_find_crossing
(
    const unsigned long long* cum_m,
    unsigned long long target,
    int limit,
    int* sh_res
)
{
    constexpr int PER_THREAD = FS_NB / FS_THREADS;
    int t = threadIdx.x;
    if (t == 0) *sh_res = limit;
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < PER_THREAD; ++j)
    {
        int b = t * PER_THREAD + j;
        if (b >= limit) continue;
        unsigned long long prev = b > 0 ? cum_m[b - 1] : 0;
        if (prev <= target && cum_m[b] > target)
            atomicMin(sh_res, b);
    }
    __syncthreads();
    return *sh_res;
}

__global__ __launch_bounds__(FS_THREADS)
void fs_select_kernel
(
    uint8_t* __restrict__ hist,
    const int phase,                            // 0: coarse, 1: consume refinement
    const int filters,
    const int top_k,
    const float top_p,
    const float minp_log,
    const float inv_temp_filter
)
{
    constexpr int PER_THREAD = FS_NB / FS_THREADS;
    int row = blockIdx.x;
    int t = threadIdx.x;
    FSCtrl* ctrl = fs_hist_ctrl(hist, row);
    if (phase == 1 && ctrl->status == FS_ST_DONE) return;

    __shared__ unsigned int sh_c[FS_THREADS];
    __shared__ unsigned long long sh_ms[FS_THREADS];
    __shared__ unsigned int cum_c[FS_NB];
    __shared__ unsigned long long cum_m[FS_NB];
    __shared__ int sh_res;

    // Tighten the kept-set bound (thread 0 only)
    auto bound_min = [&](unsigned int b, unsigned int s)
    {
        if (b < ctrl->keep_b || (b == ctrl->keep_b && s < ctrl->keep_s))
        {
            ctrl->keep_b = b;
            ctrl->keep_s = s;
        }
    };
    // Bound from a dropped crossing bin (b, s): keep everything lexicographically before it
    auto bound_drop_crossing = [&](int b, int s)
    {
        if (s > 0)      bound_min(b, s - 1);
        else if (b > 0) bound_min(b - 1, FS_NB - 1);
        else            bound_min(0, 0);  // the top token is always kept (with its exact ties)
    };

    if (phase == 0)
    {
        if (t == 0)
        {
            ctrl->keep_b = FS_NB - 1;
            ctrl->keep_s = FS_NB - 1;
        }
        fs_scan_1024(fs_hist_count(hist, row, 0), fs_hist_mass(hist, row, 0), sh_c, sh_ms, cum_c, cum_m);
        unsigned long long total = cum_m[FS_NB - 1];

        // Top-K boundary bucket: binding only if the count crosses k before the last bucket
        // (the last bucket also holds the clamped tail, so a crossing there keeps everything)
        int b_k = FS_NB;
        if (filters & FUSED_SAMPLER_F_TOPK)
        {
            if (t == 0) sh_res = FS_NB;
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < PER_THREAD; ++j)
            {
                int b = t * PER_THREAD + j;
                unsigned int prev = b > 0 ? cum_c[b - 1] : 0;
                if (prev < (unsigned int) top_k && cum_c[b] >= (unsigned int) top_k)
                    atomicMin(&sh_res, b);
            }
            __syncthreads();
            b_k = sh_res;
        }
        if (b_k < FS_NB - 1)
        {
            // Refine the top-K bucket first; top-P needs the exact truncated mass
            if (t == 0)
            {
                ctrl->ref_bucket = b_k;
                ctrl->k_rem = (unsigned int) top_k - (b_k > 0 ? cum_c[b_k - 1] : 0);
                ctrl->status = FS_ST_REFINE_K;
            }
            return;
        }

        // Top-K absent or non-binding: the top-P target over the full mass is already exact
        if (filters & FUSED_SAMPLER_F_TOPP)
        {
            unsigned long long target = (unsigned long long) ((double) top_p * (double) total);
            int b_p = fs_find_crossing(cum_m, target, FS_NB, &sh_res);
            if (b_p >= FS_NB - 1)
            {
                // Crossing in the clamped tail bucket (or none): drop at most that bucket
                if (t == 0)
                {
                    if (b_p == FS_NB - 1) bound_drop_crossing(FS_NB - 1, 0);
                    ctrl->status = FS_ST_DONE;
                }
                return;
            }
            if (t == 0)
            {
                ctrl->ref_bucket = b_p;
                ctrl->target = target - (b_p > 0 ? cum_m[b_p - 1] : 0);
                ctrl->status = FS_ST_REFINE_P;
            }
            return;
        }

        if (t == 0) ctrl->status = FS_ST_DONE;
        return;
    }

    // Phase 1: consume a refinement round
    int status = ctrl->status;
    int rb = ctrl->ref_bucket;

    if (status == FS_ST_REFINE_K)
    {
        fs_scan_1024(fs_hist_count(hist, row, 1), fs_hist_mass(hist, row, 1), sh_c, sh_ms, cum_c, cum_m);

        // Exact top-K cutoff: first sub-bucket where the count within the bucket reaches
        // k_rem; ties at the cutoff value share its sub-bucket and are all kept
        unsigned int k_rem = ctrl->k_rem;
        if (t == 0) sh_res = FS_NB - 1;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < PER_THREAD; ++j)
        {
            int b = t * PER_THREAD + j;
            unsigned int prev = b > 0 ? cum_c[b - 1] : 0;
            if (prev < k_rem && cum_c[b] >= k_rem)
                atomicMin(&sh_res, b);
        }
        __syncthreads();
        int s_k = sh_res;
        unsigned long long kept_mass_rb = cum_m[s_k];
        if (t == 0) bound_min(rb, s_k);

        if (!(filters & FUSED_SAMPLER_F_TOPP))
        {
            if (t == 0) ctrl->status = FS_ST_DONE;
            return;
        }

        // Exact top-P target over the top-K-truncated set
        fs_scan_1024(fs_hist_count(hist, row, 0), fs_hist_mass(hist, row, 0), sh_c, sh_ms, cum_c, cum_m);
        unsigned long long below = rb > 0 ? cum_m[rb - 1] : 0;
        unsigned long long z1 = below + kept_mass_rb;
        unsigned long long target = (unsigned long long) ((double) top_p * (double) z1);

        int b_p = fs_find_crossing(cum_m, target, rb + 1, &sh_res);
        if (b_p >= rb)
        {
            // Crossing lands in the already-refined bucket (target < z1 <= cum_m[rb]), so
            // resolve it against the level-1 histogram. Serial search by thread 0: 1024
            // sub-buckets, only runs in this corner case
            if (t == 0)
            {
                unsigned long long sub_target = target - below;
                const unsigned long long* g_mass1 = fs_hist_mass(hist, row, 1);
                unsigned long long prev = 0, acc = 0;
                int s_p = FS_NB;
                for (int s = 0; s < FS_NB; ++s)
                {
                    acc += g_mass1[s];
                    if (prev <= sub_target && acc > sub_target) { s_p = s; break; }
                    prev = acc;
                }
                if (s_p < FS_NB) bound_drop_crossing(rb, s_p);
                ctrl->status = FS_ST_DONE;
            }
            return;
        }

        // Crossing in an earlier bucket: schedule the top-P refinement round
        if (t == 0)
        {
            ctrl->ref_bucket = b_p;
            ctrl->target = target - (b_p > 0 ? cum_m[b_p - 1] : 0);
            ctrl->status = FS_ST_REFINE_P;
        }
        // Zero the sub-histogram for the next refinement pass
        __syncthreads();
        unsigned long long* g_mass1 = fs_hist_mass(hist, row, 1);
        unsigned int* g_cnt1 = fs_hist_count(hist, row, 1);
        for (int b = t; b < FS_NB; b += FS_THREADS)
        {
            g_mass1[b] = 0;
            g_cnt1[b] = 0;
        }
        return;
    }

    if (status == FS_ST_REFINE_P)
    {
        fs_scan_1024(fs_hist_count(hist, row, 1), fs_hist_mass(hist, row, 1), sh_c, sh_ms, cum_c, cum_m);
        unsigned long long sub_target = ctrl->target;
        int s_p = fs_find_crossing(cum_m, sub_target, FS_NB, &sh_res);
        if (t == 0)
        {
            if (s_p < FS_NB) bound_drop_crossing(rb, s_p);
            ctrl->status = FS_ST_DONE;
        }
        return;
    }
}

// Pass 2: per-block partial argmax of the sampling objective. Mode 2 first recovers the global
// max from the pass 1 partials (each block redundantly reduces num_blocks values, which is far
// cheaper than a separate kernel launch); mode 3 reads the threshold selected from the
// histogram instead

template <typename T, int MODE>
__global__ __launch_bounds__(FS_THREADS)
void fs_partial_argmax_kernel
(
    const T* __restrict__ logits,               // (bsz, dim)
    const half* __restrict__ mask,              // (1 or bsz, >= size) or nullptr
    const float* __restrict__ ws_max,           // (bsz, num_blocks), mode 2 only
    ValIdx* __restrict__ ws_vi,                 // (bsz, num_blocks)
    const uint8_t* __restrict__ hist,           // mode 3 only (selected kept-set bound)
    const int dim,
    const int size,
    const int num_blocks,
    const bool mask_per_row,
    const float inv_temp,
    const float minp_log,
    const int filters,
    const float inv_temp_filter,
    const uint32_t random
)
{
    int row = blockIdx.y;
    const T* row_logits = logits + (size_t) row * dim;
    const half* row_mask = mask ? mask + (mask_per_row ? (size_t) row * dim : 0) : nullptr;

    float threshold = FS_NEG_INF;
    if constexpr (MODE == 2)
    {
        float m = FS_NEG_INF;
        for (int j = threadIdx.x; j < num_blocks; j += FS_THREADS)
            m = fmaxf(m, ws_max[row * num_blocks + j]);
        m = fs_block_reduce_max(m);
        __shared__ float sh_m;
        if (threadIdx.x == 0) sh_m = m;
        __syncthreads();
        threshold = sh_m + minp_log;
    }

    // Mode 3 keeps elements by their (bucket, sub-bucket) bin, evaluated with the exact
    // expression the histogram passes used, so the kept set matches the select bit-exactly
    float fs3_m = 0.0f, fs3_scale = 0.0f;
    unsigned int fs3_kb = 0, fs3_ks = 0;
    if constexpr (MODE == 3)
    {
        const FSCtrl* ctrl = fs_hist_ctrl((uint8_t*) hist, row);
        fs3_m = ctrl->m;
        fs3_kb = ctrl->keep_b;
        fs3_ks = ctrl->keep_s;
        fs3_scale = (float) FS_NB / FS_HIST_RANGE * inv_temp_filter;
        if (filters & FUSED_SAMPLER_F_MINP)
            threshold = fs3_m + minp_log;
    }

    ValIdx best = { FS_NEG_INF, INT_MAX };
    for (int i = blockIdx.x * FS_THREADS + threadIdx.x; i < size; i += num_blocks * FS_THREADS)
    {
        float x = fs_to_float(row_logits[i]);
        if (row_mask) x += fs_to_float(row_mask[i]);

        float v;
        if constexpr (MODE == 0)
        {
            v = x;
        }
        else
        {
            if (MODE >= 2 && x < threshold) continue;
            if (MODE == 3 && x != fs3_m)
            {
                if (x == FS_NEG_INF) continue;
                int b, s;
                fs_bin(x, fs3_m, fs3_scale, b, s);
                if ((unsigned int) b > fs3_kb ||
                    ((unsigned int) b == fs3_kb && (unsigned int) s > fs3_ks)) continue;
            }
            // Same noise stream as the gumbel_noise_* kernels: Philox keyed on (random, flat
            // index over the batch), independent per element, so skipped elements don't shift
            // anyone else's noise
            curandStatePhilox4_32_10_t rng;
            curand_init(random, (uint64_t) row * dim + i, 0, &rng);
            v = x * inv_temp + fs_gumbel(curand_uniform(&rng));
        }
        fs_amax_first(best, v, i);
    }

    best = fs_block_reduce_argmax_first(best);
    if (threadIdx.x == 0)
        ws_vi[row * num_blocks + blockIdx.x] = best;
}

// Pass 3: reduce the per-block winners to one token index per row

__global__ __launch_bounds__(FUSED_SAMPLER_MAX_BLOCKS)
void fs_finalize_kernel
(
    const ValIdx* __restrict__ ws_vi,           // (bsz, num_blocks)
    uint64_t* __restrict__ out,                 // (bsz, 1)
    const int num_blocks
)
{
    int row = blockIdx.x;

    ValIdx v = threadIdx.x < num_blocks ?
        ws_vi[row * num_blocks + threadIdx.x] :
        ValIdx{ FS_NEG_INF, INT_MAX };

    __shared__ ValIdx shared[FUSED_SAMPLER_MAX_BLOCKS / 32];
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    v = fs_warp_reduce_argmax_first(v);
    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();
    if (warp_id == 0)
    {
        v = lane_id < FUSED_SAMPLER_MAX_BLOCKS / 32 ? shared[lane_id] : ValIdx{ FS_NEG_INF, INT_MAX };
        v = fs_warp_reduce_argmax_first(v);
        if (lane_id == 0)
            out[row] = v.idx == INT_MAX ? 0 : (uint64_t) v.idx;
    }
}

}  // namespace

void fused_sampler
(
    const at::Tensor& logits,                   // (bsz, dim), half or float
    const c10::optional<at::Tensor>& logit_mask,// (1 or bsz, >= size), half, optional
    at::Tensor& out,                            // (bsz, 1), long
    at::Tensor& workspace,                      // >= bsz * FUSED_SAMPLER_MAX_BLOCKS * 3, float
    int size,                                   // effective vocab bound, <= dim
    float inv_temp,                             // 1 / sampling temperature (modes 1/2/3)
    float minp_log,                             // additive log-space min-P threshold (modes 2/3)
    uint32_t random,
    int mode,                                   // 0: greedy, 1: gumbel, 2: gumbel + min-P,
                                                // 3: gumbel + top-K/top-P (+ min-P)
    int filters,                                // mode 3: FUSED_SAMPLER_F_* bits
    int top_k,                                  // mode 3
    float top_p,                                // mode 3
    float inv_temp_filter,                      // mode 3: 1 / T for filters computed on the
                                                // tempered distribution, 1 otherwise
    const c10::optional<at::Tensor>& histogram  // mode 3: >= bsz * FUSED_SAMPLER_HIST_STRIDE, u8
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(logits, 2);
    TORCH_CHECK(logits.is_contiguous(), "fused_sampler: logits must be contiguous");
    TORCH_CHECK_DTYPE(out, kLong);
    TORCH_CHECK_DTYPE(workspace, kFloat);
    TORCH_CHECK(mode >= 0 && mode <= 3, "fused_sampler: invalid mode");

    int bsz = logits.size(0);
    int dim = logits.size(1);
    TORCH_CHECK(size > 0 && size <= dim, "fused_sampler: invalid size bound");
    TORCH_CHECK(out.numel() == bsz, "fused_sampler: out must have bsz elements");

    const half* mask_ptr = nullptr;
    bool mask_per_row = false;
    if (logit_mask.has_value())
    {
        const at::Tensor& mask = logit_mask.value();
        TORCH_CHECK_DTYPE(mask, kHalf);
        TORCH_CHECK_DIM(mask, 2);
        TORCH_CHECK(mask.is_contiguous(), "fused_sampler: mask must be contiguous");
        TORCH_CHECK(mask.size(1) >= size, "fused_sampler: mask narrower than size bound");
        TORCH_CHECK(mask.size(0) == 1 || mask.size(0) == bsz, "fused_sampler: bad mask batch dim");
        // Row stride for the per-row case is the logits width; require identical widths there
        TORCH_CHECK(mask.size(0) == 1 || mask.size(1) == dim, "fused_sampler: per-row mask must match logits width");
        mask_ptr = (const half*) mask.data_ptr();
        mask_per_row = mask.size(0) == bsz;
    }

    int num_blocks = MIN(CEIL_DIVIDE(size, FS_THREADS * 4), FUSED_SAMPLER_MAX_BLOCKS);
    TORCH_CHECK(workspace.numel() >= bsz * FUSED_SAMPLER_MAX_BLOCKS * 3, "fused_sampler: workspace too small");
    // ws_vi is offset by the max-blocks stride (always even) so the 8-byte ValIdx slots stay aligned
    float* ws_max = (float*) workspace.data_ptr();
    ValIdx* ws_vi = (ValIdx*) (ws_max + bsz * FUSED_SAMPLER_MAX_BLOCKS);

    bool is_half = logits.dtype() == at::kHalf;
    bool is_float = logits.dtype() == at::kFloat;
    TORCH_CHECK(is_half || is_float, "fused_sampler: logits must be half or float");

    uint8_t* hist_ptr = nullptr;
    if (mode == 3)
    {
        TORCH_CHECK(histogram.has_value(), "fused_sampler: mode 3 requires a histogram buffer");
        const at::Tensor& hist = histogram.value();
        TORCH_CHECK_DTYPE(hist, kByte);
        TORCH_CHECK(hist.numel() >= bsz * FUSED_SAMPLER_HIST_STRIDE, "fused_sampler: histogram too small");
        hist_ptr = (uint8_t*) hist.data_ptr();
        TORCH_CHECK(((uintptr_t) hist_ptr) % 8 == 0, "fused_sampler: histogram must be 8-byte aligned");
        cuda_check(cudaMemsetAsync(hist_ptr, 0, (size_t) bsz * FUSED_SAMPLER_HIST_STRIDE, stream));
    }

    dim3 grid(num_blocks, bsz);

    #define FS_ARGS(T) \
        (const T*) logits.data_ptr(), mask_ptr, ws_max, ws_vi, hist_ptr, dim, size, num_blocks, \
        mask_per_row, inv_temp, minp_log, filters, inv_temp_filter, random

    if (mode >= 2)
    {
        if (is_half)
            fs_partial_max_kernel<half><<<grid, FS_THREADS, 0, stream>>>
                ((const half*) logits.data_ptr(), mask_ptr, ws_max, dim, size, num_blocks, mask_per_row);
        else
            fs_partial_max_kernel<float><<<grid, FS_THREADS, 0, stream>>>
                ((const float*) logits.data_ptr(), mask_ptr, ws_max, dim, size, num_blocks, mask_per_row);
    }

    if (mode == 3)
    {
        #define FS_HIST(T) \
            fs_histogram_kernel<T><<<grid, FS_THREADS, 0, stream>>> \
                ((const T*) logits.data_ptr(), mask_ptr, ws_max, hist_ptr, dim, size, \
                 num_blocks, mask_per_row, filters, minp_log, inv_temp_filter);
        #define FS_REFINE(T) \
            fs_histogram_refine_kernel<T><<<grid, FS_THREADS, 0, stream>>> \
                ((const T*) logits.data_ptr(), mask_ptr, hist_ptr, dim, size, \
                 num_blocks, mask_per_row, filters, minp_log, inv_temp_filter);
        #define FS_SELECT(PHASE) \
            fs_select_kernel<<<bsz, FS_THREADS, 0, stream>>> \
                (hist_ptr, PHASE, filters, top_k, top_p, minp_log, inv_temp_filter);

        if (is_half) FS_HIST(half) else FS_HIST(float)
        FS_SELECT(0)

        // One refinement round resolves a single binding criterion exactly; with both top-K
        // and top-P active, top-P may need a second round after the exact top-K mass is
        // known. Rounds no-op (status check) once the threshold is final.
        int rounds = ((filters & FUSED_SAMPLER_F_TOPK) && (filters & FUSED_SAMPLER_F_TOPP)) ? 2 : 1;
        for (int r = 0; r < rounds; ++r)
        {
            if (is_half) FS_REFINE(half) else FS_REFINE(float)
            FS_SELECT(1)
        }
        #undef FS_HIST
        #undef FS_REFINE
        #undef FS_SELECT
    }

    #define FS_LAUNCH(T, MODE) \
        fs_partial_argmax_kernel<T, MODE><<<grid, FS_THREADS, 0, stream>>>(FS_ARGS(T));
    if (is_half)
    {
        if      (mode == 0) FS_LAUNCH(half, 0)
        else if (mode == 1) FS_LAUNCH(half, 1)
        else if (mode == 2) FS_LAUNCH(half, 2)
        else                FS_LAUNCH(half, 3)
    }
    else
    {
        if      (mode == 0) FS_LAUNCH(float, 0)
        else if (mode == 1) FS_LAUNCH(float, 1)
        else if (mode == 2) FS_LAUNCH(float, 2)
        else                FS_LAUNCH(float, 3)
    }
    #undef FS_LAUNCH
    #undef FS_ARGS

    fs_finalize_kernel<<<bsz, FUSED_SAMPLER_MAX_BLOCKS, 0, stream>>>
    (
        ws_vi,
        (uint64_t*) out.data_ptr(),
        num_blocks
    );

    cuda_check(cudaPeekAtLastError());
}
