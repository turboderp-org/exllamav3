#include <cuda_fp16.h>
#include "routing.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "reduction.cuh"
#include "hgemm.cuh"

#define MAX_NUM_EXPERTS 512
#define MAX_K 16

using bfloat16 = __nv_bfloat16;

__device__ __forceinline__
float sigmoid_stable_hf(float xf)
{
    float ez = __expf(-fabsf(xf));
    float base = ez / (1.0f + ez);
    return (xf >= 0.0f) ? 1.0f - base : base;
}

// Score activations for the nogroup top-k kernels. Both are strictly increasing, so sorting
// by the raw logit when there is no selection bias remains valid for either
#define ROUTING_ACT_SIGMOID 0
#define ROUTING_ACT_SQRTSP 1

template <int ACT>
__device__ __forceinline__
float routing_act(float xf)
{
    if constexpr (ACT == ROUTING_ACT_SQRTSP)
    {
        // sqrt(softplus(x)), matching torch F.softplus(beta = 1, threshold = 20)
        float sp = xf > 20.0f ? xf : log1pf(__expf(xf));
        return sqrtf(sp);
    }
    else
        return sigmoid_stable_hf(xf);
}


__device__ __forceinline__
void warp_reduce_best_f32(float& key, float& payload, int& idx)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
        // Monotonic unsigned encoding of the float key, hardware max-reduce, then fetch the
        // winner's values from the lowest tied lane
        unsigned int ku = __float_as_uint(key);
        ku = (ku & 0x80000000u) ? ~ku : (ku | 0x80000000u);
        unsigned int m = __reduce_max_sync(0xffffffffu, ku);
        int src = __ffs(__ballot_sync(0xffffffffu, ku == m)) - 1;
        key = __shfl_sync(0xffffffffu, key, src);
        payload = __shfl_sync(0xffffffffu, payload, src);
        idx = __shfl_sync(0xffffffffu, idx, src);
    #else
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_key = __shfl_down_sync(0xffffffffu, key, offset);
            float other_payload = __shfl_down_sync(0xffffffffu, payload, offset);
            int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
            if (other_key > key)
            {
                key = other_key;
                payload = other_payload;
                idx = other_idx;
            }
        }

        key = __shfl_sync(0xffffffffu, key, 0);
        payload = __shfl_sync(0xffffffffu, payload, 0);
        idx = __shfl_sync(0xffffffffu, idx, 0);
    #endif
}


__device__ __forceinline__
void warp_radixsort_posf16(half& key, int& idx, int* src_lane_map)
{
    unsigned int lane_id = threadIdx.x % 32;
    const unsigned int active = 0xffffffffu;

    unsigned int ku = __half_as_ushort(key);

    #pragma unroll
    for (int bit = 0; bit < 15; ++bit)
    {
        unsigned int b = (ku >> bit) & 1;
        unsigned int ones = __ballot_sync(active, b);
        unsigned int zeros = active ^ ones;
        int nzeros = __popc(zeros);

        unsigned int below = (1 << lane_id) - 1;
        int r0 = __popc(zeros & below);
        int r1 = __popc(ones & below);

        int dest = b ? (nzeros + r1) : r0;
        int myrank = __popc(active & below);

        src_lane_map[dest] = lane_id;
        __syncwarp(active);
        int src = src_lane_map[myrank];

        ku = __shfl_sync(active, ku, src);
        idx = __shfl_sync(active, idx, src);
    }
    key = __ushort_as_half(ku);
}


__device__ __forceinline__
void warp_radixsort_posf32_pl(float& key, float& payload, int& idx, int* src_lane_map)
{
    unsigned int lane_id = threadIdx.x % 32;
    const unsigned int active = 0xffffffffu;

    unsigned int ku = __float_as_uint(key);

    #pragma unroll
    for (int bit = 0; bit < 31; ++bit)
    {
        unsigned int b = (ku >> bit) & 1u;
        unsigned int ones = __ballot_sync(active, b);
        unsigned int zeros = active ^ ones;
        int nzeros = __popc(zeros);

        unsigned int below = (1u << lane_id) - 1u;
        int r0 = __popc(zeros & below);
        int r1 = __popc(ones & below);

        int dest = b ? (nzeros + r1) : r0;
        int myrank = __popc(active & below);

        src_lane_map[dest] = lane_id;
        __syncwarp(active);
        int src = src_lane_map[myrank];

        ku = __shfl_sync(active, ku, src);
        payload = __shfl_sync(active, payload, src);
        idx = __shfl_sync(active, idx, src);
    }
    key = __uint_as_float(ku);
}


// Single-token router gemv on a transposed gate copy: scores = x @ gate_t.T. One warp per
// expert; cheaper than a cublas call at this size
#define RGEMV_WARPS 8

__global__ __launch_bounds__(RGEMV_WARPS * 32)
void routing_gemv_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const int k,
    const int E
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * RGEMV_WARPS + warp;
    if (row >= E) return;

    const half2* x2 = (const half2*) x;
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);

    float sum = 0.0f;
    for (int j = lane; j < k / 2; j += 32)
    {
        float2 xf = __half22float2(x2[j]);
        float2 wf = __half22float2(w2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);

    if (lane == 0)
        scores[row] = __float2half_rn(sum);
}

void routing_gemv
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    const c10::optional<at::Tensor>& gate_t,
    at::Tensor& scores,
    cudaStream_t stream
)
{
    int k = hidden.size(-1);
    int E = scores.size(-1);
    bool bsz1 = hidden.numel() == k;

    if (bsz1 && gate_t.has_value() && !(k & 1))
    {
        routing_gemv_kernel<<<CEIL_DIVIDE(E, RGEMV_WARPS), RGEMV_WARPS * 32, 0, stream>>>
        (
            (const half*) hidden.data_ptr(),
            (const half*) gate_t.value().data_ptr(),
            (half*) scores.data_ptr(),
            k, E
        );
    }
    else
    {
        hgemm(hidden, gate, scores);
    }
}


template <int ACT>
__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_ds3_nogroup_topk_kernel
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);
    bool mask = t < num_experts;

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    float* sh_key = reinterpret_cast<float*>(sh);
    float* sh_payload = reinterpret_cast<float*>(sh_key + num_warps * K);
    int* sh_idx = reinterpret_cast<int*>(sh_payload + num_warps * K);

    float logit = mask ? __half2float(scores[t]) : -1.0e30f;
    float act = bias && mask ? routing_act<ACT>(logit) : 0.0f;
    float key = mask ? (bias ? act + __half2float(bias[t]) : logit) : -1.0e30f;
    float payload = bias ? act : logit;
    int idx = mask ? t : -1;

    for (int k = 0; k < K; ++k)
    {
        float best_key = key;
        float best_payload = payload;
        int best_idx = idx;
        warp_reduce_best_f32(best_key, best_payload, best_idx);

        if (lane_id == k)
        {
            sh_key[warp_id * K + k] = best_key;
            sh_payload[warp_id * K + k] = best_payload;
            sh_idx[warp_id * K + k] = best_idx;
        }

        if (idx == best_idx) key = -1.0e30f;
    }
    __syncthreads();

    int num_candidates = num_warps * K;
    while (num_candidates > 32)
    {
        int stage_warps = CEIL_DIVIDE(num_candidates, 32);

        if (warp_id < stage_warps)
        {
            int pos = t;
            key = pos < num_candidates ? sh_key[pos] : -1.0e30f;
            payload = pos < num_candidates ? sh_payload[pos] : 0.0f;
            idx = pos < num_candidates ? sh_idx[pos] : -1;

            for (int k = 0; k < K; ++k)
            {
                float best_key = key;
                float best_payload = payload;
                int best_idx = idx;
                warp_reduce_best_f32(best_key, best_payload, best_idx);

                if (lane_id == k)
                {
                    sh_key[warp_id * K + k] = best_key;
                    sh_payload[warp_id * K + k] = best_payload;
                    sh_idx[warp_id * K + k] = best_idx;
                }

                if (idx == best_idx) key = -1.0e30f;
            }
        }
        __syncthreads();

        num_candidates = stage_warps * K;
    }

    if (warp_id == 0)
    {
        key = lane_id < num_candidates ? sh_key[lane_id] : -1.0e30f;
        payload = lane_id < num_candidates ? sh_payload[lane_id] : 0.0f;
        idx = lane_id < num_candidates ? sh_idx[lane_id] : -1;

        for (int k = 0; k < K; ++k)
        {
            float best_key = key;
            float best_payload = payload;
            int best_idx = idx;
            warp_reduce_best_f32(best_key, best_payload, best_idx);

            if (lane_id == k)
            {
                sh_payload[k] = bias ? best_payload : routing_act<ACT>(best_payload);
                sh_idx[k] = best_idx;
            }

            if (idx == best_idx) key = -1.0e30f;
        }

        __syncwarp();

        float o = lane_id < K ? sh_payload[lane_id] : 0.0f;
        float sum = warp_reduce_sum_first_k(o, K) + 1e-20f;
        if (lane_id < K)
        {
            topk_indices[lane_id] = (int64_t) sh_idx[lane_id];
            topk_weights[lane_id] = __float2half_rn(o * scaling_factor / sum);
        }
    }
}


template <int ACT>
__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_ds3_nogroup_kernel
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);
    bool mask = t < num_experts;

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    int K_ = K + (K & 1);
    float* sh_v = reinterpret_cast<float*>(sh);
    float* sh_o = reinterpret_cast<float*>(sh_v + K_ * num_warps);
    int* sh_idx = reinterpret_cast<int*>(sh_o + K_ * num_warps);
    int* perm = reinterpret_cast<int*>(sh_idx + K_ * num_warps);
    float* reduce = reinterpret_cast<float*>(perm + 32 * num_warps);

    // Input activation
    int idx = mask ? t : -1;  // output index
    float v = mask ? routing_act<ACT>(__half2float(scores[t])) : 0.0f;  // sort key
    float o = v;  // output weight

    // Add bias and shift sigmoid(logits) to be non-negative before radix sort
    if (bias)
    {
        v += mask ? __half2float(bias[t]) : 1e30;

        float minv = v;
        for (int offset = 32 >> 1; offset > 0; offset >>= 1)
            minv = fminf(minv, __shfl_down_sync(0xffffffff, minv, offset));
        if (lane_id == 0)
            reduce[warp_id] = minv;

        __syncthreads();

        if (warp_id == 0)
        {
            minv = lane_id < num_warps ? reduce[lane_id] : 1e30;
            for (int offset = 32 >> 1; offset > 0; offset >>= 1)
                minv = fminf(minv, __shfl_down_sync(0xffffffff, minv, offset));
            if (lane_id == 0)
                reduce[0] = minv;
        }

        __syncthreads();

        v -= reduce[0];
        if (!mask) v = 0.0f;
    }

    // Sort by v
    warp_radixsort_posf32_pl(v, o, idx, perm + warp_id * 32);

    while (num_warps > 1)
    {
        if (warp_id < num_warps && lane_id >= (32 - K))
        {
            int kpos = (32 - 1) - lane_id;
            sh_v[warp_id * K + kpos] = v;
            sh_o[warp_id * K + kpos] = o;
            sh_idx[warp_id * K + kpos] = idx;
        }
        __syncthreads();

        int num_experts_k = K * num_warps;
        num_warps = CEIL_DIVIDE(num_experts_k, 32);

        if (warp_id < num_warps)
        {
            if (t < num_experts_k && mask)
            {
                v = sh_v[t];
                o = sh_o[t];
                idx = sh_idx[t];
            }
            else
            {
                v = 0.0f;
                o = 0.0f;
                idx = -1;
            }
            warp_radixsort_posf32_pl(v, o, idx, perm + warp_id * 32);
        }
        __syncthreads();
    }

    // Normalize output in warp 0 lanes 32-K .. K, store result
    if (warp_id == 0)
    {
        float sum = warp_reduce_sum_last_k(o, K) + 1e-20;
        o *= scaling_factor / sum;

        if (lane_id >= (32 - K))
        {
            int kpos = (32 - 1) - lane_id;
            topk_indices[kpos] = (int64_t) idx;
            topk_weights[kpos] = __float2half_rn(o);
        }
    }
}


__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_std_topk_kernel
(
    const half* __restrict__ scores,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const bfloat16* __restrict__ per_expert_scale,
    const half* __restrict__ bias,
    int num_experts,
    int K,
    int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    float* sh_key = reinterpret_cast<float*>(sh);
    int* sh_idx = reinterpret_cast<int*>(sh_key + num_warps * K);
    float* max_red = reinterpret_cast<float*>(sh_idx + num_warps * K);

    bool mask = t < num_experts;
    float logit = mask ? __half2float(scores[t]) : -1.0e30f;
    // Router bias (gpt-oss): biased logits drive both the top-k selection and the softmax
    if (bias && mask)
        logit += __half2float(bias[t]);
    float max_logit = logit;
    max_logit = warp_reduce_max_f(max_logit);
    max_logit = __shfl_sync(0xffffffffu, max_logit, 0);

    if (num_warps > 1)
    {
        if (lane_id == 0) max_red[warp_id] = max_logit;
        __syncthreads();
        max_logit = lane_id < num_warps ? max_red[lane_id] : -1.0e30f;
        max_logit = warp_reduce_max_f(max_logit);
        max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
    }

    float key = logit;
    float payload = logit;
    int idx = mask ? t : -1;

    for (int k = 0; k < K; ++k)
    {
        float best_key = key;
        float best_payload = payload;
        int best_idx = idx;
        warp_reduce_best_f32(best_key, best_payload, best_idx);

        if (lane_id == k)
        {
            sh_key[warp_id * K + k] = best_key;
            sh_idx[warp_id * K + k] = best_idx;
        }

        if (idx == best_idx) key = -1.0e30f;
    }
    __syncthreads();

    int num_candidates = num_warps * K;
    while (num_candidates > 32)
    {
        int stage_warps = CEIL_DIVIDE(num_candidates, 32);

        if (warp_id < stage_warps)
        {
            int pos = t;
            key = pos < num_candidates ? sh_key[pos] : -1.0e30f;
            payload = key;
            idx = pos < num_candidates ? sh_idx[pos] : -1;

            for (int k = 0; k < K; ++k)
            {
                float best_key = key;
                float best_payload = payload;
                int best_idx = idx;
                warp_reduce_best_f32(best_key, best_payload, best_idx);

                if (lane_id == k)
                {
                    sh_key[warp_id * K + k] = best_key;
                    sh_idx[warp_id * K + k] = best_idx;
                }

                if (idx == best_idx) key = -1.0e30f;
            }
        }
        __syncthreads();

        num_candidates = stage_warps * K;
    }

    if (warp_id == 0)
    {
        key = lane_id < num_candidates ? sh_key[lane_id] : -1.0e30f;
        payload = key;
        idx = lane_id < num_candidates ? sh_idx[lane_id] : -1;

        for (int k = 0; k < K; ++k)
        {
            float best_key = key;
            float best_payload = payload;
            int best_idx = idx;
            warp_reduce_best_f32(best_key, best_payload, best_idx);

            if (lane_id == k)
            {
                sh_key[k] = expf(best_payload - max_logit);
                sh_idx[k] = best_idx;
            }

            if (idx == best_idx) key = -1.0e30f;
        }

        __syncwarp();

        float e = lane_id < K ? sh_key[lane_id] : 0.0f;
        float sum = warp_reduce_sum_first_k(e, K) + 1e-20f;
        e /= sum;

        if (lane_id < K)
        {
            int out_idx = sh_idx[lane_id];
            if (per_expert_scale)
                e *= __bfloat162float(per_expert_scale[out_idx]);
            topk_indices[lane_id] = (int64_t) out_idx;
            topk_weights[lane_id] = __float2half_rn(e);
        }
    }
}


__launch_bounds__(MAX_NUM_EXPERTS)
__global__ void routing_std_kernel
(
    const half* __restrict__ scores,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const bfloat16* __restrict__ per_expert_scale,
    int num_experts,
    int K,
    int bsz
)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = CEIL_DIVIDE(num_experts, 32);

    scores += num_experts * row;
    topk_indices += K * row;
    topk_weights += K * row;

    extern __shared__ unsigned char sh[];
    int K_ = K + (K & 1);
    half* sh_v = reinterpret_cast<half*>(sh);
    int* sh_idx = reinterpret_cast<int*>(sh_v + K_ * num_warps);
    int* perm = reinterpret_cast<int*>(sh_idx + K_ * num_warps);
    half* max_red = sh_v;

    // Get max logit, shift prior to sorting so int order matches float order (same sign). Also
    // stabilizes softmax at output

    half max_logit = t < num_experts ? scores[t] : __ushort_as_half(0xfbff);
    max_logit = warp_reduce_max_h(max_logit);
    max_logit = __shfl_sync(0xffffffffu, max_logit, 0);

    if (num_warps > 1)
    {
        max_red[warp_id] = max_logit;
        __syncthreads();
        max_logit = lane_id < num_warps ? max_red[lane_id] : __ushort_as_half(0xfbff);
        max_logit = warp_reduce_max_h(max_logit);
        max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
    }

    // Input logit, shifted

    int idx = t < num_experts ? t : -1;  // output index
    half v = t < num_experts ? __hsub(scores[t], max_logit) : __ushort_as_half(0xfbff);

    // Sort by v

    warp_radixsort_posf16(v, idx, perm + warp_id * 32);

    while (num_warps > 1)
    {
        if (warp_id < num_warps && lane_id < K)
        {
            int kpos = lane_id;
            sh_v[warp_id * K + kpos] = v;
            sh_idx[warp_id * K + kpos] = idx;
        }
        __syncthreads();

        int num_experts_k = K * num_warps;
        num_warps = CEIL_DIVIDE(num_experts_k, 32);

        if (warp_id < num_warps)
        {
            if (t < num_experts_k)
            {
                v = sh_v[t];
                idx = sh_idx[t];
            }
            else
            {
                v = __ushort_as_half(0xfbff);
                idx = -1;
            }
            warp_radixsort_posf16(v, idx, perm + warp_id * 32);
        }
        __syncthreads();
    }

    // Normalize output in first K lanes, store result

    if (warp_id == 0)
    {

        float e = expf(__half2float(v));
        float sum = warp_reduce_sum_first_k(e, K) + 1e-20;
        e /= sum;

        if (lane_id < K)
        {
            if (per_expert_scale)
                e *= __bfloat162float(per_expert_scale[idx]);
            int kpos = lane_id;
            topk_indices[kpos] = (int64_t) idx;
            topk_weights[kpos] = __float2half_rn(e);
        }
    }
}


/*
DS3 routing for n_group == 1, topk_group

hidden: Input hidden states, float16, shape (..., hidden_dim)
gate: Router gate matrix, float16, shape (hidden_dim, num_experts)
scores: Output routing logits buffer, float16, shape (bsz, num_experts)
bias: Pre-topk bias, float16, shape (1, num_experts)
topk_indices: int64, shape (bsz, k)
topk_weights: float16, shape (bsz, k)
routed_scaling_factor: float32
act_fn: score activation, ROUTING_ACT_SIGMOID (DS3/dots) or ROUTING_ACT_SQRTSP (DSv4)
*/

void routing_ds3_nogroup
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    routing_gemv(hidden, gate, gate_t, scores, stream);

    TORCH_CHECK_DTYPE(hidden, kHalf);
    TORCH_CHECK_DTYPE(gate, kHalf);
    TORCH_CHECK_SHAPES_OPT(bias, 0, scores, 1, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(hidden, -1, gate, 0, 1);
    TORCH_CHECK_SHAPES(gate, 1, scores, -1, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;

    // The iterative top-K kernel beats the radix-sort kernel at every measured size
    size_t shmem = num_warps * K * (2 * sizeof(float) + sizeof(int));
    auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
        routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SQRTSP> :
        routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SIGMOID>;
    kernel<<<bsz, num_threads, shmem, stream>>>
    (
        (const half*) scores.data_ptr(),
        (const half*) OPTPTR(bias),
        (int64_t*) topk_indices.data_ptr(),
        (half*) topk_weights.data_ptr(),
        scaling_factor,
        num_experts,
        K,
        bsz
    );
    cuda_check(cudaPeekAtLastError());
}


void routing_ds3_nogroup_logits
(
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const bool use_topk,
    const int act_fn
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES_OPT(bias, 0, scores, 1, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;

    if (use_topk)
    {
        size_t shmem = num_warps * K * (2 * sizeof(float) + sizeof(int));
        auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
            routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SQRTSP> :
            routing_ds3_nogroup_topk_kernel<ROUTING_ACT_SIGMOID>;
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (const half*) OPTPTR(bias),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            scaling_factor,
            num_experts,
            K,
            bsz
        );
    }
    else
    {
        int K_ = K + (K & 1);
        size_t shmem = num_warps * K_ * (2 * sizeof(float) + sizeof(int))
                     + num_threads * sizeof(int)
                     + num_warps * sizeof(float);
        auto kernel = act_fn == ROUTING_ACT_SQRTSP ?
            routing_ds3_nogroup_kernel<ROUTING_ACT_SQRTSP> :
            routing_ds3_nogroup_kernel<ROUTING_ACT_SIGMOID>;
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (const half*) OPTPTR(bias),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            scaling_factor,
            num_experts,
            K,
            bsz
        );
    }

    cuda_check(cudaPeekAtLastError());
}


/*
Routing weights for externally-selected experts (e.g. DSv4 hash-MoE, where selection is a
frozen token-id -> experts table): scores = hidden @ gate, then weights = normalized
activated scores gathered at `selected`, times scaling_factor. Selection order is preserved.

hidden: Input hidden states, float16, shape (bsz, hidden_dim)
gate: Router gate matrix, float16, shape (hidden_dim, num_experts)
scores: Output routing logits buffer, float16, shape (bsz, num_experts)
selected: Expert indices, int64, shape (bsz, k)
weights: Output, float16, shape (bsz, k)
*/

// Dynamic expert placement (CPU expert split): one pass over the selected-expert ids that
// records hit counts, translates router ids to physical slots inplace (the bsz-1 routing
// statics are read by captured graphs at baked addresses), and emits the worker-side
// selection with GPU-resident picks replaced by the -1 sentinel. Replaces a 4-5 launch
// torch chain per MoE layer per step, which measurably outweighed the placement benefit on
// 40-layer models

__global__ void moe_split_map_kernel
(
    int64_t* __restrict__ sel,           // (n,) router ids in, physical slots out
    const int64_t* __restrict__ map,     // (E,) router id -> physical slot
    float* __restrict__ hist,            // (E,) hit counts (router space)
    int64_t* __restrict__ sel_cpu,       // (n,) out: worker slot if CPU-resident else -1
    const int n,
    const int first_cpu_slot
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t r = sel[i];
    atomicAdd(&hist[r], 1.0f);
    int64_t p = map[r];
    sel_cpu[i] = p >= first_cpu_slot ? p - first_cpu_slot : -1;
    sel[i] = p;
}

void moe_split_map
(
    at::Tensor sel,
    const at::Tensor& map,
    at::Tensor hist,
    at::Tensor sel_cpu,
    const int64_t first_cpu_slot
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(map, kLong);
    TORCH_CHECK_DTYPE(hist, kFloat);
    TORCH_CHECK_DTYPE(sel_cpu, kLong);
    int n = (int) sel.numel();
    TORCH_CHECK(sel_cpu.numel() >= n, "moe_split_map: sel_cpu too small");
    int threads = n < 1024 ? n : 1024;
    int blocks = (n + threads - 1) / threads;
    moe_split_map_kernel<<<blocks, threads, 0, stream>>>
    (
        (int64_t*) sel.data_ptr(),
        (const int64_t*) map.data_ptr(),
        (float*) hist.data_ptr(),
        (int64_t*) sel_cpu.data_ptr(),
        n, (int) first_cpu_slot
    );
    cuda_check(cudaPeekAtLastError());
}

// Fused issue for the CPU expert split (decode-size jobs): one launch replaces the map/
// translate kernel, the int32 cast, the zero-pad and the three D2H memcpys of the two-phase
// submit path. The kernel writes the staged inputs straight into the device-mapped pinned
// slot with zero-copy stores, and skips the activation/weight payload entirely (no PCIe
// traffic) when no selected expert is CPU-resident (common case with a hot/cold-swapped
// tail). dev_count[slot] tells the collect kernel whether the worker produced anything. With
// `map`, also translates sel to physical slots in place and bumps the hit histogram (the
// dynamic-placement duties of moe_split_map)
__global__ void moe_split_issue_kernel
(
    int64_t* __restrict__ sel,           // (rows * topk,) router ids; physical slots out (map mode)
    const int64_t* __restrict__ map,     // (E,) or null (static split: ids are physical already)
    float* __restrict__ hist,            // (E,) or null
    const half* __restrict__ y,          // (rows, h_) device, contiguous
    const half* __restrict__ w,          // (rows, topk) device, contiguous
    int32_t* __restrict__ h_sel,         // slot sel, mapped host (rows, topk)
    half* __restrict__ h_x,              // slot x, mapped host (rows, hi)
    half* __restrict__ h_w,              // slot w, mapped host (rows, topk)
    int32_t* __restrict__ dev_count,     // (num_slots,) device
    const int n,                         // rows * topk
    const int rows,
    const int h_,
    const int hi,
    const int first_cpu,
    const int slot_idx
)
{
    __shared__ int s_any;
    const int t = threadIdx.x;
    if (t == 0) s_any = 0;
    __syncthreads();

    int any = 0;
    for (int i = t; i < n; i += blockDim.x)
    {
        int64_t p = sel[i];
        if (map)
        {
            atomicAdd(&hist[p], 1.0f);
            p = map[p];
            sel[i] = p;
        }
        int32_t lc = p >= first_cpu ? (int32_t)(p - first_cpu) : -1;
        h_sel[i] = lc;
        any |= (lc >= 0);
    }
    if (any) atomicOr(&s_any, 1);
    __syncthreads();
    if (t == 0) dev_count[slot_idx] = s_any;
    if (!s_any)
    {
        __threadfence_system();
        return;
    }

    for (int i = t; i < n; i += blockDim.x)
        h_w[i] = w[i];

    // Activations, zero-padded to the quantized input width. Vectorized when both widths
    // are 8-half aligned (they are for every real model dim); PCIe write-combining wants
    // the widest coalesced stores it can get
    if ((h_ & 7) == 0 && (hi & 7) == 0)
    {
        const int h8 = h_ >> 3, hi8 = hi >> 3;
        const int4* y4 = reinterpret_cast<const int4*>(y);
        int4* x4 = reinterpret_cast<int4*>(h_x);
        const int4 z = {0, 0, 0, 0};
        const int total = rows * hi8;
        for (int i = t; i < total; i += blockDim.x)
        {
            const int row = i / hi8;
            const int c = i - row * hi8;
            x4[i] = c < h8 ? y4[row * h8 + c] : z;
        }
    }
    else
    {
        const int total = rows * hi;
        for (int i = t; i < total; i += blockDim.x)
        {
            const int row = i / hi;
            const int c = i - row * hi;
            h_x[i] = c < h_ ? y[row * h_ + c] : __float2half_rn(0.0f);
        }
    }
    __threadfence_system();
}

void moe_split_issue
(
    at::Tensor sel,
    const c10::optional<at::Tensor>& map,
    const c10::optional<at::Tensor>& hist,
    const at::Tensor& y,
    const at::Tensor& w,
    int64_t h_sel_ptr,
    int64_t h_x_ptr,
    int64_t h_w_ptr,
    at::Tensor dev_count,
    const int64_t slot_idx,
    const int64_t hi,
    const int64_t first_cpu
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dev_count, kInt);
    TORCH_CHECK(y.is_contiguous() && w.is_contiguous() && sel.is_contiguous(),
                "moe_split_issue: inputs must be contiguous");
    TORCH_CHECK(map.has_value() == hist.has_value(), "moe_split_issue: map requires hist");
    int n = (int) sel.numel();
    int rows = (int) y.size(0);
    int h_ = (int) y.size(1);
    moe_split_issue_kernel<<<1, 1024, 0, stream>>>
    (
        (int64_t*) sel.data_ptr(),
        map ? (const int64_t*) map.value().data_ptr() : nullptr,
        hist ? (float*) hist.value().data_ptr() : nullptr,
        (const half*) y.data_ptr(),
        (const half*) w.data_ptr(),
        reinterpret_cast<int32_t*>(h_sel_ptr),
        reinterpret_cast<half*>(h_x_ptr),
        reinterpret_cast<half*>(h_w_ptr),
        (int32_t*) dev_count.data_ptr(),
        n, rows, h_, (int) hi, (int) first_cpu, (int) slot_idx
    );
    cuda_check(cudaPeekAtLastError());
}

// Fused collect for the split path: fold the worker's partial into the routed sum straight
// from the pinned slot (replacing the H2D memcpy, the width trim and the separate add), or
// do nothing at all -- no PCIe reads -- when the issue kernel recorded an empty job. Loads
// go through volatile so no stale line can be served for a slot's previous tenant
__global__ void moe_split_collect_add_kernel
(
    float* __restrict__ final_out,       // (rows, h_) device, contiguous
    const float* __restrict__ h_out,     // slot out, mapped host (rows, ho)
    const int32_t* __restrict__ dev_count,
    const int total,                     // rows * h_
    const int h_,
    const int ho,
    const int slot_idx
)
{
    if (dev_count[slot_idx] == 0) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / h_;
    const int c = i - row * h_;
    const volatile float* vo = h_out;
    final_out[i] += vo[row * ho + c];
}

void moe_split_collect_add
(
    at::Tensor final_out,
    int64_t h_out_ptr,
    const at::Tensor& dev_count,
    const int64_t slot_idx,
    const int64_t ho
)
{
    const at::cuda::OptionalCUDAGuard device_guard(final_out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(final_out, kFloat);
    TORCH_CHECK_DTYPE(dev_count, kInt);
    TORCH_CHECK(final_out.is_contiguous(), "moe_split_collect_add: output must be contiguous");
    int rows = (int) final_out.size(0);
    int h_ = (int) final_out.size(1);
    int total = rows * h_;
    int threads = total < 1024 ? total : 1024;
    int blocks = (total + threads - 1) / threads;
    moe_split_collect_add_kernel<<<blocks, threads, 0, stream>>>
    (
        (float*) final_out.data_ptr(),
        reinterpret_cast<const float*>(h_out_ptr),
        (const int32_t*) dev_count.data_ptr(),
        total, h_, (int) ho, (int) slot_idx
    );
    cuda_check(cudaPeekAtLastError());
}

__global__ void routing_sel_norm_kernel
(
    const half* __restrict__ scores,
    const int64_t* __restrict__ selected,
    half* __restrict__ weights,
    const float scaling_factor,
    const int num_experts,
    const int K,
    const int act_fn
)
{
    int row = blockIdx.x;
    int lane = threadIdx.x;

    float o = 0.0f;
    if (lane < K)
    {
        int e = (int) selected[row * K + lane];
        float logit = __half2float(scores[row * num_experts + e]);
        o = act_fn == ROUTING_ACT_SQRTSP ?
            routing_act<ROUTING_ACT_SQRTSP>(logit) :
            routing_act<ROUTING_ACT_SIGMOID>(logit);
    }

    float sum = warp_reduce_sum_first_k(o, K) + 1e-20f;
    if (lane < K)
        weights[row * K + lane] = __float2half_rn(o * scaling_factor / sum);
}

void routing_sel_norm
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const at::Tensor& selected,
    at::Tensor weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    routing_gemv(hidden, gate, gate_t, scores, stream);

    TORCH_CHECK_DTYPE(hidden, kHalf);
    TORCH_CHECK_DTYPE(gate, kHalf);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(selected, kLong);
    TORCH_CHECK_DTYPE(weights, kHalf);
    TORCH_CHECK_SHAPES(hidden, -1, gate, 0, 1);
    TORCH_CHECK_SHAPES(gate, 1, scores, -1, 1);
    TORCH_CHECK_SHAPES(scores, 0, selected, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, weights, 0, 1);
    TORCH_CHECK_SHAPES(selected, 1, weights, 1, 1);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = selected.size(1);
    TORCH_CHECK(K <= 32, "routing_sel_norm: K > 32");

    routing_sel_norm_kernel<<<bsz, 32, 0, stream>>>
    (
        (const half*) scores.data_ptr(),
        (const int64_t*) selected.data_ptr(),
        (half*) weights.data_ptr(),
        scaling_factor,
        num_experts,
        K,
        act_fn
    );
    cuda_check(cudaPeekAtLastError());
}

/*
Standard softmax routing

hidden: Input hidden states, float16, shape (..., hidden_dim)
gate: Router gate matrix, float16, shape (hidden_dim, num_experts)
scores: Output routing logits buffer, float16, shape (bsz, num_experts)
topk_indices: int64, shape (bsz, k)
topk_weights: float16, shape (bsz, k)
*/

void routing_std
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const c10::optional<at::Tensor>& gate_t,
    const c10::optional<at::Tensor>& bias
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    routing_gemv(hidden, gate, gate_t, scores, stream);

    TORCH_CHECK_DTYPE(hidden, kHalf);
    TORCH_CHECK_DTYPE(gate, kHalf);
    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(hidden, -1, gate, 0, 1);
    TORCH_CHECK_SHAPES(gate, 1, scores, -1, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_SHAPES_OPT(per_expert_scale, 0, scores, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);
    TORCH_CHECK_DTYPE_OPT(per_expert_scale, kBFloat16);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;
    int K_ = K + (K & 1);
    size_t shmem = num_warps * K_ * (sizeof(float) + sizeof(int))
                 + num_threads * sizeof(int)
                 + num_warps * sizeof(float);

    //int num_blocks = bsz;
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);
    routing_std_topk_kernel<<<bsz, num_threads, shmem, stream>>>
    (
        (const half*) scores.data_ptr(),
        (int64_t*) topk_indices.data_ptr(),
        (half*) topk_weights.data_ptr(),
        (const bfloat16*) OPTPTR(per_expert_scale),
        (const half*) OPTPTR(bias),
        num_experts,
        K,
        bsz
    );
    cuda_check(cudaPeekAtLastError());
}


void routing_std_logits
(
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const bool use_topk
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_SHAPES_OPT(per_expert_scale, 0, scores, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);
    TORCH_CHECK_DTYPE_OPT(per_expert_scale, kBFloat16);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");
    TORCH_CHECK(K <= num_experts, "K cannot exceed number of experts");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;

    if (use_topk)
    {
        size_t shmem = num_warps * K * (sizeof(float) + sizeof(int))
                     + num_warps * sizeof(float);
        routing_std_topk_kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            (const bfloat16*) OPTPTR(per_expert_scale),
            nullptr,
            num_experts,
            K,
            bsz
        );
    }
    else
    {
        int K_ = K + (K & 1);
        size_t shmem = num_warps * K_ * (sizeof(float) + sizeof(int))
                     + num_threads * sizeof(int)
                     + num_warps * sizeof(float);
        routing_std_kernel<<<bsz, num_threads, shmem, stream>>>
        (
            (const half*) scores.data_ptr(),
            (int64_t*) topk_indices.data_ptr(),
            (half*) topk_weights.data_ptr(),
            (const bfloat16*) OPTPTR(per_expert_scale),
            num_experts,
            K,
            bsz
        );
    }

    cuda_check(cudaPeekAtLastError());
}
