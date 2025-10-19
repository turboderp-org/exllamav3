#include <cuda_fp16.h>
#include "routing.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "reduction.cuh"

#define MAX_NUM_EXPERTS 512
#define MAX_K 16


__device__ __forceinline__
float sigmoid_stable_hf(float xf)
{
    float ez = __expf(-fabsf(xf));
    float base = ez / (1.0f + ez);
    return (xf >= 0.0f) ? 1.0f - base : base;
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


__launch_bounds__(1024)
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
    int num_warps = num_experts / 32;

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

    // Input sigmoid and optional bias

    int idx = t;  // output index
    float v = sigmoid_stable_hf(__half2float(scores[t]));  // sort key
    float o = v ;  // output weight

    // Add bias
    if (bias)
    {
        v += __half2float(bias[t]);

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
            if (t < num_experts_k)
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


__launch_bounds__(1024)
__global__ void routing_std_kernel
(
    const half* __restrict__ scores,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
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
            int kpos = lane_id;
            topk_indices[kpos] = (int64_t) idx;
            topk_weights[kpos] = __float2half_rn(e);
        }
    }
}


/*
DS3 routing for n_group == 1, topk_group

scores: Input scores, float16, shape (bsz, num_experts)
bias: Pre-topk bias, float16, shape (1, num_experts)
topk_indices: int64, shape (bsz, k)
topk_weights: float16, shape (bsz, k)
routed_scaling_factor: float32
*/

void routing_ds3_nogroup
(
    const at::Tensor& scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor
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
    TORCH_CHECK(num_experts % 32 == 0, "num_experts must be a multiple of 32");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;
    int K_ = K + (K & 1);
    size_t shmem = num_warps * K_ * (2 * sizeof(float) + sizeof(int))
                 + num_threads * sizeof(int)
                 + num_warps * sizeof(float);

    //int num_blocks = bsz;
    routing_ds3_nogroup_kernel<<<bsz, num_threads, shmem, stream>>>
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

/*
Standard softmax routing

scores: Input scores, float16, shape (bsz, num_experts)
topk_indices: int64, shape (bsz, k)
topk_weights: float16, shape (bsz, k)
*/

void routing_std
(
    const at::Tensor& scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(scores, 0, topk_indices, 0, 1);
    TORCH_CHECK_SHAPES(scores, 0, topk_weights, 0, 1);
    TORCH_CHECK_SHAPES(topk_indices, 1, topk_weights, 1, 1);
    TORCH_CHECK_DTYPE(scores, kHalf);
    TORCH_CHECK_DTYPE(topk_indices, kLong);
    TORCH_CHECK_DTYPE(topk_weights, kHalf);

    int bsz = scores.size(0);
    int num_experts = scores.size(1);
    int K = topk_indices.size(1);

    TORCH_CHECK(num_experts <= MAX_NUM_EXPERTS, "Too many experts");
    TORCH_CHECK(K <= MAX_K, "Too many experts per token");

    int num_warps = CEIL_DIVIDE(num_experts, 32);
    int num_threads = num_warps * 32;
    int K_ = K + (K & 1);
    size_t shmem = num_warps * K_ * (sizeof(float) + sizeof(int))
                 + num_threads * sizeof(int)
                 + num_warps * sizeof(float);

    //int num_blocks = bsz;
    routing_std_kernel<<<bsz, num_threads, shmem, stream>>>
    (
        (const half*) scores.data_ptr(),
        (int64_t*) topk_indices.data_ptr(),
        (half*) topk_weights.data_ptr(),
        num_experts,
        K,
        bsz
    );
    cuda_check(cudaPeekAtLastError());
}
