#pragma once

struct ValIdx
{
    float val;
    int idx;
};

__device__ inline ValIdx warp_reduce_argmax(ValIdx v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        if (other_val > v.val)
        {
            v.val = other_val;
            v.idx = other_idx;
        }
    }
    return v;
}

__device__ inline ValIdx block_reduce_argmax(ValIdx v)
{
    __shared__ ValIdx shared[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_argmax(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    if (warp_id == 0)
    {
        v = shared[lane_id];
        v = warp_reduce_argmax(v);
    }
    return v;
}

__device__ inline half warp_reduce_max_h(half v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        half2 other_v = __shfl_down_sync(0xffffffff, __half2half2(v), offset);
        v = __hmax(v, __low2half(other_v));
    }
    return v;
}

__device__ inline float warp_reduce_max_f(float v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v = fmaxf(v, other_v);
    }
    return v;
}

__device__ inline half block_reduce_max_h(half v, int num_threads)
{
    __shared__ half shared[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_max_h(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    int max_warp_id = num_threads / 32;
    if (warp_id == 0)
    {
        v = lane_id < max_warp_id ? shared[lane_id] : NEG_INF_F16;
        v = warp_reduce_max_h(v);
    }
    return v;
}

__device__ inline float warp_reduce_sum_f(float v)
{
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    return v;
}

__device__ inline float warp_reduce_sum_last_k(float v, int K)
{
    int lane_id = threadIdx.x % 32;
    if (lane_id < (32 - K)) v = 0.0f;
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    return v;
}

__device__ inline float warp_reduce_sum_first_k(float v, int K)
{
    int lane_id = threadIdx.x % 32;
    if (lane_id >= K) v = 0.0f;
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        float other_v = __shfl_down_sync(0xffffffff, v, offset);
        v += other_v;
    }
    v = __shfl_sync(0xffffffffu, v, 0);
    return v;
}

__device__ inline float block_reduce_sum_f(float v, int num_threads)
{
    __shared__ float shared[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_sum_f(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    int max_warp_id = num_threads / 32;
    if (warp_id == 0)
    {
        v = lane_id < max_warp_id ? shared[lane_id] : 0.0f;
        v = warp_reduce_sum_f(v);
    }
    return v;
}

__device__ inline float block_reduce_sum_broadcast_f(float v, int num_threads)
{
    __shared__ float shared[32];

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    v = warp_reduce_sum_f(v);

    if (lane_id == 0) shared[warp_id] = v;
    __syncthreads();

    int max_warp_id = num_threads / 32;
    if (warp_id == 0)
    {
        v = lane_id < max_warp_id ? shared[lane_id] : 0.0f;
        v = warp_reduce_sum_f(v);
        shared[0] = v;
    }
    __syncthreads();
    v = shared[0];
    return v;
}