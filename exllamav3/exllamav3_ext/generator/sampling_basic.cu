#include "sampling_basic.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include <limits>
#include <curand_kernel.h>
#include "../reduction.cuh"

constexpr float NEG_INF_F32 = -std::numeric_limits<float>::infinity();
constexpr float POS_INF_F32 = std::numeric_limits<float>::infinity();

#define NUM_THREADS 1024

inline __device__ float gumbel(float x)
{
    return -__logf(fmaxf(-__logf(fmaxf(x, 1e-20)), 1e-20));
}

inline __device__ ValIdx argmax2f(int idx, float& x0, float& x1)
{
    ValIdx vi;
    if (x0 >= x1)
    {
        vi.val = x0;
        vi.idx = idx;
    }
    else
    {
        vi.val = x1;
        vi.idx = idx + 1;
    }
    vi = block_reduce_argmax(vi);
    return vi;
}

inline __device__ bool read2f
(
    const half* logits_ptr,
    int idx,
    float& x0,
    float& x1,
    int num_logits,
    int max_logit
)
{
    if (idx >= num_logits)
    {
        x0 = NEG_INF_F32;
        x1 = NEG_INF_F32;
        return false;
    }
    else
    {
        half2 x0x1 = *((half2*) (logits_ptr + idx));
        if (idx < max_logit - 1) x0 = __half2float(__low2half(x0x1));
        else x0 = NEG_INF_F32;
        if (idx < max_logit) x1 = __half2float(__high2half(x0x1));
        else x1 = NEG_INF_F32;
        return true;
    }
}

__global__ __launch_bounds__(NUM_THREADS)
void argmax_sample_kernel
(
    const half* __restrict__ logits,
    uint64_t* __restrict__ ids,
    int num_logits,
    int max_logit
)
{
    const half* logits_ptr = logits + num_logits * blockIdx.x;
    uint64_t* ids_ptr = ids + blockIdx.x;

    ValIdx maxvi = { NEG_INF_F32, 0 };
    int idx = threadIdx.x * 2;
    int blocks = CEIL_DIVIDE(max_logit, NUM_THREADS * 2);
    for (int block = 0; block < blocks; ++block, idx += NUM_THREADS * 2)
    {
        float x0, x1;
        read2f(logits_ptr, idx, x0, x1, num_logits, max_logit);
        ValIdx vi = argmax2f(idx, x0, x1);
        if (threadIdx.x == 0 && vi.val > maxvi.val)
            maxvi = vi;
    }

    if (threadIdx.x == 0)
        *ids_ptr = (uint64_t) maxvi.idx;
}

__global__ __launch_bounds__(NUM_THREADS)
void gumbel_sample_kernel
(
    const half* __restrict__ logits,
    uint64_t* __restrict__ ids,
    int num_logits,
    int max_logit,
    uint32_t random
)
{
    const half* logits_ptr = logits + num_logits * blockIdx.x;
    uint64_t* ids_ptr = ids + blockIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(random, threadIdx.x, 0, &state);

    ValIdx maxvi = { NEG_INF_F32, 0 };
    int idx = threadIdx.x * 2;
    int blocks = CEIL_DIVIDE(max_logit, NUM_THREADS * 2);
    for (int block = 0; block < blocks; ++block, idx += NUM_THREADS * 2)
    {
        float x0, x1;
        if (read2f(logits_ptr, idx, x0, x1, num_logits, max_logit))
        {
            float rf0 = curand_uniform(&state);
            float rf1 = curand_uniform(&state);
            x0 += gumbel(rf0);
            x1 += gumbel(rf1);
        }
        ValIdx vi = argmax2f(idx, x0, x1);
        if (threadIdx.x == 0 && vi.val > maxvi.val)
            maxvi = vi;
    }

    if (threadIdx.x == 0)
        *ids_ptr = (uint64_t) maxvi.idx;
}

void common
(
    const at::Tensor& logits,
    at::Tensor& ids,
    int& bsz,
    int& num_logits,
    int& max_logit
)
{
    TORCH_CHECK_DIM(logits, 2);
    TORCH_CHECK_DIM(ids, 2);
    TORCH_CHECK_DTYPE(logits, kHalf);
    TORCH_CHECK_DTYPE(ids, kLong);
    TORCH_CHECK_SHAPES(logits, 0, ids, 0, 1);

    bsz = logits.size(0);
    num_logits = logits.size(1);
    if (max_logit > num_logits) max_logit = num_logits;
}

void argmax_sample
(
    const at::Tensor& logits,
    at::Tensor& ids,
    int max_logit
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (!max_logit) max_logit = logits.size(-1);

    int bsz, num_logits;
    common(logits, ids, bsz, num_logits, max_logit);
    argmax_sample_kernel<<<bsz, NUM_THREADS, 0, stream>>>
    (
        (const half*) logits.data_ptr(),
        (uint64_t*) ids.data_ptr(),
        num_logits,
        max_logit
    );
}

void gumbel_sample
(
    const at::Tensor& logits,
    at::Tensor& ids,
    int max_logit,
    uint32_t random
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (!max_logit) max_logit = logits.size(-1);

    int bsz, num_logits;
    common(logits, ids, bsz, num_logits, max_logit);
    gumbel_sample_kernel<<<bsz, NUM_THREADS, 0, stream>>>
    (
        (const half*) logits.data_ptr(),
        (uint64_t*) ids.data_ptr(),
        num_logits,
        max_logit,
        random
    );
}
