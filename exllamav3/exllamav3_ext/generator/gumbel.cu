#include "sampling_basic.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include <limits>
#include <curand_kernel.h>

#define NUM_THREADS 1024

inline __device__ float gumbel(float x)
{
    return -__logf(fmaxf(-__logf(fmaxf(x, 1e-20)), 1e-20));
}

__global__ __launch_bounds__(NUM_THREADS)
void gumbel_noise_kernel_f16
(
    const half* __restrict__ in_logits,
    half* __restrict__ logits,
    int size,
    uint32_t random
)
{
    int idx = (threadIdx.x + NUM_THREADS * blockIdx.x) * 2;
    if (idx >= size) return;

    curandStatePhilox4_32_10_t state;
    curand_init(random, idx, 0, &state);

    half2 x01 = *((half2*) (in_logits + idx));
    float x0 = __half2float(__low2half(x01));
    float x1 = __half2float(__high2half(x01));
    float rf0 = curand_uniform(&state);
    curand_init(random, idx + 1, 0, &state);
    float rf1 = curand_uniform(&state);
    x0 += gumbel(rf0);
    x1 += gumbel(rf1);
    x01 = __floats2half2_rn(x0, x1);
    *((half2*) (logits + idx)) = x01;
}

__global__ __launch_bounds__(NUM_THREADS)
void gumbel_noise_kernel_f32
(
    const float* __restrict__ in_logits,
    float* __restrict__ logits,
    int size,
    uint32_t random
)
{
    int idx = threadIdx.x + NUM_THREADS * blockIdx.x;
    if (idx >= size) return;

    curandStatePhilox4_32_10_t state;
    curand_init(random, idx, 0, &state);

    float x = in_logits[idx];
    float rf = curand_uniform(&state);
    x += gumbel(rf);
    logits[idx] = x;
}


__global__ __launch_bounds__(NUM_THREADS)
void gumbel_noise_kernel_log
(
    const float* __restrict__ probs,
    float* __restrict__ logits,
    int size,
    uint32_t random
)
{
    int idx = threadIdx.x + NUM_THREADS * blockIdx.x;
    if (idx >= size) return;

    curandStatePhilox4_32_10_t state;
    curand_init(random, idx, 0, &state);

    float x = probs[idx];
    x = __logf(x);
    float rf = curand_uniform(&state);
    x += gumbel(rf);
    logits[idx] = x;
}

void gumbel_noise_f16
(
    const at::Tensor& logits_in,
    at::Tensor& logits,
    uint32_t random
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(logits_in, kHalf);
    TORCH_CHECK_DTYPE(logits, kHalf);

    int size = logits.numel();
    int blocks = CEIL_DIVIDE(size / 2, NUM_THREADS);

    gumbel_noise_kernel_f16<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const half*) logits_in.data_ptr(),
        (half*) logits.data_ptr(),
        size,
        random
    );
}

void gumbel_noise_f32
(
    const at::Tensor& logits_in,
    at::Tensor& logits,
    uint32_t random
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(logits_in, kFloat);
    TORCH_CHECK_DTYPE(logits, kFloat);

    int size = logits.numel();
    int blocks = CEIL_DIVIDE(size, NUM_THREADS);

    gumbel_noise_kernel_f32<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const float*) logits_in.data_ptr(),
        (float*) logits.data_ptr(),
        size,
        random
    );
}

void gumbel_noise_log
(
    const at::Tensor& probs,
    at::Tensor& logits,
    uint32_t random
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(probs, kFloat);
    TORCH_CHECK_DTYPE(logits, kFloat);
    TORCH_CHECK_SHAPES_FULL(probs, logits);

    int size = probs.numel();
    int blocks = CEIL_DIVIDE(size, NUM_THREADS);

    gumbel_noise_kernel_log<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const float*) probs.data_ptr(),
        (float*) logits.data_ptr(),
        size,
        random
    );
}