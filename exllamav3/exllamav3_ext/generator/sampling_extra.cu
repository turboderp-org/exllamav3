#include <cuda_fp16.h>
#include "sampling_basic.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include <limits>
#include <curand_kernel.h>

#define NUM_THREADS 1024

inline __device__ float gumbel(float x)
{
    return -__logf(fmaxf(-__logf(fmaxf(x, 1e-20)), 1e-20));
}

constexpr float NEG_INF_F32 = -std::numeric_limits<float>::infinity();

__global__ __launch_bounds__(NUM_THREADS)
void adaptivep_gumbel_noise_kernel_f32
(
    const float* __restrict__ probs_in,
    float* __restrict__ logits,
    const int size,
    const uint32_t random,
    float adapted_target,
    float inv_width,
    float peak_logit_value,
    float sharpness
)
{
    int idx = threadIdx.x + NUM_THREADS * blockIdx.x;
    if (idx >= size) return;

    float x = probs_in[idx];
    if (x < 1e-8)
    {
        x = NEG_INF_F32;
    }
    else
    {
        curandStatePhilox4_32_10_t state;
        curand_init(random, idx, 0, &state);

        float adapted_prob = fabs(x - adapted_target) * inv_width;
        x = peak_logit_value - sharpness * adapted_prob * adapted_prob / (adapted_prob + 1.0);
        float rf = curand_uniform(&state);
        x += gumbel(rf);
    }

    logits[idx] = x;
}

// Produces adaptive-P faux-logits from truncated probabilities, then adds gumbel noise

void adaptivep_gumbel_noise_f32
(
    const at::Tensor& probs_in,
    at::Tensor& logits,
    uint32_t random,
    float adapted_target,
    float inv_width,
    float peak_logit_value,
    float sharpness
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(probs_in, kFloat);
    TORCH_CHECK_DTYPE(logits, kFloat);

    int size = logits.numel();
    int blocks = CEIL_DIVIDE(size, NUM_THREADS);

    adaptivep_gumbel_noise_kernel_f32<<<blocks, NUM_THREADS, 0, stream>>>
    (
        (const float*) probs_in.data_ptr(),
        (float*) logits.data_ptr(),
        size,
        random,
        adapted_target,
        inv_width,
        peak_logit_value,
        sharpness
    );
}