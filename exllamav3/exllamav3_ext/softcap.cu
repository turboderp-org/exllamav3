#include "softcap.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "util.h"
#include "util.cuh"

#define NUM_THREADS 1024

__launch_bounds__(NUM_THREADS)
__global__ void softcap_kernel
(
    const float* __restrict__ x,
    float* __restrict__ y,
    const uint64_t numel,
    const float scale
)
{
    uint64_t idx = (uint64_t)blockIdx.x * NUM_THREADS + (uint64_t)threadIdx.x;
    if (idx >= numel) return;

    float v = x[idx];
    v /= scale;
    v = tanhf(v);
    v *= scale;
    y[idx] = v;
}

__launch_bounds__(NUM_THREADS)
__global__ void softcap_h_kernel
(
    const half* __restrict__ x,
    half* __restrict__ y,
    const uint64_t numel,
    const float scale
)
{
    uint64_t idx = ((uint64_t)blockIdx.x * NUM_THREADS + (uint64_t)threadIdx.x) * 2;
    if (idx >= numel) return;

    half2 v01 = *((half2*)(x + idx));
    float v0 = __low2float(v01);
    float v1 = __high2float(v01);
    v0 /= scale;
    v1 /= scale;
    v0 = tanhf(v0);
    v1 = tanhf(v1);
    v0 *= scale;
    v1 *= scale;
    v01 = __floats2half2_rn(v0, v1);
    *((half2*)(y + idx)) = v01;
}

/*
Apply softcapping: y <-scale * tanh(x/scale)
Works inplace if x == y
*/

void softcap
(
    at::Tensor x,
    at::Tensor y,
    float scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    uint64_t numel = x.numel();

    if (x.dtype() == at::kFloat)
    {
        softcap_kernel<<<CEIL_DIVIDE(numel, NUM_THREADS), NUM_THREADS, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (float*) y.data_ptr(),
            numel,
            scale
        );
    }
    else if (x.dtype() == at::kHalf)
    {
        softcap_h_kernel<<<CEIL_DIVIDE(numel / 2, NUM_THREADS), NUM_THREADS, 0, stream>>>
        (
            (const half*) x.data_ptr(),
            (half*) y.data_ptr(),
            numel,
            scale
        );
    }
    else
    {
        TORCH_CHECK(false, "softcap wrong dtype");
    }
}
