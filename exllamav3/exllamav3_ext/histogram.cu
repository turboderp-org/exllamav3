#include <cuda_fp16.h>
#include "histogram.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include <cmath>

#define NUM_THREADS 1024
#define MAX_BINS 1024

template <typename T>
__global__ __launch_bounds__(NUM_THREADS)
void histogram_kernel
(
    const T* __restrict__ input_ptr,
    unsigned long long* __restrict__ output_ptr,
    uint64_t numel,
    uint64_t num_bins,
    float min_value,
    float max_value,
    bool exclusive
)
{
    __shared__ unsigned long long histogram[MAX_BINS];

    // Clear
    for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS)
        histogram[i] = 0;
    __syncthreads();

    // Count
    for (size_t i = threadIdx.x; i < numel; i += NUM_THREADS)
    {
        float val;
        if constexpr (std::is_same_v<T, float>)
            val = input_ptr[i];
        else
            val = __half2float(input_ptr[i]);

        if (exclusive)
        {
            if (val < min_value) continue;
            if (val > max_value) continue;
        }

        val -= min_value;
        val /= (max_value - min_value);
        val *= (float) num_bins;
        int idx = (int) val;
        if (idx < 0) idx = 0;
        if (idx >= num_bins) idx = num_bins - 1;
        atomicAdd(&histogram[idx], 1);
    }
    __syncthreads();

    // Write
    for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS)
        output_ptr[i] = histogram[i];
}

/*
Compare tensor distribution to codebook (not optimized)

input: tensor, float, any shape
output: (empty) output histogram, uint64_t, shape (num_bins,)
*/

void histogram
(
    at::Tensor& input,
    at::Tensor& output,
    float min_value,
    float max_value,
    bool exclusive
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    bool float32;
    if (input.dtype() == at::kFloat)
        float32 = true;
    else if (input.dtype() == at::kHalf)
        float32 = false;
    else TORCH_CHECK(false, "incorrect input datatype");

    TORCH_CHECK_DTYPE(output, kLong);

    uint64_t numel = input.numel();
    uint64_t num_bins = output.numel();
    TORCH_CHECK(num_bins <= MAX_BINS, "Too many bins");

    if (float32)
        histogram_kernel<float><<<1, NUM_THREADS, 0, stream>>>
        (
            (const float*) input.data_ptr(),
            (unsigned long long*) output.data_ptr(),
            numel,
            num_bins,
            min_value,
            max_value,
            exclusive
        );
    else
        histogram_kernel<half><<<1, NUM_THREADS, 0, stream>>>
        (
            (const half*) input.data_ptr(),
            (unsigned long long*) output.data_ptr(),
            numel,
            num_bins,
            min_value,
            max_value,
            exclusive
        );

    cuda_check(cudaPeekAtLastError());
}