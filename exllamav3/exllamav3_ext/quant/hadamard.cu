#include "quantize.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "hadamard_inner.cuh"

__global__ __launch_bounds__(32)
void had_hf_r_128_kernel
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    float r_scale
)
{
    input_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner(input_ptr, output_ptr, pre_scale, post_scale, r_scale);
}

__global__ __launch_bounds__(32)
void had_ff_r_128_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    float r_scale
)
{
    input_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner(input_ptr, output_ptr, pre_scale, post_scale, r_scale);
}

/*
Compute y = (x.view(-1, 128) @ had_128).view(x.shape)
Works inplace if y == x
x and y must be same dtype, either float16 or float32
*/
void had_r_128
(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    float scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES_FULL(input, output);
    TORCH_CHECK_DIM(input, 2);
    TORCH_CHECK_DIV(input, 1, 128);
    int rows = input.size(0);
    int cols = input.size(1);

    int blocks = cols / 128;
    float r_scale = scale * 0.088388347648f; // scale / sqrt(128)

    dim3 blockDim(32);
    dim3 gridDim(rows, blocks);

    if (input.dtype() == at::kHalf)
    {
        TORCH_CHECK_DTYPE(output, kHalf);
        had_hf_r_128_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const half*) input.data_ptr(),
            (half*) output.data_ptr(),
            (const half*) OPTPTR(pre_scale),
            (const half*) OPTPTR(post_scale),
            r_scale
        );
    }

    else if (input.dtype() == at::kFloat)
    {
        TORCH_CHECK_DTYPE(output, kFloat);
        had_ff_r_128_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const float*) input.data_ptr(),
            (float*) output.data_ptr(),
            (const half*) OPTPTR(pre_scale),
            (const half*) OPTPTR(post_scale),
            r_scale
        );
    }

    else TORCH_CHECK(false, "unsupported datatype");
}