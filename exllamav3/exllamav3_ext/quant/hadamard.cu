#include <cuda_fp16.h>
#include "quantize.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "hadamard_inner.cuh"

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_hf_r_128_kernel
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_ff_r_128_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale
)
{
    input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input_ptr, output_ptr, scale, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_hf_r_128_dual_kernel
(
    const half* __restrict__ input1_ptr,
    half* __restrict__ output1_ptr,
    const half* __restrict__ scale_1,
    const half* __restrict__ input2_ptr,
    half* __restrict__ output2_ptr,
    const half* __restrict__ scale_2,
    const float r_scale
)
{
    input1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input1_ptr, output1_ptr, scale_1, r_scale);

    input2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input2_ptr, output2_ptr, scale_2, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_ff_r_128_dual_kernel
(
    const float* __restrict__ input1_ptr,
    float* __restrict__ output1_ptr,
    const half* __restrict__ scale_1,
    const float* __restrict__ input2_ptr,
    float* __restrict__ output2_ptr,
    const half* __restrict__ scale_2,
    const float r_scale
)
{
    input1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output1_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input1_ptr, output1_ptr, scale_1, r_scale);

    input2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output2_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input2_ptr, output2_ptr, scale_2, r_scale);
}

// Batched-expert variant: the rows are B blocks of rows_per_expert rows, and the scale vector
// of a row comes from a [E, cols] table selected by ids[row / rows_per_expert] (the slab
// layout of moe_batch_recon.py)
template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_hf_r_128_batch_kernel
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ table,
    const int64_t* __restrict__ ids,
    const int rows_per_expert,
    const float r_scale
)
{
    size_t off = (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    const half* scale = table + (size_t) ids[blockIdx.x / rows_per_expert] * gridDim.y * 128;
    had_hf_r_128_inner<pre_scale, post_scale>(input_ptr + off, output_ptr + off, scale, r_scale);
}

template <bool pre_scale, bool post_scale>
__global__ __launch_bounds__(32)
void had_ff_r_128_batch_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ table,
    const int64_t* __restrict__ ids,
    const int rows_per_expert,
    const float r_scale
)
{
    size_t off = (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    const half* scale = table + (size_t) ids[blockIdx.x / rows_per_expert] * gridDim.y * 128;
    had_ff_r_128_inner<pre_scale, post_scale>(input_ptr + off, output_ptr + off, scale, r_scale);
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
    const float scale
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
        if (pre_scale.has_value())
            had_hf_r_128_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input.data_ptr(),
                (half*) output.data_ptr(),
                (const half*) OPTPTR(pre_scale),
                r_scale
            );
        else if (post_scale.has_value())
            had_hf_r_128_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input.data_ptr(),
                (half*) output.data_ptr(),
                (const half*) OPTPTR(post_scale),
                r_scale
            );
        else
            had_hf_r_128_kernel<false, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input.data_ptr(),
                (half*) output.data_ptr(),
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else if (input.dtype() == at::kFloat)
    {
        TORCH_CHECK_DTYPE(output, kFloat);
        if (pre_scale.has_value())
            had_ff_r_128_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input.data_ptr(),
                (float*) output.data_ptr(),
                (const half*) OPTPTR(pre_scale),
                r_scale
            );
        else if (post_scale.has_value())
            had_ff_r_128_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input.data_ptr(),
                (float*) output.data_ptr(),
                (const half*) OPTPTR(post_scale),
                r_scale
            );
        else
            had_ff_r_128_kernel<false, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input.data_ptr(),
                (float*) output.data_ptr(),
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else TORCH_CHECK(false, "unsupported datatype");
}

void had_r_128_dual
(
    const at::Tensor& input1,
    const at::Tensor& output1,
    const c10::optional<at::Tensor>& pre_scale1,
    const c10::optional<at::Tensor>& post_scale1,
    const at::Tensor& input2,
    const at::Tensor& output2,
    const c10::optional<at::Tensor>& pre_scale2,
    const c10::optional<at::Tensor>& post_scale2,
    const float scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input1.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES_FULL(input1, output1);
    TORCH_CHECK_SHAPES_FULL(input1, input2);
    TORCH_CHECK_SHAPES_FULL(output1, output2);
    TORCH_CHECK_DIM(input1, 2);
    TORCH_CHECK_DIV(input1, 1, 128);
    int rows = input1.size(0);
    int cols = input1.size(1);

    TORCH_CHECK(
        pre_scale1.has_value() == pre_scale2.has_value() &&
        post_scale1.has_value() == post_scale2.has_value(),
        "Cannot mix scaling modes in dual had"
    )

    int blocks = cols / 128;
    float r_scale = scale * 0.088388347648f; // scale / sqrt(128)

    dim3 blockDim(32);
    dim3 gridDim(rows, blocks);

    if (input1.dtype() == at::kHalf)
    {
        TORCH_CHECK_DTYPE(output1, kHalf);
        if (pre_scale1.has_value())
            had_hf_r_128_dual_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input1.data_ptr(),
                (half*) output1.data_ptr(),
                (const half*) OPTPTR(pre_scale1),
                (const half*) input2.data_ptr(),
                (half*) output2.data_ptr(),
                (const half*) OPTPTR(pre_scale2),
                r_scale
            );
        else if (post_scale1.has_value())
            had_hf_r_128_dual_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input1.data_ptr(),
                (half*) output1.data_ptr(),
                (const half*) OPTPTR(post_scale1),
                (const half*) input2.data_ptr(),
                (half*) output2.data_ptr(),
                (const half*) OPTPTR(post_scale2),
                r_scale
            );
        else
            had_hf_r_128_dual_kernel<false, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const half*) input1.data_ptr(),
                (half*) output1.data_ptr(),
                (const half*) nullptr,
                (const half*) input2.data_ptr(),
                (half*) output2.data_ptr(),
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else if (input1.dtype() == at::kFloat)
    {
        TORCH_CHECK_DTYPE(output1, kFloat);
        if (pre_scale1.has_value())
            had_ff_r_128_dual_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input1.data_ptr(),
                (float*) output1.data_ptr(),
                (const half*) OPTPTR(pre_scale1),
                (const float*) input2.data_ptr(),
                (float*) output2.data_ptr(),
                (const half*) OPTPTR(pre_scale2),
                r_scale
            );
        else if (post_scale1.has_value())
            had_ff_r_128_dual_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input1.data_ptr(),
                (float*) output1.data_ptr(),
                (const half*) OPTPTR(post_scale1),
                (const float*) input2.data_ptr(),
                (float*) output2.data_ptr(),
                (const half*) OPTPTR(post_scale2),
                r_scale
            );
        else
            had_ff_r_128_dual_kernel<false, false><<<gridDim, blockDim, 0, stream>>>
            (
                (const float*) input1.data_ptr(),
                (float*) output1.data_ptr(),
                (const half*) nullptr,
                (const float*) input2.data_ptr(),
                (float*) output2.data_ptr(),
                (const half*) nullptr,
                r_scale
            );
        cuda_check(cudaPeekAtLastError());
    }

    else TORCH_CHECK(false, "unsupported datatype");
}


/*
had_r_128 over a [B * rows_per_expert, cols] slab with per-expert pre/post scale vectors: the
scale of row r is table[ids[r / rows_per_expert]], tables are [E, cols] half. Works in place.
*/
void had_r_128_batch
(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& pre_table,
    const c10::optional<at::Tensor>& post_table,
    const at::Tensor& ids,
    int rows_per_expert,
    const float scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES_FULL(input, output);
    TORCH_CHECK_DIM(input, 2);
    TORCH_CHECK_DIV(input, 1, 128);
    TORCH_CHECK_DTYPE(ids, kLong);
    TORCH_CHECK(ids.is_contiguous(), "ids must be contiguous");
    TORCH_CHECK(rows_per_expert > 0, "rows_per_expert must be positive");
    TORCH_CHECK(pre_table.has_value() != post_table.has_value(), "exactly one of pre_table / post_table");
    int rows = input.size(0);
    int cols = input.size(1);
    TORCH_CHECK(rows % rows_per_expert == 0, "rows must be a multiple of rows_per_expert");
    TORCH_CHECK(ids.numel() >= rows / rows_per_expert, "ids too short for the slab");
    const at::Tensor& table = pre_table.has_value() ? pre_table.value() : post_table.value();
    TORCH_CHECK_DTYPE(table, kHalf);
    TORCH_CHECK_DIM(table, 2);
    TORCH_CHECK(table.is_contiguous() && table.size(1) == cols, "scale table must be contiguous [E, cols]");
    if (!rows) return;

    int blocks = cols / 128;
    float r_scale = scale * 0.088388347648f;
    dim3 blockDim(32);
    dim3 gridDim(rows, blocks);
    bool pre = pre_table.has_value();

    if (input.dtype() == at::kHalf)
    {
        TORCH_CHECK_DTYPE(output, kHalf);
        if (pre)
            had_hf_r_128_batch_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            ((const half*) input.data_ptr(), (half*) output.data_ptr(), (const half*) table.data_ptr(),
             (const int64_t*) ids.data_ptr(), rows_per_expert, r_scale);
        else
            had_hf_r_128_batch_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            ((const half*) input.data_ptr(), (half*) output.data_ptr(), (const half*) table.data_ptr(),
             (const int64_t*) ids.data_ptr(), rows_per_expert, r_scale);
    }
    else if (input.dtype() == at::kFloat)
    {
        TORCH_CHECK_DTYPE(output, kFloat);
        if (pre)
            had_ff_r_128_batch_kernel<true, false><<<gridDim, blockDim, 0, stream>>>
            ((const float*) input.data_ptr(), (float*) output.data_ptr(), (const half*) table.data_ptr(),
             (const int64_t*) ids.data_ptr(), rows_per_expert, r_scale);
        else
            had_ff_r_128_batch_kernel<false, true><<<gridDim, blockDim, 0, stream>>>
            ((const float*) input.data_ptr(), (float*) output.data_ptr(), (const half*) table.data_ptr(),
             (const int64_t*) ids.data_ptr(), rows_per_expert, r_scale);
    }
    else TORCH_CHECK(false, "unsupported datatype");
    cuda_check(cudaPeekAtLastError());
}
