#include <cuda_fp16.h>
#include "quantize.cuh"
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"
#include "exl3_devctx.cuh"
#include <cmath>

#define NUM_THREADS 512
#define H_INF __ushort_as_half(0x7c00)

#include "comp_units/quantize_tiles_instances.cuh"

#define __(i, cb) quantize_tiles_kernel_k##i##_cb##cb()
static const std::array<fp_quantize_tiles_kernel, 24> quantize_tiles_kernel_instances
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __


void quantize_tiles
(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int K,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(input_tiles, 2);
    TORCH_CHECK_SIZE(input_tiles, 1, 256);
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_indices);
    TORCH_CHECK_DTYPE(input_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_indices, kShort);
    TORCH_CHECK(K >= 1 && K <= 8, "quantize_tiles K must be in range 1..8");

    const int edges = 65536 >> K;
    const int num_tiles = input_tiles.size(0);
    if (!num_tiles) return;

    TORCH_CHECK_DTYPE(temp_costs, kHalf);
    TORCH_CHECK_DIM(temp_costs, 3);
    TORCH_CHECK_SIZE(temp_costs, 1, 2);
    TORCH_CHECK_SIZE(temp_costs, 2, edges);
    TORCH_CHECK_DTYPE(temp_edges, kShort);
    TORCH_CHECK_DIM(temp_edges, 3);
    TORCH_CHECK_SIZE(temp_edges, 1, 256);
    TORCH_CHECK_SIZE(temp_edges, 2, edges);

    int device;
    cudaGetDevice(&device);
    const int max_batch_size = MIN((int) temp_costs.size(0), 2 * DevCtx::instance().get_num_sms(device));
    const int shmem = (K >= 2 ? 2 * edges * sizeof(half) : 0) + 512 + 64 + 128;
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;
    auto kernel = quantize_tiles_kernel_instances[K - 1 + 8 * cb];
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    cuda_check(cudaPeekAtLastError());

    for (int batch_i = 0; batch_i < num_tiles; batch_i += max_batch_size)
    {
        const int bsz = MIN(max_batch_size, num_tiles - batch_i);
        kernel<<<bsz, NUM_THREADS, shmem, stream>>>
        (
            ((const float*) input_tiles.data_ptr()) + 256 * batch_i,
            ((float*) output_tiles.data_ptr()) + 256 * batch_i,
            ((uint16_t*) output_indices.data_ptr()) + 256 * batch_i,
            (half*) temp_costs.data_ptr(),
            (uint16_t*) temp_edges.data_ptr()
        );
        cuda_check(cudaPeekAtLastError());
    }
}

template <typename T>
__global__ //__launch_bounds__(64)
void decode_kernel
(
    const uint16_t* __restrict__ input_tiles_ptr,
    T* __restrict__ output_tiles_ptr,
    int cols,
    bool mcg,
    bool mul1
)
{
    int col = threadIdx.x + blockIdx.x * 64;
    if (col >= cols) return;
    int row = blockIdx.y;
    int idx = row * cols + col;

    uint32_t enc = (uint32_t) input_tiles_ptr[idx];
    half w;
    if (mcg)
        w = decode_3inst<1>(enc);
    else if (mul1)
        w = decode_3inst<2>(enc);
    else
        w = decode_3inst<0>(enc);

    if constexpr (std::is_same_v<T, float>)
        output_tiles_ptr[idx] = __half2float(w);
    else
        output_tiles_ptr[idx] = w;
}

/*
Decode tensor

input_indices: uint16_t
output_tiles: float or half
mcg: use mcg codebook
mul1: use mcg codebook
*/

void decode
(
    at::Tensor input_indices,
    at::Tensor output_tiles,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_indices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(input_indices, 2);
    TORCH_CHECK_SHAPES_FULL(input_indices, output_tiles);
    TORCH_CHECK_DTYPE(input_indices, kShort);

    int rows = input_indices.size(0);
    int cols = input_indices.size(1);

    dim3 blockDim(64);
    dim3 gridDim(cols / 64, rows);

    if (output_tiles.dtype() == at::kFloat)
        decode_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const uint16_t*) input_indices.data_ptr(),
            (float*) output_tiles.data_ptr(),
            cols,
            mcg,
            mul1
        );
    else if (output_tiles.dtype() == at::kHalf)
        decode_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const uint16_t*) input_indices.data_ptr(),
            (half*) output_tiles.data_ptr(),
            cols,
            mcg,
            mul1
        );
}


#define NUM_THREADS_TD 1024
#define MAX_BINS 1024

__global__ __launch_bounds__(NUM_THREADS_TD)
void test_distribution_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ dist_output_ptr,
    float* __restrict__ ref_output_ptr,
    uint64_t numel,
    uint64_t num_bins,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
)
{
    __shared__ int histogram[MAX_BINS];
    auto reset_histogram = [&]()
    {
        for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS_TD)
            histogram[i] = 0;
        __syncthreads();
    };

    auto write_histogram = [&](float* output_ptr, uint64_t sc)
    {
        float scf = (float) sc;
        for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS_TD)
            output_ptr[i] = ((float) histogram[i]) / scf;
        __syncthreads();
    };

    auto count = [&](float val)
    {
        val -= min_value;
        val /= (max_value - min_value);
        val *= (float) num_bins;
        int idx = (int) val;
        if (idx < 0) idx = 0;
        if (idx > num_bins - 1) idx = num_bins - 1;
        atomicAdd(&histogram[idx], 1);
    };

    if (ref_output_ptr)
    {
        reset_histogram();
        for (uint64_t i = threadIdx.x; i < 65536; i += NUM_THREADS_TD)
        {
            if (mcg)
                count(decode_3inst_f<1>((uint16_t) (i & 0xffff)));
            else if (mul1)
                count(decode_3inst_f<2>((uint16_t) (i & 0xffff)));
            else
                count(decode_3inst_f<0>((uint16_t) (i & 0xffff)));
        }
        __syncthreads();
        write_histogram(ref_output_ptr, 65536);
    }

    reset_histogram();
    for (uint64_t i = threadIdx.x; i < numel; i += NUM_THREADS_TD)
        count(input_ptr[i]);
    __syncthreads();
    write_histogram(dist_output_ptr, numel);
}

/*
Compare tensor distribution to codebook (not optimized)

input: tensor, float, any shape
dist_output: (empty) output histogram, float, shape (num_bins,)
ref_output, optional: (empty) output codebook histogram, float, shape (num_bins,)
*/

void test_distribution
(
    at::Tensor& input,
    at::Tensor& dist_output,
    const c10::optional<at::Tensor>& ref_output,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(input, kFloat);

    uint64_t numel = input.numel();
    float* ref_output_ptr = (float*) OPTPTR(ref_output);
    uint64_t num_bins = dist_output.numel();
    TORCH_CHECK(num_bins <= MAX_BINS, "Too many bins");
    if (ref_output_ptr)
        TORCH_CHECK(num_bins == ref_output.value().numel());

    test_distribution_kernel<<<1, NUM_THREADS_TD, 0, stream>>>
    (
        (const float*) input.data_ptr(),
        (float*) dist_output.data_ptr(),
        (float*) ref_output_ptr,
        numel,
        num_bins,
        min_value,
        max_value,
        mcg,
        mul1
    );
}
