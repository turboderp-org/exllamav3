#include <cuda_fp16.h>
#include "tq3_dequant.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"

// TQ3 weight dequantization kernel
// Input: packed bitplanes (uint32) + scales (fp16)
// Output: fp16 weight matrix
// Each block of 32 weights: 2 bitplane uint32s + 1 fp16 scale
// Bitplane 0 = nonzero mask, Bitplane 1 = positive mask
// Value = nonzero * (2*positive - 1) * scale

__global__ void dequant_tq3_weight_kernel(
    const uint32_t* __restrict__ packed,    // (num_blocks * 2, out_features)
    const half* __restrict__ scales,        // (num_blocks, out_features)
    half* __restrict__ output,              // (in_features, out_features)
    int num_blocks,                         // in_features / 32
    int out_features
)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_features) return;

    int block_idx = blockIdx.y;
    if (block_idx >= num_blocks) return;

    // Read bitplanes for this block and column
    uint32_t bp0 = packed[(block_idx * 2) * out_features + col];      // nonzero mask
    uint32_t bp1 = packed[(block_idx * 2 + 1) * out_features + col];  // positive mask
    float s = __half2float(scales[block_idx * out_features + col]);

    // Expand 32 values
    int base_row = block_idx * 32;
    #pragma unroll
    for (int bit = 0; bit < 32; bit++) {
        int nz = (bp0 >> bit) & 1;
        int pos = (bp1 >> bit) & 1;
        float val = nz ? (pos ? s : -s) : 0.0f;
        output[(base_row + bit) * out_features + col] = __float2half(val);
    }
}

void dequant_tq3_weight(
    const at::Tensor& packed,
    const at::Tensor& scales,
    at::Tensor& output
)
{
    const at::cuda::OptionalCUDAGuard device_guard(packed.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(packed, kInt);
    TORCH_CHECK_DTYPE(scales, kHalf);
    TORCH_CHECK_DTYPE(output, kHalf);

    int out_features = output.size(1);
    int in_features = output.size(0);
    int num_blocks = in_features / 32;

    TORCH_CHECK(in_features % 32 == 0, "in_features must be multiple of 32");

    dim3 grid(CEIL_DIVIDE(out_features, 256), num_blocks);
    dim3 threads(256);

    dequant_tq3_weight_kernel<<<grid, threads, 0, stream>>>(
        (const uint32_t*) packed.data_ptr(),
        (const half*) scales.data_ptr(),
        (half*) output.data_ptr(),
        num_blocks,
        out_features
    );
    cuda_check(cudaPeekAtLastError());
}
