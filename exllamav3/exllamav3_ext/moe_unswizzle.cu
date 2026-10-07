#include "moe_unswizzle.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/bits_k.cuh"

// CPU expert tiles use native order or groups of two/eight output tiles. Streaming
// stages these bytes verbatim; restore native order here with one read and one write.
// Each CTA still moves eight native tiles, gathering four subruns for group2.

#define NUM_THREADS 128

__global__ __launch_bounds__(NUM_THREADS)
void moe_unswizzle_kernel
(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const size_t expert_stride_b,
    const size_t proj_off_b,
    const int tiles_k,
    const int tiles_n,
    const int tile_b,
    const int group
)
{
    const int groups = tiles_n / 8;
    const int run = blockIdx.x;             // (kt, g) run index, native order
    const int kt = run / groups;
    const int g = run % groups;
    const size_t base = (size_t) blockIdx.y * expert_stride_b + proj_off_b;
    const size_t run_b = (size_t) 8 * tile_b;
    const size_t dst_off = base + ((size_t) kt * tiles_n + (size_t) g * 8) * tile_b;
    uint4* d = reinterpret_cast<uint4*>(dst + dst_off);
    for (int i = threadIdx.x; i < (int) (run_b / 16); i += NUM_THREADS)
    {
        const int member_bytes = i * 16;
        const int nt = g * 8 + member_bytes / tile_b;
        const int in_tile = member_bytes % tile_b;
        const size_t src_tile = group
            ? ((size_t) (nt / group) * tiles_k + kt) * group + nt % group
            : (size_t) kt * tiles_n + nt;
        d[i] = *reinterpret_cast<const uint4*>(src + base + src_tile * tile_b + in_tile);
    }
}

void moe_unswizzle_trellis
(
    const at::Tensor& src,
    const at::Tensor& dst,
    int64_t num_experts,
    int64_t expert_stride_b,
    int64_t proj_off_b,
    int64_t tiles_k,
    int64_t tiles_n,
    double K,
    int64_t group
)
{
    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "moe_unswizzle: tensors must share a CUDA device");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "moe_unswizzle: tensors must be contiguous");
    TORCH_CHECK(group == 0 || group == 2 || group == 8, "moe_unswizzle: group must be 0, 2 or 8");
    TORCH_CHECK(tiles_n % 8 == 0, "moe_unswizzle: tiles_n must be a multiple of 8");
    const BitsK bk = bits_from_K((float) K);
    const int tile_b = bk.bits * 32 + (bk.half ? 16 : 0);
    const int64_t proj_b = tiles_k * tiles_n * tile_b;
    const int64_t need = (num_experts - 1) * expert_stride_b + proj_off_b + proj_b;
    TORCH_CHECK(num_experts >= 1 && need <= (int64_t) src.numel() * src.element_size()
                && need <= (int64_t) dst.numel() * dst.element_size(), "moe_unswizzle: batch exceeds the buffers");
    TORCH_CHECK((expert_stride_b | proj_off_b) % 16 == 0, "moe_unswizzle: offsets must be 16-byte aligned");
    dim3 grid((unsigned) (tiles_k * (tiles_n / 8)), (unsigned) num_experts);
    moe_unswizzle_kernel<<<grid, NUM_THREADS, 0, stream>>>(
        (const uint8_t*) src.data_ptr(), (uint8_t*) dst.data_ptr(),
        (size_t) expert_stride_b, (size_t) proj_off_b, (int) tiles_k, (int) tiles_n, tile_b, (int) group);
    cuda_check(cudaPeekAtLastError());
}
