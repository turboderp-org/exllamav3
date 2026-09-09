#include "moe_unswizzle.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

// Expert weights offloaded to the CPU live in the arena band-swizzled for the VBMI kernels:
// tile (kt, nt) at (nt / 8) * tiles_k * 8 + kt * 8 + nt % 8 instead of kt * tiles_n + nt. When
// experts stream back to the GPU for prefill they are staged and DMA'd verbatim, and this kernel
// restores the native order in VRAM: per (expert, kt, group) the 8 tiles of a group are
// contiguous in both layouts, so each block moves one such run. One read + one write of the
// bytes at VRAM bandwidth, instead of tiles_k * groups scattered memcpys on the stager thread.

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
    const bool swizzled
)
{
    const int groups = tiles_n / 8;
    const int run = blockIdx.x;             // (kt, g) run index, native order
    const int kt = run / groups;
    const int g = run % groups;
    const size_t base = (size_t) blockIdx.y * expert_stride_b + proj_off_b;
    const size_t run_b = (size_t) 8 * tile_b;
    const size_t dst_off = base + ((size_t) kt * tiles_n + (size_t) g * 8) * tile_b;
    const size_t src_off = swizzled ? base + ((size_t) g * tiles_k + kt) * run_b : dst_off;
    const uint4* s = reinterpret_cast<const uint4*>(src + src_off);
    uint4* d = reinterpret_cast<uint4*>(dst + dst_off);
    for (int i = threadIdx.x; i < (int) (run_b / 16); i += NUM_THREADS)
        d[i] = s[i];
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
    int64_t bits,
    bool swizzled
)
{
    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "moe_unswizzle: tensors must share a CUDA device");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "moe_unswizzle: tensors must be contiguous");
    TORCH_CHECK(tiles_n % 8 == 0, "moe_unswizzle: tiles_n must be a multiple of 8");
    TORCH_CHECK(bits >= 1 && bits <= 8, "moe_unswizzle: bits out of range");
    const int tile_b = (int) bits * 32;
    const int64_t proj_b = tiles_k * tiles_n * tile_b;
    const int64_t need = (num_experts - 1) * expert_stride_b + proj_off_b + proj_b;
    TORCH_CHECK(num_experts >= 1 && need <= (int64_t) src.numel() * src.element_size()
                && need <= (int64_t) dst.numel() * dst.element_size(), "moe_unswizzle: batch exceeds the buffers");
    TORCH_CHECK((expert_stride_b | proj_off_b) % 16 == 0, "moe_unswizzle: offsets must be 16-byte aligned");
    dim3 grid((unsigned) (tiles_k * (tiles_n / 8)), (unsigned) num_experts);
    moe_unswizzle_kernel<<<grid, NUM_THREADS, 0, stream>>>(
        (const uint8_t*) src.data_ptr(), (uint8_t*) dst.data_ptr(),
        (size_t) expert_stride_b, (size_t) proj_off_b, (int) tiles_k, (int) tiles_n, tile_b, swizzled);
    cuda_check(cudaPeekAtLastError());
}
