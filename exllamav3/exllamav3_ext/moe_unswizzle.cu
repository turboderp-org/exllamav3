#include "moe_unswizzle.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/bits_k.cuh"

// Expert weights offloaded to the CPU live in the arena band-swizzled for the banded CPU
// kernels: tile (kt, nt) at (nt / g) * tiles_k * g + kt * g + nt % g instead of
// kt * tiles_n + nt, with the group g per matrix from exl3_moe_cpu_swizzle_group (8 on the
// AVX-512 tiers, 2 on AVX2). When experts stream back to the GPU for prefill they are staged
// and DMA'd verbatim, and this kernel restores the native order in VRAM: per (expert, kt,
// group) the g tiles of a group are contiguous in both layouts, so each block moves such
// runs -- one read + one write of the bytes at VRAM bandwidth, instead of tiles_k * groups
// scattered memcpys on the stager thread. group = 0 (matrix never swizzled) is a plain copy.
//
// planar = 1 (AVX2 tier only, integer rates; exl3_moe_cpu_planar_layout) additionally means
// each staged tile's dwords are in the planar order: dword q holds native word
// bits*(q % 8) + q / 8. The restore then inverts that per-dword within each tile (4-byte
// accesses inside a tile of 32*bits bytes; the pass stays bandwidth-bound).

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
    const int group,              // 0 (native) | 2 | 8
    const int planar              // 1: also invert the planar dword order (group > 0, integer K)
)
{
    const int groups = tiles_n / 8;             // launch granularity: 8-tile native runs (g | 8)
    const int run = blockIdx.x;
    const int kt = run / groups;
    const int g = run % groups;
    const size_t base = (size_t) blockIdx.y * expert_stride_b + proj_off_b;
    const size_t run_b = (size_t) 8 * tile_b;
    const size_t dst_off = base + ((size_t) kt * tiles_n + (size_t) g * 8) * tile_b;
    if (group)
    {
        // An 8-tile native run spans 8/group swizzled groups: each contiguous (group tiles)
        // but consecutive groups sit tiles_k groups apart in the swizzled order
        const size_t sub_b = (size_t) group * tile_b;
        const uint4* s0 = reinterpret_cast<const uint4*>(
            src + base + ((size_t) g * (8 / group) * tiles_k + kt) * sub_b);
        uint4* d0 = reinterpret_cast<uint4*>(dst + dst_off);
        if (planar)
        {
            // Per-dword inverse of the planar repack within each tile (see file header);
            // integer rates only: tile_b = 32 * bits, dword residue math in unsigned int
            const int bits = tile_b / 32;
            const int wd = 8 * bits;                       // dwords per tile
            #pragma unroll
            for (int sub = 0; sub < 8 / group; ++sub)
            {
                const uint32_t* s = reinterpret_cast<const uint32_t*>(s0 + (size_t) sub * tiles_k * (sub_b / 16));
                uint32_t* d = reinterpret_cast<uint32_t*>(d0 + (size_t) sub * (sub_b / 16));
                for (int i = threadIdx.x; i < (int) (sub_b / 4); i += NUM_THREADS)
                {
                    const int q = i % wd;
                    d[(i / wd) * wd + bits * (q & 7) + (q >> 3)] = s[i];
                }
            }
            return;
        }
        #pragma unroll
        for (int sub = 0; sub < 8 / group; ++sub)
        {
            const uint4* s = s0 + (size_t) sub * tiles_k * (sub_b / 16);
            uint4* d = d0 + (size_t) sub * (sub_b / 16);
            for (int i = threadIdx.x; i < (int) (sub_b / 16); i += NUM_THREADS)
                d[i] = s[i];
        }
        return;
    }
    const uint4* s = reinterpret_cast<const uint4*>(src + dst_off);
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
    double K,
    int64_t group,                // swizzle group of this projection (0/2/8); 0: plain copy
    int64_t planar                // 1: planar dword order to invert (from exl3_moe_cpu_planar_layout)
)
{
    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "moe_unswizzle: tensors must share a CUDA device");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "moe_unswizzle: tensors must be contiguous");
    TORCH_CHECK(tiles_n % 8 == 0, "moe_unswizzle: tiles_n must be a multiple of 8");
    TORCH_CHECK(group == 0 || group == 2 || group == 8, "moe_unswizzle: group must be 0, 2 or 8");
    TORCH_CHECK(!planar || group != 0, "moe_unswizzle: planar requires a swizzled projection");
    const BitsK bk = bits_from_K((float) K);
    const int tile_b = bk.bits * 32 + (bk.half ? 16 : 0);
    TORCH_CHECK(!planar || !bk.half, "moe_unswizzle: planar is undefined for half-integer rates");
    const int64_t proj_b = tiles_k * tiles_n * tile_b;
    const int64_t need = (num_experts - 1) * expert_stride_b + proj_off_b + proj_b;
    TORCH_CHECK(num_experts >= 1 && need <= (int64_t) src.numel() * src.element_size()
                && need <= (int64_t) dst.numel() * dst.element_size(), "moe_unswizzle: batch exceeds the buffers");
    TORCH_CHECK((expert_stride_b | proj_off_b) % 16 == 0, "moe_unswizzle: offsets must be 16-byte aligned");
    dim3 grid((unsigned) (tiles_k * (tiles_n / 8)), (unsigned) num_experts);
    moe_unswizzle_kernel<<<grid, NUM_THREADS, 0, stream>>>(
        (const uint8_t*) src.data_ptr(), (uint8_t*) dst.data_ptr(),
        (size_t) expert_stride_b, (size_t) proj_off_b, (int) tiles_k, (int) tiles_n, tile_b,
        (int) group, (int) planar);
    cuda_check(cudaPeekAtLastError());
}
