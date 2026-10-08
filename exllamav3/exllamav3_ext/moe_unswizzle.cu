#include "moe_unswizzle.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/bits_k.cuh"

// Expert weights offloaded to the CPU live in the arena in a packed trellis layout for the
// banded CPU kernels (exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout): tile (kt, nt)
// at (nt / g) * tiles_k * g + kt * g + nt % g instead of kt * tiles_n + nt, and on the AVX2
// tier (g = 2) each tile's dwords planar-repacked (native dword w at 8 * (w % bits) + w / bits).
// When experts stream back to the GPU for prefill they are staged and DMA'd verbatim, and this
// kernel restores the native order in VRAM for the HIP build (the CUDA kernels read the packed
// bytes directly). Per (expert, kt, native 8-tile run) the run is one contiguous block in the
// native layout and g | 8/g contiguous sub-runs of g tiles in the packed one, so each block
// moves its run at VRAM bandwidth -- one read + one write, instead of tiles_k * groups
// scattered memcpys on the stager thread. group = 0 is a plain copy.

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
    const int runs = tiles_n / 8;
    const int kt = blockIdx.x / runs;
    const int r = blockIdx.x % runs;            // native 8-tile run index
    const size_t base = (size_t) blockIdx.y * expert_stride_b + proj_off_b;
    const size_t run_off = base + ((size_t) kt * tiles_n + (size_t) r * 8) * tile_b;
    const size_t run_b = (size_t) 8 * tile_b;
    uint4* d = reinterpret_cast<uint4*>(dst + run_off);
    if (!group)
    {
        const uint4* s = reinterpret_cast<const uint4*>(src + run_off);
        for (int i = threadIdx.x; i < (int) (run_b / 16); i += NUM_THREADS)
            d[i] = s[i];
        return;
    }
    const size_t sub_b = (size_t) group * tile_b;
    for (int s = 0; s < 8 / group; ++s)
    {
        const size_t sub_src = base + ((size_t) (r * (8 / group) + s) * tiles_k + kt) * sub_b;
        uint4* sd = d + (size_t) s * (sub_b / 16);
        if (!planar)
        {
            const uint4* ss = reinterpret_cast<const uint4*>(src + sub_src);
            for (int i = threadIdx.x; i < (int) (sub_b / 16); i += NUM_THREADS)
                sd[i] = ss[i];
            continue;
        }
        // Planar inverse: native dword w of a tile is stored at 8 * (w % bits) + w / bits
        // (integer rates only, tile_b = 32 * bits; per-dword traffic inside one tile)
        const int bits = tile_b / 32;
        const int wd = tile_b / 4;                      // dwords per tile
        const uint32_t* ss = reinterpret_cast<const uint32_t*>(src + sub_src);
        uint32_t* dd = reinterpret_cast<uint32_t*>(sd);
        for (int i = threadIdx.x; i < (int) (sub_b / 4); i += NUM_THREADS)
        {
            const int w = i % wd;
            dd[i] = ss[(i - w) + 8 * (w % bits) + w / bits];
        }
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
    int64_t group,                // packed tile group of this projection (0/2/8)
    int64_t planar                // 1: planar dword order to invert
)
{
    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "moe_unswizzle: tensors must share a CUDA device");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "moe_unswizzle: tensors must be contiguous");
    TORCH_CHECK(tiles_n % 8 == 0, "moe_unswizzle: tiles_n must be a multiple of 8");
    TORCH_CHECK(group == 0 || group == 2 || group == 8, "moe_unswizzle: group must be 0, 2 or 8");
    TORCH_CHECK(!planar || group, "moe_unswizzle: planar requires a swizzled projection");
    TORCH_CHECK(num_experts >= 0 && tiles_k >= 0 && tiles_n >= 0 && expert_stride_b >= 0 && proj_off_b >= 0,
                "moe_unswizzle: negative size or offset");
    const BitsK bk = bits_from_K((float) K);
    const int tile_b = bk.bits * 32 + (bk.half ? 16 : 0);
    TORCH_CHECK(!planar || !bk.half, "moe_unswizzle: planar is undefined for half-integer rates");
    TORCH_CHECK((expert_stride_b | proj_off_b) % 16 == 0, "moe_unswizzle: offsets must be 16-byte aligned");

    // No experts or no tiles: nothing to copy
    if (!num_experts || !tiles_k || !tiles_n) return;

    const int64_t proj_b = tiles_k * tiles_n * tile_b;
    const int64_t need = (num_experts - 1) * expert_stride_b + proj_off_b + proj_b;
    TORCH_CHECK(need <= (int64_t) src.numel() * src.element_size()
                && need <= (int64_t) dst.numel() * dst.element_size(), "moe_unswizzle: batch exceeds the buffers");
    dim3 grid((unsigned) (tiles_k * (tiles_n / 8)), (unsigned) num_experts);
    moe_unswizzle_kernel<<<grid, NUM_THREADS, 0, stream>>>(
        (const uint8_t*) src.data_ptr(), (uint8_t*) dst.data_ptr(),
        (size_t) expert_stride_b, (size_t) proj_off_b, (int) tiles_k, (int) tiles_n, tile_b,
        (int) group, (int) planar);
    cuda_check(cudaPeekAtLastError());
}