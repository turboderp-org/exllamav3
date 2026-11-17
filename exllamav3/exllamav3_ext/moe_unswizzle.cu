#include "moe_unswizzle.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/bits_k.cuh"
#include "cpu/moe_handoff.h"

// Expert weights offloaded to the CPU live in the arena in a packed trellis layout for the
// banded CPU kernels (exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout): tile (kt, nt)
// at (nt / g) * tiles_k * g + kt * g + nt % g instead of kt * tiles_n + nt, and on the AVX2
// tier (g = 2) each integer tile's dwords planar-repacked (native dword w at
// 8 * (w % bits) + w / bits). Fractional tiles instead have Q = 2*bits+1 halfword planes:
// native halfword h is at 8 * ((h ^ 1) % Q) + (h ^ 1) / Q, without a byte swap.
// When experts stream back to the GPU for prefill they are staged and DMA'd verbatim, and this
// kernel restores the native order in VRAM, which is the only order the GEMM and reconstruct
// kernels read. Per (expert, kt, native 8-tile run) the run is one contiguous block in the
// native layout and g | 8/g contiguous sub-runs of g tiles in the packed one, so each block
// moves its run at VRAM bandwidth -- one read + one write, instead of tiles_k * groups
// scattered memcpys on the stager thread. group = 0 is a plain copy.

#define NUM_THREADS 128

// One packed sub-run (group tiles, sub_b bytes) back to native order: a plain copy, or the planar
// dword / fractional halfword inverse within each tile
template <int Q>
__device__ __forceinline__ void restore_sub_run
(
    const uint8_t* __restrict__ src,
    uint4* __restrict__ sd,
    const size_t sub_b,
    const int tile_b,
    const int planar
)
{
    if (!planar)
    {
        const uint4* ss = reinterpret_cast<const uint4*>(src);
        for (int i = threadIdx.x; i < (int) (sub_b / 16); i += NUM_THREADS)
            sd[i] = ss[i];
        return;
    }
    if (tile_b % 32)
    {
        // Fractional planar inverse on dwords: native dword w holds halfwords 2w (low) and 2w + 1 (high),
        // and native halfword h sits at packed halfword 8 * ((h ^ 1) % Q) + (h ^ 1) / Q, so the low half
        // comes from chronological halfword 2w + 1 and the high half from 2w. Each source halfword is read
        // as the dword containing it and the two are merged with one byte permute: dword traffic only
        const int q = Q ? Q : tile_b / 16;
        const int wd = tile_b / 4;                  // dwords per tile (4 * q)
        const uint32_t* ss = reinterpret_cast<const uint32_t*>(src);
        uint32_t* dd = reinterpret_cast<uint32_t*>(sd);
        for (int i = threadIdx.x; i < (int) (sub_b / 4); i += NUM_THREADS)
        {
            const int w = i % wd;
            const int tile0 = i - w;
            const int hl = 2 * w + 1, hh = 2 * w;
            const int pl = 8 * (hl % q) + hl / q;   // packed halfword holding the low half
            const int ph = 8 * (hh % q) + hh / q;   // ... and the high half
            const uint32_t a = ss[tile0 + (pl >> 1)];
            const uint32_t b = ss[tile0 + (ph >> 1)];
            const int sel = ((pl & 1) ? 0x0032 : 0x0010) | ((ph & 1) ? 0x7600 : 0x5400);
            dd[i] = __byte_perm(a, b, sel);
        }
        return;
    }
    // Planar inverse: native dword w of a tile is stored at 8 * (w % bits) + w / bits
    // (integer rates only, tile_b = 32 * bits; per-dword traffic inside one tile)
    const int bits = tile_b / 32;
    const int wd = tile_b / 4;                      // dwords per tile
    const uint32_t* ss = reinterpret_cast<const uint32_t*>(src);
    uint32_t* dd = reinterpret_cast<uint32_t*>(sd);
    for (int i = threadIdx.x; i < (int) (sub_b / 4); i += NUM_THREADS)
    {
        const int w = i % wd;
        dd[i] = ss[(i - w) + 8 * (w % bits) + w / bits];
    }
}

// Q = 0 keeps the runtime layout; Q = 3/5/7 specializes fractional planar group 2.
template <int Q = 0>
__global__ __launch_bounds__(NUM_THREADS)
void moe_unswizzle_kernel
(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const size_t expert_stride_b,
    const size_t proj_off_b,
    const int tiles_k,
    const int tiles_n,
    const int runtime_tile_b,
    const int runtime_group,      // 0 (native) | 2 | 8
    const int runtime_planar      // 1: also invert planar dwords or fractional halfwords
)
{
    const int tile_b = Q ? 16 * Q : runtime_tile_b;
    const int group = Q ? 2 : runtime_group;
    const int planar = Q ? 1 : runtime_planar;
    const int runs = tiles_n / 8;
    const int kt = blockIdx.x / runs;
    const int r = blockIdx.x % runs;            // native 8-tile run index
    const size_t base = (size_t) blockIdx.y * expert_stride_b + proj_off_b;
    const size_t run_off = base + ((size_t) kt * tiles_n + (size_t) r * 8) * tile_b;
    const size_t run_b = (size_t) 8 * tile_b;
    uint4* d = reinterpret_cast<uint4*>(dst + run_off);
    if (!group)
    {
        restore_sub_run<Q>(src + run_off, d, run_b, tile_b, 0);
        return;
    }
    const size_t sub_b = (size_t) group * tile_b;
    for (int s = 0; s < 8 / group; ++s)
    {
        const size_t sub_src = base + ((size_t) (r * (8 / group) + s) * tiles_k + kt) * sub_b;
        restore_sub_run<Q>(src + sub_src, d + (size_t) s * (sub_b / 16), sub_b, tile_b, planar);
    }
}

// Multi-NUMA N-split: the batch holds each worker's slice of every expert, packed as that worker's
// CPU tier stores it (a [k/16, n/16 slice] or [k/16 slice, n/16] trellis in the same layout), worker
// w's slice of expert e at w_off[w] + e * w_stride[w] + w_proj[w]. Slices cut the projection at
// split[w] (tiles along n, or along k with sliced_k), at multiples of 8 tiles, so every native
// 8-tile run of the full projection lies in one worker's slice; the kernel writes the full native
// projection into dst exactly as moe_unswizzle_kernel does for one worker's bytes
struct NSplitSrc
{
    int workers;
    int sliced_k;
    int split[MOE_MAX_WORKERS + 1];
    int64_t w_off[MOE_MAX_WORKERS];
    int64_t w_stride[MOE_MAX_WORKERS];
    int64_t w_proj[MOE_MAX_WORKERS];
};

template <int Q = 0>
__global__ __launch_bounds__(NUM_THREADS)
void moe_unswizzle_nsplit_kernel
(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const size_t expert_stride_b,
    const size_t proj_off_b,
    const int tiles_k,
    const int tiles_n,
    const int runtime_tile_b,
    const int runtime_group,
    const int runtime_planar,
    const NSplitSrc ns
)
{
    const int tile_b = Q ? 16 * Q : runtime_tile_b;
    const int group = Q ? 2 : runtime_group;
    const int planar = Q ? 1 : runtime_planar;
    const int runs = tiles_n / 8;
    const int kt = blockIdx.x / runs;
    const int r = blockIdx.x % runs;
    const int e = blockIdx.y;
    const size_t run_b = (size_t) 8 * tile_b;
    uint4* d = reinterpret_cast<uint4*>(dst + (size_t) e * expert_stride_b + proj_off_b
                                        + ((size_t) kt * tiles_n + (size_t) r * 8) * tile_b);
    // The worker holding this run, and the run in that worker's slice
    const int t = ns.sliced_k ? kt : r * 8;
    int w = 0;
    while (w + 1 < ns.workers && t >= ns.split[w + 1]) ++w;
    const int kt_l = ns.sliced_k ? kt - ns.split[w] : kt;
    const int nt_l = ns.sliced_k ? r * 8 : r * 8 - ns.split[w];
    const int tk_w = ns.sliced_k ? ns.split[w + 1] - ns.split[w] : tiles_k;
    const int tn_w = ns.sliced_k ? tiles_n : ns.split[w + 1] - ns.split[w];
    const uint8_t* sbase = src + ns.w_off[w] + (int64_t) e * ns.w_stride[w] + ns.w_proj[w];
    if (!group)
    {
        restore_sub_run<Q>(sbase + ((size_t) kt_l * tn_w + nt_l) * tile_b, d, run_b, tile_b, 0);
        return;
    }
    const size_t sub_b = (size_t) group * tile_b;
    for (int s = 0; s < 8 / group; ++s)
    {
        const size_t sub_src = ((size_t) (nt_l / group + s) * tk_w + kt_l) * sub_b;
        restore_sub_run<Q>(sbase + sub_src, d + (size_t) s * (sub_b / 16), sub_b, tile_b, planar);
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
    int64_t planar                // 1: planar dword / fractional halfword order to invert
)
{
    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "moe_unswizzle: tensors must share a CUDA device");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "moe_unswizzle: tensors must be contiguous");
    TORCH_CHECK(tiles_n % 8 == 0, "moe_unswizzle: tiles_n must be a multiple of 8");
    TORCH_CHECK(group == 0 || group == 2 || group == 8, "moe_unswizzle: group must be 0, 2 or 8");
    TORCH_CHECK(planar == 0 || planar == 1, "moe_unswizzle: planar must be 0 or 1");
    TORCH_CHECK(!planar || group, "moe_unswizzle: planar requires a swizzled projection");
    TORCH_CHECK(num_experts >= 0 && tiles_k >= 0 && tiles_n >= 0 && expert_stride_b >= 0 && proj_off_b >= 0,
                "moe_unswizzle: negative size or offset");
    const BitsK bk = bits_from_K((float) K);
    const int tile_b = bk.bits * 32 + (bk.half ? 16 : 0);
    TORCH_CHECK(!planar || !bk.half || group == 2,
                "moe_unswizzle: fractional planar layout requires group 2");
    TORCH_CHECK((expert_stride_b | proj_off_b) % 16 == 0, "moe_unswizzle: offsets must be 16-byte aligned");

    // No experts or no tiles: nothing to copy
    if (!num_experts || !tiles_k || !tiles_n) return;

    const int64_t proj_b = tiles_k * tiles_n * tile_b;
    const int64_t need = (num_experts - 1) * expert_stride_b + proj_off_b + proj_b;
    TORCH_CHECK(need <= (int64_t) src.numel() * src.element_size()
                && need <= (int64_t) dst.numel() * dst.element_size(), "moe_unswizzle: batch exceeds the buffers");
    dim3 grid((unsigned) (tiles_k * (tiles_n / 8)), (unsigned) num_experts);
    auto kernel = moe_unswizzle_kernel<>;
    if (planar && bk.half)
    {
        if      (bk.bits == 1) kernel = moe_unswizzle_kernel<3>;
        else if (bk.bits == 2) kernel = moe_unswizzle_kernel<5>;
        else if (bk.bits == 3) kernel = moe_unswizzle_kernel<7>;
    }
    kernel<<<grid, NUM_THREADS, 0, stream>>>(
        (const uint8_t*) src.data_ptr(), (uint8_t*) dst.data_ptr(),
        (size_t) expert_stride_b, (size_t) proj_off_b, (int) tiles_k, (int) tiles_n, tile_b,
        (int) group, (int) planar);
    cuda_check(cudaPeekAtLastError());
}

void moe_unswizzle_trellis_nsplit
(
    const at::Tensor& src,
    const at::Tensor& dst,
    int64_t num_experts,
    int64_t expert_stride_b,
    int64_t proj_off_b,
    int64_t tiles_k,
    int64_t tiles_n,
    double K,
    int64_t group,
    int64_t planar,
    bool sliced_k,
    const std::vector<int64_t>& split,
    const std::vector<int64_t>& w_off,
    const std::vector<int64_t>& w_stride,
    const std::vector<int64_t>& w_proj
)
{
    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "moe_unswizzle_nsplit: tensors must share a CUDA device");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "moe_unswizzle_nsplit: tensors must be contiguous");
    TORCH_CHECK(tiles_n % 8 == 0, "moe_unswizzle_nsplit: tiles_n must be a multiple of 8");
    TORCH_CHECK(group == 0 || group == 2 || group == 8, "moe_unswizzle_nsplit: group must be 0, 2 or 8");
    TORCH_CHECK(!planar || group, "moe_unswizzle_nsplit: planar requires a swizzled projection");
    const int workers = (int) w_off.size();
    TORCH_CHECK(workers >= 1 && workers <= MOE_MAX_WORKERS && (int) split.size() == workers + 1
                && (int) w_stride.size() == workers && (int) w_proj.size() == workers,
                "moe_unswizzle_nsplit: bad worker tables");
    TORCH_CHECK(split.front() == 0 && split.back() == (sliced_k ? tiles_k : tiles_n), "moe_unswizzle_nsplit: split must cover the projection");
    const BitsK bk = bits_from_K((float) K);
    const int tile_b = bk.bits * 32 + (bk.half ? 16 : 0);
    TORCH_CHECK(!planar || !bk.half || group == 2, "moe_unswizzle_nsplit: fractional planar layout requires group 2");
    NSplitSrc ns {};
    ns.workers = workers;
    ns.sliced_k = sliced_k ? 1 : 0;
    for (int w = 0; w <= workers; ++w)
    {
        ns.split[w] = (int) split[w];
        TORCH_CHECK(split[w] % 8 == 0 && (w == 0 || split[w] > split[w - 1]), "moe_unswizzle_nsplit: split must ascend in 8-tile steps");
    }
    const int64_t src_b = (int64_t) src.numel() * src.element_size();
    for (int w = 0; w < workers; ++w)
    {
        ns.w_off[w] = w_off[w];
        ns.w_stride[w] = w_stride[w];
        ns.w_proj[w] = w_proj[w];
        TORCH_CHECK((w_off[w] | w_stride[w] | w_proj[w]) % 16 == 0, "moe_unswizzle_nsplit: offsets must be 16-byte aligned");
        const int64_t span = (int64_t) (split[w + 1] - split[w]) * (sliced_k ? tiles_n : tiles_k) * tile_b;
        TORCH_CHECK(num_experts == 0 || w_off[w] + (num_experts - 1) * w_stride[w] + w_proj[w] + span <= src_b,
                    "moe_unswizzle_nsplit: batch exceeds the source buffer");
    }
    TORCH_CHECK((expert_stride_b | proj_off_b) % 16 == 0, "moe_unswizzle_nsplit: offsets must be 16-byte aligned");
    if (!num_experts || !tiles_k || !tiles_n) return;
    const int64_t need = (num_experts - 1) * expert_stride_b + proj_off_b + tiles_k * tiles_n * tile_b;
    TORCH_CHECK(need <= (int64_t) dst.numel() * dst.element_size(), "moe_unswizzle_nsplit: batch exceeds the destination");
    dim3 grid((unsigned) (tiles_k * (tiles_n / 8)), (unsigned) num_experts);
    auto kernel = moe_unswizzle_nsplit_kernel<>;
    if (planar && bk.half)
    {
        if      (bk.bits == 1) kernel = moe_unswizzle_nsplit_kernel<3>;
        else if (bk.bits == 2) kernel = moe_unswizzle_nsplit_kernel<5>;
        else if (bk.bits == 3) kernel = moe_unswizzle_nsplit_kernel<7>;
    }
    kernel<<<grid, NUM_THREADS, 0, stream>>>(
        (const uint8_t*) src.data_ptr(), (uint8_t*) dst.data_ptr(),
        (size_t) expert_stride_b, (size_t) proj_off_b, (int) tiles_k, (int) tiles_n, tile_b,
        (int) group, (int) planar, ns);
    cuda_check(cudaPeekAtLastError());
}
