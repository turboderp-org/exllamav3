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

__device__ __forceinline__ void moe_unswizzle_run
(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const size_t base,
    const int run,
    const int tiles_k,
    const int tiles_n,
    const int tile_b,
    const int group
)
{
    const int groups = tiles_n / 8;
    const int kt = run / groups;
    const int g = run % groups;
    const size_t run_b = (size_t) 8 * tile_b;
    const size_t dst_off = base + ((size_t) kt * tiles_n + (size_t) g * 8) * tile_b;
    uint4* d = reinterpret_cast<uint4*>(dst + dst_off);
    if (group == 2)
    {
        for (int i = threadIdx.x; i < (int) (run_b / 16); i += NUM_THREADS)
        {
            const int member_bytes = i * 16;
            const int nt = g * 8 + member_bytes / tile_b;
            const int in_tile = member_bytes % tile_b;
            const size_t src_tile = ((size_t) (nt / group) * tiles_k + kt) * group + nt % group;
            d[i] = *reinterpret_cast<const uint4*>(src + base + src_tile * tile_b + in_tile);
        }
    }
    else
    {
        const size_t src_off = group == 8 ? base + ((size_t) g * tiles_k + kt) * run_b : dst_off;
        const uint4* s = reinterpret_cast<const uint4*>(src + src_off);
        for (int i = threadIdx.x; i < (int) (run_b / 16); i += NUM_THREADS) d[i] = s[i];
    }
}

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
    moe_unswizzle_run(src, dst, (size_t) blockIdx.y * expert_stride_b + proj_off_b,
                      (int) blockIdx.x, tiles_k, tiles_n, tile_b, group);
}

struct MoeUnswizzleProjection
{
    size_t offset_b;
    int tiles_k, tiles_n, tile_b, group;
    int end_run;
};

struct MoeUnswizzleBatch
{
    MoeUnswizzleProjection projections[3];
};

__global__ __launch_bounds__(NUM_THREADS)
void moe_unswizzle_batch_kernel
(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const size_t expert_stride_b,
    const MoeUnswizzleBatch batch
)
{
    const int run = (int) blockIdx.x;
    // Every thread in the CTA selects the same projection and relative native run.
    const bool first = run < batch.projections[0].end_run;
    const bool second = run < batch.projections[1].end_run;
    const MoeUnswizzleProjection proj = first ? batch.projections[0] :
                                         second ? batch.projections[1] : batch.projections[2];
    const int first_run = first ? 0 : second ? batch.projections[0].end_run : batch.projections[1].end_run;
    moe_unswizzle_run(src, dst, (size_t) blockIdx.y * expert_stride_b + proj.offset_b,
                      run - first_run, proj.tiles_k, proj.tiles_n, proj.tile_b, proj.group);
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

void moe_unswizzle_trellis_batch
(
    const at::Tensor& src,
    const at::Tensor& dst,
    int64_t num_experts,
    int64_t expert_stride_b,
    const std::array<std::array<int64_t, 5>, 3>& projections
)
{
    TORCH_CHECK(src.is_cuda() && dst.is_cuda() && src.device() == dst.device(), "moe_unswizzle: tensors must share a CUDA device");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "moe_unswizzle: tensors must be contiguous");
    TORCH_CHECK(num_experts >= 1 && expert_stride_b > 0, "moe_unswizzle: invalid batch shape");
    TORCH_CHECK(expert_stride_b % 16 == 0, "moe_unswizzle: offsets must be 16-byte aligned");

    MoeUnswizzleBatch batch = {};
    int64_t runs = 0, end_b = 0;
    for (int p = 0; p < 3; ++p)
    {
        const auto& desc = projections[p];
        const int64_t off = desc[0], tk = desc[1], tn = desc[2], tb = desc[3], group = desc[4];
        if (tk != 0)
        {
            TORCH_CHECK(tk > 0 && tn > 0 && tn % 8 == 0,
                        "moe_unswizzle: invalid projection shape");
            TORCH_CHECK(group == 0 || group == 2 || group == 8, "moe_unswizzle: group must be 0, 2 or 8");
            TORCH_CHECK(tb >= 32 && tb <= 256, "moe_unswizzle: invalid tile bytes");
            bits_from_K((float) tb / 32.0f);
            TORCH_CHECK(off >= 0 && off <= expert_stride_b && off % 16 == 0,
                        "moe_unswizzle: offsets must be within the expert and 16-byte aligned");
            TORCH_CHECK(tk <= (expert_stride_b - off) / tb / tn, "moe_unswizzle: projection exceeds expert stride");
            const int64_t proj_end = off + tk * tn * tb;
            if (proj_end > end_b) end_b = proj_end;
            runs += tk * (tn / 8);
            batch.projections[p] = {(size_t) off, (int) tk, (int) tn, (int) tb, (int) group, (int) runs};
        }
        else batch.projections[p].end_run = (int) runs;
    }
    const int64_t src_b = src.numel() * src.element_size();
    const int64_t dst_b = dst.numel() * dst.element_size();
    TORCH_CHECK(end_b <= src_b && end_b <= dst_b &&
                num_experts - 1 <= (src_b - end_b) / expert_stride_b &&
                num_experts - 1 <= (dst_b - end_b) / expert_stride_b,
                "moe_unswizzle: batch exceeds the buffers");
    if (runs == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid((unsigned) runs, (unsigned) num_experts);
    moe_unswizzle_batch_kernel<<<grid, NUM_THREADS, 0, stream>>>(
        (const uint8_t*) src.data_ptr(), (uint8_t*) dst.data_ptr(), (size_t) expert_stride_b, batch);
    cuda_check(cudaPeekAtLastError());
}
