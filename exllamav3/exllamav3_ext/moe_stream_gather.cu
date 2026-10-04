#include "moe_stream_gather.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

// Streamed single-row decode for CPU-offloaded MoE layers (model/moe_stream_decode.py): each
// selected expert's contiguous [gate | up | down] trellis block is read straight out of the
// worker's pinned arena (mapped into the device address space, EXL3_MOE_PINNED_ARENA) into a
// VRAM staging slot, 16 bytes per thread, grid-stride over the block. The PCIe reads are the
// whole cost, so blockIdx.x spreads each block's requests over many SMs; blockIdx.y is the
// staging slot (= position in the top-k). The optional aux gather copies the selected experts'
// entries of the per-expert pointer tables (suh/svh) into a per-slot table in the same launch,
// so the caller needs no separate index_select for the fused kernel's tables.

#define GATHER_THREADS 256
#define GATHER_BLOCKS_PER_EXPERT 64

__global__ __launch_bounds__(GATHER_THREADS)
void moe_stream_gather_kernel
(
    uint4* __restrict__ dst,
    const int64_t* __restrict__ chunk_base,
    const int32_t* __restrict__ blk_chunk,
    const int64_t* __restrict__ blk_off,
    const int64_t* __restrict__ sel,
    const int64_t n16,                              // uint4s per expert block
    const int64_t* __restrict__ aux_ptrs,
    int64_t* __restrict__ aux_out,
    const int aux_rows,
    const int num_experts,
    const int num_slots
)
{
    const int slot = blockIdx.y;
    const int64_t e = sel[slot];
    if (e < 0 || e >= num_experts) return;
    if (blockIdx.x == 0 && threadIdx.x < aux_rows)
        aux_out[threadIdx.x * num_slots + slot] = aux_ptrs[threadIdx.x * num_experts + e];
    const uint4* src = reinterpret_cast<const uint4*>(chunk_base[blk_chunk[e]] + blk_off[e]);
    uint4* d = dst + (int64_t) slot * n16;
    for (int64_t i = (int64_t) blockIdx.x * GATHER_THREADS + threadIdx.x; i < n16;
         i += (int64_t) gridDim.x * GATHER_THREADS)
        d[i] = src[i];
}

void moe_stream_gather
(
    const at::Tensor& dst,
    const at::Tensor& chunk_base,
    const at::Tensor& blk_chunk,
    const at::Tensor& blk_off,
    const at::Tensor& sel,
    int64_t expert_bytes,
    const c10::optional<at::Tensor>& aux_ptrs,
    const c10::optional<at::Tensor>& aux_out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(dst.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "moe_stream_gather: dst must be a contiguous CUDA tensor");
    TORCH_CHECK(expert_bytes > 0 && expert_bytes % 16 == 0, "moe_stream_gather: expert_bytes must be a positive multiple of 16");
    TORCH_CHECK(chunk_base.is_cuda() && chunk_base.scalar_type() == at::kLong && chunk_base.is_contiguous(),
                "moe_stream_gather: chunk_base must be a contiguous int64 CUDA tensor");
    TORCH_CHECK(blk_chunk.is_cuda() && blk_chunk.scalar_type() == at::kInt && blk_chunk.is_contiguous(),
                "moe_stream_gather: blk_chunk must be a contiguous int32 CUDA tensor");
    TORCH_CHECK(blk_off.is_cuda() && blk_off.scalar_type() == at::kLong && blk_off.is_contiguous()
                && blk_off.numel() == blk_chunk.numel(), "moe_stream_gather: blk_off must be a contiguous int64 CUDA tensor, one entry per expert");
    TORCH_CHECK(sel.is_cuda() && sel.scalar_type() == at::kLong && sel.is_contiguous() && sel.dim() == 1,
                "moe_stream_gather: sel must be a contiguous 1D int64 CUDA tensor");
    const int64_t slots = sel.numel();
    TORCH_CHECK(slots >= 1 && slots <= 65535, "moe_stream_gather: slot count out of range");
    TORCH_CHECK((int64_t) dst.numel() * dst.element_size() >= slots * expert_bytes, "moe_stream_gather: dst too small");

    const int64_t* _aux_ptrs = nullptr;
    int64_t* _aux_out = nullptr;
    int aux_rows = 0;
    if (aux_ptrs.has_value())
    {
        TORCH_CHECK(aux_out.has_value(), "moe_stream_gather: aux_ptrs needs aux_out");
        const at::Tensor& ap = aux_ptrs.value();
        const at::Tensor& ao = aux_out.value();
        TORCH_CHECK(ap.is_cuda() && ap.scalar_type() == at::kLong && ap.is_contiguous() && ap.dim() == 2
                    && ap.size(1) == blk_chunk.numel() && ap.size(0) <= GATHER_THREADS,
                    "moe_stream_gather: aux_ptrs must be a contiguous int64 [rows, num_experts] CUDA tensor");
        TORCH_CHECK(ao.is_cuda() && ao.scalar_type() == at::kLong && ao.is_contiguous() && ao.dim() == 2
                    && ao.size(0) == ap.size(0) && ao.size(1) == slots,
                    "moe_stream_gather: aux_out must be a contiguous int64 [rows, slots] CUDA tensor");
        _aux_ptrs = ap.data_ptr<int64_t>();
        _aux_out = ao.data_ptr<int64_t>();
        aux_rows = (int) ap.size(0);
    }

    dim3 grid(GATHER_BLOCKS_PER_EXPERT, (unsigned) slots);
    moe_stream_gather_kernel<<<grid, GATHER_THREADS, 0, stream>>>(
        (uint4*) dst.data_ptr(), chunk_base.data_ptr<int64_t>(), blk_chunk.data_ptr<int32_t>(),
        blk_off.data_ptr<int64_t>(), sel.data_ptr<int64_t>(), expert_bytes / 16,
        _aux_ptrs, _aux_out, aux_rows, (int) blk_chunk.numel(), (int) slots);
    cuda_check(cudaPeekAtLastError());
}
