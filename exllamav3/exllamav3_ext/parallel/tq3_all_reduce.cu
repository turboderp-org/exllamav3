#include <cuda_fp16.h>
#include "tq3_all_reduce.cuh"
#include "tq3_compress.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include "barrier_inner.cuh"

// ---------------------------------------------------------------------------
// Shared buffer layout for TQ3 all-reduce (all-gather + local-sum pattern)
//
// The pinned shared buffer (shm_b, 16 MB by default) is sliced into
// num_ranks equal slots.  Each slot holds the TQ3-compressed representation
// of one rank's fp16 tensor:
//
//   slot_bytes = num_tq3_blocks * 10
//   10 bytes per block = 4 (bp0) + 4 (bp1) + 2 (scale, fp16)
//
// Slot for rank r starts at:  shbuf_ptr + r * slot_bytes
//
// Algorithm
//   Phase 1 — every thread processes one or more TQ3 blocks from its rank's
//              local fp16 data and stores compressed output to shbuf[this_rank].
//   Phase 2 — __threadfence_system() + cooperative grid.sync() + barrier_inner
//              ensures all GPUs see each other's writes.
//   Phase 3 — every thread accumulates ALL ranks' compressed data for its
//              assigned blocks into a float accumulator, then stores fp16 result
//              back to the data tensor in-place.
//   Phase 4 — second barrier to make results visible before the kernel exits.
// ---------------------------------------------------------------------------

#define TQ3_AR_MAX_THREADS 1024

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(TQ3_AR_MAX_THREADS)
void tq3_all_reduce_kernel
(
    PGContext* __restrict__  ctx,
    const uint32_t           device_mask,
    int                      this_device,
    int                      master_device,
    half* __restrict__       data_ptr,        // fp16 input/output on this GPU
    uint8_t* __restrict__    shbuf_ptr,       // pinned shared ring buffer
    const size_t             num_elements,    // number of fp16 elements
    const size_t             slot_bytes,      // bytes per rank slot in shbuf
    bool                     contribution,    // false → treat local data as zeros
    uint32_t* __restrict__   abort_flag
)
{
    auto grid = cg::this_grid();

    const int num_ranks  = __popc(device_mask);
    const int this_rank  = __popc(device_mask & ((1 << this_device) - 1));

    const size_t num_blocks = (num_elements + TQ3_BLOCK_SIZE - 1) / TQ3_BLOCK_SIZE;
    const size_t block_bytes = 10u;  // bp0(4) + bp1(4) + scale(2)

    // Pointer to this rank's write slot
    uint8_t* my_slot = shbuf_ptr + (size_t)this_rank * slot_bytes;

    // ------------------------------------------------------------------
    // Phase 1: compress local fp16 data → TQ3, write to our shbuf slot
    // ------------------------------------------------------------------
    {
        int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int stride     = (int)(gridDim.x  * blockDim.x);

        for (size_t blk = (size_t)global_tid; blk < num_blocks; blk += (size_t)stride)
        {
            size_t elem_off         = blk * TQ3_BLOCK_SIZE;
            size_t elems_this_block = min((size_t)TQ3_BLOCK_SIZE, num_elements - elem_off);

            half src_buf[TQ3_BLOCK_SIZE];
            #pragma unroll
            for (int i = 0; i < TQ3_BLOCK_SIZE; ++i)
            {
                if ((size_t)i < elems_this_block && contribution)
                    src_buf[i] = data_ptr[elem_off + (size_t)i];
                else
                    src_buf[i] = __float2half(0.0f);
            }

            uint32_t bp0, bp1;
            half     scale;
            tq3_compress_block(src_buf, &bp0, &bp1, &scale);

            uint8_t* dst = my_slot + blk * block_bytes;
            *reinterpret_cast<uint32_t*>(dst + 0) = bp0;
            *reinterpret_cast<uint32_t*>(dst + 4) = bp1;
            *reinterpret_cast<half*>    (dst + 8) = scale;
        }
    }

    // Make writes visible to peer GPUs before the barrier
    __threadfence_system();

    // ------------------------------------------------------------------
    // Phase 2: global barrier — wait for all ranks to finish writing
    // ------------------------------------------------------------------
    grid.sync();
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
    if (*abort_flag) return;

    // ------------------------------------------------------------------
    // Phase 3: accumulate all rank slots → write result in-place
    // ------------------------------------------------------------------
    {
        int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int stride     = (int)(gridDim.x  * blockDim.x);

        for (size_t blk = (size_t)global_tid; blk < num_blocks; blk += (size_t)stride)
        {
            size_t elem_off         = blk * TQ3_BLOCK_SIZE;
            size_t elems_this_block = min((size_t)TQ3_BLOCK_SIZE, num_elements - elem_off);

            // Accumulate in float for numerical accuracy
            float acc[TQ3_BLOCK_SIZE];
            #pragma unroll
            for (int i = 0; i < TQ3_BLOCK_SIZE; ++i) acc[i] = 0.0f;

            for (int r = 0; r < num_ranks; ++r)
            {
                const uint8_t* src = shbuf_ptr + (size_t)r * slot_bytes + blk * block_bytes;

                uint32_t bp0   = *reinterpret_cast<const uint32_t*>(src + 0);
                uint32_t bp1   = *reinterpret_cast<const uint32_t*>(src + 4);
                half     scale = *reinterpret_cast<const half*>    (src + 8);
                float    fscale = __half2float(scale);

                #pragma unroll
                for (int i = 0; i < TQ3_BLOCK_SIZE; ++i)
                {
                    uint32_t mag  = (bp1 >> i) & 1u;
                    uint32_t sign = (bp0 >> i) & 1u;
                    if (mag) acc[i] += sign ? -fscale : fscale;
                }
            }

            // Write results back to the data tensor
            #pragma unroll
            for (int i = 0; i < TQ3_BLOCK_SIZE; ++i)
            {
                if ((size_t)i < elems_this_block)
                    data_ptr[elem_off + (size_t)i] = __float2half(acc[i]);
            }
        }
    }

    // ------------------------------------------------------------------
    // Phase 4: second barrier so all ranks have finished reading before
    //          the caller re-uses or frees the shbuf slot
    // ------------------------------------------------------------------
    __threadfence_system();
    grid.sync();
    pg_barrier_inner(ctx, device_mask, this_device, master_device, abort_flag);
}


// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------
void tq3_all_reduce
(
    const at::Tensor&       data,
    uintptr_t               ctx_ptr,
    std::vector<uintptr_t>  devices,
    int                     this_device,
    int                     master_device,
    uintptr_t               shbuf,
    size_t                  shbuf_size,
    bool                    contribution
)
{
    TORCH_CHECK(data.scalar_type() == at::kHalf,
                "tq3_all_reduce: input tensor must be fp16 (torch.float16)");
    TORCH_CHECK(data.is_contiguous(),
                "tq3_all_reduce: input tensor must be contiguous");
    TORCH_CHECK(data.is_cuda(),
                "tq3_all_reduce: input tensor must be on a CUDA device");

    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    pg_check_timeout(ctx_ptr);
    PGContext* ctx = reinterpret_cast<PGContext*>(ctx_ptr);

    // Build device mask
    uint32_t device_mask = 0;
    for (uintptr_t d : devices) device_mask |= (1u << (int)d);

    int num_ranks = __builtin_popcount(device_mask);
    if (num_ranks <= 1) return;

    size_t num_elements   = (size_t) data.numel();
    size_t num_tq3_blocks = (num_elements + TQ3_BLOCK_SIZE - 1) / TQ3_BLOCK_SIZE;
    size_t slot_bytes     = num_tq3_blocks * 10u;  // 10 bytes per TQ3 block
    size_t total_needed   = slot_bytes * (size_t)num_ranks;

    TORCH_CHECK(total_needed <= shbuf_size,
                "tq3_all_reduce: tensor too large for shared buffer. "
                "Need ", total_needed, " bytes, have ", shbuf_size);

    uint8_t* shbuf_ptr = reinterpret_cast<uint8_t*>(shbuf);

    // Thread count: one thread handles one TQ3 block at a time (strided loop).
    // Cap at TQ3_AR_MAX_THREADS, round up to warp boundary.
    int threads = (int) min((size_t) TQ3_AR_MAX_THREADS, num_tq3_blocks);
    if (threads < 1) threads = 1;
    threads = ((threads + 31) / 32) * 32;
    if (threads > TQ3_AR_MAX_THREADS) threads = TQ3_AR_MAX_THREADS;

    // Single cooperative block per GPU (grid.sync() requires cooperative launch;
    // one block is sufficient — threads loop over all TQ3 blocks internally).
    dim3 grid_dim(1);
    dim3 block_dim(threads);

    // Per-call abort flag — small temporary device tensor
    at::Tensor abort_tensor = torch::zeros(
        {1},
        at::TensorOptions().dtype(torch::kInt32).device(data.device())
    );
    uint32_t* abort_flag_ptr = reinterpret_cast<uint32_t*>(abort_tensor.data_ptr());

    half*    data_dev_ptr = reinterpret_cast<half*>(data.data_ptr());

    void* kernelArgs[] =
    {
        (void*) &ctx,
        (void*) &device_mask,
        (void*) &this_device,
        (void*) &master_device,
        (void*) &data_dev_ptr,
        (void*) &shbuf_ptr,
        (void*) &num_elements,
        (void*) &slot_bytes,
        (void*) &contribution,
        (void*) &abort_flag_ptr
    };

    cudaLaunchCooperativeKernel(
        (void*) tq3_all_reduce_kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        0,
        stream
    );

    cuda_check(cudaPeekAtLastError());
}
