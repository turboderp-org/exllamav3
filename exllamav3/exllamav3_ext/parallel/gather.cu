#include <cuda_fp16.h>
#include "gather.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include "ll.cuh"
#include "barrier_inner.cuh"

#define NUM_THREADS 1024
#define BATCH_STAGE 2
#define STAGE_SIZE (NUM_THREADS * 16)

struct Offsets {
    int v[MAX_DEVICES + 1];
    __host__ __device__ int& operator[](int i)       { return v[i]; }
    __host__ __device__ int  operator[](int i) const { return v[i]; }
};

__global__ __launch_bounds__(NUM_THREADS)
void pg_gather_kernel
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    const int this_device,
    const int out_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ out_data_ptr,
    const Offsets __grid_constant__ all_offsets,
    const int batch,
    uint8_t* __restrict__ shbuf_ptr,
    const size_t data_size,
    const size_t shbuf_size,
    uint32_t *abort_flag
)
{
    int t = threadIdx.x;
    auto grid = cg::this_grid();

    __shared__ uint32_t r;
    uint8_t* data_end = data_ptr + data_size;

    // Divide shared buffer among ranks
    int num_ranks = __popc(device_mask);
    if (num_ranks <= 1) return;
    int this_rank = __popc(device_mask & ((1 << this_device) - 1));

    const size_t stage_size = blockDim.x * sizeof(uint4);
    size_t rank_shbuf_size = shbuf_size / num_ranks / stage_size * stage_size;
    int num_buf_stages = rank_shbuf_size /stage_size;

    // Our slice
    int ldim = all_offsets[this_device + 1] - all_offsets[this_device];

    // Indexing
    auto shbuf_stage_ptr = [&] (int rank, int stage_idx)
    {
        return shbuf_ptr +
               rank * rank_shbuf_size +
               (stage_idx % num_buf_stages) * stage_size;
    };

    auto data_stage_ptr = [&] (int stage_idx)
    {
        return data_ptr + stage_idx * stage_size;
    };

    // Sync
    auto wait_min_stage = [&] (uint32_t* stage_ptr, int min_stage, uint64_t deadline)
    {
        if (t == 0)
        {
            uint32_t sleep = SYNC_MIN_SLEEP;
            while ((int) ldg_acquire_sys_u32(stage_ptr) < min_stage)
            {
                __nanosleep(sleep);
                if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                else *abort_flag = check_timeout(ctx, deadline, "gather (0)");
                if (*abort_flag) break;
            }
        }
        __syncthreads();
    };

    auto stage_ready = [&] (uint32_t* stage_ptr, int min_stage)
    {
        if (t == 0)
            r = ((int) ldg_acquire_sys_u32(stage_ptr) >= min_stage);
        __syncthreads();
        return r;
    };

    // Consumer
    bool is_consumer = this_device == out_device;
    if (is_consumer)
    {
        // Find source device for this block
        int src_rank = blockIdx.x;
        int src_device = __fns(device_mask, 0, src_rank + 1);

        size_t stride = (size_t) all_offsets[MAX_DEVICES];

        // Source is self, copy in gmem
        if (src_rank == this_rank)
        {
            size_t bytes_to_copy = data_size;
            size_t dst_offset = (size_t) all_offsets[this_device];
            int num_stages = (int) CEIL_DIVIDE(bytes_to_copy, stage_size);

            for (int stage = 0; stage < num_stages; ++stage)
            {
                uint4* src = (uint4*) data_stage_ptr(stage);
                size_t copy_t = (size_t) stage * stage_size + t * 16;
                if (copy_t < bytes_to_copy)
                {
                    size_t row = copy_t / ldim;
                    size_t col = copy_t % ldim;
                    size_t out_t = row * stride + col;
                    *((uint4*) (out_data_ptr + out_t)) = src[t];
                }
            }
        }

        // Source is remote, stream from other device
        else
        {
            size_t src_ldim = (size_t) (all_offsets[src_device + 1] - all_offsets[src_device]);
            size_t bytes_to_recv = src_ldim * (size_t) batch;
            size_t dst_offset = (size_t) (all_offsets[src_device]);
            int num_stages = (int) CEIL_DIVIDE(bytes_to_recv, stage_size);
            uint32_t sleep = SYNC_MIN_SLEEP;
            uint64_t deadline = sync_deadline();

            int stage_recv = 0;
            while (stage_recv < num_stages)
            {
                if (t == 0)
                    r = (int) ldg_acquire_sys_u32(ctx->gather_stage_produced + src_rank);
                __syncthreads();
                uint32_t stage_ready = r;

                if (stage_recv < stage_ready)
                {
                    while (stage_recv < stage_ready)
                    {
                        sleep = SYNC_MIN_SLEEP;
                        uint4* src = (uint4*) shbuf_stage_ptr(src_rank, stage_recv);
                        size_t recv_t = (size_t) stage_recv * stage_size + t * 16;
                        if (recv_t < bytes_to_recv)
                        {
                            size_t row = recv_t / src_ldim;
                            size_t col = recv_t % src_ldim;
                            size_t out_t = row * stride + col + dst_offset;
                            *((uint4*) (out_data_ptr + out_t)) = src[t];
                        }

                        // Advance
                        stage_recv++;
                    }
                    __syncthreads();

                    if (t == 0)
                    {
                        stg_release_sys_u32(ctx->gather_stage_consumed + src_rank, stage_recv);
                    }
                }
                else
                {
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "gather (1)");
                    if (*abort_flag) break;
                }
            }
        }
    }

    // Producer
    else
    {
        size_t bytes_to_send = data_size;
        int num_stages = (int) CEIL_DIVIDE(bytes_to_send, stage_size);
        bool no_overflow = num_stages < num_buf_stages - 2;
        uint32_t sleep = SYNC_MIN_SLEEP;
        uint64_t deadline = sync_deadline();

        int stage_send = 0;
        while (stage_send < num_stages)
        {
            bool ready_send = stage_send < num_stages;
            if (ready_send && !no_overflow)
                if (!stage_ready(ctx->gather_stage_consumed + this_rank, stage_send - num_buf_stages + 1 + BATCH_STAGE))
                   ready_send = false;

            if (ready_send)
            {
                for (int i = 0; i < BATCH_STAGE && stage_send < num_stages; ++i)
                {
                    sleep = SYNC_MIN_SLEEP;
                    uint4* src = (uint4*) data_stage_ptr(stage_send);
                    uint4* dst = (uint4*) shbuf_stage_ptr(this_rank, stage_send);
                    if (src + t < (uint4*) data_end) dst[t] = src[t];

                    // Advance
                    stage_send++;
                }

                if (t == 0)
                {
                    stg_release_sys_u32(ctx->gather_stage_produced + this_rank, stage_send);
                }
            }
            else
            {
                __nanosleep(sleep);
                if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                else *abort_flag = check_timeout(ctx, deadline, "gather (2)");
                if (*abort_flag) break;
            }
        }

        // Wait for receive to finish and reset
        wait_min_stage(ctx->gather_stage_consumed + this_rank, num_stages, deadline);
        if (t == 0)
        {
            ctx->gather_stage_consumed[this_rank] = 0;
            ctx->gather_stage_produced[this_rank] = 0;
            __threadfence_system();
        }
    }

    grid.sync();
    pg_barrier_inner(ctx, device_mask, this_device, out_device, abort_flag);
}


void pg_gather
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    uint8_t* out_data_ptr = (uint8_t*) OPTPTR(out_tensor);
    uint8_t* shbuf_ptr = (uint8_t*) shbuf;

    size_t esize = tensor.element_size();
    size_t send_size = tensor.numel() * esize;
    size_t send_ldim = tensor.size(-1) * esize;
    TORCH_CHECK(send_ldim % 128 == 0, "send_ldim must be multiple of 128");
    TORCH_CHECK(devices.size() == ldims.size(), "Must have one ldim per active device");
    int batch = out_data_ptr ? out_tensor.value().numel() / out_tensor.value().size(-1)
                             : tensor.numel() / tensor.size(-1);

    Offsets all_offsets = {};
    for (int i = 0; i < MAX_DEVICES + 1; ++i) all_offsets[i] = 0;
    for (int i = 0; i < devices.size(); ++i) all_offsets[devices[i]] = ldims[i] * esize;
    int p = 0;
    for (int i = 0; i < MAX_DEVICES + 1; ++i) { int q = p; p += all_offsets[i]; all_offsets[i] = q; }
    if (out_data_ptr)
        TORCH_CHECK(p == out_tensor.value().size(-1) * esize, "Gather: Output tensor last dimension mismatch");

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);
    int num_ranks = devices.size();
    int num_blocks = 1;
    if (this_device == out_device)
        num_blocks = num_ranks;

    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    void* kernelArgs[] =
    {
        (void*)& ctx,  // Shared, pinned
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& out_device,
        (void*)& data_ptr,
        (void*)& out_data_ptr,
        (void*)& all_offsets,
        (void*)& batch,
        (void*)& shbuf_ptr,
        (void*)& send_size,
        (void*)& shbuf_size,
        (void*)& abort_flag_ptr
    };

    dim3 block_grid(num_blocks);
    dim3 block_dim(NUM_THREADS);

    cudaLaunchCooperativeKernel
    (
        (void*)pg_gather_kernel,
        block_grid,
        block_dim,
        kernelArgs,
        0,
        stream
    );
    cuda_check(cudaPeekAtLastError());
}