#include <cuda_fp16.h>
#include "broadcast.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"
#include "ll.cuh"
#include "barrier_inner.cuh"

#define NUM_THREADS 1024
#define NUM_THREADS_LL 256

template <bool is_producer>
__global__ __launch_bounds__(NUM_THREADS)
void pg_broadcast_kernel
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    const int this_device,
    const int src_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ shbuf_ptr,
    const size_t data_size,
    const size_t shbuf_size,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;

    const size_t stage_size = BROADCAST_STAGE_SIZE;
    int num_buffered_stages = shbuf_size / stage_size;
    int num_stages = CEIL_DIVIDE(data_size, stage_size);
    int local_stage = 0;

    uint8_t* data_end = data_ptr + data_size;
    uint32_t* broadcast_stages_ptr = &ctx->broadcast_stage_device[0];

    // Producer
    if constexpr (is_producer)
    {
        // Copy one stage to host buffer
        auto produce_next_stage = [&] ()
        {
            uint8_t* src_ptr = data_ptr + local_stage * stage_size;
            uint8_t* dst_ptr = shbuf_ptr + (local_stage % num_buffered_stages) * stage_size;
            size_t s_chunk_size = MIN(stage_size, data_end - src_ptr);

            uint8_t* src_end = src_ptr + s_chunk_size;
            src_ptr += 16 * t;
            dst_ptr += 16 * t;
            while(src_ptr < src_end)
            {
                *((uint4*) dst_ptr) = *((uint4*) src_ptr);
                src_ptr += 16 * NUM_THREADS;
                dst_ptr += 16 * NUM_THREADS;
            }
        };

        // Wait for all consumers to reach stage
        auto wait_consumers_stage = [&] (uint32_t stage)
        {
            if (threadIdx.x == 0)
            {
                uint64_t deadline = sync_deadline();

                uint32_t pending = device_mask & ~(1 << this_device);
                uint32_t sleep = SYNC_MIN_SLEEP;
                while (pending)
                {
                    const int r = __ffs(pending) - 1;
                    if (ldg_acquire_sys_u32(broadcast_stages_ptr + r) >= stage)
                    {
                        pending &= (pending - 1);
                        sleep = SYNC_MIN_SLEEP;
                    }
                    else __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "broadcast");
                    if (*abort_flag) break;
                }
            }
            __syncthreads();
        };

        // Producer loop
        while (local_stage < num_stages && !(*abort_flag))
        {
            produce_next_stage();
            local_stage++;
            stg_release_sys_u32(broadcast_stages_ptr + this_device, local_stage);

            // Wait for all consumers to be at most NUM_BROADCAST_STAGES - 2 behind. After last stage, wait for
            // consumers to finish
            wait_consumers_stage(local_stage < num_stages ? MAX(0, local_stage - num_buffered_stages + 2) : local_stage);
        }
    }

    // Consumer
    else
    {
        // Wait for producer to be past stage, poll with acquire to make sure all changes to host memory (including
        // the payload from the producer) are visible
        auto wait_producer_stage = [&] (uint32_t stage)
        {
            if (threadIdx.x == 0)
            {
                uint64_t deadline = sync_deadline();
                uint64_t sleep = SYNC_MIN_SLEEP;
                while (ldg_acquire_sys_u32(broadcast_stages_ptr + src_device) <= stage)
                {
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "broadcast");
                    if (*abort_flag) break;
                }
            }
            __syncthreads();
        };

        // Copy one stage from host buffer
        auto consume_next_stage = [&] ()
        {
            uint8_t* src_ptr = shbuf_ptr + (local_stage % num_buffered_stages) * stage_size;
            uint8_t* dst_ptr = data_ptr + local_stage * stage_size;
            size_t r_chunk_size = MIN(stage_size, data_end - dst_ptr);

            uint8_t* src_end = src_ptr + r_chunk_size;
            src_ptr += 16 * t;
            dst_ptr += 16 * t;
            while(src_ptr < src_end)
            {
                *((uint4*) dst_ptr) = *((uint4*)src_ptr);
                src_ptr += 16 * NUM_THREADS;
                dst_ptr += 16 * NUM_THREADS;
            }
        };

        // Consumer loop
        while (local_stage < num_stages && !(*abort_flag))
        {
            wait_producer_stage(local_stage);
            consume_next_stage();

            // Signal stage is produced
            local_stage++;
            stg_release_sys_u32(&broadcast_stages_ptr[this_device], local_stage);
        }
    }

    // Finished. Barrier to make sure buffer is free on kernel exit
    pg_barrier_inner(ctx, device_mask, this_device, src_device, abort_flag);

    // Clear stages for next kernel
    if constexpr (is_producer)
    {
        if (threadIdx.x == 0)
        {
            for (uint32_t pending = device_mask; pending; pending &= (pending - 1))
            {
                const int r = __ffs(pending) - 1;
                if (__popc(pending) > 1)
                    broadcast_stages_ptr[r] = 0;
                else
                    stg_release_sys_u32(broadcast_stages_ptr + r, 0);
            }
        }
    };
}


template <bool is_producer>
__global__ __launch_bounds__(NUM_THREADS_LL)
void pg_broadcast_ll_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int src_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ shbuf_ptr,
    size_t data_size,
    size_t shbuf_size,
    uint32_t* abort_flag
)
{
    int t = threadIdx.x;

    // Use barrier epoch to synchronize. The barrier at the end of this kernel ensures it will increment at
    // least once before the next broadcast. Only load in thread 0 to reduce PCIe traffic.
    __shared__ uint32_t cookie_s;
    if (threadIdx.x == 0)
        cookie_s = ldg_cv_u32(&ctx->barrier_epoch);
    __syncthreads();
    uint32_t cookie = cookie_s;

    // Effective buffer length, accounting for 100% cookie overhead
    size_t iter_size = shbuf_size / sizeof(uint64_t);

    // Divide into chunks, each using the whole buffer
    int num_iters = CEIL_DIVIDE(data_size / sizeof(uint32_t), iter_size);
    for (int iter = 0; iter < num_iters; ++iter)
    {
        // Indexing shared between producer and consumers
        uint32_t* chunk = ((uint32_t*) data_ptr) + iter * iter_size;
        uint32_t* chunk_end = MIN(chunk + iter_size, (uint32_t*) (data_ptr + data_size));
        int chunk_items = chunk_end - chunk;
        uint64_t* shbuf_pack = (uint64_t*) shbuf_ptr;

        // Producer
        if constexpr (is_producer)
        {
            for (int i = t; i < chunk_items; i += NUM_THREADS_LL)
                synced_write_uint32(shbuf_pack + i, chunk[i], cookie);
        }

        // Consumer
        else
        {
            uint64_t deadline = sync_deadline();
            for (int i = t; i < chunk_items && !(*abort_flag); i += NUM_THREADS_LL)
                chunk[i] = synced_read_uint32(ctx, shbuf_pack + i, cookie, deadline, abort_flag, "pg_broadcast_ll_kernel");
        }

        __syncthreads();

        // Make sure buffer is free for next iteration or kernel
        pg_barrier_inner(ctx, device_mask, this_device, src_device, abort_flag);
    }
}


void pg_broadcast
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    uint8_t* shbuf_ptr = (uint8_t*) shbuf;
    size_t data_size = tensor.numel() * tensor.element_size();
    TORCH_CHECK(data_size % 16 == 0, "data_size must be multiple of 16");

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);

    #define ARGS \
        (PGContext*) ctx, \
        device_mask, \
        this_device, \
        src_device, \
        data_ptr, \
        shbuf_ptr, \
        data_size, \
        shbuf_size, \
        (uint32_t*) abort_flag.data_ptr()

    if (this_device == src_device)
        pg_broadcast_kernel<true><<<1, NUM_THREADS, 0, stream>>>(ARGS);
    else
        pg_broadcast_kernel<false><<<1, NUM_THREADS, 0, stream>>>(ARGS);
    cuda_check(cudaPeekAtLastError());

    #undef ARGS
}


void pg_broadcast_ll
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    uint8_t* shbuf_ptr = (uint8_t*) shbuf;
    size_t data_size = tensor.numel() * tensor.element_size();
    TORCH_CHECK(data_size % 4 == 0, "data_size must be multiple of 4");

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);

    #define ARGS \
        (PGContext*) ctx, \
        device_mask, \
        this_device, \
        src_device, \
        data_ptr, \
        shbuf_ptr, \
        data_size, \
        shbuf_size, \
        (uint32_t*) abort_flag.data_ptr()

    if (this_device == src_device)
        pg_broadcast_ll_kernel<true><<<1, NUM_THREADS_LL, 0, stream>>>(ARGS);
    else
        pg_broadcast_ll_kernel<false><<<1, NUM_THREADS_LL, 0, stream>>>(ARGS);
    cuda_check(cudaPeekAtLastError());

    #undef ARGS
}

