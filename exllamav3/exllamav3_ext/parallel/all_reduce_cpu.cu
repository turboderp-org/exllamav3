#include <cuda_fp16.h>
#include "all_reduce.cuh"
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
#include <thread>
#include <cstring>
#include <chrono>
#include "../avx2_target.h"
#include "all_reduce_cpu_avx2.h"

// Schedule CPU reduce job. Called by master device proces in all_reduce_cpu and assumes all
// CUDA streams have kernels scheduled at the same time
void push_reduce_job
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask
)
{
    atomic_ref<uint32_t> tail_(&ctx->reduce_jobs_tail);
    uint32_t tail = tail_.load_acquire();
    uint32_t next = (tail + 1) % MAX_REDUCE_JOBS;
    ctx->reduce_jobs[tail] = ReduceJob{ data_size, device_mask };
    tail_.store_release(next);
}


// Mark the end of the CPU reduce job queue. CPU process will sit in spin loop processing
// reductions until this is reached, after which the function returns. Scheduled once, at the
// end of a forward pass, by the master device process.
void push_reduce_job_end(PGContext* ctx)
{
    push_reduce_job(ctx, 0, 0);
}


// Pop reduce job off the front of the queue, called by CPU reduce process
bool pop_reduce_job
(
    PGContext* ctx,
    ReduceJob& out
)
{
    atomic_ref<uint32_t> head_(&ctx->reduce_jobs_head);
    atomic_ref<uint32_t> tail_(&ctx->reduce_jobs_tail);
    uint32_t head = head_.load_relaxed();
    uint32_t tail = tail_.load_acquire();
    if (head == tail) return false;
    out = ctx->reduce_jobs[head];
    head_.store_release((head + 1) % MAX_REDUCE_JOBS);
    return true;
}


void perform_cpu_reduce
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
);


// Busy loop to process all incoming jobs until the end marker is reached. Called at the start
// of a forward pass by the CPU reduce process
void run_cpu_reduce_jobs
(
    uintptr_t ctx_ptr,
    uintptr_t shbuf,
    size_t shbuf_size
)
{
    enable_fast_fp();

    PGContext* ctx = (PGContext*) ctx_ptr;
    uint8_t* shbuf_ptr = (uint8_t*) shbuf;
    ReduceJob current_job;

    uint32_t max_wait = 8000; // ms
    while (true)
    {
        // Wait for next job. Should never pause for more than one millisecond, since jobs are
        // scheduled along side all-reduce kernels. The reduction itself (below) pauses in a
        // tighter spin loop to reduce latency if the CUDA streams have not caught up yet
        // (typical case)
        while (true)
        {
            if (pop_reduce_job(ctx, current_job))
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            max_wait--;
            if (!max_wait)
            {
                // After 8 seconds of waiting for the queue to build, something surely crashed
                printf(" ## CPU reduce wait timeout");
                TORCH_CHECK(false, "CPU reduce wait timeout");
            }
        }

        // No more work
        if (current_job.data_size == 0)
            return;

        // Next job ready
        perform_cpu_reduce
        (
            ctx,
            current_job.data_size,
            current_job.device_mask,
            shbuf_ptr,
            shbuf_size
        );
    }
}


void end_cpu_reduce_jobs
(
    uintptr_t ctx_ptr
)
{
    PGContext* ctx = (PGContext*) ctx_ptr;
    push_reduce_job_end(ctx);
}

#define PARCK_MODE_FLOAT 0
#define PARCK_MODE_HALF 1
#define PARCK_MODE_BF16 2

template <int dtype>
__global__ __launch_bounds__(NUM_THREADS)
void pg_all_reduce_cpu_kernel
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int master_device,
    uint8_t* __restrict__ data_ptr,
    uint8_t* __restrict__ shbuf_ptr,
    size_t data_size,
    size_t shbuf_size,
    bool contributor,
    uint32_t* abort_flag
)
{
    // Indexing
    const uint32_t buf_slot_size = (shbuf_size / (MAX_DEVICES + 1) / 1024) * 1024;
    const uint32_t max_buf_stages = buf_slot_size / CPUREDUCE_CHUNK_SIZE;
    auto host_ptr = [&] (int device, uint32_t stage_idx)
    {
        return shbuf_ptr + buf_slot_size * device + (stage_idx % max_buf_stages) * CPUREDUCE_CHUNK_SIZE;
    };
    auto dev_ptr = [&] (uint32_t chunk_idx)
    {
        if constexpr (dtype == PARCK_MODE_FLOAT)
            return data_ptr + chunk_idx * CPUREDUCE_CHUNK_SIZE * 2;
        if constexpr (dtype == PARCK_MODE_HALF || dtype == PARCK_MODE_BF16)
            return data_ptr + chunk_idx * CPUREDUCE_CHUNK_SIZE;
    };
    uint8_t* data_end = data_ptr + data_size;

    int t = threadIdx.x;
    int dir = blockIdx.x;
    auto grid = cg::this_grid();

    // Get device stage
    __shared__ uint32_t cc;
    if (t == 0)
        cc = (uint32_t)ldg_acquire_sys_u32(ctx->cpusum_stage_device + this_device * REDUCE_STAGE_STRIDE);
    __syncthreads();

    grid.sync();
    uint32_t stage = cc & 0x7fffffffu;

    int num_iter;
    if constexpr (dtype == PARCK_MODE_FLOAT)
        num_iter = CEIL_DIVIDE(data_size, CPUREDUCE_CHUNK_SIZE * 2);
    if constexpr (dtype == PARCK_MODE_HALF || dtype == PARCK_MODE_BF16)
        num_iter = CEIL_DIVIDE(data_size, CPUREDUCE_CHUNK_SIZE);

    for (int iter = 0; iter < num_iter + 1; ++iter)
    {
        int send_idx = iter;
        int recv_idx = iter - 1;

        // Threadblock 0 writes contribution to shared buffer
        if (dir == 0 && iter < num_iter)
        {
            // Device has contribution
            if (contributor)
            {
                // Send float
                if constexpr (dtype == PARCK_MODE_FLOAT)
                {
                    uint4* src = (uint4*) dev_ptr(send_idx);
                    uint2* dst = (uint2*) host_ptr(this_device, stage);
                    for (int i = 0; i < 2 && src + t < (uint4*) data_end; ++i)
                    {
                        // Truncate FP32 -> BF16
                        uint4 v = src[t];
                        uint2 pack{ (v.y & 0xffff0000u) | (v.x >> 16), (v.w & 0xffff0000u) | (v.z >> 16) };
                        dst[t] = pack;
                        src += NUM_THREADS;
                        dst += NUM_THREADS;
                    }
                }

                // Send bfloat16
                if constexpr (dtype == PARCK_MODE_BF16)
                {
                    uint2* src = (uint2*) dev_ptr(send_idx);
                    uint2* dst = (uint2*) host_ptr(this_device, stage);
                    for (int i = 0; i < 2 && src + t < (uint2*) data_end; ++i)
                    {
                        // Copy BF16 -> BF16
                        uint2 pack = src[t];
                        dst[t] = pack;
                        src += NUM_THREADS;
                        dst += NUM_THREADS;
                    }
                }

                // Send half
                if constexpr (dtype == PARCK_MODE_HALF)
                {
                    half4* src = (half4*) dev_ptr(send_idx);
                    uint2* dst = (uint2*) host_ptr(this_device, stage);
                    for (int i = 0; i < 2 && src + t < (half4*) data_end; ++i)
                    {
                        half4 h = src[t];
                        float4 f{ __half2float(h.x.x), __half2float(h.x.y), __half2float(h.y.x), __half2float(h.y.y) };
                        uint4 v{ __float_as_uint(f.x), __float_as_uint(f.y), __float_as_uint(f.z), __float_as_uint(f.w) };
                        uint2 pack{ (v.y & 0xffff0000) | (v.x >> 16), (v.w & 0xffff0000) | (v.z >> 16) };
                        dst[t] = pack;
                        src += NUM_THREADS;
                        dst += NUM_THREADS;
                    }
                }

                // Signal CPU process
                stage = (stage + 1u) & 0x7fffffffu;
                __syncthreads();
                if (t == 0)
                {
                    stg_release_sys_u32(ctx->cpusum_stage_device + this_device * REDUCE_STAGE_STRIDE, stage);
                }
            }

            // Device has no contribution
            else
            {
                stage = (stage + 1u) & 0x7fffffffu;
                uint32_t estage = stage | 0x80000000u;
                __syncthreads();
                if (t == 0)
                {
                    stg_release_sys_u32(ctx->cpusum_stage_device + this_device * REDUCE_STAGE_STRIDE, estage);
                }
            }
        }

        // Threadblock 1 reads back sum
        if (dir == 1 && iter > 0)
        {
            // Wait for CPU process
            __shared__ bool to;
            if (t == 0)
            {
                to = false;
                uint64_t deadline = sync_deadline();
                uint64_t sleep = SYNC_MIN_SLEEP;
                while (true)
                {
                    uint32_t ep = (int)ldg_acquire_sys_u32(&ctx->cpusum_stage_cpu);
                    if (ep >= stage + 1u) break;
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else if (check_timeout(ctx, deadline, "pg_all_reduce_cpu_kernel"))
                    {
                        *abort_flag = 1;
                        break;
                    }
                }
            }
            __syncthreads();

            // Recv float
            if constexpr (dtype == PARCK_MODE_FLOAT)
            {
                uint4* dst = (uint4*) dev_ptr(recv_idx);
                uint2* src = (uint2*) host_ptr(MAX_DEVICES, stage);
                for (int i = 0; i < 2 && dst + t < (uint4*) data_end; ++i)
                {
                    uint2 pack = src[t];
                    uint4 v{ pack.x << 16, pack.x & 0xffff0000, pack.y << 16, pack.y & 0xffff0000 };
                    dst[t] = v;
                    src += NUM_THREADS;
                    dst += NUM_THREADS;
                }
            }

            // Recv bfloat16
            if constexpr (dtype == PARCK_MODE_BF16)
            {
                uint2* dst = (uint2*) dev_ptr(recv_idx);
                uint2* src = (uint2*) host_ptr(MAX_DEVICES, stage);
                for (int i = 0; i < 2 && dst + t < (uint2*) data_end; ++i)
                {
                    uint2 pack = src[t];
                    dst[t] = pack;
                    src += NUM_THREADS;
                    dst += NUM_THREADS;
                }
            }

            // Recv half
            if constexpr (dtype == PARCK_MODE_HALF)
            {
                half4* dst = (half4*) dev_ptr(recv_idx);
                uint2* src = (uint2*) host_ptr(MAX_DEVICES, stage);
                for (int i = 0; i < 2 && dst + t < (half4*) data_end; ++i)
                {
                    uint2 pack = src[t];
                    uint4 v{ pack.x << 16, pack.x & 0xffff0000, pack.y << 16, pack.y & 0xffff0000 };
                    float4 f{ __uint_as_float(v.x), __uint_as_float(v.y), __uint_as_float(v.z), __uint_as_float(v.w) };
                    half4 h(__float2half_rn(f.x), __float2half_rn(f.y), __float2half_rn(f.z), __float2half_rn(f.w));
                    dst[t] = h;
                    src += NUM_THREADS;
                    dst += NUM_THREADS;
                }
            }

            stage = (stage + 1u) & 0x7fffffffu;
        }

        grid.sync();
        if (*abort_flag) break;
    }
}

void pg_all_reduce_cpu
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t shbuf,
    size_t shbuf_size,
    bool is_master,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    TORCH_CHECK(is_avx2_supported(), "AVX2 is required for tensor-parallel inference using native backend");

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);

    uint8_t* data_ptr = (uint8_t*) tensor.data_ptr();
    uint8_t* shbuf_ptr = (uint8_t*) shbuf;
    size_t cpu_data_size = tensor.numel() * 2;
    size_t device_data_size = tensor.numel() * tensor.element_size();

    TORCH_CHECK(cpu_data_size % 16 == 0, "data_size must be multiple of 16");

    uint32_t* abort_flag_ptr = (uint32_t*) abort_flag.data_ptr();
    void* kernelArgs[] =
    {
        (void*)& ctx,
        (void*)& device_mask,
        (void*)& this_device,
        (void*)& master_device,
        (void*)& data_ptr,
        (void*)& shbuf_ptr,
        (void*)& device_data_size,
        (void*)& shbuf_size,
        (void*)& contributor,
        (void*)& abort_flag_ptr
    };

    dim3 block_grid(2);
    dim3 block_dim(NUM_THREADS);

    if (tensor.dtype() == at::kFloat)
    {
        cudaLaunchCooperativeKernel
        (
            (void*)pg_all_reduce_cpu_kernel<PARCK_MODE_FLOAT>,
            block_grid,
            block_dim,
            kernelArgs,
            0,
            stream
        );
    }
    else if (tensor.dtype() == at::kHalf)
    {
        cudaLaunchCooperativeKernel
        (
            (void*)pg_all_reduce_cpu_kernel<PARCK_MODE_HALF>,
            block_grid,
            block_dim,
            kernelArgs,
            0,
            stream
        );
    }
    else if (tensor.dtype() == at::kBFloat16)
    {
        cudaLaunchCooperativeKernel
        (
            (void*)pg_all_reduce_cpu_kernel<PARCK_MODE_BF16>,
            block_grid,
            block_dim,
            kernelArgs,
            0,
            stream
        );
    }
    else TORCH_CHECK(false, "pg_all_reduce_cpu: Unknown dtype");

    cuda_check(cudaPeekAtLastError());

    // Master also queues up CPU reduction. Job will be popped in separate CPU reduce process
    if (is_master)
        push_reduce_job((PGContext*) ctx, cpu_data_size, device_mask);
}