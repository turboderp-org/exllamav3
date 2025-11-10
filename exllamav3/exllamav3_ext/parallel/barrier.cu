#include <cuda_fp16.h>
#include "barrier.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "timeout.cuh"

#include "barrier_inner.cuh"

__global__ void pg_barrier_kernel
(
    PGContext* __restrict__ ctx,
    const uint32_t device_mask,
    const int this_device,
    const int coordinator_device,
    uint32_t* abort_flag
)
{
    pg_barrier_inner(ctx, device_mask, this_device, coordinator_device, abort_flag);
}

void pg_barrier
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    uint32_t device_mask = 0;
    for (int i : devices) device_mask |= (1 << i);

    pg_barrier_kernel<<<1, 1, 0, stream>>>
    (
        (PGContext*) ctx,  // Shared, pinned
        device_mask,
        this_device,
        devices[0],
        (uint32_t*) abort_flag.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}
