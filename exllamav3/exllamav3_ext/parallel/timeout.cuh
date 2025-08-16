#pragma once
#include "context.cuh"

__device__ __forceinline__ uint64_t sync_deadline()
{
    return globaltimer_ns() + SYNC_TIMEOUT * 1000000000ull;
}

__device__ __forceinline__ bool check_timeout(PGContext* ctx, uint64_t deadline, const char* name)
{
    bool timeout = globaltimer_ns() >= deadline;
    if (timeout && threadIdx.x == 0)
    {
        stg_release_sys_u32(&ctx->sync_timeout, 1);
        printf(" ## Synchronization timeout in kernel: %s\n\n", name);
    }
    return timeout;
}
