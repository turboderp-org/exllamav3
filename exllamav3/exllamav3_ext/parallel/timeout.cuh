#pragma once
#include "context.cuh"
#include "../ptx.cuh"

__device__ __forceinline__ uint64_t sync_deadline()
{
    return globaltimer_ns() + SYNC_TIMEOUT * 45000000000ull;
}

__device__ __forceinline__ uint32_t check_timeout(PGContext* ctx, uint64_t deadline, const char* name)
{
    // Sticky: once any collective on any rank has timed out, every later wait aborts at once. The host
    // enqueues a whole forward's collectives ahead of time, so without this each queued kernel would sit
    // out its own full deadline before the stream drains and the host can raise
    if (ldg_acquire_sys_u32(&ctx->sync_timeout)) return 1;
    uint32_t timeout = globaltimer_ns() >= deadline ? 1 : 0;
    if (timeout && threadIdx.x == 0)
    {
        // Device printf breaks the launch of these kernels on ROCm without PCIe atomics
        // (gfx1100): copy the name out for the host to print instead
        #if defined(USE_ROCM)
            char* dst = ctx->sync_timeout_name;
            int i = 0;
            #pragma unroll 8
            for (; i < 63 && name[i]; ++i) dst[i] = name[i];
            dst[i] = 0;
        #else
            printf(" ## Synchronization timeout in kernel: %s\n\n", name);
        #endif
        stg_release_sys_u32(&ctx->sync_timeout, 1);
    }
    return timeout;
}
