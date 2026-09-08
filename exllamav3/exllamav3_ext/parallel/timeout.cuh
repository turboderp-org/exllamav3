#pragma once
#include "context.cuh"
#include "../ptx.cuh"

// globaltimer_ns() and its tick rate GLOBALTIMER_HZ come from ptx.cuh /
// ptx_rocm_compat.cuh (ROCm's clock64() fallback is budgeted high-side, so the
// abort can only fire late, never early)

__device__ __forceinline__ uint64_t sync_deadline()
{
    return globaltimer_ns() + SYNC_TIMEOUT * GLOBALTIMER_HZ;
}

__device__ __forceinline__ uint32_t check_timeout(PGContext* ctx, uint64_t deadline, const char* name)
{
    // Sticky: once any collective on any rank has timed out, every later wait aborts at once. The host
    // enqueues a whole forward's collectives ahead of time, so without this each queued kernel would sit
    // out its own full deadline before the stream drains and the host can raise
    if (ldg_acquire_sys_u32(&ctx->sync_timeout)) return 1;
    uint32_t timeout = globaltimer_ns() >= deadline ? 1 : 0;
    // name first, flag last (release): the flag store publishes the name bytes;
    // host-side printing because device printf is unreliable from a wedged collective
    if (timeout && threadIdx.x == 0)
    {
        char* dst = ctx->sync_timeout_name;
        int i = 0;
        #pragma unroll 8
        for (; i < 63 && name[i]; ++i) dst[i] = name[i];
        dst[i] = 0;
        __threadfence_system();
        stg_release_sys_u32(&ctx->sync_timeout, 1);
    }
    return timeout;
}
