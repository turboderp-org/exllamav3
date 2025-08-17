#pragma once

__device__ __forceinline__ uint32_t synced_read_uint32
(
    PGContext* ctx,
    uint64_t* p,
    uint32_t cookie,
    uint64_t deadline,
    uint32_t *timeout,
    const char* timeout_name
)
{
    uint64_t sleep = SYNC_MIN_SLEEP;
    uint64_t packed;

    while(true)
    {
        packed = ldg_acquire_sys_u64(p);
        uint32_t got_cookie = uint32_t(packed >> 32);
        if (cookie == got_cookie) break;
        __nanosleep(sleep);
        if (sleep < SYNC_MAX_SLEEP) sleep *= 2;
        else
        {
            *timeout = check_timeout(ctx, deadline, timeout_name);
            if (*timeout) break;
        }
    }

    return uint32_t(packed & 0xffffffffu);
}

__device__ __forceinline__ void synced_write_uint32(uint64_t* p, uint32_t v, uint32_t cookie)
{
    uint64_t packed = (uint64_t(cookie) << 32) | uint64_t(v);
    stg_release_sys_u64(p, packed);
}