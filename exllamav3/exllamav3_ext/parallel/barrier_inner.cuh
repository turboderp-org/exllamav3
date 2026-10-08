// Barrier over the devices in device_mask, coordinated by coordinator_device. Each other
// participant publishes arrive[d] = release[d] + 1 and spins until the coordinator copies that
// value into release[d]; the coordinator spins until every participant's arrive counter is ahead
// of its release counter, then releases each of them. The counters are per device, so devices
// outside the mask are never touched and barriers over different subsets cannot release each
// other's participants. A device must not take part in two barriers at once (one stream per
// device enforces this). The coordinator itself needs no handshake.
__device__ __forceinline__ void pg_barrier_inner
(
    PGContext* __restrict__ ctx,
    uint32_t device_mask,
    int this_device,
    int coordinator_device,
    uint32_t* abort_flag
)
{
    if (!blockIdx.x && !blockIdx.y && !blockIdx.z && !threadIdx.x && !threadIdx.y && !threadIdx.z)
    {
        uint32_t* arrive  = ctx->barrier_arrive;
        uint32_t* release = ctx->barrier_release;

        if (this_device == coordinator_device)
        {
            uint64_t deadline = sync_deadline();
            uint32_t pending = device_mask & ~(1 << this_device);
            uint32_t seq[MAX_DEVICES];

            // Wait for the other participants to arrive. Arrive and release counters are loaded
            // together, so each poll costs one round trip
            uint32_t sleep = SYNC_MIN_SLEEP;
            while (pending)
            {
                uint32_t pending_t = pending;
                #pragma unroll
                for (int i = 0; i < MAX_DEVICES; i += 4)
                {
                    uint32_t pmask = (pending >> i) & 0x0f;
                    if (!pmask) continue;

                    uint4 a = ldg_cv_u128((const uint4*) (arrive + i));
                    uint4 r = ldg_cv_u128((const uint4*) (release + i));
                    if ((pmask & 1) && a.x != r.x) { pending &= ~(1 << (i + 0)); seq[i + 0] = a.x; }
                    if ((pmask & 2) && a.y != r.y) { pending &= ~(1 << (i + 1)); seq[i + 1] = a.y; }
                    if ((pmask & 4) && a.z != r.z) { pending &= ~(1 << (i + 2)); seq[i + 2] = a.z; }
                    if ((pmask & 8) && a.w != r.w) { pending &= ~(1 << (i + 3)); seq[i + 3] = a.w; }
                }

                if (pending == pending_t)
                {
                    __nanosleep(sleep);
                    if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                    else *abort_flag = check_timeout(ctx, deadline, "barrier");
                    if (*abort_flag) break;
                }
                else sleep = SYNC_MIN_SLEEP;
            }

            // Release every participant that arrived (all of them unless aborted)
            uint32_t arrived = device_mask & ~(1 << this_device) & ~pending;
            #pragma unroll
            for (int i = 0; i < MAX_DEVICES; ++i)
                if (arrived & (1 << i)) stg_wt_u32(release + i, seq[i]);
        }
        else
        {
            uint64_t deadline = sync_deadline();

            // Between barriers arrive == release for this device, and only this device writes its
            // arrive counter
            const uint32_t s = ldg_cv_u32(release + this_device) + 1;
            stg_wt_u32(arrive + this_device, s);

            // Wait for the coordinator to release this device
            uint64_t sleep = SYNC_MIN_SLEEP;
            while (ldg_cv_u32(release + this_device) != s)
            {
                __nanosleep(sleep);
                if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                else *abort_flag = check_timeout(ctx, deadline, "barrier");
                if (*abort_flag) break;
            }
        }
    }
    __syncthreads();
}
