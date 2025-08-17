
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
        uint32_t* epoch_ptr     = &ctx->barrier_epoch;
        uint32_t* epoch_dev_ptr = ctx->barrier_epoch_device;

        // Snapshot current epoch
        const uint32_t epoch = ldg_cv_u32(epoch_ptr);

        // Publish arrival
        stg_wt_u32(&epoch_dev_ptr[this_device], epoch);

        if (this_device == coordinator_device)
        {
            uint64_t deadline = sync_deadline();
            uint32_t pending = device_mask & ~(1 << this_device);

            // Wait for other participants to arrive at epoch
            uint32_t sleep = SYNC_MIN_SLEEP;
            while (pending)
            {
                uint32_t pending_t = pending;
                uint4* p = (uint4*) epoch_dev_ptr;
                for (int i = 0; i < MAX_DEVICES; i += 4, p++)
                {
                    uint32_t pmask = (pending >> i) & 0x0f;
                    if (!pmask) continue;

                    uint4 s = ldg_cv_u128(p);
                    if ((pmask & 1) && s.x == epoch) pending &= ~(1 << (i + 0));
                    if ((pmask & 2) && s.y == epoch) pending &= ~(1 << (i + 1));
                    if ((pmask & 4) && s.z == epoch) pending &= ~(1 << (i + 2));
                    if ((pmask & 8) && s.w == epoch) pending &= ~(1 << (i + 3));
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

            // Release: bump epoch
            stg_wt_u32(epoch_ptr, epoch + 1);
        }
        else
        {
            uint64_t deadline = sync_deadline();

            // Wait for coordinator to bump epoch
            uint64_t sleep = SYNC_MIN_SLEEP;
            while (ldg_cv_u32(epoch_ptr) == epoch)
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