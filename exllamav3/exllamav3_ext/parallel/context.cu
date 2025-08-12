#include "context.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"

void pg_init_context(uintptr_t ctx)
{
    PGContext* ctx_ptr = (PGContext*) ctx;

    ctx_ptr->sync_timeout = 0;
    ctx_ptr->barrier_epoch = 1;

    for (int i = 0; i < MAX_DEVICES; ++i)
        ctx_ptr->barrier_epoch_device[i] = 0;
}

void pg_check_timeout(uintptr_t ctx)
{
    PGContext* ctx_ptr = (PGContext*) ctx;
    if (ctx_ptr->sync_timeout)
    {
        TORCH_CHECK(false, "Synchronization timeout");
    }
}