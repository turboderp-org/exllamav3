#pragma once

#include <ATen/Tensor.h>

#define MAX_DEVICES 16
#define BROADCAST_STAGE_SIZE 16384
#define MAX_REDUCE_JOBS 2048
#define REDUCE_STAGE_STRIDE (64 / sizeof(uint32_t))

// Sync delay in nanoseconds
#define SYNC_MIN_SLEEP 64
#define SYNC_MAX_SLEEP 1024

// Timeout in seconds
#define SYNC_TIMEOUT 2ull

struct ReduceJob
{
    size_t data_size;
    uint32_t device_mask;
};

struct alignas(64) PGContext
{
    uint32_t sync_timeout;
    uint32_t barrier_epoch;
    alignas(16) uint32_t barrier_epoch_device[MAX_DEVICES];
    alignas(16) uint32_t broadcast_stage_device[MAX_DEVICES];
    alignas(16) uint32_t reduce_stage_produced[MAX_DEVICES];
    alignas(16) uint32_t reduce_stage_consumed[MAX_DEVICES];
    alignas(16) uint32_t gather_stage_produced[MAX_DEVICES];
    alignas(16) uint32_t gather_stage_consumed[MAX_DEVICES];

    // Maintain flags in separate 64-byte regions/cache lines
    alignas(64) uint32_t reduce_jobs_head; char _pad1[64 - sizeof(uint32_t)];
    alignas(64) uint32_t reduce_jobs_tail; char _pad2[64 - sizeof(uint32_t)];
    alignas(64) uint32_t cpusum_stage_device[MAX_DEVICES * REDUCE_STAGE_STRIDE];
    alignas(64) uint32_t cpusum_stage_cpu; char _pad4[64 - sizeof(uint32_t)];
    ReduceJob reduce_jobs[MAX_REDUCE_JOBS];
};

void pg_init_context(uintptr_t ctx);
void pg_check_timeout(uintptr_t ctx);