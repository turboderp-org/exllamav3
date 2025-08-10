#pragma once

#include <ATen/Tensor.h>

#define MAX_DEVICES 16
#define BROADCAST_STAGE_SIZE 16384

// Sync delay in nanoseconds
#define SYNC_MIN_SLEEP 64
#define SYNC_MAX_SLEEP 1024

// Timeout in seconds
#define SYNC_TIMEOUT 2ull

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
};

void pg_init_context(uintptr_t ctx);
void pg_check_timeout(uintptr_t ctx);