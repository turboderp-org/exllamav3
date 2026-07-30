#pragma once

#include <ATen/Tensor.h>

#define MAX_DEVICES 16
#define BROADCAST_STAGE_SIZE 16384
#define MAX_REDUCE_JOBS 2048
#define REDUCE_STAGE_STRIDE (64 / sizeof(uint32_t))

// Threadblocks per direction in the multi-chunk (bandwidth) CPU-reduce kernel. One block cannot
// saturate pinned-memory reads (measured: 17 of 26 GB/s on PCIe4, 20 of 53 GB/s on PCIe5); each
// block owns a stripe of every chunk and its own stage flag, so per-block flag releases stay
// ordered behind that block's own PCIe writes
#define CPUREDUCE_MB_BLOCKS 4

// Sync delay in nanoseconds
#define SYNC_MIN_SLEEP 64
#define SYNC_MAX_SLEEP 1024

// Timeout in seconds
#define SYNC_TIMEOUT 2ull

// Wire format of a CPU-assisted reduce. BF16 wire carries fp32/bf16 payloads (range-safe for
// fp32 residual streams); FP16 wire carries fp16 payloads verbatim, making the reduce exact for
// two ranks (single round-to-nearest of the fp32 sum) at identical PCIe traffic. Requires F16C
// on the CPU for the fp16 accumulate.
#define REDUCE_WIRE_BF16 0
#define REDUCE_WIRE_FP16 1

struct ReduceJob
{
    size_t data_size;
    uint32_t device_mask;
    uint32_t wire_dtype;
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
    alignas(64) uint32_t broadcast_ll_epoch; char _pad0[64 - sizeof(uint32_t)];
    alignas(64) uint32_t broadcast_ll_sequence_device[MAX_DEVICES];
    alignas(64) uint32_t reduce_jobs_head; char _pad1[64 - sizeof(uint32_t)];
    alignas(64) uint32_t reduce_jobs_tail; char _pad2[64 - sizeof(uint32_t)];
    alignas(64) uint32_t cpusum_stage_device[MAX_DEVICES * REDUCE_STAGE_STRIDE];
    // Per-(device, block) stage flags for the multi-chunk reduce kernel (absolute stage values,
    // same sequence as cpusum_stage_device, which remains the per-device source of truth for
    // send-side stage discovery at kernel start)
    alignas(64) uint32_t cpusum_stage_device_mb[MAX_DEVICES * CPUREDUCE_MB_BLOCKS * REDUCE_STAGE_STRIDE];
    // Recv-side mirror of the stage sequence: cpusum_stage_recv is the per-device discovery
    // counter (advanced by every recv kernel), cpusum_stage_recv_mb the per-(device, block)
    // consumption progress the CPU throttles on before reusing accumulator ring slots
    alignas(64) uint32_t cpusum_stage_recv[MAX_DEVICES * REDUCE_STAGE_STRIDE];
    alignas(64) uint32_t cpusum_stage_recv_mb[MAX_DEVICES * CPUREDUCE_MB_BLOCKS * REDUCE_STAGE_STRIDE];
    alignas(64) uint32_t cpusum_stage_cpu; char _pad4[64 - sizeof(uint32_t)];
    ReduceJob reduce_jobs[MAX_REDUCE_JOBS];
};

void pg_init_context(uintptr_t ctx);
void pg_check_timeout(uintptr_t ctx);
