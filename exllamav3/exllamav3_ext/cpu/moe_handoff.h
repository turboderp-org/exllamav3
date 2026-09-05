#pragma once

#include <cstdint>

// Persistent-worker handoff for CPU-offloaded MoE experts (pattern follows the native TP
// backend's CPU-assisted all-reduce: pinned shared-memory flags, GPU-side publish/wait kernels,
// a spin-parked child process consuming a job ring).
//
// Shared-memory layout (allocated and cudaHostRegistered by the parent, plain-mapped by the
// child; all offsets in bytes from the region base):
//
//   [ctrl block]
//     u32 quit          @ 0    parent -> child: exit the worker loop
//     u32 pass_wake     @ 64   parent bumps at the start of every forward pass: child spins hard
//     u32 abort         @ 128  set by the GPU wait kernel on timeout; parent checks per pass
//     u32 ready         @ 192  child -> parent: worker loop entered, layers loaded
//     u32 jobs_tail     @ 256  parent producer index (job ring)
//     u32 jobs_head     @ 320  child consumer index
//     MoeJob ring       @ 384  MOE_JOB_RING entries
//     u32 data_ready[]  @ SLOT_FLAGS_OFFSET       per slot, stride 64: GPU publishes job seq
//     u32 done[]        @ SLOT_FLAGS_OFFSET + 64*MOE_MAX_SLOTS, stride 64: child publishes seq
//     u32 consumed[]    @ SLOT_FLAGS_OFFSET + 2*64*MOE_MAX_SLOTS, stride 64: the collecting
//                         device's stream publishes seq after reading the slot output back.
//                         Slot reuse waits on this, not done: with offloaded layers spread
//                         over multiple GPUs, the next tenant's issue runs on a different
//                         stream than the previous tenant's readback, and done (child
//                         finished writing) alone lets the child overwrite the output while
//                         the readback is still queued on the other device
//   [slot data]         @ MOE_CTRL_SIZE, num_slots x slot_size
//     per slot: [x fp16 cap_rows*Hi][sel i32 cap_rows*topk][w fp16 cap_rows*topk]
//               [out fp32 cap_rows*Ho], each section 64-byte aligned
//
// Streamed prefill DMAs expert weights straight out of the child's arena chunks (shared
// sections the parent maps and page-locks, see moe_cpu_host.py); there is no staging ring.
//
// Ordering: the parent writes a job descriptor and bumps jobs_tail before enqueueing the GPU
// memcpys and the data_ready publish for that seq, so the child always sees the descriptor
// before it can see the data flag. Sequence numbers increase monotonically from 1; slot reuse is
// safe because the GPU stream serializes on the wait kernel of the oldest outstanding job.

#define MOE_JOB_RING 256
#define MOE_MAX_SLOTS 8
#define MOE_JOB_KIND_COMPUTE 0
// GATED: issued by the fused-issue kernel, whose collect side reads the output only when at
// least one selected expert was CPU-resident. The worker may skip the compute and the
// output write entirely for an all-inactive job. Plain COMPUTE jobs must keep writing
// (zeroing) the output, since their collect reads it back unconditionally
#define MOE_JOB_KIND_COMPUTE_GATED 2

struct MoeJob
{
    uint32_t seq;
    uint32_t layer;
    uint32_t rows;
    uint32_t topk;
    uint32_t slot;
    uint32_t kind;
    uint32_t _pad[2];
};

#define MOE_CTRL_JOBS_OFFSET 384
#define MOE_SLOT_FLAGS_OFFSET (MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * sizeof(MoeJob))

// Flag order: data_ready[MAX_SLOTS], done[MAX_SLOTS], consumed[MAX_SLOTS], each entry 64-byte
// strided
#define MOE_FLAGS_SIZE (3 * 64 * MOE_MAX_SLOTS)
#define MOE_CTRL_SIZE (MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE)

// GPU-side flag ops on the current torch CUDA stream (addresses inside the registered region).
// Prefer stream memory operations (front-end executed, no SM occupancy, no launch cost) over the
// kernel fallback; exl3_moe_cpu_set_memops toggles this process-wide (called once, parent-side
// only -- these ops are only ever issued from the process enqueueing the GPU work).
void exl3_moe_flag_write(uintptr_t flag, int64_t value);
void exl3_moe_flag_wait(uintptr_t flag, int64_t value, uintptr_t abort_flag);
void exl3_moe_cpu_set_memops(bool enabled);

// Child worker loop: consumes jobs until quit is set. Layer indices in job descriptors refer to
// the registration order of exl3_moe_cpu_make_layer calls made in this (child) process.
void exl3_moe_cpu_worker_run
(
    uintptr_t shm_base,
    int64_t num_slots,
    int64_t slot_size,
    int64_t cap_rows,
    int64_t max_hi,
    int64_t max_ho,
    int64_t max_topk,
    int64_t threads
);
