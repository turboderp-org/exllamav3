#pragma once

#include <cstdint>
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

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
//     u32 hot           @ 68   parent -> child: keep polling between passes, the next one is a draft/verify
//                              handoff away (bounded by MOE_HOT_SPIN_LIMIT after the last job)
//     u32 abort         @ 128  set by the GPU wait kernel on timeout; parent checks per pass
//     u32 ready         @ 192  child -> parent: worker loop entered, layers loaded
//     u32 jobs_tail     @ 256  parent producer index (job ring)
//     u32 jobs_head     @ 320  child consumer index
//     MoeJob ring       @ 384  MOE_JOB_RING entries
//     flag data_ready[] @ SLOT_FLAGS_OFFSET       per slot, stride 64: GPU publishes job seq
//     flag done[]       @ SLOT_FLAGS_OFFSET + 64*MOE_MAX_SLOTS, stride 64: child publishes seq
//     flag consumed[]   @ SLOT_FLAGS_OFFSET + 2*64*MOE_MAX_SLOTS, stride 64: the collecting
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
// Ordering: the parent writes a job descriptor and bumps jobs_tail before enqueueing the GPU
// memcpys and the data_ready publish for that seq, so the child always sees the descriptor
// before it can see the data flag. Sequence numbers increase monotonically from 1; slot reuse is
// safe because the GPU stream serializes on the wait kernel of the oldest outstanding job.
//
// Flag width: the GPU waits for a flag to reach a sequence number with a stream memory operation.
// CUDA's 32-bit wait compares cyclically ((int32_t)(*addr - value) >= 0), so 32-bit flags may wrap.
// HIP's compares as plain unsigned >=, which a wrapped 32-bit flag would satisfy early, so ROCm
// builds use 64-bit flags that never wrap. Either way each flag owns a 64-byte line, and the job
// descriptor carries the sequence number as two 32-bit halves. CPU-side waits compare the low
// halves cyclically, which is exact for any width.
#if defined(USE_ROCM)
#define MOE_FLAG_BITS 64
typedef uint64_t moe_flag_t;
#else
#define MOE_FLAG_BITS 32
typedef uint32_t moe_flag_t;
#endif

#define MOE_JOB_RING 256
#define MOE_MAX_SLOTS 8
#define MOE_MAX_WSLOTS 8
#define MOE_JOB_KIND_COMPUTE 0
#define MOE_JOB_KIND_STAGE 1
// GATED: issued by the fused-issue kernel, whose collect side reads the output only when at
// least one selected expert was CPU-resident. The worker may skip the compute and the
// output write entirely for an all-inactive job. Plain COMPUTE jobs must keep writing
// (zeroing) the output, since their collect reads it back unconditionally
#define MOE_JOB_KIND_COMPUTE_GATED 2

// kind == STAGE: the worker memcpys `rows` experts' packed weights (ids in experts[]) of layer
// `layer` into weight-staging slot `slot`, after waiting for the parent's pinned_free flag to
// reach prev_seq (the slot's previous tenant has been DMA'd to VRAM). Weight-staging slots
// follow the compute slots in the shared region; their geometry travels in the layout dict.
//
// MOE_JOB_MAX_EXPERTS is the structural capacity of the experts[] array (just sizes a uint32_t
// array in shared memory; overprovisioning has no runtime cost since only the first `rows`
// entries are ever read). The runtime batch size actually used per stage job is a separate,
// Python-side tunable (MoeCpuTuning.batch_experts in moe_cpu_host.py, default 24) clamped to
// this capacity.
#define MOE_JOB_MAX_EXPERTS 256

struct MoeJob
{
    uint32_t seq;
    uint32_t layer;
    uint32_t rows;       // compute: token rows; stage: expert count
    uint32_t topk;
    uint32_t slot;
    uint32_t kind;
    uint32_t prev_seq;   // stage: pinned_free value to wait for before overwriting the slot
    uint32_t experts[MOE_JOB_MAX_EXPERTS];
    uint32_t seq_hi;     // high 32 bits of seq (published in full on 64-bit flags)
};

#define MOE_CTRL_JOBS_OFFSET 384
#define MOE_SLOT_FLAGS_OFFSET (MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * sizeof(MoeJob))

// Flag order: data_ready[MAX_SLOTS], done[MAX_SLOTS], stage_done[MAX_WSLOTS],
// pinned_free[MAX_WSLOTS], each entry 64-byte strided
#define MOE_FLAGS_SIZE (3 * 64 * MOE_MAX_SLOTS + 2 * 64 * MOE_MAX_WSLOTS)

// Stage jobs travel in their own ring, consumed by a dedicated stager thread in the child, so
// weight staging (pure memcpy) overlaps the compute pool's work on the token tail instead of
// queuing behind it
#define MOE_STAGE_RING 64
#define MOE_STAGE_TAIL_OFFSET (MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE)
#define MOE_STAGE_HEAD_OFFSET (MOE_STAGE_TAIL_OFFSET + 64)
#define MOE_STAGE_JOBS_OFFSET (MOE_STAGE_TAIL_OFFSET + 128)
#define MOE_CTRL_SIZE (MOE_STAGE_JOBS_OFFSET + MOE_STAGE_RING * sizeof(MoeJob))

// Multi-NUMA N-split (moe_cpu_numa_nodes): W worker processes, one per NUMA node, each holding a
// slice of every CPU-resident expert's intermediate dimension and writing a partial output that
// the collecting stream sums. Worker 0 uses the single-worker layout above unchanged. Workers 1..W-1 each get an
// extension block with their own rings, followed by one shared bank of arrival counters, all
// appended after the rest of the segment at the layout's wext_off (a single worker allocates
// none):
//
//   [extension block w]  @ wext_off + (w - 1) * MOE_WEXT_SIZE
//     u32 ready, jobs_tail, jobs_head, the job ring, stage_tail, stage_head and the stage ring,
//     as in the legacy block
//   [arrival counters]   @ wext_off + (W - 1) * MOE_WEXT_SIZE, MOE_ARRIVE_SIZE bytes
//     u32 arrive_done[MAX_SLOTS], arrive_stage[MAX_WSLOTS], each 64-byte strided
//
// Shared by all workers: quit, pass_wake, hot, abort, every flag (data_ready[], done[], consumed[],
// stage_done[], pinned_free[] at their single-worker places) and the slot inputs. A worker
// finishing its part of a job (or a stage job) bumps the slot's arrival counter; the one that
// completes the set resets it and publishes the flag, so the GPU waits on one flag whatever the
// worker count. The reset can't race the next tenant: a slot's next job starts only after the
// GPU consumed this one, which follows the flag, which follows the reset. Each slot carries one
// fp32 output section per worker, worker w's at off_out + w * align64(cap_rows * max_ho * 4) (a
// single worker's sits where it always did). Each worker stages into its own weight-staging
// region; worker 0's is the legacy one.
#define MOE_MAX_WORKERS 8
#define MOE_WEXT_READY 0
#define MOE_WEXT_JOBS_TAIL 64
#define MOE_WEXT_JOBS_HEAD 128
#define MOE_WEXT_JOBS 192
#define MOE_WEXT_STAGE_TAIL (MOE_WEXT_JOBS + MOE_JOB_RING * sizeof(MoeJob))
#define MOE_WEXT_STAGE_HEAD (MOE_WEXT_STAGE_TAIL + 64)
#define MOE_WEXT_STAGE_JOBS (MOE_WEXT_STAGE_HEAD + 64)
#define MOE_WEXT_SIZE (MOE_WEXT_STAGE_JOBS + MOE_STAGE_RING * sizeof(MoeJob))
#define MOE_ARRIVE_SIZE (64 * (MOE_MAX_SLOTS + MOE_MAX_WSLOTS))

// One worker's control words and rings (worker 0: the legacy block)
struct MoeWorkerCtrl
{
    uint32_t* ready;
    uint32_t* jobs_tail;
    uint32_t* jobs_head;
    MoeJob* jobs;
    uint32_t* stage_tail;
    uint32_t* stage_head;
    MoeJob* stage_jobs;
};

inline MoeWorkerCtrl moe_worker_ctrl(uint8_t* base, int64_t worker, int64_t wext_offset)
{
    if (worker == 0)
        return {
            reinterpret_cast<uint32_t*>(base + 192),
            reinterpret_cast<uint32_t*>(base + 256),
            reinterpret_cast<uint32_t*>(base + 320),
            reinterpret_cast<MoeJob*>(base + MOE_CTRL_JOBS_OFFSET),
            reinterpret_cast<uint32_t*>(base + MOE_STAGE_TAIL_OFFSET),
            reinterpret_cast<uint32_t*>(base + MOE_STAGE_HEAD_OFFSET),
            reinterpret_cast<MoeJob*>(base + MOE_STAGE_JOBS_OFFSET),
        };
    uint8_t* b = base + wext_offset + size_t(worker - 1) * MOE_WEXT_SIZE;
    return {
        reinterpret_cast<uint32_t*>(b + MOE_WEXT_READY),
        reinterpret_cast<uint32_t*>(b + MOE_WEXT_JOBS_TAIL),
        reinterpret_cast<uint32_t*>(b + MOE_WEXT_JOBS_HEAD),
        reinterpret_cast<MoeJob*>(b + MOE_WEXT_JOBS),
        reinterpret_cast<uint32_t*>(b + MOE_WEXT_STAGE_TAIL),
        reinterpret_cast<uint32_t*>(b + MOE_WEXT_STAGE_HEAD),
        reinterpret_cast<MoeJob*>(b + MOE_WEXT_STAGE_JOBS),
    };
}

// The arrival counters of a W-worker segment (W > 1)
inline uint32_t* moe_arrive_done(uint8_t* base, int64_t wext_offset, int64_t num_workers)
{
    return reinterpret_cast<uint32_t*>(base + wext_offset + size_t(num_workers - 1) * MOE_WEXT_SIZE);
}
inline uint32_t* moe_arrive_stage(uint8_t* base, int64_t wext_offset, int64_t num_workers)
{
    return moe_arrive_done(base, wext_offset, num_workers) + 16 * MOE_MAX_SLOTS;
}

// GPU-side flag ops on the current torch CUDA stream (addresses inside the registered region).
// Prefer stream memory operations (front-end executed, no SM occupancy, no launch cost) over the
// kernel fallback; exl3_moe_cpu_set_memops toggles this process-wide (called once, parent-side
// only -- these ops are only ever issued from the process enqueueing the GPU work).
void exl3_moe_flag_write(uintptr_t flag, int64_t value);
void exl3_moe_flag_wait(uintptr_t flag, int64_t value, uintptr_t abort_flag);

// Fused split-path handoff (see moe_handoff.cu): the job descriptor, the slot's consumed wait,
// the issue kernel and the data_ready flag in one call; the done wait, the collect kernel and the
// consumed flag in the other. num_workers > 1 publishes the descriptor into every worker's ring
// (wext_offset locates workers 1..W-1, see above) and sums the W partials, which sit
// partial_stride floats apart starting at h_out_ptr, once the last worker has published done
bool exl3_moe_split_issue_fused
(
    uintptr_t shm_base, int64_t seq, int64_t layer, int64_t rows, int64_t topk, int64_t slot, int64_t prev_seq,
    at::Tensor sel, const c10::optional<at::Tensor>& map, const c10::optional<at::Tensor>& hist,
    const at::Tensor& y, const at::Tensor& w, int64_t h_sel_ptr, int64_t h_x_ptr, int64_t h_w_ptr,
    at::Tensor dev_count, int64_t hi, int64_t first_cpu,
    uintptr_t data_ready_flag, uintptr_t consumed_flag, uintptr_t abort_flag,
    int64_t num_workers, int64_t wext_offset
);
void exl3_moe_split_collect_fused
(
    at::Tensor final_out, int64_t h_out_ptr, const at::Tensor& dev_count, int64_t slot, int64_t ho,
    uintptr_t done_flag, int64_t seq, uintptr_t consumed_flag, uintptr_t abort_flag,
    int64_t num_workers, int64_t partial_stride
);
void exl3_moe_cpu_set_memops(bool enabled);

// Child worker loop: consumes jobs until quit is set. Layer indices in job descriptors refer to
// the registration order of exl3_moe_cpu_make_layer calls made in this (child) process.
// worker_idx selects this worker's control block and output section (0 for a single worker, with
// num_workers 1); wstage_offset is this worker's own weight-staging region.
void exl3_moe_cpu_worker_run
(
    uintptr_t shm_base,
    int64_t num_slots,
    int64_t slot_size,
    int64_t cap_rows,
    int64_t max_hi,
    int64_t max_ho,
    int64_t max_topk,
    int64_t wstage_offset,
    int64_t num_wslots,
    int64_t wslot_size,
    int64_t threads,
    int64_t stage_threads,
    int64_t worker_idx,
    int64_t num_workers,
    int64_t wext_offset
);
