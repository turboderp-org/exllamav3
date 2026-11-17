#include "moe_handoff.h"
#include "moe_mul1.h"
#include "../routing.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "../ptx.cuh"

#ifdef __linux__
#include <dlfcn.h>
#else
#include <intrin.h>
#include <windows.h>
#endif

// Wait timeout: generous, the CPU may be chewing a full prefill chunk
#define MOE_WAIT_TIMEOUT_NS 30000000000ull
#define MOE_SLEEP_MIN 64
#define MOE_SLEEP_MAX 2048

// -------------------------------------------------------------------------------------------
//   GPU flag kernels
// -------------------------------------------------------------------------------------------

// Fallback for when the stream memory operations are unavailable or disabled
__global__ void moe_flag_write_kernel(moe_flag_t* flag, moe_flag_t value)
{
#if MOE_FLAG_BITS == 64
    stg_release_sys_u64(flag, value);
#else
    stg_release_sys_u32(flag, value);
#endif
}

// Fallback for when the stream memory operations are unavailable or disabled
__global__ void moe_flag_wait_kernel(moe_flag_t* flag, moe_flag_t value, uint32_t* abort_flag)
{
    uint64_t sleep = MOE_SLEEP_MIN;
    uint64_t waited = 0;
    while (true)
    {
#if MOE_FLAG_BITS == 64
        if (ldg_acquire_sys_u64(flag) >= value) return;
#else
        uint32_t v = (uint32_t) ldg_acquire_sys_u32(flag);
        if ((int32_t)(v - value) >= 0) return;
#endif
        __nanosleep(sleep);
        waited += sleep;
        if (sleep < MOE_SLEEP_MAX) sleep <<= 1;
        if (waited > MOE_WAIT_TIMEOUT_NS)
        {
            stg_release_sys_u32(abort_flag, 1);
            return;
        }
    }
}

// Stream memory operations: same semantics as the flag kernels (CU_STREAM_WAIT_VALUE_GEQ is
// documented as the cyclic comparison (int32_t)(*addr - value) >= 0, identical to the kernel's
// predicate), but executed by the GPU front-end: no SM occupancy (nothing to co-schedule against
// exl3_moe's all-blocks-resident launch) and no kernel-launch cost per wait. The kernel path
// remains as a fallback (EXL3_MOE_MEMOPS=0 forces it); note the memop wait has no timeout, so
// dead-worker detection moves to the host-side watchdog, which unblocks pending waits by
// writing satisfying values into the flags.
namespace {

#if defined(USE_ROCM)
// HIP has the memops as runtime entry points in the library already linked. Its wait compares with a plain
// unsigned >= rather than CUDA's cyclic one, hence the 64-bit flags (see moe_handoff.h)
typedef hipError_t (*fn_stream_wait64)(hipStream_t, void*, uint64_t, unsigned int, uint64_t);
typedef hipError_t (*fn_stream_write64)(hipStream_t, void*, uint64_t, unsigned int);

struct MemOps
{
    fn_stream_wait64 wait = &hipStreamWaitValue64;
    fn_stream_write64 write = &hipStreamWriteValue64;
    bool resolved = true;
};
#else
typedef CUresult (CUDAAPI* fn_stream_wait32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
typedef CUresult (CUDAAPI* fn_stream_write32)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);

struct MemOps
{
    fn_stream_wait32 wait = nullptr;
    fn_stream_write32 write = nullptr;
    bool resolved = false;
    MemOps()
    {
        // Symbol resolution is unconditional (independent of exl3_moe_cpu_set_memops): whether
        // the ops are used is a separate, mutable runtime switch, not a one-time decision
#ifdef __linux__
        void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (!h) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!h) return;
        wait = (fn_stream_wait32) dlsym(h, "cuStreamWaitValue32_v2");
        if (!wait) wait = (fn_stream_wait32) dlsym(h, "cuStreamWaitValue32");
        write = (fn_stream_write32) dlsym(h, "cuStreamWriteValue32_v2");
        if (!write) write = (fn_stream_write32) dlsym(h, "cuStreamWriteValue32");
#else
        HMODULE h = GetModuleHandleA("nvcuda.dll");
        if (!h) return;
        wait = (fn_stream_wait32) GetProcAddress(h, "cuStreamWaitValue32_v2");
        if (!wait) wait = (fn_stream_wait32) GetProcAddress(h, "cuStreamWaitValue32");
        write = (fn_stream_write32) GetProcAddress(h, "cuStreamWriteValue32_v2");
        if (!write) write = (fn_stream_write32) GetProcAddress(h, "cuStreamWriteValue32");
#endif
        resolved = wait && write;
    }
};
#endif

MemOps& memops() { static MemOps m; return m; }
std::atomic<bool> g_memops_ok { true };
std::atomic<bool> g_memops_enabled { true };

inline uint32_t load_acquire_u32(const uint32_t* p)
{
#if defined(_MSC_VER) && !defined(__clang__)
    long v = _InterlockedCompareExchange(
        reinterpret_cast<volatile long*>(const_cast<uint32_t*>(p)), 0L, 0L);
    return static_cast<uint32_t>(v);
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

inline void store_release_u32(uint32_t* p, uint32_t v)
{
#if defined(_MSC_VER) && !defined(__clang__)
    (void)_InterlockedExchange(reinterpret_cast<volatile long*>(p), static_cast<long>(v));
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

} // namespace

void exl3_moe_cpu_set_memops(bool enabled)
{
    g_memops_enabled.store(enabled, std::memory_order_relaxed);
}

void exl3_moe_flag_write(uintptr_t flag, int64_t value)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    MemOps& m = memops();
    if (m.resolved && g_memops_enabled.load(std::memory_order_relaxed)
        && g_memops_ok.load(std::memory_order_relaxed))
    {
#if defined(USE_ROCM)
        hipError_t r = m.write(stream, (void*) flag, (uint64_t) value, 0);
        if (r == hipSuccess) return;
#else
        CUresult r = m.write((CUstream) stream, (CUdeviceptr) flag, (cuuint32_t) value, 0);
        if (r == CUDA_SUCCESS) return;
#endif
        g_memops_ok.store(false, std::memory_order_relaxed);
    }
    moe_flag_write_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<moe_flag_t*>(flag), static_cast<moe_flag_t>(value));
}

void exl3_moe_flag_wait(uintptr_t flag, int64_t value, uintptr_t abort_flag)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    MemOps& m = memops();
    if (m.resolved && g_memops_enabled.load(std::memory_order_relaxed)
        && g_memops_ok.load(std::memory_order_relaxed))
    {
#if defined(USE_ROCM)
        hipError_t r = m.wait(stream, (void*) flag, (uint64_t) value, hipStreamWaitValueGte, ~0ull);
        if (r == hipSuccess) return;
#else
        CUresult r = m.wait(
            (CUstream) stream,
            (CUdeviceptr) flag,
            (cuuint32_t) value,
            CU_STREAM_WAIT_VALUE_GEQ
        );
        if (r == CUDA_SUCCESS) return;
#endif
        g_memops_ok.store(false, std::memory_order_relaxed);
    }
    moe_flag_wait_kernel<<<1, 1, 0, stream>>>
    (
        reinterpret_cast<moe_flag_t*>(flag),
        static_cast<moe_flag_t>(value),
        reinterpret_cast<uint32_t*>(abort_flag)
    );
}

// ---- fused split issue / collect -----------------------------------------------------------
//
// One call per layer per step for each half of the decode handoff, replacing the host-side job
// descriptor write, ring check and the separate flag-wait / kernel / flag-write calls. The job
// descriptor goes into the shared ring here (same layout as the Python writer); the flag memops
// and the kernel are enqueued on the stream of the tensors' device. Returns false without
// touching anything when the job ring is within its safety margin of overflowing, so the caller
// can throttle and retry (a long prefill is the only way to get there).

namespace {

inline void flag_write_on(cudaStream_t stream, uintptr_t flag, int64_t value)
{
    MemOps& m = memops();
    if (m.resolved && g_memops_enabled.load(std::memory_order_relaxed)
        && g_memops_ok.load(std::memory_order_relaxed))
    {
#if defined(USE_ROCM)
        if (m.write(stream, (void*) flag, (uint64_t) value, 0) == cudaSuccess) return;
#else
        if (m.write((CUstream) stream, (CUdeviceptr) flag, (cuuint32_t) value, 0) == CUDA_SUCCESS) return;
#endif
        g_memops_ok.store(false, std::memory_order_relaxed);
    }
    moe_flag_write_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<moe_flag_t*>(flag), static_cast<moe_flag_t>(value));
}

inline void flag_wait_on(cudaStream_t stream, uintptr_t flag, int64_t value, uintptr_t abort_flag)
{
    MemOps& m = memops();
    if (m.resolved && g_memops_enabled.load(std::memory_order_relaxed)
        && g_memops_ok.load(std::memory_order_relaxed))
    {
#if defined(USE_ROCM)
        if (m.wait(stream, (void*) flag, (uint64_t) value, hipStreamWaitValueGte, ~0ull) == cudaSuccess) return;
#else
        if (m.wait((CUstream) stream, (CUdeviceptr) flag, (cuuint32_t) value, CU_STREAM_WAIT_VALUE_GEQ) == CUDA_SUCCESS) return;
#endif
        g_memops_ok.store(false, std::memory_order_relaxed);
    }
    moe_flag_wait_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<moe_flag_t*>(flag), static_cast<moe_flag_t>(value), reinterpret_cast<uint32_t*>(abort_flag));
}

} // namespace

bool exl3_moe_split_issue_fused
(
    uintptr_t shm_base,
    int64_t seq,
    int64_t layer,
    int64_t rows,
    int64_t topk,
    int64_t slot,
    int64_t prev_seq,
    at::Tensor sel,
    const c10::optional<at::Tensor>& map,
    const c10::optional<at::Tensor>& hist,
    const at::Tensor& y,
    const at::Tensor& w,
    int64_t h_sel_ptr,
    int64_t h_x_ptr,
    int64_t h_w_ptr,
    at::Tensor dev_count,
    int64_t hi,
    int64_t first_cpu,
    uintptr_t data_ready_flag,
    uintptr_t consumed_flag,
    uintptr_t abort_flag,
    int64_t num_workers,
    int64_t wext_offset
)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(shm_base);
    TORCH_CHECK(num_workers >= 1 && num_workers <= MOE_MAX_WORKERS, "exl3_moe_split_issue_fused: bad worker count");
    // Every worker's ring must have room before any of them gets the descriptor, so a declined
    // call leaves all rings untouched
    for (int64_t wi = 0; wi < num_workers; ++wi)
    {
        const MoeWorkerCtrl c = moe_worker_ctrl(base, wi, wext_offset);
        if (*c.jobs_tail - load_acquire_u32(c.jobs_head) >= MOE_JOB_RING - 4) return false;
    }
    // Descriptor first: host-visible before the GPU can publish the data flag
    for (int64_t wi = 0; wi < num_workers; ++wi)
    {
        const MoeWorkerCtrl c = moe_worker_ctrl(base, wi, wext_offset);
        const uint32_t tail = *c.jobs_tail;
        MoeJob& job = c.jobs[tail % MOE_JOB_RING];
        job.seq = (uint32_t) (seq & 0xFFFFFFFFull);
        job.seq_hi = (uint32_t) ((uint64_t) seq >> 32);
        job.layer = (uint32_t) layer;
        job.rows = (uint32_t) rows;
        job.topk = (uint32_t) topk;
        job.slot = (uint32_t) slot;
        job.kind = MOE_JOB_KIND_COMPUTE_GATED;
        store_release_u32(c.jobs_tail, tail + 1);
    }

    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(y.get_device()).stream();
    if (prev_seq) flag_wait_on(stream, consumed_flag, prev_seq, abort_flag);
    moe_split_issue(sel, map, hist, y, w, h_sel_ptr, h_x_ptr, h_w_ptr, dev_count, slot, hi, first_cpu);
    flag_write_on(stream, data_ready_flag, seq);
    return true;
}

void exl3_moe_split_collect_fused
(
    at::Tensor final_out,
    int64_t h_out_ptr,
    const at::Tensor& dev_count,
    int64_t slot,
    int64_t ho,
    uintptr_t done_flag,
    int64_t seq,
    uintptr_t consumed_flag,
    uintptr_t abort_flag,
    int64_t num_workers,
    int64_t partial_stride
)
{
    const at::cuda::OptionalCUDAGuard device_guard(final_out.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(final_out.get_device()).stream();
    // With several workers the last one to finish publishes done, so every partial is complete
    flag_wait_on(stream, done_flag, seq, abort_flag);
    moe_split_collect_add(final_out, h_out_ptr, dev_count, slot, ho, num_workers, partial_stride);
    // Consumption is published AFTER the readback on this same stream: the slot's next tenant
    // (possibly issuing from another device) gates on it
    flag_write_on(stream, consumed_flag, seq);
}

// ---- child worker loop --------------------------------------------------------------------

namespace {

inline void cpu_pause_()
{
#if defined(__aarch64__) || defined(__arm__)  // EXL3_AARCH64_PAUSE
    asm volatile("yield" ::: "memory");
#elif defined(__linux__)
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}

// Publish a job's full sequence number into a flag (only the 64-bit flags take the high half)
inline void store_release_flag(uint32_t* p, const MoeJob& job)
{
#if MOE_FLAG_BITS == 64
    uint64_t v = (uint64_t(job.seq_hi) << 32) | job.seq;
#if defined(_MSC_VER) && !defined(__clang__)
    (void)_InterlockedExchange64(reinterpret_cast<volatile long long*>(p), static_cast<long long>(v));
#else
    __atomic_store_n(reinterpret_cast<uint64_t*>(p), v, __ATOMIC_RELEASE);
#endif
#else
    store_release_u32(p, job.seq);
#endif
}

// Publish a job's completion: directly with one worker; with several, the worker whose arrival
// completes the set resets the counter and publishes (the others' parts happen-before through the
// counter's acq_rel read-modify-writes, so the GPU sees every partial once it sees the flag)
inline void publish_flag(uint32_t* flag, uint32_t* arrive, uint32_t num_workers, const MoeJob& job)
{
    if (arrive)
    {
#if defined(_MSC_VER) && !defined(__clang__)
        const uint32_t n = (uint32_t) _InterlockedIncrement(reinterpret_cast<volatile long*>(arrive));
#else
        const uint32_t n = __atomic_add_fetch(arrive, 1u, __ATOMIC_ACQ_REL);
#endif
        if (n != num_workers) return;
        store_release_u32(arrive, 0);
    }
    store_release_flag(flag, job);
}

inline size_t align64(size_t x) { return (x + 63) & ~size_t(63); }

} // namespace

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
)
{
    // GIL is released by the binding's call_guard; do not release it again here
    uint8_t* base = reinterpret_cast<uint8_t*>(shm_base);
    uint32_t* quit = reinterpret_cast<uint32_t*>(base + 0);
    uint32_t* pass_wake = reinterpret_cast<uint32_t*>(base + 64);
    uint32_t* hot = reinterpret_cast<uint32_t*>(base + 68);
    // This worker's control block (worker 0: the legacy block; see moe_handoff.h)
    const MoeWorkerCtrl ctrl = moe_worker_ctrl(base, worker_idx, wext_offset);
    uint32_t* ready = ctrl.ready;
    uint32_t* jobs_tail = ctrl.jobs_tail;
    uint32_t* jobs_head = ctrl.jobs_head;
    MoeJob* jobs = ctrl.jobs;
    uint32_t* data_ready = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET);
    uint32_t* done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS);
    uint32_t* stage_done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS);
    uint32_t* pinned_free = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS + 64 * MOE_MAX_WSLOTS);
    // Several workers: the shared flags are published by whichever worker completes the set
    uint32_t* arrive_done = num_workers > 1 ? moe_arrive_done(base, wext_offset, num_workers) : nullptr;
    uint32_t* arrive_stage = num_workers > 1 ? moe_arrive_stage(base, wext_offset, num_workers) : nullptr;
    uint32_t* stage_tail = ctrl.stage_tail;
    uint32_t* stage_head = ctrl.stage_head;
    MoeJob* stage_jobs = ctrl.stage_jobs;
    uint8_t* data = base + MOE_CTRL_SIZE;
    uint8_t* wstage = base + wstage_offset;

    // Dedicated stager: consumes the stage ring so weight memcpys run concurrently with the
    // compute pool's work on the token tail. Uses its own scratch threads, never the pool.
    // stage_threads comes from the parent (MoeCpuTuning), passed through the "start" layout.
    int stage_threads_ = stage_threads > 0 ? (int) stage_threads : 1;
    std::thread stager([&]()
    {
        uint32_t shead = load_acquire_u32(stage_head);
        uint32_t s_last_wake = 0;
        int s_idle = 0;
        std::chrono::steady_clock::time_point s_idle_since{};
        while (true)
        {
            if (load_acquire_u32(quit)) return;
            const uint32_t wake = load_acquire_u32(pass_wake);
            if (wake != s_last_wake) { s_last_wake = wake; s_idle = 0; }
            if (load_acquire_u32(stage_tail) == shead)
            {
                if (++s_idle < 65536) { cpu_pause_(); continue; }
                {
                    // Nap length by time idle, so the wake penalty stays a small fraction of any gap: a decode step
                    // with a long GPU-only window keeps 100 us naps, 10 ms naps only after seconds between requests
                    if (s_idle == 65536) s_idle_since = std::chrono::steady_clock::now();
                    const auto idle_for = std::chrono::steady_clock::now() - s_idle_since;
                    if (idle_for < std::chrono::milliseconds(100)) std::this_thread::sleep_for(std::chrono::microseconds(100));
                    else if (idle_for < std::chrono::seconds(2)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    else std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    s_idle = 65536 + 1;
                }
                continue;
            }
            s_idle = 0;

            const MoeJob job = stage_jobs[shead % MOE_STAGE_RING];
            shead++;
            store_release_u32(stage_head, shead);

            // Wait until the slot's previous tenant has been DMA'd out (low half of the flag, cyclically)
            uint32_t* pf = pinned_free + size_t(job.slot) * 16;
            while ((int32_t)(load_acquire_u32(pf) - job.prev_seq) < 0)
            {
                if (load_acquire_u32(quit)) return;
                cpu_pause_();
            }
            exl3_moe_cpu_stage_experts(
                static_cast<int64_t>(job.layer),
                job.experts,
                static_cast<int>(job.rows),
                wstage + size_t(job.slot) * wslot_size,
                stage_threads_
            );
            publish_flag(stage_done + size_t(job.slot) * 16, arrive_stage ? arrive_stage + size_t(job.slot) * 16 : nullptr,
                         (uint32_t) num_workers, job);
        }
    });

    // Fixed slot section offsets from the registered maxima; with several workers, each writes
    // its partial output into its own section after the first
    const size_t off_x = 0;
    const size_t off_sel = align64(off_x + size_t(cap_rows) * max_hi * 2);
    const size_t off_w = align64(off_sel + size_t(cap_rows) * max_topk * 4);
    const size_t off_out = align64(off_w + size_t(cap_rows) * max_topk * 2)
        + size_t(worker_idx) * align64(size_t(cap_rows) * max_ho * 4);

    // Handoff profiling (EXL3_MOE_HANDOFF_PROF): per-job wall times for the three segments the
    // worker can observe -- idle gap since the previous job, spin-on-data_ready, and compute
    const bool hprof = getenv("EXL3_MOE_HANDOFF_PROF") != nullptr;
    double hp_gap = 0.0, hp_spin = 0.0, hp_comp = 0.0;
    double hp_gap_mx = 0.0, hp_spin_mx = 0.0, hp_comp_mx = 0.0;
    long hp_jobs = 0, hp_empty = 0, hp_assign = 0, hp_rows = 0;
    auto hp_prev_end = std::chrono::steady_clock::now();

    const char* prime_env = getenv("EXL3_MOE_POOL_PRIME");
    const bool prime_pool = !prime_env || prime_env[0] != '0';
    if (prime_pool) exl3_moe_cpu_pool_prime((int) threads);
    store_release_u32(ready, 1);

    // Hot polling (the hot word): a host that forgets to clear it holds the cores this long at most
    constexpr auto MOE_HOT_SPIN_LIMIT = std::chrono::seconds(1);
    auto last_job = std::chrono::steady_clock::now() - MOE_HOT_SPIN_LIMIT;

    uint32_t head = load_acquire_u32(jobs_head);
    uint32_t last_wake = 0;
    int idle = 0;
    std::chrono::steady_clock::time_point idle_since{};
    while (true)
    {
        if (load_acquire_u32(quit)) break;

        const uint32_t wake = load_acquire_u32(pass_wake);
        if (wake != last_wake)
        {
            last_wake = wake;
            idle = 0;
            if (prime_pool) exl3_moe_cpu_pool_prime((int) threads);
        }

        if (load_acquire_u32(jobs_tail) == head)
        {
            // Spin hard in-pass (jobs arrive within microseconds of the GPU reaching the layer),
            // back off to two-tier naps (100us for first ~50ms, then 10ms between prompts)
            if (++idle < 65536) { cpu_pause_(); continue; }
            // The host expects the next pass within milliseconds (the MTP draft/verify handoff), sooner than a nap's
            // wake-up: keep the pool primed and polling instead
            if (load_acquire_u32(hot) && std::chrono::steady_clock::now() - last_job < MOE_HOT_SPIN_LIMIT)
            {
                if (prime_pool) exl3_moe_cpu_pool_prime((int) threads);
                idle = 65536 - 16384;
                continue;
            }
                {
                // Nap length by time idle, so the wake penalty stays a small fraction of any gap: a decode step
                // with a long GPU-only window keeps 100 us naps, 10 ms naps only after seconds between requests
                if (idle == 65536) idle_since = std::chrono::steady_clock::now();
                const auto idle_for = std::chrono::steady_clock::now() - idle_since;
                if (idle_for < std::chrono::milliseconds(100)) std::this_thread::sleep_for(std::chrono::microseconds(100));
                else if (idle_for < std::chrono::seconds(2)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                else std::this_thread::sleep_for(std::chrono::milliseconds(10));
                idle = 65536 + 1;
            }
            continue;
        }
        idle = 0;

        const MoeJob job = jobs[head % MOE_JOB_RING];
        head++;
        store_release_u32(jobs_head, head);

        const auto hp_t0 = std::chrono::steady_clock::now();

        // Wait for the GPU to publish the staged inputs for this seq (low half of the flag, cyclically)
        uint32_t* drdy = data_ready + size_t(job.slot) * 16;
        while ((int32_t)(load_acquire_u32(drdy) - job.seq) < 0)
        {
            if (load_acquire_u32(quit)) goto out;
            cpu_pause_();
        }

        const auto hp_t1 = std::chrono::steady_clock::now();

        {
            uint8_t* slot = data + size_t(job.slot) * slot_size;
            bool run = true;
            if (job.kind == MOE_JOB_KIND_COMPUTE_GATED)
            {
                // Fused-issue job: the collecting kernel reads the output only when some
                // selected expert is CPU-resident, so an all-inactive job is a pure no-op
                const int32_t* selp = reinterpret_cast<const int32_t*>(slot + off_sel);
                const int total = (int) job.rows * (int) job.topk;
                if (hprof)
                {
                    // Full-count variant: CPU-assignment stats for the profiler report
                    int n = 0;
                    for (int i = 0; i < total; ++i)
                        if (selp[i] >= 0) n++;
                    run = n > 0;
                    hp_assign += n;
                    hp_rows += (int) job.rows;
                    if (!run) hp_empty++;
                }
                else
                {
                    run = false;
                    for (int i = 0; i < total; ++i)
                        if (selp[i] >= 0) { run = true; break; }
                }
            }
            if (run)
                exl3_moe_cpu_forward_raw(
                    static_cast<int64_t>(job.layer),
                    reinterpret_cast<const at::Half*>(slot + off_x),
                    reinterpret_cast<const int32_t*>(slot + off_sel),
                    reinterpret_cast<const at::Half*>(slot + off_w),
                    reinterpret_cast<float*>(slot + off_out),
                    static_cast<int>(job.rows),
                    static_cast<int>(job.topk),
                    static_cast<int>(threads)
                );
        }

        publish_flag(done + size_t(job.slot) * 16, arrive_done ? arrive_done + size_t(job.slot) * 16 : nullptr,
                     (uint32_t) num_workers, job);
        last_job = std::chrono::steady_clock::now();

        if (hprof)
        {
            const auto hp_t2 = std::chrono::steady_clock::now();
            auto ms = [](auto a, auto b)
                { return std::chrono::duration<double, std::milli>(b - a).count(); };
            const double g = ms(hp_prev_end, hp_t0), s = ms(hp_t0, hp_t1), c = ms(hp_t1, hp_t2);
            hp_gap += g; hp_spin += s; hp_comp += c;
            if (g > hp_gap_mx) hp_gap_mx = g;
            if (s > hp_spin_mx) hp_spin_mx = s;
            if (c > hp_comp_mx) hp_comp_mx = c;
            hp_prev_end = hp_t2;
            if (++hp_jobs % 64 == 0)
            {
                // Several workers: each labels its lines (a single worker's are unchanged)
                char wtag[16] = "";
                if (wext_offset) snprintf(wtag, sizeof(wtag), " w%d", (int) worker_idx);
                printf(" -- handoff prof%s (%ld jobs, ms/job avg|max): gap %.3f|%.3f "
                       "spin %.3f|%.3f compute %.3f|%.3f | empty %ld/64, "
                       "cpu-assign/row %.2f\n",
                       wtag, hp_jobs, hp_gap / 64, hp_gap_mx, hp_spin / 64, hp_spin_mx,
                       hp_comp / 64, hp_comp_mx, hp_empty,
                       hp_rows ? (double) hp_assign / (double) hp_rows : 0.0);
                fflush(stdout);
                hp_gap = hp_spin = hp_comp = 0.0;
                hp_gap_mx = hp_spin_mx = hp_comp_mx = 0.0;
                hp_empty = 0; hp_assign = 0; hp_rows = 0;
            }
        }
    }
    out:;
    stager.join();
}
