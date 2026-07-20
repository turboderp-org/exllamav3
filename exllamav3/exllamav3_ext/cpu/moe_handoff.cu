#include "moe_handoff.h"
#include "moe_mul1.h"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
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

// No longer used
__global__ void moe_flag_write_kernel(uint32_t* flag, uint32_t value)
{
    stg_release_sys_u32(flag, value);
}

// No longer used
__global__ void moe_flag_wait_kernel(uint32_t* flag, uint32_t value, uint32_t* abort_flag)
{
    uint64_t sleep = MOE_SLEEP_MIN;
    uint64_t waited = 0;
    while (true)
    {
        uint32_t v = (uint32_t) ldg_acquire_sys_u32(flag);
        if ((int32_t)(v - value) >= 0) return;
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

MemOps& memops() { static MemOps m; return m; }
std::atomic<bool> g_memops_ok { true };
std::atomic<bool> g_memops_enabled { true };

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
        CUresult r = m.write((CUstream) stream, (CUdeviceptr) flag, (cuuint32_t) value, 0);
        if (r == CUDA_SUCCESS) return;
        g_memops_ok.store(false, std::memory_order_relaxed);
    }
    moe_flag_write_kernel<<<1, 1, 0, stream>>>(reinterpret_cast<uint32_t*>(flag), static_cast<uint32_t>(value));
}

void exl3_moe_flag_wait(uintptr_t flag, int64_t value, uintptr_t abort_flag)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    MemOps& m = memops();
    if (m.resolved && g_memops_enabled.load(std::memory_order_relaxed)
        && g_memops_ok.load(std::memory_order_relaxed))
    {
        CUresult r = m.wait(
            (CUstream) stream,
            (CUdeviceptr) flag,
            (cuuint32_t) value,
            CU_STREAM_WAIT_VALUE_GEQ
        );
        if (r == CUDA_SUCCESS) return;
        g_memops_ok.store(false, std::memory_order_relaxed);
    }
    moe_flag_wait_kernel<<<1, 1, 0, stream>>>
    (
        reinterpret_cast<uint32_t*>(flag),
        static_cast<uint32_t>(value),
        reinterpret_cast<uint32_t*>(abort_flag)
    );
}

// ---- child worker loop --------------------------------------------------------------------

namespace {

inline void cpu_pause_()
{
#ifdef __linux__
    __builtin_ia32_pause();
#else
    _mm_pause();
#endif
}

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
    int64_t stage_threads
)
{
    // GIL is released by the binding's call_guard; do not release it again here
    uint8_t* base = reinterpret_cast<uint8_t*>(shm_base);
    uint32_t* quit = reinterpret_cast<uint32_t*>(base + 0);
    uint32_t* pass_wake = reinterpret_cast<uint32_t*>(base + 64);
    uint32_t* ready = reinterpret_cast<uint32_t*>(base + 192);
    uint32_t* jobs_tail = reinterpret_cast<uint32_t*>(base + 256);
    uint32_t* jobs_head = reinterpret_cast<uint32_t*>(base + 320);
    MoeJob* jobs = reinterpret_cast<MoeJob*>(base + MOE_CTRL_JOBS_OFFSET);
    uint32_t* data_ready = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET);
    uint32_t* done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS);
    uint32_t* stage_done = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS);
    uint32_t* pinned_free = reinterpret_cast<uint32_t*>(base + MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS + 64 * MOE_MAX_WSLOTS);
    uint32_t* stage_tail = reinterpret_cast<uint32_t*>(base + MOE_STAGE_TAIL_OFFSET);
    uint32_t* stage_head = reinterpret_cast<uint32_t*>(base + MOE_STAGE_HEAD_OFFSET);
    MoeJob* stage_jobs = reinterpret_cast<MoeJob*>(base + MOE_STAGE_JOBS_OFFSET);
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
        while (true)
        {
            if (load_acquire_u32(quit)) return;
            const uint32_t wake = load_acquire_u32(pass_wake);
            if (wake != s_last_wake) { s_last_wake = wake; s_idle = 0; }
            if (load_acquire_u32(stage_tail) == shead)
            {
                if (++s_idle < 65536) { cpu_pause_(); continue; }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            s_idle = 0;

            const MoeJob job = stage_jobs[shead % MOE_STAGE_RING];
            shead++;
            store_release_u32(stage_head, shead);

            // Wait until the slot's previous tenant has been DMA'd out
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
            store_release_u32(stage_done + size_t(job.slot) * 16, job.seq);
        }
    });

    // Fixed slot section offsets from the registered maxima
    const size_t off_x = 0;
    const size_t off_sel = align64(off_x + size_t(cap_rows) * max_hi * 2);
    const size_t off_w = align64(off_sel + size_t(cap_rows) * max_topk * 4);
    const size_t off_out = align64(off_w + size_t(cap_rows) * max_topk * 2);

    store_release_u32(ready, 1);

    uint32_t head = load_acquire_u32(jobs_head);
    uint32_t last_wake = 0;
    int idle = 0;
    while (true)
    {
        if (load_acquire_u32(quit)) break;

        const uint32_t wake = load_acquire_u32(pass_wake);
        if (wake != last_wake) { last_wake = wake; idle = 0; }

        if (load_acquire_u32(jobs_tail) == head)
        {
            // Spin hard in-pass (jobs arrive within microseconds of the GPU reaching the layer),
            // back off to naps when the queue has been dry for a while
            if (++idle < 65536) { cpu_pause_(); continue; }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        idle = 0;

        const MoeJob job = jobs[head % MOE_JOB_RING];
        head++;
        store_release_u32(jobs_head, head);

        // Wait for the GPU to publish the staged inputs for this seq
        uint32_t* drdy = data_ready + size_t(job.slot) * 16;
        while ((int32_t)(load_acquire_u32(drdy) - job.seq) < 0)
        {
            if (load_acquire_u32(quit)) goto out;
            cpu_pause_();
        }

        {
            uint8_t* slot = data + size_t(job.slot) * slot_size;
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

        store_release_u32(done + size_t(job.slot) * 16, job.seq);
    }
    out:;
    stager.join();
}
