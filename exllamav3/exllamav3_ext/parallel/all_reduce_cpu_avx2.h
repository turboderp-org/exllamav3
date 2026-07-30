#pragma once
#include <cstdint>
#include "context.cuh"

#define NUM_THREADS 1024
// 128KB wire chunks: large enough that per-chunk flag traffic and pipeline sync amortize against
// the PCIe copy time, small enough that a single CPU thread's accumulate (~31 GB/s wire rate)
// stays ahead of one link's copy time at 2 ranks. Also the boundary between the single-chunk
// (split kernels) and multi-chunk (striped pipeline) reduce paths
#define CPUREDUCE_CHUNK_SIZE (NUM_THREADS * 128)

void enable_fast_fp();
void enable_fast_fp_avx2();

void perform_cpu_reduce
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint32_t wire_dtype,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
);

void perform_cpu_reduce_avx2
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint32_t wire_dtype,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
);

// Basic process-safe atomic reference with acquire/release semantics, Linux/Windows compatible
template <typename T>
struct atomic_ref
{
    T* p;
    explicit atomic_ref(T* ptr) : p(ptr) {}

    T load_relaxed() const noexcept
    {
        return *p;
    }

    T load_acquire() const noexcept
    {
        #if defined(_MSC_VER) && !defined(__clang__)
            static_assert(sizeof(T) == 4, "MSVC path assumes 32-bit T");
            long v = _InterlockedCompareExchange(reinterpret_cast<volatile long*>(p), 0L, 0L);
            return static_cast<T>(static_cast<uint32_t>(v));
        #else
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        #endif
    }

    void store_release(T v)
    {
        #if defined(_MSC_VER) && !defined(__clang__)
            static_assert(sizeof(T) == 4, "MSVC path assumes 32-bit T");
            (void)_InterlockedExchange(reinterpret_cast<volatile long*>(p), static_cast<long>(static_cast<uint32_t>(v)));
        #else
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        #endif
    }
};

// True when `device` has fully published the chunk following `stage`. Single-chunk jobs (split
// kernels) release the per-device flag; multi-chunk jobs (the striped bandwidth kernel) release
// one flag per send block, and the chunk is complete only when all of them have advanced.
// `no_contrib` is valid only when the function returns true.
inline bool cpusum_device_arrived(PGContext* ctx, int device, uint32_t stage, bool multi, bool* no_contrib)
{
    if (!multi)
    {
        atomic_ref<uint32_t> f_(&ctx->cpusum_stage_device[device * REDUCE_STAGE_STRIDE]);
        uint32_t v = f_.load_acquire();
        if ((v & 0x7fffffffu) == stage) return false;
        *no_contrib = (v & 0x80000000u) != 0;
        return true;
    }
    uint32_t target = (stage + 1u) & 0x7fffffffu;
    bool nc = false;
    for (int b = 0; b < CPUREDUCE_MB_BLOCKS; ++b)
    {
        atomic_ref<uint32_t> f_(&ctx->cpusum_stage_device_mb[(device * CPUREDUCE_MB_BLOCKS + b) * REDUCE_STAGE_STRIDE]);
        uint32_t v = f_.load_acquire();
        if ((int32_t)((v & 0x7fffffffu) - target) < 0) return false;
        if (b == 0) nc = (v & 0x80000000u) != 0;
    }
    *no_contrib = nc;
    return true;
}

// True when every participating device's recv blocks have consumed the chunk whose absolute
// stage precedes `target` — the CPU's accumulator-ring throttle (only meaningful for
// multi-chunk jobs; the elastic reduce kernel has no lockstep bounding the CPU's lead)
inline bool cpusum_recv_ready(PGContext* ctx, uint32_t device_mask, uint32_t target)
{
    for (int device = 0; device < MAX_DEVICES; ++device)
    {
        if (!(device_mask & (1u << device))) continue;
        for (int b = 0; b < CPUREDUCE_MB_BLOCKS; ++b)
        {
            atomic_ref<uint32_t> f_(&ctx->cpusum_stage_recv_mb[(device * CPUREDUCE_MB_BLOCKS + b) * REDUCE_STAGE_STRIDE]);
            if ((int32_t)((f_.load_acquire() & 0x7fffffffu) - target) < 0) return false;
        }
    }
    return true;
}

// Run an accumulate split across worker threads (persistent, spawned lazily, spin-parked between
// jobs). Exactly one of fn3 (dst = a + b) / fn2 (dst += b) is non-null; slices are 128-element
// aligned. threads <= 1 or small counts run inline. Implemented in all_reduce_cpu_avx2.cpp.
void cpu_reduce_parallel
(
    void (*fn3)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*fn2)(uint16_t*, const uint16_t*, size_t),
    uint16_t* dst,
    const uint16_t* a,
    const uint16_t* b,
    size_t count,
    int threads
);
