#pragma once
#include <cstdint>
#include "context.cuh"

#define NUM_THREADS 1024
#define CPUREDUCE_CHUNK_SIZE (NUM_THREADS * 16)

void enable_fast_fp();

void perform_cpu_reduce
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
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
            return (T)_InterlockedCompareExchange(reinterpret_cast<volatile T*>(p), 0, 0);
        #else
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        #endif
    }

    void store_release(T v)
    {
        #if defined(_MSC_VER) && !defined(__clang__)
            _InterlockedExchange(reinterpret_cast<volatile T*>(p), (T)v);
        #else
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        #endif
    }
};
