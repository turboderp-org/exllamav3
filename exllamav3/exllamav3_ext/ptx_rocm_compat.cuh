#pragma once

// AMDGCN equivalents of ptx.cuh's inline-PTX primitives. The originals the TP
// collectives use are system-scope (cross-GPU) acquire/release ops, hence the
// explicit __hip_atomic_* scopes (plain __atomic defaults to agent scope).

#include <hip/hip_runtime.h>
#include <cstdint>

// by-value struct parameters work on AMDGCN without it
#ifndef __grid_constant__
#define __grid_constant__
#endif

// no __nanosleep on HIP; s_sleep takes a constant immediate, so the backoff ladder
// becomes a fixed ~1 us yield between polls
#if !defined(__nanosleep)
#define __nanosleep(ns) ((void) (ns), __builtin_amdgcn_s_sleep(1))
#endif

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{
    return __hip_atomic_load((unsigned int*) p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{
    return __hip_atomic_load((unsigned long long*) p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{
    __hip_atomic_store((unsigned int*) p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{
    __hip_atomic_store((unsigned long long*) p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// no write-through store on AMDGCN; a system-scope release store is the closest
// equivalent for host-pinned flags
__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{
    __hip_atomic_store((unsigned int*) p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{
    return *(const volatile unsigned int*) p;
}

__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
    // volatile word loads (HIP's uint4 has no volatile copy constructor)
    const volatile unsigned int* q = reinterpret_cast<const volatile unsigned int*>(p);
    uint4 v;
    v.x = q[0];
    v.y = q[1];
    v.z = q[2];
    v.w = q[3];
    return v;
}

// no wall-clock counter on gfx1100; clock64() budgeted at a high-side 5 GHz so
// deadlines can only fire late (see parallel/timeout.cuh)
__device__ __forceinline__ uint64_t globaltimer_ns()
{
    return (uint64_t) clock64();
}

#define GLOBALTIMER_HZ 5000000000ull

// bitfield primitives with ptx.cuh's API
struct FragB { half2 elems[2]; __device__ half2& operator[](int i) { return elems[i]; } };

#define FSHF_IMM(dst, lo, hi, imm) \
    do { uint64_t _m = (static_cast<uint64_t>(hi) << 32) | static_cast<uint32_t>(lo); (dst) = static_cast<uint32_t>(_m >> (imm)); } while(0)
#define BFE16_IMM(dst, src, imm) (dst) = ((src) >> (imm)) & 0xFFFFu

static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    return static_cast<uint32_t>((value >> offset) & ((1ULL << length) - 1));
}

// cuda::atomic_ref shim so shared kernel sources compile unmodified; the real
// libcu++ wins via __has_include. Scope is not differentiated: every use here is
// agent-scope.
#if !__has_include(<cuda/atomic>)
namespace cuda
{
    enum memory_order
    {
        memory_order_relaxed = __ATOMIC_RELAXED,
        memory_order_consume = __ATOMIC_CONSUME,
        memory_order_acquire = __ATOMIC_ACQUIRE,
        memory_order_release = __ATOMIC_RELEASE,
        memory_order_acq_rel = __ATOMIC_ACQ_REL,
        memory_order_seq_cst = __ATOMIC_SEQ_CST
    };

    enum thread_scope { thread_scope_block, thread_scope_device, thread_scope_system };

    template <typename T, cuda::thread_scope Scope = cuda::thread_scope_device>
    class atomic_ref
    {
        T* ptr;
    public:
        __device__ explicit atomic_ref(T& ref) : ptr(&ref) {}
        __device__ T fetch_add(T value, cuda::memory_order order)
        {
            return __hip_atomic_fetch_add(ptr, value, (int) order, __HIP_MEMORY_SCOPE_AGENT);
        }
        __device__ T load(cuda::memory_order order)
        {
            return __hip_atomic_load(ptr, (int) order, __HIP_MEMORY_SCOPE_AGENT);
        }
        __device__ void store(T value, cuda::memory_order order)
        {
            __hip_atomic_store(ptr, value, (int) order, __HIP_MEMORY_SCOPE_AGENT);
        }
    };
}
#endif

// sense-reversing inter-block barrier; counter layout [2*group_id] = count, [+1] = sense
__device__ inline void group_barrier
(
    int group_id,
    int group_size,
    int* barrier_counters_sense
)
{
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int* counter_ptr = barrier_counters_sense + group_id * 2;
        int* sense_ptr = counter_ptr + 1;

        int old_sense = __hip_atomic_load(sense_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        int old = __hip_atomic_fetch_add(counter_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);

        if (old == group_size - 1)
        {
            __hip_atomic_store(counter_ptr, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(sense_ptr, 1 - old_sense, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        else
        {
            while (__hip_atomic_load(sense_ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == old_sense)
                __nanosleep(32);
        }
    }

    __syncthreads();
}
