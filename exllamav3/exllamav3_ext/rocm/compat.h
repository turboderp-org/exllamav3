#pragma once

// ROCm compatibility layer, force-included (-include) into every translation unit of a ROCm build, host C++
// and HIP alike (util/cuda_flags.py, hip_include_flags). The sources reach it already hipified, so this only
// fills the gaps hipify leaves: CUDA constructs with no HIP spelling, or a different signature. The names
// stay CUDA's so the call sites are shared with the CUDA build; if a later HIP grows one of them, the build
// breaks on the redefinition and the shim here should be deleted, not renamed.

#if !defined(USE_ROCM)
    #error "rocm/compat.h is for ROCm builds only"
#endif

// HIP's host_defines.h makes __noinline__ a macro (empty for the host compiler), which breaks the C++11
// attribute spelling [[__gnu__::__noinline__]] that newer libstdc++ uses in <format> and <stacktrace>. Parse
// those headers before HIP's; their include guards make every later inclusion a no-op
#if __cplusplus >= 202002L && __has_include(<format>)
    #include <format>
#endif
#if __cplusplus >= 202302L && __has_include(<stacktrace>)
    #include <stacktrace>
#endif

// Everything that declares the intrinsics redefined below is included up front: the warp-sync macros would
// otherwise rewrite HIP's own declarations of them in a header included later
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <initializer_list>

// Host-only translation units (.cpp, compiled by the host compiler) include the kernel headers for their
// declarations; give them the CUDA qualifiers hipcc would otherwise supply

#if !defined(__HIPCC__)
    #if !defined(__align__)
        #define __align__(x) __attribute__((aligned(x)))
    #endif
    #if !defined(__forceinline__)
        #define __forceinline__ inline
    #endif
    // Device-only intrinsics referenced from inline __device__ bodies in shared headers (util.cuh). The host
    // compiler parses those bodies but never emits them, so a declaration is all it needs
    __half2 __halves2half2(__half, __half);
    __hip_bfloat162 __halves2bfloat162(__hip_bfloat16, __hip_bfloat16);
#endif

// hipify rewrites std::min to ::min for the device's sake, and HIP's global min only takes two arguments; host
// code also uses the initializer-list form

template <typename T>
__host__ inline T min(std::initializer_list<T> values)
{
    return std::min(values);
}

// HIP takes the kernel as const void* only; CUDA's templates accept typed kernel pointers

template <typename F>
__host__ inline hipError_t hipFuncSetAttribute(F* func, hipFuncAttribute attr, int value)
{
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

template <typename F>
__host__ inline hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, F* func)
{
    return hipFuncGetAttributes(attr, reinterpret_cast<const void*>(func));
}

// Graph API. hipify translates the runtime spellings but not the driver-side ones that graph.cuh uses for
// nodes captured from Triton modules; HIP has one kernel-node parameter struct for both

using cudaKernelNodeParams = hipKernelNodeParams;
using CUDA_KERNEL_NODE_PARAMS = hipKernelNodeParams;
using CUgraphNode = hipGraphNode_t;
using CUgraphExec = hipGraphExec_t;
#define cuGraphKernelNodeGetParams hipGraphKernelNodeGetParams
#define cuGraphExecKernelNodeSetParams hipGraphExecKernelNodeSetParams

// __grid_constant__ only affects how CUDA passes a const kernel parameter; HIP has no equivalent and the
// plain const parameter is the same program

#if !defined(__grid_constant__)
    #define __grid_constant__
#endif

// Carveout preference: HIP takes the percentage directly and has no enum for it

#if !defined(cudaSharedmemCarveoutMaxShared)
    #define cudaSharedmemCarveoutMaxShared 100
#endif

// Warp-sync intrinsics, mapped onto HIP's plain wave intrinsics. HIP's own *_sync forms take a 64-bit mask
// and run a convergence check loop before every operation; inside the GEMV kernels' unrolled extraction loops
// that loop never exits (the shuffle variant of exl3_gemv_kernel hangs, its shared-memory variant does not).
// Dropping the mask is exact for every call site here: each passes either the full warp or the set of lanes
// that is active at that point, and RDNA runs wave32 (arch_list.py rejects wave64 devices). __ballot_sync
// still clears the lanes outside the mask, as CUDA's does. __syncwarp keeps HIP's mask-free form, which
// fences LDS at wavefront scope (cross-lane exchange through shared memory relies on that).

#define __shfl_sync(mask, var, ...)       __shfl(var, __VA_ARGS__)
#define __shfl_up_sync(mask, var, ...)    __shfl_up(var, __VA_ARGS__)
#define __shfl_down_sync(mask, var, ...)  __shfl_down(var, __VA_ARGS__)
#define __shfl_xor_sync(mask, var, ...)   __shfl_xor(var, __VA_ARGS__)
#define __ballot_sync(mask, pred)         ((unsigned) (__ballot(pred) & (unsigned long long) (unsigned) (mask)))
#define __all_sync(mask, pred)            __all(pred)
#define __any_sync(mask, pred)            __any(pred)
#define __syncwarp(...)                   __syncwarp()

// cache/lmq.cuh picks its clamp by __CUDA_ARCH__, which HIP never defines, so the device pass would take the
// host helper. Defined first here, lmq.cuh's #ifndef keeps this one, usable from both sides

__host__ __device__ __forceinline__ int exl3_lm_clamp(int x, int lo, int hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
#define LM_CLAMP_IDX(idx, lo, hi) exl3_lm_clamp((idx), (lo), (hi))

#if defined(__HIPCC__)

// HIP has __hmax2/__hmin2 for bfloat162 only

__device__ __forceinline__ __half2 __hmax2(const __half2 a, const __half2 b)
{
    return __halves2half2(__hmax(__low2half(a), __low2half(b)), __hmax(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ __half2 __hmin2(const __half2 a, const __half2 b)
{
    return __halves2half2(__hmin(__low2half(a), __low2half(b)), __hmin(__high2half(a), __high2half(b)));
}

// bfloat16 conversions with an explicit rounding mode. HIP's __float2bfloat16 rounds to nearest even; toward
// zero is truncation of the fp32 pattern

__host__ __device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(const float f)
{
    return __float2bfloat16(f);
}

__host__ __device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rz(const float f)
{
    unsigned int u;
    __builtin_memcpy(&u, &f, sizeof(u));
    return __hip_bfloat16(__hip_bfloat16_raw{static_cast<unsigned short>(u >> 16)});
}

// __nanosleep only paces spin-waits. s_sleep takes an immediate in units of 64 clocks, so it cannot follow a
// runtime duration; the shortest sleep keeps the back-off close to the CUDA one instead of overshooting it.
// Declared for the host pass too, which parses device function bodies

__device__ __forceinline__ void __nanosleep(unsigned int)
{
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#endif
}

// __dp4a, unsigned form only (the only one the kernels use: byte sums against 0x01010101). A signed call
// fails to resolve rather than getting the wrong extension

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c)
{
    return __builtin_amdgcn_udot4(a, b, c, false);
}

// Correctly rounded fp32 arithmetic for the deterministic kernels (det_gemm.cuh and its users), whose results
// must match the CUDA build bit for bit. HIP's own __fadd_rn / __fmul_rn are plain operators, which the compiler
// is free to contract into FMAs with a neighboring operation, and its __fsqrt_rn is the native approximation
// (the correctly rounded OCML forms behind OCML_BASIC_ROUNDED_OPERATIONS are missing from the device libraries).
// Contraction is decided per operation as it is emitted, so switching it off inside these functions holds after
// inlining; __builtin_sqrtf is lowered correctly rounded, as is HIP's division (__fdiv_rn is fine as it is)

__device__ __forceinline__ float exl3_fadd_rn(float a, float b)
{
    #pragma clang fp contract(off)
    return a + b;
}
__device__ __forceinline__ float exl3_fmul_rn(float a, float b)
{
    #pragma clang fp contract(off)
    return a * b;
}
__device__ __forceinline__ float exl3_fsqrt_rn(float a)
{
    return __builtin_sqrtf(a);
}
#define __fadd_rn exl3_fadd_rn
#define __fmul_rn exl3_fmul_rn
#define __fsqrt_rn exl3_fsqrt_rn

// __ldcs (streaming, evict-first) is only a cache hint; a nontemporal load is the RDNA counterpart.
// __ldcg (cache at L2, bypassing L1) is load-bearing: the kernels use it to read values other blocks wrote,
// and RDNA's per-CU L0 is just as incoherent as NVIDIA's L1. A relaxed agent-scope atomic load reads at the
// coherent level

template <typename T>
__device__ __forceinline__ T __ldcs(const T* p)
{
    return __builtin_nontemporal_load(p);
}

template <typename T>
__device__ __forceinline__ T __ldcg(const T* p)
{
    if constexpr (sizeof(T) <= 8)
        return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    else
    {
        // Vector types: word by word (an ld.global.cg vector load is not single-copy atomic as a whole either)
        static_assert(sizeof(T) % 4 == 0);
        T v;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(p);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&v);
        #pragma unroll
        for (int i = 0; i < (int) (sizeof(T) / 4); ++i)
            dst[i] = __hip_atomic_load(src + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        return v;
    }
}

// HIP declares rsqrtf for the device only; host code computes softmax scales with it

__host__ inline float rsqrtf(float x)
{
    return 1.0f / std::sqrt(x);
}

#endif  // __HIPCC__
