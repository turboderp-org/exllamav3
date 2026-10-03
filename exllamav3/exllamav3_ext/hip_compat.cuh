#pragma once

// HIP/ROCm compatibility shims for CUDA constructs that hipify does not translate.
// Included from ptx.cuh and util.cuh; also standalone from headers that use warp
// intrinsics. The qualifier fallbacks below let host-only TUs (.cpp under gcc)
// parse this header; under nvcc/hipcc the real definitions already exist.

#if !defined(__HIPCC__) && !defined(__CUDACC__)
#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif
#if !defined(__device__)
#define __device__
#endif
#if !defined(__host__)
#define __host__
#endif
#if !defined(__forceinline__)
#define __forceinline__ inline
#endif
#if !defined(__global__)
#define __global__
#endif
#endif

// Tensor core fragment types shared with ptx.cuh. Defined here (outside the
// USE_ROCM guard) so this header can be included standalone from any file that
// needs the warp-intrinsic shims below.

template <typename T, int n>
struct Vec
{
    T elems[n];
    __device__ T& operator[](int i) { return elems[i]; }
    __device__ const T& operator[](int i) const { return elems[i]; }
};

#if defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragC_h = Vec<half2, 2>;

#define EXL3_FULL_WARP_MASK 0xffffffffffffffffULL

// Widen the mask argument of the *_sync warp builtins. The macro name is not
// re-expanded inside its own replacement list, so these forward to the real
// HIP builtins.

#define __shfl_sync(mask, ...)        __shfl_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_up_sync(mask, ...)     __shfl_up_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_down_sync(mask, ...)   __shfl_down_sync((unsigned long long)(mask), __VA_ARGS__)
#define __shfl_xor_sync(mask, ...)    __shfl_xor_sync((unsigned long long)(mask), __VA_ARGS__)
#define __ballot_sync(mask, ...)      __ballot_sync((unsigned long long)(mask), __VA_ARGS__)
#define __all_sync(mask, ...)         __all_sync((unsigned long long)(mask), __VA_ARGS__)
#define __any_sync(mask, ...)         __any_sync((unsigned long long)(mask), __VA_ARGS__)
#define EXL3_SYNCWARP_0()             __syncwarp()
#define EXL3_SYNCWARP_1(m)            __syncwarp((unsigned long long)(m))
#define EXL3_SYNCWARP_PICK(_0, _1, NAME, ...) NAME
#define __syncwarp(...)               EXL3_SYNCWARP_PICK(_, ##__VA_ARGS__, EXL3_SYNCWARP_1, EXL3_SYNCWARP_0)(__VA_ARGS__)
#define __reduce_add_sync(mask, ...)  __reduce_add_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_min_sync(mask, ...)  __reduce_min_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_max_sync(mask, ...)  __reduce_max_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_or_sync(mask, ...)   __reduce_or_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_and_sync(mask, ...)  __reduce_and_sync((unsigned long long)(mask), __VA_ARGS__)
#define __reduce_xor_sync(mask, ...)  __reduce_xor_sync((unsigned long long)(mask), __VA_ARGS__)

// __grid_constant__ is CUDA-only; a plain const kernel argument is the portable form.

#ifndef __grid_constant__
#define __grid_constant__
#endif

// hipify does not map the carveout enum; HIP takes a plain percentage (0-100).

#ifndef cudaSharedmemCarveoutMaxShared
#define cudaSharedmemCarveoutMaxShared 100
#endif

#if defined(__HIPCC__)

// The functions below intentionally keep the CUDA builtin names so call sites stay
// identical across platforms. If a future HIP version adds any of them, the build
// breaks loudly (redefinition) — delete the shim then; do not rename the call sites.

// HIP provides __hmax/__hmin for __half but not the __half2 forms.

__device__ __forceinline__ __half2 __hmax2(__half2 a, __half2 b)
{
    return __halves2half2(__hmax(__low2half(a), __low2half(b)),
                          __hmax(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ __half2 __hmin2(__half2 a, __half2 b)
{
    return __halves2half2(__hmin(__low2half(a), __low2half(b)),
                          __hmin(__high2half(a), __high2half(b)));
}

// hipify does not map the _rn/_rz-suffixed bf16 conversions; HIP's __float2bfloat16
// is already round-to-nearest, and truncation is the _rz form.

__device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rn(float f)
{
    return __float2bfloat16(f);
}

__device__ __forceinline__ __hip_bfloat16 __float2bfloat16_rz(float f)
{
    return __ushort_as_bfloat16((unsigned short) (__float_as_uint(f) >> 16));
}

__device__ __forceinline__ void __nanosleep(uint32_t ns)
{
    // s_sleep takes a constant cycle count (max 127); ~1ns/cycle at ~1GHz.
    // Use 127-cycle chunks to minimize iteration count and oversleep ratio.
    int iters = (int)(ns / 127) + 1;
    while (iters-- > 0) __builtin_amdgcn_s_sleep(127);
}

// __dp4a: 4-way byte dot product. gfx11+ has v_dot4_i32_iu8 / v_dot4_u32_u8
// (exposed as __builtin_amdgcn_sudot4 / __builtin_amdgcn_udot4); the signed×signed
// form is expressed via the u8 bias trick: s8(x) = u8(x^0x80) - 128 per byte.
// Older targets fall back to per-byte multiply-adds. Signedness follows the CUDA
// overloads.

#if defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_sudot4)
#define EXL3_HAS_DOT4 1
#endif
#endif

#if defined(EXL3_HAS_DOT4)

__device__ __forceinline__ int __dp4a(int a, int b, int c)
{
    // s8·s8 = s8·(u8^0x80) - 128·Σs8(a); Σs8(a) = s8(a)·u8(1)
    return __builtin_amdgcn_sudot4(true, a, false, b ^ 0x80808080, c, 0)
         - 128 * __builtin_amdgcn_sudot4(true, a, false, 0x01010101, 0, 0);
}

__device__ __forceinline__ int __dp4a(unsigned int a, int b, int c)
{
    return __builtin_amdgcn_sudot4(false, (int) a, true, b, c, 0);
}

__device__ __forceinline__ int __dp4a(int a, unsigned int b, int c)
{
    return __builtin_amdgcn_sudot4(true, a, false, (int) b, c, 0);
}

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c)
{
    return __builtin_amdgcn_udot4(a, b, c, false);
}

#else

__device__ __forceinline__ int __dp4a(int a, int b, int c)
{
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += (int)(int8_t) ((a >> (8 * i)) & 0xff) * (int)(int8_t) ((b >> (8 * i)) & 0xff);
    return r;
}

__device__ __forceinline__ int __dp4a(unsigned int a, int b, int c)
{
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += (int) ((a >> (8 * i)) & 0xff) * (int)(int8_t) ((b >> (8 * i)) & 0xff);
    return r;
}

__device__ __forceinline__ int __dp4a(int a, unsigned int b, int c)
{
    int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += (int)(int8_t) ((a >> (8 * i)) & 0xff) * (int) ((b >> (8 * i)) & 0xff);
    return r;
}

__device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c)
{
    unsigned int r = c;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        r += ((a >> (8 * i)) & 0xff) * ((b >> (8 * i)) & 0xff);
    return r;
}

#endif  // EXL3_HAS_DOT4

// L2-coherent loads (ld.global.cg). On AMD a plain load is L2-coherent; the hint only
// affects L1, so these degrade to normal loads.

__device__ __forceinline__ int __ldcg(const int* p) { return *p; }
__device__ __forceinline__ uint32_t __ldcg(const uint32_t* p) { return *p; }
__device__ __forceinline__ uint2 __ldcg(const uint2* p) { return *p; }
__device__ __forceinline__ uint4 __ldcg(const uint4* p) { return *p; }
__device__ __forceinline__ float4 __ldcg(const float4* p) { return *p; }

// Streaming/volatile global accesses. On AMD, __builtin_nontemporal_* generates
// loads/stores with .slc (streaming) cache hints. For 128-bit ops, use a single
// flat_load_dwordx4 / flat_store_dwordx4 via reinterpret_cast to a vector type
// instead of four separate 32-bit ops.

__device__ __forceinline__ uint32_t __ldcs(const uint32_t* p)
{
    return __builtin_nontemporal_load(p);
}

__device__ __forceinline__ uint64_t __ldcs(const uint64_t* p)
{
    return __builtin_nontemporal_load(p);
}

__device__ __forceinline__ void __stwt(uint32_t* p, uint32_t v)
{
    __builtin_nontemporal_store(v, p);
}

__device__ __forceinline__ void __stwt(uint64_t* p, uint64_t v)
{
    __builtin_nontemporal_store(v, p);
}

__device__ __forceinline__ void stg_wt_u32(uint32_t* p, uint32_t v)
{
    __builtin_nontemporal_store(v, p);
}

__device__ __forceinline__ void stg_wt_u128(uint4* p, const uint4 v)
{
    // Single 128-bit nontemporal store via vector reinterpret. One instruction, but
    // single-copy atomicity is not guaranteed the way PTX st.global.wt.v4.u32 is;
    // do not use for flag+data packed words.
    typedef int __attribute__((ext_vector_type(4))) int4_v;
    *reinterpret_cast<int4_v*>(p) = *reinterpret_cast<const int4_v*>(&v);
}

__device__ __forceinline__ uint32_t ldg_cv_u32(const uint32_t* p)
{
    return __builtin_nontemporal_load(p);
}

__device__ __forceinline__ uint4 ldg_cv_u128(const uint4* p)
{
    // Single 128-bit load. On AMD a plain 128-bit load is more efficient than
    // four 32-bit nontemporal loads and the cache hint matters less with the
    // unified L2 hierarchy.
    return *p;
}

// System-scope acquire/release for the TP collectives (ll.cuh, barrier_inner.cuh).
// __atomic_* builtins emit agent-scope (device) atomics on AMDGCN; the __hip_atomic_*
// forms take an explicit scope, and these paths synchronize with the host and peer
// GPUs, so they need __HIP_MEMORY_SCOPE_SYSTEM.

__device__ __forceinline__ uint32_t ldg_acquire_sys_u32(const uint32_t* p)
{
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t ldg_acquire_sys_u64(const uint64_t* p)
{
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void stg_release_sys_u32(uint32_t* p, uint32_t v)
{
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void stg_release_sys_u64(uint64_t* p, uint64_t v)
{
    __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Device-scope acquire/release used by the inter-block barriers in ptx.cuh.

__device__ __forceinline__ int ldg_acquire_gpu_i32(const int* p)
{
    return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void red_relaxed_gpu_add_i32(int* p, int v)
{
    __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Global real-time clock in nanoseconds (matching %globaltimer). CDNA has
// s_memrealtime (ns); RDNA has the 100 MHz steady counter instead.

__device__ __forceinline__ uint64_t globaltimer_ns()
{
#if __AMDGCN_WAVEFRONT_SIZE == 64
    return (uint64_t) __builtin_amdgcn_s_memrealtime();
#else
    return (uint64_t) __builtin_readsteadycounter() * 10;
#endif
}

// Bitfield extract / funnel shift used by the trellis decode paths.

__device__ __forceinline__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    uint64_t mask = (length >= 64) ? ~0ULL : ((1ULL << length) - 1ULL);
    return static_cast<uint32_t>((value >> offset) & mask);
}

#define FSHF_IMM(dst, lo, hi, imm) dst = __funnelshift_r((lo), (hi), (imm))
#define BFE16_IMM(dst, src, imm) dst = ((src) >> (imm)) & 0xffff

// __fns: Find nth set bit in mask starting from position prev+1.
// CUDA-only intrinsic; emulate with ctz + bit-clear loop for ROCm.
// Returns the bit position (0-indexed) or -1 if not found.

__device__ __forceinline__ int __fns(unsigned int mask, int prev, int n)
{
    if (prev >= 0)
    {
        if (prev >= 31) return -1;
        mask &= ~((1u << (prev + 1)) - 1);  // Clear bits 0..prev
    }
    for (int i = 1; i < n && mask; ++i)
        mask &= mask - 1;  // Clear (n-1) lowest set bits
    return mask ? __builtin_ctz(mask) : -1;
}

// lop3.b32: 3-input bitwise LUT. The immediate selects the output bit for each
// combination of (a, b, c); with a compile-time imm this folds to a few logic ops.

__device__ __forceinline__ uint32_t lop3_u32(uint32_t a, uint32_t b, uint32_t c, uint32_t imm)
{
    uint32_t r = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        if (imm & (1u << i))
            r |= ((i & 4) ? a : ~a) & ((i & 2) ? b : ~b) & ((i & 1) ? c : ~c);
    return r;
}

// cp.async emulation: synchronous 16-byte copies. cp_async_fence/wait become
// no-ops; correctness relies on the __syncthreads() the kernels already issue
// before consuming staged data.
//
// True async global-to-LDS copies on gfx11 require inline asm (global_load_lds
// or buffer_load_lds with s_waitcnt vmcnt tracking). The __builtin_amdgcn_global_load_lds
// builtin is gated on vmem-to-lds-load-insts (gfx9/CDNA only). Future optimization:
// implement via inline asm when RDNA3 hardware is available for testing.
//
// The load and store are split into separate statements to give the compiler
// more scheduling freedom — the global load is non-blocking on AMD, so the
// compiler can overlap it with preceding computation.

__device__ __forceinline__ void cp_async_pred(void* smem_ptr, const void* glob_ptr, bool pred = true)
{
    if (pred)
    {
        uint4 data = *((const uint4*) glob_ptr);
        *((uint4*) smem_ptr) = data;
    }
}

__device__ __forceinline__ void cp_async(void* smem_ptr, const void* glob_ptr)
{
    uint4 data = *((const uint4*) glob_ptr);
    *((uint4*) smem_ptr) = data;
}

__device__ __forceinline__ void cp_async_stream(void* smem_ptr, const void* glob_ptr)
{
    uint4 data = *((const uint4*) glob_ptr);
    *((uint4*) smem_ptr) = data;
}

__device__ __forceinline__ void cp_async_fence() {}

template <int n>
__device__ __forceinline__ void cp_async_wait() {}

// ldmatrix.x4 emulation: lane l receives row l/4 of each of the four 8x8 b16
// matrices, 4 bytes at column (l%4)*2. The row base addresses are supplied by
// lanes 0..31 (lanes 8i+r hold matrix i row r) and fetched via shuffles.

__device__ __forceinline__ void ldsm4(FragA& frag_a, const void* smem_ptr)
{
    int lane = threadIdx.x & 31;
    uintptr_t base = (uintptr_t) smem_ptr;
    int row = lane >> 2;
    int col = (lane & 3) * 4;
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        // Lane 8*i + row holds the row pointer for matrix i
        uintptr_t row_ptr = (uintptr_t) __shfl_sync(
            EXL3_FULL_WARP_MASK, (unsigned long long) base, 8 * i + row);
        a[i] = *((const uint32_t*) (row_ptr + col));
    }
}

// mma.m16n8k16 emulation via shuffles + FMA. Fragment layouts follow the PTX doc
// (and the ldmatrix.x4 order TL, BL, TR, BR):
//   A (row-major 16x16): lane l = (g, t) = (l/4, l%4); a[j] is a half2 holding
//     A[g + 8*(j%2)][2t + 8*(j/2)] and A[g + 8*(j%2)][2t + 8*(j/2) + 1].
//   B (col-major 16x8):  b[i] is a half2 holding B[2t + 8*i][g] and B[2t + 8*i + 1][g].
//   C/D (16x8):          c[i] holds D[g + 8*(i/2)][2t + (i%2)].
// Each lane gathers its two A rows and two B columns (16 half2 shuffles total),
// then runs 16 FMAs per output element.

__device__ __forceinline__ half2 shfl_h2(half2 v, int src)
{
    // Use __builtin_bit_cast to avoid taking the address of v, which would
    // force a stack spill and cause __shfl_sync to read from memory instead of VGPR.
    uint32_t u = __builtin_bit_cast(uint32_t, v);
    u = __shfl_sync(EXL3_FULL_WARP_MASK, u, src);
    return __builtin_bit_cast(half2, u);
}

__device__ __forceinline__ void mma_m16n8k16_gather
(
    const FragA& frag_a,
    const FragB& frag_b,
    half2 (&a_rows)[2][8],
    half2 (&b_cols)[2][8]
)
{
    int lane = threadIdx.x & 31;
    int g = lane >> 2, t = lane & 3;

    // A row m lives in lanes 4*(m%8)..4*(m%8)+3. For row g: k pairs 0..7 come from
    // frag_a[0], k pairs 8..15 from frag_a[2]; for row g+8: frag_a[1] / frag_a[3].
    // Lane 4g+i holds k = 2i, 2i+1.
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        a_rows[0][i]     = shfl_h2(frag_a[0], 4 * g + i);
        a_rows[0][4 + i] = shfl_h2(frag_a[2], 4 * g + i);
        a_rows[1][i]     = shfl_h2(frag_a[1], 4 * g + i);
        a_rows[1][4 + i] = shfl_h2(frag_a[3], 4 * g + i);
    }

    // B column n lives in lanes 4n..4n+3: lane 4n+i holds k = 2i, 2i+1 in b[0]
    // and k = 2i+8, 2i+9 in b[1]. Columns 2t and 2t+1 are at lane bases 8t, 8t+4.
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        b_cols[0][i]     = shfl_h2(frag_b[0], 8 * t + i);
        b_cols[0][4 + i] = shfl_h2(frag_b[1], 8 * t + i);
        b_cols[1][i]     = shfl_h2(frag_b[0], 8 * t + 4 + i);
        b_cols[1][4 + i] = shfl_h2(frag_b[1], 8 * t + 4 + i);
    }
}

__device__ __forceinline__ void mma_m16n8k16_f32_emu
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC& frag_c
)
{
    half2 a_rows[2][8], b_cols[2][8];
    mma_m16n8k16_gather(frag_a, frag_b, a_rows, b_cols);

    float* c = reinterpret_cast<float*>(&frag_c);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        c[0] += __half2float(__low2half(a_rows[0][i]))  * __half2float(__low2half(b_cols[0][i]));
        c[0] += __half2float(__high2half(a_rows[0][i])) * __half2float(__high2half(b_cols[0][i]));
        c[1] += __half2float(__low2half(a_rows[0][i]))  * __half2float(__low2half(b_cols[1][i]));
        c[1] += __half2float(__high2half(a_rows[0][i])) * __half2float(__high2half(b_cols[1][i]));
        c[2] += __half2float(__low2half(a_rows[1][i]))  * __half2float(__low2half(b_cols[0][i]));
        c[2] += __half2float(__high2half(a_rows[1][i])) * __half2float(__high2half(b_cols[0][i]));
        c[3] += __half2float(__low2half(a_rows[1][i]))  * __half2float(__low2half(b_cols[1][i]));
        c[3] += __half2float(__high2half(a_rows[1][i])) * __half2float(__high2half(b_cols[1][i]));
    }
}

__device__ __forceinline__ void mma_m16n8k16_f16_emu
(
    const FragA& frag_a,
    const FragB& frag_b,
    FragC_h& frag_c
)
{
    half2 a_rows[2][8], b_cols[2][8];
    mma_m16n8k16_gather(frag_a, frag_b, a_rows, b_cols);

    // Accumulate each output element in a half2 (even/odd k in the two lanes),
    // then fold. c[0] = {D[g][2t], D[g][2t+1]}, c[1] = {D[g+8][2t], D[g+8][2t+1]}.
    half2* c = reinterpret_cast<half2*>(&frag_c);
    half2 acc00 = __float2half2_rn(0.0f), acc01 = acc00, acc10 = acc00, acc11 = acc00;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        acc00 = __hfma2(a_rows[0][i], b_cols[0][i], acc00);
        acc01 = __hfma2(a_rows[0][i], b_cols[1][i], acc01);
        acc10 = __hfma2(a_rows[1][i], b_cols[0][i], acc10);
        acc11 = __hfma2(a_rows[1][i], b_cols[1][i], acc11);
    }
    c[0] = __hadd2(c[0], __halves2half2(
        __hadd(__low2half(acc00), __high2half(acc00)),
        __hadd(__low2half(acc01), __high2half(acc01))));
    c[1] = __hadd2(c[1], __halves2half2(
        __hadd(__low2half(acc10), __high2half(acc10)),
        __hadd(__low2half(acc11), __high2half(acc11))));
}



#endif  // __HIPCC__
#endif  // USE_ROCM
