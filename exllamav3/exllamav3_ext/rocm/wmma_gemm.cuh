// =============================================================================
// wmma_gemm.cuh -- WMMA GEMM backend for hgemm on RDNA3 / RDNA3.5
// =============================================================================
//
// ROCm-only. Wraps the MIT-licensed rocm_wmma_gemm kernels (Copyright (c) 2024 Adel Johar,
// vendored unchanged except for one marked edit under rocm/vendor/rocm_wmma_gemm/, see
// its LICENSE) as a faster backend for hgemm / hgemm_recon: A[M,K] fp16 row-major,
// B[K,N] fp16 row-major, C[M,N] fp32 or fp16 row-major, fp32 accumulation. fp16 output
// also accumulates in fp32 (the vendored kernel was given a separate output type for
// it), matching hipBLAS's CUBLAS_COMPUTE_32F contract that hgemm uses; the library's own
// 16-bit-accumulator kernels are not used.
//
// Selection (wmma_gemm.cu): runtime gcnArchName -> per-arch table (wmma_gemm_table.cuh)
// built from the library's tuned configs plus local tuning, then the library's own
// rule -- exact (M,N,K) hit, else the entries with the closest K, then the closest (M,N)
// by squared distance. The chosen entry's route flags decide whether the call runs here
// or on hipBLAS for its output dtype. Tables: gfx1151 (library + local tuning), gfx1100/gfx1101
// (library only: fp32 output, m >= 512). Anything else (gfx1200/1201, gfx1150, ...) has no
// table and stays on hipBLAS.
//
// Kernel bodies are compiled only for the device passes whose arch table uses them
// (EXL3_WMMA_DEV_ARCH below); in every other pass, gfx12 included, the kernel is an empty
// stub that the host never launches. That keeps build time proportional to one table and
// keeps gfx11-only WMMA builtins out of gfx12 code objects.
// =============================================================================

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Arch bit of the current device pass. Must agree with the arch bits in
// wmma_gemm_table.cuh and with wmma_gemm_arch_bit() in wmma_gemm.cu.
#define EXL3_WMMA_ARCH_GFX1151 1
#define EXL3_WMMA_ARCH_GFX1100 2
#if defined(__gfx1151__)
    #define EXL3_WMMA_DEV_ARCH EXL3_WMMA_ARCH_GFX1151
#elif defined(__gfx1100__) || defined(__gfx1101__)
    #define EXL3_WMMA_DEV_ARCH EXL3_WMMA_ARCH_GFX1100
#else
    #define EXL3_WMMA_DEV_ARCH 0   // host pass, and every arch without a table
#endif

#if EXL3_WMMA_DEV_ARCH != 0
    #include "vendor/rocm_wmma_gemm/kernel/kernel.hpp"
#endif

// One launcher per unique config; defined in the generated wmma_gemm_inst*.cu units.
typedef void (*exl3_wmma_launch_fn)
(
    void* c, const half* a, const half* b,
    int m, int n, int k,
    bool out_fp32, bool aligned,
    hipStream_t stream
);

struct Exl3WmmaCfg
{
    int warps_m, warps_n, warp_tile_m, warp_tile_n, k_slices, single_buffer, swizzle, bits;
    int arch_mask;
    exl3_wmma_launch_fn launch;
};

struct Exl3WmmaEntry
{
    int m, n, k;
    int cfg;
    int route;      // 1: route fp32 output, 2: route fp16 output (shapes where WMMA wins)
};

struct Exl3WmmaTable
{
    int arch_bit;
    const Exl3WmmaEntry* entries;
    int count;
    int min_m;      // > 0 for tables without local tuning
};

#ifdef EXL3_WMMA_INSTANTIATE

// The library builds its kernels with -mcumode (a workgroup's waves stay on one CU of the
// WGP, sharing that CU's LDS); its tuned configs assume it. setup.py compiles every TU with
// one flag set, so it is applied per kernel instead. Device pass only: the host target does
// not know the feature.
#if defined(__HIP_DEVICE_COMPILE__)
    #define EXL3_WMMA_CUMODE __attribute__((target("cumode")))
#else
    #define EXL3_WMMA_CUMODE
#endif

// Kernel entry. The k_slices == 1 variant carries the waves-per-EU hint the library's
// kernel_gemm_impl specialization uses (row-major A and B: min 2, max 8).
template <class TO, int WM, int WN, int TM, int TN, int KS, int SB, int SW, int BITS, int AL, int MASK>
__global__ __launch_bounds__(32 * WM * WN) EXL3_WMMA_CUMODE
void exl3_wmma_gemm_kernel(TO* __restrict__ C, const half* __restrict__ A, const half* __restrict__ B, int M, int N, int K)
{
    #if EXL3_WMMA_DEV_ARCH != 0
    if constexpr ((MASK & EXL3_WMMA_DEV_ARCH) != 0)
        rocm_wmma_gemm::gemm_impl<float, half,
            rocm_wmma_gemm::m_layout::row_major, rocm_wmma_gemm::m_layout::row_major, rocm_wmma_gemm::m_layout::row_major,
            WM, WN, TM, TN, KS, SB, SW, BITS, AL, TO>(C, A, B, M, N, K);
    #endif
}

template <class TO, int WM, int WN, int TM, int TN, int SB, int SW, int BITS, int AL, int MASK>
__global__ __launch_bounds__(32 * WM * WN) __attribute__((amdgpu_waves_per_eu(2, 8))) EXL3_WMMA_CUMODE
void exl3_wmma_gemm_kernel_ks1(TO* __restrict__ C, const half* __restrict__ A, const half* __restrict__ B, int M, int N, int K)
{
    #if EXL3_WMMA_DEV_ARCH != 0
    if constexpr ((MASK & EXL3_WMMA_DEV_ARCH) != 0)
        rocm_wmma_gemm::gemm_impl<float, half,
            rocm_wmma_gemm::m_layout::row_major, rocm_wmma_gemm::m_layout::row_major, rocm_wmma_gemm::m_layout::row_major,
            WM, WN, TM, TN, 1, SB, SW, BITS, AL, TO>(C, A, B, M, N, K);
    #endif
}

template <class TO, int WM, int WN, int TM, int TN, int KS, int SB, int SW, int BITS, int AL, int MASK>
static inline void exl3_wmma_launch_t(TO* c, const half* a, const half* b, int m, int n, int k, hipStream_t stream)
{
    constexpr int block_m = WM * TM * 16;
    constexpr int block_n = WN * TN * 16;
    const int grid = ((m + block_m - 1) / block_m) * ((n + block_n - 1) / block_n);
    if constexpr (KS == 1)
        exl3_wmma_gemm_kernel_ks1<TO, WM, WN, TM, TN, SB, SW, BITS, AL, MASK>
            <<<dim3(grid, 1), dim3(32 * WM * WN), 0, stream>>>(c, a, b, m, n, k);
    else
        exl3_wmma_gemm_kernel<TO, WM, WN, TM, TN, KS, SB, SW, BITS, AL, MASK>
            <<<dim3(grid, 1), dim3(32 * WM * WN), 0, stream>>>(c, a, b, m, n, k);
}

// Body of one generated launcher: fp32/fp16 output x aligned/unaligned tile edges
#define EXL3_WMMA_LAUNCHER(name, WM, WN, TM, TN, KS, SB, SW, BITS, MASK)                         \
    void name(void* c, const half* a, const half* b, int m, int n, int k,                       \
              bool out_fp32, bool aligned, hipStream_t stream)                                  \
    {                                                                                           \
        if (out_fp32)                                                                           \
        {                                                                                       \
            if (aligned) exl3_wmma_launch_t<float, WM, WN, TM, TN, KS, SB, SW, BITS, 1, MASK>   \
                             ((float*) c, a, b, m, n, k, stream);                               \
            else         exl3_wmma_launch_t<float, WM, WN, TM, TN, KS, SB, SW, BITS, 0, MASK>   \
                             ((float*) c, a, b, m, n, k, stream);                               \
        }                                                                                       \
        else                                                                                    \
        {                                                                                       \
            if (aligned) exl3_wmma_launch_t<half, WM, WN, TM, TN, KS, SB, SW, BITS, 1, MASK>    \
                             ((half*) c, a, b, m, n, k, stream);                                \
            else         exl3_wmma_launch_t<half, WM, WN, TM, TN, KS, SB, SW, BITS, 0, MASK>    \
                             ((half*) c, a, b, m, n, k, stream);                                \
        }                                                                                       \
    }

#endif  // EXL3_WMMA_INSTANTIATE

// Entry point used by hgemm_gemmex_impl (hgemm.cu). Returns false, having
// launched nothing, whenever the call is not covered; the caller then uses hipBLAS.
// No allocation and no host sync: safe under HIP graph capture.
bool wmma_gemm_try
(
    const half* a_ptr,
    const half* b_ptr,
    void* c_ptr,
    bool output_fp32,
    int size_m,
    int size_k,
    int size_n,
    int64_t c_stride_m,
    int device,
    hipStream_t stream
);
