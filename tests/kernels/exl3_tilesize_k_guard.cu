// Test-only TU that instantiates the EXL3 GEMM inner kernel at the TILESIZE_K given by
// -DPROBE_TILESIZE_K. Compiled on its own to check which values the kernel's own static_asserts
// accept, so a shape whose A-fragment swizzle would run past the end of a row cannot be built.

#include <cuda_fp16.h>
#include <cstdint>

#include "util.h"
#include "util.cuh"
#include "quant/hadamard_inner.cuh"
#include "quant/exl3_kernel_map.cuh"
#include "quant/exl3_gemm_inner.cuh"

#ifndef PROBE_TILESIZE_K
#define PROBE_TILESIZE_K 32
#endif

__global__ void probe_tilesize_k
(
    const half* A, const uint16_t* B, void* C,
    int size_m, int size_k, int size_n, int* locks, const half* post_scale
)
{
    exl3_gemm_kernel_inner<4, false, 0, 16, PROBE_TILESIZE_K, 128, 4, 3, true>
        (A, B, C, size_m, size_k, size_n, locks, post_scale);
}
