#pragma once

// Cross-lane moves for the hyper-connection kernels (hc_mix.cu, hc_fuse.cuh). The CUDA forms are the warp
// shuffles the kernels were written with; RDNA replaces them with DPP row operations (rocm/hc_dpp_rdna.cuh),
// the same lanes consumed in the same order, so both backends produce bit-identical results.

#if defined(USE_ROCM)

#include "rocm/hc_dpp_rdna.cuh"

// __shfl_xor(v, O) for O in {1, 2, 4, 8, 16}; mask is the CUDA participation mask and unused here
template <int O>
__device__ __forceinline__ float hc_xor(float v, unsigned mask = 0xffffffffu)
{
    (void) mask;
    if constexpr (O == 16) return hc_permlanex16(v);
    else return hc_xor16<O>(v);
}

#else

// The __shfl_down tree (offsets 16, 8, 4, 2, 1); lane 0 ends with the sum
__device__ __forceinline__ float hc_warp_sum_lane0(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, offset);
    return v;
}

template <int O>
__device__ __forceinline__ float hc_xor(float v, unsigned mask = 0xffffffffu)
{
    return __shfl_xor_sync(mask, v, O);
}

#endif

// Butterfly trees over lane offsets O, 2O, ... below END (the unrolled loops of the sinkhorn: row sums over
// offsets 1 .. H / 2, column sums over H .. H * H / 2)
template <int O, int END>
__device__ __forceinline__ float hc_xor_sum(float v, unsigned mask)
{
    if constexpr (O < END)
    {
        v += hc_xor<O>(v, mask);
        return hc_xor_sum<O * 2, END>(v, mask);
    }
    else return v;
}

template <int O, int END>
__device__ __forceinline__ float hc_xor_max(float v, unsigned mask)
{
    if constexpr (O < END)
    {
        v = fmaxf(v, hc_xor<O>(v, mask));
        return hc_xor_max<O * 2, END>(v, mask);
    }
    else return v;
}
