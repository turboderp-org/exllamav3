// =============================================================================
// exl3_gemm_kernel / exl3_mgemm_kernel for RDNA -- generated from
// quant/exl3_gemm_kernel.cuh, plus one bounds guard marked RDNA below. Otherwise
// the changes listed here and nothing else. Keeping it mechanical keeps the diff
// auditable.
//
//  1. Includes point at the RDNA kernel map and GEMM inner.
//
//  2. cooperative_groups is included and aliased here. The CUDA header relies on the
//     including .cu to have done that; the RDNA comp_units include this header
//     directly, so it has to be self-contained.
//
//  3. grid.sync() is replaced by a device barrier (group_barrier on the counter pair at
//     EXL3_GRID_BARRIER_OFFSET of the lock buffer): ROCm launches these kernels as plain launches,
//     since back-to-back cooperative launches fault in the runtime. The grid is sized from occupancy
//     (at most one block per multiprocessor per autotune candidate), so every block is resident.
//     The per-z-group syncs of the multi-matrix kernel take the group_barrier form the CUDA kernel
//     uses on Blackwell.
// =============================================================================

#ifndef EXL3_ROCM_GEMM_KERNEL_RDNA_H
#define EXL3_ROCM_GEMM_KERNEL_RDNA_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>
#ifndef EXL3_ROCM_CG_ALIAS
#define EXL3_ROCM_CG_ALIAS
namespace cg = cooperative_groups;
#endif


#include "exl3_kernel_map_rdna.cuh"
#include "../../quant/hadamard_inner.cuh"
#include "exl3_gemm_inner_rdna.cuh"
#include "../../quant/exl3_devctx.cuh"

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_gemm_kernel(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();

    // if (suh)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        group_barrier(0, gridDim.x * gridDim.y * gridDim.z, locks + EXL3_GRID_BARRIER_OFFSET);
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, half_k, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (A_, B, C_, MIN(size_m_, 16), size_k, size_n, locks, svh);

        A_ += 16 * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * size_n);
        else                  C_ = (void*) (((half*) C_) + 16 * size_n);
        size_m_ -= 16;

        if (size_m_ > 0 || svh)
            group_barrier(0, gridDim.x * gridDim.y * gridDim.z, locks + EXL3_GRID_BARRIER_OFFSET);
    }

    // if (svh)
    /*
    {
        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true>
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner<false, true>
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
     */
}

#define MAX_INDICES 128

// Per translation unit, as on CUDA (the build is not relocatable-device-code)
__device__ int64_t v_indices[128];
__device__ half v_weights[128];
__device__ int bszm_sync;

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_mgemm_kernel(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    auto grid = cg::this_grid();
        int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Pack indices within min_index <= idx < max_index

    if (min_index >= 0)
    {
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
        {
            if (num_tokens > 1)
            {
                // Position-preserving mask: the grouped reduction below sums each token's
                // fixed run of (bszm / num_tokens) slots, and with bszm_in > 1 slot j also
                // addresses input row j, so out-of-range picks are marked inactive in place
                // (skipped by the compute stages and the reduction) instead of compacted away
                for (int i = 0; i < bszm; ++i)
                {
                    int idx = B_indices[i];
                    bool keep = idx >= min_index && idx < max_index;
                    v_indices[i] = keep ? idx - min_index : -1;
                    if (B_weights) v_weights[i] = keep ? B_weights[i] : __float2half(0.0f);
                }
                bszm_sync = bszm;
            }
            else
            {
                int j = 0;
                for (int i = 0; i < bszm; ++i)
                {
                    int idx = B_indices[i];
                    if (idx >= min_index && idx < max_index)
                    {
                        v_indices[j] = idx - min_index;
                        if (B_weights) v_weights[j] = B_weights[i];
                        j++;
                    }
                }
                bszm_sync = j;
                for (; j < bszm; ++j)
                {
                    v_indices[j] = -1;
                }
            }
        }
        __threadfence();
        group_barrier(0, gridDim.x * gridDim.y * gridDim.z, locks + EXL3_GRID_BARRIER_OFFSET);
        B_indices = v_indices;
        if (B_weights) B_weights = v_weights;
        bszm = bszm_sync;
    }

    // Sliced mode: the entries are equal-width column slices of fewer source matrices, so the
    // input transform runs once per source (suh_list is per source, A_had holds one slab per
    // source) with every block cooperating, and each slice then reads its source's slab
    if (had_src_list)
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * gridDim.z * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * (blockIdx.x + gridDim.x * blockIdx.z);
        for (; this_warp < num_had_src * total_warps; this_warp += warps_grid)
        {
            int src = this_warp / total_warps;
            int w = this_warp - src * total_warps;
            had_hf_r_128_inner<true, false>
            (
                A + w * 128,
                A_had + src * size_m * size_k + w * 128,
                suh_list[src] + (w * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );
        }
        group_barrier(0, gridDim.x * gridDim.y * gridDim.z, locks + EXL3_GRID_BARRIER_OFFSET);
    }

    for (int i = 0; i < bszm; i += gridDim.z)
    {
        int j = i + blockIdx.z;
        int mat_index = -1;
        const uint16_t* B = nullptr;
        if (j >= bszm) j = -1;
        else
        {
            mat_index = B_indices ? (int) B_indices[j] : j;
            if (mat_index >= 0)
            {
                B = B_list[mat_index];
            }
        }

        // Had and input scales (sliced mode: done once per source above)

        if (B && !had_src_list)
        {
            int total_warps = size_m * size_k / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* suh = suh_list[mat_index];
            const half* A_ = bszm_in == 1 ? A : A + j * size_m * size_k;
            half* A_had_ = A_had + j * size_m * size_k;

            for(; this_warp < total_warps; this_warp += warps_grid)
                had_hf_r_128_inner<true, false>
                (
                    A_ + this_warp * 128,
                    A_had_ + this_warp * 128,
                    suh + (this_warp * 128) % size_k,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);

        // Matmul. Per-matrix output width/pointer when the caller supplies the lists
        // (size_n then only sizes the per-z-slice lock ranges and must be the max width);
        // resolved once per matrix, outside all inner loops

        int n_j = (size_n_list && mat_index >= 0) ? size_n_list[mat_index] : size_n;
        // Column slices write into a wider row (their source matrix's full width)
        int n_stride_j = (n_stride_list && mat_index >= 0) ? n_stride_list[mat_index] : n_j;
        int size_m_ = size_m;
        // RDNA: guarded on mat_index >= 0 (the CUDA kernel indexes had_src_list[-1] for the idle
        // z-slots past bszm; A_ is unused there, but the read is out of bounds)
        half* A_ = A_had + ((had_src_list && mat_index >= 0) ? had_src_list[mat_index] : j) * size_m * size_k;
        void* C_;
        if (C_list && mat_index >= 0) C_ = C_list[mat_index];
        else if constexpr (c_fp32) C_ = (void*) (((float*) C) + j * size_m * size_n);
        else                       C_ = (void*) (((half*) C) + j * size_m * size_n);
        void* C_base = C_;

        while (size_m_ > 0)
        {
            if (B)
            {
                int lock_offs = blockIdx.z * size_n / 128;

                exl3_gemm_kernel_inner
                <bits, half_k, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false>
                (A_, B, C_, MIN(size_m_, 16), size_k, n_j, locks + lock_offs, nullptr, n_stride_j);
            }

            A_ += 16 * size_k;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_) + 16 * n_stride_j);
            else                  C_ = (void*) (((half*) C_) + 16 * n_stride_j);
            size_m_ -= 16;
                group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
        }

        // Had and output scales

        if (B)
        {
            int total_warps = size_m * n_j / 128;
            int warps_grid = gridDim.x * blockDim.x / 32;
            int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

            const half* svh = svh_list[mat_index];
            float scale = 0.088388347648f;  // 1/sqrt(128)
            if (B_weights) scale *= __half2float(B_weights[j]);

            C_ = C_base;

            int cols = n_j / 128;
            for(; this_warp < total_warps; this_warp += warps_grid)
            {
                int row = this_warp / cols;
                int col = this_warp - row * cols;
                int offs = row * n_stride_j + col * 128;
                if constexpr (c_fp32)
                    had_ff_r_128_inner<false, true>
                    (
                        ((const float*) C_) + offs,
                        ((float*) C_) + offs,
                        svh + col * 128,
                        scale
                    );
                else
                    had_hf_r_128_inner<false, true>
                    (
                        ((const half*) C_) + offs,
                        ((half*) C_) + offs,
                        svh + col * 128,
                        scale
                    );
            }
        }
    }

    if (B_weights)
        group_barrier(0, gridDim.x * gridDim.y * gridDim.z, locks + EXL3_GRID_BARRIER_OFFSET);

    // Final reduction: each of the num_tokens groups of (bszm / num_tokens) contiguous slots is
    // summed into its own output row (row t for group t), instead of always collapsing into row
    // 0. num_tokens == 1 (the legacy single-token case) reduces to exactly the original
    // single-row behavior. Groups MUST be processed in increasing t order per column: row t is
    // only ever read by group floor(t / stride), which is <= t, so it has already been fully
    // read (and, if that group's index equals t, is only then correctly overwritten) by the time
    // group t's own write happens.
    if (B_weights && blockIdx.z == 0)
    {
        int total_warps = size_m * size_n / 32;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        int this_lane = threadIdx.x % 32;
        int stride = bszm / num_tokens;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            for (int t = 0; t < num_tokens; ++t)
            {
                int col = this_warp * 32 + this_lane;
                if constexpr (c_fp32)
                {
                    float* C___ = ((float*) C) + t * stride * size_m * size_n + col;
                    float sum = 0.0f;
                    for (int j = 0; j < stride; ++j)
                    {
                        // Inactive slots (masked by range filtering, or -1 selections) were
                        // never written by the compute stages: their scratch is stale
                        if (!B_indices || B_indices[t * stride + j] >= 0)
                            sum += *C___;
                        C___ += size_m * size_n;
                    }
                    ((float*) C)[t * size_m * size_n + col] = sum;
                }
                else
                {
                    half* C___ = ((half*) C) + t * stride * size_m * size_n + col;
                    half sum = {};
                    for (int j = 0; j < stride; ++j)
                    {
                        if (!B_indices || B_indices[t * stride + j] >= 0)
                            sum = __hadd(sum, *C___);
                        C___ += size_m * size_n;
                    }
                    ((half*) C)[t * size_m * size_n + col] = sum;
                }
            }
        }
    }
}

#endif // EXL3_ROCM_GEMM_KERNEL_RDNA_H
