#pragma once

#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"

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

        grid.sync();
        A = A_had;
    }

    int size_m_ = size_m;
    const half* A_ = A;
    void* C_ = C;

    while (size_m_ > 0)
    {
        exl3_gemm_kernel_inner
        <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (A_, B, C_, MIN(size_m_, TILESIZE_M), size_k, size_n, locks, svh);

        A_ += TILESIZE_M * size_k;
        if constexpr (c_fp32) C_ = (void*) (((float*) C_) + TILESIZE_M * size_n);
        else                  C_ = (void*) (((half*) C_) + TILESIZE_M * size_n);
        size_m_ -= TILESIZE_M;

        if (size_m_ > 0 || svh)
            grid.sync();
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

template<EXL3_GEMM_T_ARGS>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16)
void exl3_mgemm_kernel(EXL3_MGEMM_ARGS)
{
    int bszm = MAX(bszm_in, bszm_out);
    auto grid = cg::this_grid();

    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
        int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;
    #endif

    // Local matrix for slot j, or -1 if the slot is inactive. With min_index >= 0 the tables are
    // local to an expert shard: selections outside [min_index, max_index) are inactive and the
    // rest are rebased. Slots keep their position either way, so slot j always refers to input
    // row j, output row j and weight j
    auto slot_matrix = [&] (int j) -> int
    {
        int idx = B_indices ? (int) B_indices[j] : j;
        if (min_index >= 0 && (idx < min_index || idx >= max_index)) return -1;
        return min_index >= 0 ? idx - min_index : idx;
    };

    // List the active slots so the loop below can skip the inactive ones. The list holds slot
    // POSITIONS rather than renumbered indices, which is what lets a filtered launch keep the
    // fixed per-token slot groups that the reduction needs. One warp per chunk counts, then
    // writes at the chunk's prefix, so the grid does not wait on a serial scan
    int* slot_total = locks + MGEMM_SLOTS_OFFSET;
    int* chunk_count = slot_total + 1;
    int* slot_list = chunk_count + MGEMM_CHUNKS;
    int num_slots = bszm;

    if (min_index >= 0)
    {
        int chunk = blockIdx.z * gridDim.x + blockIdx.x;
        int num_chunks = MIN((int) (gridDim.x * gridDim.z), MGEMM_CHUNKS);
        int chunk_size = CEIL_DIVIDE(bszm, num_chunks);
        int beg = MIN(chunk * chunk_size, bszm);
        int end = MIN(beg + chunk_size, bszm);
        int lane = threadIdx.x % 32;

        if (chunk < num_chunks && threadIdx.x < 32)
        {
            int count = 0;
            for (int i = beg; i < end; i += 32)
                count += __popc(__ballot_sync(0xffffffff, i + lane < end && slot_matrix(i + lane) >= 0));
            if (lane == 0) chunk_count[chunk] = count;
        }
        __threadfence();
        grid.sync();

        if (chunk < num_chunks && threadIdx.x < 32)
        {
            int c = lane < num_chunks ? chunk_count[lane] : 0;
            int total = c;
            int base = lane < chunk ? c : 0;
            for (int s = 16; s; s >>= 1)
            {
                total += __shfl_xor_sync(0xffffffff, total, s);
                base += __shfl_xor_sync(0xffffffff, base, s);
            }
            for (int i = beg; i < end; i += 32)
            {
                bool active = i + lane < end && slot_matrix(i + lane) >= 0;
                uint32_t mask = __ballot_sync(0xffffffff, active);
                if (active) slot_list[base + __popc(mask & ((1u << lane) - 1))] = i + lane;
                base += __popc(mask);
            }
            if (chunk == 0 && lane == 0) slot_total[0] = total;
        }
        __threadfence();
        grid.sync();
        num_slots = slot_total[0];
    }

    for (int i = 0; i < num_slots; i += gridDim.z)
    {
        int p = i + blockIdx.z;
        int j = -1;
        int mat_index = -1;
        const uint16_t* B = nullptr;
        if (p < num_slots)
        {
            j = min_index >= 0 ? slot_list[p] : p;
            mat_index = slot_matrix(j);
            if (mat_index >= 0)
            {
                B = B_list[mat_index];
            }
        }

        // Had and input scales

        if (B)
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

        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
        #else
            grid.sync();
        #endif

        // Matmul. Per-matrix output width/pointer when the caller supplies the lists
        // (size_n then only sizes the per-z-slice lock ranges and must be the max width);
        // resolved once per matrix, outside all inner loops

        int n_j = (size_n_list && mat_index >= 0) ? size_n_list[mat_index] : size_n;
        int size_m_ = size_m;
        half* A_ = A_had + j * size_m * size_k;
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
                <bits, c_fp32, cb, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, false>
                (A_, B, C_, MIN(size_m_, TILESIZE_M), size_k, n_j, locks + lock_offs, nullptr);
            }

            A_ += TILESIZE_M * size_k;
            if constexpr (c_fp32) C_ = (void*) (((float*) C_) + TILESIZE_M * n_j);
            else                  C_ = (void*) (((half*) C_) + TILESIZE_M * n_j);
            size_m_ -= TILESIZE_M;

            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
                group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
            #else
                grid.sync();
            #endif
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

            for(; this_warp < total_warps; this_warp += warps_grid)
            {
                if constexpr (c_fp32)
                    had_ff_r_128_inner<false, true>
                    (
                        ((const float*) C_) + this_warp * 128,
                        ((float*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % n_j,
                        scale
                    );
                else
                    had_hf_r_128_inner<false, true>
                    (
                        ((const half*) C_) + this_warp * 128,
                        ((half*) C_) + this_warp * 128,
                        svh + (this_warp * 128) % n_j,
                        scale
                    );
            }
        }
    }

    if (B_weights)
        grid.sync();

    // Final reduction: each of the num_tokens groups of (bszm / num_tokens) contiguous slots is
    // summed into its own output row (row t for group t), instead of always collapsing into row
    // 0. num_tokens == 1 (the legacy single-token case) reduces to exactly the original
    // single-row behavior. Inactive slots produced no output and are skipped. Groups MUST be
    // processed in increasing t order per column: row t is only ever read by group
    // floor(t / stride), which is <= t, so it has already been fully read (and, if that group's
    // index equals t, is only then correctly overwritten) by the time group t's own write happens.
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
                        if (slot_matrix(t * stride + j) >= 0) sum += *C___;
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
                        if (slot_matrix(t * stride + j) >= 0) sum = __hadd(sum, *C___);
                        C___ += size_m * size_n;
                    }
                    ((half*) C)[t * size_m * size_n + col] = sum;
                }
            }
        }
    }
}
