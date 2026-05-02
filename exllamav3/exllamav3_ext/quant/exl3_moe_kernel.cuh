#pragma once

#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#include "exl3_moe_common.cuh"
#include "../util.h"
#include "../util.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"
#include "../ptx.cuh"

template<int t_bits, int MOE_TILESIZE_N>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;  // blockDim.x
    const int group_threads = MOE_SMS_PER_EXPERT * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    // Buffers for group
    temp_state_g += group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += group_idx * max_tokens_per_expert * intermediate_dim;

    // Barriers for group sync
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    // Loop over experts
    int start = 0;
    int end = 0;
    int expert_idx = 0;
    int expert_idx_assign = 0;
    for (; expert_idx < num_experts; ++expert_idx)
    {
        // Token span for current expert
        start = end;
        end += expert_count[expert_idx];
        int token_count = end - start;

        // Skip if no tokens or too many tokens for fused kernel (batch is handled by reconstruct path outside kernel)
        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;

        // Skip if expert is assigned to different group
        if (expert_idx_assign++ % concurrency != group_idx) continue;

        // EXL3 weights for g, u, d
        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx];
        const half* exp_gate_suh = gate_suh[expert_idx];
        const half* exp_gate_svh = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis = up_trellis[expert_idx];
        const half* exp_up_suh = up_suh[expert_idx];
        const half* exp_up_svh = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx];
        const half* exp_down_suh = down_suh[expert_idx];
        const half* exp_down_svh = down_svh[expert_idx];

        // Gather + input hadamard for g, u
        auto had_gather_gu_in = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_g + 128 * warp_idx,
                    exp_gate_suh + 128 * token_off,
                    0.088388347648f
                );
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_gather_gu_in();

        // g, u GEMM
        auto gemm_up = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    MIN(size_m, 16),    \
                    hidden_dim,         \
                    intermediate_dim,   \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    MOE_FRAG_STAGES
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, 1, SHAPE_ARGS, false>(ARGS);
                else switch(K)
                {
                    case 1: exl3_gemm_kernel_inner<1, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 2: exl3_gemm_kernel_inner<2, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 5: exl3_gemm_kernel_inner<5, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 6: exl3_gemm_kernel_inner<6, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 7: exl3_gemm_kernel_inner<7, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 8: exl3_gemm_kernel_inner<8, false, 1, SHAPE_ARGS, false>(ARGS); break;
                };
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr += 16 * hidden_dim;
                out_addr += 16 * intermediate_dim;
                size_m -= 16;
            }
        };

        gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate);
        gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);

        // Output hadamard for g, u + activation+gate + input hadamard for d
        auto had_guad = [&]()
        {
            const int warps_per_token = intermediate_dim / 128;
            const int total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner
                (
                    temp_intermediate_g + 128 * warp_idx,
                    temp_intermediate_u + 128 * warp_idx,
                    temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off,
                    exp_up_svh + 128 * token_off,
                    exp_down_suh + 128 * token_off,
                    0.088388347648f,
                    act_limit,
                    act_function
                );
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_guad();

        // d GEMM
        auto gemm_down = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    MIN(size_m, 16),    \
                    intermediate_dim,   \
                    hidden_dim,         \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    MOE_FRAG_STAGES
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, 1, SHAPE_ARGS, false>(ARGS);
                else switch(K)
                {
                    case 1: exl3_gemm_kernel_inner<1, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 2: exl3_gemm_kernel_inner<2, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 5: exl3_gemm_kernel_inner<5, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 6: exl3_gemm_kernel_inner<6, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 7: exl3_gemm_kernel_inner<7, false, 1, SHAPE_ARGS, false>(ARGS); break;
                    case 8: exl3_gemm_kernel_inner<8, false, 1, SHAPE_ARGS, false>(ARGS); break;
                };
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr += 16 * intermediate_dim;
                out_addr += 16 * hidden_dim;
                size_m -= 16;
            }
        };

        gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);
        group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);

        // Output hadamard for d + scatter add
        auto had_d_out = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            const half* weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                half weight = weights[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                float* out_ptr = output_state + token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_d_inner
                (
                    temp_state_g + 128 * warp_idx,
                    out_ptr,
                    exp_down_svh + 128 * token_off,
                    0.088388347648f * __half2float(weight)
                );
            }
            group_barrier(group_idx, MOE_SMS_PER_EXPERT, barrier_counters_sense);
        };

        had_d_out();
    }
}
