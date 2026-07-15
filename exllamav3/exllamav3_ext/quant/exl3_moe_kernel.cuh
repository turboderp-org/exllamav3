#pragma once

#include <cuda_bf16.h>
#include <cuda/atomic>
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

template<int t_bits, int MOE_TILESIZE_N, bool a1_retile = false>
#if MOE_SMEMFIX
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16, 2)
#else
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
#endif
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int group_size = gridDim.x;  // SMs per group, set at launch
    const int num_groups = gridDim.z;
    const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;  // blockDim.x
    const int group_threads = group_size * block_threads;
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

    // Expert scheduler state, self-resetting: [0] next ticket,
    // [1] retired groups, [2 + group] ticket for the group.
    int* sched = locks + MOE_SCHED_OFFSET;

    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    // In dynamic mode each schedulable expert tile has a monotonically
    // increasing ticket. Initial tickets are the group indices; completed
    // groups greedily draw the next unclaimed tile. A negative num_active on
    // the host selects the legacy round-robin path and never touches sched[].
    int ticket = group_idx;

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

        // Skip empty experts. The default kernel retains max_tokens_per_expert
        // chunks. A1 makes every M32 route block an independent scheduler item,
        // so no group serializes multiple M tiles before another can steal work.
        if (token_count == 0) continue;
        // Eager-prefill hybrid dispatch: high-row experts are handled by the
        // reconstruct-once -> tensor-core HGEMM path after this launch.
        if (dq_threshold > 0 && token_count >= dq_threshold) continue;

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

        const int expert_start = start;
        const int expert_end = end;
        const int route_block_rows = a1_retile ? MOE_TILESIZE_M : max_tokens_per_expert;
        for (int tile_start = expert_start; tile_start < expert_end;
             tile_start += route_block_rows)
        {
            start = tile_start;
            token_count = MIN(route_block_rows, expert_end - tile_start);

            // A1 tickets route blocks at exactly M32 granularity. Each winning
            // eight-CTA group still uses the legacy K/N split and lock order;
            // only ownership of independent row tiles changes.
            const int assignment = expert_idx_assign++;
            if (use_ticket_scheduler)
            {
                if (assignment != ticket) continue;
            }
            else if (assignment % concurrency != group_idx) continue;

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
            group_barrier(group_idx, group_size, barrier_counters_sense);
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
                    MIN(size_m, MOE_TILESIZE_M), \
                    hidden_dim,         \
                    intermediate_dim,   \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    (a1_retile ? MOE_A1_SH_STAGES : MOE_SH_STAGES), \
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

                in_addr += MOE_TILESIZE_M * hidden_dim;
                out_addr += MOE_TILESIZE_M * intermediate_dim;
                size_m -= MOE_TILESIZE_M;
            }
        };

        gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate);
        gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
        group_barrier(group_idx, group_size, barrier_counters_sense);

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
            group_barrier(group_idx, group_size, barrier_counters_sense);
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
                    MIN(size_m, MOE_TILESIZE_M), \
                    intermediate_dim,   \
                    hidden_dim,         \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    (a1_retile ? MOE_A1_SH_STAGES : MOE_SH_STAGES), \
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

                in_addr += MOE_TILESIZE_M * intermediate_dim;
                out_addr += MOE_TILESIZE_M * hidden_dim;
                size_m -= MOE_TILESIZE_M;
            }
        };

        gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);
        group_barrier(group_idx, group_size, barrier_counters_sense);

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
        };

            had_d_out();

            // In dynamic mode publish the next ticket through the end-of-tile
            // barrier, which also protects the per-group temp buffers for reuse.
            // Tickets num_groups.. are claimed atomically; 0..num_groups-1 are
            // the implicit initial assignments.
            if (use_ticket_scheduler)
            {
                if (block_idx == 0 && threadIdx.x == 0)
                    sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);
                group_barrier(group_idx, group_size, barrier_counters_sense);
                ticket = sched[2 + group_idx];
            }
            else
            {
                group_barrier(group_idx, group_size, barrier_counters_sense);
            }
        }
    }

    // Last dynamic group out resets scheduler state for the next launch. The
    // acq_rel retirement orders every earlier relaxed ticket grab before reset.
    if (use_ticket_scheduler && block_idx == 0 && threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> next_ticket(sched[0]);
        cuda::atomic_ref<int, cuda::thread_scope_device> retired_groups(sched[1]);
        int retired = retired_groups.fetch_add(1, cuda::memory_order_acq_rel);
        if (retired == num_groups - 1)
        {
            next_ticket.store(0, cuda::memory_order_relaxed);
            retired_groups.store(0, cuda::memory_order_relaxed);
        }
    }
}
