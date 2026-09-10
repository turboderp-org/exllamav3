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

// Prefill tile dispatch for one expert's rows. Small experts pay M=16's low fixed cost; large
// ones amortise B decode over M=64. All tiers use the MoE-only _mt inner; the shared dense inner
// is untouched. FS is the fragment pipeline depth per tier.
template<int t_bits, int M_TILE, int N_TILE, int cb, int FS>
__device__ __forceinline__
void moe_gemm_tile
(
    const half* __restrict__ in_addr,
    const uint16_t* __restrict__ trellis,
    half* __restrict__ out_addr,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const int K
)
{
    #define MOE_TILE_ARGS \
        in_addr, trellis, out_addr, MIN(size_m, M_TILE), size_k, size_n, locks, nullptr
    #define MOE_TILE_SHAPE \
        M_TILE, MOE_TILESIZE_K, N_TILE, MOE_SH_STAGES, FS
    if constexpr (t_bits)
        exl3_gemm_kernel_inner_mt<t_bits, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS);
    else switch (K)
    {
        case 1: exl3_gemm_kernel_inner_mt<1, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
        case 2: exl3_gemm_kernel_inner_mt<2, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
        case 3: exl3_gemm_kernel_inner_mt<3, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
        case 4: exl3_gemm_kernel_inner_mt<4, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
        case 5: exl3_gemm_kernel_inner_mt<5, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
        case 6: exl3_gemm_kernel_inner_mt<6, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
        case 7: exl3_gemm_kernel_inner_mt<7, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
        case 8: exl3_gemm_kernel_inner_mt<8, false, cb, MOE_TILE_SHAPE, false>(MOE_TILE_ARGS); break;
    }
    #undef MOE_TILE_ARGS
    #undef MOE_TILE_SHAPE
}

// Whole-expert loop for the M=32/64 tiers, outlined so their (larger) register frames do not
// inflate the kernel's live set. The M=16 tier is left inline at the call site: it is small and
// every small expert pays a call otherwise.
template<int t_bits, int M_TILE, int N_TILE, int cb, int FS>
__device__ __noinline__
void moe_gemm_rows
(
    const half* __restrict__ in_addr,
    const uint16_t* __restrict__ trellis,
    half* __restrict__ out_addr,
    int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const int K
)
{
    while (size_m > 0)
    {
        moe_gemm_tile<t_bits, M_TILE, N_TILE, cb, FS>
        (in_addr, trellis, out_addr, size_m, size_k, size_n, locks, K);
        in_addr += M_TILE * size_k;
        out_addr += M_TILE * size_n;
        size_m -= M_TILE;
    }
}

template<int t_bits, int MOE_TILESIZE_N, int cb>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int group_size = gridDim.x;  // SMs per expert, set at launch
    const int num_groups = gridDim.z;
    const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;  // blockDim.x
    const int group_threads = group_size * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    // Per-expert tile dispatch thresholds and the large tile for this N shape. N=128 fits M=64
    // with single-buffered A (FS=2 to chase STACK 0); N=256's reduction scratch does not, so its
    // large tier stays M=32.
    constexpr int MOE_BIG_M = (MOE_TILESIZE_N == 128) ? 64 : MOE_TILESIZE_M;
    constexpr int MOE_BIG_FS = (MOE_TILESIZE_N == 128) ? 2 : MOE_FRAG_STAGES;

    // Buffers for group
    temp_state_g += group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += group_idx * max_tokens_per_expert * intermediate_dim;

    // Barriers for group sync
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Expert scheduler state, self-resetting: [0] next ticket, [1] retired groups, [2 + g] ticket for group g
    int* sched = locks + MOE_SCHED_OFFSET;

    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    // Dynamic expert assignment: active experts are numbered in scan order, and each group processes the active
    // expert matching its current ticket. Initial tickets are the group indices; after finishing an expert, a group
    // draws the next unclaimed ticket, so load balances greedily without assuming uniform cost per expert
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

        // Skip if no tokens or too many tokens for fused kernel (batch is handled by reconstruct path outside kernel)
        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;

        // Skip if expert is claimed by a different group
        if (expert_idx_assign++ != ticket) continue;

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

        // Gather + input hadamard for g, u. Non-gated mode skips the g staging (and the g GEMM
        // below); the activation synthesizes the gate lane from u
        const bool gated = act_function != MOE_ACT_RELU2_NOGATE;
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
                if (gated)
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
            if (token_count <= 16)
                moe_gemm_tile<t_bits, 16, MOE_TILESIZE_N, cb, MOE_FRAG_STAGES>
                (in_addr, trellis, out_addr, token_count, hidden_dim, intermediate_dim, locks, K);
            else if (token_count <= 48)
                moe_gemm_rows<t_bits, 32, MOE_TILESIZE_N, cb, MOE_FRAG_STAGES>
                (in_addr, trellis, out_addr, token_count, hidden_dim, intermediate_dim, locks, K);
            else
                moe_gemm_rows<t_bits, MOE_BIG_M, MOE_TILESIZE_N, cb, MOE_BIG_FS>
                (in_addr, trellis, out_addr, token_count, hidden_dim, intermediate_dim, locks, K);
        };

        if (gated)
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
            if (token_count <= 16)
                moe_gemm_tile<t_bits, 16, MOE_TILESIZE_N, cb, MOE_FRAG_STAGES>
                (in_addr, trellis, out_addr, token_count, intermediate_dim, hidden_dim, locks, K);
            else if (token_count <= 48)
                moe_gemm_rows<t_bits, 32, MOE_TILESIZE_N, cb, MOE_FRAG_STAGES>
                (in_addr, trellis, out_addr, token_count, intermediate_dim, hidden_dim, locks, K);
            else
                moe_gemm_rows<t_bits, MOE_BIG_M, MOE_TILESIZE_N, cb, MOE_BIG_FS>
                (in_addr, trellis, out_addr, token_count, intermediate_dim, hidden_dim, locks, K);
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

        // Draw the next ticket and publish it to the group through the end-of-expert barrier, which also protects
        // the temp buffers for reuse. Grabbed tickets continue from num_groups since 0..num_groups-1 are implicit
        if (block_idx == 0 && threadIdx.x == 0)
            sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);
        group_barrier(group_idx, group_size, barrier_counters_sense);
        ticket = sched[2 + group_idx];
    }

    // Retire group; last group out resets the scheduler for the next launch. The acq_rel increment orders each
    // group's earlier ticket grabs before the last group's reset (plain atomics are relaxed, so without this a
    // straggler's in-flight grab could land after the reset and leak into the next launch)
    if (block_idx == 0 && threadIdx.x == 0)
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
