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
#if defined(USE_ROCM)
    // The RDNA inner (WMMA) and the pipelined mainloop the PIPE instances run (rocm/quant). With
    // MOE_TILESIZE_K = 32 this kernel is the RDNA inner's only live caller of its split-K path (every RDNA
    // GEMM shape has TILESIZE_K = 16), so a wrong fused-MoE output on RDNA points there first
    #include "../rocm/quant/exl3_gemm_inner_rdna.cuh"
    #include "../rocm/quant/exl3_moe_inner_rdna.cuh"
#else
    #include "exl3_gemm_inner.cuh"
#endif
#include "exl3_devctx.cuh"
#include "../ptx.cuh"

// M_TILE: rows per GEMM tile. 16 is the general instance; 32 and 64 (mul1 only) amortise the
// B dequant over more rows and are launched separately over the experts whose token count
// warrants them ([count_lo, count_hi], see exl3_moe). Separate instances rather than one
// kernel with a runtime tier switch: the tiers' register frames would otherwise share one
// 128-register budget and the 16-row path pays for tiles it never runs. On RDNA the inner implements
// the 16-row tile only; the wide tiles come from the pipelined mainloop below
template<int t_bits, int cb, int MT, int N_TILE, bool t_half>
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
    const int K,
    const int group,
    const int planar
)
{
    // Fragment pipeline depth: the 64-row tile keeps two B stages so its A fragments fit
    constexpr int FS = (MT >= 64) ? 2 : MOE_FRAG_STAGES;
#if defined(USE_ROCM)
    // The ROCm inner keeps native addressing (the stream path restores packed layouts there;
    // see the stream layout policy in moe_cpu_host.py)
    #define ARGS in_addr, trellis, out_addr, MIN(size_m, MT), size_k, size_n, locks, nullptr
#else
    #define ARGS in_addr, trellis, out_addr, MIN(size_m, MT), size_k, size_n, locks, nullptr, 0, group, planar
#endif
    #define SHAPE_ARGS MT, MOE_TILESIZE_K, N_TILE, MOE_SH_STAGES, FS
    // Compile-time rate: t_bits, or t_bits + 0.5 with t_half (mul1 codebook only). Otherwise the runtime K
    // arrives in half-bit units (2 * bits + half, see bits_k.cuh): even = integer rates, odd = the
    // half-integer rates 1.5 / 2.5 / 3.5 (mul1 codebook only, the host checks)
    if constexpr (t_bits)
        exl3_gemm_kernel_inner<t_bits, t_half, false, cb, SHAPE_ARGS, false>(ARGS);
    else switch(K)
    {
        case 2:  exl3_gemm_kernel_inner<1, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 4:  exl3_gemm_kernel_inner<2, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 6:  exl3_gemm_kernel_inner<3, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 8:  exl3_gemm_kernel_inner<4, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 10: exl3_gemm_kernel_inner<5, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 12: exl3_gemm_kernel_inner<6, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 14: exl3_gemm_kernel_inner<7, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 16: exl3_gemm_kernel_inner<8, false, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 3:  if constexpr (cb == 2) exl3_gemm_kernel_inner<1, true, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 5:  if constexpr (cb == 2) exl3_gemm_kernel_inner<2, true, false, cb, SHAPE_ARGS, false>(ARGS); break;
        case 7:  if constexpr (cb == 2) exl3_gemm_kernel_inner<3, true, false, cb, SHAPE_ARGS, false>(ARGS); break;
    };
    #undef ARGS
    #undef SHAPE_ARGS
}

#if defined(USE_ROCM)

// Pipelined mainloop (rocm/quant/exl3_moe_inner_rdna.cuh): one expert GEMM of size_m rows in row tiles of
// 64 / 48 / 32 / 16 (the smallest that covers the rest, capped at 64), sharing each dequantized B
// fragment across the row blocks. Mixed-K kernels (t_bits == 0) keep the 16-row tile only, so the K
// switch does not multiply into three row-tile copies. The mainloop takes half-integer rates as the
// pseudo width EXL3_HALF_BITS(K) (exl3_gemv_tiles_rdna.cuh); mixed half-integer rates never reach it
// (the host routes them to the non-pipelined kernel)
template<int t_bits, int cb, int N_TILE, bool t_half>
__device__ __forceinline__
void moe_gemm_rows_pipe
(
    const half* in_addr,
    const uint16_t* trellis,
    half* out_addr,
    int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const int K
)
{
    constexpr int PB = t_half ? EXL3_HALF_BITS(t_bits) : t_bits;
    #define PIPE_ARGS(MT) in_addr, trellis, out_addr, MIN(size_m, MT), size_k, size_n, locks
    while (size_m > 0)
    {
        int tm;
        if constexpr (t_bits)
        {
            if (size_m > 48)      { moe_pipe::moe_gemm_pipe<PB, cb, 4, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(64)); tm = 64; }
            else if (size_m > 32) { moe_pipe::moe_gemm_pipe<PB, cb, 3, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(48)); tm = 48; }
            else if (size_m > 16) { moe_pipe::moe_gemm_pipe<PB, cb, 2, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(32)); tm = 32; }
            else                  { moe_pipe::moe_gemm_pipe<PB, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); tm = 16; }
        }
        else
        {
            // Runtime K in half-bit units, integer rates only (see above)
            switch (K)
            {
                case 2:  moe_pipe::moe_gemm_pipe<1, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
                case 4:  moe_pipe::moe_gemm_pipe<2, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
                case 6:  moe_pipe::moe_gemm_pipe<3, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
                case 8:  moe_pipe::moe_gemm_pipe<4, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
                case 10: moe_pipe::moe_gemm_pipe<5, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
                case 12: moe_pipe::moe_gemm_pipe<6, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
                case 14: moe_pipe::moe_gemm_pipe<7, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
                case 16: moe_pipe::moe_gemm_pipe<8, cb, 1, MOE_TILESIZE_K, N_TILE>(PIPE_ARGS(16)); break;
            }
            tm = 16;
        }
        in_addr += tm * size_k;
        out_addr += tm * size_n;
        size_m -= tm;
    }
    #undef PIPE_ARGS
}

// The pipelined instances are built for EXL3_MOE_PIPE_WPE waves per SIMD, so two blocks fit one WGP; the
// host launches that many only after the runtime occupancy query confirms it (exl3_moe.cu)
#define EXL3_MOE_KERNEL_ATTR __attribute__((amdgpu_waves_per_eu(PIPE ? EXL3_MOE_PIPE_WPE : 1)))

#else
#define EXL3_MOE_KERNEL_ATTR
#endif

// PIPE (RDNA only): true = the g/u/d GEMMs run the pipelined mainloop (EXL3_ROCM_MOE_PIPE=1, default),
// false = the shared inner through moe_gemm_tile (the CUDA path, EXL3_ROCM_MOE_PIPE=0 on RDNA)
template<int t_bits, int MOE_TILESIZE_N, int cb, int M_TILE = MOE_TILESIZE_M, bool t_half = false, bool PIPE = false>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
EXL3_MOE_KERNEL_ATTR
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
#if !defined(USE_ROCM)
    static_assert(!PIPE, "the pipelined mainloop exists for RDNA only");
#endif
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
        // Skip if outside this launch's row-tile tier
        if (token_count < count_lo || token_count > count_hi) continue;

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

        // GEMM over the expert's rows in tiles of M_TILE. The wide instances finish an
        // expert's remainder with the largest smaller tile that covers it (64 -> 32 -> 16) so a
        // 96-row expert runs 64 + 32 instead of two 64-row tiles with half of one idle
        auto gemm = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K,
                        const int size_k, const int size_n, const int group = 0, const int planar = 0)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                int tm;
                if constexpr (M_TILE >= 64)
                {
                    if (size_m > 32)      { moe_gemm_tile<t_bits, cb, 64, MOE_TILESIZE_N, t_half>(in_addr, trellis, out_addr, size_m, size_k, size_n, locks, K, group, planar); tm = 64; }
                    else if (size_m > 16) { moe_gemm_tile<t_bits, cb, 32, MOE_TILESIZE_N, t_half>(in_addr, trellis, out_addr, size_m, size_k, size_n, locks, K, group, planar); tm = 32; }
                    else                  { moe_gemm_tile<t_bits, cb, 16, MOE_TILESIZE_N, t_half>(in_addr, trellis, out_addr, size_m, size_k, size_n, locks, K, group, planar); tm = 16; }
                }
                else if constexpr (M_TILE == 32)
                {
                    if (size_m > 16)      { moe_gemm_tile<t_bits, cb, 32, MOE_TILESIZE_N, t_half>(in_addr, trellis, out_addr, size_m, size_k, size_n, locks, K, group, planar); tm = 32; }
                    else                  { moe_gemm_tile<t_bits, cb, 16, MOE_TILESIZE_N, t_half>(in_addr, trellis, out_addr, size_m, size_k, size_n, locks, K, group, planar); tm = 16; }
                }
                else
                {
                    moe_gemm_tile<t_bits, cb, 16, MOE_TILESIZE_N, t_half>(in_addr, trellis, out_addr, size_m, size_k, size_n, locks, K, group, planar); tm = 16;
                }
                in_addr += tm * size_k;
                out_addr += tm * size_n;
                size_m -= tm;
            }
        };

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

#if defined(USE_ROCM)
        // g, u and d through ONE call site: with three, clang outlines the GEMM lambda (FLAT loads,
        // callee-saved spills), and the outlined runtime-K instance hangs on gfx11 for half-integer rates at
        // the N = 256 tile (every wave returns into the lambda's own epilogue: s[30:31] holds the epilogue
        // address when s_setpc_b64 runs). Same order and barriers as the CUDA sequence below
        #pragma nounroll
        for (int p = gated ? 0 : 1; p < 3; ++p)
        {
            if (p == 2)
            {
                group_barrier(group_idx, group_size, barrier_counters_sense);
                had_guad();
            }
            const half* in_addr     = p == 0 ? temp_state_g : (p == 1 ? temp_state_u : temp_intermediate_g);
            half* out_addr          = p == 0 ? temp_intermediate_g : (p == 1 ? temp_intermediate_u : temp_state_g);
            const uint16_t* trellis = p == 0 ? exp_gate_trellis : (p == 1 ? exp_up_trellis : exp_down_trellis);
            const int K             = p == 0 ? K_gate : (p == 1 ? K_up : K_down);
            const int size_k        = p == 2 ? intermediate_dim : hidden_dim;
            const int size_n        = p == 2 ? hidden_dim : intermediate_dim;
            if constexpr (PIPE)
                moe_gemm_rows_pipe<t_bits, cb, MOE_TILESIZE_N, t_half>(in_addr, trellis, out_addr, token_count, size_k, size_n, locks, K);
            else
                gemm(in_addr, out_addr, trellis, K, size_k, size_n);
        }
#else
        auto gemm_up = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K,
                           const int group, const int planar)
        {
            gemm(in_addr, out_addr, trellis, K, hidden_dim, intermediate_dim, group, planar);
        };
        auto gemm_down = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K,
                             const int group, const int planar)
        {
            gemm(in_addr, out_addr, trellis, K, intermediate_dim, hidden_dim, group, planar);
        };

        if (gated)
            gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate, group_g, planar_g);
        gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up, group_u, planar_u);
        group_barrier(group_idx, group_size, barrier_counters_sense);
        had_guad();
        gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down, group_d, planar_d);
#endif
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for d + scatter add
        auto had_d_out = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            const half* weights = weight_sorted + start;
            // Deterministic mode (output_scratch set): every fused assignment owns a compact
            // slot (fused_base[expert] + row within the expert) and the weighted output is
            // stored there; exl3_moe_gather then sums each token's top-k slots in a fixed
            // order. Otherwise the contributions are atomically added into the token row in
            // arrival order, which is not bit-reproducible run to run
            const int64_t slot_base = output_scratch ? fused_base[expert_idx] : 0;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int row = warp_idx / warps_per_token;
                int token_idx = top_x[row];
                half weight = weights[row];
                int token_off = warp_idx % warps_per_token;
                if (output_scratch)
                {
                    float* out_ptr = output_scratch + (slot_base + row) * hidden_dim + token_off * 128;
                    had_hf_r_128_d_inner<false>
                    (
                        temp_state_g + 128 * warp_idx,
                        out_ptr,
                        exp_down_svh + 128 * token_off,
                        0.088388347648f * __half2float(weight)
                    );
                }
                else
                {
                    float* out_ptr = output_state + token_idx * hidden_dim + token_off * 128;
                    had_hf_r_128_d_inner<true>
                    (
                        temp_state_g + 128 * warp_idx,
                        out_ptr,
                        exp_down_svh + 128 * token_off,
                        0.088388347648f * __half2float(weight)
                    );
                }
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
