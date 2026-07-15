#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

#define MOE_ACT_SILU 0
#define MOE_ACT_GELU 1

#define MOE_SMS_PER_EXPERT 8
#define MOE_MAX_SMS_PER_EXPERT 32
#define MOE_TILESIZE_K 32
#define MOE_TILESIZE_M 32
#define MOE_SH_STAGES 3
#define MOE_FRAG_STAGES 3

// A1 prefill-only re-tile. Keep the established M32/K32 MMA geometry and
// eight-CTA K partition, but deepen the async-copy pipeline for the separately
// instantiated route-block kernel. The kernel has 1,024 B of static shared
// storage, so a 100,352 B dynamic opt-in reaches the 101,376 B SM120 ceiling
// without exceeding it. Launches still allocate only the exact four-stage
// working set; registers keep the cooperative pool at one CTA/SM.
#define MOE_A1_SH_STAGES 4
#define MOE_A1_SMEM_MAX_BYTES 100352

// Variant switches. Each requested build is selected by these two values.
#define MOE_SMEMFIX 0

#ifndef EXL3_GEMM_BASE_THREADS
#define EXL3_GEMM_BASE_THREADS 256
#endif

#ifndef EXL3_SMEM_MAX_BYTES
#define EXL3_SMEM_MAX_BYTES (90 * 1024)
#endif

#ifndef SMEM_MAX
#define SMEM_MAX EXL3_SMEM_MAX_BYTES
#endif

// Exact extern-shared allocation used by one fused MoE GEMM launch. This
// mirrors the byte budget in exl3_gemm_inner.cuh for shmem_out_had == false.
inline constexpr int exl3_moe_smem_bytes
(
    const int bits,
    const int tilesize_n,
    const int sh_stages = MOE_SH_STAGES
)
{
    constexpr int tileblocks_k = MOE_TILESIZE_K / 16;
    constexpr int warps = EXL3_GEMM_BASE_THREADS / 32;
    const int tileblocks_n = tilesize_n / 16;
    const int frags_n_per_warp = 2 * tileblocks_n / warps;
    const int sh_a_stage_size = MOE_TILESIZE_M * MOE_TILESIZE_K;
    const int sh_b_stage_size = tileblocks_k * tileblocks_n * 256 / 16 * bits;
    const int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS * frags_n_per_warp;
    return sh_stages * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size;
}

#define EXL3_MOE_KERNEL_ARGS                    \
    const half* __restrict__ hidden_state,      \
    half* __restrict__ temp_state_g,            \
    half* __restrict__ temp_state_u,            \
    half* __restrict__ temp_intermediate_g,     \
    half* __restrict__ temp_intermediate_u,     \
    float* __restrict__ output_state,           \
                                                \
    const uint16_t** __restrict__ gate_trellis, \
    const half** __restrict__ gate_suh,         \
    const half** __restrict__ gate_svh,         \
    const uint16_t** __restrict__ up_trellis,   \
    const half** __restrict__ up_suh,           \
    const half** __restrict__ up_svh,           \
    const uint16_t** __restrict__ down_trellis, \
    const half** __restrict__ down_suh,         \
    const half** __restrict__ down_svh,         \
                                                \
    const int64_t* __restrict__ expert_count,   \
    const int64_t* __restrict__ token_sorted,   \
    const half* __restrict__ weight_sorted,     \
                                                \
    const int hidden_dim,                       \
    const int intermediate_dim,                 \
    const int num_experts,                      \
    const int num_experts_per_tok,              \
    const int max_tokens_per_expert,            \
    const int concurrency,                      \
    const float act_limit,                      \
    const int act_function,                     \
    const int K_gate,                           \
    const int K_up,                             \
    const int K_down,                           \
    const int dq_threshold,                     \
    const int use_ticket_scheduler,             \
                                                \
    int* __restrict__ locks
