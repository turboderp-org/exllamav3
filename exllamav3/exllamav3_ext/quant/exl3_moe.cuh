#pragma once

#include <ATen/Tensor.h>
#include "../graph.cuh"

// Exported by the Python extension as an integer module attribute with the
// same name. Consumers must require an exact supported value instead of
// inferring compatibility from symbol presence or callable arity. Increment
// whenever the additive entry points' signatures, pointer-table layout,
// routing workspace semantics, residual encoding, or overflow behavior change
// incompatibly.
constexpr int EXL3_MOE_ADDITIVE_ABI_VERSION = 1;

int exl3_moe_max_concurrency(int device);

void exl3_moe
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,

    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,

    const int act_function,

    const int K_gate,
    const int K_up,
    const int K_down,

    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,

    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,

    const float act_limit,
    const int num_active
);

// Additive residual contract:
// - all tensor arguments are contiguous CUDA tensors on hidden_state.device();
//   fused routing workspaces must be disjoint (expert_count and expert_offsets
//   are checked explicitly);
// - residual trellises use the MCG codebook and reuse the corresponding base
//   projection's suh/svh vectors;
// - residual K metadata is int32, one value in 1..max_residual_bits (<= 8) per
//   stage, and must be validated before graph capture;
// - a zero residual scale denotes a sparse/missing projection and skips its
//   GEMM;
// - int64 pointer tables do not retain their pointees. Callers must keep every
//   base/residual trellis and base suh/svh allocation alive and unmoved on the
//   same device through asynchronous completion and the lifetime of any graph;
// - additive kernels tile oversized expert route spans internally, so
//   num_active counts all nonempty experts and is only a launch-size hint.
void exl3_moe_additive
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,
    const int act_function,
    const int K_gate,
    const int K_up,
    const int K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const at::Tensor& residual_gate_ptrs_trellis,
    const at::Tensor& residual_up_ptrs_trellis,
    const at::Tensor& residual_down_ptrs_trellis,
    const at::Tensor& residual_gate_scales,
    const at::Tensor& residual_up_scales,
    const at::Tensor& residual_down_scales,
    const at::Tensor& residual_gate_k,
    const at::Tensor& residual_up_k,
    const at::Tensor& residual_down_k,
    const int max_residual_bits,
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int num_active
);

void exl3_moe_additive_fused
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& topk_ids,
    const at::Tensor& topk_weights,
    const at::Tensor& expert_map,
    const at::Tensor& expert_count,
    const at::Tensor& expert_offsets,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,
    const int act_function,
    const int K_gate,
    const int K_up,
    const int K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const at::Tensor& residual_gate_ptrs_trellis,
    const at::Tensor& residual_up_ptrs_trellis,
    const at::Tensor& residual_down_ptrs_trellis,
    const at::Tensor& residual_gate_scales,
    const at::Tensor& residual_up_scales,
    const at::Tensor& residual_down_scales,
    const at::Tensor& residual_gate_k,
    const at::Tensor& residual_up_k,
    const at::Tensor& residual_down_k,
    const int max_residual_bits,
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int num_active
);
