#pragma once

#include <ATen/Tensor.h>
#include "../graph.cuh"

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

void exl3_moe_fused
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
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int dq_threshold
);

// Prefill-only A1 entrypoint. It shares route packing and numerics with
// exl3_moe_fused, then selects the separately instantiated M32 ticket kernel.
void exl3_moe_fused_retile
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
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int dq_threshold
);
