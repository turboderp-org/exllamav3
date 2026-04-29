#pragma once

#include <ATen/Tensor.h>

void routing_ds3_nogroup
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor
);

void routing_ds3_nogroup_logits
(
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const bool use_topk
);

void routing_std
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale
);

void routing_std_logits
(
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const bool use_topk
);
