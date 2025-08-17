#pragma once

#include <ATen/Tensor.h>

void routing_ds3_nogroup
(
    const at::Tensor& scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor
);

void routing_std
(
    const at::Tensor& scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights
);
