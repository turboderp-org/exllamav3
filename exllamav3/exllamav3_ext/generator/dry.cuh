#pragma once

#include <ATen/Tensor.h>
#include <optional>

void dry_penalty
(
    const at::Tensor& in_logits,
    const at::Tensor& out_logits,
    const at::Tensor& past_ids,
    const std::optional<at::Tensor>& breakers,
    const at::Tensor& workspace,
    const at::Tensor& counters,
    float multiplier,
    float base,
    int allowed_length,
    int range,
    int max_exponent,
    int match_cap
);
