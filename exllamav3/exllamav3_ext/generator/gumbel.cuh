#pragma once

#include <ATen/Tensor.h>

void gumbel_noise_f16
(
    const at::Tensor& logits_in,
    at::Tensor& logits,
    uint32_t random
);

void gumbel_noise_f32
(
    const at::Tensor& logits_in,
    at::Tensor& logits,
    uint32_t random
);

void gumbel_noise_log
(
    const at::Tensor& probs,
    at::Tensor& logits,
    uint32_t random
);