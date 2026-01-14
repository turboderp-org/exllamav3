#pragma once

#include <ATen/Tensor.h>

void adaptivep_gumbel_noise_f32
(
    const at::Tensor& probs_in,
    at::Tensor& logits,
    uint32_t random,
    float adapted_target,
    float inv_width,
    float peak_logit_value,
    float sharpness
);
