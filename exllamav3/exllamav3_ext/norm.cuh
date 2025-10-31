#pragma once

#include <ATen/Tensor.h>

void rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    bool span_heads
);

void gated_rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias
);
