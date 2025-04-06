#pragma once

#include <ATen/Tensor.h>

void rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    float epsilon,
    float constant_bias
);
