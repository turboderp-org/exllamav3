#pragma once

#include <ATen/Tensor.h>

void histogram
(
    at::Tensor& input,
    at::Tensor& output,
    float min_value,
    float max_value,
    bool exclusive
);