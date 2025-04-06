#pragma once

#include <ATen/Tensor.h>

void silu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);

void gelu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);