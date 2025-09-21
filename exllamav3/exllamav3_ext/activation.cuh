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

void relu2_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);

void xielu
(
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& alpha_p,
    const at::Tensor& alpha_n
);

void add_sigmoid_gate
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);