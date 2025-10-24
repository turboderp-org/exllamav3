#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

void silu_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
);

void silu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);

void gelu_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
);

void gelu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);

void relu2_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
);

void relu2_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);

void xielu_gr
(
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& alpha_p,
    const at::Tensor& alpha_n,
    Graph* graph
);

void xielu
(
    const at::Tensor& x,
    at::Tensor& y,
    const at::Tensor& alpha_p,
    const at::Tensor& alpha_n
);

void add_sigmoid_gate_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
);

void add_sigmoid_gate
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);

void add_sigmoid_gate_proj_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const at::Tensor& w,
    Graph* graph
);

void add_sigmoid_gate_proj
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const at::Tensor& w
);