#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

void silu_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const float act_limit,
    Graph* graph
);

void silu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const float act_limit
);

void gelu_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const float act_limit,
    Graph* graph
);

void gelu_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const float act_limit
);

void relu2_mul_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const float act_limit,
    Graph* graph
);

void relu2_mul
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    const float act_limit
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

void mul_sigmoid_
(
    at::Tensor& x,
    const at::Tensor& y
);

void mul_sigmoid__gr
(
    at::Tensor& x,
    const at::Tensor& y,
    Graph* graph
);

void deinterleave_qg
(
    const at::Tensor& qg,
    at::Tensor& q,
    at::Tensor& g,
    int head_dim
);

void deinterleave_qg_gr
(
    const at::Tensor& qg,
    at::Tensor& q,
    at::Tensor& g,
    int head_dim,
    Graph* graph
);

void mul_sigmoid_broadcast_
(
    at::Tensor& x,
    const at::Tensor& y
);

void mul_sigmoid_broadcast__gr
(
    at::Tensor& x,
    const at::Tensor& y,
    Graph* graph
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
