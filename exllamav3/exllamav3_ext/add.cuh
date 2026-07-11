#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

void add_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
);

void add
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
);

void copy2d_gr
(
    const at::Tensor& src,
    at::Tensor& dst,
    Graph* graph
);

void moe_bias_add_gr
(
    at::Tensor& interm,
    const at::Tensor& bias_ptrs,
    const at::Tensor& sel,
    Graph* graph
);

void moe_bias_add_weighted_gr
(
    at::Tensor& out,
    const at::Tensor& bias_ptrs,
    const at::Tensor& sel,
    const at::Tensor& weights,
    Graph* graph
);