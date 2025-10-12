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