#pragma once

#include <ATen/Tensor.h>

class Graph;

void dsa_topk_gr
(
    const at::Tensor& scores,
    at::Tensor& indices,
    int k,
    Graph* graph
);

void dsa_topk
(
    const at::Tensor& scores,
    at::Tensor indices,
    int64_t k
);
