#pragma once

#include <ATen/Tensor.h>

class Graph;

void dsa_topk_gr
(
    const at::Tensor& scores,
    at::Tensor& indices,
    int k,
    Graph* graph,
    const c10::optional<at::Tensor>& t_ptr = {},
    int t_seq = 0
);

void dsa_topk
(
    const at::Tensor& scores,
    at::Tensor indices,
    int64_t k,
    const c10::optional<at::Tensor>& t_ptr,
    int64_t t_seq
);
