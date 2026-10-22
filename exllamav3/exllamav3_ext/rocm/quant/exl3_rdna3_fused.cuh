#pragma once

#include <ATen/Tensor.h>

class Graph;

// C = (silu(g) * u) @ W, or with gate_mode C = (g * sigmoid(u)) @ W (attention output gate), with the
// activation in the RDNA3 matmul's input transform
bool exl3_gemm_silu_gr
(
    const at::Tensor& g,
    const at::Tensor& u,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& svh,
    bool mcg,
    bool mul1,
    Graph* graph,
    bool gate_mode = false
);

// Gated DeltaNet output projection: C = gated_rms_norm(x, w, g) @ W
bool exl3_gemm_gnorm_gr
(
    const at::Tensor& x,
    const at::Tensor& g,
    const at::Tensor& w,
    float eps,
    float bias,
    bool sigmoid,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& svh,
    bool mcg,
    bool mul1,
    Graph* graph
);
