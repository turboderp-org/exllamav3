#pragma once

#include <ATen/Tensor.h>

class Graph;

void rms_norm
(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool span_heads,
    bool add_residual
);

void rms_norm_gr
(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    Graph* graph
);

void rms_norm_res_in
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    at::Tensor r,
    float epsilon,
    float constant_bias,
    float constant_scale
);

void gated_rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias
);

void gated_rms_norm_gr
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias,
    Graph* graph
);
