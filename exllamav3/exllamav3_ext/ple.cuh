#pragma once

#include <ATen/Tensor.h>

class Graph;

void ple_gate
(
    const at::Tensor& gate,
    const at::Tensor& value,
    at::Tensor& out,
    float gate_scale
);

void ple_gate_gr
(
    const at::Tensor& gate,
    const at::Tensor& value,
    at::Tensor& out,
    float gate_scale,
    Graph* graph
);

void ple_forward_streams
(
    const at::Tensor& streams,
    const at::Tensor& emb,
    const at::Tensor& key_w,
    const at::Tensor& value_w,
    const at::Tensor& norm_key_w,
    const at::Tensor& norm_query_w,
    const at::Tensor& norm_conv_w,
    const at::Tensor& conv_w,
    const c10::optional<at::Tensor>& conv_state,
    double rms_eps,
    double gate_scale,
    int64_t conv_dilation,
    at::Tensor delta,
    at::Tensor conv_stream
);
