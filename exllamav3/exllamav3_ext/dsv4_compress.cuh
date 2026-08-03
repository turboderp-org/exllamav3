#pragma once

#include <ATen/Tensor.h>

class Graph;

// Fused DSv4 compressor step (cached path, bsz 1): windows compute + snapshot + ring store.
// See dsv4_compress.cu for layout and ordering invariants.

void dsv4_compress_gr
(
    const at::Tensor& kv_new,
    const at::Tensor& gate_new,
    at::Tensor& ring_kv,
    at::Tensor& ring_gate,
    c10::optional<at::Tensor>& ovl,
    const at::Tensor& ape,
    const at::Tensor& norm_w,
    float rms_norm_eps,
    const at::Tensor& inv_freq,
    at::Tensor& dest_a,
    c10::optional<at::Tensor>& dest_b,
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m,
    Graph* graph
);

void dsv4_compress
(
    const at::Tensor& kv_new,
    const at::Tensor& gate_new,
    at::Tensor ring_kv,
    at::Tensor ring_gate,
    c10::optional<at::Tensor> ovl,
    const at::Tensor& ape,
    const at::Tensor& norm_w,
    float rms_norm_eps,
    const at::Tensor& inv_freq,
    at::Tensor dest_a,
    c10::optional<at::Tensor> dest_b,
    int position,
    const c10::optional<at::Tensor>& position_tensor,
    int m
);

void dsv4_ring_append_gr
(
    const at::Tensor& kv,
    at::Tensor& ring,
    const at::Tensor& pos,
    const at::Tensor& ring_beg,
    Graph* graph
);

void dsv4_ring_append
(
    const at::Tensor& kv,
    at::Tensor ring,
    const at::Tensor& pos,
    const at::Tensor& ring_beg
);
