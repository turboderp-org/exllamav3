#pragma once

#include <ATen/Tensor.h>

// Flash-attention forward on sm_75 HMMA for head_dim 256 (prefill): q [Tq, Hq, 256], k/v [Tkv, Hkv, 256]
// (16-byte aligned row/head strides, contiguous last dim), o [Tq, Hq, 256] contiguous, fp16. causal is
// bottom-right aligned (query i sees keys j <= i + Tkv - Tq); GQA by Hq / Hkv
void fa75_fwd
(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    double scale,
    bool causal
);
