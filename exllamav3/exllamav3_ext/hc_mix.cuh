#pragma once

#include <ATen/Tensor.h>

class Graph;

// Fused mHC HyperConnection mix / HyperHead collapse

int hc_mix_num_chunks(int R, int row_len);

void hc_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor collapsed
);

void hc_head
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    at::Tensor partials,
    at::Tensor collapsed
);

void hc_apply
(
    at::Tensor x,
    const at::Tensor& y,
    const at::Tensor& post,
    const c10::optional<at::Tensor>& comb
);

void gr_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& upt,
    const at::Tensor& w,
    double rms_eps,
    at::Tensor dots,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);
