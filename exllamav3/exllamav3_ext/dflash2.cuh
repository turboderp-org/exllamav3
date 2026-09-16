#pragma once

#include <ATen/Tensor.h>

void dflash2_dynconv
(
    const at::Tensor& x,
    const at::Tensor& dyn,
    const at::Tensor& base,
    at::Tensor& out,
    int64_t group_size,
    bool accumulate
);

void dflash2_selector_walk
(
    const at::Tensor& unary,
    const at::Tensor& cands,
    const at::Tensor& gate,
    const at::Tensor& pred_cb,
    const at::Tensor& succ_cb,
    const at::Tensor& anchor,
    at::Tensor& out,
    const c10::optional<at::Tensor>& conf
);

void dflash2_topk
(
    const at::Tensor& logits,
    int64_t vocab,
    double scale,
    double softcap,
    at::Tensor& values,
    at::Tensor& indices
);
