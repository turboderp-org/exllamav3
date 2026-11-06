#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

void sparse_head_argmax
(
    at::Tensor x,
    at::Tensor w,
    const c10::optional<at::Tensor>& bias,
    int64_t n_active,
    at::Tensor cols,
    at::Tensor out_ids,
    at::Tensor out_conf,
    at::Tensor scratch,
    const c10::optional<at::Tensor>& emb_rows,
    const c10::optional<at::Tensor>& out_emb
);
