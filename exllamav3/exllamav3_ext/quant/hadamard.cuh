#pragma once

#include <ATen/Tensor.h>

void had_r_128
(
    const at::Tensor& input,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& pre_flip,
    const c10::optional<at::Tensor>& post_flip,
    float scale
);
