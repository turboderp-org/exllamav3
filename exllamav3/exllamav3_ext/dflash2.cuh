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
