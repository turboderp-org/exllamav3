#pragma once
#include <ATen/Tensor.h>

void dequant_tq3_weight(
    const at::Tensor& packed,
    const at::Tensor& scales,
    at::Tensor& output
);
