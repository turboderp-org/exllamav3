#pragma once

#include <ATen/Tensor.h>

void cache_rotate
(
    const at::Tensor& cache,
    const at::Tensor& order,
    const at::Tensor& temp
);