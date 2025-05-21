#pragma once

#include <ATen/Tensor.h>

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    uint32_t mcg_mult,
    uint32_t mul1_mult
);
