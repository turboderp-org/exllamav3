#pragma once

#include <ATen/Tensor.h>

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);

void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1,
    int64_t n_offset
);

void reconstruct_fp8dg_nt
(
    at::Tensor q_nt,
    at::Tensor scales,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
);
