#pragma once

#include <ATen/Tensor.h>

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1
);

void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1,
    int64_t n_offset
);

void reconstruct_had_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    at::Tensor suh,
    at::Tensor svh,
    float K,
    bool mcg,
    bool mul1,
    int64_t n_offset
);

void reconstruct_had_batch
(
    at::Tensor unpacked,
    at::Tensor packed_ptrs,
    at::Tensor suh_ptrs,
    at::Tensor svh_ptrs,
    float K,
    bool mcg,
    bool mul1
);

void reconstruct_batch
(
    at::Tensor unpacked,
    at::Tensor packed_ptrs,
    float K,
    bool mcg,
    bool mul1
);
