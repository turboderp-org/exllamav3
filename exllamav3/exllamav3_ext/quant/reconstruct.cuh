#pragma once

#include <ATen/Tensor.h>

// group/planar (default 0/0 = native layout): read a CPU-packed trellis directly, tile group
// 0/2/8 plus the planar dword order; see cpu/moe_mul1.h and quant/reconstruct.cu.

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1,
    int64_t group = 0,
    int64_t planar = 0
);

void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    float K,
    bool mcg,
    bool mul1,
    int64_t n_offset,
    int64_t group = 0,
    int64_t planar = 0
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
    int64_t n_offset,
    int64_t group = 0,
    int64_t planar = 0
);

void reconstruct_had_batch
(
    at::Tensor unpacked,
    at::Tensor packed_ptrs,
    at::Tensor suh_ptrs,
    at::Tensor svh_ptrs,
    float K,
    bool mcg,
    bool mul1,
    int64_t group = 0,
    int64_t planar = 0
);

void reconstruct_batch
(
    at::Tensor unpacked,
    at::Tensor packed_ptrs,
    float K,
    bool mcg,
    bool mul1,
    int64_t group = 0,
    int64_t planar = 0
);
