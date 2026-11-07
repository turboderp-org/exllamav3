#pragma once

#include <ATen/Tensor.h>

// DSA prefill attention, gathered mode, gfx11 WMMA (ROCm only; see dsa_prefill_rdna.cu)
void dsa_prefill_rdna
(
    const at::Tensor& q,
    const at::Tensor& ring,
    const at::Tensor& kv_chunk,
    const at::Tensor& pool_c,
    const at::Tensor& pool_r,
    const at::Tensor& block_table,
    const at::Tensor& indices,
    const at::Tensor& sinks,
    const at::Tensor& derot,
    at::Tensor& out,
    int64_t k_len,
    int64_t win_len,
    int64_t q_pos0,
    int64_t win_floor,
    int64_t ring_beg,
    int64_t page_size,
    double scale
);
