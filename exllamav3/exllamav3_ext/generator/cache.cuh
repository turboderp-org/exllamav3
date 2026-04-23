#pragma once

#include <ATen/Tensor.h>

void cache_rotate
(
    const at::Tensor& cache,
    const at::Tensor& order,
    const at::Tensor& temp
);

void paged_kv_cache_update
(
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& block_table,
    at::Tensor& cache_seqlens
);