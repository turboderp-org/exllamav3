#pragma once

#include <ATen/Tensor.h>

bool sm120_tma_attn_supported(int device);

at::Tensor sm120_tma_attn_paged(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor k_cache,
    at::Tensor v_cache,
    at::Tensor block_table,
    at::Tensor cache_seqlens,
    bool causal,
    float sm_scale,
    float softcap,
    int split_k,
    int q_group_mode
);
