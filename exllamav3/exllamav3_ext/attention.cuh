#pragma once

#include <ATen/Tensor.h>

void bighead_attn_paged
(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    const at::Tensor& o,
    // const at::Tensor& workspace,
    int kv_chunk_size,
    bool causal,
    float sm_scale
);

void bighead_attn
(
    const at::Tensor& q,         // [bsz, q_len, n_q_heads, D]    fp16
    const at::Tensor& k,         // [bsz, kv_len, n_kv_heads, D]  fp16
    const at::Tensor& v,         // [bsz, kv_len, n_kv_heads, D]  fp16
    const at::Tensor& o,         // [bsz, q_len, n_q_heads, D]    fp16  (pre-allocated output)
    // const at::Tensor& workspace, // fp32, at least workspace_size() elements
    int  kv_chunk_size,
    bool causal,
    float sm_scale
);

size_t bighead_attn_workspace_size
(
    int bsz,
    int q_len,
    int n_q_heads,
    int max_kv_len,
    int kv_chunk_size,
    int dim
);