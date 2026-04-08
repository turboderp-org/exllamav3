#pragma once

#include <ATen/Tensor.h>

void quant_cache_cont
(
    const at::Tensor& in,
    const at::Tensor& out,
    const at::Tensor& out_scales
);

void dequant_cache_cont
(
    const at::Tensor& in,
    const at::Tensor& in_scales,
    const at::Tensor& out
);

void quant_cache_paged
(
    const at::Tensor& k_in,
    const at::Tensor& k_out,
    const at::Tensor& k_out_scales,
    const at::Tensor& v_in,
    const at::Tensor& v_out,
    const at::Tensor& v_out_scales,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
);

void quant_cache_paged_delta
(
    const at::Tensor& k_in,
    const at::Tensor& k_out,
    const at::Tensor& k_out_scales,
    const at::Tensor& v_in,
    const at::Tensor& v_out,
    const at::Tensor& v_out_scales,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
);

void dequant_cache_paged
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size
);

void dequant_cache_paged_gather
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
);

void dequant_cache_paged_gather_heads
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
);

void dequant_cache_paged_gather_delta
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_delta,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_delta,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len,
    int delta_len
);

void dequant_cache_paged_gather_delta_heads
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_delta,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_delta,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len,
    int delta_len
);

void dequant_cache_paged_select_delta_heads
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_delta,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_delta,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    const at::Tensor& selected_positions,
    const at::Tensor& selected_counts,
    int page_size,
    int seq_len,
    int delta_len
);
