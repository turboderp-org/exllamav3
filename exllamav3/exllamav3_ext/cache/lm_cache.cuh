#pragma once

#include <ATen/Tensor.h>

void quant_lm_cache_cont
(
    const at::Tensor& in,
    const at::Tensor& out,
    const at::Tensor& out_scales
);

void dequant_lm_cache_cont
(
    const at::Tensor& in,
    const at::Tensor& in_scales,
    const at::Tensor& out
);

void quant_lm_cache_paged
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

void dequant_lm_cache_paged
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

// Sub-block scale variants (sub_size=8, 4 scales per 32-element block)

void quant_lm_cache_cont_sub
(
    const at::Tensor& in,
    const at::Tensor& out,
    const at::Tensor& out_scales
);

void dequant_lm_cache_cont_sub
(
    const at::Tensor& in,
    const at::Tensor& in_scales,
    const at::Tensor& out
);

void quant_lm_cache_paged_sub
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

void dequant_lm_cache_paged_sub
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

// Asymmetric K (scale + zero-point, uniform grid) / symmetric Lloyd-Max V variants

void quant_lm_cache_cont_asym
(
    const at::Tensor& in,
    const at::Tensor& out,
    const at::Tensor& out_scales,
    const at::Tensor& out_zeros
);

void dequant_lm_cache_cont_asym
(
    const at::Tensor& in,
    const at::Tensor& in_scales,
    const at::Tensor& in_zeros,
    const at::Tensor& out
);

void quant_lm_cache_paged_asym
(
    const at::Tensor& k_in,
    const at::Tensor& k_out,
    const at::Tensor& k_out_scales,
    const at::Tensor& k_out_zeros,
    const at::Tensor& v_in,
    const at::Tensor& v_out,
    const at::Tensor& v_out_scales,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
);

void dequant_lm_cache_paged_asym
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_in_zeros,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size
);
