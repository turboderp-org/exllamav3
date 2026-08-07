#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

#define ROPESTYLE_NONE 0
#define ROPESTYLE_GPTJ 1
#define ROPESTYLE_NEOX 2
#define ROPESTYLE_NANOCHAT 3

void l4_scale_q_gr
(
    at::Tensor& q,
    int bsz,
    int seq_len,
    int row_width,
    int row_stride,
    uint32_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    float llama_4_scaling_beta,
    int llama_4_scaling_original,
    int position_ids_stride,
    Graph* graph
);

void rope_gr
(
    const at::Tensor& q,
    at::Tensor& out_q,
    const c10::optional<at::Tensor>& k,
    c10::optional<at::Tensor>& out_k,
    const at::Tensor& inv_freq,
    uint32_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    int rope_mode,
    float attn_factor,
    const c10::optional<at::Tensor>& q_norm,
    const c10::optional<at::Tensor>& k_norm,
    float norm_eps,
    float norm_constant_bias,
    float llama_4_scaling_beta,
    int llama_4_scaling_original,
    bool post_rope_norm,
    int rotate_dims,
    int rotate_offset,
    Graph* graph
);

void rope
(
    const at::Tensor& q,
    at::Tensor& out_q,
    const c10::optional<at::Tensor>& k,
    c10::optional<at::Tensor>& out_k,
    const at::Tensor& inv_freq,
    uint32_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    int rope_mode,
    float attn_factor,
    const c10::optional<at::Tensor>& q_norm,
    const c10::optional<at::Tensor>& k_norm,
    float norm_eps,
    float norm_constant_bias,
    float llama_4_scaling_beta,
    int llama_4_scaling_original,
    bool post_rope_norm,
    int rotate_dims,
    int rotate_offset
);

int64_t gen_mrope_pos_ids
(
    at::Tensor mrope_pos_ids,
    at::Tensor ids,
    int merge_size,
    const std::vector<std::tuple<int64_t, int64_t>> &spans,
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> &grids
);