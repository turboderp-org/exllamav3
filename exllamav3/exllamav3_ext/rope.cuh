#pragma once

#include <ATen/Tensor.h>

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
    float norm_constant_bias
);

int64_t gen_mrope_pos_ids
(
    at::Tensor mrope_pos_ids,
    at::Tensor ids,
    int merge_size,
    const std::vector<std::tuple<int64_t, int64_t>> &spans,
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> &grids
);