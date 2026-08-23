#pragma once

#include <ATen/Tensor.h>

void routing_ds3_nogroup
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn
);

void routing_ds3_nogroup_logits
(
    at::Tensor scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const float scaling_factor,
    const bool use_topk,
    const int act_fn
);

void moe_split_map
(
    at::Tensor sel,
    const at::Tensor& map,
    at::Tensor hist,
    at::Tensor sel_cpu,
    const int64_t first_cpu_slot
);

void moe_split_issue
(
    at::Tensor sel,
    const c10::optional<at::Tensor>& map,
    const c10::optional<at::Tensor>& hist,
    const at::Tensor& y,
    const at::Tensor& w,
    int64_t h_sel_ptr,
    int64_t h_x_ptr,
    int64_t h_w_ptr,
    at::Tensor dev_count,
    const int64_t slot_idx,
    const int64_t hi,
    const int64_t first_cpu
);

void moe_split_collect_add
(
    at::Tensor final_out,
    int64_t h_out_ptr,
    const at::Tensor& dev_count,
    const int64_t slot_idx,
    const int64_t ho
);

void routing_sel_norm
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    const at::Tensor& selected,
    at::Tensor weights,
    const float scaling_factor,
    const c10::optional<at::Tensor>& gate_t,
    const int act_fn
);

void routing_std
(
    const at::Tensor& hidden,
    const at::Tensor& gate,
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const c10::optional<at::Tensor>& gate_t,
    const c10::optional<at::Tensor>& bias
);

void routing_std_logits
(
    at::Tensor scores,
    at::Tensor topk_indices,
    at::Tensor topk_weights,
    const c10::optional<at::Tensor>& per_expert_scale,
    const bool use_topk
);
