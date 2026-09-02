#pragma once

#include <ATen/Tensor.h>

class Graph;

void dsa_topk_gr
(
    const at::Tensor& scores,
    at::Tensor& indices,
    int k,
    Graph* graph,
    const c10::optional<at::Tensor>& t_ptr = {},
    int t_seq = 0
);

void dsa_topk
(
    const at::Tensor& scores,
    at::Tensor indices,
    int64_t k,
    const c10::optional<at::Tensor>& t_ptr,
    int64_t t_seq
);

void dsa_topk_tile
(
    const at::Tensor& scores,
    at::Tensor ws_idx,
    at::Tensor ws_scr,
    at::Tensor ws_cnt,
    int64_t slot,
    int64_t k,
    int64_t idx_offset
);
void dsa_topk_merge_tiles
(
    const at::Tensor& ws_idx,
    const at::Tensor& ws_scr,
    const at::Tensor& ws_cnt,
    at::Tensor out_idx,
    const c10::optional<at::Tensor>& out_scr,
    const c10::optional<at::Tensor>& out_cnt,
    int64_t k
);
void dsa_seq_state_gr
(
    const at::Tensor& cache_seqlens,
    at::Tensor& arr,
    int bsz,
    int q_len,
    Graph* graph
);
