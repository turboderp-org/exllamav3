#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <memory>
#include "../graph.cuh"
#include "../triton_kernel.h"
#include "linear.h"

// Query-only decode attention over another model's K/V cache (drafters that share the target's keys and
// values, see modules/shared_kv_attn.py): q_proj -> q_norm/RoPE -> paged attention over the source layer's
// pages -> o_proj as one native call, captured as one CUDA graph per (bsz, q_len) slot after a warmup run,
// or issued stage by stage where graphs are unavailable. No K/V is projected or appended: the source cache
// holds every key the step attends to, and the split kernel is compiled with no appended tokens.

struct BC_SharedKVAttention
{
    static constexpr int MAX_BSZ = 8;
    static constexpr int MAX_QLEN = 16;

    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int hidden_size;
    int hidden_size_padded;
    int page_size;
    std::shared_ptr<BC_LinearEXL3> q_proj;
    std::shared_ptr<BC_LinearEXL3> o_proj;
    c10::optional<at::Tensor> q_norm;
    float norm_eps;
    float norm_constant_bias;
    at::Tensor inv_freq;
    int rope_style;
    float attn_factor;
    float l4_scaling_beta;
    int l4_scaling_original;
    int rotate_dims;
    at::Tensor cache_k;          // source layer's pages (pages, page_size, kv_heads, head_dim), fp16
    at::Tensor cache_v;
    at::Tensor xh;               // Hadamard staging for the projections
    at::Tensor h32;              // dummy for the kernel's unused quantized-cache argument

    struct Slot
    {
        bool configured = false;
        int runs = 0;
        int block_n = 0;
        int splits_cap = 0;
        int programs = 0;
        at::Tensor q, o, partial_o, partial_ml, xp, yp;
        at::Tensor q2, q4, o2;
        std::shared_ptr<TritonKernel> k_split;
        std::shared_ptr<TritonKernel> k_combine;
        std::unique_ptr<Graph> graph;
    };
    std::vector<Slot> slots;

    BC_SharedKVAttention
    (
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int hidden_size,
        int hidden_size_padded,
        int page_size,
        std::shared_ptr<BC_LinearEXL3> q_proj,
        std::shared_ptr<BC_LinearEXL3> o_proj,
        c10::optional<at::Tensor> q_norm,
        float norm_eps,
        float norm_constant_bias,
        at::Tensor inv_freq,
        int rope_style,
        float attn_factor,
        float l4_scaling_beta,
        int l4_scaling_original,
        int rotate_dims,
        at::Tensor cache_k,
        at::Tensor cache_v,
        at::Tensor xh,
        at::Tensor h32
    );

    Slot& slot(int bsz, int q_len) { return slots[(bsz - 1) * MAX_QLEN + (q_len - 1)]; }
    bool needs_configure(int bsz, int q_len);

    void configure_slot
    (
        int bsz,
        int q_len,
        at::Tensor q,
        at::Tensor o,
        at::Tensor partial_o,
        at::Tensor partial_ml,
        std::shared_ptr<TritonKernel> k_split,
        std::shared_ptr<TritonKernel> k_combine,
        int block_n,
        int splits_cap,
        int programs,
        c10::optional<at::Tensor> xp,
        c10::optional<at::Tensor> yp
    );

    void run_gr
    (
        int bsz,
        int q_len,
        Slot& s,
        const at::Tensor& x,
        at::Tensor& y,
        const at::Tensor& cache_seqlens,
        const at::Tensor& block_table,
        int64_t position,
        const c10::optional<at::Tensor>& positions,
        const c10::optional<at::Tensor>& position_ids,
        Graph* graph
    );

    void run
    (
        int bsz,
        int q_len,
        const at::Tensor& x,
        at::Tensor& y,
        const at::Tensor& cache_seqlens,
        const at::Tensor& block_table,
        int64_t position,
        const c10::optional<at::Tensor>& positions,
        const c10::optional<at::Tensor>& position_ids
    );
};
