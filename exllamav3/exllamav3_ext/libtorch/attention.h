#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <memory>
#include "../graph.cuh"
#include "../triton_kernel.h"
#include "linear.h"

// Graph-captured decode attention block, following the BC_GatedDeltaNetSplit pattern: run() is
// the bound entry point controlling the mode (first run eager, second run captured, later runs
// patch-and-replay), run_gr() is the workload — q/k/v projections, fused q/k norm + RoPE, cache
// append and the flash-decoding attention kernels (AOT-compiled Triton cubins launched through
// TritonKernel), then o_proj — all recorded as one graph. Intermediates live in static buffers
// allocated python-side through g_tensor_cache (shared between layers on the same device); the
// only per-call updates are the pointers patched into the graph: input x, output y,
// cache_seqlens, block_table and RoPE positions.
//
// One lazily configured slot per (bsz, q_len, regime) with bsz <= 4, q_len <= 16. The attention
// kernels bake the split configuration per slot; the block-table width and split length are
// runtime kernel arguments frozen at capture, so when the generator's block table grows the slot
// is reconfigured (recapture, no recompile unless the split count changes). Instances are built
// per cache layer, since the cache tensors are baked into the captured graphs.
//
// QSA (Qwen3.8-Flash-Next): set_qsa() arms the sparse-indexer stages, following the DSA-on-MLA
// pattern (mla_attention.h). Armed instances project + append raw indexer keys and rebuild the
// touched pooled block keys every step in BOTH regimes, so the planes stay complete while dense
// slots serve; regime-1 slots (context past the sparse threshold, bsz == 1, q_len == 1)
// additionally norm + rope the indexer queries, score the pooled plane (fewq with uniform head
// weights), select top-k blocks (capture-safe radix top-k), expand them to token indices plus
// the query's tail block, and run the gathered GQA attention instead of the dense flash-decoding
// kernels. Causality lives in the selection; the index list is a slot static, so sparse replay
// patches nothing beyond what the dense path patches plus the scoring bounds.

struct BC_Attention
{
    static constexpr int MAX_BSZ = 8;
    static constexpr int MAX_QLEN = 16;

    // Model config. hidden_size_padded > hidden_size when the model dim is not a multiple of
    // the EXL3 tile alignment (gpt-oss): the projections' K/N pad to 128 and the graph stages
    // the input through a zero-padded static and trims the o_proj output back down
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int hidden_size;
    int hidden_size_padded;
    int page_size;

    // Learned attention sinks (one logit per q head, fp32), passed to the combine kernel
    // (compiled with HAS_SINKS) as a static pointer
    c10::optional<at::Tensor> sinks;

    // Projections. K/V run as one fused mgemm when the pointer tables are given (and
    // bsz * q_len is small enough), otherwise as separate GEMMs
    std::shared_ptr<BC_LinearEXL3> q_proj;
    std::shared_ptr<BC_LinearEXL3> k_proj;
    std::shared_ptr<BC_LinearEXL3> v_proj;
    c10::optional<at::Tensor> kv_ptrs_trellis;
    c10::optional<at::Tensor> kv_ptrs_suh;
    c10::optional<at::Tensor> kv_ptrs_svh;
    int kv_K;
    bool kv_mcg;
    bool kv_mul1;
    std::shared_ptr<BC_LinearEXL3> o_proj;

    // V shares the K projection output (copied to a static before norm + RoPE touch K)
    bool use_k_as_v;

    // Output gate: 0 = none, 1 = headwise (g_proj emits one gate per head, broadcast across
    // head_dim; always an unquantized fp16 projection, so it runs as a captured cublas gemm over
    // the statically staged input, like the fp16 full gate), 2 = full (o *= act(g), g from its
    // own projection or the fused q+g mgemm), 3 = interleaved (q_proj emits q/g interleaved per
    // head). gate_softplus selects softplus instead of sigmoid for the headwise gate (Laguna)
    int gate_mode;
    bool gate_softplus;
    std::shared_ptr<BC_LinearEXL3> g_proj;
    c10::optional<at::Tensor> g_weight;   // unquantized full gate: fp16 (hidden, qh*hd)
    c10::optional<at::Tensor> qg_ptrs_trellis;
    c10::optional<at::Tensor> qg_ptrs_suh;
    c10::optional<at::Tensor> qg_ptrs_svh;
    int qg_K;
    bool qg_mcg;
    bool qg_mul1;

    // Head norms, fused into the RoPE kernel
    c10::optional<at::Tensor> q_norm;
    c10::optional<at::Tensor> k_norm;
    float norm_eps;
    float norm_constant_bias;

    // Optional V norm (RMSNorm, possibly unweighted), applied to the raw projection output
    bool v_norm;
    c10::optional<at::Tensor> v_norm_w;
    float v_norm_eps;
    float v_norm_constant_bias;
    float v_norm_constant_scale;

    // RoPE; nullopt for NoPE models (rope stage skipped entirely; head norms, which share the
    // rope kernel, must be absent in that case)
    c10::optional<at::Tensor> inv_freq;
    int rope_style;
    float attn_factor;
    float l4_scaling_beta;
    int l4_scaling_original;
    int rotate_dims;

    // Cache tensors: fp16 (pages, page_size, kvh, hd) or packed int32 + scales
    bool quant_cache;
    at::Tensor cache_k;
    at::Tensor cache_v;
    c10::optional<at::Tensor> cache_k_scales;
    c10::optional<at::Tensor> cache_v_scales;

    // Shared statics (python tensor cache): hadamard scratch for the EXL3 GEMMs, sized
    // (2, MAX_R, max(hidden_size, num_q_heads * head_dim)), and the H32 rotation matrix for
    // quantized-cache kernels (any small tensor when unused)
    at::Tensor xh;
    at::Tensor h32;

    // QSA indexer (set_qsa): fused q/raw-key projection, per-head norm weights, the paged raw
    // and pooled key planes (flat views of the CacheLayer_qsa side planes)
    bool qsa = false;
    std::shared_ptr<BC_LinearEXL3> qsa_qk_proj;
    at::Tensor qsa_q_norm_w;
    at::Tensor qsa_k_norm_w;
    float qsa_norm_eps = 0.0f;
    int qsa_n_heads = 0;
    int qsa_head_dim = 0;
    int qsa_topk = 0;            // blocks selected per query (token_budget / compress_ratio)
    int qsa_cr = 0;              // compress ratio P (tokens per block)
    at::Tensor qsa_raw_plane;    // flat (pages * page_size, D_i) fp16
    at::Tensor qsa_pool_plane;   // flat (pages * page_size / P, D_i) fp16

    struct Slot
    {
        bool configured = false;
        int runs = 0;
        int block_n = 0;
        int splits_cap = 0;   // grid height of the split kernel; the live split count is a
                              // runtime argument patched per call, extra splits idle
        int programs = 0;
        dim3 upd_grid;

        // Static intermediates (python tensor cache) and precomputed views. gate_a/gate_b by
        // gate mode: full = qg (2, R, qh*hd) with q aliasing qg[0]; interleaved = qg_i
        // (R, 2*qh*hd) staging + g (R, qh*hd). xp: input staging, present when the hidden dim
        // is padded or the gate projection is fp16 (the cublas node needs a static input).
        // yp: padded o_proj output, only when hidden_size_padded > hidden_size
        at::Tensor q, kv, o, partial_o, partial_ml, gate_a, gate_b, xp, yp;
        at::Tensor q2, q4, k4, v4, o2, o4, qg2, g2;

        std::shared_ptr<TritonKernel> k_split;
        std::shared_ptr<TritonKernel> k_combine;   // null when num_splits == 1
        std::shared_ptr<TritonKernel> k_update;    // null when quant_cache

        // QSA statics/kernels (qsa == true). The projection/stage/append/pool pieces exist in
        // both regimes; scoring, selection and the gathered attention only in regime-1 slots
        at::Tensor qsa_qk;       // (R, (H_i + 1) * D_i) fused projection output
        at::Tensor qsa_q;        // (R, H_i, D_i) normed queries
        at::Tensor qsa_q4;       // rope view (bsz, q_len, H_i, rotate_dims)
        at::Tensor qsa_kraw;     // (R, D_i) raw keys
        at::Tensor qsa_wts;      // (R, H_i) fp16, filled with 1.0 (uniform head weights)
        at::Tensor qsa_scores;   // (R, S_max) fp16, -inf filled once
        at::Tensor qsa_pool_idx; // (R, KP_pool) i32 selected block ids
        at::Tensor qsa_indices;  // (R, K_pad) i32 expanded token indices
        std::shared_ptr<TritonKernel> k_qsa_stage, k_qsa_raw_append, k_qsa_pool_update,
            k_qsa_fewq, k_qsa_expand, k_qsa_split, k_qsa_combine;
        int qsa_fewq_gy = 0;
        int qsa_splits = 0;
        int qsa_split_len = 0;
        int qsa_programs = 0;

        std::unique_ptr<Graph> graph;
    };
    std::vector<Slot> slots;

    BC_Attention
    (
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int hidden_size,
        int hidden_size_padded,
        int page_size,
        std::shared_ptr<BC_LinearEXL3> q_proj,
        std::shared_ptr<BC_LinearEXL3> k_proj,
        std::shared_ptr<BC_LinearEXL3> v_proj,
        c10::optional<at::Tensor> kv_ptrs_trellis,
        c10::optional<at::Tensor> kv_ptrs_suh,
        c10::optional<at::Tensor> kv_ptrs_svh,
        int kv_K,
        bool kv_mcg,
        bool kv_mul1,
        std::shared_ptr<BC_LinearEXL3> o_proj,
        bool use_k_as_v,
        int gate_mode,
        bool gate_softplus,
        std::shared_ptr<BC_LinearEXL3> g_proj,
        c10::optional<at::Tensor> g_weight,
        c10::optional<at::Tensor> qg_ptrs_trellis,
        c10::optional<at::Tensor> qg_ptrs_suh,
        c10::optional<at::Tensor> qg_ptrs_svh,
        int qg_K,
        bool qg_mcg,
        bool qg_mul1,
        c10::optional<at::Tensor> q_norm,
        c10::optional<at::Tensor> k_norm,
        float norm_eps,
        float norm_constant_bias,
        bool v_norm,
        c10::optional<at::Tensor> v_norm_w,
        float v_norm_eps,
        float v_norm_constant_bias,
        float v_norm_constant_scale,
        c10::optional<at::Tensor> inv_freq,
        int rope_style,
        float attn_factor,
        float l4_scaling_beta,
        int l4_scaling_original,
        int rotate_dims,
        bool quant_cache,
        at::Tensor cache_k,
        at::Tensor cache_v,
        c10::optional<at::Tensor> cache_k_scales,
        c10::optional<at::Tensor> cache_v_scales,
        at::Tensor xh,
        at::Tensor h32,
        c10::optional<at::Tensor> sinks
    );

    void set_qsa
    (
        std::shared_ptr<BC_LinearEXL3> qk_proj,
        at::Tensor q_norm_w,
        at::Tensor k_norm_w,
        float norm_eps,
        int n_heads,
        int head_dim,
        int topk,
        int compress_ratio,
        at::Tensor raw_plane,
        at::Tensor pool_plane
    );

    bool needs_configure(int bsz, int q_len, int regime);

    void configure_slot
    (
        int bsz,
        int q_len,
        int regime,
        at::Tensor q,
        at::Tensor kv,
        at::Tensor o,
        at::Tensor partial_o,
        at::Tensor partial_ml,
        c10::optional<at::Tensor> gate_a,
        c10::optional<at::Tensor> gate_b,
        std::shared_ptr<TritonKernel> k_split,
        std::shared_ptr<TritonKernel> k_combine,
        std::shared_ptr<TritonKernel> k_update,
        int block_n,
        int splits_cap,
        c10::optional<at::Tensor> xp,
        c10::optional<at::Tensor> yp
    );

    // Attaches the QSA statics/kernels to an already-configured slot. Selection pieces are
    // nullopt/null for regime-0 slots
    void configure_slot_qsa
    (
        int bsz,
        int q_len,
        int regime,
        at::Tensor qk,
        at::Tensor q,
        at::Tensor kraw,
        std::shared_ptr<TritonKernel> k_stage,
        std::shared_ptr<TritonKernel> k_raw_append,
        std::shared_ptr<TritonKernel> k_pool_update,
        int rotate_dims,
        c10::optional<at::Tensor> wts,
        c10::optional<at::Tensor> scores,
        c10::optional<at::Tensor> pool_idx,
        c10::optional<at::Tensor> indices,
        std::shared_ptr<TritonKernel> k_fewq,
        std::shared_ptr<TritonKernel> k_expand,
        std::shared_ptr<TritonKernel> k_split,
        std::shared_ptr<TritonKernel> k_combine,
        int fewq_gy,
        int qsa_splits,
        int qsa_split_len,
        int qsa_programs
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
        const c10::optional<at::Tensor>& position_ids,
        const c10::optional<at::Tensor>& inv_freq_override,
        int regime,
        int64_t t_total
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
        const c10::optional<at::Tensor>& inv_freq_override,
        int regime,
        int64_t t_total,
        Graph* graph
    );

private:
    Slot& slot(int bsz, int q_len, int regime)
        { return slots[(regime * MAX_BSZ + bsz - 1) * MAX_QLEN + (q_len - 1)]; }
};
