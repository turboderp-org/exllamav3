#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <memory>
#include "../graph.cuh"
#include "../triton_kernel.h"
#include "linear.h"

// Graph-captured decode attention block for MLA (DeepSeek-style latent attention), following
// the BC_Attention pattern: run() controls the mode (first run eager, second run captured,
// later runs patch-and-replay), run_gr() is the workload -- q projections (direct or LoRA with
// q_a norm), the latent projection, a fused staging kernel (kv_a norm + rope-key split + per-
// head rope-query gather), partial RoPE on the rope halves, W_UK absorption, cache append
// (fp16 or packed-quant), the absorbed flash-decoding split/combine kernels, W_UV unfold and
// o_proj -- all recorded as one graph. The absorb/unfold GEMMs and the attention kernels are
// AOT-compiled Triton cubins launched through TritonKernel; kv_b exists only as the flat
// (D_c, H * d) tensors those kernels read per-head column blocks from.
//
// One lazily configured slot per (bsz, q_len, regime). The block-table width and split
// configuration are runtime kernel arguments patched per call, so context growth never
// recaptures. Instances are built per cache layer, since the cache tensors are baked into the
// captured graphs.
//
// DSA-on-MLA (GLM-5.2): set_indexer() arms the optional lightning-indexer stages. Full-indexer
// layers append roped indexer keys to a paged side plane every step; in the sparse regime
// (context past index_topk) they additionally score the plane, select top-k per query
// (capture-safe radix top-k) and attend over the gathered latent rows instead of the dense
// flash-decoding kernels: the gathered kernels read the absorbed q_lat and rope q_pe statics
// directly (Q_SPLIT) and the combine emits the head-major latent the unfold consumes
// (OUT_LATENT), so no packed query or output staging exists.
//
// Shared-indexer layers skip scoring and gather through an EXTERNAL
// index list whose pointer is patched per call (GP_dsa_indices), so the selection can come
// from another layer's graph, the eager path, or across devices. The sparse regime follows
// the BC_DSV4Attention pattern (dsv4_attn.cpp).
//
// Batched sparse slots (bsz > 1) run the scoring / top-k / gather stages in MULTIROW mode:
// the per-job scan widths and causal clamps live in a small device state array derived from
// the (already patched) cache_seqlens tensor by a leading kernel (dsa_seq_state_gr), the
// block table is consumed one row per job, and the top-k takes the per-job bound as a device
// pointer, so batched sparse replay patches nothing beyond what the dense path patches.

struct BC_MLAttention
{
    static constexpr int MAX_BSZ = 8;
    static constexpr int MAX_QLEN = 16;

    // Model config
    int num_q_heads;
    int hidden_size;
    int page_size;
    int kv_lora_rank;       // D_c, latent width (cached)
    int qk_rope_head_dim;   // D_r, shared rope key width (cached separately, always fp16)
    int qk_nope_head_dim;
    int qk_head_dim;        // nope + rope
    int v_head_dim;
    int q_lora_rank;        // 0 = direct q projection

    // Projections. q_proj is q_b_proj when q_lora_rank > 0
    std::shared_ptr<BC_LinearEXL3> q_proj;
    std::shared_ptr<BC_LinearEXL3> q_a_proj;
    c10::optional<at::Tensor> q_a_norm_w;
    std::shared_ptr<BC_LinearEXL3> kv_a_proj;
    std::shared_ptr<BC_LinearEXL3> o_proj;
    at::Tensor kv_norm_w;   // kv_a_layernorm weight, applied inside the staging kernel
    float norm_eps;

    // Partial RoPE over the rope halves (q_pe/k_pe), GPTJ or NEOX per rope_style
    at::Tensor inv_freq;
    int rope_style;
    float attn_factor;
    int rotate_dims;

    // Llama-4 position scale on the full query (queries only, applied to q_full before the
    // pe-gather/rope/absorb stages, matching the module's eager-path multiply); 0 = disabled
    float l4_scaling_beta;
    int l4_scaling_original;

    // kv_b flats: (D_c, H * qk_nope_head_dim) and (D_c, H * v_head_dim)
    at::Tensor w_uk_flat;
    at::Tensor w_uv_flat;

    // Cache pages: latent fp16 (pages, page_size, 1, D_c) or packed int32 + group scales;
    // the rope key pages are fp16 either way
    bool quant_cache;
    at::Tensor cache_ckv;
    at::Tensor cache_kpe;
    c10::optional<at::Tensor> cache_scales;

    // Shared statics: hadamard scratch for the EXL3 GEMMs and the H32 rotation matrix for
    // quantized-cache kernels (any small tensor when unused)
    at::Tensor xh;
    at::Tensor h32;

    // DSA lightning indexer (set_indexer): 0 = none, 1 = full, 2 = shared
    int idx_mode = 0;
    std::shared_ptr<BC_LinearEXL3> idx_wq_b;   // full only, quantized (q_lora_rank -> H_i * D_i)
    c10::optional<at::Tensor> idx_wk_w;        // (hidden, D_i) fp16, full only
    c10::optional<at::Tensor> idx_k_norm_w;    // (D_i,) fp16
    c10::optional<at::Tensor> idx_k_norm_b;    // (D_i,) fp16
    c10::optional<at::Tensor> idx_weights_w;   // (hidden, H_i) fp16
    c10::optional<at::Tensor> cache_kidx;      // flat (pages * page_size, D_i) fp16 plane
    int index_n_heads = 0;
    int index_head_dim = 0;
    int index_topk = 0;
    // K-pool compression (GLM5.3): plane rows are packed [k || gate_scores] (2 * D_i wide),
    // pooled keys live in a second plane and scoring/top-k run over pools
    int index_kpool = 0;
    bool index_kpool_tail = true;
    c10::optional<at::Tensor> idx_gate_w;      // (hidden, D_i) fp16
    c10::optional<at::Tensor> idx_kpool_ape;   // (P, D_i) fp16
    c10::optional<at::Tensor> cache_kpool;     // flat (pages * page_size / P, D_i) fp16

    struct Slot
    {
        bool configured = false;
        int runs = 0;
        int block_n = 0;
        int splits_cap = 0;   // grid height of the split kernel; the live split count is a
                              // runtime argument patched per call, extra splits idle
        int programs = 0;
        int absorb_gx = 0;
        int absorb_gy = 0;
        int unfold_gx = 0;

        // Static intermediates (python tensor cache) and precomputed views
        at::Tensor q_full;    // (R, q_proj out width) fp16
        at::Tensor q_a;       // (R, q_a out width) fp16, q_lora only
        at::Tensor ckv_kpe;   // (R, kv_a out width) fp16
        at::Tensor ckv;       // (R, D_c) fp16, normalized latent
        at::Tensor kpe;       // (R, D_r) fp16
        at::Tensor q_pe;      // (R, H, D_r) fp16 token-major
        at::Tensor q_lat;     // (H, R, D_c) fp16 head-major
        at::Tensor o_lat;     // (H, R, D_c) fp16 head-major
        at::Tensor o;         // (R, H * D_v) fp16
        at::Tensor partial_o, partial_ml;
        at::Tensor qtmp, stmp;   // quant append temporaries
        at::Tensor q_pe4, kpe4;  // rope views (bsz, q_len, heads, D_r)

        std::shared_ptr<TritonKernel> k_stage;
        std::shared_ptr<TritonKernel> k_absorb;
        std::shared_ptr<TritonKernel> k_append;   // fp16 update or quant scatter
        std::shared_ptr<TritonKernel> k_split;
        std::shared_ptr<TritonKernel> k_combine;
        std::shared_ptr<TritonKernel> k_unfold;

        // DSA statics (idx_mode > 0). Full-indexer slots carry the key stages in both
        // regimes; sparse slots add scoring/selection (full) and the gathered attention
        at::Tensor x_st;      // (R_pad, hidden) staged input, zero-padded rows for cuBLASLt
        at::Tensor kidx;      // (R_pad, D_i) raw wk output
        at::Tensor kidx_n;    // (R, D_i) normed keys, roped in place on the leading D_r dims
        at::Tensor kidx4;     // rope view (bsz, q_len, 1, D_r)
        at::Tensor qidx;      // (R, H_i * D_i), sparse full only
        at::Tensor qidx4;     // rope view (bsz, q_len, H_i, D_r) of the leading dims
        at::Tensor wts;       // (R_pad, H_i)
        at::Tensor scores;    // (R, S_max)
        at::Tensor indices;   // (R, K_pad): own selection (full) or patch target (shared)
        at::Tensor dsa_arr;   // (2, MAX_BSZ) i32 per-job [q_pos0; past + q_len], bsz > 1 only:
                              // filled on device from cache_seqlens each step (dsa_seq_state)
        at::Tensor dsa_ws_ml, dsa_ws_acc;
        at::Tensor gidx;      // (R, D_i) gate-score rows (kpool)
        at::Tensor pool_idx;  // (R, KP_pool) selected pool ids (kpool sparse)
        std::shared_ptr<TritonKernel> k_idx_norm, k_plane_append, k_fewq,
            k_dsa_split, k_dsa_combine;
        std::shared_ptr<TritonKernel> k_gate_append, k_pool_update, k_pool_expand;
        int dsa_hb = 0;
        int dsa_splits = 0;
        int fewq_gy = 0;

        std::unique_ptr<Graph> graph;
    };
    std::vector<Slot> slots;

    BC_MLAttention
    (
        int num_q_heads,
        int hidden_size,
        int page_size,
        int kv_lora_rank,
        int qk_rope_head_dim,
        int qk_nope_head_dim,
        int v_head_dim,
        int q_lora_rank,
        std::shared_ptr<BC_LinearEXL3> q_proj,
        std::shared_ptr<BC_LinearEXL3> q_a_proj,
        c10::optional<at::Tensor> q_a_norm_w,
        std::shared_ptr<BC_LinearEXL3> kv_a_proj,
        std::shared_ptr<BC_LinearEXL3> o_proj,
        at::Tensor kv_norm_w,
        float norm_eps,
        at::Tensor inv_freq,
        int rope_style,
        float attn_factor,
        int rotate_dims,
        float l4_scaling_beta,
        int l4_scaling_original,
        at::Tensor w_uk_flat,
        at::Tensor w_uv_flat,
        bool quant_cache,
        at::Tensor cache_ckv,
        at::Tensor cache_kpe,
        c10::optional<at::Tensor> cache_scales,
        at::Tensor xh,
        at::Tensor h32
    );

    void set_indexer
    (
        int mode,
        std::shared_ptr<BC_LinearEXL3> wq_b,
        c10::optional<at::Tensor> wk_w,
        c10::optional<at::Tensor> k_norm_w,
        c10::optional<at::Tensor> k_norm_b,
        c10::optional<at::Tensor> weights_w,
        c10::optional<at::Tensor> kidx,
        int n_heads,
        int head_dim,
        int topk,
        int kpool = 0,
        bool kpool_tail = true,
        c10::optional<at::Tensor> gate_w = {},
        c10::optional<at::Tensor> kpool_ape = {},
        c10::optional<at::Tensor> kpool_plane = {}
    );

    bool needs_configure(int bsz, int q_len, int regime);

    void configure_slot
    (
        int bsz,
        int q_len,
        int regime,
        at::Tensor q_full,
        c10::optional<at::Tensor> q_a,
        at::Tensor ckv_kpe,
        at::Tensor ckv,
        at::Tensor kpe,
        at::Tensor q_pe,
        at::Tensor q_lat,
        at::Tensor o_lat,
        at::Tensor o,
        at::Tensor partial_o,
        at::Tensor partial_ml,
        c10::optional<at::Tensor> qtmp,
        c10::optional<at::Tensor> stmp,
        std::shared_ptr<TritonKernel> k_stage,
        std::shared_ptr<TritonKernel> k_absorb,
        std::shared_ptr<TritonKernel> k_append,
        std::shared_ptr<TritonKernel> k_split,
        std::shared_ptr<TritonKernel> k_combine,
        std::shared_ptr<TritonKernel> k_unfold,
        int block_n,
        int splits_cap,
        int programs,
        int absorb_gx,
        int absorb_gy,
        int unfold_gx
    );

    // Attaches the DSA statics/kernels to an already-configured slot. Sparse-only pieces are
    // nullopt for regime 0; scoring pieces are nullopt for shared-indexer instances
    void configure_slot_dsa
    (
        int bsz,
        int q_len,
        int regime,
        c10::optional<at::Tensor> x_st,
        c10::optional<at::Tensor> kidx,
        c10::optional<at::Tensor> kidx_n,
        c10::optional<at::Tensor> qidx,
        c10::optional<at::Tensor> wts,
        c10::optional<at::Tensor> scores,
        c10::optional<at::Tensor> indices,
        c10::optional<at::Tensor> dsa_arr,
        c10::optional<at::Tensor> dsa_ws_ml,
        c10::optional<at::Tensor> dsa_ws_acc,
        std::shared_ptr<TritonKernel> k_idx_norm,
        std::shared_ptr<TritonKernel> k_plane_append,
        std::shared_ptr<TritonKernel> k_fewq,
        std::shared_ptr<TritonKernel> k_dsa_split,
        std::shared_ptr<TritonKernel> k_dsa_combine,
        int dsa_hb,
        int dsa_splits,
        int fewq_gy,
        c10::optional<at::Tensor> gidx = {},
        c10::optional<at::Tensor> pool_idx = {},
        std::shared_ptr<TritonKernel> k_gate_append = nullptr,
        std::shared_ptr<TritonKernel> k_pool_update = nullptr,
        std::shared_ptr<TritonKernel> k_pool_expand = nullptr
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
        int regime,
        int64_t t_total,
        const c10::optional<at::Tensor>& ext_indices
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
        int regime,
        int64_t t_total,
        const c10::optional<at::Tensor>& ext_indices,
        Graph* graph
    );

private:
    Slot& slot(int bsz, int q_len, int regime)
        { return slots[(regime * MAX_BSZ + bsz - 1) * MAX_QLEN + (q_len - 1)]; }
};
