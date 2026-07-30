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
// One lazily configured slot per (bsz, q_len). The block-table width and split configuration
// are runtime kernel arguments patched per call, so context growth never recaptures. Instances
// are built per cache layer, since the cache tensors are baked into the captured graphs.

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
        at::Tensor w_uk_flat,
        at::Tensor w_uv_flat,
        bool quant_cache,
        at::Tensor cache_ckv,
        at::Tensor cache_kpe,
        c10::optional<at::Tensor> cache_scales,
        at::Tensor xh,
        at::Tensor h32
    );

    bool needs_configure(int bsz, int q_len);

    void configure_slot
    (
        int bsz,
        int q_len,
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

private:
    Slot& slot(int bsz, int q_len) { return slots[(bsz - 1) * MAX_QLEN + (q_len - 1)]; }
};
