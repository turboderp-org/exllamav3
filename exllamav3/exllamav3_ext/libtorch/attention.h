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
// One lazily configured slot per (bsz, q_len) with bsz <= 4, q_len <= 16. The attention kernels
// bake the split configuration per slot; the block-table width and split length are runtime
// kernel arguments frozen at capture, so when the generator's block table grows the slot is
// reconfigured (recapture, no recompile unless the split count changes). Instances are built
// per cache layer, since the cache tensors are baked into the captured graphs.

struct BC_Attention
{
    static constexpr int MAX_BSZ = 4;
    static constexpr int MAX_QLEN = 16;

    // Model config
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int hidden_size;
    int page_size;

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

    // Output gate: 0 = none, 2 = full (o *= sigmoid(g), g from its own projection or the fused
    // q+g mgemm), 3 = interleaved (q_proj emits q/g interleaved per head). Headwise (1) is not
    // supported: its fp16 gate projection runs through cublas, which has no patchable sites
    int gate_mode;
    std::shared_ptr<BC_LinearEXL3> g_proj;
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

    // RoPE
    at::Tensor inv_freq;
    int rope_style;
    float attn_factor;
    float l4_scaling_beta;
    int l4_scaling_original;
    bool post_rope_norm;
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

    struct Slot
    {
        bool configured = false;
        int runs = 0;
        int bt_width = 0;
        int num_splits = 0;
        int split_len = 0;
        int programs = 0;
        dim3 upd_grid;

        // Static intermediates (python tensor cache) and precomputed views. gate_a/gate_b by
        // gate mode: full = qg (2, R, qh*hd) with q aliasing qg[0]; interleaved = qg_i
        // (R, 2*qh*hd) staging + g (R, qh*hd)
        at::Tensor q, kv, o, partial_o, partial_ml, gate_a, gate_b;
        at::Tensor q2, q4, k4, v4, o2, o4, qg2, g2;

        std::shared_ptr<TritonKernel> k_split;
        std::shared_ptr<TritonKernel> k_combine;   // null when num_splits == 1
        std::shared_ptr<TritonKernel> k_update;    // null when quant_cache

        std::unique_ptr<Graph> graph;
    };
    std::vector<Slot> slots;

    BC_Attention
    (
        int num_q_heads,
        int num_kv_heads,
        int head_dim,
        int hidden_size,
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
        std::shared_ptr<BC_LinearEXL3> g_proj,
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
        at::Tensor inv_freq,
        int rope_style,
        float attn_factor,
        float l4_scaling_beta,
        int l4_scaling_original,
        bool post_rope_norm,
        int rotate_dims,
        bool quant_cache,
        at::Tensor cache_k,
        at::Tensor cache_v,
        c10::optional<at::Tensor> cache_k_scales,
        c10::optional<at::Tensor> cache_v_scales,
        at::Tensor xh,
        at::Tensor h32
    );

    // True when the slot must be (re)configured before run(): not yet configured, or captured
    // against a different block-table width
    bool needs_configure(int bsz, int q_len, int bt_width);

    void configure_slot
    (
        int bsz,
        int q_len,
        int bt_width,
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
        int num_splits,
        int split_len
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
        const c10::optional<at::Tensor>& inv_freq_override
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
        Graph* graph
    );

private:
    Slot& slot(int bsz, int q_len) { return slots[(bsz - 1) * MAX_QLEN + (q_len - 1)]; }
};
