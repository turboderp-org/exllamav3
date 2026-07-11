#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include "linear.h"
#include "gated_rmsnorm.h"

struct BC_GatedDeltaNet
{
    at::Tensor mixed_qkv;
    at::Tensor z;
    at::Tensor beta;
    at::Tensor g;
    at::Tensor qkvz;
    at::Tensor ba;
    at::Tensor conv_temp_a;
    at::Tensor conv_temp_b;
    at::Tensor core_attn_out;
    at::Tensor core_attn_out_f;
    std::shared_ptr<BC_LinearEXL3> qkvz_proj;
    std::shared_ptr<BC_LinearFP16> ba_proj;
    at::Tensor dt_bias;
    at::Tensor a_log;
    int num_k_heads;
    int num_v_heads;
    int k_head_dim;
    int v_head_dim;
    at::Tensor conv1d_weight;
    c10::optional<at::Tensor> conv1d_bias;
    std::shared_ptr<BC_GatedRMSNorm> norm;
    std::shared_ptr<BC_LinearEXL3> o_proj;
    const float beta_scale;

    BC_GatedDeltaNet
    (
        at::Tensor _mixed_qkv,
        at::Tensor _z,
        at::Tensor _beta,
        at::Tensor _g,
        at::Tensor _qkvz,
        at::Tensor _ba,
        at::Tensor _conv_temp_a,
        at::Tensor _conv_temp_b,
        at::Tensor _core_attn_out,
        at::Tensor _core_attn_out_f,
        std::shared_ptr<BC_LinearEXL3> _qkvz_proj,
        std::shared_ptr<BC_LinearFP16> _ba_proj,
        at::Tensor _dt_bias,
        at::Tensor _a_log,
        int _num_k_heads,
        int _num_v_heads,
        int _k_head_dim,
        int _v_head_dim,
        at::Tensor _conv1d_weight,
        c10::optional<at::Tensor> _conv1d_bias,
        std::shared_ptr<BC_GatedRMSNorm> _norm,
        std::shared_ptr<BC_LinearEXL3> _o_proj,
        const float _beta_scale
    ) :
        mixed_qkv       (std::move(_mixed_qkv)),
        z               (std::move(_z)),
        beta            (std::move(_beta)),
        g               (std::move(_g)),
        qkvz            (std::move(_qkvz)),
        ba              (std::move(_ba)),
        conv_temp_a     (std::move(_conv_temp_a)),
        conv_temp_b     (std::move(_conv_temp_b)),
        core_attn_out   (std::move(_core_attn_out)),
        core_attn_out_f (std::move(_core_attn_out_f)),
        qkvz_proj       (_qkvz_proj),
        ba_proj         (_ba_proj),
        dt_bias         (std::move(_dt_bias)),
        a_log           (std::move(_a_log)),
        num_k_heads     (_num_k_heads),
        num_v_heads     (_num_v_heads),
        k_head_dim      (_k_head_dim),
        v_head_dim      (_v_head_dim),
        conv1d_weight   (std::move(_conv1d_weight)),
        conv1d_bias     (std::move(_conv1d_bias)),
        norm            (_norm),
        o_proj          (_o_proj),
        beta_scale      (_beta_scale)
    {}

    at::Tensor run_bsz1_a
    (
        const at::Tensor& x
    );

    void run_bsz1_b
    (
        at::Tensor& mixed_qkv,
        at::Tensor& y,
        at::Tensor& recurrent_state
    );
};


// Split-projection variant (Qwen3.5: in_proj_qkv/in_proj_z/in_proj_b/in_proj_a). Runs the whole
// GDN layer for a single decode token in one call, replayed via an internal CUDA graph after the
// second invocation. b/a projections are merged into one fp16 GEMV at construction

struct BC_GatedDeltaNetSplit
{
    // Preallocated bsz-1 scratch buffers (shared between layers via the python tensor cache)
    at::Tensor qkv;               // (1, 1, F) float
    at::Tensor z;                 // (1, 1, Nv, Hv) float
    at::Tensor ba;                // (1, 1, 2*Nv) float
    at::Tensor beta;              // (1, 1, Nv) bfloat16
    at::Tensor g;                 // (1, 1, Nv) float
    at::Tensor mixed_qkv;         // (1, F, 1) bfloat16
    at::Tensor conv_out;          // (1, 1, F) bfloat16
    at::Tensor core_attn_out;     // (1, 1, Nv, Hv) bfloat16
    at::Tensor core_attn_out_f;   // (1, 1, Nv*Hv) half
    at::Tensor z_flat;            // (1, 1, Nv*Hv) view of z

    std::shared_ptr<BC_LinearEXL3> qkv_proj;
    std::shared_ptr<BC_LinearEXL3> z_proj;
    std::shared_ptr<BC_LinearEXL3> o_proj;
    at::Tensor ba_weight_t;       // (2*Nv, hidden) half
    c10::optional<at::Tensor> ba_bias;
    at::Tensor dt_bias;
    at::Tensor a_log;
    int num_k_heads;
    int num_v_heads;
    int k_head_dim;
    int v_head_dim;
    at::Tensor conv1d_weight;     // (F, K) bfloat16
    c10::optional<at::Tensor> conv1d_bias;
    std::shared_ptr<BC_GatedRMSNorm> norm;
    const float beta_scale;

    Graph graph_bsz1;
    int graph_state_size;
    int graph_hist_stride;

    BC_GatedDeltaNetSplit
    (
        at::Tensor _qkv,
        at::Tensor _z,
        at::Tensor _ba,
        at::Tensor _beta,
        at::Tensor _g,
        at::Tensor _mixed_qkv,
        at::Tensor _conv_out,
        at::Tensor _core_attn_out,
        at::Tensor _core_attn_out_f,
        std::shared_ptr<BC_LinearEXL3> _qkv_proj,
        std::shared_ptr<BC_LinearEXL3> _z_proj,
        std::shared_ptr<BC_LinearEXL3> _o_proj,
        at::Tensor _ba_weight_t,
        c10::optional<at::Tensor> _ba_bias,
        at::Tensor _dt_bias,
        at::Tensor _a_log,
        int _num_k_heads,
        int _num_v_heads,
        int _k_head_dim,
        int _v_head_dim,
        at::Tensor _conv1d_weight,
        c10::optional<at::Tensor> _conv1d_bias,
        std::shared_ptr<BC_GatedRMSNorm> _norm,
        const float _beta_scale
    ) :
        qkv             (std::move(_qkv)),
        z               (std::move(_z)),
        ba              (std::move(_ba)),
        beta            (std::move(_beta)),
        g               (std::move(_g)),
        mixed_qkv       (std::move(_mixed_qkv)),
        conv_out        (std::move(_conv_out)),
        core_attn_out   (std::move(_core_attn_out)),
        core_attn_out_f (std::move(_core_attn_out_f)),
        qkv_proj        (_qkv_proj),
        z_proj          (_z_proj),
        o_proj          (_o_proj),
        ba_weight_t     (std::move(_ba_weight_t)),
        ba_bias         (std::move(_ba_bias)),
        dt_bias         (std::move(_dt_bias)),
        a_log           (std::move(_a_log)),
        num_k_heads     (_num_k_heads),
        num_v_heads     (_num_v_heads),
        k_head_dim      (_k_head_dim),
        v_head_dim      (_v_head_dim),
        conv1d_weight   (std::move(_conv1d_weight)),
        conv1d_bias     (std::move(_conv1d_bias)),
        norm            (_norm),
        beta_scale      (_beta_scale),
        graph_state_size(-1),
        graph_hist_stride(-1)
    {
        z_flat = z.view({1, 1, -1});
    }

    void run_bsz1_gr
    (
        const at::Tensor& x,
        at::Tensor& y,
        at::Tensor& conv_state,
        at::Tensor& recurrent_state,
        const at::Tensor& slots,
        Graph* graph
    );

    void run_bsz1
    (
        const at::Tensor& x,
        at::Tensor& y,
        at::Tensor& conv_state,
        at::Tensor& recurrent_state,
        const at::Tensor& slots
    );
};