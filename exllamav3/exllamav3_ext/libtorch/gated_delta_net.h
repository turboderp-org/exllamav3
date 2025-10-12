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
        std::shared_ptr<BC_LinearEXL3> _o_proj
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
        o_proj          (_o_proj)
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