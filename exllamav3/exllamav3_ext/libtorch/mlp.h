#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "linear.h"
#include "../graph.cuh"

// Gate and up projections run as one fused MGEMM when the pointer tables are given, otherwise
// as two separate GEMV/GEMM calls through the gate/up BC handles (the unfused configuration the
// int8-activation GEMV path prefers for wide matrices)

struct BC_GatedMLP
{
    at::Tensor guh;
    at::Tensor gu;
    at::Tensor a;
    c10::optional<at::Tensor> gu_ptrs_trellis;
    c10::optional<at::Tensor> gu_ptrs_suh;
    c10::optional<at::Tensor> gu_ptrs_svh;
    int gu_K;
    bool gu_mcg;
    bool gu_mul1;
    bool act_silu;
    bool act_gelu;
    bool act_relu2;
    std::shared_ptr<BC_LinearEXL3> gate;
    std::shared_ptr<BC_LinearEXL3> up;
    std::shared_ptr<BC_LinearEXL3> down;
    float act_limit;

    Graph graph_bsz1;

    BC_GatedMLP
    (
        at::Tensor _guh,
        at::Tensor _gu,
        at::Tensor _a,
        c10::optional<at::Tensor> _gu_ptrs_trellis,
        c10::optional<at::Tensor> _gu_ptrs_suh,
        c10::optional<at::Tensor> _gu_ptrs_svh,
        int _gu_K,
        bool _gu_mcg,
        bool _gu_mul1,
        bool _act_silu,
        bool _act_gelu,
        bool _act_relu2,
        std::shared_ptr<BC_LinearEXL3> _gate,
        std::shared_ptr<BC_LinearEXL3> _up,
        std::shared_ptr<BC_LinearEXL3> _down,
        float _act_limit
    ) :
        guh                 (std::move(_guh)),
        gu                  (std::move(_gu)),
        a                   (std::move(_a)),
        gu_ptrs_trellis     (std::move(_gu_ptrs_trellis)),
        gu_ptrs_suh         (std::move(_gu_ptrs_suh)),
        gu_ptrs_svh         (std::move(_gu_ptrs_svh)),
        gu_K                (_gu_K),
        gu_mcg              (_gu_mcg),
        gu_mul1             (_gu_mul1),
        act_silu            (_act_silu),
        act_gelu            (_act_gelu),
        act_relu2           (_act_relu2),
        gate                (_gate),
        up                  (_up),
        down                (_down),
        act_limit           (_act_limit)
    {
        TORCH_CHECK(gu_ptrs_trellis.has_value() || (gate && up), "BC_GatedMLP: need fused mgemm tensors or gate/up handles");
    }

    void run_bsz1_gr
    (
        const at::Tensor& x,
        at::Tensor& d,
        Graph* graph
    );

    void run_bsz1
    (
        const at::Tensor& x,
        at::Tensor& d
    );
};