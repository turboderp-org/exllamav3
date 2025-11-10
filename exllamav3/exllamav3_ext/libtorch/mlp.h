#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "linear.h"
#include "../graph.cuh"

struct BC_GatedMLP
{
    at::Tensor guh;
    at::Tensor gu;
    at::Tensor a;
    at::Tensor gu_ptrs_trellis;
    at::Tensor gu_ptrs_suh;
    at::Tensor gu_ptrs_svh;
    int gu_K;
    bool gu_mcg;
    bool gu_mul1;
    bool act_silu;
    bool act_gelu;
    bool act_relu2;
    std::shared_ptr<BC_LinearEXL3> down;

    Graph graph_bsz1;

    BC_GatedMLP
    (
        at::Tensor _guh,
        at::Tensor _gu,
        at::Tensor _a,
        at::Tensor _gu_ptrs_trellis,
        at::Tensor _gu_ptrs_suh,
        at::Tensor _gu_ptrs_svh,
        int _gu_K,
        bool _gu_mcg,
        bool _gu_mul1,
        bool _act_silu,
        bool _act_gelu,
        bool _act_relu2,
        std::shared_ptr<BC_LinearEXL3> _down
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
        down                (_down)
    {}

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