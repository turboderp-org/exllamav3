#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct BC_GatedRMSNorm
{
    at::Tensor weight;
    float rms_norm_eps;
    float constant_bias;
    int w_groups;       // Mamba2 group norm: weight row selected by (row % w_groups)
    bool gate_first;    // Mamba2 style: y = norm(x * silu(g)) * w

    BC_GatedRMSNorm
    (
        at::Tensor _weight,
        float _rms_norm_eps,
        float _constant_bias,
        int _w_groups = 1,
        bool _gate_first = false
    ) :
        weight(std::move(_weight)),
        rms_norm_eps(_rms_norm_eps),
        constant_bias(_constant_bias),
        w_groups(_w_groups),
        gate_first(_gate_first)
    {}

    void run(const at::Tensor& x, at::Tensor& y, const at::Tensor& gate);
    void run_gr(const at::Tensor& x, at::Tensor& y, const at::Tensor& gate, class Graph* graph);
};