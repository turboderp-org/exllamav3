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

    BC_GatedRMSNorm
    (
        at::Tensor _weight,
        float _rms_norm_eps,
        float _constant_bias
    ) :
        weight(std::move(_weight)),
        rms_norm_eps(_rms_norm_eps),
        constant_bias(_constant_bias)
    {}

    void run(const at::Tensor& x, at::Tensor& y, const at::Tensor& gate);
};