#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

struct BC_LinearFP16
{
    at::Tensor weight;
    c10::optional<at::Tensor> bias;

    BC_LinearFP16
    (
        at::Tensor _weight,
        c10::optional<at::Tensor> _bias
    ) :
        weight(std::move(_weight)),
        bias(std::move(_bias))
    {}

    void run(const at::Tensor& x, at::Tensor& y);
    void run_cublas(const at::Tensor& x, at::Tensor& y);
};

struct BC_LinearEXL3
{
    at::Tensor trellis;
    at::Tensor suh;
    at::Tensor svh;
    int K;
    c10::optional<at::Tensor> bias;
    uint32_t mcg_mult;
    uint32_t mul1_mult;
    at::Tensor xh;

    BC_LinearEXL3
    (
        at::Tensor _trellis,
        at::Tensor _suh,
        at::Tensor _svh,
        int _K,
        c10::optional<at::Tensor> _bias,
        uint32_t _mcg_mult,
        uint32_t _mul1_mult,
        at::Tensor _xh
    ) :
        trellis(std::move(_trellis)),
        suh(std::move(_suh)),
        svh(std::move(_svh)),
        K(_K),
        bias(std::move(_bias)),
        mcg_mult(_mcg_mult),
        mul1_mult(_mul1_mult),
        xh(std::move(_xh))
    {}

    void run(const at::Tensor& x, at::Tensor& y);
};
