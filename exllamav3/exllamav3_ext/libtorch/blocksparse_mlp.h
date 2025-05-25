#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

std::tuple<at::Tensor, at::Tensor> blocksparse_mlp_routing(
    int bsz,
    const py::object& cfg,
    const at::Tensor& y,
    const py::dict& params
);