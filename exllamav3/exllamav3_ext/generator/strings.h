#pragma once

#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <ATen/Tensor.h>

namespace py = pybind11;

int partial_strings_match
(
    py::buffer match,
    py::buffer offsets,
    py::buffer strings
);

int count_match_tensor
(
    at::Tensor a,
    at::Tensor b,
    int max_a
);
