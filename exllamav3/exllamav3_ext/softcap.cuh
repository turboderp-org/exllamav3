#pragma once

#include <ATen/Tensor.h>

void softcap
(
    at::Tensor x,
    at::Tensor y,
    float softcap_factor
);
