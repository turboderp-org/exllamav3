#pragma once

#include <ATen/Tensor.h>

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
);