#pragma once

#include <ATen/Tensor.h>
#include "context.cuh"

void pg_barrier
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
);