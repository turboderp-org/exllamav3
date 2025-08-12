#pragma once

#include <ATen/Tensor.h>

void pg_all_reduce
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf,
    size_t shbuf_size
);
