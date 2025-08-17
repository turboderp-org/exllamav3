#pragma once

#include <ATen/Tensor.h>

void pg_broadcast
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
);

void pg_broadcast_ll
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t shbuf,
    size_t shbuf_size,
    at::Tensor& abort_flag
);
