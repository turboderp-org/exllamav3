#pragma once

#include <ATen/Tensor.h>
#include "context.cuh"

// ctx is the host VA of the registered shared-memory region, used for host-side access. ctx_dev is the device
// alias of the same region (cudaHostGetDevicePointer), passed to kernels. The two are identical on platforms
// where the host pointer is directly usable in kernels (Linux), but differ under WDDM.
void pg_barrier
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& abort_flag
);
