#pragma once

#include <ATen/Tensor.h>

// ctx is the host VA of the registered shared-memory region, used for host-side access. ctx_dev and shbuf_dev are
// device aliases (cudaHostGetDevicePointer) of the context and transfer buffer, passed to kernels. Aliases are
// identical to the host VAs on platforms where the host pointer is directly usable in kernels (Linux), but differ
// under WDDM.
void pg_gather
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
);

void pg_gather_small
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int out_device,
    at::Tensor& tensor,
    c10::optional<at::Tensor>& out_tensor,
    std::vector<size_t> ldims,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
);
