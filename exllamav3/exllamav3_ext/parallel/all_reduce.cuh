#pragma once

#include <ATen/Tensor.h>

// ctx is the host VA of the registered shared-memory region, used for host-side access (timeout check, CPU-reduce
// job queue). ctx_dev and shbuf_dev are device aliases (cudaHostGetDevicePointer) of the context and transfer
// buffer, passed to kernels. Aliases are identical to the host VAs on platforms where the host pointer is directly
// usable in kernels (Linux), but differ under WDDM. run_cpu_reduce_jobs/end_cpu_reduce_jobs run in the CPU helper
// process and take host VAs only.
void pg_all_reduce
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    at::Tensor& abort_flag
);

void pg_all_reduce_cpu
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t shbuf_dev,
    size_t shbuf_size,
    bool is_master,
    at::Tensor& abort_flag
);

void run_cpu_reduce_jobs
(
    uintptr_t ctx_ptr,
    uintptr_t shbuf,
    size_t shbuf_size
);

void end_cpu_reduce_jobs
(
    uintptr_t ctx_ptr
);
