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

void pg_all_reduce_cpu
(
    uintptr_t ctx,
    std::vector<uintptr_t> devices,
    int this_device,
    int master_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t shbuf,
    size_t shbuf_size,
    bool is_master
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