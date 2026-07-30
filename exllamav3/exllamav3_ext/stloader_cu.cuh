#pragma once

#include <cuda_runtime.h>

void inplace_bf16_to_fp16_cpu
(
    void* buffer,
    const size_t numel
);

cudaError_t inplace_bf16_to_fp16_cuda
(
    void* buffer,
    const size_t numel,
    cudaStream_t stream
);
