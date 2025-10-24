#pragma once

void inplace_bf16_to_fp16_cpu
(
    void* buffer,
    const size_t numel
);

void inplace_bf16_to_fp16_cuda
(
    void* buffer,
    const size_t numel
);
