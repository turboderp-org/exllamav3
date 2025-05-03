#pragma once

void inplace_bf16_to_fp16_cpu
(
    void* buffer,
    size_t numel
);

void inplace_bf16_to_fp16_cuda
(
    void* buffer,
    size_t numel
 );
