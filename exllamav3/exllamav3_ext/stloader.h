#pragma once

#include <ATen/Tensor.h>
#include <vector>

#define STLOADER_BLOCK_SIZE (512*1024)
#define STLOADER_THREADS 8

void stloader_read
(
    std::vector<uintptr_t> handles,
    size_t offset,
    size_t size,
    at::Tensor target
);

std::vector<uintptr_t> stloader_open_file(const char* filename);
void stloader_close_file(std::vector<uintptr_t> handles);

struct TensorLoadJob {
    std::vector<uintptr_t> handles;
    size_t file_offset;
    size_t bytesize;
    uintptr_t destination;
    bool bf16_to_fp16;
    bool fp32_to_fp16;
    bool cuda;
    int device_id;
};

void stloader_deferred_cpu(std::vector<TensorLoadJob> const& jobs);
void stloader_deferred_cuda(std::vector<TensorLoadJob> const& jobs, size_t max_chunk_size);
