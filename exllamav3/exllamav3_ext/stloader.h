#pragma once

#include <ATen/Tensor.h>
#include <vector>

#define STLOADER_BLOCK_SIZE (512*1024)
#define STLOADER_THREADS 5

void stloader_read
(
    std::vector<uintptr_t> handles,
    size_t offset,
    size_t size,
    at::Tensor target
);

std::vector<uintptr_t> stloader_open_file(const char* filename);
void stloader_close_file(std::vector<uintptr_t> handles);
