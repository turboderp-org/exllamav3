#pragma once

#include <ATen/Tensor.h>
#include <vector>

#define STLOADER_BLOCK_SIZE (512*1024)
#define STLOADER_THREADS 8

// CUDA loads stage through a shared pool of pinned ring buffers: one ring of
// STLOADER_SLOTS_PER_THREAD slots of STLOADER_SLOT_SIZE bytes per worker thread. Adjacent jobs
// are coalesced into single reads of up to one slot, and slot recycling waits on a per-slot
// event rather than a device sync, so disk reads overlap H2D copies.
#define STLOADER_SLOT_SIZE (4*1024*1024)
#define STLOADER_SLOTS_PER_THREAD 4

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
