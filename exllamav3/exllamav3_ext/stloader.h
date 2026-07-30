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

// bf16 -> fp16 conversion runs per job, in place, over bytesize/2 elements. Every read length has
// to be even or a bf16 element would straddle two jobs and be converted as two halves of nothing.
// That holds for tensor sizes trivially, but not for the boundaries a large tensor is split on,
// so the block/slot sizes those boundaries derive from are pinned here. The caller-side chunk
// size is checked to match in stloader_deferred_cuda() and in the loader's Python front end.
static_assert(STLOADER_BLOCK_SIZE % 2 == 0, "STLOADER_BLOCK_SIZE must be even");
static_assert(STLOADER_SLOT_SIZE % 2 == 0, "STLOADER_SLOT_SIZE must be even");

void stloader_read
(
    std::vector<uintptr_t> handles,
    size_t offset,
    size_t size,
    at::Tensor target
);

std::vector<uintptr_t> stloader_open_file(const char* filename);
void stloader_close_file(std::vector<uintptr_t> handles);

// A job writes bytesize bytes to a raw destination pointer, so the tensor it was allocated
// against is not visible from here and its extent cannot be recovered. dest_size is the caller's
// declared capacity at that pointer, checked against bytesize before any worker runs; without it
// the loader has no way to reject a job that would write past the end of its destination.
struct TensorLoadJob {
    std::vector<uintptr_t> handles;
    size_t file_offset;
    size_t bytesize;
    uintptr_t destination;
    size_t dest_size;
    bool bf16_to_fp16;
    bool fp32_to_fp16;
    bool cuda;
    int device_id;
};

void stloader_deferred_cpu(std::vector<TensorLoadJob> const& jobs);
void stloader_deferred_cuda(std::vector<TensorLoadJob> const& jobs, size_t max_chunk_size);
