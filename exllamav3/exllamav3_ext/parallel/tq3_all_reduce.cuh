#pragma once

#include <ATen/Tensor.h>

// TQ3-compressed all-reduce over the native parallel-group shared memory fabric.
//
// Uses an all-gather + local-reduce pattern:
//   1. Each rank TQ3-compresses its fp16 tensor (6.4× smaller) and writes the
//      result into its dedicated slot inside the pinned shared buffer.
//   2. A cross-GPU barrier ensures every rank has finished writing.
//   3. Each rank decompresses all slots and accumulates them locally (in-place).
//
// Parameters
//   data         — fp16 tensor on the calling GPU (modified in-place)
//   ctx_ptr      — uintptr_t of the process-group's pinned PGContext block
//   devices      — ordered list of participating GPU indices (same as pg_all_reduce)
//   this_device  — GPU index of the calling process
//   master_device— coordinator GPU index (lowest rank, used by barrier_inner)
//   shbuf        — uintptr_t of the pinned shared ring buffer (shm_b)
//   shbuf_size   — total size of the shared buffer in bytes
//   contribution — if false this rank is a non-contributing observer: it writes
//                  all-zeros into its slot but still participates in barriers
void tq3_all_reduce
(
    const at::Tensor&        data,
    uintptr_t                ctx_ptr,
    std::vector<uintptr_t>   devices,
    int                      this_device,
    int                      master_device,
    uintptr_t                shbuf,
    size_t                   shbuf_size,
    bool                     contribution
);
