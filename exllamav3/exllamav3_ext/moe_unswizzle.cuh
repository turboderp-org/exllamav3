#pragma once

#include <ATen/Tensor.h>
#include <vector>

void moe_unswizzle_trellis
(
    const at::Tensor& src,      // staged batch (int16 flat), band-swizzled tiles
    const at::Tensor& dst,      // same size, receives native (k/16, n/16, 16K) tile order
    int64_t num_experts,
    int64_t expert_stride_b,    // bytes per expert in the batch
    int64_t proj_off_b,         // byte offset of this projection within an expert
    int64_t tiles_k,
    int64_t tiles_n,
    double K,
    int64_t group = 0,            // packed tile group of these bytes (0/2/8, exl3_moe_cpu_swizzle_group);
                                  // 0: plain copy (e.g. never-swizzled projections)
    int64_t planar = 0            // 1: invert planar dword/halfword order (exl3_moe_cpu_planar_layout)
);

// Multi-NUMA N-split: the batch holds each worker's packed slice of every expert (worker w's
// slice of expert e at w_off[w] + e * w_stride[w] + w_proj[w]; slices cut the projection at the
// tile boundaries in split, along n, or along k with sliced_k); writes the full native projection
// into dst like moe_unswizzle_trellis does for one worker's bytes. group 0 is a plain gather
void moe_unswizzle_trellis_nsplit
(
    const at::Tensor& src,
    const at::Tensor& dst,
    int64_t num_experts,
    int64_t expert_stride_b,    // dst: bytes per expert
    int64_t proj_off_b,         // dst: byte offset of this projection within an expert
    int64_t tiles_k,
    int64_t tiles_n,
    double K,
    int64_t group,
    int64_t planar,
    bool sliced_k,
    const std::vector<int64_t>& split,
    const std::vector<int64_t>& w_off,
    const std::vector<int64_t>& w_stride,
    const std::vector<int64_t>& w_proj
);
