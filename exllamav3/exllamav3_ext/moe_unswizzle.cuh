#pragma once

#include <ATen/Tensor.h>
#include <array>

void moe_unswizzle_trellis
(
    const at::Tensor& src,      // staged batch (int16 flat), source layout given by group
    const at::Tensor& dst,      // same size, receives native (k/16, n/16, 16K) tile order
    int64_t num_experts,
    int64_t expert_stride_b,    // bytes per expert in the batch
    int64_t proj_off_b,         // byte offset of this projection within an expert
    int64_t tiles_k,
    int64_t tiles_n,
    double K,
    int64_t group              // source layout: native (0), paired (2), or eight-tile groups (8)
);

void moe_unswizzle_trellis_batch
(
    const at::Tensor& src,
    const at::Tensor& dst,
    int64_t num_experts,
    int64_t expert_stride_b,
    // Three records: offset_bytes, tiles_k, tiles_n, tile_bytes, group; tiles_k == 0 is inactive.
    const std::array<std::array<int64_t, 5>, 3>& projections
);
