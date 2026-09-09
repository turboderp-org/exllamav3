#pragma once

#include <ATen/Tensor.h>

void moe_unswizzle_trellis
(
    const at::Tensor& src,      // staged batch (int16 flat), band-swizzled tiles
    const at::Tensor& dst,      // same size, receives native (k/16, n/16, 16K) tile order
    int64_t num_experts,
    int64_t expert_stride_b,    // bytes per expert in the batch
    int64_t proj_off_b,         // byte offset of this projection within an expert
    int64_t tiles_k,
    int64_t tiles_n,
    int64_t bits,
    bool swizzled               // false: plain copy (e.g. K8 matrices are never swizzled)
);
