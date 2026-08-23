#pragma once

#include <ATen/Tensor.h>

// Dense EXL3 CPU reconstruction, fused with both-side Hadamard transforms: emits
// W = diag(suh) . H128 . W_hat . H128 . diag(svh) (per 128x128 tile, 1/sqrt(128) per side),
// i.e. the ORIGINAL-basis weights, matching quant/reconstruct.cu's reconstruct_had_slice.
//
// CPU tensors only; the caller keeps the compressed trellis/suh/svh storage alive. This is the
// dense vision-tower execution path -- unlike cpu/moe_mul1.cpp it performs exact dequantization
// (no int8 activation approximation) and supports all codebooks (default/3inst, mcg, mul1).

void reconstruct_had_slice_cpu
(
    at::Tensor unpacked,    // [k, requested_n] fp16, CPU, contiguous
    at::Tensor packed,      // [k/16, n_blocks, K*16] int16, CPU, contiguous
    at::Tensor suh,         // [k] fp16
    at::Tensor svh,         // [requested_n] fp16, pre-offset by caller for slices
    int64_t K,
    bool mcg,
    bool mul1,
    int64_t n_offset        // column offset into the full matrix, multiple of 128
);
