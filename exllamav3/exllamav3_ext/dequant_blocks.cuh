#pragma once

#include <ATen/Tensor.h>

// Natively quantized checkpoint tensors to fp16, written straight into LinearFP16's (in, out) layout:
// fp8 e4m3 / e5m2 bytes or packed e2m1 nibbles (two per byte, low nibble first), scaled per block of a
// scale grid (E8M0 exponent bytes, 2^(b - 127), or float32 multipliers), with an optional global
// multiplier applied after the fp16 rounding the way the eager loaders do

void dequant_blocks
(
    const at::Tensor& packed,                   // (rows, bytes per row) uint8; rows = out features unless src_transposed
    const c10::optional<at::Tensor>& scale,     // (sr, sc) uint8 (E8M0) or float32, or none
    at::Tensor out,                             // (in_pad, out_pad) half, transposed
    int64_t fmt,                                // 0 = e4m3, 1 = e2m1 nibbles, 2 = e5m2
    double global_scale,                        // multiplier applied to the rounded fp16 value (1.0 = none)
    int64_t col0,                               // output column (out feature) offset of this segment
    bool src_transposed,                        // packed is (in, out) bytes (fp8 only)
    int64_t block_rows,                         // out features per scale row (0: rows / scale rows, exact)
    int64_t block_cols                          // in features per scale column (0: cols / scale cols, exact)
);
