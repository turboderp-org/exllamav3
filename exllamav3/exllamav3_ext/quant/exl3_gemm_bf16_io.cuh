#pragma once

#include <ATen/Tensor.h>

// Decode-only K5/K6 MCG path. Input and output stay BF16 at the public
// boundary while the existing EXL3 Hadamard/GEMM arithmetic remains FP16.
// final_grid_sync controls only the redundant terminal cooperative barrier.
// force_num_sms == 0 derives a valid cooperative grid from the current device.
int exl3_gemm_bf16_io
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    int force_shape_idx,
    int force_num_sms,
    bool mcg,
    bool final_grid_sync
);
