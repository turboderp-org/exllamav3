#pragma once

#include <ATen/Tensor.h>

// Fixed-width decode bundles with a BF16 serving boundary. direct_output
// selects the fused accumulator-to-BF16 epilogue; final_group_barrier controls
// only the terminal barrier and may be false only when all matrices reside.
// force_num_sms == 0 shares the available cooperative grid across matrices.
int exl3_mgemm_bf16_io
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    int bits,
    int force_num_sms,
    int output_stride,
    bool mcg,
    bool direct_output,
    bool final_group_barrier
);

// As above, but unique_suh/had_group_ids share transformed inputs among
// matrices that use the same input scale/flip vector. had_group_ids is static
// launch metadata supplied as a contiguous CPU int32 tensor, so its values can
// be validated before capture and copied into CUDA Graph kernel parameters.
int exl3_mgemm_bf16_io_grouped_had
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& unique_suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& had_group_ids,
    int bits,
    int force_num_sms,
    int output_stride,
    bool mcg,
    bool direct_output,
    bool final_group_barrier
);
