#pragma once

#include <ATen/Tensor.h>

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& sv,
    int force_kernel_idx
);

int exl3_gemm_num_kernel_variants();