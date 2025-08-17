#pragma once

#include <ATen/Tensor.h>

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult
);

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    c10::optional<at::Tensor>& indices,
    c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index
);
