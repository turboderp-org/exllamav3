#pragma once

#include <torch/extension.h>

at::Tensor p2b_fused_moe_cuda(const at::Tensor& x, at::Tensor& out,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rows, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg, bool mul1,
    int64_t hidden, int64_t inter);

#include <vector>
std::vector<at::Tensor> p2b_stage_debug_cuda(const at::Tensor& x,
    const at::Tensor& gt, const at::Tensor& gu, const at::Tensor& gv,
    const at::Tensor& ut, const at::Tensor& uu, const at::Tensor& uv,
    const at::Tensor& dt, const at::Tensor& du, const at::Tensor& dv,
    const at::Tensor& ids, const at::Tensor& rows, const at::Tensor& rw,
    int64_t kg, int64_t ku, int64_t kd, bool mcg, bool mul1,
    int64_t hidden, int64_t inter, int64_t stop_after);
