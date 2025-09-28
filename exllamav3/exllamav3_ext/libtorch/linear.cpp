#include <Python.h>
#include "linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"

void BC_LinearFP16::run(const at::Tensor& x, at::Tensor& y)
{
    if (x.dtype() == y.dtype())
        at::matmul_out(weight, x, y);
    else
        hgemm(x, weight, y);

    if (bias)
        y.add_(bias.value());
}


void BC_LinearFP16::run_cublas(const at::Tensor& x, at::Tensor& y)
{
    hgemm(x, weight, y);
    if (bias)
        y.add_(bias.value());
}


void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    if (x.numel() == x.size(-1))
    {
        exl3_gemm(x, trellis, y, suh, xh, svh, -1, mcg_mult, mul1_mult);
    }
    else
    {
        at::Tensor xh_ = at::empty_like(x);
        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg_mult, mul1_mult);
    }

    if (bias) y.add_(bias.value());
}

