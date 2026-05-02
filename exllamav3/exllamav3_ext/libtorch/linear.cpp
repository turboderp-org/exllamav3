#include <Python.h>
#include "linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../add.cuh"

void BC_LinearFP16::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (x.dtype() == y.dtype() && !graph)
        at::matmul_out(weight, x, y);
    else
        hgemm_gr(x, weight, y, graph);

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearFP16::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

//void BC_LinearFP16::run_cublas(const at::Tensor& x, at::Tensor& y)
//{
//    hgemm(x, weight, y);
//    if (bias)
//        y.add_(bias.value());
//}

void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (x.numel() == x.size(-1))
    {
        exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0, graph);
    }
    else
    {
        TORCH_CHECK(!graph, "BC_LinearEXL3 invoked with graph and bsz > 1");
        at::Tensor xh_ = at::empty_like(x);
        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);
    }

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

at::Tensor BC_LinearEXL3::run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32)
{
    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() = out_features;

    at::Tensor y = at::empty(
        out_shape,
        x.options().dtype(output_fp32 ? at::kFloat : at::kHalf)
    );

    at::Tensor x_flat = x.view({-1, x.size(-1)});
    at::Tensor y_flat = y.view({-1, out_features});
    run(x_flat, y_flat);
    return y;
}
