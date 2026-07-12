#include <Python.h>
#include "gated_rmsnorm.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
//#include <torch/extension.h>
#include "../util.h"
#include "../norm.cuh"

void BC_GatedRMSNorm::run(const at::Tensor& x, at::Tensor& y, const at::Tensor& gate)
{
    gated_rms_norm(x, weight, y, gate, rms_norm_eps, constant_bias, w_groups, gate_first);
}

void BC_GatedRMSNorm::run_gr(const at::Tensor& x, at::Tensor& y, const at::Tensor& gate, Graph* graph)
{
    gated_rms_norm_gr(x, weight, y, gate, rms_norm_eps, constant_bias, graph, w_groups, gate_first);
}