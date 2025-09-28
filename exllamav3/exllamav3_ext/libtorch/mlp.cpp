#include <Python.h>
#include "mlp.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../activation.cuh"

using namespace torch::indexing;

void BC_GatedMLP::run_bsz1
(
    const at::Tensor& x,
    at::Tensor& d
)
{
    exl3_mgemm
    (
        x,
        gu_ptrs_trellis,
        gu,
        gu_ptrs_suh,
        guh,
        gu_ptrs_svh,
        {},
        {},
        gu_K,
        -1,
        gu_mcg_mult,
        gu_mul1_mult,
        -1,
        -1
    );

    at::Tensor g = gu.select(0, 0).unsqueeze(0);
    at::Tensor u = gu.select(0, 1).unsqueeze(0);

    if (act_silu)
        silu_mul(g, u, a);
    else if (act_gelu)
        gelu_mul(g, u, a);
    else if (act_relu2)
        relu2_mul(g, u, a);

    down->run(a, d);
}