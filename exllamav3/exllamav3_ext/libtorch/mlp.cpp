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

void BC_GatedMLP::run_bsz1_gr
(
    const at::Tensor& x,
    at::Tensor& d,
    Graph* graph
)
{
    exl3_mgemm_gr
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
        gu_mcg,
        gu_mul1,
        -1,
        -1,
        0,
        graph
    );

    at::Tensor g = gu.select(0, 0).unsqueeze(0);
    at::Tensor u = gu.select(0, 1).unsqueeze(0);

    if (act_silu)
        silu_mul_gr(g, u, a, graph);
    else if (act_gelu)
        gelu_mul_gr(g, u, a, graph);
    else if (act_relu2)
        relu2_mul_gr(g, u, a, graph);

    down->run_gr(a, d, graph);
}

void BC_GatedMLP::run_bsz1
(
    const at::Tensor& x,
    at::Tensor& d
)
{
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    #define USE_GRAPH
    #ifndef USE_GRAPH

        run_bsz1_gr(x, d, nullptr);

    #else

        if (!graph_bsz1.ready)
        {
            graph_bsz1.capture_begin();
            run_bsz1_gr(x, d, &graph_bsz1);
            graph_bsz1.capture_end();
        }

        auto args = std::vector<PPTR>
        {
            PPTR(GP_mgemm_A,            (void*) x.data_ptr()),
            PPTR(GP_gemm_C,             (void*) d.data_ptr())
        };

        graph_bsz1.launch(args, stream);

    #endif
}