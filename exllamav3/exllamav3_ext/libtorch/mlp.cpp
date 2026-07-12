#include <Python.h>
#include "mlp.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../activation.cuh"
#include "../add.cuh"

using namespace torch::indexing;

void BC_GatedMLP::run_bsz1_gr
(
    const at::Tensor& x,
    at::Tensor& d,
    Graph* graph
)
{
    if (gu_ptrs_trellis)
    {
        exl3_mgemm_gr
        (
            x,
            gu_ptrs_trellis.value(),
            gu,
            gu_ptrs_suh.value(),
            guh,
            gu_ptrs_svh.value(),
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
    }
    else
    {
        at::Tensor g2 = gu.select(0, 0);
        at::Tensor u2 = gu.select(0, 1);
        gate->run_gr(x, g2, graph);
        up->run_gr(x, u2, graph);
    }

    at::Tensor g = gu.select(0, 0).unsqueeze(0);
    at::Tensor u = gu.select(0, 1).unsqueeze(0);

    if (act_silu)
        silu_mul_gr(g, u, a, act_limit, graph);
    else if (act_gelu)
        gelu_mul_gr(g, u, a, act_limit, graph);
    else if (act_relu2)
        relu2_mul_gr(g, u, a, act_limit, graph);

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

    if (graph_bsz1.disabled || (!graph_bsz1.ready && !graph_bsz1.ready_to_record))
    {
        run_bsz1_gr(x, d, nullptr);
        graph_bsz1.ready_to_record = true;
    }
    else
    {
        if (!graph_bsz1.ready)
        {
            graph_bsz1.capture_begin();
            run_bsz1_gr(x, d, &graph_bsz1);
            graph_bsz1.capture_end();
        }

        std::vector<PPTR> args;
        if (gu_ptrs_trellis)
        {
            args.emplace_back(GP_mgemm_A, (void*) x.data_ptr());
        }
        else
        {
            // The gate/up GEMMs record their own GP_gemm_C sites ahead of the down projection's;
            // patch them with their (static) values so the site walk stays aligned and the final
            // GP_gemm_C entry binds to the down projection
            args.emplace_back(GP_gemm_A, (void*) x.data_ptr());
            args.emplace_back(GP_gemm_C, (void*) gu.select(0, 0).data_ptr());
            args.emplace_back(GP_gemm_A, (void*) x.data_ptr());
            args.emplace_back(GP_gemm_C, (void*) gu.select(0, 1).data_ptr());
        }
        args.emplace_back(GP_gemm_C, (void*) d.data_ptr());
        if (down->bias)
        {
            args.emplace_back(GP_add_x, (void*) d.data_ptr());
            args.emplace_back(GP_add_z, (void*) d.data_ptr());
        }

        graph_bsz1.launch(args, stream);
    }
}

void BC_MLP::run_bsz1_gr
(
    const at::Tensor& x,
    at::Tensor& d,
    Graph* graph
)
{
    // Padded up K: stage x through the zero-padded static
    at::Tensor x_in = x;
    if (xp)
    {
        at::Tensor x2 = x.view({1, hidden_size});
        at::Tensor xp2 = xp.value();
        copy2d_gr(x2, xp2, graph);
        x_in = xp2.view({1, 1, -1});
    }

    up->run_gr(x_in, u, graph);

    if (act_silu)
        silu_mul_gr(u, ones, u, act_limit, graph);
    else if (act_gelu)
        gelu_mul_gr(u, ones, u, act_limit, graph);
    else if (act_relu2)
        relu2_mul_gr(u, ones, u, act_limit, graph);

    // Padded down N: the GEMM writes the padded static, the exact width copies out to d
    if (yp)
    {
        at::Tensor yp3 = yp.value().view({1, 1, -1});
        down->run_gr(u, yp3, graph);
        at::Tensor d2 = d.view({1, out_size});
        at::Tensor yp2 = yp.value();
        copy2d_gr(yp2, d2, graph);
    }
    else
        down->run_gr(u, d, graph);
}

void BC_MLP::run_bsz1
(
    const at::Tensor& x,
    at::Tensor& d
)
{
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (graph_bsz1.disabled || (!graph_bsz1.ready && !graph_bsz1.ready_to_record))
    {
        run_bsz1_gr(x, d, nullptr);
        graph_bsz1.ready_to_record = true;
    }
    else
    {
        if (!graph_bsz1.ready)
        {
            graph_bsz1.capture_begin();
            run_bsz1_gr(x, d, &graph_bsz1);
            graph_bsz1.capture_end();
        }

        std::vector<PPTR> args;
        // Intermediate same-type sites (the head staging copy's dst, the up projection's C) are
        // patched with their static values to keep the site walk aligned
        if (xp)
        {
            args.emplace_back(GP_copy2d_src, (void*) x.data_ptr());
            args.emplace_back(GP_copy2d_dst, (void*) xp.value().data_ptr());
        }
        else
            args.emplace_back(GP_gemm_A, (void*) x.data_ptr());
        args.emplace_back(GP_gemm_C, (void*) u.data_ptr());
        if (yp)
        {
            args.emplace_back(GP_copy2d_src, (void*) yp.value().data_ptr());
            args.emplace_back(GP_copy2d_dst, (void*) d.data_ptr());
        }
        else
        {
            args.emplace_back(GP_gemm_C, (void*) d.data_ptr());
            if (down->bias)
            {
                args.emplace_back(GP_add_x, (void*) d.data_ptr());
                args.emplace_back(GP_add_z, (void*) d.data_ptr());
            }
        }

        graph_bsz1.launch(args, stream);
    }
}