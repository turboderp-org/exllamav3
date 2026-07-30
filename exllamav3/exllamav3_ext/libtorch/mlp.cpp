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

void BC_GatedMLP::run_bszN_gr
(
    const at::Tensor& x,
    at::Tensor& d,
    int num_tokens,
    Graph* graph
)
{
    // guh/gu hold 2 slots (gate, up); slicing the static (2, MAX_BSZN, width) buffers along dim 1
    // would leave dim 0's stride at MAX_BSZN*width instead of num_tokens*width, corrupting the
    // fused mgemm kernel's raw j*size_m*size_k slot addressing for any num_tokens != MAX_BSZN --
    // use exactly-shaped, lazily-cached buffers instead (a/down_xh have no such issue: their slot
    // dim is size 1, so slicing the token dim -- the true leading dim there -- stays contiguous)
    at::Tensor& guh_n_ref = guh_cache[num_tokens - 1];
    if (!guh_n_ref.defined())
        guh_n_ref = at::empty({2, num_tokens, guh.size(2)}, guh.options());
    at::Tensor& gu_n_ref = gu_cache[num_tokens - 1];
    if (!gu_n_ref.defined())
        gu_n_ref = at::empty({2, num_tokens, gu.size(2)}, gu.options());
    at::Tensor guh_n     = guh_n_ref;
    at::Tensor gu_n      = gu_n_ref;
    at::Tensor a_n       = a.slice(1, 0, num_tokens);
    at::Tensor down_xh_n = down_xh.slice(1, 0, num_tokens);

    if (gu_ptrs_trellis)
    {
        exl3_mgemm_gr
        (
            x,
            gu_ptrs_trellis.value(),
            gu_n,
            gu_ptrs_suh.value(),
            guh_n,
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
            graph,
            1   // mgemm's reduction-group param, unused here (no weights -> no reduction runs);
                // bszm=2 is the gate/up slot pair, unrelated to num_tokens (carried via size_m)
        );
    }
    else
    {
        // num_tokens > 1 bypasses BC_LinearEXL3::run_gr (which hard-refuses graphing above bsz 1
        // and whose own xh scratch is a single global cache entry shared by every quantized
        // Linear layer of the same shape) -- call exl3_gemm_gr directly with this class's own
        // scratch instead, for every num_tokens including 1
        at::Tensor g2 = gu_n.select(0, 0);
        at::Tensor u2 = gu_n.select(0, 1);
        at::Tensor gate_xh = guh_n.select(0, 0);
        at::Tensor up_xh   = guh_n.select(0, 1);
        exl3_gemm_gr(x, gate->trellis, g2, gate->suh, gate_xh, gate->svh, -1, gate->mcg, gate->mul1, 0, graph);
        exl3_gemm_gr(x, up->trellis, u2, up->suh, up_xh, up->svh, -1, up->mcg, up->mul1, 0, graph);
        if (gate->bias) add_gr(g2, gate->bias.value(), g2, graph);
        if (up->bias) add_gr(u2, up->bias.value(), u2, graph);
    }

    at::Tensor g = gu_n.select(0, 0).unsqueeze(0);
    at::Tensor u = gu_n.select(0, 1).unsqueeze(0);

    if (act_silu)
        silu_mul_gr(g, u, a_n, act_limit, graph);
    else if (act_gelu)
        gelu_mul_gr(g, u, a_n, act_limit, graph);
    else if (act_relu2)
        relu2_mul_gr(g, u, a_n, act_limit, graph);

    exl3_gemm_gr(a_n, down->trellis, d, down->suh, down_xh_n, down->svh, -1, down->mcg, down->mul1, 0, graph);
    if (down->bias)
        add_gr(d, down->bias.value(), d, graph);
}

void BC_GatedMLP::run_bszN
(
    const at::Tensor& x,
    at::Tensor& d
)
{
    int num_tokens = (int) x.numel() / (int) x.size(-1);
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= MAX_BSZN, "run_bszN: bsz out of supported range");
    int graphidx = num_tokens - 1;

    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    Graph& g = graph_bszN[graphidx];

    if (g.disabled || (!g.ready && !g.ready_to_record))
    {
        run_bszN_gr(x, d, num_tokens, nullptr);
        g.ready_to_record = true;
    }
    else
    {
        if (!g.ready)
        {
            g.capture_begin();
            run_bszN_gr(x, d, num_tokens, &g);
            g.capture_end();
        }

        std::vector<PPTR> args;
        if (gu_ptrs_trellis)
        {
            args.emplace_back(GP_mgemm_A, (void*) x.data_ptr());
        }
        else
        {
            at::Tensor gu_n = gu_cache[num_tokens - 1];
            // The gate/up GEMMs record their own GP_gemm_C sites ahead of the down projection's;
            // patch them with their (static) values so the site walk stays aligned and the final
            // GP_gemm_C entry binds to the down projection
            args.emplace_back(GP_gemm_A, (void*) x.data_ptr());
            args.emplace_back(GP_gemm_C, (void*) gu_n.select(0, 0).data_ptr());
            args.emplace_back(GP_gemm_A, (void*) x.data_ptr());
            args.emplace_back(GP_gemm_C, (void*) gu_n.select(0, 1).data_ptr());
        }
        args.emplace_back(GP_gemm_C, (void*) d.data_ptr());
        if (down->bias)
        {
            args.emplace_back(GP_add_x, (void*) d.data_ptr());
            args.emplace_back(GP_add_z, (void*) d.data_ptr());
        }

        g.launch(args, stream);
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