#include <Python.h>
#include "blocksparse_mlp.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../activation.cuh"

std::tuple<at::Tensor, at::Tensor> blocksparse_mlp_routing(
    int bsz,
    const py::object& cfg,
    const at::Tensor& y,
    const py::dict& params
)
{
    bool activate_all = false;
    if (params.contains("activate_all_experts"))
        activate_all = params["activate_all_experts"].cast<bool>();

    at::Tensor gate_tensor = cfg.attr("gate_tensor").cast<at::Tensor>();
    int64_t num_experts = cfg.attr("num_experts").cast<int64_t>();
    int64_t num_exp_per_tok = cfg.attr("num_experts_per_tok").cast<int64_t>();

    if (!activate_all && bsz == 1)
    {
        at::Tensor router_logits_bsz1 = cfg.attr("router_logits_bsz1").cast<at::Tensor>();
        at::Tensor routing_weights_bsz1 = cfg.attr("routing_weights_bsz1").cast<at::Tensor>();
        at::Tensor selected_experts_bsz1 = cfg.attr("selected_experts_bsz1").cast<at::Tensor>();

        at::matmul_out(router_logits_bsz1, y, gate_tensor);
        at::topk_out
        (
            routing_weights_bsz1,
            selected_experts_bsz1,
            router_logits_bsz1,
            num_exp_per_tok,
            -1,
            true,
            false
        );

        at::softmax_out(routing_weights_bsz1, routing_weights_bsz1, -1);
        return {selected_experts_bsz1, routing_weights_bsz1};
    }
    else
    {
        int64_t k = activate_all ? num_experts : num_exp_per_tok;

        at::Tensor router_logits = at::matmul(y, gate_tensor);

        auto topk_result = at::topk(router_logits, k, -1);
        at::Tensor routing_weights = std::get<0>(topk_result);
        at::Tensor selected_experts = std::get<1>(topk_result);

        routing_weights = at::softmax(routing_weights, -1);

        return {selected_experts, routing_weights};
    }
}

void BC_BlockSparseMLP::run_bsz1
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
)
{
    py::gil_scoped_release _;
    const at::Tensor& yi = y.unsqueeze(0);

    exl3_mgemm
    (
        yi,
        gate_ptrs_trellis,
        interm_g,
        gate_ptrs_suh,
        yh,
        gate_ptrs_svh,
        selected_experts,
        {},
        gate_K,
        -1,
        gate_mcg_mult,
        gate_mul1_mult,
        min_expert,
        max_expert
    );

    exl3_mgemm(
        yi,
        up_ptrs_trellis,
        interm_u,
        up_ptrs_suh,
        yh,
        up_ptrs_svh,
        selected_experts,
        {},
        up_K,
        -1,
        up_mcg_mult,
        up_mul1_mult,
        min_expert,
        max_expert
    );

    if (act_silu)
        silu_mul(interm_g, interm_u, interm_a);
    else if (act_gelu)
        gelu_mul(interm_g, interm_u, interm_a);

    exl3_mgemm(
        interm_a,
        down_ptrs_trellis,
        out_d,
        down_ptrs_suh,
        interm_a,
        down_ptrs_svh,
        selected_experts,
        routing_weights,
        down_K,
        -1,
        down_mcg_mult,
        down_mul1_mult,
        min_expert,
        max_expert
    );

    if (shared_experts)
    {
        shared_experts->run_bsz1(yi, out_d_sh.value());
        if (shared_gate)
        {
            shared_gate->run_cublas(yi, z.value());
            add_sigmoid_gate(out_d_sh.value(), z.value(), out_d);
        }
        else
        {
            out_d.add_(out_d_sh.value());
        }
    }

}