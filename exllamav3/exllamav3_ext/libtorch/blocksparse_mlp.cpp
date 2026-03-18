#include <Python.h>
#include "blocksparse_mlp.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../quant/hadamard.cuh"
#include "../quant/reconstruct.cuh"
#include "../quant/exl3_devctx.cuh"
#include "../activation.cuh"
#include "../add.cuh"

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

void BC_BlockSparseMLP::run_bsz1_gr
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights,
    Graph* graph
)
{
    //py::gil_scoped_release _;
    const at::Tensor& yi = y.unsqueeze(0);

    exl3_mgemm_gr
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
        gate_mcg,
        gate_mul1,
        min_expert,
        max_expert,
        0,
        graph
    );

    exl3_mgemm_gr
    (
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
        up_mcg,
        up_mul1,
        min_expert,
        max_expert,
        0,
        graph
    );

    if (act_silu)
        silu_mul_gr(interm_g, interm_u, interm_a, act_limit, graph);
    else if (act_gelu)
        gelu_mul_gr(interm_g, interm_u, interm_a, act_limit, graph);

    exl3_mgemm_gr
    (
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
        down_mcg,
        down_mul1,
        min_expert,
        max_expert,
        0,
        graph
    );

    if (shared_experts)
    {
        shared_experts->run_bsz1_gr(yi, out_d_sh.value(), graph);
        if (shared_gate)
        {
            add_sigmoid_gate_proj_gr(out_d_sh.value(), yi, out_d, shared_gate->weight, graph);
        }
        else
        {
            add_gr(out_d, out_d_sh.value(), out_d, graph);
        }
    }
}

void BC_BlockSparseMLP::run_bsz1
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
)
{
    c10::cuda::CUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    #define USE_GRAPH
    #ifndef USE_GRAPH

        run_bsz1_gr(y, selected_experts, routing_weights, nullptr);

    #else

        if (!graph_bsz1.ready)
        {
            graph_bsz1.capture_begin();
            run_bsz1_gr(y, selected_experts, routing_weights, &graph_bsz1);
            graph_bsz1.capture_end();
        }

        auto args = std::vector<PPTR>
        {
            PPTR(GP_mgemm_A,            (void*) y.data_ptr()),
            PPTR(GP_mgemm_indices,      (void*) selected_experts.data_ptr()),
            PPTR(GP_end,                nullptr),
            PPTR(GP_mgemm_A,            (void*) y.data_ptr()),
            PPTR(GP_mgemm_indices,      (void*) selected_experts.data_ptr()),
            PPTR(GP_end,                nullptr),
        };

        if (shared_experts && shared_gate)
        {
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
            args.push_back(PPTR(GP_end,                         nullptr));
            args.push_back(PPTR(GP_mgemm_A,                     (void*) y.data_ptr()));
            args.push_back(PPTR(GP_add_sigmoid_gate_proj_y,     (void*) y.data_ptr()));
            args.push_back(PPTR(GP_add_sigmoid_gate_proj_z,     (void*) out_d.data_ptr()));
        }
        else if (shared_experts)
        {
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
            args.push_back(PPTR(GP_end,                         nullptr));
            args.push_back(PPTR(GP_mgemm_A,                     (void*) y.data_ptr()));
            args.push_back(PPTR(GP_add_x,                       (void*) out_d.data_ptr()));
            args.push_back(PPTR(GP_add_z,                       (void*) out_d.data_ptr()));
        }
        else
        {
            args.push_back(PPTR(GP_mgemm_C,                     (void*) out_d.data_ptr()));
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
        }

        graph_bsz1.launch(args, stream);

    #endif
    #undef USE_GRAPH
}

BC_BlockSparseMLP::BC_BlockSparseMLP
(
    at::Tensor _yh2,
    at::Tensor _yh,
    at::Tensor _interm_gu,
    at::Tensor _interm_g,
    at::Tensor _interm_u,
    at::Tensor _interm_a,
    at::Tensor _interm_a2,
    at::Tensor _out_d,
    at::Tensor _out_d2,
    c10::optional<at::Tensor> _out_d_sh,
    c10::optional<at::Tensor> _z,
    at::Tensor _dq_temp_up,
    at::Tensor _dq_temp_down,
    int _min_expert,
    int _max_expert,
    at::Tensor _gate_ptrs_trellis,
    at::Tensor _gate_ptrs_suh,
    at::Tensor _gate_ptrs_svh,
    int _gate_K,
    bool _gate_mcg,
    bool _gate_mul1,
    at::Tensor _up_ptrs_trellis,
    at::Tensor _up_ptrs_suh,
    at::Tensor _up_ptrs_svh,
    int _up_K,
    bool _up_mcg,
    bool _up_mul1,
    at::Tensor _down_ptrs_trellis,
    at::Tensor _down_ptrs_suh,
    at::Tensor _down_ptrs_svh,
    int _down_K,
    bool _down_mcg,
    bool _down_mul1,
    bool _act_silu,
    bool _act_gelu,
    std::shared_ptr<BC_GatedMLP> _shared_experts,
    std::shared_ptr<BC_LinearFP16> _shared_gate,
    float _act_limit,
    std::vector<std::shared_ptr<BC_LinearEXL3>> _gates,
    std::vector<std::shared_ptr<BC_LinearEXL3>> _ups,
    std::vector<std::shared_ptr<BC_LinearEXL3>> _downs,
    at::Tensor _gu_trellis_ptr,
    at::Tensor _gu_suh_ptr,
    at::Tensor _gu_svh_ptr
) :
        yh2                 (std::move(_yh2)),
        yh                  (std::move(_yh)),
        interm_gu           (std::move(_interm_gu)),
        interm_g            (std::move(_interm_g)),
        interm_u            (std::move(_interm_u)),
        interm_a            (std::move(_interm_a)),
        interm_a2           (std::move(_interm_a2)),
        out_d               (std::move(_out_d)),
        out_d2              (std::move(_out_d2)),
        out_d_sh            (std::move(_out_d_sh)),
        z                   (std::move(_z)),
        dq_temp_up          (std::move(_dq_temp_up)),
        dq_temp_down        (std::move(_dq_temp_down)),
        min_expert          (_min_expert),
        max_expert          (_max_expert),
        gate_ptrs_trellis   (std::move(_gate_ptrs_trellis)),
        gate_ptrs_suh       (std::move(_gate_ptrs_suh)),
        gate_ptrs_svh       (std::move(_gate_ptrs_svh)),
        gate_K              (_gate_K),
        gate_mcg            (_gate_mcg),
        gate_mul1           (_gate_mul1),
        up_ptrs_trellis     (std::move(_up_ptrs_trellis)),
        up_ptrs_suh         (std::move(_up_ptrs_suh)),
        up_ptrs_svh         (std::move(_up_ptrs_svh)),
        up_K                (_up_K),
        up_mcg              (_up_mcg),
        up_mul1             (_up_mul1),
        down_ptrs_trellis   (std::move(_down_ptrs_trellis)),
        down_ptrs_suh       (std::move(_down_ptrs_suh)),
        down_ptrs_svh       (std::move(_down_ptrs_svh)),
        down_K              (_down_K),
        down_mcg            (_down_mcg),
        down_mul1           (_down_mul1),
        act_silu            (_act_silu),
        act_gelu            (_act_gelu),
        shared_experts      (_shared_experts),
        shared_gate         (_shared_gate),
        act_limit           (_act_limit),
        gates               (_gates),
        ups                 (_ups),
        downs               (_downs),
        gu_trellis_ptr      (_gu_trellis_ptr),
        gu_suh_ptr          (_gu_suh_ptr),
        gu_svh_ptr          (_gu_svh_ptr)
{
    gate_ptrs_trellis_cpu   = gate_ptrs_trellis.cpu();
    gate_ptrs_suh_cpu       = gate_ptrs_suh.cpu();
    gate_ptrs_svh_cpu       = gate_ptrs_svh.cpu();
    up_ptrs_trellis_cpu     = up_ptrs_trellis.cpu();
    up_ptrs_suh_cpu         = up_ptrs_suh.cpu();
    up_ptrs_svh_cpu         = up_ptrs_svh.cpu();
    down_ptrs_trellis_cpu   = down_ptrs_trellis.cpu();
    down_ptrs_suh_cpu       = down_ptrs_suh.cpu();
    down_ptrs_svh_cpu       = down_ptrs_svh.cpu();

    max_experts_per_token = interm_g.size(0);
    max_tokens_per_expert = max_experts_per_token;

    for (int i = 0; i < max_tokens_per_expert; ++i)
    {
        interm_g_single.push_back(interm_g.squeeze(1).slice(0, 0, i + 1));
        interm_u_single.push_back(interm_u.squeeze(1).slice(0, 0, i + 1));
        interm_a_single.push_back(interm_a.squeeze(1).slice(0, 0, i + 1));
        out_d_single.push_back(out_d.squeeze(1).slice(0, 0, i + 1));
    }

    TORCH_CHECK(max_expert <= MAX_EXPERTS, "BC_BlockSparseMLP: Too many experts");

    use_mgemm = gate_K == up_K;
}

void BC_BlockSparseMLP::run_single_expert_gr
(
    const at::Tensor& y,
    const int expert_idx,
    Graph* graph
)
{
    int bsz = y.size(0);

    at::Tensor ai = interm_a2.slice(0, 0, bsz);
    at::Tensor oi = out_d2.slice(0, 0, bsz);

//    if (use_mgemm)
//    {
//        at::Tensor yb = y.unsqueeze(0);
//        at::Tensor gui = interm_gu.slice(0, 0, bsz * 2).view({2, bsz, interm_gu.size(1)});
//        at::Tensor yh2i = yh2.slice(0, 0, 2).view({2, 1, yh2.size(1)});
//
//        exl3_mgemm
//        (
//            yb,
//            gu_trellis_ptr[expert_idx],
//            gui,
//            gu_suh_ptr[expert_idx],
//            yh2,
//            gu_svh_ptr[expert_idx],
//            c10::nullopt,
//            c10::nullopt,
//            gate_K,
//            -1,
//            gate_mcg,
//            gate_mul1,
//            -1,
//            -1,
//            0
//        );
//
//        at::Tensor gi = gui[0];
//        at::Tensor ui = gui[1];
//
//        if (act_silu)
//            silu_mul(gi, ui, ai, act_limit);
//        else if (act_gelu)
//            gelu_mul(gi, ui, ai, act_limit);
//    }
//    else
    {
        at::Tensor gi = interm_gu.slice(0, 0, bsz);
        at::Tensor ui = interm_gu.slice(0, bsz, bsz * 2);

        exl3_gemm_gr
        (
            y,
            gates[expert_idx]->trellis,
            gi,
            gates[expert_idx]->suh,
            yh,
            gates[expert_idx]->svh,
            -1,
            gate_mcg,
            gate_mul1,
            0,
            graph
        );

        exl3_gemm_gr
        (
            y,
            ups[expert_idx]->trellis,
            ui,
            ups[expert_idx]->suh,
            yh,
            ups[expert_idx]->svh,
            -1,
            up_mcg,
            up_mul1,
            0,
            graph
        );

        if (act_silu)
            silu_mul_gr(gi, ui, ai, act_limit, graph);
        else if (act_gelu)
            gelu_mul_gr(gi, ui, ai, act_limit, graph);
    }

    exl3_gemm_gr
    (
        ai,
        downs[expert_idx]->trellis,
        oi,
        downs[expert_idx]->suh,
        ai,
        downs[expert_idx]->svh,
        -1,
        down_mcg,
        down_mul1,
        0,
        graph
    );
}

void BC_BlockSparseMLP::run_single_expert
(
    const at::Tensor& y,
    const int expert_idx
)
{
    int bsz = y.size(0);
    TORCH_CHECK(bsz <= TEMP_ROWS_GRAPH);
    int graphidx = bsz - 1;

    c10::cuda::CUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    #define USE_GRAPH
    #ifndef USE_GRAPH

        run_single_expert_gr(y, expert_idx, nullptr);

    #else

        if (!graph_single[graphidx].ready)
        {
            prepare_ctx(y.get_device());

            graph_single[graphidx].capture_begin();
            run_single_expert_gr(y, expert_idx, &graph_single[graphidx]);
            graph_single[graphidx].capture_end();
        }

        auto args = std::vector<PPTR>
        {
            PPTR(GP_gemm_A,             (void*) y.data_ptr()),
            PPTR(GP_gemm_B_trellis,     (void*) gates[expert_idx]->trellis.data_ptr()),
            PPTR(GP_gemm_B_suh,         (void*) gates[expert_idx]->suh.data_ptr()),
            PPTR(GP_gemm_B_svh,         (void*) gates[expert_idx]->svh.data_ptr()),
            PPTR(GP_end,                nullptr),
            PPTR(GP_gemm_A,             (void*) y.data_ptr()),
            PPTR(GP_gemm_B_trellis,     (void*) ups[expert_idx]->trellis.data_ptr()),
            PPTR(GP_gemm_B_suh,         (void*) ups[expert_idx]->suh.data_ptr()),
            PPTR(GP_gemm_B_svh,         (void*) ups[expert_idx]->svh.data_ptr()),
            PPTR(GP_end,                nullptr),
            PPTR(GP_gemm_B_trellis,     (void*) downs[expert_idx]->trellis.data_ptr()),
            PPTR(GP_gemm_B_suh,         (void*) downs[expert_idx]->suh.data_ptr()),
            PPTR(GP_gemm_B_svh,         (void*) downs[expert_idx]->svh.data_ptr()),
        };

        graph_single[graphidx].launch(args, stream);

    #endif
    #undef USE_GRAPH
}


void BC_BlockSparseMLP::run_single_expert_dq
(
    const at::Tensor& y,
    const int expert_idx,
    at::Tensor& yh,
    at::Tensor& interm,
    at::Tensor& interm_a,
    at::Tensor& out
)
{
    int bsz = y.size(0);

    at::Tensor yh1 = yh.slice(0, 0, bsz);
    at::Tensor yh2 = yh.slice(0, bsz, bsz * 2);
    at::Tensor interm1 = interm.slice(0, 0, bsz);
    at::Tensor interm2 = interm.slice(0, bsz, bsz * 2);

    had_r_128_dual(y, yh1, gates[expert_idx]->suh, c10::nullopt,
                   y, yh2, ups[expert_idx]->suh, c10::nullopt, 1.0);

    reconstruct(dq_temp_up, gates[expert_idx]->trellis, gate_K, gate_mcg, gate_mul1);
    hgemm(yh1, dq_temp_up, interm1);
    reconstruct(dq_temp_up, ups[expert_idx]->trellis, up_K, up_mcg, up_mul1);
    hgemm(yh2, dq_temp_up, interm2);

    had_r_128_dual(interm1, interm1, c10::nullopt, gates[expert_idx]->svh,
                   interm2, interm2, c10::nullopt, ups[expert_idx]->svh, 1.0);

    if (act_silu)
        silu_mul(interm1, interm2, interm_a, act_limit);
    else if (act_gelu)
        gelu_mul(interm1, interm2, interm_a, act_limit);

    had_r_128(interm_a, interm_a, downs[expert_idx]->suh, c10::nullopt, 1.0);
    reconstruct(dq_temp_down, downs[expert_idx]->trellis, down_K, down_mcg, down_mul1);
    hgemm(interm_a, dq_temp_down, out);
    had_r_128(out, out, c10::nullopt, downs[expert_idx]->svh, 1.0);
}
