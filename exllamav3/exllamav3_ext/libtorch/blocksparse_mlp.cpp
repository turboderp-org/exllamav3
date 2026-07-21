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

void BC_BlockSparseMLP::run_bszN_gr
(
    const at::Tensor& A_in,
    const at::Tensor& x_dense,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights,
    int num_tokens,
    Graph* graph
)
{
    //py::gil_scoped_release _;

    int numex = (int) selected_experts.size(-1);
    int bszm = num_tokens * numex;

    // num_tokens == 1: original bsz=1 behavior -- zero-copy broadcast (the kernel's bszm_in==1
    // fast path), with the padded-hidden-dim staging captured as part of the graph. num_tokens >
    // 1: A_in is already the gathered (and padded, if applicable) [bszm, 1, Hi] input, built
    // eagerly outside (see run_bszN) since it depends on a real per-call gather, not just a view
    at::Tensor yi;
    if (num_tokens == 1)
    {
        if (y_pad)
        {
            // y_pad is sized [MAX_BSZN, Hi] to also serve the num_tokens > 1 gather staging in
            // run_bszN; only row 0 is used here
            at::Tensor yp = y_pad.value().slice(0, 0, 1);
            copy2d_gr(A_in, yp, graph);
            yi = yp.unsqueeze(0);
        }
        else
            yi = A_in.unsqueeze(0);
    }
    else
        yi = A_in;

    at::Tensor yh_n       = yh.slice(0, 0, bszm);
    at::Tensor interm_g_n = interm_g.slice(0, 0, bszm);
    at::Tensor interm_u_n = interm_u.slice(0, 0, bszm);
    at::Tensor interm_a_n = interm_a.slice(0, 0, bszm);
    at::Tensor out_d_n    = out_d.slice(0, 0, bszm);

    // exl3_mgemm's indices/weights arguments want a flat (1, bszm) view (num_tokens == 1: already
    // that shape, reshape is a no-op view); the bias-add kernels want the natural
    // (num_tokens, top_k) shape -- both view the same storage, so patched pointers stay identical
    at::Tensor sel_idx = selected_experts.reshape({1, -1});
    at::Tensor w_idx    = routing_weights.reshape({1, -1});

    if (gated)
    {
        exl3_mgemm_gr
        (
            yi,
            gate_ptrs_trellis,
            interm_g_n,
            gate_ptrs_suh,
            yh_n,
            gate_ptrs_svh,
            sel_idx,
            {},
            gate_K,
            -1,
            gate_mcg,
            gate_mul1,
            min_expert,
            max_expert,
            0,
            graph,
            num_tokens
        );
        if (gate_bias_ptrs)
            moe_bias_add_gr(interm_g_n, gate_bias_ptrs.value(), selected_experts, min_expert, max_expert, graph);
    }

    exl3_mgemm_gr
    (
        yi,
        up_ptrs_trellis,
        interm_u_n,
        up_ptrs_suh,
        yh_n,
        up_ptrs_svh,
        sel_idx,
        {},
        up_K,
        -1,
        up_mcg,
        up_mul1,
        min_expert,
        max_expert,
        0,
        graph,
        num_tokens
    );

    if (up_bias_ptrs)
        moe_bias_add_gr(interm_u_n, up_bias_ptrs.value(), selected_experts, min_expert, max_expert, graph);

    if (!gated)
        // relu(u) * u = relu^2(u), the non-gated activation
        relu_mul_gr(interm_u_n, interm_u_n, interm_a_n, act_limit, graph);
    else if (act_silu)
        silu_mul_gr(interm_g_n, interm_u_n, interm_a_n, act_limit, graph);
    else if (act_gelu)
        gelu_mul_gr(interm_g_n, interm_u_n, interm_a_n, act_limit, graph);
    else if (act_silu_oai)
        silu_oai_mul_gr(interm_g_n, interm_u_n, interm_a_n, act_limit, graph);
    else if (act_relu2)
        relu2_mul_gr(interm_g_n, interm_u_n, interm_a_n, act_limit, graph);

    // A_had must not alias A: the kernel stages the rotated input in A_had, and the autotuner
    // relaunches the (otherwise idempotent) kernel on the first call. interm_g_n is free here
    exl3_mgemm_gr
    (
        interm_a_n,
        down_ptrs_trellis,
        out_d_n,
        down_ptrs_suh,
        interm_g_n,
        down_ptrs_svh,
        sel_idx,
        w_idx,
        down_K,
        -1,
        down_mcg,
        down_mul1,
        min_expert,
        max_expert,
        0,
        graph,
        num_tokens
    );
    if (down_bias_ptrs)
        moe_bias_add_weighted_gr(out_d_n, down_bias_ptrs.value(), selected_experts, routing_weights, min_expert, max_expert, graph);
    if (out_trim)
    {
        // Exact-width copy out of the padded down result (rows 0..num_tokens-1 hold the weighted
        // reductions)
        at::Tensor src = out_d_n.slice(0, 0, num_tokens).squeeze(1);
        at::Tensor dst = out_trim.value().slice(0, 0, num_tokens);
        copy2d_gr(src, dst, graph);
    }

    if (shared_experts)
    {
        // x_dense is the natural (ungathered) [1, num_tokens, Hi] view -- distinct from yi/A_in,
        // which for num_tokens > 1 holds the per-slot GATHERED (duplicated) routed-expert input
        at::Tensor out_d_sh_n = out_d_sh.value().slice(1, 0, num_tokens);
        shared_experts->run_bszN_gr(x_dense, out_d_sh_n, num_tokens, graph);
        if (shared_gate)
        {
            add_sigmoid_gate_proj_gr(out_d_sh_n, x_dense, out_d_n, shared_gate->weight, graph);
        }
        else
        {
            add_gr(out_d_n, out_d_sh_n, out_d_n, graph);
        }
    }
}

void BC_BlockSparseMLP::run_bszN
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
)
{
    int num_tokens = (int) y.size(0);
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= MAX_BSZN, "run_bszN: bsz out of supported range");
    int graphidx = num_tokens - 1;
    int numex = (int) selected_experts.size(-1);

    c10::cuda::CUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // num_tokens > 1: build the gathered input eagerly (not graphed -- this is a real per-call
    // op, unlike the bsz=1 zero-copy broadcast). flat_token depends only on (num_tokens, numex),
    // never on routing outcome, so it's built once per num_tokens and cached. x_dense is the
    // natural (ungathered) [1, num_tokens, Hi] view, used only for the shared-experts merge (the
    // constructor already requires y_pad to be absent whenever shared_experts is present, so no
    // padded-dim handling is needed there)
    at::Tensor A_in;
    at::Tensor x_dense;
    if (num_tokens == 1)
    {
        A_in = y;
        x_dense = y.unsqueeze(0);
    }
    else
    {
        int bszm = num_tokens * numex;
        at::Tensor& flat_token = flat_token_cache[graphidx];
        if (!flat_token.defined())
        {
            flat_token = at::arange(num_tokens, at::TensorOptions().dtype(at::kLong).device(y.device()))
                .unsqueeze(1).expand({num_tokens, numex}).reshape({num_tokens * numex}).contiguous();
        }

        at::Tensor gather_src = y;
        if (y_pad)
        {
            at::Tensor yp_n = y_pad.value().slice(0, 0, num_tokens);
            yp_n.slice(1, 0, y.size(1)).copy_(y);
            gather_src = yp_n;
        }
        x_dense = gather_src.unsqueeze(0);

        at::Tensor gathered = gather_src.index_select(0, flat_token);
        at::Tensor ag_n = a_gather.slice(0, 0, bszm);
        ag_n.slice(1, 0, gathered.size(1)).copy_(gathered);
        A_in = ag_n.view({bszm, 1, ag_n.size(1)});
    }

    Graph& g = graph_bszN[graphidx];

    if (g.disabled || (!g.ready && !g.ready_to_record))
    {
        run_bszN_gr(A_in, x_dense, selected_experts, routing_weights, num_tokens, nullptr);
        g.ready_to_record = true;
    }
    else
    {
        if (!g.ready)
        {
            g.capture_begin();
            run_bszN_gr(A_in, x_dense, selected_experts, routing_weights, num_tokens, &g);
            g.capture_end();
        }

        // Padded hidden dim at num_tokens == 1: y feeds the staging copy at the head of the graph
        // and the mgemms read the (static) padded buffer. At num_tokens > 1, A_in already points
        // at the (static, per-slot) gathered buffer -- built fresh above, but at the same address
        // every call, so no patching is strictly needed there, though patching it anyway keeps
        // this code path uniform across num_tokens
        void* yptr = (num_tokens == 1 && y_pad) ? y_pad.value().data_ptr() : (void*) A_in.data_ptr();
        // Distinct from yptr: shared_experts consumes the natural (ungathered) view, not the
        // routed experts' per-slot gathered input
        void* x_dense_ptr = (void*) x_dense.data_ptr();
        auto args = std::vector<PPTR>();
        if (num_tokens == 1 && y_pad)
            args.push_back(PPTR(GP_copy2d_src, (void*) y.data_ptr()));

        if (gated)
        {
            args.push_back(PPTR(GP_mgemm_A,            yptr));
            args.push_back(PPTR(GP_mgemm_indices,      (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_end,                nullptr));

            if (gate_bias_ptrs)
            {
                args.push_back(PPTR(GP_moe_bias_add_sel,   (void*) selected_experts.data_ptr()));
                args.push_back(PPTR(GP_end,                nullptr));
            }
        }

        args.push_back(PPTR(GP_mgemm_A,            yptr));
        args.push_back(PPTR(GP_mgemm_indices,      (void*) selected_experts.data_ptr()));
        args.push_back(PPTR(GP_end,                nullptr));

        if (up_bias_ptrs)
        {
            args.push_back(PPTR(GP_moe_bias_add_sel,   (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_end,                nullptr));
        }

        auto patch_bias = [&]()
        {
            args.push_back(PPTR(GP_moe_bias_add_weighted_sel,       (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_moe_bias_add_weighted_weights,   (void*) routing_weights.data_ptr()));
            args.push_back(PPTR(GP_end,                             nullptr));
        };

        if (shared_experts && shared_gate)
        {
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
            args.push_back(PPTR(GP_end,                         nullptr));
            if (down_bias_ptrs) patch_bias();
            args.push_back(PPTR(GP_mgemm_A,                     x_dense_ptr));
            args.push_back(PPTR(GP_add_sigmoid_gate_proj_y,     x_dense_ptr));
            args.push_back(PPTR(GP_add_sigmoid_gate_proj_z,     (void*) out_d.data_ptr()));
        }
        else if (shared_experts)
        {
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
            args.push_back(PPTR(GP_end,                         nullptr));
            if (down_bias_ptrs) patch_bias();
            args.push_back(PPTR(GP_mgemm_A,                     x_dense_ptr));
            args.push_back(PPTR(GP_add_x,                       (void*) out_d.data_ptr()));
            args.push_back(PPTR(GP_add_z,                       (void*) out_d.data_ptr()));
        }
        else
        {
            args.push_back(PPTR(GP_mgemm_C,                     (void*) out_d.data_ptr()));
            args.push_back(PPTR(GP_mgemm_indices,               (void*) selected_experts.data_ptr()));
            args.push_back(PPTR(GP_mgemm_weights,               (void*) routing_weights.data_ptr()));
            if (down_bias_ptrs) patch_bias();
        }

        g.launch(args, stream);
    }
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
    bool _act_silu_oai,
    std::shared_ptr<BC_GatedMLP> _shared_experts,
    std::shared_ptr<BC_LinearFP16> _shared_gate,
    float _act_limit,
    std::vector<std::shared_ptr<BC_LinearEXL3>> _gates,
    std::vector<std::shared_ptr<BC_LinearEXL3>> _ups,
    std::vector<std::shared_ptr<BC_LinearEXL3>> _downs,
    at::Tensor _gu_trellis_ptr,
    at::Tensor _gu_suh_ptr,
    at::Tensor _gu_svh_ptr,
    at::Tensor _a_gather,
    c10::optional<at::Tensor> _gate_bias_ptrs,
    c10::optional<at::Tensor> _up_bias_ptrs,
    c10::optional<at::Tensor> _down_bias_ptrs,
    c10::optional<at::Tensor> _y_pad,
    c10::optional<at::Tensor> _out_trim,
    bool _act_relu2
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
        act_silu_oai        (_act_silu_oai),
        act_relu2           (_act_relu2),
        shared_experts      (_shared_experts),
        shared_gate         (_shared_gate),
        act_limit           (_act_limit),
        gates               (_gates),
        ups                 (_ups),
        downs               (_downs),
        gu_trellis_ptr      (_gu_trellis_ptr),
        gu_suh_ptr          (_gu_suh_ptr),
        gu_svh_ptr          (_gu_svh_ptr),
        gate_bias_ptrs      (std::move(_gate_bias_ptrs)),
        up_bias_ptrs        (std::move(_up_bias_ptrs)),
        down_bias_ptrs      (std::move(_down_bias_ptrs)),
        y_pad               (std::move(_y_pad)),
        out_trim            (std::move(_out_trim)),
        a_gather            (std::move(_a_gather))
{
    flat_token_cache.resize(MAX_BSZN);

    // Non-gated experts (NemotronH): python passes an empty gates vector (the gate pointer
    // tables are unused placeholders) and act_relu2; the gate GEMMs are skipped throughout
    gated = !gates.empty();
    TORCH_CHECK(gated || act_relu2, "BC_BlockSparseMLP: gateless experts require act_relu2");
    TORCH_CHECK(!(shared_experts && (down_bias_ptrs || y_pad)),
        "BC_BlockSparseMLP: shared experts not supported with expert biases or padded dims");
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

    {
        at::Tensor gi = interm_gu.slice(0, 0, bsz);
        at::Tensor ui = interm_gu.slice(0, bsz, bsz * 2);

        if (gated)
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

        if (!gated)
            relu_mul_gr(ui, ui, ai, act_limit, graph);
        else if (act_silu)
            silu_mul_gr(gi, ui, ai, act_limit, graph);
        else if (act_gelu)
            gelu_mul_gr(gi, ui, ai, act_limit, graph);
        else if (act_silu_oai)
            silu_oai_mul_gr(gi, ui, ai, act_limit, graph);
        else if (act_relu2)
            relu2_mul_gr(gi, ui, ai, act_limit, graph);
    }

    // A_had must not alias A (autotune relaunches on the first call); the gate slice is free
    // after the activation
    at::Tensor gi_scratch = interm_gu.slice(0, 0, bsz);
    exl3_gemm_gr
    (
        ai,
        downs[expert_idx]->trellis,
        oi,
        downs[expert_idx]->suh,
        gi_scratch,
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

    if (graph_single[graphidx].disabled || (!graph_single[graphidx].ready && !graph_single[graphidx].ready_to_record))
    {
        run_single_expert_gr(y, expert_idx, nullptr);
        graph_single[graphidx].ready_to_record = true;
    }
    else
    {
        if (!graph_single[graphidx].ready)
        {
            prepare_ctx(y.get_device());

            graph_single[graphidx].capture_begin();
            run_single_expert_gr(y, expert_idx, &graph_single[graphidx]);
            graph_single[graphidx].capture_end();
        }

        auto args = std::vector<PPTR>();
        if (gated)
        {
            args.push_back(PPTR(GP_gemm_A,             (void*) y.data_ptr()));
            args.push_back(PPTR(GP_gemm_B_trellis,     (void*) gates[expert_idx]->trellis.data_ptr()));
            args.push_back(PPTR(GP_gemm_B_suh,         (void*) gates[expert_idx]->suh.data_ptr()));
            args.push_back(PPTR(GP_gemm_B_svh,         (void*) gates[expert_idx]->svh.data_ptr()));
            args.push_back(PPTR(GP_end,                nullptr));
        }
        args.push_back(PPTR(GP_gemm_A,             (void*) y.data_ptr()));
        args.push_back(PPTR(GP_gemm_B_trellis,     (void*) ups[expert_idx]->trellis.data_ptr()));
        args.push_back(PPTR(GP_gemm_B_suh,         (void*) ups[expert_idx]->suh.data_ptr()));
        args.push_back(PPTR(GP_gemm_B_svh,         (void*) ups[expert_idx]->svh.data_ptr()));
        args.push_back(PPTR(GP_end,                nullptr));
        args.push_back(PPTR(GP_gemm_B_trellis,     (void*) downs[expert_idx]->trellis.data_ptr()));
        args.push_back(PPTR(GP_gemm_B_suh,         (void*) downs[expert_idx]->suh.data_ptr()));
        args.push_back(PPTR(GP_gemm_B_svh,         (void*) downs[expert_idx]->svh.data_ptr()));

        graph_single[graphidx].launch(args, stream);
    }
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

    if (gated)
    {
        had_r_128_dual(y, yh1, gates[expert_idx]->suh, c10::nullopt,
                       y, yh2, ups[expert_idx]->suh, c10::nullopt, 1.0);

        reconstruct(dq_temp_up, gates[expert_idx]->trellis, gate_K, gate_mcg, gate_mul1);
        hgemm(yh1, dq_temp_up, interm1);
        reconstruct(dq_temp_up, ups[expert_idx]->trellis, up_K, up_mcg, up_mul1);
        hgemm(yh2, dq_temp_up, interm2);

        had_r_128_dual(interm1, interm1, c10::nullopt, gates[expert_idx]->svh,
                       interm2, interm2, c10::nullopt, ups[expert_idx]->svh, 1.0);
    }
    else
    {
        had_r_128(y, yh2, ups[expert_idx]->suh, c10::nullopt, 1.0);
        reconstruct(dq_temp_up, ups[expert_idx]->trellis, up_K, up_mcg, up_mul1);
        hgemm(yh2, dq_temp_up, interm2);
        had_r_128(interm2, interm2, c10::nullopt, ups[expert_idx]->svh, 1.0);
    }

    if (!gated)
        relu_mul(interm2, interm2, interm_a, act_limit);
    else if (act_silu)
        silu_mul(interm1, interm2, interm_a, act_limit);
    else if (act_gelu)
        gelu_mul(interm1, interm2, interm_a, act_limit);
    else if (act_silu_oai)
        silu_oai_mul(interm1, interm2, interm_a, act_limit);
    else if (act_relu2)
        relu2_mul(interm1, interm2, interm_a, act_limit);

    had_r_128(interm_a, interm_a, downs[expert_idx]->suh, c10::nullopt, 1.0);
    reconstruct(dq_temp_down, downs[expert_idx]->trellis, down_K, down_mcg, down_mul1);
    hgemm(interm_a, dq_temp_down, out);
    had_r_128(out, out, c10::nullopt, downs[expert_idx]->svh, 1.0);
}
