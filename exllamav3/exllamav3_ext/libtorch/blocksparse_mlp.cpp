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

void BC_BlockSparseMLP::run_bszN
(
    const at::Tensor& y,
    at::Tensor& selected_experts,
    at::Tensor& routing_weights
)
{
    // Two fused launches per layer cover every (token, expert) slot of the batch (see
    // exl3_moe_coop_kernel.cuh): input rotation + gate/up GEMVs + activation, then down GEMVs +
    // the weighted per-token reduction, with expert biases, padded dims (x zero-padded to the
    // quantized input width in the kernel, output trimmed to H) and expert-range masking
    // (out-of-range picks contribute exact zeros to the partial sum) handled inside. Nothing is
    // captured or patched: routing tensors are read directly, so no per-bsz statics or graphs
    int num_tokens = (int) y.size(0);
    TORCH_CHECK(num_tokens >= 1 && num_tokens <= MAX_BSZN, "run_bszN: bsz out of supported range");

    c10::cuda::CUDAGuard device_guard(y.device());

    // Shared experts run through their own multi-row graph first (same stream); the kernel adds
    // the result into the routed sum, through the sigmoid gate when there is one
    c10::optional<at::Tensor> sh_o;
    if (sh_coop)
    {
        // One-expert fused launch at the shared expert's own bit width (expert 0, weight 1);
        // the sigmoid gate, if any, is applied by the routed launch when it merges the result
        at::Tensor out_d_sh_n = out_d_sh.value().slice(1, 0, num_tokens);
        exl3_moe_coop_run(sh_coop_p, sh_K_gu, sh_K_d, sh_cb, y, sh_sel.slice(0, 0, num_tokens), sh_rw.slice(0, 0, num_tokens), c10::nullopt);
        sh_o = out_d_sh_n;
    }
    else if (shared_experts)
    {
        at::Tensor x_dense = y.unsqueeze(0);
        at::Tensor out_d_sh_n = out_d_sh.value().slice(1, 0, num_tokens);
        shared_experts->run_bszN(x_dense, out_d_sh_n);
        sh_o = out_d_sh_n;
    }
    exl3_moe_coop_run(coop_p, coop_K_gu, coop_K_d, coop_cb, y, selected_experts, routing_weights, sh_o);
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
    at::Tensor _coop_ctr,
    at::Tensor _had_u,
    at::Tensor _dq_temp_up,
    at::Tensor _dq_temp_down,
    int _min_expert,
    int _max_expert,
    at::Tensor _gate_ptrs_trellis,
    at::Tensor _gate_ptrs_suh,
    at::Tensor _gate_ptrs_svh,
    float _gate_K,
    bool _gate_mcg,
    bool _gate_mul1,
    at::Tensor _up_ptrs_trellis,
    at::Tensor _up_ptrs_suh,
    at::Tensor _up_ptrs_svh,
    float _up_K,
    bool _up_mcg,
    bool _up_mul1,
    at::Tensor _down_ptrs_trellis,
    at::Tensor _down_ptrs_suh,
    at::Tensor _down_ptrs_svh,
    float _down_K,
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
    at::Tensor _out_bszn,
    c10::optional<at::Tensor> _gate_bias_ptrs,
    c10::optional<at::Tensor> _up_bias_ptrs,
    c10::optional<at::Tensor> _down_bias_ptrs,
    bool _act_relu2,
    bool _sh_coop
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
        coop_ctr            (std::move(_coop_ctr)),
        had_u               (std::move(_had_u)),
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
        out_bszn            (std::move(_out_bszn))
{
    // Non-gated experts (NemotronH): python passes an empty gates vector (the gate pointer
    // tables are unused placeholders) and act_relu2; the gate GEMMs are skipped throughout
    gated = !gates.empty();
    TORCH_CHECK(gated || act_relu2, "BC_BlockSparseMLP: gateless experts require act_relu2");
    // The fused decode kernels decode all three projections with one codebook instantiation
    TORCH_CHECK(gate_mcg == up_mcg && up_mcg == down_mcg && gate_mul1 == up_mul1 && up_mul1 == down_mul1,
                "BC_BlockSparseMLP: gate/up/down must share a codebook");
    TORCH_CHECK(!shared_gate || shared_experts, "BC_BlockSparseMLP: shared gate without shared experts");
    TORCH_CHECK(!gated || gate_K == up_K, "BC_BlockSparseMLP: gate and up must share a bit width");
    TORCH_CHECK(coop_ctr.scalar_type() == at::kInt && coop_ctr.numel() >=
                exl3_moe_coop_ctr_len(interm_u.size(0), out_bszn.size(0), interm_u.size(-1), out_d.size(-1)),
                "BC_BlockSparseMLP: counter scratch too small");
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

    // Static part of the fused decode kernels' parameters, validated once
    {
        int act = act_silu_oai ? MOE_COOP_ACT_SILU_OAI :
                  act_gelu ? MOE_COOP_ACT_GELU :
                  act_relu2 ? MOE_COOP_ACT_RELU2 :
                  MOE_COOP_ACT_SILU;
        c10::optional<at::Tensor> sh_w;
        if (shared_gate) sh_w = shared_gate->weight;
        coop_p = exl3_moe_coop_prepare
        (
            (int) yh.size(-1),
            gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
            up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
            down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
            gate_bias_ptrs, up_bias_ptrs, down_bias_ptrs,
            gate_K, up_K, down_K,
            up_mcg, up_mul1,
            act, act_limit, gated,
            yh, had_u, interm_g, interm_u, interm_a, out_d, coop_ctr, out_bszn,
            sh_w, coop_K_gu, coop_K_d, coop_cb
        );
        coop_p.min_expert = min_expert;
        coop_p.max_expert = max_expert;
    }

    // Shared expert fused launch. The shared BC_GatedMLP carries its gate and up either as one
    // merged pointer table (gu_ptrs_*: [gate, up], the multi-GEMM form) or as separate bound
    // linears; the down projection is always a bound linear
    if (_sh_coop && shared_experts && out_d_sh && shared_experts->down && (shared_experts->gu_ptrs_trellis || (shared_experts->gate && shared_experts->up)))
    {
        auto& se = shared_experts; auto& d = se->down;
        auto dev = yh.device();
        auto lopt = at::TensorOptions().dtype(at::kLong);
        auto table = [&](const at::Tensor& t) { return at::tensor({(int64_t) t.data_ptr()}, lopt).to(dev); };
        at::Tensor gt, gs, gv, ut, us, uv;
        float Kgu; int Hi_sh, I_sh; bool mcg_gu, mul1_gu;
        c10::optional<at::Tensor> gb, ub;
        if (se->gu_ptrs_trellis)
        {
            gt = se->gu_ptrs_trellis->slice(0, 0, 1).contiguous(); ut = se->gu_ptrs_trellis->slice(0, 1, 2).contiguous();
            gs = se->gu_ptrs_suh->slice(0, 0, 1).contiguous();     us = se->gu_ptrs_suh->slice(0, 1, 2).contiguous();
            gv = se->gu_ptrs_svh->slice(0, 0, 1).contiguous();     uv = se->gu_ptrs_svh->slice(0, 1, 2).contiguous();
            Kgu = se->gu_K; mcg_gu = se->gu_mcg; mul1_gu = se->gu_mul1;
            Hi_sh = (int) se->guh.size(-1); I_sh = (int) se->gu.size(-1);
        }
        else
        {
            auto& g = se->gate; auto& u = se->up;
            gt = table(g->trellis); gs = table(g->suh); gv = table(g->svh);
            ut = table(u->trellis); us = table(u->suh); uv = table(u->svh);
            Kgu = g->K; mcg_gu = u->mcg; mul1_gu = u->mul1;
            Hi_sh = (int) g->trellis.size(0) * 16; I_sh = (int) g->trellis.size(1) * 16;
            if (g->K != u->K || g->mcg != u->mcg || g->mul1 != u->mul1) Kgu = -1;
            if (g->bias) gb = table(g->bias.value());
            if (u->bias) ub = table(u->bias.value());
        }
        const int Ho_sh = (int) d->trellis.size(1) * 16;
        const bool ok = Kgu >= 1 && Hi_sh % 128 == 0 && I_sh % 128 == 0 && Ho_sh % 128 == 0 &&
                        (int) d->trellis.size(0) * 16 == I_sh && mcg_gu == d->mcg && mul1_gu == d->mul1 &&
                        out_d_sh->size(-1) <= Ho_sh && (int) yh.size(-1) == Hi_sh;
        if (ok)
        {
            at::Tensor dt = table(d->trellis), ds = table(d->suh), dv = table(d->svh);
            c10::optional<at::Tensor> db;
            if (d->bias) db = table(d->bias.value());
            auto hopt = at::TensorOptions().dtype(at::kHalf).device(dev);
            auto gopt = at::TensorOptions().dtype(interm_g.scalar_type()).device(dev);
            at::Tensor had_g_sh = at::empty({MAX_BSZN, Hi_sh}, hopt), had_u_sh = at::empty({MAX_BSZN, Hi_sh}, hopt);
            at::Tensor gu_g_sh = at::empty({MAX_BSZN, I_sh}, gopt), gu_u_sh = at::empty({MAX_BSZN, I_sh}, gopt);
            at::Tensor act_out_sh = at::empty({MAX_BSZN, I_sh}, hopt);
            at::Tensor d_out_sh = at::empty({MAX_BSZN, Ho_sh}, at::TensorOptions().dtype(at::kFloat).device(dev));
            at::Tensor ctr_sh = at::zeros({exl3_moe_coop_ctr_len(MAX_BSZN, MAX_BSZN, I_sh, Ho_sh)}, at::TensorOptions().dtype(at::kInt).device(dev));
            int act_sh = se->act_gelu ? MOE_COOP_ACT_GELU : se->act_relu2 ? MOE_COOP_ACT_RELU2 : MOE_COOP_ACT_SILU;
            sh_coop_p = exl3_moe_coop_prepare
            (
                Hi_sh, gt, gs, gv, ut, us, uv, dt, ds, dv, gb, ub, db,
                Kgu, Kgu, d->K, mcg_gu, mul1_gu, act_sh, se->act_limit, true,
                had_g_sh, had_u_sh, gu_g_sh, gu_u_sh, act_out_sh, d_out_sh, ctr_sh, out_d_sh.value(),
                c10::nullopt, sh_K_gu, sh_K_d, sh_cb
            );
            sh_coop_p.min_expert = -1;
            sh_coop_p.max_expert = -1;
            sh_tables = { gt, gs, gv, ut, us, uv, dt, ds, dv, had_g_sh, had_u_sh, gu_g_sh, gu_u_sh, act_out_sh, d_out_sh, ctr_sh };
            if (gb) sh_tables.push_back(gb.value());
            if (ub) sh_tables.push_back(ub.value());
            if (db) sh_tables.push_back(db.value());
            sh_sel = at::zeros({MAX_BSZN, 1}, lopt.device(dev));
            sh_rw = at::ones({MAX_BSZN, 1}, hopt);
            sh_coop = true;
        }
    }
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
        hgemm_recon(yh1, dq_temp_up, interm1);
        reconstruct(dq_temp_up, ups[expert_idx]->trellis, up_K, up_mcg, up_mul1);
        hgemm_recon(yh2, dq_temp_up, interm2);

        had_r_128_dual(interm1, interm1, c10::nullopt, gates[expert_idx]->svh,
                       interm2, interm2, c10::nullopt, ups[expert_idx]->svh, 1.0);
    }
    else
    {
        had_r_128(y, yh2, ups[expert_idx]->suh, c10::nullopt, 1.0);
        reconstruct(dq_temp_up, ups[expert_idx]->trellis, up_K, up_mcg, up_mul1);
        hgemm_recon(yh2, dq_temp_up, interm2);
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
    hgemm_recon(interm_a, dq_temp_down, out);
    had_r_128(out, out, c10::nullopt, downs[expert_idx]->svh, 1.0);
}
