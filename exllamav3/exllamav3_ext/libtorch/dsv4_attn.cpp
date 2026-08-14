#include <Python.h>
#include "dsv4_attn.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../add.cuh"
#include "../norm.cuh"
#include "../rope.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../dsv4_compress.cuh"
#include "../dsa_topk.cuh"
#include "../graph.cuh"

bool BC_DSV4Attention::needs_configure(int seq, int regime)
{
    TORCH_CHECK(1 <= seq && seq <= MAX_QLEN && (regime == 0 || regime == 1),
                "BC_DSV4Attention: shape out of range");
    return !slot(seq, regime).configured;
}

void BC_DSV4Attention::configure_slot
(
    int seq, int regime,
    at::Tensor x_st, at::Tensor qa_st, at::Tensor qres_st, at::Tensor q_st,
    at::Tensor kv_st, at::Tensor xh_a, at::Tensor xh_b,
    c10::optional<at::Tensor> mgc_comp, c10::optional<at::Tensor> mgc_idx,
    c10::optional<at::Tensor> qidx_st, c10::optional<at::Tensor> wts_st,
    c10::optional<at::Tensor> scores_st, c10::optional<at::Tensor> indices_st,
    at::Tensor ws_ml, at::Tensor ws_acc, at::Tensor attn_out_st,
    at::Tensor woa_c_st, at::Tensor woa_t_st, at::Tensor woa_xh, at::Tensor y_st,
    std::shared_ptr<TritonKernel> k_split,
    std::shared_ptr<TritonKernel> k_combine,
    std::shared_ptr<TritonKernel> k_fewq,
    c10::optional<at::Tensor> fan_c_ptrs,
    c10::optional<at::Tensor> fan_ahad
)
{
    Slot& s = slot(seq, regime);
    s.x_st = std::move(x_st);
    s.qa_st = std::move(qa_st);
    s.qres_st = std::move(qres_st);
    s.q_st = std::move(q_st);
    s.kv_st = std::move(kv_st);
    s.xh_a = std::move(xh_a);
    s.xh_b = std::move(xh_b);
    s.mgc_comp = std::move(mgc_comp);
    s.mgc_idx = std::move(mgc_idx);
    s.qidx_st = std::move(qidx_st);
    s.wts_st = std::move(wts_st);
    s.scores_st = std::move(scores_st);
    s.indices_st = std::move(indices_st);
    s.ws_ml = std::move(ws_ml);
    s.ws_acc = std::move(ws_acc);
    s.attn_out_st = std::move(attn_out_st);
    s.woa_c_st = std::move(woa_c_st);
    s.woa_t_st = std::move(woa_t_st);
    s.woa_xh = std::move(woa_xh);
    s.y_st = std::move(y_st);
    s.k_split = k_split;
    s.k_combine = k_combine;
    s.k_fewq = k_fewq;
    s.fan_c_ptrs = std::move(fan_c_ptrs);
    s.fan_ahad = std::move(fan_ahad);
    s.graph = std::make_unique<Graph>();
    s.configured = true;
}

void BC_DSV4Attention::run_gr
(
    const at::Tensor& x, int seq, int regime, int pos, int win_beg,
    Slot& s, Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    int D = head_dim;
    bool has_comp = comp_bc != nullptr;
    bool topk_regime = regime == 1;

    // Stage x (the only per-call input pointer). x_st is padded to >= 8 rows so the wts
    // hgemm can run full-height (cuBLASLt's M = 2..7 kernel choice is ~13x slower);
    // every other consumer reads the seq-row view
    at::Tensor x2 = x.view({seq, hidden_size});
    at::Tensor x_rows = s.x_st.narrow(0, 0, seq);
    copy2d_gr(x2, x_rows, graph);

    // x-side projections: single per-matrix-N mgemm over [q_a, wkv, comp pair, idx pair]
    // when the fan is available (uniform bits/format), else individual gemms
    bool use_fan = fan_trellis.has_value();
    if (use_fan)
    {
        at::Tensor A = x_rows.unsqueeze(0);
        at::Tensor Cd = s.qa_st.view({1, seq, q_lora});   // dtype + max-width carrier
        c10::optional<at::Tensor> no_w = {};
        exl3_mgemm_gr(A, fan_trellis.value(), Cd, fan_suh.value(), s.fan_ahad.value(),
                      fan_svh.value(), fan_indices, no_w, q_a->K, -1, q_a->mcg, q_a->mul1,
                      -1, -1, 0, graph, 1, fan_n, s.fan_c_ptrs);
    }
    else
    {
        exl3_gemm_gr(x_rows, q_a->trellis, s.qa_st, q_a->suh, s.xh_a, q_a->svh, -1, q_a->mcg, q_a->mul1, 0, graph);
        exl3_gemm_gr(x_rows, wkv->trellis, s.kv_st, wkv->suh, s.xh_a, wkv->svh, -1, wkv->mcg, wkv->mul1, 0, graph);
    }
    {
        c10::optional<at::Tensor> qnw = q_norm_w;
        rms_norm_gr(s.qa_st, qnw, s.qres_st, rms_norm_eps, 0.0f, 1.0f, false, graph);
    }
    exl3_gemm_gr(s.qres_st, q_b->trellis, s.q_st, q_b->suh, s.xh_b, q_b->svh,
                 -1, q_b->mcg, q_b->mul1, 0, graph);

    // Fused head norms + partial rope, positions from the device scalar
    {
        at::Tensor q4 = s.q_st.view({1, seq, num_q_heads, D});
        at::Tensor kv4 = s.kv_st.view({1, seq, 1, D});
        c10::optional<at::Tensor> k_in = kv4;
        c10::optional<at::Tensor> k_out = kv4;
        c10::optional<at::Tensor> positions = pos_dev;
        c10::optional<at::Tensor> no_pid = {};
        c10::optional<at::Tensor> qn = q_ones;
        c10::optional<at::Tensor> kn = kv_norm_w;
        rope_gr
        (
            q4, q4, k_in, k_out, inv_freq, 0, positions, no_pid,
            ROPESTYLE_GPTJ, 1.0f, qn, kn, rms_norm_eps, 0.0f, 0.0f, 0,
            1, D - rope_dim, graph
        );
    }

    // Compressors: emitted entries into the paged pools via the per-job block table
    // static, ring rows stored
    if (has_comp)
    {
        c10::optional<at::Tensor> pt = pos_dev;
        c10::optional<at::Tensor> bt = pool_bt;
        at::Tensor bkv = comp_buf_kv.value(), bg = comp_buf_gate.value();
        at::Tensor pc = pool_c.value(), pr = pool_r.value();
        c10::optional<at::Tensor> pr_opt = pr;
        comp_bc->run_gr(x_rows, bkv, bg, comp_ovl, pc, pr_opt, pos, pt, s.mgc_comp, graph,
                        use_fan, bt, pool_epp);
        if (idx_bc)
        {
            at::Tensor ikv = idx_buf_kv.value(), ig = idx_buf_gate.value();
            at::Tensor pi = pool_idx.value();
            c10::optional<at::Tensor> no_b = {};
            idx_bc->run_gr(x_rows, ikv, ig, idx_ovl, pi, no_b, pos, pt, s.mgc_idx, graph,
                           use_fan, bt, pool_epp);
        }
    }

    int ec = has_comp ? (pos + seq) / m : 0;

    // Long-ctx regime: indexer scoring + capture-safe top-k selection
    if (topk_regime)
    {
        exl3_gemm_gr(s.qres_st, idx_wq_b->trellis, s.qidx_st.value(), idx_wq_b->suh,
                     s.xh_b, idx_wq_b->svh, -1, idx_wq_b->mcg, idx_wq_b->mul1, 0, graph);
        {
            at::Tensor qi4 = s.qidx_st.value().view({1, seq, index_n_heads, index_head_dim});
            at::Tensor qi_sl = qi4.narrow(3, index_head_dim - rope_dim, rope_dim);
            c10::optional<at::Tensor> no_k = {};
            c10::optional<at::Tensor> no_ko = {};
            c10::optional<at::Tensor> positions = pos_dev;
            c10::optional<at::Tensor> no_pid = {};
            c10::optional<at::Tensor> no_n = {};
            rope_gr
            (
                qi_sl, qi_sl, no_k, no_ko, inv_freq, 0, positions, no_pid,
                ROPESTYLE_GPTJ, 1.0f, no_n, no_n, 1e-6f, 0.0f, 0.0f, 0,
                1, 0, graph
            );
        }
        hgemm_gr(s.x_st, idx_weights_w.value(), s.wts_st.value(), graph);  // full padded height

        // Few-query indexer kernel; T / q_pos0 / bound_max are patched per call
        {
            at::Tensor& sc = s.scores_st.value();
            int T = ec;
            std::vector<void*> args = {
                (void*) s.qidx_st.value().data_ptr(),
                (void*) s.wts_st.value().data_ptr(),
                (void*) pool_idx.value().data_ptr(),
                (void*) sc.data_ptr(),
                (void*) (uintptr_t) (uint32_t) T,
                (void*) (uintptr_t) (uint32_t) seq,
                (void*) (uintptr_t) (uint32_t) pos,
                (void*) (uintptr_t) (uint32_t) ec,
                (void*) pool_bt.data_ptr(),         // block_table (paged pool_idx)
                (void*) (uintptr_t) (uint32_t) 0,   // num_pages_per_row (MULTIROW only)
            };
            int s_max = (int) sc.size(1);
            int gy = CEIL_DIVIDE(s_max, 128);
            s.k_fewq->launch(seq, gy, 1, args, stream);
            if (graph)
            {
                graph->record_param(s.k_fewq->handle(), GP_dsa_T, 4, 4);
                graph->record_param(s.k_fewq->handle(), GP_dsa_qpos, 6, 4);
                graph->record_param(s.k_fewq->handle(), GP_dsa_bound_max, 7, 4);
                graph->record_param(s.k_fewq->handle(), GP_end, 0);
            }
        }
        dsa_topk_gr(s.scores_st.value(), s.indices_st.value(), index_topk, graph);
    }

    // Split + combine attention (AOT kernels)
    int hb = CEIL_DIVIDE(num_q_heads, block_h);
    int n_prev = std::min(std::min(window - 1, pos - win_beg), pos);
    int win_floor = pos - n_prev;
    int k_len = topk_regime ? index_topk : 0;
    int pool_len = ec;
    {
        std::vector<void*> args =
        {
            (void*) s.q_st.data_ptr(),
            (void*) ring.data_ptr(),
            (void*) s.kv_st.data_ptr(),
            (void*) (has_comp ? pool_c.value().data_ptr() : s.kv_st.data_ptr()),
            (void*) (has_comp ? pool_r.value().data_ptr() : s.kv_st.data_ptr()),
            (void*) pool_bt.data_ptr(),
            (void*) (topk_regime ? s.indices_st.value().data_ptr() : pool_bt.data_ptr()),
            (void*) s.ws_ml.data_ptr(),
            (void*) s.ws_acc.data_ptr(),
            (void*) (uintptr_t) (uint32_t) k_len,
            (void*) (uintptr_t) (uint32_t) window,
            (void*) (uintptr_t) (uint32_t) pool_len,
            (void*) (uintptr_t) (uint32_t) 0,   // num_pages_per_row (shared single-row bt)
            (void*) (uintptr_t) (uint32_t) pos,
            (void*) (uintptr_t) (uint32_t) win_floor,
            (void*) (uintptr_t) (uint32_t) win_beg,
            (void*) (uintptr_t) (uint32_t) 0,   // slot_ids (MULTIROW only)
            (void*) (uintptr_t) (uint32_t) 0,   // ring_stride (MULTIROW only)
        };
        s.k_split->launch(seq * hb, n_splits, 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_split->handle(), GP_dsa_pool_len, 11, 4);
            graph->record_param(s.k_split->handle(), GP_dsa_qpos, 13, 4);
            graph->record_param(s.k_split->handle(), GP_dsa_win_floor, 14, 4);
            graph->record_param(s.k_split->handle(), GP_dsa_ring_beg, 15, 4);
            graph->record_param(s.k_split->handle(), GP_end, 0);
        }
    }
    {
        std::vector<void*> args =
        {
            (void*) s.ws_ml.data_ptr(),
            (void*) s.ws_acc.data_ptr(),
            (void*) sinks.data_ptr(),
            (void*) inv_freq_neg.data_ptr(),
            (void*) s.attn_out_st.data_ptr(),
            (void*) (uintptr_t) (uint32_t) pos,
            (void*) (uintptr_t) (uint32_t) seq,
            (void*) (uintptr_t) (uint32_t) n_splits,
        };
        s.k_combine->launch(seq * hb, CEIL_DIVIDE(D, 128), 1, args, stream);
        if (graph)
        {
            graph->record_param(s.k_combine->handle(), GP_dsa_qpos, 5, 4);
            graph->record_param(s.k_combine->handle(), GP_end, 0);
        }
    }

    // Grouped o_proj: wo_a as an 8-expert mgemm over the group-major attention output
    {
        at::Tensor A = s.attn_out_st;    // (G, seq, hpg * hd)
        at::Tensor C = s.woa_c_st;       // (G, seq, o_lora)
        at::Tensor ah = s.woa_xh.view({o_groups, seq, A.size(2)});
        c10::optional<at::Tensor> mi = woa_indices;
        c10::optional<at::Tensor> no_w = {};
        exl3_mgemm_gr(A, woa_trellis, C, woa_suh, ah, woa_svh, mi, no_w, woa_k, -1, woa_mcg, woa_mul1, -1, -1, 0, graph, 1);
    }
    at::Tensor wo_b_in;
    if (seq == 1)
    {
        // (G, 1, o_lora) is memory-identical to the concatenated (1, G * o_lora) row
        wo_b_in = s.woa_c_st.view({1, o_groups * o_lora});
    }
    else
    {
        // Per-group 2D transpose into (seq, G * o_lora)
        for (int g = 0; g < o_groups; ++g)
        {
            at::Tensor src = s.woa_c_st.select(0, g);
            at::Tensor dst = s.woa_t_st.narrow(1, g * o_lora, o_lora);
            copy2d_gr(src, dst, graph);
        }
        wo_b_in = s.woa_t_st;
    }
    at::Tensor wo_b_xh = s.woa_xh.view({-1}).narrow(0, 0, wo_b_in.numel()).view({(int64_t) seq, (int64_t) o_groups * o_lora});
    exl3_gemm_gr(wo_b_in, wo_b->trellis, s.y_st, wo_b->suh, wo_b_xh, wo_b->svh, -1, wo_b->mcg, wo_b->mul1, 0, graph);

    // SWA ring append (device-scalar addressed; shift/rebase handled host-side pre-replay)
    dsv4_ring_append_gr(s.kv_st, ring, pos_dev, ring_beg_dev, graph);
}

bool BC_DSV4BatchAttention::needs_configure(int B, int S)
{
    TORCH_CHECK(1 <= B && B <= MAX_B && 1 <= S && S <= MAX_S,
                "BC_DSV4BatchAttention: shape out of range");
    return !slot(B, S).configured;
}

void BC_DSV4BatchAttention::configure_slot
(
    int B, int S,
    at::Tensor x_st, at::Tensor qa_st, at::Tensor qres_st, at::Tensor q_st,
    at::Tensor kv_st, at::Tensor xh_b,
    c10::optional<at::Tensor> mgc_comp, c10::optional<at::Tensor> mgc_idx,
    c10::optional<at::Tensor> qidx_st, c10::optional<at::Tensor> wts_st,
    c10::optional<at::Tensor> scores_st, c10::optional<at::Tensor> indices_st,
    at::Tensor ws_ml, at::Tensor ws_acc, at::Tensor attn_out_st,
    at::Tensor woa_c_st, at::Tensor woa_t_st, at::Tensor woa_xh, at::Tensor y_st,
    at::Tensor fan_c_ptrs, at::Tensor fan_ahad,
    std::shared_ptr<TritonKernel> k_split,
    std::shared_ptr<TritonKernel> k_combine,
    std::shared_ptr<TritonKernel> k_fewq,
    c10::optional<at::Tensor> fan2_c_ptrs,
    c10::optional<at::Tensor> fan2_ahad
)
{
    Slot& s = slot(B, S);
    s.x_st = std::move(x_st);
    s.qa_st = std::move(qa_st);
    s.qres_st = std::move(qres_st);
    s.q_st = std::move(q_st);
    s.kv_st = std::move(kv_st);
    s.xh_b = std::move(xh_b);
    s.mgc_comp = std::move(mgc_comp);
    s.mgc_idx = std::move(mgc_idx);
    s.qidx_st = std::move(qidx_st);
    s.wts_st = std::move(wts_st);
    s.scores_st = std::move(scores_st);
    s.indices_st = std::move(indices_st);
    s.ws_ml = std::move(ws_ml);
    s.ws_acc = std::move(ws_acc);
    s.attn_out_st = std::move(attn_out_st);
    s.woa_c_st = std::move(woa_c_st);
    s.woa_t_st = std::move(woa_t_st);
    s.woa_xh = std::move(woa_xh);
    s.y_st = std::move(y_st);
    s.fan_c_ptrs = std::move(fan_c_ptrs);
    s.fan_ahad = std::move(fan_ahad);
    s.k_split = k_split;
    s.k_combine = k_combine;
    s.k_fewq = k_fewq;
    s.fan2_c_ptrs = std::move(fan2_c_ptrs);
    s.fan2_ahad = std::move(fan2_ahad);
    s.graph = std::make_unique<Graph>();
    s.configured = true;
}

void BC_DSV4BatchAttention::run_gr(const at::Tensor& x, int B, int S, Slot& s, Graph* graph)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    int R = B * S;
    int D = head_dim;
    bool has_comp = pool_c.has_value();
    bool has_idx = pool_idx.has_value();

    // Per-job device state rows (fixed pointers into the shared (6, MAX_B) array)
    at::Tensor a_pos   = arr.select(0, 0);
    at::Tensor a_floor = arr.select(0, 1);
    at::Tensor a_beg   = arr.select(0, 2);
    at::Tensor a_ec    = arr.select(0, 3);
    at::Tensor a_slot  = arr.select(0, 4).narrow(0, 0, B);
    c10::optional<at::Tensor> pos_opt = a_pos.narrow(0, 0, B);
    c10::optional<at::Tensor> slot_opt = a_slot;

    // Stage x (the only per-call input pointer). x_st is padded to >= 8 rows: the wts
    // hgemm below runs full-height so cuBLASLt picks its M >= 8 kernel (the M = 2..7
    // heuristic choice is ~13x slower); everything else reads the R-row view
    at::Tensor x2 = x.view({R, hidden_size});
    at::Tensor x_rows = s.x_st.narrow(0, 0, R);
    copy2d_gr(x2, x_rows, graph);

    // x-side projections: one per-matrix-N mgemm over [q_a, wkv, (comp kv/gate), (idx
    // kv/gate)] for all R rows
    {
        at::Tensor A = x_rows.unsqueeze(0);
        at::Tensor Cd = s.qa_st.view({1, R, q_lora});
        c10::optional<at::Tensor> no_w = {};
        c10::optional<at::Tensor> fi = fan_indices;
        c10::optional<at::Tensor> fn = fan_n;
        c10::optional<at::Tensor> fc = s.fan_c_ptrs;
        exl3_mgemm_gr(A, fan_trellis, Cd, fan_suh, s.fan_ahad, fan_svh, fi, no_w,
                      fan_k, -1, fan_mcg, fan_mul1, -1, -1, 0, graph, 1, fn, fc);
    }
    {
        c10::optional<at::Tensor> qnw = q_norm_w;
        rms_norm_gr(s.qa_st, qnw, s.qres_st, rms_norm_eps, 0.0f, 1.0f, false, graph);
    }
    // q-side: q_b + idx_wq_b as one 2-expert mgemm when the q fan is available
    bool use_fan2 = fan2_trellis.has_value();
    if (use_fan2)
    {
        at::Tensor A = s.qres_st.unsqueeze(0);
        at::Tensor Cd = s.q_st.view({1, R, num_q_heads * D});
        c10::optional<at::Tensor> no_w = {};
        exl3_mgemm_gr(A, fan2_trellis.value(), Cd, fan2_suh.value(), s.fan2_ahad.value(),
                      fan2_svh.value(), fan2_indices, no_w, fan2_k, -1, fan2_mcg, fan2_mul1,
                      -1, -1, 0, graph, 1, fan2_n, s.fan2_c_ptrs);
    }
    else
        exl3_gemm_gr(s.qres_st, q_b->trellis, s.q_st, q_b->suh, s.xh_b, q_b->svh,
                     -1, q_b->mcg, q_b->mul1, 0, graph);

    // Fused head norms + partial rope, per-job base positions from the state array
    {
        at::Tensor q4 = s.q_st.view({B, S, num_q_heads, D});
        at::Tensor kv4 = s.kv_st.view({B, S, 1, D});
        c10::optional<at::Tensor> k_in = kv4;
        c10::optional<at::Tensor> k_out = kv4;
        c10::optional<at::Tensor> no_pid = {};
        c10::optional<at::Tensor> qn = q_ones;
        c10::optional<at::Tensor> kn = kv_norm_w;
        rope_gr
        (
            q4, q4, k_in, k_out, inv_freq, 0, pos_opt, no_pid,
            ROPESTYLE_GPTJ, 1.0f, qn, kn, rms_norm_eps, 0.0f, 0.0f, 0,
            1, D - rope_dim, graph
        );
    }

    // Compressors on the fan outputs: grid.y-batched, entries written to the paged pools
    // through each job's block-table row
    if (has_comp)
    {
        at::Tensor bkv = comp_buf_kv.value(), bg = comp_buf_gate.value();
        at::Tensor pc = pool_c.value();
        c10::optional<at::Tensor> pr_opt = pool_r.value();
        c10::optional<at::Tensor> bt = bt_st.narrow(0, 0, B);
        dsv4_compress_gr(s.mgc_comp.value().select(0, 0), s.mgc_comp.value().select(0, 1),
                         bkv, bg, comp_ovl, comp_ape.value(), comp_norm_w.value(), comp_eps,
                         comp_inv_freq.value(), pc, pr_opt, 0, pos_opt, m, graph,
                         slot_opt, bt, pool_epp);
        if (has_idx)
        {
            at::Tensor ikv = idx_buf_kv.value(), ig = idx_buf_gate.value();
            at::Tensor pi = pool_idx.value();
            c10::optional<at::Tensor> no_b = {};
            dsv4_compress_gr(s.mgc_idx.value().select(0, 0), s.mgc_idx.value().select(0, 1),
                             ikv, ig, idx_ovl, idx_ape.value(), idx_norm_w.value(), idx_eps,
                             idx_inv_freq.value(), pi, no_b, 0, pos_opt, m, graph,
                             slot_opt, bt, pool_epp);
        }
    }

    // Unified selection for ALL rows (dense jobs = degenerate bounded top-k). Per-job
    // T / q_pos0 / bound_max are pointer args into the state array; the per-row scan
    // bound in top-k makes the score buffer's stale region unreachable, no fill needed
    if (has_idx)
    {
        if (!use_fan2)
            exl3_gemm_gr(s.qres_st, idx_wq_b->trellis, s.qidx_st.value(), idx_wq_b->suh,
                         s.xh_b, idx_wq_b->svh, -1, idx_wq_b->mcg, idx_wq_b->mul1, 0, graph);
        {
            at::Tensor qi4 = s.qidx_st.value().view({B, S, index_n_heads, index_head_dim});
            at::Tensor qi_sl = qi4.narrow(3, index_head_dim - rope_dim, rope_dim);
            c10::optional<at::Tensor> no_k = {};
            c10::optional<at::Tensor> no_ko = {};
            c10::optional<at::Tensor> no_pid = {};
            c10::optional<at::Tensor> no_n = {};
            rope_gr
            (
                qi_sl, qi_sl, no_k, no_ko, inv_freq, 0, pos_opt, no_pid,
                ROPESTYLE_GPTJ, 1.0f, no_n, no_n, 1e-6f, 0.0f, 0.0f, 0,
                1, 0, graph
            );
        }
        hgemm_gr(s.x_st, idx_weights_w.value(), s.wts_st.value(), graph);  // full padded height
        {
            at::Tensor& sc = s.scores_st.value();
            std::vector<void*> args = {
                (void*) s.qidx_st.value().data_ptr(),
                (void*) s.wts_st.value().data_ptr(),
                (void*) pool_idx.value().data_ptr(),
                (void*) sc.data_ptr(),
                (void*) a_ec.data_ptr(),                    // T (per job)
                (void*) (uintptr_t) (uint32_t) R,
                (void*) a_pos.data_ptr(),                   // q_pos0 (per job)
                (void*) a_ec.data_ptr(),                    // bound_max (per job)
                (void*) bt_st.data_ptr(),                   // block table (row per job)
                (void*) (uintptr_t) (uint32_t) bt_st.size(1),
            };
            int gy = CEIL_DIVIDE((int) sc.size(1), 128);
            s.k_fewq->launch(R, gy, 1, args, stream);
        }
        c10::optional<at::Tensor> ec_opt = a_ec;
        dsa_topk_gr(s.scores_st.value(), s.indices_st.value(), index_topk, graph, ec_opt, S);
    }

    // Split + combine attention (MULTIROW AOT kernels, all bounds device-driven)
    int hb = CEIL_DIVIDE(num_q_heads, block_h);
    {
        std::vector<void*> args =
        {
            (void*) s.q_st.data_ptr(),
            (void*) ring.data_ptr(),
            (void*) s.kv_st.data_ptr(),
            (void*) (has_comp ? pool_c.value().data_ptr() : s.kv_st.data_ptr()),
            (void*) (has_comp ? pool_r.value().data_ptr() : s.kv_st.data_ptr()),
            (void*) bt_st.data_ptr(),
            (void*) (has_idx ? s.indices_st.value().data_ptr() : bt_st.data_ptr()),
            (void*) s.ws_ml.data_ptr(),
            (void*) s.ws_acc.data_ptr(),
            (void*) arr.select(0, 5).data_ptr(),            // k_len (per job)
            (void*) (uintptr_t) (uint32_t) window,
            (void*) a_ec.data_ptr(),                        // pool_len (per job)
            (void*) (uintptr_t) (uint32_t) bt_st.size(1),
            (void*) a_pos.data_ptr(),                       // q_pos0 (per job)
            (void*) a_floor.data_ptr(),                     // win_floor (per job)
            (void*) a_beg.data_ptr(),                       // ring_beg (per job)
            (void*) arr.select(0, 4).data_ptr(),            // slot_ids
            (void*) (uintptr_t) (uint32_t) ((int) ring.size(1) * D),
        };
        s.k_split->launch(R * hb, n_splits, 1, args, stream);
    }
    {
        std::vector<void*> args =
        {
            (void*) s.ws_ml.data_ptr(),
            (void*) s.ws_acc.data_ptr(),
            (void*) sinks.data_ptr(),
            (void*) inv_freq_neg.data_ptr(),
            (void*) s.attn_out_st.data_ptr(),
            (void*) a_pos.data_ptr(),                       // q_pos0 (per job)
            (void*) (uintptr_t) (uint32_t) R,
            (void*) (uintptr_t) (uint32_t) n_splits,
        };
        s.k_combine->launch(R * hb, CEIL_DIVIDE(D, 128), 1, args, stream);
    }

    // Grouped o_proj
    {
        at::Tensor A = s.attn_out_st;
        at::Tensor C = s.woa_c_st;
        at::Tensor ah = s.woa_xh.view({o_groups, R, A.size(2)});
        c10::optional<at::Tensor> mi = woa_indices;
        c10::optional<at::Tensor> no_w = {};
        exl3_mgemm_gr(A, woa_trellis, C, woa_suh, ah, woa_svh, mi, no_w, woa_k, -1, woa_mcg, woa_mul1, -1, -1, 0, graph, 1);
    }
    // Group-major -> row-major staging as ONE elementwise copy (static src/dst, nothing
    // to patch). ATen launches on torch's current stream, so during capture it must be
    // redirected to the capture stream explicitly
    {
        c10::optional<c10::cuda::CUDAStreamGuard> sg;
        if (graph)
            sg.emplace(c10::cuda::getStreamFromExternal(
                graph->capture_stream, s.woa_t_st.device().index()));
        s.woa_t_st.view({(int64_t) R, (int64_t) o_groups, (int64_t) o_lora})
            .copy_(s.woa_c_st.transpose(0, 1));
    }
    // wo_b as a plain gemm: the eager first instance of the graph protocol autotunes the
    // shape, capture records the tuned kernel (BC_MLP::run_bsz1 pattern)
    at::Tensor wo_b_xh = s.woa_xh.view({-1}).narrow(0, 0, s.woa_t_st.numel()).view({(int64_t) R, (int64_t) o_groups * o_lora});
    exl3_gemm_gr(s.woa_t_st, wo_b->trellis, s.y_st, wo_b->suh, wo_b_xh, wo_b->svh, -1, wo_b->mcg, wo_b->mul1, 0, graph);

    // SWA ring appends (batched, slot-indexed; shift/rebase handled host-side pre-replay)
    dsv4_ring_append_gr(s.kv_st, ring, a_pos, a_beg, graph, slot_opt);
}

at::Tensor BC_DSV4BatchAttention::run(const at::Tensor& x, int B, int S)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    Slot& s = slot(B, S);
    TORCH_CHECK(s.configured, "BC_DSV4BatchAttention: slot not configured");

    if (s.graph->disabled || (!s.graph->ready && !s.graph->ready_to_record))
    {
        run_gr(x, B, S, s, nullptr);
        s.graph->ready_to_record = true;
        return s.y_st;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_gr(x, B, S, s, s.graph.get());
        s.graph->capture_end();
    }

    // Everything is device-driven; the input pointer is the only patched parameter
    auto args = std::vector<PPTR> { PPTR(GP_copy2d_src, (void*) x.data_ptr()) };
    s.graph->launch(args, stream);
    return s.y_st;
}

at::Tensor BC_DSV4Attention::run(const at::Tensor& x, int pos, int win_beg, int regime)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int seq = (int) x.numel() / hidden_size;
    Slot& s = slot(seq, regime);
    TORCH_CHECK(s.configured, "BC_DSV4Attention: slot not configured");

    if (s.graph->disabled || (!s.graph->ready && !s.graph->ready_to_record))
    {
        run_gr(x, seq, regime, pos, win_beg, s, nullptr);
        s.graph->ready_to_record = true;
        return s.y_st;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_gr(x, seq, regime, pos, win_beg, s, s.graph.get());
        s.graph->capture_end();
    }

    int ec = comp_bc ? (pos + seq) / m : 0;
    int n_prev = std::min(std::min(window - 1, pos - win_beg), pos);
    int win_floor = pos - n_prev;

    // Params must be passed in RECORD (capture) order: x staging, then the indexer sites
    // (topk regime), then split, then combine
    auto args = std::vector<PPTR> { PPTR(GP_copy2d_src, (void*) x.data_ptr()) };
    if (regime == 1)
    {
        args.emplace_back(GP_dsa_T,         (void*) (uintptr_t) (uint32_t) ec);
        args.emplace_back(GP_dsa_qpos,      (void*) (uintptr_t) (uint32_t) pos);   // indexer
        args.emplace_back(GP_dsa_bound_max, (void*) (uintptr_t) (uint32_t) ec);
        args.emplace_back(GP_dsa_T,         (void*) (uintptr_t) (uint32_t) ec);    // topk
    }
    args.emplace_back(GP_dsa_pool_len,   (void*) (uintptr_t) (uint32_t) ec);
    args.emplace_back(GP_dsa_qpos,       (void*) (uintptr_t) (uint32_t) pos);      // split
    args.emplace_back(GP_dsa_win_floor,  (void*) (uintptr_t) (uint32_t) win_floor);
    args.emplace_back(GP_dsa_ring_beg,   (void*) (uintptr_t) (uint32_t) win_beg);
    args.emplace_back(GP_dsa_qpos,       (void*) (uintptr_t) (uint32_t) pos);      // combine
    s.graph->launch(args, stream);
    return s.y_st;
}
