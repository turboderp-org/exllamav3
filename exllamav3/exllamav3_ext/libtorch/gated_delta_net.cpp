#include <Python.h>
#include "gated_delta_net.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../gdn.cuh"
#include "../add.cuh"

using namespace torch::indexing;

at::Tensor BC_GatedDeltaNet::run_bsz1_a
(
    const at::Tensor& x
)
{
    py::gil_scoped_release _;

    qkvz_proj->run(x, qkvz);
    ba_proj->run(x, ba);

    gated_delta_net_fused_op
    (
        qkvz, ba,
        dt_bias, a_log,
        mixed_qkv, z, beta, g,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        beta_scale
    );

    return mixed_qkv;
}

void BC_GatedDeltaNet::run_bsz1_b
(
    at::Tensor& mixed_qkv,
    at::Tensor& y,
    at::Tensor& recurrent_state
)
{
    cuda_recurrent_gated_delta_rule
    (
        mixed_qkv.transpose(1, 2),
        g,
        beta,
        recurrent_state,
        core_attn_out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        c10::nullopt,
        false
    );

    norm->run(core_attn_out, core_attn_out_f, z);
    o_proj->run(core_attn_out_f, y);
}

bool BC_GatedDeltaNetSplit::needs_configure(int bsz, int seqlen, bool history)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= seqlen && seqlen <= MAX_QLEN,
                "BC_GatedDeltaNetSplit: shape out of range");
    return !slot(bsz, seqlen, history).configured;
}

void BC_GatedDeltaNetSplit::configure_slot
(
    int bsz,
    int seqlen,
    bool history,
    at::Tensor qkv,
    at::Tensor z,
    at::Tensor ba,
    at::Tensor beta,
    at::Tensor g,
    at::Tensor mixed_qkv,
    at::Tensor conv_out,
    at::Tensor core_attn_out,
    at::Tensor core_attn_out_f,
    at::Tensor qkv_xh,
    at::Tensor z_xh,
    at::Tensor o_xh
)
{
    Slot& s = slot(bsz, seqlen, history);

    s.qkv             = std::move(qkv);
    s.z               = std::move(z);
    s.ba              = std::move(ba);
    s.beta            = std::move(beta);
    s.g               = std::move(g);
    s.mixed_qkv       = std::move(mixed_qkv);
    s.conv_out        = std::move(conv_out);
    s.core_attn_out   = std::move(core_attn_out);
    s.core_attn_out_f = std::move(core_attn_out_f);
    s.qkv_xh          = std::move(qkv_xh);
    s.z_xh            = std::move(z_xh);
    s.o_xh            = std::move(o_xh);
    s.z_flat = s.z.view({bsz, seqlen, -1});

    TORCH_CHECK(s.qkv.is_contiguous() && s.z.is_contiguous() && s.ba.is_contiguous() &&
                s.beta.is_contiguous() && s.g.is_contiguous() && s.mixed_qkv.is_contiguous() &&
                s.conv_out.is_contiguous() && s.core_attn_out.is_contiguous() &&
                s.core_attn_out_f.is_contiguous(),
                "BC_GatedDeltaNetSplit: statics must be contiguous");

    s.graph = std::make_unique<Graph>();
    s.configured = true;
}

void BC_GatedDeltaNetSplit::run_bszN_gr
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history,
    Slot& s,
    Graph* graph
)
{
    // qkv/z projections: bypass BC_LinearEXL3::run_gr, which hard-refuses graph capture above 1
    // row -- call exl3_gemm_gr directly with this slot's own xh scratch instead, exactly like
    // BC_GatedMLP::run_bszN_gr does for shared-expert projections
    exl3_gemm_gr(x, qkv_proj->trellis, s.qkv, qkv_proj->suh, s.qkv_xh, qkv_proj->svh, -1, qkv_proj->mcg, qkv_proj->mul1, 0, graph);
    if (qkv_proj->bias)
        add_gr(s.qkv, qkv_proj->bias.value(), s.qkv, graph);
    exl3_gemm_gr(x, z_proj->trellis, s.z_flat, z_proj->suh, s.z_xh, z_proj->svh, -1, z_proj->mcg, z_proj->mul1, 0, graph);
    if (z_proj->bias)
        add_gr(s.z_flat, z_proj->bias.value(), s.z_flat, graph);

    gdn_ba_gemv_gr(x, ba_weight_t, ba_bias, s.ba, graph);

    gated_delta_net_fused_op_3_gr
    (
        s.qkv, s.ba,
        dt_bias, a_log,
        s.mixed_qkv, s.beta, s.g,
        beta_scale,
        graph
    );

    cuda_causal_conv1d_update_gr
    (
        s.mixed_qkv,
        conv_state,
        slots,
        conv1d_weight,
        conv1d_bias,
        s.conv_out,
        true,
        history,
        graph
    );

    cuda_recurrent_gated_delta_rule_gr
    (
        s.conv_out,
        s.g,
        s.beta,
        recurrent_state,
        s.core_attn_out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        slots,
        history,
        graph
    );

    norm->run_gr(s.core_attn_out, s.core_attn_out_f, s.z, graph);

    exl3_gemm_gr(s.core_attn_out_f, o_proj->trellis, y, o_proj->suh, s.o_xh, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
    if (o_proj->bias)
        add_gr(y, o_proj->bias.value(), y, graph);
}

void BC_GatedDeltaNetSplit::run_bszN
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = (int) x.size(0);
    int seqlen = (int) x.size(1);
    TORCH_CHECK(bsz >= 1 && bsz <= MAX_BSZ && seqlen >= 1 && seqlen <= MAX_QLEN,
                "BC_GatedDeltaNetSplit::run_bszN: shape out of range");
    Slot& s = slot(bsz, seqlen, history);
    TORCH_CHECK(s.configured, "BC_GatedDeltaNetSplit::run_bszN: slot not configured");

    if (s.graph->disabled || (!s.graph->ready && !s.graph->ready_to_record))
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        s.graph->ready_to_record = true;
        graph_state_size = (int) conv_state.size(2);
        graph_hist_stride = (int) recurrent_state.size(1);
        return;
    }

    // The captured graph bakes in the state-buffer geometry (scalar kernel args can't be patched),
    // so a cache with different dimensions falls back to the eager path
    if ((int) conv_state.size(2) != graph_state_size ||
        (int) recurrent_state.size(1) != graph_hist_stride)
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, s.graph.get());
        s.graph->capture_end();
    }

    auto args = std::vector<PPTR>
    {
        PPTR(GP_gemm_A,         (void*) x.data_ptr()),          // qkv_proj input
        PPTR(GP_gemm_A,         (void*) x.data_ptr()),          // z_proj input
        PPTR(GP_gdn_ba_x,       (void*) x.data_ptr()),
        PPTR(GP_conv1d_state,   (void*) conv_state.data_ptr()),
        PPTR(GP_conv1d_slots,   (void*) slots.data_ptr()),
        PPTR(GP_gdn_rule_state, (void*) recurrent_state.data_ptr()),
        PPTR(GP_gdn_rule_slots, (void*) slots.data_ptr()),
        PPTR(GP_gemm_C,         (void*) y.data_ptr())           // o_proj output
    };
    s.graph->launch(args, stream);
}

bool BC_Mamba2::needs_configure(int bsz, int seqlen, bool history)
{
    TORCH_CHECK(1 <= bsz && bsz <= MAX_BSZ && 1 <= seqlen && seqlen <= MAX_QLEN,
                "BC_Mamba2: shape out of range");
    return !slot(bsz, seqlen, history).configured;
}

void BC_Mamba2::configure_slot
(
    int bsz,
    int seqlen,
    bool history,
    c10::optional<at::Tensor> xp,
    at::Tensor proj,
    at::Tensor mixed_xbc,
    at::Tensor dt,
    at::Tensor g,
    at::Tensor z_gate,
    at::Tensor conv_out,
    at::Tensor core_attn_out,
    at::Tensor core_attn_out_f,
    c10::optional<at::Tensor> yp,
    at::Tensor in_xh,
    at::Tensor o_xh
)
{
    Slot& s = slot(bsz, seqlen, history);

    s.xp              = std::move(xp);
    s.proj            = std::move(proj);
    s.mixed_xbc       = std::move(mixed_xbc);
    s.dt              = std::move(dt);
    s.g               = std::move(g);
    s.conv_out        = std::move(conv_out);
    s.core_attn_out   = std::move(core_attn_out);
    s.core_attn_out_f = std::move(core_attn_out_f);
    s.yp              = std::move(yp);
    s.in_xh           = std::move(in_xh);
    s.o_xh            = std::move(o_xh);

    TORCH_CHECK(padded_in == s.xp.has_value(), "BC_Mamba2: xp presence must match padded_in");
    TORCH_CHECK(padded_out == s.yp.has_value(), "BC_Mamba2: yp presence must match padded_out");

    TORCH_CHECK(s.proj.is_contiguous() && s.mixed_xbc.is_contiguous() && s.dt.is_contiguous() &&
                s.g.is_contiguous() && s.conv_out.is_contiguous() && s.core_attn_out.is_contiguous() &&
                s.core_attn_out_f.is_contiguous() && z_gate.is_contiguous(),
                "BC_Mamba2: statics must be contiguous");

    int gs = v_dim / num_k_heads;
    s.z_gate = z_gate.view({bsz, seqlen, num_k_heads, gs});
    s.core_g = s.core_attn_out.view({bsz, seqlen, num_k_heads, gs});
    s.core_f_g = s.core_attn_out_f.view({bsz, seqlen, num_k_heads, gs});

    s.graph = std::make_unique<Graph>();
    s.configured = true;
}

void BC_Mamba2::run_bszN_gr
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history,
    Slot& s,
    Graph* graph
)
{
    int bsz = (int) x.size(0);
    int seqlen = (int) x.size(1);
    int R = bsz * seqlen;

    // Padded in_proj K: stage x through the zero-padded static (pad columns are zeroed at
    // configure time and only the exact width is ever copied in)
    at::Tensor x_in = x;
    if (s.xp)
    {
        at::Tensor x2 = x.reshape({R, hidden_size});
        at::Tensor xp2 = s.xp.value();
        copy2d_gr(x2, xp2, graph);
        x_in = xp2.view({bsz, seqlen, -1});
    }

    // in_proj: bypass BC_LinearEXL3::run_gr for the same reason as BC_GatedDeltaNetSplit above
    exl3_gemm_gr(x_in, in_proj->trellis, s.proj, in_proj->suh, s.in_xh, in_proj->svh, -1, in_proj->mcg, in_proj->mul1, 0, graph);
    if (in_proj->bias)
        add_gr(s.proj, in_proj->bias.value(), s.proj, graph);

    mamba2_fused_op_gr(s.proj, s.mixed_xbc, s.dt, s.g, s.z_gate, dt_bias, a_log, v_dim, dt_first, dt_min, dt_max, graph);

    cuda_causal_conv1d_update_gr
    (
        s.mixed_xbc,
        conv_state,
        slots,
        conv1d_weight,
        conv1d_bias,
        s.conv_out,
        true,
        history,
        graph
    );

    cuda_recurrent_mamba2_gr
    (
        s.conv_out,
        s.g,
        s.dt,
        d_skip,
        recurrent_state,
        s.core_attn_out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        slots,
        history,
        graph
    );

    norm->run_gr(s.core_g, s.core_f_g, s.z_gate, graph);

    // Padded o_proj N: the GEMM writes the padded static, then the exact width copies out to y
    if (s.yp)
    {
        at::Tensor yp3 = s.yp.value().view({bsz, seqlen, -1});
        exl3_gemm_gr(s.core_attn_out_f, o_proj->trellis, yp3, o_proj->suh, s.o_xh, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
        if (o_proj->bias)
            add_gr(yp3, o_proj->bias.value(), yp3, graph);
        at::Tensor y2 = y.reshape({R, hidden_size});
        at::Tensor yp2 = s.yp.value();
        copy2d_gr(yp2, y2, graph);
    }
    else
    {
        exl3_gemm_gr(s.core_attn_out_f, o_proj->trellis, y, o_proj->suh, s.o_xh, o_proj->svh, -1, o_proj->mcg, o_proj->mul1, 0, graph);
        if (o_proj->bias)
            add_gr(y, o_proj->bias.value(), y, graph);
    }
}

void BC_Mamba2::run_bszN
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    bool history
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = (int) x.size(0);
    int seqlen = (int) x.size(1);
    TORCH_CHECK(bsz >= 1 && bsz <= MAX_BSZ && seqlen >= 1 && seqlen <= MAX_QLEN,
                "BC_Mamba2::run_bszN: shape out of range");
    Slot& s = slot(bsz, seqlen, history);
    TORCH_CHECK(s.configured, "BC_Mamba2::run_bszN: slot not configured");

    if (s.graph->disabled || (!s.graph->ready && !s.graph->ready_to_record))
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        s.graph->ready_to_record = true;
        graph_state_size = (int) conv_state.size(2);
        graph_hist_stride = (int) recurrent_state.size(1);
        return;
    }

    // The captured graph bakes in the state-buffer geometry (scalar kernel args can't be patched),
    // so a cache with different dimensions falls back to the eager path
    if ((int) conv_state.size(2) != graph_state_size ||
        (int) recurrent_state.size(1) != graph_hist_stride)
    {
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, nullptr);
        return;
    }

    if (!s.graph->ready)
    {
        s.graph->capture_begin();
        run_bszN_gr(x, y, conv_state, recurrent_state, slots, history, s, s.graph.get());
        s.graph->capture_end();
    }

    std::vector<PPTR> args;
    args.reserve(8);
    if (s.xp)
        args.emplace_back(GP_copy2d_src, (void*) x.data_ptr());
    else
        args.emplace_back(GP_gemm_A, (void*) x.data_ptr());     // in_proj input
    args.emplace_back(GP_conv1d_state,   (void*) conv_state.data_ptr());
    args.emplace_back(GP_conv1d_slots,   (void*) slots.data_ptr());
    args.emplace_back(GP_gdn_rule_state, (void*) recurrent_state.data_ptr());
    args.emplace_back(GP_gdn_rule_slots, (void*) slots.data_ptr());
    if (s.yp)
        args.emplace_back(GP_copy2d_dst, (void*) y.data_ptr());
    else
        args.emplace_back(GP_gemm_C, (void*) y.data_ptr());     // o_proj output
    s.graph->launch(args, stream);
}
