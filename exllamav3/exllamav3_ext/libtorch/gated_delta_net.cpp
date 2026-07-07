#include <Python.h>
#include "gated_delta_net.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../gdn.cuh"

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

void BC_GatedDeltaNetSplit::run_bsz1_gr
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots,
    Graph* graph
)
{
    qkv_proj->run_gr(x, qkv, graph);
    z_proj->run_gr(x, z_flat, graph);
    gdn_ba_gemv_gr(x, ba_weight_t, ba_bias, ba, graph);

    gated_delta_net_fused_op_3_gr
    (
        qkv, ba,
        dt_bias, a_log,
        mixed_qkv, beta, g,
        beta_scale,
        graph
    );

    cuda_causal_conv1d_update_gr
    (
        mixed_qkv,
        conv_state,
        slots,
        conv1d_weight,
        conv1d_bias,
        conv_out,
        true,
        false,
        graph
    );

    cuda_recurrent_gated_delta_rule_gr
    (
        conv_out,
        g,
        beta,
        recurrent_state,
        core_attn_out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        slots,
        false,
        graph
    );

    norm->run_gr(core_attn_out, core_attn_out_f, z, graph);
    o_proj->run_gr(core_attn_out_f, y, graph);
}

void BC_GatedDeltaNetSplit::run_bsz1
(
    const at::Tensor& x,
    at::Tensor& y,
    at::Tensor& conv_state,
    at::Tensor& recurrent_state,
    const at::Tensor& slots
)
{
    py::gil_scoped_release release;
    c10::cuda::CUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (graph_bsz1.disabled || (!graph_bsz1.ready && !graph_bsz1.ready_to_record))
    {
        run_bsz1_gr(x, y, conv_state, recurrent_state, slots, nullptr);
        graph_bsz1.ready_to_record = true;
        graph_state_size = (int) conv_state.size(2);
        graph_hist_stride = (int) recurrent_state.size(1);
        return;
    }

    // The captured graph bakes in the state-buffer geometry (scalar kernel args can't be patched),
    // so a cache with different dimensions falls back to the eager path
    if ((int) conv_state.size(2) != graph_state_size ||
        (int) recurrent_state.size(1) != graph_hist_stride)
    {
        run_bsz1_gr(x, y, conv_state, recurrent_state, slots, nullptr);
        return;
    }

    if (!graph_bsz1.ready)
    {
        graph_bsz1.capture_begin();
        run_bsz1_gr(x, y, conv_state, recurrent_state, slots, &graph_bsz1);
        graph_bsz1.capture_end();
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
    graph_bsz1.launch(args, stream);
}
