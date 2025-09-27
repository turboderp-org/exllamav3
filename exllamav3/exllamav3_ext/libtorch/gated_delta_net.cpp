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
        v_head_dim
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
        v_head_dim
    );

    norm->run(core_attn_out, core_attn_out_f, z);
    o_proj->run(core_attn_out_f, y);
}
