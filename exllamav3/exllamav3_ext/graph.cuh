#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

using PPTR = std::tuple<int, void*>;

enum GraphedParams
{
    GP_end,

    GP_gemm_A,
    GP_gemm_B_trellis,
    GP_gemm_C,
    GP_gemm_B_suh,
    GP_gemm_A_had,
    GP_gemm_B_svh,

    GP_mgemm,
    GP_mgemm_A,
    GP_mgemm_C,
    GP_mgemm_indices,
    GP_mgemm_weights,

    GP_silu_mul_x,
    GP_silu_mul_y,
    GP_silu_mul_z,

    GP_gelu_mul_x,
    GP_gelu_mul_y,
    GP_gelu_mul_z,

    GP_relu2_mul_x,
    GP_relu2_mul_y,
    GP_relu2_mul_z,

    GP_xielu_x,
    GP_xielu_y,

    GP_add_sigmoid_gate,
    GP_add_sigmoid_gate_x,
    GP_add_sigmoid_gate_y,
    GP_add_sigmoid_gate_z,

    GP_add_sigmoid_gate_proj,
    GP_add_sigmoid_gate_proj_x,
    GP_add_sigmoid_gate_proj_y,
    GP_add_sigmoid_gate_proj_z,

    GP_add_x,
    GP_add_y,
    GP_add_z,

    GP_gdn_ba_x,
    GP_conv1d_state,
    GP_conv1d_slots,
    GP_gdn_rule_state,
    GP_gdn_rule_slots,

    GP_rope_inv_freq,
    GP_rope_position,
    GP_rope_positions,
    GP_rope_position_ids,
    GP_rope_pid_stride,

    GP_qcache_seqlens,
    GP_qcache_block_table,
    GP_qcache_blocks_per_seq,

    GP_attn_seqlens,
    GP_attn_block_table,
    GP_attn_num_pages,
    GP_attn_split_len,
    GP_attn_num_splits
};

class Graph
{
public:
    cudaStream_t capture_stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    // (kernel, param_id, param_offset, value size in bytes). Scalar args (4-byte ints) can be
    // patched too: kernelParams[i] points to each argument's own storage, but adjacent args may
    // be packed, so only `size` bytes are written
    std::vector<std::tuple<void*, int, int, int>> graph_sites;
    std::vector<std::tuple<int, int, int, int>> graph_node_sites;

    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> node_params;
    // Kernel nodes captured from driver-API launches (Triton cubins) cannot be read through the
    // runtime API ("invalid device function"); their params live here instead
    std::vector<CUDA_KERNEL_NODE_PARAMS> node_params_drv;
    std::vector<char> node_is_driver;
    std::vector<void*> current_values;
    std::vector<bool> node_needs_update;

    bool need_cublas;
    bool ready;
    bool ready_to_record;
    bool disabled;

    Graph();
    ~Graph();

    cudaStream_t capture_begin();
    void capture_end();

    void record_param(void* kernel, int param_id, int param_offset, int size = 8);
    void launch(std::vector<PPTR> params, cudaStream_t stream);

    void inspect_graph();
};

