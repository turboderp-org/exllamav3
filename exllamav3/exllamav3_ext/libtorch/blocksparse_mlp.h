#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "mlp.h"
#include "linear.h"
#include "../graph.cuh"

#define MAX_EXPERTS 512
#define TEMP_ROWS_GRAPH 32  // must match TEMP_ROWS_GRAPH in BlockSparseMLP.py

std::tuple<at::Tensor, at::Tensor> blocksparse_mlp_routing(
    int bsz,
    const py::object& cfg,
    const at::Tensor& y,
    const py::dict& params
);

struct BC_BlockSparseMLP
{
    at::Tensor yh2;
    at::Tensor yh;
    at::Tensor interm_gu;
    at::Tensor interm_g;
    at::Tensor interm_u;
    at::Tensor interm_a;
    at::Tensor interm_a2;
    at::Tensor out_d;
    at::Tensor out_d2;
    c10::optional<at::Tensor> out_d_sh;
    c10::optional<at::Tensor> z;
    at::Tensor dq_temp_up;
    at::Tensor dq_temp_down;
    int min_expert;
    int max_expert;
    at::Tensor gate_ptrs_trellis;   at::Tensor gate_ptrs_trellis_cpu;
    at::Tensor gate_ptrs_suh;       at::Tensor gate_ptrs_suh_cpu;
    at::Tensor gate_ptrs_svh;       at::Tensor gate_ptrs_svh_cpu;
    int gate_K;
    bool gate_mcg;
    bool gate_mul1;
    at::Tensor up_ptrs_trellis;     at::Tensor up_ptrs_trellis_cpu;
    at::Tensor up_ptrs_suh;         at::Tensor up_ptrs_suh_cpu;
    at::Tensor up_ptrs_svh;         at::Tensor up_ptrs_svh_cpu;
    int up_K;
    bool up_mcg;
    bool up_mul1;
    at::Tensor down_ptrs_trellis;   at::Tensor down_ptrs_trellis_cpu;
    at::Tensor down_ptrs_suh;       at::Tensor down_ptrs_suh_cpu;
    at::Tensor down_ptrs_svh;       at::Tensor down_ptrs_svh_cpu;
    int down_K;
    bool down_mcg;
    bool down_mul1;
    bool act_silu;
    bool act_gelu;
    std::shared_ptr<BC_GatedMLP> shared_experts;
    std::shared_ptr<BC_LinearFP16> shared_gate;
    float act_limit;
    std::vector<std::shared_ptr<BC_LinearEXL3>> gates;
    std::vector<std::shared_ptr<BC_LinearEXL3>> ups;
    std::vector<std::shared_ptr<BC_LinearEXL3>> downs;
    at::Tensor gu_trellis_ptr;
    at::Tensor gu_suh_ptr;
    at::Tensor gu_svh_ptr;

    int max_experts_per_token;
    int max_tokens_per_expert;
    std::vector<at::Tensor> interm_g_single;
    std::vector<at::Tensor> interm_u_single;
    std::vector<at::Tensor> interm_a_single;
    std::vector<at::Tensor> out_d_single;

    bool use_mgemm;

    Graph graph_bsz1;
    Graph graph_single[TEMP_ROWS_GRAPH];

    BC_BlockSparseMLP
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
    );

    void run_bsz1_gr
    (
        const at::Tensor& y,
        at::Tensor& selected_experts,
        at::Tensor& routing_weights,
        Graph* graph
    );

    void run_bsz1
    (
        const at::Tensor& y,
        at::Tensor& selected_experts,
        at::Tensor& routing_weights
    );

    void run_single_expert_gr
    (
        const at::Tensor& y,
        const int expert_idx,
        Graph* graph
    );

    void run_single_expert
    (
        const at::Tensor& y,
        const int expert_idx
    );

    void run_single_expert_dq
    (
        const at::Tensor& y,
        const int expert_idx,
        at::Tensor& yh,
        at::Tensor& interm,
        at::Tensor& interm_a,
        at::Tensor& out
    );
};