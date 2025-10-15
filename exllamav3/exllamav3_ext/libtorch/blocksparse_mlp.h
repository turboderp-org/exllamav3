#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "mlp.h"
#include "linear.h"
#include "../graph.cuh"

std::tuple<at::Tensor, at::Tensor> blocksparse_mlp_routing(
    int bsz,
    const py::object& cfg,
    const at::Tensor& y,
    const py::dict& params
);

struct BC_BlockSparseMLP
{
    at::Tensor yh;
    at::Tensor interm_g;
    at::Tensor interm_u;
    at::Tensor interm_a;
    at::Tensor out_d;
    c10::optional<at::Tensor> out_d_sh;
    c10::optional<at::Tensor> z;
    int min_expert;
    int max_expert;
    at::Tensor gate_ptrs_trellis;
    at::Tensor gate_ptrs_suh;
    at::Tensor gate_ptrs_svh;
    int gate_K;
    bool gate_mcg;
    bool gate_mul1;
    at::Tensor up_ptrs_trellis;
    at::Tensor up_ptrs_suh;
    at::Tensor up_ptrs_svh;
    int up_K;
    bool up_mcg;
    bool up_mul1;
    at::Tensor down_ptrs_trellis;
    at::Tensor down_ptrs_suh;
    at::Tensor down_ptrs_svh;
    int down_K;
    bool down_mcg;
    bool down_mul1;
    bool act_silu;
    bool act_gelu;
    std::shared_ptr<BC_GatedMLP> shared_experts;
    std::shared_ptr<BC_LinearFP16> shared_gate;

    Graph graph_bsz1;

    BC_BlockSparseMLP
    (
        at::Tensor _yh,
        at::Tensor _interm_g,
        at::Tensor _interm_u,
        at::Tensor _interm_a,
        at::Tensor _out_d,
        c10::optional<at::Tensor> _out_d_sh,
        c10::optional<at::Tensor> _z,
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
        std::shared_ptr<BC_LinearFP16> _shared_gate
    ) :
        yh                  (std::move(_yh)),
        interm_g            (std::move(_interm_g)),
        interm_u            (std::move(_interm_u)),
        interm_a            (std::move(_interm_a)),
        out_d               (std::move(_out_d)),
        out_d_sh            (std::move(_out_d_sh)),
        z                   (std::move(_z)),
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
        shared_gate         (_shared_gate)
    {}

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
};