#pragma once

#include <ATen/Tensor.h>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include "linear.h"

// Companion bound class for modules/dsv4.py DSV4Compressor (cached path, bsz 1): the whole
// forward -- wkv/wgate projections into fixed scratch, then the fused compress kernels
// (window pooling + norm + rope + pool store + snapshot + ring store) -- in one C++
// transition. Callable from other C++ code and graph-capturable: scratch is fixed-size for
// seqlen <= MAX_QLEN, and run_gr with a device position tensor records position-independent
// launches (the kernels read the position from memory and pad the window grid).

struct BC_DSV4Compressor
{
    static constexpr int MAX_QLEN = 32;

    // Exactly one of each pair is set (quantized vs unquantized checkpoints)
    std::shared_ptr<BC_LinearEXL3> wkv_exl3;
    std::shared_ptr<BC_LinearFP16> wkv_fp16;
    std::shared_ptr<BC_LinearEXL3> wgate_exl3;
    std::shared_ptr<BC_LinearFP16> wgate_fp16;

    at::Tensor ape;          // (m, W) float
    at::Tensor norm_w;       // (hd) half
    float rms_norm_eps;
    at::Tensor inv_freq;     // (rd / 2) float
    int m;

    at::Tensor kv_scratch;   // (MAX_QLEN, W) half
    at::Tensor gate_scratch; // (MAX_QLEN, W) half
    at::Tensor xh_scratch;   // (MAX_QLEN, hidden) half, exl3 gemm hadamard scratch

    // Batched wkv+wgate projection (one exl3_mgemm over 2 "experts"); set when both are
    // EXL3 with matching K/mcg/mul1, else the two-gemm fallback runs
    c10::optional<at::Tensor> mg_trellis;   // (2,) long: tensor data pointers
    c10::optional<at::Tensor> mg_suh;
    c10::optional<at::Tensor> mg_svh;
    c10::optional<at::Tensor> mg_indices;   // (1, 2) long

    BC_DSV4Compressor
    (
        std::shared_ptr<BC_LinearEXL3> _wkv_exl3,
        std::shared_ptr<BC_LinearFP16> _wkv_fp16,
        std::shared_ptr<BC_LinearEXL3> _wgate_exl3,
        std::shared_ptr<BC_LinearFP16> _wgate_fp16,
        at::Tensor _ape,
        at::Tensor _norm_w,
        float _rms_norm_eps,
        at::Tensor _inv_freq,
        int _m,
        at::Tensor _kv_scratch,
        at::Tensor _gate_scratch,
        at::Tensor _xh_scratch,
        c10::optional<at::Tensor> _mg_trellis,
        c10::optional<at::Tensor> _mg_suh,
        c10::optional<at::Tensor> _mg_svh,
        c10::optional<at::Tensor> _mg_indices
    ) :
        wkv_exl3        (_wkv_exl3),
        wkv_fp16        (_wkv_fp16),
        wgate_exl3      (_wgate_exl3),
        wgate_fp16      (_wgate_fp16),
        ape             (std::move(_ape)),
        norm_w          (std::move(_norm_w)),
        rms_norm_eps    (_rms_norm_eps),
        inv_freq        (std::move(_inv_freq)),
        m               (_m),
        kv_scratch      (std::move(_kv_scratch)),
        gate_scratch    (std::move(_gate_scratch)),
        xh_scratch      (std::move(_xh_scratch)),
        mg_trellis      (std::move(_mg_trellis)),
        mg_suh          (std::move(_mg_suh)),
        mg_svh          (std::move(_mg_svh)),
        mg_indices      (std::move(_mg_indices))
    {}

    void run_gr
    (
        const at::Tensor& x,                            // (seq, hidden) half, seq <= MAX_QLEN
        at::Tensor& ring_kv,                            // (buf_rows, W) half
        at::Tensor& ring_gate,                          // (buf_rows, W) half
        c10::optional<at::Tensor>& ovl,                 // (depth, 2, m, hd) float
        at::Tensor& dest_a,                             // (cap, Wa) half
        c10::optional<at::Tensor>& dest_b,              // (cap, hd - Wa) half
        int position,
        const c10::optional<at::Tensor>& position_tensor,
        const c10::optional<at::Tensor>& mg_c,          // (2, seq, W) preallocated scratch
                                                        // (graph capture: no allocations)
        class Graph* graph,
        bool proj_precomputed = false,                  // kv/gate already in mg_c (fan mgemm)
        const c10::optional<at::Tensor>& pool_bt = {},  // paged pools: job's block table row
        int pool_epp = 0
    );

    void run
    (
        const at::Tensor& x,
        at::Tensor& ring_kv,
        at::Tensor& ring_gate,
        c10::optional<at::Tensor>& ovl,
        at::Tensor& dest_a,
        c10::optional<at::Tensor>& dest_b,
        int position,
        const c10::optional<at::Tensor>& position_tensor,
        const c10::optional<at::Tensor>& mg_c,
        const c10::optional<at::Tensor>& pool_bt,
        int pool_epp
    );
};
