#include <Python.h>
#include "dsv4_compressor.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../dsv4_compress.cuh"
#include "../graph.cuh"

void BC_DSV4Compressor::run_gr
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
    Graph* graph,
    bool proj_precomputed,
    const c10::optional<at::Tensor>& pool_bt,
    int pool_epp
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());

    int seq = x.size(0);
    TORCH_CHECK(seq <= MAX_QLEN, "BC_DSV4Compressor: seq > MAX_QLEN");

    // Projections into scratch. Preferred path: ONE exl3_mgemm over 2 "experts" (wkv,
    // wgate share input and shape); fallbacks: two exl3 gemms (mixed K), two hgemms (fp16).
    // proj_precomputed: the caller's fan mgemm already filled mg_c
    at::Tensor kv, gate;
    if (proj_precomputed)
    {
        kv = mg_c.value().select(0, 0);
        gate = mg_c.value().select(0, 1);
    }
    else if (mg_trellis)
    {
        int W = kv_scratch.size(1);
        at::Tensor A = x.unsqueeze(0);
        // The mgemm kernel writes one hadamard-transformed input slab PER EXPERT
        // (A_had + j * m * k): two slabs here, not a broadcast view
        TORCH_CHECK(xh_scratch.size(0) >= 2 * seq, "BC_DSV4Compressor: xh scratch too small");
        at::Tensor A_had = xh_scratch.narrow(0, 0, 2 * seq).view({2, seq, x.size(1)});
        at::Tensor C = mg_c ? mg_c.value() : at::empty({2, seq, W}, kv_scratch.options());
        c10::optional<at::Tensor> no_weights = {};
        exl3_mgemm_gr(A, mg_trellis.value(), C, mg_suh.value(), A_had, mg_svh.value(),
                      mg_indices, no_weights, wkv_exl3->K, -1,
                      wkv_exl3->mcg, wkv_exl3->mul1, -1, -1, 0, graph, 1);
        kv = C.select(0, 0);
        gate = C.select(0, 1);
    }
    else if (wkv_exl3)
    {
        kv = kv_scratch.narrow(0, 0, seq);
        gate = gate_scratch.narrow(0, 0, seq);
        at::Tensor xh = xh_scratch.narrow(0, 0, seq);
        exl3_gemm_gr(x, wkv_exl3->trellis, kv, wkv_exl3->suh, xh, wkv_exl3->svh,
                     -1, wkv_exl3->mcg, wkv_exl3->mul1, 0, graph);
        exl3_gemm_gr(x, wgate_exl3->trellis, gate, wgate_exl3->suh, xh, wgate_exl3->svh,
                     -1, wgate_exl3->mcg, wgate_exl3->mul1, 0, graph);
    }
    else
    {
        kv = kv_scratch.narrow(0, 0, seq);
        gate = gate_scratch.narrow(0, 0, seq);
        hgemm_gr(x, wkv_fp16->weight, kv, graph);
        hgemm_gr(x, wgate_fp16->weight, gate, graph);
    }

    dsv4_compress_gr(kv, gate, ring_kv, ring_gate, ovl, ape, norm_w, rms_norm_eps,
                     inv_freq, dest_a, dest_b, position, position_tensor, m, graph,
                     {}, pool_bt, pool_epp);
}

void BC_DSV4Compressor::run
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
)
{
    run_gr(x, ring_kv, ring_gate, ovl, dest_a, dest_b, position, position_tensor, mg_c,
           nullptr, false, pool_bt, pool_epp);
}
