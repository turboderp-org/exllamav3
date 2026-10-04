#pragma once

// =============================================================================
// Multi-row GEMV for RDNA, m = 2..8 -- host interface
// =============================================================================
//
// See exl3_gemv_multirow_rdna.cu for the design. Both entry points return
// false, having launched nothing, when the call must fall through to the
// cooperative GEMM / mgemm.

#include <cuda_fp16.h>
#include <cstdint>

class Graph;

// EXL3_GEMV_MAX_M (default 8, clamp 1..8): the largest m the multi-row GEMV
// takes; 1 switches the path off. Re-read per call.
int exl3_gemv_max_m();

// EXL3_ROCM_HALF_GEMV (default on; =0 sends half-integer bitrates back to the
// cooperative GEMM / mgemm). Callers pass K = EXL3_HALF_BITS(K) (16 + K, see
// exl3_gemv_tiles_rdna.cuh) for a K + 0.5 bpw mul1 tensor when this is on.
bool exl3_rocm_half_gemv_enabled();
#ifndef EXL3_HALF_BITS
#define EXL3_HALF_BITS(ka) (16 + (ka))   // pseudo width of a ka + 0.5 bpw tensor (exl3_gemv_tiles_rdna.cuh)
#endif

// Allocates the per-device parameter block outside capture (hipMalloc would
// invalidate an active capture). Called from the m == 1 paths' non-graph
// sites, which every BC module runs eagerly before it captures.
void exl3_gemv_multirow_prewarm(int device);

// Single matrix: the exl3_gemm contract (A m x K, B trellis, C m x N, suh,
// A_had scratch of m x K, svh). Records the six GP_gemm_* sites when graph.
bool exl3_gemv_multirow_try_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    Graph* graph
);

// Multi matrix: the exl3_mgemm contract at m > 1 without routing weights
// (A bszm_in x m x K, C bszm_out x m x N, A_had bszm x m x K, per-matrix
// pointer tables, optional indices / expert-range packing / per-matrix width
// and output lists). Records the four GP_mgemm_* sites when graph.
bool exl3_mgemv_multirow_try_launch
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list,
    void** c_list,
    int device,
    cudaStream_t stream,
    Graph* graph
);

// Fused bsz <= 8 MoE decode (gate|up as one 2S-slot GEMV, activation folded into the down projection's input
// rotation, weighted grouped down GEMV), called by BC_BlockSparseMLP::run_bszN on ROCm
#include <ATen/ATen.h>
bool exl3_rocm_moe_decode_fits(int S, int Hi, int I, int Ho, int K_gu, int cb_gu, int K_d, int cb_d);
void exl3_rocm_moe_decode
(
    const at::Tensor& y,
    const at::Tensor& sel,
    const at::Tensor& weights,
    const at::Tensor& gu_trellis,
    const at::Tensor& gu_suh,
    const at::Tensor& gu_svh,
    const at::Tensor& d_trellis,
    const at::Tensor& d_suh,
    const at::Tensor& d_svh,
    at::Tensor& yh,
    at::Tensor& gu,
    at::Tensor& a_had,
    at::Tensor& out,
    int64_t K_gu,
    int64_t cb_gu,
    int64_t K_d,
    int64_t cb_d,
    double act_limit
);
