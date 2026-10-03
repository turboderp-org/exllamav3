#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "../graph.cuh"

// Fused int8-activation GEMV for mul1 (cb 2) tensors: single cooperative launch covering input
// Hadamard, activation quantization, dp4a GEMV and output Hadamard. EXL3_INT8_GEMV=2 (default) is
// the plain int8 mode, =1 the error-feedback residual mode (~15-16 bit effective activation
// precision), =0 disables the path. Tensors outside the mul1 codebook are unaffected. Arbitrary m is handled as
// sequential row passes within the launch, so intended for the small-m regime.
// See benchmarks/exl3_m1_bench (variants 15-18) for derivation, microbenchmarks and profiling.

bool exl3_gemv_int8_enabled();

// Kill switch for the multi-matrix/sliced path only (EXL3_INT8_MSQ=0)
bool exl3_gemv_int8_msq_enabled();

// Highest K the int8 path accepts on `device` (per-arch, EXL3_INT8_GEMV_MAX_K override). Also
// bound to Python so the mgemm fusion heuristic can mirror the gate
int exl3_gemv_int8_max_k(int device);

// Returns true if the operation was handled (false -> caller should fall through to the regular kernel)
// num_sms: the caller's SM count (exl3_gemm's force_num_sms override, 0 = hardware default). The sq
// path sizes its grid and its slice height from it, so a non-zero value acts as a grid multiplier
// for launch-geometry sweeps without touching code.
bool exl3_gemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int num_sms,
    cudaStream_t stream,
    Graph* graph
);

// Multi-matrix/sliced variant for the mgemm entry (m == 1, bszm_in == 1, unfiltered). Takes the
// caller's cooked pointer arguments; the kernel signature matches exl3_mgemm_kernel so graph
// parameter recording is identical. Returns true if the operation was handled.
bool exl3_gemv_int8_msq
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    int size_m,
    int size_k,
    int size_n,
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list_ptr,
    void** c_list_ptr,
    const int* n_stride_list_ptr,
    const int* had_src_list_ptr,
    int num_had_src,
    int K,
    bool c_fp32,
    int device,
    int num_sms,
    cudaStream_t stream,
    Graph* graph
);
