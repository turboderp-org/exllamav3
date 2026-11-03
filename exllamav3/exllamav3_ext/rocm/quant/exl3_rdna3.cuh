#pragma once

// EXL3 small-m matmul for AMD RDNA3 (gfx11, wave32), dispatched by exl3_gemm ahead of the RDNA GEMVs for every m up
// to the reconstruct threshold. Adapted from exllamav3-rocm (github.com/phoenixhaxor/exllamav3-rocm, MIT).
//
// - a small kernel applies suh and the 128-point Hadamard to x once per matmul, writing fp16 in the main kernel's
//   LDS pair layout plus per-128-block sums of the rounded values
// - each block owns 128 output columns (8 n-tiles, one wave per tile) of up to MR rows and a k-range (split)
// - trellis words stream straight to registers (buffer loads, prefetch ring); mul1 at 2..6 bits decodes raw
//   (byte sums of the multiplied window, codebook affine map applied to the dot products), the rest through
//   dq_dispatch; products accumulate in fp32 via v_dot2_f32_f16
// - k-splits reduce through a fp32 workspace; the last block per column group (atomic counter) sums the
//   partials, applies the output Hadamard and svh and writes C
//
// exl3_mgemm runs shared-input bundles (gate/up, sliced Q/K/V) as one input-transform launch plus one matmul launch,
// one blockIdx.z per entry. EXL3_RDNA3_GEMM=0 disables both.

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define EXL3_RDNA3_NT 8
#define EXL3_RDNA3_THREADS (EXL3_RDNA3_NT * 32)
#define EXL3_RDNA3_WS_BYTES (64ull << 20)
#define EXL3_RDNA3_MAX_COUNTERS (1 << 16)
#define EXL3_RDNA3_XH_BYTES (16ull << 20)
#define EXL3_RDNA3_XCS_FLOATS (1 << 18)

// Multi-matrix mode (blockIdx.z = entry): per-entry trellis / svh / output pointers, output row stride and input
// source. A null b table selects plain single-matrix mode
struct Exl3Rdna3MTab
{
    const uint64_t* b;         // trellis pointers (pre-offset to the entry's first column)
    const uint64_t* svh;       // svh pointers (pre-offset)
    const uint64_t* c;         // output pointers, or null: C + e * size_m * size_n
    const int* n_stride;       // row stride of trellis and output, or null: size_n
    const int* src;            // input source (transformed-input slab), or null: e
};

// Elementwise prologue of the input transform (exl3_gemm_silu_gr / exl3_gemm_gnorm_gr, exl3_rdna3_fused.cu), in
// place of a separate kernel in front of the matmul, same rounding:
// - A_up given: the input is silu(A) * A_up (act_mul_kernel_h), or with GN_UP_SIGMOID A * sigmoid(A_up)
//   (mul_sigmoid_kernel_h, attention output gate)
// - GN_ACTIVE: gated RMSNorm, one head per 128-element block, y = norm(A) * (w + bias) * act(g) as gated_rms_norm
//   computes it; A is bf16; GN_W_BF16 / GN_G_BF16 select bf16 weight / gate (else fp32), GN_SIGMOID the gate's
//   activation (else silu)
enum { GN_ACTIVE = 1, GN_W_BF16 = 2, GN_G_BF16 = 4, GN_SIGMOID = 8, GN_UP_SIGMOID = 16 };
struct Exl3Rdna3GNorm
{
    const void* w;
    const void* g;
    float eps;
    float bias;
    int flags;
};

typedef void (*fp_exl3_rdna3_kernel)
(
    const uint2* __restrict__ xh,
    const uint32_t* __restrict__ B,
    void* __restrict__ C,
    int size_m,
    int size_k,
    int size_n,
    int* __restrict__ counters,
    const float* __restrict__ xcs,
    float* __restrict__ ws,
    const half* __restrict__ svh,
    int splits,
    int ks_per_split,
    Exl3Rdna3MTab mt
);

// mr_idx: rows per pass 1, 2, 4, 8, 16, 3, 5, 6, 12
fp_exl3_rdna3_kernel exl3_rdna3_get_k1(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k2(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k3(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k4(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k5(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k6(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k7(bool half_k, int cb, int mr_idx, bool c_fp32);
fp_exl3_rdna3_kernel exl3_rdna3_get_k8(bool half_k, int cb, int mr_idx, bool c_fp32);

class Graph;

// Returns true if the matmul was launched
bool exl3_rdna3_gemm
(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const half* suh,
    const half* svh,
    int device,
    cudaStream_t stream,
    Graph* graph,
    const half* A_up = nullptr,
    const Exl3Rdna3GNorm* gn = nullptr
);

// Several matrices sharing one input (MultiLinear / SlicedMultiLinear tables), one launch pair
bool exl3_rdna3_mgemm
(
    const half* A,
    const uint64_t* b_tab,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const uint64_t* suh_tab,
    const uint64_t* svh_tab,
    const uint64_t* c_tab,
    const int* n_stride_tab,
    const int* src_tab,
    int num_entries,
    int num_src,
    int device,
    cudaStream_t stream,
    Graph* graph
);

void exl3_rdna3_prepare(int device);
bool exl3_rdna3_enabled();
