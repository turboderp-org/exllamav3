#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "../arch.cuh"

// Shape table and kernel instance lists for the EXL3 GEMM (exl3_gemm_kernel.cuh), shared by both
// backends. Only the shape table itself is per backend: the CUDA inner (exl3_gemm_inner.cuh, MMA) and
// the RDNA inner (rocm/quant/exl3_gemm_inner_rdna.cuh, WMMA) stage different tiles against different
// shared memory budgets, so each lists the tiles it instantiates. Everything built on the table (the
// instance arrays, the per-(K, cb) translation units under comp_units/, the smem accounting the kernels
// static_assert against and the selectors in exl3_kernel_map.cu) is derived from it.

// Thread count of one row group of the GEMM block (blockDim is this times TILESIZE_K / 16)
#define EXL3_GEMM_BASE_THREADS 256

// Dynamic shared memory the kernel instantiations are compiled against: the static_asserts in the
// inner kernels must accept every shape in the table below. What a launch actually requests is
// the per-device value (DevCtx::get_smem_request), which is this capped by the device's limit
#define SMEM_MAX EXL3_SMEM_MAX_DEFAULT

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits, bool multi);
int exl3_gemm_num_kernel_shapes();
bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int bits, bool half_k = false);
// Dynamic shared memory (bytes) a (shape, bitrate) instantiation requests at launch
int exl3_gemm_shape_smem(int shape_idx, int bits, bool half_k);
// Throws if the shape does not fit the current device; for paths that bypass shape_compat
void exl3_gemm_check_smem(int shape_idx, int bits, bool half_k, const char* who);

// bits: integer part of the bitrate; half: bitrate is bits + 0.5 (mul1 codebook only, 16 * bits + 8 uint16 per tile)
#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool half_k, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

// Dual-matrix GEMV variant: EXL3_GEMM_ARGS + a second matrix slot
// (B2/C2/suh2/A_had2/svh2). Fuses two back-to-back GEMV calls into
// one launch — see docs/sm70_dual_gemv_design.md.
#define EXL3_GEMM_ARGS_DUAL \
    EXL3_GEMM_ARGS, \
    const uint16_t* __restrict__ B2, \
    void* __restrict__ C2, \
    const half* __restrict__ suh2, \
    half* __restrict__ A_had2, \
    const half* __restrict__ svh2

#define EXL3_MGEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t** __restrict__ B_list, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half** __restrict__ suh_list, \
    half* __restrict__ A_had, \
    const half** __restrict__ svh_list, \
    int64_t* B_indices, \
    half* B_weights, \
    const int bszm_in, \
    const int bszm_out, \
    const int min_index, \
    const int max_index, \
    const int num_tokens, \
    const int* __restrict__ size_n_list, \
    void** __restrict__ C_list, \
    const int* __restrict__ n_stride_list, \
    const int* __restrict__ had_src_list, \
    const int num_had_src

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);

// K-split variant: EXL3_GEMM_ARGS + fp32 partial-sum workspace.
// CFG==2 kernels write per-k-chunk partials to the workspace and
// reduce across chunks after a grid barrier; other variants ignore it.
#define EXL3_GEMM_ARGS_KS \
    EXL3_GEMM_ARGS, \
    float* __restrict__ ws
typedef void (*fp_exl3_gemm_kernel_ks) (EXL3_GEMM_ARGS_KS);

// Dual K-split: dual args + workspace
#define EXL3_GEMM_ARGS_DUAL_KS \
    EXL3_GEMM_ARGS_DUAL, \
    float* __restrict__ ws
typedef void (*fp_exl3_gemm_kernel_dual_ks) (EXL3_GEMM_ARGS_DUAL_KS);
typedef void (*fp_exl3_mgemm_kernel) (EXL3_MGEMM_ARGS);

// Shape table: TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES. EXL3_GEMM_FOREACH_SHAPE(X, ...)
// expands X(shape_idx, ...) once per shape, in index order; the parallel tables are indexed by shape
// with an unused slot 0.
#if defined(USE_ROCM)

// RDNA (64 KB of LDS per workgroup): 256 threads, TILESIZE_K 16 and three or four stages keep every
// bitrate inside the budget; TILESIZE_N must be a multiple of 128 and of 16 x the warp count (the
// WMMA inner has no remainder pass over N-blocks). The inner implements TILESIZE_M == 16 only.
#define EXL3_GEMM_SHAPE_1     16,     16,    128,     4,     3
#define EXL3_GEMM_SHAPE_2     16,     16,    256,     3,     3
#define EXL3_GEMM_SHAPE_3     16,     16,    384,     3,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     3,     3

#define EXL3_GEMM_TILESIZE_M  0, 16, 16, 16, 16
#define EXL3_GEMM_TILESIZE_K  0, 16, 16, 16, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 256, 384, 512
#define EXL3_GEMM_BLOCKDIM    0, 256, 256, 256, 256

#define EXL3_GEMM_NUM_SHAPES 4
#define EXL3_GEMM_FOREACH_SHAPE(X, ...) \
    X(1, __VA_ARGS__) X(2, __VA_ARGS__) X(3, __VA_ARGS__) X(4, __VA_ARGS__)

// Row stride, in halves, of the per-warp B staging tile the RDNA inner transposes dequantized
// fragments through on the way to WMMA layout. 18 rather than 17: at 17 the adjacent active-lane
// groups (0-3 vs 16-19, 8-11 vs 24-27) land on the same LDS banks; 18 halves = 9 dwords, coprime
// with 32, so every bank is covered. One definition for the kernel's indexing and the smem
// accounting below: a retyped literal once made every launch under-allocate dynamic LDS
#define EXL3_GEMM_SH_B_DQ_STRIDE 18

#else

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3
// Multi-row tiles: all rows of a tile share each decoded B fragment, where the 16-row shapes
// read and decode the whole weight matrix again for every 16 rows
#define EXL3_GEMM_SHAPE_5     32,     32,    128,     3,     3
#define EXL3_GEMM_SHAPE_6     48,     32,    128,     3,     3
#define EXL3_GEMM_SHAPE_7     64,     32,    128,     3,     3

#define EXL3_GEMM_TILESIZE_M  0, 16, 16, 16, 16, 32, 48, 64
#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16, 32, 32, 32
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512, 128, 128, 128
#define EXL3_GEMM_BLOCKDIM    0, 256, 512, 512, 256, 512, 512, 512

#define EXL3_GEMM_NUM_SHAPES 7
#define EXL3_GEMM_FOREACH_SHAPE(X, ...) \
    X(1, __VA_ARGS__) X(2, __VA_ARGS__) X(3, __VA_ARGS__) X(4, __VA_ARGS__) \
    X(5, __VA_ARGS__) X(6, __VA_ARGS__) X(7, __VA_ARGS__)

#endif

// Instance lists, indexed by shape (slot 0 unused). Shape 1 is not currently selected anywhere
#define EXL3_GEMM_INST(S, _bits, _half, _c_fp32, cb)  exl3_gemm_kernel<_bits, _half, _c_fp32, cb, EXL3_GEMM_SHAPE_##S>,
#define EXL3_MGEMM_INST(S, _bits, _half, _c_fp32, cb) exl3_mgemm_kernel<_bits, _half, _c_fp32, cb, EXL3_GEMM_SHAPE_##S>,

#define EXL3_GEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, EXL3_GEMM_FOREACH_SHAPE(EXL3_GEMM_INST, _bits, false, _c_fp32, cb)
#define EXL3_MGEMM_KERNEL_INSTANCES(_bits, _c_fp32, cb) \
    nullptr, EXL3_GEMM_FOREACH_SHAPE(EXL3_MGEMM_INST, _bits, false, _c_fp32, cb)

// Half-integer bitrates (bits + 0.5), mul1 codebook
#define EXL3_GEMM_KERNEL_INSTANCES_H(_bits, _c_fp32) \
    nullptr, EXL3_GEMM_FOREACH_SHAPE(EXL3_GEMM_INST, _bits, true, _c_fp32, 2)
#define EXL3_MGEMM_KERNEL_INSTANCES_H(_bits, _c_fp32) \
    nullptr, EXL3_GEMM_FOREACH_SHAPE(EXL3_MGEMM_INST, _bits, true, _c_fp32, 2)

#define EXL3_KERNEL_INSTANCES_H(K) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES_H(K, true) }; \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[] = { EXL3_GEMM_KERNEL_INSTANCES_H(K, false) }; \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_h##K[] = { EXL3_MGEMM_KERNEL_INSTANCES_H(K, true) }; \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_h##K[] = { EXL3_MGEMM_KERNEL_INSTANCES_H(K, false) };

#define EXL3_KERNEL_EXTERNS_H(K) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_h##K[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_h##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_h##K[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_h##K[];

// Dynamic shared memory a (shape, bitrate) instantiation stages, as laid out by the backend's inner
// kernel. Single definition, used both by that kernel's static_assert and by the host-side shape
// filter, so the two cannot disagree about what a shape costs.
//
// shmem_out_had is the GEMM's sh_c variant (it stages a full output tile for the fused output
// Hadamard); the MoE kernel passes false and only needs the reduction scratch.
// Parameters are ordered to match the EXL3_GEMM_SHAPE_n expansion (TILESIZE_M, TILESIZE_K,
// TILESIZE_N, SH_STAGES, FRAG_STAGES) so the macro can be splatted in directly; frag_stages
// is a register-pipelining depth and does not affect shared memory. half_k adds the extra
// 8 uint16 per tile of a half-integer bitrate (mul1 codebook).
__host__ __device__ constexpr int exl3_gemm_smem_bytes(
    int tilesize_m, int tilesize_k, int tilesize_n, int sh_stages, int frag_stages,
    int bits, bool half_k, bool shmem_out_had)
{
    (void) frag_stages;
    int tileblocks_m = tilesize_m / 16;
    int tileblocks_k = tilesize_k / 16;
    int tileblocks_n = tilesize_n / 16;
    int num_warps = EXL3_GEMM_BASE_THREADS / 32;
    int tile_u16 = 16 * bits + (half_k ? 8 : 0);
    int sh_b_stage_size = tileblocks_k * tileblocks_n * tile_u16;              // uint16s
    int sh_c_had = shmem_out_had ? tilesize_n * tilesize_m : 0;                // floats
#if defined(USE_ROCM)
    // RDNA inner: A rows padded by one 8-half group in place of the CUDA XOR swizzle, a per-warp
    // B staging tile for the transpose into WMMA layout, and one WMMA N-block per fragment
    (void) tileblocks_m;
    int frags_n_per_warp = tileblocks_n / num_warps;
    int sh_a_stage_size = tilesize_m * (tilesize_k + 8);                       // halfs
    int sh_b_dq_size = num_warps * tileblocks_k * 16 * EXL3_GEMM_SH_B_DQ_STRIDE;   // halfs
    int sh_c_reduce = tileblocks_k > 1 ? 8 * EXL3_GEMM_BASE_THREADS * frags_n_per_warp : 0;
    int sh_c_size = sh_c_reduce > sh_c_had ? sh_c_reduce : sh_c_had;
    return sh_stages * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 2 * sh_b_dq_size + 4 * sh_c_size;
#else
    int frags_n_per_warp = 2 * tileblocks_n / num_warps;
    int sh_a_stage_size = tilesize_m * tilesize_k;                             // halfs
    int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS * frags_n_per_warp * tileblocks_m;   // floats
    if (sh_c_had > sh_c_size) sh_c_size = sh_c_had;
    return sh_stages * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size;
#endif
}

// Same, addressed by shape index: expands the EXL3_GEMM_SHAPE_n macro so the tile dims and
// stage count come from the one place they are declared, rather than a parallel table that
// has to be updated by hand whenever a shape changes.
#define EXL3_GEMM_SMEM_FOR_SHAPE(_shape, _bits, _half, _had) \
    exl3_gemm_smem_bytes(_shape, _bits, _half, _had)
#define EXL3_GEMM_SMEM_CASE(S, _bits, _half, _had) \
    case S: return EXL3_GEMM_SMEM_FOR_SHAPE(EXL3_GEMM_SHAPE_##S, _bits, _half, _had);

__host__ __device__ constexpr int exl3_gemm_smem_bytes_for_shape(
    int shape_idx, int bits, bool half_k, bool shmem_out_had)
{
    switch (shape_idx)
    {
        EXL3_GEMM_FOREACH_SHAPE(EXL3_GEMM_SMEM_CASE, bits, half_k, shmem_out_had)
        default: return 0;
    }
}

// Instance arrays are indexed by shape and defined per (K, cb) so each codebook compiles as a separate
// translation unit (see comp_units/exl3_comp_unit_K_cbX.cu)

#define EXL3_KERNEL_EXTERNS_CB(K, cb) \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[]; \
    extern fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[]; \
    extern fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[]; \

#define ALL_EXL3_KERNEL_EXTERNS(K) \
    EXL3_KERNEL_EXTERNS_CB(K, 0) \
    EXL3_KERNEL_EXTERNS_CB(K, 1) \
    EXL3_KERNEL_EXTERNS_CB(K, 2) \

#define EXL3_KERNEL_INSTANCES_CB(K, cb) \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b##K##_cb##cb[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, true, cb) \
    }; \
    \
    fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b##K##_cb##cb[] = { \
        EXL3_GEMM_KERNEL_INSTANCES(K, false, cb) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b##K##_cb##cb[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, true, cb) \
    }; \
    \
    fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b##K##_cb##cb[] = { \
        EXL3_MGEMM_KERNEL_INSTANCES(K, false, cb) \
    };

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int bits,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb,
    const bool half_k = false
);

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
(
    const int cc,
    const int size_m,
    const int size_k,
    const int size_n,
    const int K,
    const bool c_fp32,
    const int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms,
    const int cb,
    const int bszm_in,
    const int bszm_out,
    const bool half_k = false
);

fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k = false);
fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb, bool half_k = false);
