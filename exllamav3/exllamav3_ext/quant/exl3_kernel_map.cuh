#pragma once

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits);
int exl3_gemm_num_kernel_shapes();
bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int bits);

#define EXL3_GEMM_T_ARGS \
    int bits, \
    bool c_fp32, \
    int TILESIZE_M, \
    int TILESIZE_K, \
    int TILESIZE_N, \
    int SH_STAGES, \
    int FRAG_STAGES \

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    int size_m, \
    int size_k, \
    int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

typedef void (*fp_exl3_gemm_kernel) (EXL3_GEMM_ARGS);

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     3,     1
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     2
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     2
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     2

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512

#define EXL3_GEMM_BASE_THREADS 256

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int bits,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* out_num_sms
);
