#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include <tuple>
#include <mutex>
#include <map>
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "comp_units/exl3_comp_unit_1.cuh"
#include "comp_units/exl3_comp_unit_2.cuh"
#include "comp_units/exl3_comp_unit_3.cuh"
#include "comp_units/exl3_comp_unit_4.cuh"
#include "comp_units/exl3_comp_unit_5.cuh"
#include "comp_units/exl3_comp_unit_6.cuh"
#include "comp_units/exl3_comp_unit_7.cuh"
#include "comp_units/exl3_comp_unit_8.cuh"

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int K, bool multi, int bszm_in, int bszm_out)
{
    bool mod_256 = (size_n % 256 == 0);
    bool mod_512 = (size_n % 512 == 0);

    size_k *= bszm_in;
    size_n *= bszm_out;

    switch(cc)
    {
        case CC_OLD:
        case CC_AMPERE:
            if (mod_256 && K <= 4)
            {
                if (size_n <= 2048 || size_k <= 2048) return 2;
                return 3;
            }
            if (mod_256 && size_n < 4096) return size_k > 8192 ? 3 : 2;
            if (mod_512 && (size_n * size_k) > (4096 * 4096) && K <= 6) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_ADA:
            if (mod_256 && K <= 3)
            {
                if (size_k <= 2048 && !multi) return 2;
                if (size_n < 4096 && size_k <= 12288) return 2;
                return 3;
            }
            if (size_n <= 16384) return 2;
            if (mod_512 && size_n >= 32768) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_HOPPER:
        case CC_BLACKWELL:
            if ((K == 4 || K == 2) && !multi)
            {
                if (size_k <= 2048) return 1;
            }
            if (K >= 7)
            {
                if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
                if (mod_512 && size_n > 32768) return 4;
                return 2;
            }
            if (mod_256 && size_n <= 4096) return size_k > 8192 && K >= 3 ? 3 : 2;
            if (mod_512 && size_n > 16384) return 4;
            if (mod_256) return 3;
            return 2;
    }
    return 0;
}

int exl3_gemm_num_kernel_shapes()
{
    return EXL3_GEMM_NUM_SHAPES;
}

int exl3_gemm_tilesize_k[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim[] = {EXL3_GEMM_BLOCKDIM};

// Shared memory a shape/bitrate instantiation needs, mirroring the layout computed in
// exl3_gemm_kernel_inner (sh_a and sh_b staged SH_STAGES deep, then the fp32 sh_c region).
// Kept in sync with the static_assert there; used to reject shapes that exceed what a
// device can actually give a block, which on Turing is 64 KB rather than 90.
static const int exl3_gemm_sh_stages[] = {0, 6, 4, 4, 4};

int exl3_gemm_shape_smem(int shape_idx, int K)
{
    const int TILESIZE_M = 16;
    int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
    int sh_stages = exl3_gemm_sh_stages[shape_idx];

    int sh_a_stage_size = TILESIZE_M * tilesize_k;                              // halfs
    int sh_b_stage_size = (tilesize_k / 16) * (tilesize_n / 16) * 256 / 16 * K; // uint16s
    int frags_n_per_warp = 2 * (tilesize_n / 16) / (EXL3_GEMM_BASE_THREADS / 32);
    int sh_c_size = MAX(4 * EXL3_GEMM_BASE_THREADS * frags_n_per_warp,          // floats
                        tilesize_n * TILESIZE_M);

    return sh_stages * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size;
}

bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int K)
{
    int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
    if (size_k % tilesize_k || size_n % tilesize_n) return false;

    // Device-dependent: callers with tensors on a non-current device must set a device guard
    // first (every in-tree caller runs under OptionalCUDAGuard). Only matters on a mixed-arch
    // host, where a 90 KB-capable device would otherwise vouch for a 64 KB one.
    int device;
    cudaGetDevice(&device);
    return exl3_gemm_shape_smem(shape_idx, K) <= DevCtx::instance().get_smem_request(device);
}

// Hard gate for explicitly forced shapes, which skip the autotuner's shape_compat filter.
// Launching a shape whose static layout exceeds what the launch can request would read past
// the end of the extern __shared__ block - silent corruption rather than a failed launch.
void exl3_gemm_check_smem(int shape_idx, int K, const char* who)
{
    int device;
    cudaGetDevice(&device);
    int need = exl3_gemm_shape_smem(shape_idx, K);
    int have = DevCtx::instance().get_smem_request(device);
    TORCH_CHECK(need <= have, who, ": shape ", shape_idx, " at ", K,
                " bpw needs ", need, " B of shared memory, device provides ", have);
}

// Instance tables, [K][cb] -> array indexed by shape_idx. Row 0 unused (no K = 0 instances)

#define EXL3_KERNEL_TABLE_ROW(fp, K) \
    { tfp_exl3_gemm_kernel_##fp##_b##K##_cb0, tfp_exl3_gemm_kernel_##fp##_b##K##_cb1, tfp_exl3_gemm_kernel_##fp##_b##K##_cb2 }
#define EXL3_MKERNEL_TABLE_ROW(fp, K) \
    { tfp_exl3_mgemm_kernel_##fp##_b##K##_cb0, tfp_exl3_mgemm_kernel_##fp##_b##K##_cb1, tfp_exl3_mgemm_kernel_##fp##_b##K##_cb2 }

static fp_exl3_gemm_kernel* const tab_gemm_fp32[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_KERNEL_TABLE_ROW(fp32, 1), EXL3_KERNEL_TABLE_ROW(fp32, 2), EXL3_KERNEL_TABLE_ROW(fp32, 3),
    EXL3_KERNEL_TABLE_ROW(fp32, 4), EXL3_KERNEL_TABLE_ROW(fp32, 5), EXL3_KERNEL_TABLE_ROW(fp32, 6),
    EXL3_KERNEL_TABLE_ROW(fp32, 7), EXL3_KERNEL_TABLE_ROW(fp32, 8)
};

static fp_exl3_gemm_kernel* const tab_gemm_fp16[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_KERNEL_TABLE_ROW(fp16, 1), EXL3_KERNEL_TABLE_ROW(fp16, 2), EXL3_KERNEL_TABLE_ROW(fp16, 3),
    EXL3_KERNEL_TABLE_ROW(fp16, 4), EXL3_KERNEL_TABLE_ROW(fp16, 5), EXL3_KERNEL_TABLE_ROW(fp16, 6),
    EXL3_KERNEL_TABLE_ROW(fp16, 7), EXL3_KERNEL_TABLE_ROW(fp16, 8)
};

static fp_exl3_mgemm_kernel* const tab_mgemm_fp32[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_MKERNEL_TABLE_ROW(fp32, 1), EXL3_MKERNEL_TABLE_ROW(fp32, 2), EXL3_MKERNEL_TABLE_ROW(fp32, 3),
    EXL3_MKERNEL_TABLE_ROW(fp32, 4), EXL3_MKERNEL_TABLE_ROW(fp32, 5), EXL3_MKERNEL_TABLE_ROW(fp32, 6),
    EXL3_MKERNEL_TABLE_ROW(fp32, 7), EXL3_MKERNEL_TABLE_ROW(fp32, 8)
};

static fp_exl3_mgemm_kernel* const tab_mgemm_fp16[9][3] =
{
    { nullptr, nullptr, nullptr },
    EXL3_MKERNEL_TABLE_ROW(fp16, 1), EXL3_MKERNEL_TABLE_ROW(fp16, 2), EXL3_MKERNEL_TABLE_ROW(fp16, 3),
    EXL3_MKERNEL_TABLE_ROW(fp16, 4), EXL3_MKERNEL_TABLE_ROW(fp16, 5), EXL3_MKERNEL_TABLE_ROW(fp16, 6),
    EXL3_MKERNEL_TABLE_ROW(fp16, 7), EXL3_MKERNEL_TABLE_ROW(fp16, 8)
};

fp_exl3_gemm_kernel select_exl3_gemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* num_sms,
    int cb
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, K, false, 1, 1) : force_shape_idx;

    TORCH_CHECK(shape_idx > 0, "exl3_gemm: no compatible kernel");
    exl3_gemm_check_smem(shape_idx, K, "exl3_gemm");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n;
        *num_sms = MAX(MIN(max_slices, *num_sms), 1);
    }

    TORCH_CHECK(K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape");
    return (c_fp32 ? tab_gemm_fp32 : tab_gemm_fp16)[K][cb][shape_idx];
}

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
(
    int cc,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int force_shape_idx,
    int* out_block_dim,
    int* out_shape_idx,
    int* num_sms,
    int cb,
    int bszm_in,
    int bszm_out
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, K, true, bszm_in, bszm_out) : force_shape_idx;
    TORCH_CHECK(shape_idx > 0, "exl3_mgemm: no compatible kernel");
    exl3_gemm_check_smem(shape_idx, K, "exl3_mgemm");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        int max_slices = size_k / tilesize_k * size_n / tilesize_n / (*num_sms > 128 ? 20 : 24);
        *num_sms = MIN(max_slices, *num_sms);
    }

    TORCH_CHECK(K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape");
    return (c_fp32 ? tab_mgemm_fp32 : tab_mgemm_fp16)[K][cb][shape_idx];
}


fp_exl3_gemm_kernel get_gemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb)
{
    TORCH_CHECK(K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape");
    return (c_fp32 ? tab_gemm_fp32 : tab_gemm_fp16)[K][cb][shape_idx];
}


fp_exl3_mgemm_kernel get_mgemm_kernel_ptr(int K, int shape_idx, bool c_fp32, int cb)
{
    TORCH_CHECK(K >= 1 && K <= 8 && cb >= 0 && cb <= 2, "No kernel for GEMM shape");
    return (c_fp32 ? tab_mgemm_fp32 : tab_mgemm_fp16)[K][cb][shape_idx];
}
