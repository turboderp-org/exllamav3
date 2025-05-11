#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include <tuple>
#include <mutex>
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

int select_gemm_shape(int cc, int size_m, int size_k, int size_n, int bits)
{
    bool mod_256 = (size_n % 256 == 0);
    bool mod_512 = (size_n % 512 == 0);

    switch(cc)
    {
        case CC_OLD:
        case CC_AMPERE:
            if (bits <= 4)
            {
                if (size_n <= 2048) return 2;
                return 3;
            }
            if (size_n < 4096) return size_k > 8192 ? 3 : 2;
            if (mod_512 && (size_n * size_k) > (4096 * 4096) && bits <= 6) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_ADA:
            if (bits <= 3)
            {
                if (size_n < 4096 && size_k <= 12288) return 2;
                return 3;
            }
            if (size_n <= 16384) return 2;
            if (mod_512 && size_n >= 32768) return 4;
            if (mod_256) return 3;
            return 2;

        case CC_HOPPER:
        case CC_BLACKWELL:
            if (bits >= 7)
            {
                if (size_n <= 8192) return size_k > 32768 ? 3 : 2;
                if (mod_512 && size_n > 32768) return 4;
                return 2;
            }
            if (size_n <= 4096) return (size_k && bits >= 3) > 8192 ? 3 : 2;
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

bool exl3_gemm_shape_compat(int shape_idx, int size_m, int size_k, int size_n, int bits)
{
    int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
    return (size_k % tilesize_k == 0) && (size_n % tilesize_n == 0);
}

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
    int* num_sms
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, bits) : force_shape_idx;
    TORCH_CHECK(shape_idx > 0, "exl3_gemm: no compatible kernel");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        // decided experimentally, TODO: evaluate if Ampere would benefit from larger grid
        int max_slices = size_k / tilesize_k * size_n / tilesize_n / 12;
        *num_sms = MIN(max_slices, *num_sms);
    }

    if (c_fp32)
    {
        switch (bits)
        {
            case 1: return tfp_exl3_gemm_kernel_fp32_b1[shape_idx];
            case 2: return tfp_exl3_gemm_kernel_fp32_b2[shape_idx];
            case 3: return tfp_exl3_gemm_kernel_fp32_b3[shape_idx];
            case 4: return tfp_exl3_gemm_kernel_fp32_b4[shape_idx];
            case 5: return tfp_exl3_gemm_kernel_fp32_b5[shape_idx];
            case 6: return tfp_exl3_gemm_kernel_fp32_b6[shape_idx];
            case 7: return tfp_exl3_gemm_kernel_fp32_b7[shape_idx];
            case 8: return tfp_exl3_gemm_kernel_fp32_b8[shape_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
    else
    {
        switch (bits)
        {
            case 1: return tfp_exl3_gemm_kernel_fp16_b1[shape_idx];
            case 2: return tfp_exl3_gemm_kernel_fp16_b2[shape_idx];
            case 3: return tfp_exl3_gemm_kernel_fp16_b3[shape_idx];
            case 4: return tfp_exl3_gemm_kernel_fp16_b4[shape_idx];
            case 5: return tfp_exl3_gemm_kernel_fp16_b5[shape_idx];
            case 6: return tfp_exl3_gemm_kernel_fp16_b6[shape_idx];
            case 7: return tfp_exl3_gemm_kernel_fp16_b7[shape_idx];
            case 8: return tfp_exl3_gemm_kernel_fp16_b8[shape_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
}

fp_exl3_mgemm_kernel select_exl3_mgemm_kernel
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
    int* num_sms
)
{
    int shape_idx = force_shape_idx <= 0 ? select_gemm_shape(cc, size_m, size_k, size_n, bits) : force_shape_idx;
    TORCH_CHECK(shape_idx > 0, "exl3_mgemm: no compatible kernel");
    if (out_shape_idx) *out_shape_idx = shape_idx;
    if (out_block_dim) *out_block_dim = exl3_gemm_blockdim[shape_idx];

    // Avoid empty blocks
    if (num_sms)
    {
        int tilesize_k = exl3_gemm_tilesize_k[shape_idx];
        int tilesize_n = exl3_gemm_tilesize_n[shape_idx];
        // decided experimentally, TODO: evaluate if Ampere would benefit from larger grid
        int max_slices = size_k / tilesize_k * size_n / tilesize_n / 12;
        *num_sms = MIN(max_slices, *num_sms);
    }

    if (c_fp32)
    {
        switch (bits)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp32_b1[shape_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp32_b2[shape_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp32_b3[shape_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp32_b4[shape_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp32_b5[shape_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp32_b6[shape_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp32_b7[shape_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp32_b8[shape_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
    else
    {
        switch (bits)
        {
            case 1: return tfp_exl3_mgemm_kernel_fp16_b1[shape_idx];
            case 2: return tfp_exl3_mgemm_kernel_fp16_b2[shape_idx];
            case 3: return tfp_exl3_mgemm_kernel_fp16_b3[shape_idx];
            case 4: return tfp_exl3_mgemm_kernel_fp16_b4[shape_idx];
            case 5: return tfp_exl3_mgemm_kernel_fp16_b5[shape_idx];
            case 6: return tfp_exl3_mgemm_kernel_fp16_b6[shape_idx];
            case 7: return tfp_exl3_mgemm_kernel_fp16_b7[shape_idx];
            case 8: return tfp_exl3_mgemm_kernel_fp16_b8[shape_idx];
            default: TORCH_CHECK(false, "No kernel for GEMM shape");
        }
    }
}