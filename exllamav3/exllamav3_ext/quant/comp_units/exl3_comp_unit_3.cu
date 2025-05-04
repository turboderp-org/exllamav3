#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../../ptx.cuh"
#include "../exl3_gemm_kernel.cuh"
#include "exl3_comp_unit_3.cuh"

fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b3[] = {
    nullptr,
    exl3_gemm_kernel<3, true, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<3, true, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<3, true, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<3, true, EXL3_GEMM_SHAPE_4>
};

fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b3[] = {
    nullptr,
    exl3_gemm_kernel<3, false, EXL3_GEMM_SHAPE_1>,
    exl3_gemm_kernel<3, false, EXL3_GEMM_SHAPE_2>,
    exl3_gemm_kernel<3, false, EXL3_GEMM_SHAPE_3>,
    exl3_gemm_kernel<3, false, EXL3_GEMM_SHAPE_4>
};


