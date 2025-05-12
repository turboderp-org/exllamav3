#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../../ptx.cuh"
#include "../exl3_gemm_kernel.cuh"
#include "exl3_comp_unit_2.cuh"

fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp32_b2[] = {
    EXL3_GEMM_KERNEL_INSTANCES(2, true)
};

fp_exl3_gemm_kernel tfp_exl3_gemm_kernel_fp16_b2[] = {
    EXL3_GEMM_KERNEL_INSTANCES(2, false)
};

fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp32_b2[] = {
    EXL3_MGEMM_KERNEL_INSTANCES(2, true)
};

fp_exl3_mgemm_kernel tfp_exl3_mgemm_kernel_fp16_b2[] = {
    EXL3_MGEMM_KERNEL_INSTANCES(2, false)
};