// RDNA instantiation unit: half-integer bitrate 1.5 bpw, mul1 codebook.
//
// Mirrors quant/comp_units/exl3_comp_unit_h1.cu: defines exactly the
// tfp_exl3_{gemm,mgemm}_kernel_{fp32,fp16}_h1 symbols that
// exl3_kernel_map_rdna.cu declares with EXL3_KERNEL_EXTERNS_H(1), against the
// RDNA kernel (exl3_gemm_kernel_rdna.cuh with half_k = true).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../../../util.h"
#include "../../../util.cuh"
#include "../exl3_gemm_kernel_rdna.cuh"

EXL3_KERNEL_INSTANCES_H(1)
