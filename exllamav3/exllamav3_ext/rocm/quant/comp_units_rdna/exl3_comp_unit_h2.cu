// RDNA instantiation unit: half-integer bitrate 2.5 bpw, mul1 codebook.
//
// Mirrors quant/comp_units/exl3_comp_unit_h2.cu: defines exactly the
// tfp_exl3_{gemm,mgemm}_kernel_{fp32,fp16}_h2 symbols that
// exl3_kernel_map_rdna.cu declares with EXL3_KERNEL_EXTERNS_H(2), against the
// RDNA kernel (exl3_gemm_kernel_rdna.cuh with half_k = true).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../../../util.h"
#include "../../../util.cuh"
#include "../exl3_gemm_kernel_rdna.cuh"

EXL3_KERNEL_INSTANCES_H(2)
