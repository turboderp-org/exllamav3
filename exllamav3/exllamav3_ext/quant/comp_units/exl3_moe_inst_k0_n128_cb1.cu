#include "exl3_moe_inst_common.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k0_n128_cb1() { return exl3_moe_kernel<0, 128, 1>; }

#if defined(USE_ROCM)
fp_exl3_moe_kernel exl3_moe_kernel_k0_n128_cb1_pipe() { return exl3_moe_kernel<0, 128, 1, MOE_TILESIZE_M, false, true>; }
#endif
