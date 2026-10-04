#include "exl3_moe_inst_common.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb2() { return exl3_moe_kernel<3, 128, 2>; }
fp_exl3_moe_kernel exl3_moe_kernel_k3_n256_cb2() { return exl3_moe_kernel<3, 256, 2>; }

#if defined(USE_ROCM)
fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb2_pipe() { return exl3_moe_kernel<3, 128, 2, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k3_n256_cb2_pipe() { return exl3_moe_kernel<3, 256, 2, MOE_TILESIZE_M, true>; }
#endif
