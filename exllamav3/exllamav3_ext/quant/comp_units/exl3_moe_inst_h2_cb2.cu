#include "exl3_moe_inst_common.cuh"

#if defined(USE_ROCM)
fp_exl3_moe_kernel exl3_moe_kernel_h2_n128_cb2_pipe() { return exl3_moe_kernel<2, 128, 2, MOE_TILESIZE_M, true, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_h2_n256_cb2_pipe() { return exl3_moe_kernel<2, 256, 2, MOE_TILESIZE_M, true, true>; }
#else
fp_exl3_moe_kernel exl3_moe_kernel_h2_n128_cb2() { return exl3_moe_kernel<2, 128, 2, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_h2_n256_cb2() { return exl3_moe_kernel<2, 256, 2, MOE_TILESIZE_M, true>; }
#endif
