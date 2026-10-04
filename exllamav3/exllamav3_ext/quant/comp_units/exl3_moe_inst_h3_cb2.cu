#include "exl3_moe_inst_common.cuh"

#if defined(USE_ROCM)
fp_exl3_moe_kernel exl3_moe_kernel_h3_n128_cb2_pipe() { return exl3_moe_kernel<EXL3_HALF_BITS(3), 128, 2, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_h3_n256_cb2_pipe() { return exl3_moe_kernel<EXL3_HALF_BITS(3), 256, 2, MOE_TILESIZE_M, true>; }
#else
fp_exl3_moe_kernel exl3_moe_kernel_h3_n128_cb2() { return exl3_moe_kernel<3, 128, 2, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_h3_n256_cb2() { return exl3_moe_kernel<3, 256, 2, MOE_TILESIZE_M, true>; }
#endif
