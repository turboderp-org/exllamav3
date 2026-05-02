#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k1_n128() { return exl3_moe_kernel<1, 128>; }
fp_exl3_moe_kernel exl3_moe_kernel_k1_n256() { return exl3_moe_kernel<1, 256>; }
