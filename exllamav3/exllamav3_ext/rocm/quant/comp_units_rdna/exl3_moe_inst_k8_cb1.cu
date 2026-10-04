// Generated MoE instantiation for RDNA. Mirrors
// quant/comp_units/exl3_moe_inst_k8_cb1.cu, with two changes:
//
//   - the kernel header is the RDNA sibling
//   - it is included BEFORE exl3_moe_instances.cuh, which is load-bearing:
//     exl3_moe_common.cuh sets SMEM_MAX to 90 KB behind an #ifndef, and the
//     RDNA kernel map has to win that race or the inner's LDS static_assert
//     admits shapes this part cannot launch. There is a static_assert below
//     that fails loudly if the order is ever swapped.
//
// MoE has no cb0 -- only cb1 (mcg) and cb2 (mul1).

#include "../exl3_moe_kernel_rdna.cuh"
#include "../../../quant/comp_units/exl3_moe_instances.cuh"
#include "../exl3_moe_pipe_instances_rdna.cuh"

// Asserts the include race was won, NOT a specific size: exl3_moe_common.cuh
// would set SMEM_MAX to 90 KB behind an #ifndef if it got here first, which on
// a 64 KB part lets the inner's LDS static_assert admit unlaunchable shapes.
// Compare against EXL3_RDNA_SMEM_MAX so a -DEXL3_RDNA_SMEM_MAX build still works.
static_assert(SMEM_MAX == EXL3_RDNA_SMEM_MAX,
    "SMEM_MAX is not the RDNA value -- exl3_moe_kernel_rdna.cuh must be included first");

fp_exl3_moe_kernel exl3_moe_kernel_k8_n128_cb1() { return exl3_moe_kernel<8, 128, 1>; }
fp_exl3_moe_kernel exl3_moe_kernel_k8_n256_cb1() { return exl3_moe_kernel<8, 256, 1>; }

// Pipelined mainloop (exl3_moe_inner_rdna.cuh); EXL3_ROCM_MOE_PIPE=0 selects the getters above
fp_exl3_moe_kernel exl3_moe_kernel_k8_n128_cb1_pipe() { return exl3_moe_kernel<8, 128, 1, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k8_n256_cb1_pipe() { return exl3_moe_kernel<8, 256, 1, MOE_TILESIZE_M, true>; }
