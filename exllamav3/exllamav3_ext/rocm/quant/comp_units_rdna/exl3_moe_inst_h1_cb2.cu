// Generated MoE instantiation for RDNA: HALF-INTEGER rate 1.5 bpw (mul1), pipelined mainloop only
// (EXL3_ROCM_HALF_MOE_PIPE). t_bits is the pseudo width EXL3_HALF_BITS(1) = 17 of
// exl3_gemv_tiles_rdna.cuh; exl3_moe_rdna.cu selects it for uniform 1.5 bpw gate / up / down.
// Include order as in exl3_moe_inst_k2_cb2.cu (the SMEM_MAX race; asserted below).

#include "../exl3_moe_kernel_rdna.cuh"
#include "../../../quant/comp_units/exl3_moe_instances.cuh"
#include "../exl3_moe_pipe_instances_rdna.cuh"

static_assert(SMEM_MAX == EXL3_RDNA_SMEM_MAX,
    "SMEM_MAX is not the RDNA value -- exl3_moe_kernel_rdna.cuh must be included first");

fp_exl3_moe_kernel exl3_moe_kernel_h1_n128_cb2_pipe() { return exl3_moe_kernel<EXL3_HALF_BITS(1), 128, 2, MOE_TILESIZE_M, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_h1_n256_cb2_pipe() { return exl3_moe_kernel<EXL3_HALF_BITS(1), 256, 2, MOE_TILESIZE_M, true>; }
