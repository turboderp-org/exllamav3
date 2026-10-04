#pragma once

// Includes for one MoE instance unit (exl3_moe_inst_*.cu). The getters are defined per
// backend because the kernel templates differ: CUDA's trailing parameter is half_k (the half-
// integer rates come as separate getters, exl3_moe_kernel_h*), RDNA's is PIPE (a pipelined
// mainloop variant of every instance, with the half-integer rates folded into the bit width by
// EXL3_HALF_BITS). The RDNA kernel header has to come first: it establishes SMEM_MAX for the
// shape constants in exl3_moe_common.cuh
#if defined(USE_ROCM)
    #include "../../rocm/quant/exl3_moe_kernel_rdna.cuh"
    #include "exl3_moe_instances.cuh"
    #include "../../rocm/quant/exl3_moe_pipe_instances_rdna.cuh"
    static_assert(SMEM_MAX == EXL3_RDNA_SMEM_MAX,
        "SMEM_MAX is not the RDNA value -- exl3_moe_kernel_rdna.cuh must be included first");
#else
    #include "exl3_moe_instances.cuh"
    #include "../exl3_moe_kernel.cuh"
#endif
