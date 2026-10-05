#pragma once

// Includes for one MoE instance unit (exl3_moe_inst_*.cu). The kernel template is shared; the units
// differ per backend only in which instances they define: CUDA adds the 32 / 64-row tiles and the
// non-pipelined half-integer rates, ROCm adds the pipelined (PIPE) variant of every instance
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"
#if defined(USE_ROCM)
    #include "../../rocm/quant/exl3_moe_pipe_instances_rdna.cuh"
#endif
