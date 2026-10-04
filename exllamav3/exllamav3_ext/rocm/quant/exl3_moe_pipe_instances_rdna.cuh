#pragma once

// Getters for the pipelined-mainloop MoE kernels (exl3_moe_kernel<..., PIPE = true>).
// Defined next to the CUDA-named getters in quant/comp_units/exl3_moe_inst_*.cu (ROCm arm);
// include after quant/comp_units/exl3_moe_instances.cuh (for fp_exl3_moe_kernel).

#define EXL3_MOE_DECLARE_PIPE_GETTERS(K) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1_pipe(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1_pipe(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2_pipe(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2_pipe(); \

EXL3_MOE_DECLARE_PIPE_GETTERS(0);
EXL3_MOE_DECLARE_PIPE_GETTERS(1);
EXL3_MOE_DECLARE_PIPE_GETTERS(2);
EXL3_MOE_DECLARE_PIPE_GETTERS(3);
EXL3_MOE_DECLARE_PIPE_GETTERS(4);
EXL3_MOE_DECLARE_PIPE_GETTERS(5);
EXL3_MOE_DECLARE_PIPE_GETTERS(6);
EXL3_MOE_DECLARE_PIPE_GETTERS(7);
EXL3_MOE_DECLARE_PIPE_GETTERS(8);

// Half-integer rates, mul1 only (quant/comp_units/exl3_moe_inst_h*_cb2.cu, ROCm arm): t_bits =
// EXL3_HALF_BITS(K), uniform gate / up / down, pipelined mainloop only
#define EXL3_MOE_DECLARE_PIPE_GETTERS_H(K) \
    fp_exl3_moe_kernel exl3_moe_kernel_h##K##_n128_cb2_pipe(); \
    fp_exl3_moe_kernel exl3_moe_kernel_h##K##_n256_cb2_pipe(); \

EXL3_MOE_DECLARE_PIPE_GETTERS_H(1);
EXL3_MOE_DECLARE_PIPE_GETTERS_H(2);
EXL3_MOE_DECLARE_PIPE_GETTERS_H(3);

#undef EXL3_MOE_DECLARE_PIPE_GETTERS_H
#undef EXL3_MOE_DECLARE_PIPE_GETTERS
