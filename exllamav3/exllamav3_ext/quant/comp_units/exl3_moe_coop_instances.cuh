#pragma once

// Per-K compilation units of the fused decode MoE kernels (exl3_moe_coop_kernel.cuh): each
// exl3_moe_coop_inst_k{K}.cu instantiates the gate/up (A) and down (B) kernels for the three
// codebooks. Returns the kernel pointer for cudaLaunchKernel and the dynamic shared memory size
// (A: for the given Hi; B: fixed)

struct MoeCoopKernel
{
    void* kernel;
    int smem;
};

#define EXL3_MOE_COOP_INST_DECL(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_k##K(int cb, int Hi, bool wide); \
    MoeCoopKernel exl3_moe_coop_kernel_b_k##K(int cb, bool wide);

EXL3_MOE_COOP_INST_DECL(1)
EXL3_MOE_COOP_INST_DECL(2)
EXL3_MOE_COOP_INST_DECL(3)
EXL3_MOE_COOP_INST_DECL(4)
EXL3_MOE_COOP_INST_DECL(5)
EXL3_MOE_COOP_INST_DECL(6)
EXL3_MOE_COOP_INST_DECL(7)
EXL3_MOE_COOP_INST_DECL(8)

#undef EXL3_MOE_COOP_INST_DECL

// Half-integer rates K + 0.5 (exl3_moe_coop_inst_h{K}.cu), mul1 codebook only
#define EXL3_MOE_COOP_INST_DECL_H(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_h##K(int Hi, bool wide); \
    MoeCoopKernel exl3_moe_coop_kernel_b_h##K(bool wide);

EXL3_MOE_COOP_INST_DECL_H(1)
EXL3_MOE_COOP_INST_DECL_H(2)
EXL3_MOE_COOP_INST_DECL_H(3)

#undef EXL3_MOE_COOP_INST_DECL_H

#define EXL3_MOE_COOP_INST_DEF_H(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_h##K(int Hi, bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_a_bytes<K, true>(Hi); \
        if (wide) return { (void*) exl3_moe_coop_a_kernel<K, 2, true, true>, smem }; \
        return { (void*) exl3_moe_coop_a_kernel<K, 2, false, true>, smem }; \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_b_h##K(bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_b_bytes<K, true>(); \
        if (wide) return { (void*) exl3_moe_coop_b_kernel<K, 2, true, true>, smem }; \
        return { (void*) exl3_moe_coop_b_kernel<K, 2, false, true>, smem }; \
    }

// Body of an instance file
#define EXL3_MOE_COOP_INST_DEF(K) \
    MoeCoopKernel exl3_moe_coop_kernel_a_k##K(int cb, int Hi, bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_a_bytes<K>(Hi); \
        if (wide) switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_a_kernel<K, 0, true>, smem }; \
            case 1: return { (void*) exl3_moe_coop_a_kernel<K, 1, true>, smem }; \
            default: return { (void*) exl3_moe_coop_a_kernel<K, 2, true>, smem }; \
        } \
        switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_a_kernel<K, 0, false>, smem }; \
            case 1: return { (void*) exl3_moe_coop_a_kernel<K, 1, false>, smem }; \
            default: return { (void*) exl3_moe_coop_a_kernel<K, 2, false>, smem }; \
        } \
    } \
    MoeCoopKernel exl3_moe_coop_kernel_b_k##K(int cb, bool wide) \
    { \
        using namespace exl3_moe_coop_ns; \
        const int smem = smem_b_bytes<K>(); \
        if (wide) switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_b_kernel<K, 0, true>, smem }; \
            case 1: return { (void*) exl3_moe_coop_b_kernel<K, 1, true>, smem }; \
            default: return { (void*) exl3_moe_coop_b_kernel<K, 2, true>, smem }; \
        } \
        switch (cb) \
        { \
            case 0: return { (void*) exl3_moe_coop_b_kernel<K, 0, false>, smem }; \
            case 1: return { (void*) exl3_moe_coop_b_kernel<K, 1, false>, smem }; \
            default: return { (void*) exl3_moe_coop_b_kernel<K, 2, false>, smem }; \
        } \
    }
