#pragma once

// Architecture feature gates.
//
// EXL3 was written against sm_80+. Turing (sm_75) is missing three things the EXL3 GEMM
// relies on, and each one is handled in a different place:
//
//   1. cp.async         -> ptx.cuh falls back to plain synchronous 16 B smem stores. The
//                          pipeline's existing __syncthreads() calls already provide the
//                          ordering the async waits used to provide, so the fallback is
//                          correct, just without global->shared overlap.
//   2. mma.m16n8k16     -> ptx.cuh issues two mma.m16n8k8 instead. The k=16 fragment splits
//                          exactly into the two k=8 fragments (A: {a0,a1} then {a2,a3},
//                          B: b0 then b1), so this is bit-equivalent, not an approximation.
//   3. 90 KB shared mem -> Turing caps dynamic shared memory at 64 KB per block. The limit is
//                          a host-side launch parameter, so it is resolved per device at
//                          runtime (DevCtx::get_smem_max) and shapes that do not fit are
//                          filtered out of kernel selection rather than failing at launch.
//
// ldmatrix, __nanosleep, __dp4a and tanh.approx.f32 are all sm_75-capable and need no gate.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    #define EXL3_SM75 1
#else
    #define EXL3_SM75 0
#endif

// Dynamic shared memory per block. SMEM_MAX stays at the sm_86 value because it only bounds
// the compile-time static_assert in exl3_gemm_inner.cuh, which must keep accepting every
// template instantiation. What actually gets requested at launch is the per-device value.
#define EXL3_SMEM_MAX_DEFAULT (90 * 1024)
#define EXL3_SMEM_MAX_SM75    (64 * 1024)
