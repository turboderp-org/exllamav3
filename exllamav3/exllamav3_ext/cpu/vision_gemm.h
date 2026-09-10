#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>

// Native vision-tower CPU GEMM replacing torch matmul on the vision CPU offload path.
//
// Computes out = x @ w (+ bias) with fp32 accumulation in registers, fp32 residents, and
// the output cast fused into the C-epilogue (no activation upcast temporary, no separate
// bias-add or cast passes). x is streamed with on-the-fly fp16/bf16 -> fp32 conversion.
//
// w arrives in repacked panel-major layout [NB, K, 48] (NB = ceil(N / 48), last panel
// zero-padded): each k-step then walks B sequentially. A strided walk over plain [K, N]
// stalls ~5x on L3 latency with the reuse pattern here (measured: 200 vs 1000 GFLOP/s),
// so the Python layer repacks resident weights once (see repack_64 in util/tensor.py)
// and caches the panels; this entry never sees plain B.
//
// Layouts: x [M, K], wb [NB, K, 48], out [M, N]; all CPU, contiguous, floating
// (half / bfloat16 / float32). bias [N] or undefined/None. M == 0 is a no-op.
//
// Threading runs on the persistent process-wide pool shared with the MoE CPU path
// (moe_mul1.cpp); ISA dispatch (AVX-512 / AVX2 / scalar) reuses the MoE ISA detection,
// overridable per process with EXL3_VISION_GEMM_ISA=avx512|avx2|scalar (testing only).
// Numerics: fp32 accumulation from fp32 residents, the same envelope as the oneDNN
// cpu_matmul fast path it replaces.
void vision_gemm_blocked_cpu(
    at::Tensor x,
    at::Tensor wb,
    c10::optional<at::Tensor> bias,
    at::Tensor out);
