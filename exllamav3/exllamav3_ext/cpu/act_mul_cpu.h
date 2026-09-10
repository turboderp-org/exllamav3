#pragma once

#include <ATen/Tensor.h>
#include <cstdint>

// Native fused gate*up activation for CPU tensors (vision-tower CPU offload path).
//
// Same math as the CUDA act_mul kernels (activation_kernels.cuh), computed in float32 with a
// single round to the output dtype: z = act(g) * u, where act is silu / tanh-gelu / relu2.
// act_limit applies the same asymmetric clamp as the CUDA kernels (u clamped to [-L, +L],
// the activated gate clamped from above to +L only); silu_oai (gpt-oss) is out of scope,
// only the MoE path uses it.
//
// Layouts: g/u/a any contiguous CPU shape with equal numel (flattened internally);
// g/u in {float32, half, bfloat16}, a in {float32, half, bfloat16} in any combination.
// a may alias g or u (the GatedMLP interm_dtype == half path passes a is u): every block
// reads its g/u lanes into registers before writing a, so aliasing is exact.
// numel == 0 is a no-op. Anything outside this contract throws; the Python caller falls
// back to the torch reference, so a forward never fails here.
//
// Threading runs on the persistent process-wide pool shared with the MoE CPU path
// (moe_mul1.cpp), tiled in 4096-element chunks. ISA dispatch (AVX-512 / AVX2 / scalar)
// reuses the MoE ISA detection, overridable per process with
// EXL3_ACT_MUL_CPU_ISA=avx512|avx2|scalar (testing only).
void silu_mul_cpu(const at::Tensor& g, const at::Tensor& u, at::Tensor& a, float act_limit);
void gelu_mul_cpu(const at::Tensor& g, const at::Tensor& u, at::Tensor& a, float act_limit);
void relu2_mul_cpu(const at::Tensor& g, const at::Tensor& u, at::Tensor& a, float act_limit);
