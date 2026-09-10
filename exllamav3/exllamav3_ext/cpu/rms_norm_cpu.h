#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>

// Native fused RMSNorm for CPU tensors (vision-tower CPU offload path).
//
// Same math as the CUDA rms_norm kernels (norm.cu), computed in float32 with the
// weight folded into the second pass: y = rms(x) * (w + bias) * scale.
//
// rms_norm_cpu covers the standard path (add_residual = fused post-norm accumulate,
// y aliases the residual buffer), grouped weights (row % groups) and caller-flattened
// span_heads inputs (the caller flattens x/y over (-2), mirroring rms_norm_impl).
// rms_norm_res_in_cpu covers fused pre-norm (r += x with half rounding, y = norm(r)).
// Neither entry takes span_heads: res_in hardcodes it false, like rms_norm_res_in.
//
// Layouts: x/y/r any contiguous CPU shape, norm over last dim (dim > 0); rows =
// numel / dim. w is 1-D contiguous with numel == dim (or groups * dim), or null/None
// (unweighted). x/w/r in {float32, half, bfloat16}; y/r in {float32, half} run
// vectorized, bfloat16 outputs take the threaded scalar kernel. dim need not be a
// multiple of the vector width (scalar tails). rows == 0 is a no-op.
//
// Threading runs on the persistent process-wide pool shared with the MoE CPU path
// (moe_mul1.cpp). ISA dispatch (AVX-512 / AVX2 / scalar) reuses the MoE ISA detection,
// overridable per process with EXL3_RMSNORM_CPU_ISA=avx512|avx2|scalar (testing only).
void rms_norm_cpu(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool add_residual,
    int64_t w_groups = 1);

void rms_norm_res_in_cpu(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    at::Tensor r,
    float epsilon,
    float constant_bias,
    float constant_scale);
