// EXL3 matmuls with the preceding elementwise op folded into the RDNA3 matmul's input transform
// (rocm/quant/exl3_rdna3.cu): one launch pair instead of an elementwise kernel plus the matmul, same rounding as the
// separate kernels. Each returns false, launching nothing, where the RDNA3 path does not apply; callers then run the
// unfused ops. Adapted from exllamav3-rocm (github.com/phoenixhaxor/exllamav3-rocm, MIT).
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_rdna3.cuh"
#include "exl3_rdna3_fused.cuh"

// Gated MLP down projection with the activation folded into the input transform:
// C = (silu(g) * u) @ W, bit-identical to silu_mul (fp16) followed by exl3_gemm. Returns false
// (nothing launched) where the fused RDNA3 path doesn't apply
bool exl3_gemm_silu_gr
(
    const at::Tensor& g,
    const at::Tensor& u,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& svh,
    bool mcg,
    bool mul1,
    Graph* graph,
    bool gate_mode
)
{
    // gate_mode: input is g * sigmoid(u) (attention output gate, mul_sigmoid_ rounding) instead
    const Exl3Rdna3GNorm gn { nullptr, nullptr, 0.0f, 0.0f, GN_UP_SIGMOID };
    #if defined(USE_ROCM)
        if (g.dtype() != at::kHalf || u.dtype() != at::kHalf) return false;
        if (!g.is_contiguous() || !u.is_contiguous() || g.numel() != u.numel()) return false;
        bool c_fp32 = C.dtype() == at::kFloat;
        if (!c_fp32 && C.dtype() != at::kHalf) return false;

        const at::cuda::OptionalCUDAGuard device_guard(g.device());
        cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

        const int tile_u16 = B.size(2);
        const bool half_k = (tile_u16 % 16) != 0;
        const int K = tile_u16 / 16;
        const int size_k = g.size(-1);
        const int size_m = (int) (g.numel() / size_k);
        const int size_n = B.size(1) * 16;
        TORCH_CHECK(B.size(0) * 16 == size_k, "exl3_gemm_silu: k mismatch");
        TORCH_CHECK(C.numel() == (int64_t) size_m * size_n, "exl3_gemm_silu: output shape mismatch");
        const int cb = mul1 ? 2 : (mcg ? 1 : 0);

        return exl3_rdna3_gemm
        (
            (const half*) g.data_ptr(), (const uint16_t*) B.data_ptr(), C.data_ptr(), size_m, size_k, size_n,
            K, half_k, cb, c_fp32, (const half*) suh.data_ptr(), (const half*) svh.data_ptr(), g.device().index(), stream,
            graph, (const half*) u.data_ptr(), gate_mode ? &gn : nullptr
        );
    #else
        return false;
    #endif
}

// GatedDeltaNet output projection with the gated RMSNorm folded into the input transform:
// C = gated_rms_norm(x, w, g) @ W. x is bf16 (..., heads, 128), g the gate (bf16 or fp32, same shape),
// w the norm weight (128, bf16 or fp32). Returns false (nothing launched) where it doesn't apply
bool exl3_gemm_gnorm_gr
(
    const at::Tensor& x,
    const at::Tensor& g,
    const at::Tensor& w,
    float eps,
    float bias,
    bool sigmoid,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& svh,
    bool mcg,
    bool mul1,
    Graph* graph
)
{
    #if defined(USE_ROCM)
        if (x.dtype() != at::kBFloat16 || x.size(-1) != 128 || w.numel() != 128) return false;
        if (g.dtype() != at::kBFloat16 && g.dtype() != at::kFloat) return false;
        if (w.dtype() != at::kBFloat16 && w.dtype() != at::kFloat) return false;
        if (!x.is_contiguous() || !g.is_contiguous() || !w.is_contiguous() || g.numel() != x.numel()) return false;
        bool c_fp32 = C.dtype() == at::kFloat;
        if (!c_fp32 && C.dtype() != at::kHalf) return false;
        const int size_k = B.size(0) * 16;
        if (x.numel() % size_k) return false;
        const int size_m = (int) (x.numel() / size_k);
        const int size_n = B.size(1) * 16;
        if (C.numel() != (int64_t) size_m * size_n) return false;

        const at::cuda::OptionalCUDAGuard device_guard(x.device());
        cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
        const int tile_u16 = B.size(2);
        const bool half_k = (tile_u16 % 16) != 0;
        const int K = tile_u16 / 16;
        const int cb = mul1 ? 2 : (mcg ? 1 : 0);
        Exl3Rdna3GNorm gn { w.data_ptr(), g.data_ptr(), eps, bias,
            GN_ACTIVE | (w.dtype() == at::kBFloat16 ? GN_W_BF16 : 0) | (g.dtype() == at::kBFloat16 ? GN_G_BF16 : 0) |
            (sigmoid ? GN_SIGMOID : 0) };
        return exl3_rdna3_gemm
        (
            (const half*) x.data_ptr(), (const uint16_t*) B.data_ptr(), C.data_ptr(), size_m, size_k, size_n,
            K, half_k, cb, c_fp32, (const half*) suh.data_ptr(), (const half*) svh.data_ptr(), x.device().index(), stream,
            graph, nullptr, &gn
        );
    #else
        return false;
    #endif
}
