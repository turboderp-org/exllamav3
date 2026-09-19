#include <Python.h>
#include "linear.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "../util.h"
#include "../hgemm.cuh"
#include "../quant/exl3_gemm.cuh"
#include "../quant/exl3_gemv.cuh"
#include "../quant/exl3_devctx.cuh"
#include "../quant/reconstruct.cuh"
#include "../quant/hadamard.cuh"
#include "../add.cuh"

void BC_LinearFP16::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    if (x.dtype() == y.dtype() && !graph)
        at::matmul_out(y, x, weight);
    else
        hgemm_gr(x, weight, y, graph);

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearFP16::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

//void BC_LinearFP16::run_cublas(const at::Tensor& x, at::Tensor& y)
//{
//    hgemm(x, weight, y);
//    if (bias)
//        y.add_(bias.value());
//}

void BC_LinearEXL3::run_gr(const at::Tensor& x, at::Tensor& y, Graph* graph)
{
    // K > 4: the sm70 GEMV kernel covers K 5-8 at m <= 8 — route
    // through the normal dispatch, which tries the GEMV first and
    // falls back to reconstruct + cuBLAS hgemm for whatever it
    // declines (exl3_gemm_gr's K > 4 && cc < 8 fallback). The old
    // short-circuit here dequantized the weights on EVERY call —
    // nsys showed reconstruct + gemv2N eating ~88% of decode GPU
    // time.
    if (K > 4)
    {
        TORCH_CHECK(!graph || graph->disabled, "BC_LinearEXL3 K > 4 invoked with graph capture");
        exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0, graph);
        if (bias)
            add_gr(y, bias.value(), y, graph);
        return;
    }

    if (x.numel() == x.size(-1))
    {
        exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg, mul1, 0, graph);
    }
    else
    {
        TORCH_CHECK(!graph || graph->disabled, "BC_LinearEXL3 invoked with graph and bsz > 1");
        at::Tensor xh_ = at::empty_like(x);
        exl3_gemm(x, trellis, y, suh, xh_, svh, -1, mcg, mul1, 0);
    }

    if (bias)
        add_gr(y, bias.value(), y, graph);
}

void BC_LinearEXL3::run(const at::Tensor& x, at::Tensor& y)
{
    run_gr(x, y, nullptr);
}

at::Tensor BC_LinearEXL3::run_alloc(const at::Tensor& x, int64_t out_features, bool output_fp32)
{
    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() = out_features;

    at::Tensor y = at::empty(
        out_shape,
        x.options().dtype(output_fp32 ? at::kFloat : at::kHalf)
    );
    if (out_features == 0) return y;

    at::Tensor x_flat = x.view({-1, x.size(-1)});
    at::Tensor y_flat = y.view({-1, out_features});
    run(x_flat, y_flat);
    return y;
}

// Fused gate+up for the sm70 MoE per-expert loop: one exl3_gemv2
// launch instead of two exl3_gemm calls. Eligibility mirrors the
// dual kernel's coverage (sm_70 GEMV, m == 1, matching K/cb/dtype,
// su/sv present on both). Falls back to two run_alloc calls when
// ineligible — callers can treat this as a drop-in replacement.
std::pair<at::Tensor, at::Tensor> BC_LinearEXL3::run_alloc_pair
(
    const at::Tensor& x,
    BC_LinearEXL3& other
)
{
    // Fused path eligibility: the dual GEMV kernel covers sm_70, m == 1,
    // matching K/cb, half output, and requires su/sv on both slots.
    // Anything else falls back to two separate calls.
    int device = x.get_device();
    int cc = DevCtx::instance().get_cc(device);
    bool eligible =
        cc < 8 &&
        x.numel() == x.size(-1) &&            // m == 1 (GEMV shape)
        K == other.K && K == 3 &&  // dual kernel instantiated for bits=3 only (DSV4 3bpw); other bitrates fall back to two forwards
        mcg == other.mcg && mul1 == other.mul1 &&
        !bias && !other.bias &&
        suh.defined() && svh.defined() && xh.defined() &&
        other.suh.defined() && other.svh.defined() && other.xh.defined();

    if (eligible)
    {
        // Widths derive from the trellis layout: size_n = B.size(1) * 16
        // (same derivation as exl3_gemm.cu's size_n).
        int64_t n_up = trellis.size(1) * 16;
        int64_t n_gate = other.trellis.size(1) * 16;
        at::Tensor u = at::empty({n_up}, x.options().dtype(at::kHalf));
        at::Tensor g = at::empty({n_gate}, x.options().dtype(at::kHalf));
        // xh scratch is cache-aliased across same-shaped Linears
        // (GTensorCache keys on shape/dtype), so other.xh may alias xh.
        // The dual kernel's phase-1 writes A_had and A_had2 disjointly —
        // aliased, slot 1's Had would clobber slot 0's input. Allocate a
        // dedicated scratch for slot 1.
        at::Tensor xh2 = at::empty(x.sizes(), x.options().dtype(at::kHalf));
        exl3_gemv2
        (
            x, trellis, u, other.trellis, g,
            suh, xh, svh,
            other.suh, xh2, other.svh,
            mcg, mul1
        );
        return {u, g};
    }

    int64_t n_up = trellis.size(1) * 16;
    int64_t n_gate = other.trellis.size(1) * 16;
    at::Tensor u = run_alloc(x, n_up, false);
    at::Tensor g = other.run_alloc(x, n_gate, false);
    return {u, g};
}
