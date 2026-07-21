#include <cuda_fp16.h>
#include "add.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

#define NUM_THREADS 1024

#define KERNEL_DEF(xt, yt, zt, kernel, fn) \
__launch_bounds__(NUM_THREADS) \
__global__ void kernel \
( \
    const xt* __restrict__ x, \
    const yt* __restrict__ y, \
    zt* __restrict__ z, \
    const uint64_t numel_x, \
    const uint64_t numel_y \
) \
{ \
    uint64_t idx = ((uint64_t)blockIdx.x * NUM_THREADS + (uint64_t)threadIdx.x); \
    if (idx >= numel_x) return; \
    xt a = x[idx]; \
    yt b = y[idx % numel_y]; \
    z[idx] = fn; \
}

KERNEL_DEF(half,  half,  half,  add_kernel_hhh, __hadd(a, b))
KERNEL_DEF(half,  half,  float, add_kernel_hhf, __half2float(__hadd(a, b)))
KERNEL_DEF(half,  float, half,  add_kernel_hfh, __float2half_rn(__half2float(a) + b))
KERNEL_DEF(half,  float, float, add_kernel_hff, __half2float(a) + b)
KERNEL_DEF(float, half,  half,  add_kernel_fhh, __float2half_rn(a + __half2float(b)))
KERNEL_DEF(float, half,  float, add_kernel_fhf, a + __half2float(b))
KERNEL_DEF(float, float, half,  add_kernel_ffh, __float2half_rn(a + b))
KERNEL_DEF(float, float, float, add_kernel_fff, a + b)

#undef KERNEL_DEF

/*
x + y -> z
Works inplace if x == z or y == z
*/

void add_gr
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    auto xt = x.dtype();
    auto yt = y.dtype();
    auto zt = z.dtype();
    uint64_t numel_x = x.numel();
    int blocks = (int) CEIL_DIVIDE(numel_x, (uint64_t) NUM_THREADS);

    uint64_t numel_y = y.numel();
    if (numel_y != numel_x)
    {
        TORCH_CHECK(numel_y < numel_x, "Tensor shape mismatch (y > x)");
        TORCH_CHECK(numel_x % numel_y == 0, "Tensor shape mismatch (y must divide x)");
    }

    #define INSTANCE(xt_, yt_, zt_, xt__, yt__, zt__, kernel) \
    if (xt == xt_ && yt == yt_ && zt == zt_) \
    { \
        kernel<<<blocks, NUM_THREADS, 0, stream>>> \
        ( \
            (const xt__*) x.data_ptr(), \
            (const yt__*) y.data_ptr(), \
            (zt__*) z.data_ptr(), \
            numel_x, \
            numel_y \
        ); \
        if (graph) graph->record_param((void*) &kernel, GP_add_x, 0); \
        if (graph) graph->record_param((void*) &kernel, GP_add_y, 1); \
        if (graph) graph->record_param((void*) &kernel, GP_add_z, 2); \
        if (graph) graph->record_param((void*) &kernel, GP_end, 0); \
        cuda_check(cudaPeekAtLastError()); \
    }

    INSTANCE(at::kHalf,  at::kHalf,  at::kHalf,  half,  half,  half , add_kernel_hhh)
    INSTANCE(at::kHalf,  at::kHalf,  at::kFloat, half,  half,  float, add_kernel_hhf)
    INSTANCE(at::kHalf,  at::kFloat, at::kHalf,  half,  float, half , add_kernel_hfh)
    INSTANCE(at::kHalf,  at::kFloat, at::kFloat, half,  float, float, add_kernel_hff)
    INSTANCE(at::kFloat, at::kHalf,  at::kHalf,  float, half,  half , add_kernel_fhh)
    INSTANCE(at::kFloat, at::kHalf,  at::kFloat, float, half,  float, add_kernel_fhf)
    INSTANCE(at::kFloat, at::kFloat, at::kHalf,  float, float, half , add_kernel_ffh)
    INSTANCE(at::kFloat, at::kFloat, at::kFloat, float, float, float, add_kernel_fff)

    #undef INSTANCE

    cuda_check(cudaPeekAtLastError());
}

void add
(
    const at::Tensor& x,
    const at::Tensor& y,
    at::Tensor& z
)
{
    add_gr(x, y, z, nullptr);
}

// Strided row-block copy: dst[r, :width] = src[r, :width] for 2D tensors whose row strides may
// differ (zero-padded staging buffers around GEMMs with padded dims)

#define C2D_THREADS 256

template <typename T>
__global__ void copy2d_kernel
(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const int src_stride,
    const int dst_stride,
    const int width
)
{
    int col = blockIdx.x * C2D_THREADS + threadIdx.x;
    int row = blockIdx.y;
    if (col >= width) return;
    dst[(int64_t) row * dst_stride + col] = src[(int64_t) row * src_stride + col];
}

void copy2d_gr
(
    const at::Tensor& src,
    at::Tensor& dst,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(src.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(src.dim() == 2 && dst.dim() == 2, "copy2d: tensors must be 2D");
    TORCH_CHECK(src.size(0) == dst.size(0), "copy2d: row count mismatch");
    TORCH_CHECK(src.dtype() == dst.dtype(), "copy2d: dtype mismatch");
    int rows = (int) src.size(0);
    int width = (int) MIN(src.size(1), dst.size(1));
    dim3 grid(CEIL_DIVIDE(width, C2D_THREADS), rows);

    #define INSTANCE(dt, T) \
    if (src.dtype() == dt) \
    { \
        copy2d_kernel<T><<<grid, C2D_THREADS, 0, stream>>> \
        ( \
            (const T*) src.data_ptr(), \
            (T*) dst.data_ptr(), \
            (int) src.stride(0), \
            (int) dst.stride(0), \
            width \
        ); \
        if (graph) graph->record_param((void*) &copy2d_kernel<T>, GP_copy2d_src, 0); \
        if (graph) graph->record_param((void*) &copy2d_kernel<T>, GP_copy2d_dst, 1); \
        if (graph) graph->record_param((void*) &copy2d_kernel<T>, GP_end, 0); \
    }

    INSTANCE(at::kHalf, half)
    INSTANCE(at::kFloat, float)

    #undef INSTANCE

    cuda_check(cudaPeekAtLastError());
}

// Per-expert bias adds for the block-sparse MLP graphs. All inputs are static buffers (the
// routing kernel writes sel/weights in place), so the nodes never need patching:
// - moe_bias_add: interm[k, :] += bias[sel[k]] for the gate/up intermediates
// - moe_bias_add_weighted: out[0, :] += sum_k w[k] * bias[sel[k]], correcting the weighted
//   expert reduction for the down bias (bias applies before the routing weight)

__global__ void moe_bias_add_kernel
(
    half* __restrict__ interm,
    const uintptr_t* __restrict__ bias_ptrs,
    const int64_t* __restrict__ sel,
    const int stride,
    const int width,
    const int min_expert,
    const int max_expert
)
{
    int col = blockIdx.x * C2D_THREADS + threadIdx.x;
    int k = blockIdx.y;
    if (col >= width) return;
    // Expert-parallel split: sel holds global expert indices but the pointer table only covers the
    // local range, and the mgemm PACKS its output rows to the local entries of sel (in order). Skip
    // foreign experts and add each local expert's bias at its packed row, not its position in sel
    int64_t e = sel[k];
    int row = k;
    if (min_expert >= 0)
    {
        if (e < min_expert || e >= max_expert) return;
        e -= min_expert;
        row = 0;
        for (int i = 0; i < k; ++i)
        {
            int64_t ei = sel[i];
            if (ei >= min_expert && ei < max_expert) row++;
        }
    }
    const half* b = (const half*) bias_ptrs[e];
    int64_t i = (int64_t) row * stride + col;
    interm[i] = __hadd(interm[i], b[col]);
}

__global__ void moe_bias_add_weighted_kernel
(
    float* __restrict__ out,
    const uintptr_t* __restrict__ bias_ptrs,
    const int64_t* __restrict__ sel,
    const half* __restrict__ weights,
    const int num_sel,      // experts per token (top_k); grid.y = token
    const int width,
    const int min_expert,
    const int max_expert,
    const int out_stride    // row stride of out, in elements
)
{
    int col = blockIdx.x * C2D_THREADS + threadIdx.x;
    int t = blockIdx.y;
    if (col >= width) return;
    const int64_t* sel_t = sel + (int64_t) t * num_sel;
    const half* weights_t = weights + (int64_t) t * num_sel;
    float* out_t = out + (int64_t) t * out_stride;
    float acc = out_t[col];
    for (int k = 0; k < num_sel; ++k)
    {
        // Foreign experts contribute on their own rank only
        int64_t e = sel_t[k];
        if (min_expert >= 0)
        {
            if (e < min_expert || e >= max_expert) continue;
            e -= min_expert;
        }
        const half* b = (const half*) bias_ptrs[e];
        acc += __half2float(weights_t[k]) * __half2float(b[col]);
    }
    out_t[col] = acc;
}

void moe_bias_add_gr
(
    at::Tensor& interm,
    const at::Tensor& bias_ptrs,
    const at::Tensor& sel,
    int min_expert,
    int max_expert,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(interm.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(interm, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    int num_sel = (int) sel.numel();
    int width = (int) interm.size(-1);
    int stride = (int) interm.stride(0);
    dim3 grid(CEIL_DIVIDE(width, C2D_THREADS), num_sel);

    moe_bias_add_kernel<<<grid, C2D_THREADS, 0, stream>>>
    (
        (half*) interm.data_ptr(),
        (const uintptr_t*) bias_ptrs.data_ptr(),
        (const int64_t*) sel.data_ptr(),
        stride,
        width,
        min_expert,
        max_expert
    );
    if (graph)
    {
        graph->record_param((void*) &moe_bias_add_kernel, GP_moe_bias_add_sel, 2);
        graph->record_param((void*) &moe_bias_add_kernel, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());

}

void moe_bias_add_weighted_gr
(
    at::Tensor& out,
    const at::Tensor& bias_ptrs,
    const at::Tensor& sel,
    const at::Tensor& weights,
    int min_expert,
    int max_expert,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(out.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(out, kFloat);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(weights, kHalf);
    // sel/weights are [num_tokens, experts_per_token]; out has a row per token (row 0 only when
    // num_tokens == 1, the legacy/bsz-1 case -- unchanged behavior there)
    int num_tokens = (int) sel.size(0);
    int num_sel = (int) sel.size(-1);
    int width = (int) out.size(-1);
    int out_stride = (int) out.stride(0);
    dim3 grid(CEIL_DIVIDE(width, C2D_THREADS), num_tokens);

    moe_bias_add_weighted_kernel<<<grid, C2D_THREADS, 0, stream>>>
    (
        (float*) out.data_ptr(),
        (const uintptr_t*) bias_ptrs.data_ptr(),
        (const int64_t*) sel.data_ptr(),
        (const half*) weights.data_ptr(),
        num_sel,
        width,
        min_expert,
        max_expert,
        out_stride
    );
    if (graph)
    {
        graph->record_param((void*) &moe_bias_add_weighted_kernel, GP_moe_bias_add_weighted_sel, 2);
        graph->record_param((void*) &moe_bias_add_weighted_kernel, GP_moe_bias_add_weighted_weights, 3);
        graph->record_param((void*) &moe_bias_add_weighted_kernel, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());
}
