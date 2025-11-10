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
