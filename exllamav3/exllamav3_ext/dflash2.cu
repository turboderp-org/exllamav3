#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dflash2.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

// DFlash2 grouped dynamic causal convolution over a draft block (dflash.model
// _grouped_dynamic_convolve):
//
//     out[b, t, c] = sum_k (base[k, c] + dyn[b, t, k, c / group_size]) * x[b, t - k, c]
//
// Taps reach back within the block only (t - k < 0 contributes nothing), so the kernel is
// stateless across rounds. dyn is the kernel projection's output viewed as [b, l, K, groups]
// with arbitrary strides (the projection emits both the prepare and the finish deltas in one
// tensor). accumulate: out (fp32 residual stream) += conv(x), fusing the block's residual add
// into the finish() variant; otherwise out = conv(x)

#define NUM_THREADS 256

template <typename T> __device__ __forceinline__ float to_f(T v) { return (float) v; }
template <> __device__ __forceinline__ float to_f(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v) { return (T) v; }
template <> __device__ __forceinline__ half from_f(float v) { return __float2half(v); }

template <typename TX, typename TB, typename TO, bool accumulate>
__global__ __launch_bounds__(NUM_THREADS)
void dflash2_dynconv_kernel
(
    const TX* __restrict__ x,
    const half* __restrict__ dyn,
    const TB* __restrict__ base,
    TO* __restrict__ out,
    const int seqlen,
    const int hidden,
    const int group_size,
    const int taps,
    const int64_t ds_b,
    const int64_t ds_t,
    const int64_t ds_k,
    const int64_t ds_g
)
{
    int c = blockIdx.x * NUM_THREADS + threadIdx.x;
    if (c >= hidden) return;
    int t = blockIdx.y;
    int b = blockIdx.z;
    int g = c / group_size;

    const half* dyn_bt = dyn + b * ds_b + t * ds_t + g * ds_g;
    const TX* x_bt = x + ((int64_t) b * seqlen + t) * hidden + c;

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < taps; ++k)
    {
        if (k > t) break;
        float w = to_f(base[k * hidden + c]) + __half2float(dyn_bt[k * ds_k]);
        acc += w * to_f(x_bt[-(int64_t) k * hidden]);
    }

    TO* o = out + ((int64_t) b * seqlen + t) * hidden + c;
    if constexpr (accumulate)
        *o = from_f<TO>(to_f(*o) + acc);
    else
        *o = from_f<TO>(acc);
}

template <typename TX, typename TB, typename TO, bool accumulate>
void launch
(
    const at::Tensor& x, const at::Tensor& dyn, const at::Tensor& base, at::Tensor& out,
    int seqlen, int hidden, int group_size, int taps, int bsz, cudaStream_t stream
)
{
    dim3 grid(CEIL_DIVIDE(hidden, NUM_THREADS), seqlen, bsz);
    dflash2_dynconv_kernel<TX, TB, TO, accumulate><<<grid, NUM_THREADS, 0, stream>>>
    (
        (const TX*) x.data_ptr(),
        (const half*) dyn.data_ptr(),
        (const TB*) base.data_ptr(),
        (TO*) out.data_ptr(),
        seqlen, hidden, group_size, taps,
        dyn.stride(0), dyn.stride(1), dyn.stride(2), dyn.stride(3)
    );
}

/*
x:      (bsz, seqlen, hidden), fp16 or fp32, contiguous
dyn:    (bsz, seqlen, taps, hidden / group_size), fp16, any strides
base:   (taps, hidden), fp16 or bf16, contiguous
out:    (bsz, seqlen, hidden), fp16 or fp32, contiguous; accumulated into when accumulate
*/

void dflash2_dynconv
(
    const at::Tensor& x,
    const at::Tensor& dyn,
    const at::Tensor& base,
    at::Tensor& out,
    int64_t group_size,
    bool accumulate
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(x, 3);
    TORCH_CHECK_DIM(dyn, 4);
    TORCH_CHECK_DIM(base, 2);
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && base.is_contiguous(), "dflash2_dynconv: x, out and base must be contiguous");
    TORCH_CHECK_SHAPES_FULL(x, out);
    TORCH_CHECK_DTYPE(dyn, kHalf);
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "dflash2_dynconv: x must be fp16 or fp32");
    TORCH_CHECK(out.dtype() == at::kHalf || out.dtype() == at::kFloat, "dflash2_dynconv: out must be fp16 or fp32");
    TORCH_CHECK(base.dtype() == at::kHalf || base.dtype() == at::kBFloat16, "dflash2_dynconv: base must be fp16 or bf16");
    TORCH_CHECK(!accumulate || out.dtype() == at::kFloat, "dflash2_dynconv: accumulate needs an fp32 residual");

    int bsz = x.size(0);
    int seqlen = x.size(1);
    int hidden = x.size(2);
    int taps = base.size(0);
    TORCH_CHECK(group_size > 0 && hidden % group_size == 0, "dflash2_dynconv: hidden must be a multiple of group_size");
    TORCH_CHECK(base.size(1) == hidden, "dflash2_dynconv: base width mismatch");
    TORCH_CHECK(dyn.size(0) == bsz && dyn.size(1) == seqlen && dyn.size(2) == taps && dyn.size(3) == hidden / group_size,
                "dflash2_dynconv: dyn must be (bsz, seqlen, taps, hidden / group_size)");
    if (!bsz || !seqlen || !hidden) return;

    #define DISPATCH(TX, TB, TO) \
        if (accumulate) launch<TX, TB, TO, true>(x, dyn, base, out, seqlen, hidden, (int) group_size, taps, bsz, stream); \
        else            launch<TX, TB, TO, false>(x, dyn, base, out, seqlen, hidden, (int) group_size, taps, bsz, stream);
    #define DISPATCH_TO(TX, TB) \
        if (out.dtype() == at::kHalf) { DISPATCH(TX, TB, half) } else { DISPATCH(TX, TB, float) }
    #define DISPATCH_TB(TX) \
        if (base.dtype() == at::kHalf) { DISPATCH_TO(TX, half) } else { DISPATCH_TO(TX, __nv_bfloat16) }

    if (x.dtype() == at::kHalf) { DISPATCH_TB(half) } else { DISPATCH_TB(float) }

    #undef DISPATCH_TB
    #undef DISPATCH_TO
    #undef DISPATCH
    cuda_check(cudaPeekAtLastError());
}
