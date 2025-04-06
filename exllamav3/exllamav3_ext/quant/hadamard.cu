#include "quantize.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"

__device__ inline half hreduce(half2 x)
{
    return __hadd(__low2half(x), __high2half(x));
}

__device__ inline float shuffle_had_fx32(float v, int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        float pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80000000);
        v = v + pv;
    }
    return v;
}

__device__ inline half2 shuffle_had_h2x32(half2 v, int lane_id)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        half2 pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t* vi = reinterpret_cast<uint32_t*>(&v);
        int32_t sfm = -static_cast<int16_t>(lane_id & i) >> 31;
        *vi ^= (sfm & 0x80008000);
        v = __hadd2(v, pv);
    }
    return v;
}

__global__ __launch_bounds__(32)
void hadh_r_128_kernel
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale
)
{
    int t = threadIdx.x;
    input_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // Prescale
    if (pre_scale)
    {
        pre_scale += blockIdx.y * 128;
        half4 s = ((half4*) pre_scale)[t];
        v.x = __h2div(v.x, s.x);
        v.y = __h2div(v.y, s.y);
    }

    // 4 element had
    half2 vxpp = v.x;
    half2 vxpn = h2xor(vxpp, 0x80000000);
    half2 vypp = v.y;
    half2 vypn = h2xor(vypp, 0x80000000);
    half h0 = hreduce(__hadd2(vxpp, vypp));
    half h1 = hreduce(__hadd2(vxpn, vypn));
    half h2 = hreduce(__hsub2(vxpp, vypp));
    half h3 = hreduce(__hsub2(vxpn, vypn));
    v.x = __halves2half2(h0, h1);
    v.y = __halves2half2(h2, h3);

    // 32 element had, warp shuffle
    v.x = shuffle_had_h2x32(v.x, t);
    v.y = shuffle_had_h2x32(v.y, t);

    // Rescale by 1/sqrt(128)
    half2 f = __halves2half2(__float2half_rn(0.088388347648), __float2half_rn(0.088388347648));
    v.x = __hmul2(v.x, f);
    v.y = __hmul2(v.y, f);

    // Postscale
    if (post_scale)
    {
        post_scale += blockIdx.y * 128;
        half4 s = ((half4*) post_scale)[t];
        v.x = __h2div(v.x, s.x);
        v.y = __h2div(v.y, s.y);
    }

    // Store
    ((half4*) output_ptr)[t] = v;
}

__global__ __launch_bounds__(32)
void hadf_r_128_kernel
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
//    const uint16_t* __restrict__ pre_flip,
    const half* __restrict__ pre_scale,
    const uint16_t* __restrict__ post_flip,
    float r_scale
)
{
    int t = threadIdx.x;
    input_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
    output_ptr += gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // Pre flip
//    if (pre_flip)
//    {
//        int i = blockIdx.y * 32 + t;
//        uint32_t signs = (uint32_t) (pre_flip[i / 4] >> ((i % 4) * 4));
//        v.x = h2xor(v.x, ((signs & 1) << 15) | ((signs & 2) << 30));
//        v.y = h2xor(v.y, ((signs & 4) << 13) | ((signs & 8) << 28));
//    }
    __syncthreads();

    if (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) pre_scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }
    __syncthreads();

    // 4 element had
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float h0 = v0 + v1 + v2 + v3;
    float h1 = v0 - v1 + v2 - v3;
    float h2 = v0 + v1 - v2 - v3;
    float h3 = v0 - v1 - v2 + v3;

    // 32 element had, warp shuffle
    h0 = shuffle_had_fx32(h0, t);
    h1 = shuffle_had_fx32(h1, t);
    h2 = shuffle_had_fx32(h2, t);
    h3 = shuffle_had_fx32(h3, t);
    h0 *= r_scale;
    h1 *= r_scale;
    h2 *= r_scale;
    h3 *= r_scale;
    v.x = __floats2half2_rn(h0, h1);
    v.y = __floats2half2_rn(h2, h3);

    // Post flip
    if (post_flip)
    {
        int i = blockIdx.y * 32 + t;
        uint32_t signs = (uint32_t) (post_flip[i / 4] >> ((i % 4) * 4));
        v.x = h2xor(v.x, ((signs & 1) << 15) | ((signs & 2) << 30));
        v.y = h2xor(v.y, ((signs & 4) << 13) | ((signs & 8) << 28));
    }

    // Store
    ((half4*) output_ptr)[t] = v;
}

/*
Compute y = (x.view(-1, 128) @ had_128).view(x.shape)
Works inplace if y == x
*/
void had_r_128
(
    const at::Tensor& input,
    const at::Tensor& output,
//    const c10::optional<at::Tensor>& pre_flip,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_flip,
    float scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(input, 2);
    TORCH_CHECK_SHAPES_FULL(input, output);
    TORCH_CHECK_DTYPE(input, kHalf);
    TORCH_CHECK_DTYPE(output, kHalf);
    TORCH_CHECK_DIV(input, 1, 128);

    int rows = input.size(0);
    int cols = input.size(1);
    int blocks = cols / 128;
    float r_scale = scale * 0.088388347648f; // scale / sqrt(128)

    dim3 blockDim(32);
    dim3 gridDim(rows, blocks);

    hadf_r_128_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (const half*) input.data_ptr(),
        (half*) output.data_ptr(),
//        (const uint16_t*) OPTPTR(pre_flip),
        (const half*) OPTPTR(pre_scale),
        (const uint16_t*) OPTPTR(post_flip),
        r_scale
    );
}