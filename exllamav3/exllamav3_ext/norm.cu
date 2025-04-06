#include "norm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "util.h"
#include "util.cuh"

#define NUM_THREADS 1024

#if defined(USE_ROCM)
#define NUM_WARPS (1024 / warpSize)
#define WARP_SIZE (warpSize)
#else
#define NUM_WARPS 32
#define WARP_SIZE 32
#endif

__device__ inline float reduce(float sum, int warp_id, int lane_id)
{
    // Shuffle to sum across lanes
    __shared__ float sums[NUM_WARPS];
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();

    // Load partial sums from across warps, shuffle again across lanes
    #if defined(USE_ROCM)
        sum = lane_id < NUM_WARPS ? sums[lane_id] : 0.0f;
    #else
        sum = sums[lane_id];
    #endif
    for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

    return sum;
}

__device__ inline void read_half4(float4& f4, const half4* addr, bool clamp)
{
    half4 h4;
    READ64(h4, addr);
    f4.x = LOW_TO_FLOAT(h4.x);
    f4.y = HIGH_TO_FLOAT(h4.x);
    f4.z = LOW_TO_FLOAT(h4.y);
    f4.w = HIGH_TO_FLOAT(h4.y);
    if (clamp)
    {
        f4.x = CLAMP_FP16(f4.x);
        f4.y = CLAMP_FP16(f4.y);
        f4.z = CLAMP_FP16(f4.z);
        f4.w = CLAMP_FP16(f4.w);
    }
}

__device__ inline void read_float4(float4& f4, const float4* addr)
{
    READ128(f4, addr);
}

__device__ inline void write_half4(const float4& f4, half4* addr)
{
    half4 h4
    (
        __halves2half2(__float2half_rn(f4.x), __float2half_rn(f4.y)),
        __halves2half2(__float2half_rn(f4.z), __float2half_rn(f4.w))
    );
    WRITE64(addr, h4);
}

__device__ inline void write_float4(const float4& f4, float4* addr)
{
    WRITE128(addr, f4);
}

__device__ inline float sum_sq4(float lsum, const float4& f4)
{
    lsum = fma(f4.x, f4.x, lsum);
    lsum = fma(f4.y, f4.y, lsum);
    lsum = fma(f4.z, f4.z, lsum);
    lsum = fma(f4.w, f4.w, lsum);
    return lsum;
}

__device__ inline void apply4(float4& x4, const float4& w4, const float rmf)
{
    x4.x = x4.x * w4.x * rmf;
    x4.y = x4.y * w4.y * rmf;
    x4.z = x4.z * w4.z * rmf;
    x4.w = x4.w * w4.w * rmf;
}

template <typename input_t, typename output_t>
__global__ __launch_bounds__(NUM_THREADS)
void rms_norm_kernel
(
    const input_t* __restrict__ x,
    const half* __restrict__ w,
    output_t* __restrict__ y,
    const float epsilon,
    const int rows,
    const int dim,
    float constant_bias
)
{
    constexpr bool input_fp32 = std::is_same_v<input_t, float>;
    constexpr bool output_fp32 = std::is_same_v<output_t, float>;
    constexpr bool input_fp16 = std::is_same_v<input_t, half>;
    constexpr bool output_fp16 = std::is_same_v<output_t, half>;
    static_assert(input_fp32 || input_fp16, "rms_norm_kernel: input must be float or half type");
    static_assert(output_fp32 || output_fp16, "rms_norm_kernel: output must be float or half type");

    int t = threadIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x;

    int columns = dim / 4;

    // Compute sum of squares
    float sum = 0.0f;
    for (int column = t; column < columns; column += NUM_THREADS)
    {
        float4 x4;
        if constexpr (input_fp16) read_half4(x4, ((const half4*) (x + row * dim)) + column, true);
        if constexpr (input_fp32) read_float4(x4, ((const float4*) (x + row * dim)) + column);
        sum = sum_sq4(sum, x4);
    }
    sum = reduce(sum, warp_id, lane_id);

    // Get norm
    float rmf = rsqrtf(sum / (float)dim + epsilon);

    // Normalize x, scaling by w
    for (int column = t; column < columns; column += NUM_THREADS)
    {
        float4 x4;
        if constexpr (input_fp16) read_half4(x4, ((const half4*) (x + row * dim)) + column, true);
        if constexpr (input_fp32) read_float4(x4, ((const float4*) (x + row * dim)) + column);

        float4 w4;
        read_half4(w4, ((const half4*) w) + column, false);
        if (constant_bias != 0.0f)
        {
            w4.x += constant_bias;
            w4.y += constant_bias;
            w4.z += constant_bias;
            w4.w += constant_bias;
        }

        apply4(x4, w4, rmf);

        if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row * dim)) + column);
        if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row * dim)) + column);
    }
}

/*
Compute RMSNorm: y = x * w / sqrt(row_mean(x * x) + epsilon)
- Can operate in-place if y == x
- x can be either float or half dtype
- y can be either float or half dtype
- w must be half dtype
*/
void rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    float epsilon,
    float constant_bias
)
{
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DIV(x, 1, 4);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, y, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, y, 1, 1);

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    bool input_fp32 = x.dtype() == at::kFloat;
    bool output_fp32 = y.dtype() == at::kFloat;
    bool input_fp16 = !input_fp32;
    bool output_fp16 = !output_fp32;
    int rows = x.size(0);
    int dim = x.size(1);
    dim3 blockDim(NUM_THREADS, 1, 1);
    dim3 gridDim(rows, 1, 1);

    if (input_fp16 && output_fp16)
        rms_norm_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const half*) x.data_ptr(),
            (const half*) w.data_ptr(),
            (half*) y.data_ptr(),
            epsilon,
            rows,
            dim,
            constant_bias
        );
    else if (input_fp16 && output_fp32)
        rms_norm_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const half*) x.data_ptr(),
            (const half*) w.data_ptr(),
            (float*) y.data_ptr(),
            epsilon,
            rows,
            dim,
            constant_bias
        );
    else if (input_fp32 && output_fp16)
        rms_norm_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (const half*) w.data_ptr(),
            (half*) y.data_ptr(),
            epsilon,
            rows,
            dim,
            constant_bias
        );
    else if (input_fp32 && output_fp32)
        rms_norm_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const float*) x.data_ptr(),
            (const half*) w.data_ptr(),
            (float*) y.data_ptr(),
            epsilon,
            rows,
            dim,
            constant_bias
        );
    else
        TORCH_CHECK(false, "rms_norm: Invalid datatypes for input/output, must be half or float")
}