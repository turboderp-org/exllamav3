#include <cuda_fp16.h>
#include "norm.cuh"
#include "graph.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

#define NUM_THREADS 1024
using bfloat16 = __nv_bfloat16;

template <int num_threads>
__device__ inline float reduce(float sum, int warp_id, int lane_id)
{
    if constexpr (num_threads <= 32)
    {
        // Shuffle to sum across lanes
        __shared__ float sums[num_threads / 32];
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        return sum;
    }
    else
    {
        // Shuffle to sum across lanes
        __shared__ float sums[num_threads / 32];
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
        if (lane_id == 0) sums[warp_id] = sum;
        __syncthreads();

        // Load partial sums from across warps, shuffle again across lanes
        #if defined(USE_ROCM)
            sum = lane_id < num_threads / 32 ? sums[lane_id] : 0.0f;
        #else
            sum = sums[lane_id];
        #endif
        for(int offset = warpSize / 2; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);

        return sum;
    }
}

template <bool clamp>
__device__ inline void read_half4(float4& f4, const half4* addr)
{
    half4 h4;
    READ64(h4, addr);
    f4.x = LOW_TO_FLOAT(h4.x);
    f4.y = HIGH_TO_FLOAT(h4.x);
    f4.z = LOW_TO_FLOAT(h4.y);
    f4.w = HIGH_TO_FLOAT(h4.y);
    if constexpr (clamp)
    {
        f4.x = CLAMP_FP16(f4.x);
        f4.y = CLAMP_FP16(f4.y);
        f4.z = CLAMP_FP16(f4.z);
        f4.w = CLAMP_FP16(f4.w);
    }
}

__device__ inline void read_bfloat164(float4& f4, const bfloat164* addr)
{
    bfloat164 h4;
    READ64(h4, addr);
    f4.x = __bfloat162float(__low2bfloat16(h4.x));
    f4.y = __bfloat162float(__high2bfloat16(h4.x));
    f4.z = __bfloat162float(__low2bfloat16(h4.y));
    f4.w = __bfloat162float(__high2bfloat16(h4.y));
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

__device__ inline void write_bfloat164(const float4& f4, bfloat164* addr)
{
    bfloat164 h4
    (
        __halves2bfloat162(__float2bfloat16_rz(f4.x), __float2bfloat16_rz(f4.y)),
        __halves2bfloat162(__float2bfloat16_rz(f4.z), __float2bfloat16_rz(f4.w))
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

__device__ inline void apply4_nw(float4& x4, const float rmf)
{
    x4.x = x4.x * rmf;
    x4.y = x4.y * rmf;
    x4.z = x4.z * rmf;
    x4.w = x4.w * rmf;
}

__device__ __forceinline__ float _silu(float x)
{
    float e     = __expf(-x);
    float recip = __fdividef(1.0f, 1.0f + e);
    return x * recip;
}


// Block-size-agnostic reduction (any multiple of 32 threads)
__device__ inline float reduce_dyn(float sum, int warp_id, int lane_id)
{
    __shared__ float sums[32];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    int num_warps = blockDim.x / 32;
    if (num_warps == 1) return sum;
    if (lane_id == 0) sums[warp_id] = sum;
    __syncthreads();
    sum = lane_id < num_warps ? sums[lane_id] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    return sum;
}

// res_mode 0: y = norm(x) * w
// res_mode 1: y += norm(x) * w                            (post-norm residual accumulate)
// res_mode 2: r += x; y = norm(r) * w                     (fused pre-norm residual add)
#define RES_NONE 0
#define RES_POST 1
#define RES_IN 2

template <int res_mode, typename input_t, typename output_t, typename weight_t, typename residual_t>
__global__ __launch_bounds__(NUM_THREADS)
void rms_norm_kernel
(
    const input_t* __restrict__ x,
    const weight_t* __restrict__ w,
    output_t* __restrict__ y,
    residual_t* __restrict__ r,
    const float epsilon,
    const int rows,
    const int dim,
    const float constant_bias,
    const float constant_scale
)
{
    constexpr bool input_fp32 = std::is_same_v<input_t, float>;
    constexpr bool output_fp32 = std::is_same_v<output_t, float>;
    constexpr bool input_fp16 = std::is_same_v<input_t, half>;
    constexpr bool output_fp16 = std::is_same_v<output_t, half>;
    static_assert(input_fp32 || input_fp16, "rms_norm_kernel: input must be float or half type");
    static_assert(output_fp32 || output_fp16, "rms_norm_kernel: output must be float or half type");
    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool residual_fp16 = std::is_same_v<residual_t, half>;

    int t = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;

    int columns = dim / 4;
    bool single = columns <= blockDim.x;

    auto read_in = [&] (float4& f4, const input_t* addr)
    {
        if constexpr (input_fp16) read_half4<true>(f4, (const half4*) addr);
        if constexpr (input_fp32) read_float4(f4, (const float4*) addr);
    };

    auto add_resid_in = [&] (float4& x4, int column)
    {
        // r += x, rounded to the residual dtype so the result matches an unfused add
        float4 r4;
        if constexpr (residual_fp16) read_half4<false>(r4, ((const half4*) (r + row * dim)) + column);
        else                         read_float4(r4, ((const float4*) (r + row * dim)) + column);
        x4.x += r4.x;
        x4.y += r4.y;
        x4.z += r4.z;
        x4.w += r4.w;
        if constexpr (residual_fp16)
        {
            half4 h4
            (
                __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y)),
                __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w))
            );
            WRITE64(((half4*) (r + row * dim)) + column, h4);
            x4.x = LOW_TO_FLOAT(h4.x);
            x4.y = HIGH_TO_FLOAT(h4.x);
            x4.z = LOW_TO_FLOAT(h4.y);
            x4.w = HIGH_TO_FLOAT(h4.y);
        }
        else
            write_float4(x4, ((float4*) (r + row * dim)) + column);
    };

    auto apply_out = [&] (float4& x4, int column, float rmf)
    {
        if (w)
        {
            float4 w4;
            if constexpr (weight_bf16) read_bfloat164   (w4, ((const bfloat164*) w) + column);
            else                       read_half4<false>(w4, ((const half4*)     w) + column);
            if (constant_bias != 0.0f)
            {
                w4.x += constant_bias;
                w4.y += constant_bias;
                w4.z += constant_bias;
                w4.w += constant_bias;
            }
            apply4(x4, w4, rmf);
        }
        else
        {
            apply4_nw(x4, rmf);
        }

        if constexpr (res_mode == RES_POST)
        {
            float4 r4;
            if constexpr (output_fp16) read_half4<false>(r4, ((half4*) (y + row * dim)) + column);
            if constexpr (output_fp32) read_float4(r4, ((float4*) (y + row * dim)) + column);
            x4.x += r4.x;
            x4.y += r4.y;
            x4.z += r4.z;
            x4.w += r4.w;
        }

        if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row * dim)) + column);
        if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row * dim)) + column);
    };

    if (single)
    {
        // One float4 per thread: keep the value in a register between the two phases
        float4 x4 = {};
        float sum = 0.0f;
        if (t < columns)
        {
            read_in(x4, x + row * dim + 4 * t);
            if constexpr (res_mode == RES_IN) add_resid_in(x4, t);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;
        if (t < columns)
            apply_out(x4, t, rmf);
    }
    else
    {
        float sum = 0.0f;
        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            read_in(x4, x + row * dim + 4 * column);
            if constexpr (res_mode == RES_IN) add_resid_in(x4, column);
            sum = sum_sq4(sum, x4);
        }
        sum = reduce_dyn(sum, warp_id, lane_id);
        float rmf = rsqrtf(sum / (float) dim + epsilon) * constant_scale;

        for (int column = t; column < columns; column += blockDim.x)
        {
            float4 x4;
            // For RES_IN the summed values were written back to r in the first pass
            if constexpr (res_mode == RES_IN)
            {
                if constexpr (residual_fp16) read_half4<false>(x4, ((const half4*) (r + row * dim)) + column);
                else                         read_float4(x4, ((const float4*) (r + row * dim)) + column);
            }
            else
                read_in(x4, x + row * dim + 4 * column);
            apply_out(x4, column, rmf);
        }
    }
}

/*
Compute RMSNorm: y = x * w / sqrt(row_mean(x * x) + epsilon)
- Can operate in-place if y == x
- x can be either float or half dtype
- y can be either float or half dtype
- w can be either bfloat16 or half dtype
*/
void rms_norm_impl
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    c10::optional<at::Tensor> r,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool span_heads,
    int res_mode,
    Graph* graph = nullptr
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    if (span_heads)
    {
        x = x.flatten(-2);
        y = y.flatten(-2);
    }

    TORCH_CHECK_DIV(x, -1, 4);
    TORCH_CHECK_SHAPES_FULL(x, y);

    auto tx = x.scalar_type();
    auto tw = at::kHalf;  // intentional, type is irrelevant if w is None
    auto ty = y.scalar_type();

    const half* w_ptr = (const half*) OPTPTR(w);
    if (w_ptr)
    {
        TORCH_CHECK_SHAPES(x, -1, w.value(), 0, 1);
        tw = w.value().scalar_type();
    }

    void* r_ptr = OPTPTR(r);
    if (res_mode == RES_IN)
    {
        TORCH_CHECK(r_ptr, "rms_norm: res_mode RES_IN requires residual tensor");
        TORCH_CHECK_SHAPES_FULL(x, r.value());
    }

    int rows = 1;
    for (int i = 0; i < x.dim() - 1; ++i) rows *= x.size(i);
    int dim = x.size(-1);

    // Size the block to the row so short rows don't idle warps through the reduction
    int threads = MIN(NUM_THREADS, CEIL_DIVIDE(dim / 4, 32) * 32);

    dim3 blockDim(threads, 1, 1);
    dim3 gridDim(rows, 1, 1);

    auto tr = res_mode == RES_IN ? r.value().scalar_type() : tx;

    // Launch macro
    #define __(_tx, __tx, _tw, __tw, _ty, __ty, _res, _tr, __tr)                                   \
    if (tx == at::_tx && tw == at::_tw && ty == at::_ty && res_mode == _res && tr == at::_tr)      \
        rms_norm_kernel<_res, __tx, __ty, __tw, __tr><<<gridDim, blockDim, 0, stream>>>            \
        (                                                                                          \
            (const __tx*) x.data_ptr(),                                                            \
            (const __tw*) w_ptr,                                                                   \
            (__ty*) y.data_ptr(),                                                                  \
            (__tr*) r_ptr,                                                                         \
            epsilon,                                                                \
            rows,                                                                   \
            dim,                                                                    \
            constant_bias,                                                          \
            constant_scale                                                          \
        );

    //      x_type________ w_type_____________  y_type_______        mode      r_type
         __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_NONE, kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kFloat, float, RES_NONE, kHalf,  half)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_NONE, kFloat, float)
    else __(kFloat, float, kHalf,     half,     kFloat, float, RES_NONE, kFloat, float)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_NONE, kHalf,  half)
    else __(kHalf,  half,  kBFloat16, bfloat16, kFloat, float, RES_NONE, kHalf,  half)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_NONE, kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kFloat, float, RES_NONE, kFloat, float)
    else __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_POST, kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kFloat, float, RES_POST, kHalf,  half)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_POST, kFloat, float)
    else __(kFloat, float, kHalf,     half,     kFloat, float, RES_POST, kFloat, float)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_POST, kHalf,  half)
    else __(kHalf,  half,  kBFloat16, bfloat16, kFloat, float, RES_POST, kHalf,  half)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_POST, kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kFloat, float, RES_POST, kFloat, float)
    else __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kHalf,  half,  kHalf,     half,     kHalf,  half,  RES_IN,   kFloat, float)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kFloat, float, kHalf,     half,     kHalf,  half,  RES_IN,   kFloat, float)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kHalf,  half,  kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kFloat, float)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kHalf,  half)
    else __(kFloat, float, kBFloat16, bfloat16, kHalf,  half,  RES_IN,   kFloat, float)

    else TORCH_CHECK(false, "rms_norm: Invalid datatypes for input/output");
    #undef __

    cuda_check(cudaPeekAtLastError());
}

void rms_norm
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool span_heads,
    bool add_residual
)
{
    rms_norm_impl(x, w, y, {}, epsilon, constant_bias, constant_scale, span_heads,
                  add_residual ? RES_POST : RES_NONE);
}

// Graphable variant: launches on the capture stream, records nothing (BC callers norm between
// static buffers)
void rms_norm_gr
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    Graph* graph
)
{
    rms_norm_impl(x, w, y, {}, epsilon, constant_bias, constant_scale, false, RES_NONE, graph);
}

// Fused pre-norm residual add: r += x (in place), y = norm(r) * w
void rms_norm_res_in
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    at::Tensor r,
    float epsilon,
    float constant_bias,
    float constant_scale
)
{
    rms_norm_impl(x, w, y, r, epsilon, constant_bias, constant_scale, false, RES_IN);
}


template <int num_threads, typename output_t, typename weight_t, typename gate_t>
__global__ __launch_bounds__(num_threads)
void gated_rms_norm_kernel
(
    const bfloat16* __restrict__ x,
    const weight_t* __restrict__ w,
    output_t* __restrict__ y,
    const gate_t* __restrict__ g,
    const float epsilon,
    const int rows,
    const int dim,
    float constant_bias,
    const int w_groups,         // weight spans w_groups rows, cycled by row index (Mamba2 group norm)
    const bool gate_first       // apply silu(g) before the norm instead of after (Mamba2 style)
)
{
    constexpr bool output_fp32 = std::is_same_v<output_t, float>;
    constexpr bool output_fp16 = std::is_same_v<output_t, half>;
    constexpr bool weight_bf16 = std::is_same_v<weight_t, bfloat16>;
    constexpr bool gate_fp32   = std::is_same_v<gate_t,   float>;

    int t = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row = blockIdx.x;

    int columns = dim / 4;
    const weight_t* w_row = w + (size_t) (row % w_groups) * dim;

    // Compute sum of squares
    float sum = 0.0f;
    for (int column = t; column < columns; column += num_threads)
    {
        float4 x4;
        read_bfloat164(x4, ((const bfloat164*) (x + row * dim)) + column);
        if (gate_first)
        {
            float4 g4;
            if constexpr (gate_fp32)   read_float4   (g4, ((const float4*)    (g + row * dim)) + column);
            else                       read_bfloat164(g4, ((const bfloat164*) (g + row * dim)) + column);
            x4.x *= _silu(g4.x);
            x4.y *= _silu(g4.y);
            x4.z *= _silu(g4.z);
            x4.w *= _silu(g4.w);
        }
        sum = sum_sq4(sum, x4);
    }
    sum = reduce<num_threads>(sum, warp_id, lane_id);

    // Get norm
    float rmf = rsqrtf(sum / (float)dim + epsilon);

    // Normalize x, scaling by w * silu(g)
    for (int column = t; column < columns; column += num_threads)
    {
        float4 x4;
        float4 w4;
        float4 g4;

        read_bfloat164(x4, ((const bfloat164*) (x + row * dim)) + column);
        if constexpr (weight_bf16) read_bfloat164(w4, ((const bfloat164*) w_row) + column);
        else                       read_float4   (w4, ((const float4*)    w_row) + column);
        if constexpr (gate_fp32)   read_float4   (g4, ((const float4*)    (g + row * dim)) + column);
        else                       read_bfloat164(g4, ((const bfloat164*) (g + row * dim)) + column);

        if (constant_bias != 0.0f)
        {
            w4.x += constant_bias;
            w4.y += constant_bias;
            w4.z += constant_bias;
            w4.w += constant_bias;
        }

        if (gate_first)
        {
            x4.x *= _silu(g4.x);
            x4.y *= _silu(g4.y);
            x4.z *= _silu(g4.z);
            x4.w *= _silu(g4.w);

            apply4(x4, w4, rmf);
        }
        else
        {
            apply4(x4, w4, rmf);

            x4.x *= _silu(g4.x);
            x4.y *= _silu(g4.y);
            x4.z *= _silu(g4.z);
            x4.w *= _silu(g4.w);
        }

        if constexpr (output_fp16) write_half4(x4, ((half4*) (y + row * dim)) + column);
        if constexpr (output_fp32) write_float4(x4, ((float4*) (y + row * dim)) + column);
    }
}


/*
Compute RMSNorm: y = x * w / sqrt(row_mean(x * x) + epsilon) * silu(g)
- bfloat16 input only, half/float output
- w_groups > 1: w holds w_groups weight rows of size dim, selected by (row % w_groups). Used for
  Mamba2 group norm where the norm spans dim channels but the weight covers the full inner dim
- gate_first: y = norm(x * silu(g)) * w instead of norm(x) * w * silu(g) (Mamba2 style)
*/
void gated_rms_norm_gr
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias,
    Graph* graph,
    int w_groups,
    bool gate_first
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIV(x, -1, 4);
    if (w_groups == 1)
    {
        TORCH_CHECK_SHAPES(x, -1, w, 0, 1);
    }
    else
    {
        TORCH_CHECK(w.numel() == (int64_t) w_groups * x.size(-1),
                    "gated_rms_norm: w must have w_groups * dim elements");
    }
    TORCH_CHECK_SHAPES_FULL(x, g);

    int rows = 1;
    for (int i = 0; i < x.dim() - 1; ++i) rows *= x.size(i);
    int dim = x.size(-1);

    bool small = (dim <= 256);

    dim3 blockDim(small ? 32 : NUM_THREADS, 1, 1);
    dim3 gridDim(rows, 1, 1);

    auto tx = x.dtype();
    auto tw = w.dtype();
    auto ty = y.dtype();
    auto tg = g.dtype();

    // Launch macro
    #define __(_tx, __tx, _tw, __tw, _ty, __ty, _tg, __tg, _small, __num_threads)               \
    if (small == _small && tx == at::_tx && tw == at::_tw && ty == at::_ty && tg == at::_tg)    \
        gated_rms_norm_kernel<__num_threads><<<gridDim, blockDim, 0, stream>>>                  \
        (                                                                                       \
            (const __tx*) x.data_ptr(),                                                         \
            (const __tw*) w.data_ptr(),                                                         \
            (__ty*) y.data_ptr(),                                                               \
            (const __tg*) g.data_ptr(),                                                         \
            epsilon,                                                                            \
            rows,                                                                               \
            dim,                                                                                \
            constant_bias,                                                                      \
            w_groups,                                                                           \
            gate_first                                                                          \
        );

    //      x_type_____________  w_type_____________  y_type_______  g_type_____________  small  num_threads
         __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kBFloat16, bfloat16, true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kBFloat16, bfloat16, false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kHalf,  half,  kFloat,    float,    false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kFloat,    float,    kFloat, float, kFloat,    float,    false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kHalf,  half,  kFloat,    float,    false, NUM_THREADS)
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kFloat,    float,    true,  32         )
    else __(kBFloat16, bfloat16, kBFloat16, bfloat16, kFloat, float, kFloat,    float,    false, NUM_THREADS)

    else TORCH_CHECK(false, "gated_rms_norm: Invalid datatypes for input/output");
    #undef __

    cuda_check(cudaPeekAtLastError());
}

void gated_rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias,
    int w_groups,
    bool gate_first
)
{
    gated_rms_norm_gr(x, w, y, g, epsilon, constant_bias, nullptr, w_groups, gate_first);
}
