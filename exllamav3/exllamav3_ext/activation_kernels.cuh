
__device__ inline half2 clamp_half2_to_finite(half2 v)
{
    const half2 max_h2 = __float2half2_rn(65504.0f);
    const half2 min_h2 = __float2half2_rn(-65504.0f);
    return __hmax2(__hmin2(v, max_h2), min_h2);
}


__device__ __forceinline__ half _silu(half x)
{
    half one = __float2half(1.0f);
    half neg_x = __hneg(x);
    half e = hexp(neg_x);
    half sum = __hadd(one, e);
    half r = hrcp(sum);
    half result = __hmul(x, r);
    return result;
}


__device__ __forceinline__ half2 _silu(half2 x)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}


__device__ __forceinline__ float _silu(float x)
{
    float e     = __expf(-x);
    float recip = __fdividef(1.0f, 1.0f + e);
    return x * recip;
}


__device__ __forceinline__ half _gelu(half x)
{
    float xf = __half2float(x);
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float tanh_arg = c * (xf + 0.044715f * xf * xf * xf);
    xf = 0.5f * xf * (1.0 + tanh_opt(tanh_arg));
    return __float2half_rn(xf);
}


__device__ __forceinline__ float _gelu(float x)
{
    const float c = 0.797884560803f;  // sqrt(2/Pi)
    float tanh_arg = c * (x + 0.044715f * x * x * x);
    x = 0.5f * x * (1.0 + tanh_opt(tanh_arg));
    return x;
}


__device__ __forceinline__ half2 _gelu(half2 x)
{
    return __halves2half2(_gelu(__low2half(x)), _gelu(__high2half(x)));
}


__device__ __forceinline__ half _relu2(half x)
{
    float xf = __half2float(x);
    xf = fmaxf(0.0f, xf);
    xf = xf * xf;
    return __float2half_rn(xf);
}


__device__ __forceinline__ float _relu2(float x)
{
    x = fmaxf(0.0f, x);
    x = x * x;
    return x;
}


__device__ __forceinline__ half2 _relu2(half2 x)
{
    return __halves2half2(_relu2(__low2half(x)), _relu2(__high2half(x)));
}


__device__ __forceinline__ float _xielu(float x, float alpha_p, float alpha_n)
{
    const float eps = -9.9838e-07;  // -1e-6 with BF16 rounding error
    const float beta = 0.5f;
    return x > 0 ?
        alpha_p * x * x + beta * x :
        (expm1f(min(x, eps)) - x) * alpha_n + beta * x;
}


__device__ __forceinline__ float _sigmoid_fast_exp(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}


template <int activation_type>
__global__ __launch_bounds__(NUM_THREADS)
void act_mul_kernel_h
(
    const half* __restrict__ x,
    const half* __restrict__ y,
    half* __restrict__ z,
    const size_t numel
)
{
    size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);
    if (idx >= numel / 2) return;

    half2 x2 = ((const half2*) x)[idx];
    half2 y2 = ((const half2*) y)[idx];

    if constexpr (activation_type == ACT_SILU)
        x2 = _silu(x2);
    else if constexpr (activation_type == ACT_GELU)
        x2 = _gelu(x2);
    else if constexpr (activation_type == ACT_RELU2)
        x2 = _relu2(x2);

    ((half2*) z)[idx] = __hmul2(x2, y2);
}


template <int activation_type>
__global__ __launch_bounds__(NUM_THREADS)
void act_mul_kernel_f
(
    const float* __restrict__ x,
    const float* __restrict__ y,
    half* __restrict__ z,
    const size_t numel
)
{
    size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);
    if (idx >= numel / 2) return;

    float2 x2 = ((const float2*) x)[idx];
    float2 y2 = ((const float2*) y)[idx];

    if constexpr (activation_type == ACT_SILU)
    {
        x2.x = _silu(x2.x);
        x2.y = _silu(x2.y);
    }
    else if constexpr (activation_type == ACT_GELU)
    {
        x2.x = _gelu(x2.x);
        x2.y = _gelu(x2.y);
    }
    else if constexpr (activation_type == ACT_RELU2)
    {
        x2.x = _relu2(x2.x);
        x2.y = _relu2(x2.y);
    }

    x2.x *= y2.x;
    x2.y *= y2.y;
    half2 r = __float22half2_rn(x2);
    r = clamp_half2_to_finite(r);
    ((half2*) z)[idx] = r;
}


__global__ __launch_bounds__(NUM_THREADS)
void xielu_kernel_f
(
    const float* __restrict__ x,
    half* __restrict__ y,
    const size_t numel,
    float alpha_p,
    float alpha_n
)
{
    size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);
    if (idx >= numel / 2) return;

    float2 x2 = ((const float2*) x)[idx];

    x2.x = _xielu(x2.x, alpha_p, alpha_n);
    x2.y = _xielu(x2.y, alpha_p, alpha_n);

    half2 r = __float22half2_rn(x2);
    r = clamp_half2_to_finite(r);
    ((half2*) y)[idx] = r;
}


__global__ __launch_bounds__(NUM_THREADS)
void add_sigmoid_kernel_f
(
    const float* __restrict__ px,
    const float* __restrict__ py,
    float* __restrict__ pz,
    const size_t numel,
    const size_t dim
)
{
    size_t idx = (blockIdx.x * NUM_THREADS + threadIdx.x);
    size_t gidx = idx / dim;
    if (idx >= numel) return;
    float x = px[idx];
    float y = py[gidx];
    float z = pz[idx];
    z += x * _sigmoid_fast_exp(y);
    pz[idx] = z;
}

// x * sigmoid(y @ w) + z -> z,
// x: (bsz, dim)
// y: (bsz, dim)
// z: (bsz, dim)
// w: (dim, 1)

__global__ __launch_bounds__(NUM_THREADS_P)
void add_sigmoid_proj_kernel_f
(
    const float* __restrict__ px,
    const half* __restrict__ py,
    float* __restrict__ pz,
    const half* __restrict__ pw,
    const size_t bsz,
    const size_t dim
)
{
    int b = blockIdx.x;
    int t = threadIdx.x;
    const float* pxb = px + dim * b;
    const half* pyb = py + dim * b;
    const float* pzb = pz + dim * b;

    float yw = 0.0f;
    for (size_t idx = t; idx < dim; idx += NUM_THREADS_P)
    {
        float w = __half2float(pw[idx]);
        float y = __half2float(pyb[idx]);
        yw += w * y;
    }
    yw = block_reduce_sum_broadcast_f(yw, NUM_THREADS_P);
    float syw = _sigmoid_fast_exp(yw);

    if (syw < 1e-8f) return;

    for (size_t idx = t; idx < dim; idx += NUM_THREADS_P)
    {
        float x = px[idx];
        float z = pz[idx];
        z += x * syw;
        pz[idx] = z;
    }
}
