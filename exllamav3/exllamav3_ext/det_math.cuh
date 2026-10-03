#pragma once

/*

Deterministic transcendentals: FMAs and correctly-rounded intrinsics only, so results are
bit-identical across GPU architectures and across CUDA/HIP backends. Used by the routing
epilogues (selection scores must agree bit for bit across tensor-parallel ranks of different
architectures) and by the det_gemm kernels.

The build enables --use_fast_math, which routes expf/sqrtf to MUFU approximations whose
results are not guaranteed identical across architectures; the functions below avoid that.
CUDA and HIP both define __fmaf_rn/__fmul_rn/__fadd_rn/__fdiv_rn with IEEE semantics, so
nothing here needs a backend branch.

*/

#include <cuda_fp16.h>

// exp with FMAs only: 2^n * p(r), x = n ln2 + r. About 2 ulp; identical on every architecture
__device__ __forceinline__ float exp_det(float x)
{
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    const float n = rintf(__fmul_rn(x, 1.44269504088896341f));
    float r = __fmaf_rn(n, -0.693145751953125f, x);          // ln2 hi
    r = __fmaf_rn(n, -1.428606765330187e-06f, r);            // ln2 lo
    float p = 1.9875691500e-4f;
    p = __fmaf_rn(p, r, 1.3981999507e-3f);
    p = __fmaf_rn(p, r, 8.3334519073e-3f);
    p = __fmaf_rn(p, r, 4.1665795894e-2f);
    p = __fmaf_rn(p, r, 1.6666665459e-1f);
    p = __fmaf_rn(p, r, 5.0000001201e-1f);
    p = __fmaf_rn(p, __fmul_rn(r, r), r);
    p = __fadd_rn(p, 1.0f);
    return __int_as_float(__float_as_int(p) + ((int) n << 23));
}
__device__ __forceinline__ float sigmoid_det(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, exp_det(-x))); }
__device__ __forceinline__ float silu_det(float x) { return __fmul_rn(x, sigmoid_det(x)); }

// log(m 2^e) = e ln2 + 2 atanh(s), s = (m - 1) / (m + 1), FMAs only, about 2 ulp
__device__ __forceinline__ float det_log_series(float s)
{
    const float s2 = __fmul_rn(s, s);
    float p = 0.15313800f;
    p = __fmaf_rn(p, s2, 0.18182640f);
    p = __fmaf_rn(p, s2, 0.22222198f);
    p = __fmaf_rn(p, s2, 0.28571428f);
    p = __fmaf_rn(p, s2, 0.40000000f);
    p = __fmaf_rn(p, s2, 0.66666667f);
    p = __fmaf_rn(p, s2, 2.0f);
    return __fmul_rn(p, s);
}
__device__ __forceinline__ float log_det(float x)
{
    int ix = __float_as_int(x);
    int e = ((ix >> 23) & 0xff) - 127;
    float m = __int_as_float((ix & 0x007fffff) | 0x3f800000);   // [1, 2)
    if (m > 1.41421356f) { m = __fmul_rn(m, 0.5f); e += 1; }
    float r = det_log_series(__fdiv_rn(__fadd_rn(m, -1.0f), __fadd_rn(m, 1.0f)));
    r = __fmaf_rn((float) e, 0.693145751953125f, r);
    r = __fmaf_rn((float) e, 1.428606765330187e-06f, r);
    return r;
}
// log(1 + y) for y >= 0: the series in y / (2 + y) directly while 1 + y needs no exponent step
// (keeps full precision for tiny y), log_det(1 + y) beyond
__device__ __forceinline__ float log1p_det(float y)
{
    if (y < 0.41421356f) return det_log_series(__fdiv_rn(y, __fadd_rn(2.0f, y)));
    return log_det(__fadd_rn(1.0f, y));
}
// softplus matching torch (beta 1, threshold 20)
__device__ __forceinline__ float softplus_det(float x)
{
    if (x > 20.0f) return x;
    return log1p_det(exp_det(x));
}
