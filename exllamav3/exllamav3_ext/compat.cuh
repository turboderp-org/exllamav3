#pragma once

// Approximate tanh

// ROCm: hide the device helpers from host parses (the JIT loader builds .cpp files
// with the host compiler, where HIP's headers lack the intrinsics they use); the
// polyfills and host-parse shims live in compat_rocm.cuh

#if defined(USE_ROCM) && !defined(__HIPCC__)

#include "compat_rocm.cuh"

#else

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750 || CUDART_VERSION < 11000))

__inline__ __device__ float tanh_opt(float x)
{
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
}

#else

__inline__ __device__ float tanh_opt(float x)
{
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

#endif

#endif  // defined(USE_ROCM) && !defined(__HIPCC__)

#if defined(USE_ROCM) && defined(__HIPCC__)
#include "compat_rocm.cuh"
#endif
