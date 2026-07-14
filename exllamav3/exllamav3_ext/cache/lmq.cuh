#pragma once
#include <math.h>

// Portable clamp: works on both CUDA and CPU
#ifndef LM_CLAMP_IDX
#ifdef __CUDA_ARCH__
#define LM_CLAMP_IDX(idx, lo, hi) max((lo), min((hi), (idx)))
#else
static inline int lm_clamp_(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
#define LM_CLAMP_IDX(idx, lo, hi) lm_clamp_((idx), (lo), (hi))
#endif
#endif

// Cubic scheme:  f(t) = a*t + (1-a)*t^3

// Decode: index -> value
// Cost: ~3 ops (FMA + MUL + FMA for t computation)
__device__ __forceinline__
float lm_cubic_decode(int idx, int bits, float a = 0.65f) {
    float N    = (float)(1 << bits);          // 2^bits
    float t    = fmaf(2.0f * idx + 1.0f, 1.0f / N, -1.0f);  // (2*idx+1)/N - 1
    float t2   = t * t;
    float b    = 1.0f - a;
    return t * fmaf(t2, b, a);                // a*t + b*t^3 = t*(a + b*t^2)
}

// Encode: value -> index
// Cost: ~5 ops (1 sqrt + 2 cbrt + FMA + floor)
// Uses Cardano's formula to solve  b*t^3 + a*t = x  for t
// Depressed cubic: t^3 + (a/b)*t - x/b = 0
// With p = a/b > 0, there is exactly one real root
__device__ __forceinline__
int lm_cubic_encode(float x, int bits, float a = 0.65f) {
    float half_n = (float)(1 << (bits - 1));  // 2^(bits-1)
    float b      = 1.0f - a;
    float inv_b  = 1.0f / b;
    float p3     = a * inv_b * (1.0f / 3.0f); // (a/b) / 3 = p/3
    float p3_cub = p3 * p3 * p3;              // (p/3)^3

    float q_half = x * inv_b * 0.5f;          // -q/2 = x / (2b)
    float delta  = fmaf(q_half, q_half, p3_cub);
    float s      = sqrtf(delta);
    float t      = cbrtf(q_half + s) + cbrtf(q_half - s);

    int idx = __float2int_rd(fmaf(t, half_n, half_n));  // floor((t+1) * N/2)
    return LM_CLAMP_IDX(idx, 0, (1 << bits) - 1);
}

// Template wrapper to bake in N, half_N, and inv_N as compile-time constants
template <int BITS>
struct LMCubic {
    static_assert(BITS >= 2 && BITS <= 8, "Supported range: 2-8 bits");

    static constexpr int   N      = 1 << BITS;
    static constexpr int   HALF_N = 1 << (BITS - 1);
    static constexpr int   MAX_IDX = N - 1;
    static constexpr float INV_N  = 1.0f / N;

    float a, b, inv_b, p3_cub;

    __device__ __host__
    LMCubic(float a_ = 0.65f)
        : a(a_), b(1.0f - a_), inv_b(1.0f / (1.0f - a_))
    {
        float p3 = a * inv_b * (1.0f / 3.0f);
        p3_cub = p3 * p3 * p3;
    }

    __device__ __forceinline__
    float decode(int idx) const {
        float t  = fmaf(2.0f * idx + 1.0f, INV_N, -1.0f);
        float t2 = t * t;
        return t * fmaf(t2, b, a);
    }

    __device__ __forceinline__
    int encode(float x) const {
        float q_half = x * inv_b * 0.5f;
        float delta  = fmaf(q_half, q_half, p3_cub);
        float s      = sqrtf(delta);
        float t      = cbrtf(q_half + s) + cbrtf(q_half - s);
        int idx = __float2int_rd(fmaf(t, (float)HALF_N, (float)HALF_N));
        return LM_CLAMP_IDX(idx, 0, MAX_IDX);
    }
};
