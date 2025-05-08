#pragma once

// Hadamard transform 128-element vector across one warp, with optional pre and post scales

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

// Half vector, half scales

inline __device__
void had_hf_r_128_inner
(
    const half* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    float r_scale
)
{
    int t = threadIdx.x % 32;

    // Load
    half4 v = ((half4*) input_ptr)[t];

    // Pre scale
    if (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) pre_scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

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
    h0 = shuffle_had_fx32(h0, t) * r_scale;
    h1 = shuffle_had_fx32(h1, t) * r_scale;
    h2 = shuffle_had_fx32(h2, t) * r_scale;
    h3 = shuffle_had_fx32(h3, t) * r_scale;
    v.x = __floats2half2_rn(h0, h1);
    v.y = __floats2half2_rn(h2, h3);

    // Post scale
    if (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) post_scale)[i];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // Store
    ((half4*) output_ptr)[t] = v;
}

// Float vector, half scales

inline __device__
void had_ff_r_128_inner
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    const half* __restrict__ pre_scale,
    const half* __restrict__ post_scale,
    float r_scale
)
{
    int t = threadIdx.x % 32;

    // Load
    float4 v = ((float4*) input_ptr)[t];

    // Pre scale
    if (pre_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) pre_scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // 4 element had
    float v0 = v.x;
    float v1 = v.y;
    float v2 = v.z;
    float v3 = v.w;
    float h0 = v0 + v1 + v2 + v3;
    float h1 = v0 - v1 + v2 - v3;
    float h2 = v0 + v1 - v2 - v3;
    float h3 = v0 - v1 - v2 + v3;

    // 32 element had, warp shuffle
    v.x = shuffle_had_fx32(h0, t) * r_scale;
    v.y = shuffle_had_fx32(h1, t) * r_scale;
    v.z = shuffle_had_fx32(h2, t) * r_scale;
    v.w = shuffle_had_fx32(h3, t) * r_scale;

    // Post scale
    if (post_scale)
    {
        int i = blockIdx.y * 32 + t;
        half4 scales = ((half4*) post_scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    // Store
    ((float4*) output_ptr)[t] = v;
}