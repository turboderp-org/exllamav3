#pragma once

#include "lmq.cuh"

#define MAX_WARPS 8
#define ITER_PER_TB 32
#define CQ_PAGE_SIZE 256

/*
Cache quantization, groups of 32 values along the head/token dim:

- group is rotated by H32 (regularizes toward Gaussian), scaled to [-1, 1] by its absmax
  (stored as one half per group), then quantized to num_bits linearly, or through the cubic
  compander (lmq.cuh) when compand_a > 0
- packing is linear little-endian: value j of a group occupies bits [j*num_bits, (j+1)*num_bits)
  of the group's num_bits uint32 words. Unpacking any bit width is a two-word funnel window per
  lane, so dequant needs no cross-lane traffic for the payload
- each warp covers 4 groups (128 values): an 8-lane subgroup per group, 4 consecutive values per
  lane. H32 then factors as an in-register H4 times a 3-round __shfl_xor H8 — same matrix as a
  5-round per-lane butterfly, at 12 instead of 40 shuffles per warp (the old one-group-per-warp
  bitplane kernels ran at ~92% LSU-pipe utilization and 38% of DRAM)

The quantized tensors have the same shapes and sizes as the previous bitplane format.
*/

// Butterfly stages over disjoint index bits commute, so H4 (value bits 0..1, in registers) and
// H8 (bits 2..4, across the subgroup) can be applied in either order on both endpoints

__device__ __forceinline__ void had_4_inreg(float& v0, float& v1, float& v2, float& v3)
{
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    v0 = s0 + s1;
    v1 = d0 + d1;
    v2 = s0 - s1;
    v3 = d0 - d1;
}

__device__ __forceinline__ void had_8_subgroup(float& v0, float& v1, float& v2, float& v3, int lane)
{
    #pragma unroll
    for (int i = 1; i < 8; i <<= 1)
    {
        uint64_t p01 = ((uint64_t) __float_as_uint(v0)) | (((uint64_t) __float_as_uint(v1)) << 32);
        uint64_t p23 = ((uint64_t) __float_as_uint(v2)) | (((uint64_t) __float_as_uint(v3)) << 32);
        p01 = __shfl_xor_sync(0xffffffff, p01, i);
        p23 = __shfl_xor_sync(0xffffffff, p23, i);
        uint32_t sfm = (uint32_t) (-(int32_t)(lane & i) >> 31) & 0x80000000u;
        v0 = __uint_as_float(__float_as_uint(v0) ^ sfm) + __uint_as_float((uint32_t) p01);
        v1 = __uint_as_float(__float_as_uint(v1) ^ sfm) + __uint_as_float((uint32_t) (p01 >> 32));
        v2 = __uint_as_float(__float_as_uint(v2) ^ sfm) + __uint_as_float((uint32_t) p23);
        v3 = __uint_as_float(__float_as_uint(v3) ^ sfm) + __uint_as_float((uint32_t) (p23 >> 32));
    }
}

// Quantize 4 groups of 32 (one warp). in/out/out_scales point at the warp's 4-group span;
// sh_pack is a zero-initialized warp-private staging area of 4 * num_bits uint32

template <int num_bits>
__device__ __forceinline__ void quant_block_x4
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales,
    uint32_t* __restrict__ sh_pack,
    int active_groups,
    float compand_a
)
{
    constexpr int m = 1 << (num_bits - 1);
    constexpr float mf = (float) m;
    constexpr uint32_t qmax = (1u << num_bits) - 1;

    const int lane = threadIdx.x & 31;
    const int sg = lane >> 3;
    const int sl = lane & 7;
    const bool active = sg < active_groups;

    // Zero staging
    if (lane < 4 * num_bits)
        sh_pack[lane] = 0;
    __syncwarp();

    // Load 4 consecutive values, rotate, scale to 1/sqrt(32)
    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
    if (active)
    {
        half2 x01 = ((const half2*) in)[lane * 2];
        half2 x23 = ((const half2*) in)[lane * 2 + 1];
        v0 = __half2float(__low2half(x01));
        v1 = __half2float(__high2half(x01));
        v2 = __half2float(__low2half(x23));
        v3 = __half2float(__high2half(x23));
    }
    had_4_inreg(v0, v1, v2, v3);
    had_8_subgroup(v0, v1, v2, v3, lane);
    constexpr float r32 = 0.17677669529663688110f;  // 1/sqrt(32)
    v0 *= r32; v1 *= r32; v2 *= r32; v3 *= r32;

    // Group absmax
    float s = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))) + 1e-10f;
    #pragma unroll
    for (int i = 1; i < 8; i <<= 1)
        s = fmaxf(s, __shfl_xor_sync(0xffffffff, s, i));
    float inv_s = 1.0f / s;

    // Quantize
    uint32_t q0, q1, q2, q3;
    if (compand_a > 0.0f)
    {
        LMCubic<num_bits> lm(compand_a);
        q0 = lm.encode(v0 * inv_s);
        q1 = lm.encode(v1 * inv_s);
        q2 = lm.encode(v2 * inv_s);
        q3 = lm.encode(v3 * inv_s);
    }
    else
    {
        // Midpoint grid: centroids at ((2q+1)/2^bits - 1), ~5% lower MSE than the rounding grid
        auto quant1 = [&] (float v) -> uint32_t
        {
            int q = __float2int_rd(fmaf(v * inv_s, mf, mf));
            return (uint32_t) max(min(q, (int) qmax), 0);
        };
        q0 = quant1(v0); q1 = quant1(v1); q2 = quant1(v2); q3 = quant1(v3);
    }

    // Pack the lane's 16*num_bits-bit field into the group's words
    if (active)
    {
        uint32_t field = q0 | (q1 << num_bits) | (q2 << 2 * num_bits) | (q3 << 3 * num_bits);
        int off = sl * 4 * num_bits;
        int w0 = off >> 5;
        int sh = off & 31;
        atomicOr(&sh_pack[sg * num_bits + w0], field << sh);
        int w1 = (off + 4 * num_bits - 1) >> 5;
        if (w1 != w0)
            atomicOr(&sh_pack[sg * num_bits + w1], field >> (32 - sh));  // w1 != w0 implies sh > 0
    }
    __syncwarp();

    // Write words and scales
    if (lane < 4 * num_bits && (lane / num_bits) < active_groups)
        out[lane] = sh_pack[lane];
    float sw = __shfl_sync(0xffffffff, s, lane * 8);
    if (lane < active_groups)
        out_scales[lane] = __float2half_rn(sw);
}

// Dequantize 4 groups of 32 (one warp)

template <int num_bits>
__device__ __forceinline__ void dequant_block_x4
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out,
    int active_groups,
    float compand_a
)
{
    constexpr int m = 1 << (num_bits - 1);
    constexpr float inv_mf = 1.0f / (float) (1 << (num_bits - 1));
    constexpr uint32_t qmask = (1u << num_bits) - 1;

    const int lane = threadIdx.x & 31;
    const int sg = lane >> 3;
    const int sl = lane & 7;
    const bool active = sg < active_groups;

    // Two-word funnel window holds the lane's 4 values for any bit width
    const int off = sl * 4 * num_bits;
    const uint32_t* gw = in + sg * num_bits;
    uint32_t w0 = active ? gw[off >> 5] : 0;
    uint32_t w1 = active ? gw[(off + 4 * num_bits - 1) >> 5] : 0;
    uint32_t win = __funnelshift_r(w0, w1, off);

    float s = active ? __half2float(in_scales[sg]) : 0.0f;
    constexpr float r32 = 0.17677669529663688110f;  // 1/sqrt(32)
    s *= r32;

    float v0, v1, v2, v3;
    if (compand_a > 0.0f)
    {
        LMCubic<num_bits> lm(compand_a);
        v0 = lm.decode((win                ) & qmask) * s;
        v1 = lm.decode((win >>     num_bits) & qmask) * s;
        v2 = lm.decode((win >> 2 * num_bits) & qmask) * s;
        v3 = lm.decode((win >> 3 * num_bits) & qmask) * s;
    }
    else
    {
        constexpr float mh = (float) m - 0.5f;
        v0 = (float) (int) ((win                ) & qmask) - mh;
        v1 = (float) (int) ((win >>     num_bits) & qmask) - mh;
        v2 = (float) (int) ((win >> 2 * num_bits) & qmask) - mh;
        v3 = (float) (int) ((win >> 3 * num_bits) & qmask) - mh;
        float sm = s * inv_mf;
        v0 *= sm; v1 *= sm; v2 *= sm; v3 *= sm;
    }

    // Rotate back
    had_4_inreg(v0, v1, v2, v3);
    had_8_subgroup(v0, v1, v2, v3, lane);

    // Store
    if (active)
    {
        half2 o01 = __floats2half2_rn(v0, v1);
        half2 o23 = __floats2half2_rn(v2, v3);
        ((half2*) out)[lane * 2] = o01;
        ((half2*) out)[lane * 2 + 1] = o23;
    }
}


template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_cache_cont_kernel
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales,
    const int num_groups,
    const float compand_a
)
{
    __shared__ uint32_t sh_pack[MAX_WARPS][32];
    int warp = threadIdx.x >> 5;
    int g0 = (blockIdx.x * MAX_WARPS + warp) * 4;
    if (g0 >= num_groups) return;
    int active = min(4, num_groups - g0);
    quant_block_x4<bits>(in + g0 * 32, out + g0 * bits, out_scales + g0, sh_pack[warp], active, compand_a);
}

#define __(i) quant_cache_cont_kernel<i>
constexpr auto quant_cache_cont_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __


template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void dequant_cache_cont_kernel
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out,
    const int num_groups,
    const float compand_a
)
{
    int warp = threadIdx.x >> 5;
    int g0 = (blockIdx.x * MAX_WARPS + warp) * 4;
    if (g0 >= num_groups) return;
    int active = min(4, num_groups - g0);
    dequant_block_x4<bits>(in + g0 * bits, in_scales + g0, out + g0 * 32, active, compand_a);
}

#define __(i) dequant_cache_cont_kernel<i>
constexpr auto dequant_cache_cont_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __


template <int k_bits, int v_bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_cache_paged_kernel
(
    const half* __restrict__ k_in,
    uint32_t* __restrict__ k_out,
    half* __restrict__ k_out_scales,
    const half* __restrict__ v_in,
    uint32_t* __restrict__ v_out,
    half* __restrict__ v_out_scales,
    const uint32_t* __restrict__ cache_seqlens,
    const uint32_t* __restrict__ block_table,
    const int blocks_per_seq,
    const int groups_per_token,
    const float compand_a
)
{
    __shared__ uint32_t sh_pack[MAX_WARPS][32];
    int batch_idx = blockIdx.z;
    int token_idx = blockIdx.y + cache_seqlens[batch_idx];
    int page_idx = token_idx / CQ_PAGE_SIZE;
    int token_pos = block_table[blocks_per_seq * batch_idx + page_idx] * CQ_PAGE_SIZE + (token_idx % CQ_PAGE_SIZE);

    int warp = threadIdx.x >> 5;
    int g0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 4;
    if (g0 >= groups_per_token) return;
    int active = min(4, groups_per_token - g0);
    int base = token_pos * groups_per_token + g0;

    quant_block_x4<k_bits>(k_in + base * 32, k_out + base * k_bits, k_out_scales + base, sh_pack[warp], active, compand_a);
    __syncwarp();
    quant_block_x4<v_bits>(v_in + base * 32, v_out + base * v_bits, v_out_scales + base, sh_pack[warp], active, compand_a);
}

#define __(i, j) quant_cache_paged_kernel<i, j>
constexpr auto quant_cache_paged_kernel_instances = std::array
{
    std::array{ __(2, 2), __(2, 3), __(2, 4), __(2, 5), __(2, 6), __(2, 7), __(2, 8) },
    std::array{ __(3, 2), __(3, 3), __(3, 4), __(3, 5), __(3, 6), __(3, 7), __(3, 8) },
    std::array{ __(4, 2), __(4, 3), __(4, 4), __(4, 5), __(4, 6), __(4, 7), __(4, 8) },
    std::array{ __(5, 2), __(5, 3), __(5, 4), __(5, 5), __(5, 6), __(5, 7), __(5, 8) },
    std::array{ __(6, 2), __(6, 3), __(6, 4), __(6, 5), __(6, 6), __(6, 7), __(6, 8) },
    std::array{ __(7, 2), __(7, 3), __(7, 4), __(7, 5), __(7, 6), __(7, 7), __(7, 8) },
    std::array{ __(8, 2), __(8, 3), __(8, 4), __(8, 5), __(8, 6), __(8, 7), __(8, 8) }
};
#undef __


template <int k_bits, int v_bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void dequant_cache_paged_kernel
(
    const uint32_t* __restrict__ k_in,
    const half* __restrict__ k_in_scales,
    half* __restrict__ k_out,
    const uint32_t* __restrict__ v_in,
    const half* __restrict__ v_in_scales,
    half* __restrict__ v_out,
    const uint32_t* __restrict__ cache_seqlens,
    const uint32_t* __restrict__ block_table,
    const int pages_per_seq,
    const int groups_per_token,
    const int chunks_per_token,     // ceil(groups_per_token / 4)
    const int sliding_window,
    const float compand_a
)
{
    int batch_idx = blockIdx.y;
    int chunk_id = blockDim.x / 32 * (blockIdx.x * ITER_PER_TB) + (threadIdx.x >> 5);
    int d_chunks = blockDim.x / 32;
    int max_token_idx = cache_seqlens[batch_idx];
    const uint32_t* b_block_table = block_table + batch_idx * pages_per_seq;

    // Skip all whole blocks prior to the sliding window
    if (sliding_window > 0)
    {
        int nb_chunk_id = (blockDim.x / 32) * ((blockIdx.x + 1) * ITER_PER_TB);
        int nb_token_idx = nb_chunk_id / chunks_per_token;
        if (nb_token_idx <= max_token_idx - sliding_window)
            return;
    }

    #pragma unroll 4
    for (int iter = 0; iter < ITER_PER_TB; ++iter)
    {
        int token_idx = chunk_id / chunks_per_token;
        if (token_idx >= max_token_idx) break;
        int g0 = (chunk_id - token_idx * chunks_per_token) * 4;
        int active = min(4, groups_per_token - g0);
        int page_idx = token_idx / CQ_PAGE_SIZE;
        int mapped_page = b_block_table[page_idx];
        int token_pos = mapped_page * CQ_PAGE_SIZE + (token_idx % CQ_PAGE_SIZE);
        int base = token_pos * groups_per_token + g0;

        dequant_block_x4<k_bits>(k_in + base * k_bits, k_in_scales + base, k_out + base * 32, active, compand_a);
        dequant_block_x4<v_bits>(v_in + base * v_bits, v_in_scales + base, v_out + base * 32, active, compand_a);

        chunk_id += d_chunks;
    }
}

#define __(i, j) dequant_cache_paged_kernel<i, j>
constexpr auto dequant_cache_paged_kernel_instances = std::array
{
    std::array{ __(2, 2), __(2, 3), __(2, 4), __(2, 5), __(2, 6), __(2, 7), __(2, 8) },
    std::array{ __(3, 2), __(3, 3), __(3, 4), __(3, 5), __(3, 6), __(3, 7), __(3, 8) },
    std::array{ __(4, 2), __(4, 3), __(4, 4), __(4, 5), __(4, 6), __(4, 7), __(4, 8) },
    std::array{ __(5, 2), __(5, 3), __(5, 4), __(5, 5), __(5, 6), __(5, 7), __(5, 8) },
    std::array{ __(6, 2), __(6, 3), __(6, 4), __(6, 5), __(6, 6), __(6, 7), __(6, 8) },
    std::array{ __(7, 2), __(7, 3), __(7, 4), __(7, 5), __(7, 6), __(7, 7), __(7, 8) },
    std::array{ __(8, 2), __(8, 3), __(8, 4), __(8, 5), __(8, 6), __(8, 7), __(8, 8) }
};
#undef __
