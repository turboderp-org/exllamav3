#pragma once

#define MAX_WARPS 8
#define ITER_PER_TB 32
#define CQ_PAGE_SIZE 256


__device__ __forceinline__ float shuffle_had_fx32(float v, int lane_id)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
    {
        float pv = __shfl_xor_sync(0xffffffff, v, i);
        uint32_t vi = __float_as_uint(v);
        int32_t sfm = -static_cast<int32_t>(lane_id & i) >> 31;
        vi ^= (sfm & 0x80000000);
        v = __uint_as_float(vi) + pv;
    }
    return v;
}


__device__ __forceinline__ float shuffle_sum_fx32(float s)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
        s += __shfl_xor_sync(0xffffffff, s, i);
    return s;
}


__device__ __forceinline__ float shuffle_max_fx32(float s)
{
    #pragma unroll
    for (int i = 1; i < 32; i <<= 1)
        s = fmaxf(s, __shfl_xor_sync(0xffffffff, s, i));
    return s;
}


template <int num_bits>
__device__ __forceinline__ void quant_block
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    int t = threadIdx.x & 31;

    // Load, rotate and scale 32 values
    float v = __half2float(in[t]);
    v = shuffle_had_fx32(v, t);
    v *= 0.17677669529663688110f;  // = 1 / sqrt(32)
    float s = shuffle_max_fx32(fabsf(v) + 1e-10);
    half sh = __float2half_rn(s);
    v /= s;

    // Quantize and clamp
    int m = (1 << (num_bits - 1));
    constexpr float mf = (1 << (num_bits - 1));
    v *= mf;
    int q = lrintf(v) + m;
    q = max(min((1 << num_bits) - 1, q), 0);

    // Pack bits
    register uint32_t bitplanes[num_bits];
    #pragma unroll
    for (int i = 0, mask = 1; i < num_bits; ++i, mask <<= 1)
        bitplanes[i] = __ballot_sync(0xffffffff, q & mask);

    // Write output
    if (t < num_bits)
        out[t] = bitplanes[t];
    if (t == num_bits)
        *out_scales = sh;
}


template <int num_bits>
__device__ __forceinline__ void dequant_block
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out
)
{
    int lane = threadIdx.x & 31;

    // Load bitplanes
    uint32_t word = (lane < num_bits) ? in[lane] : 0u;

    // Dequantize at 1/sqrt(32) scale (preparing for Hadamard transform)
    float mbit = -0.17677669529663688110f;
    float v = -0.17677669529663688110f;
    #pragma unroll
    for (int i = num_bits - 1; i >= 0; --i)
    {
        uint32_t wi = __shfl_sync(0xffffffff, word, i);
        v -= ((wi >> lane) & 1u) ? mbit : 0.0f;
        mbit *= 0.5f;
    }

    // Scale and rotate
    v *= __half2float(*in_scales);
    v = shuffle_had_fx32(v, lane);

    // Store
    out[lane] = __float2half(v);
}


template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_cache_cont_kernel
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    in += 32 * blockIdx.x;
    out += bits * blockIdx.x;
    out_scales += blockIdx.x;
    quant_block<bits>(in, out, out_scales);
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
    half* __restrict__ out
)
{
    in += bits * blockIdx.x;
    in_scales += blockIdx.x;
    out += 32 * blockIdx.x;
    dequant_block<bits>(in, in_scales, out);
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
    // int page_size,
    const int blocks_per_seq,
    const int token_dim
)
{
    int batch_idx = blockIdx.z;
    int token_idx = blockIdx.y + cache_seqlens[batch_idx];
    int page_idx = token_idx / CQ_PAGE_SIZE;
    int token_pos = block_table[blocks_per_seq * batch_idx + page_idx] * CQ_PAGE_SIZE + (token_idx % CQ_PAGE_SIZE);
    int sub_pos = (token_pos * token_dim + blockDim.x * blockIdx.x + threadIdx.x) / 32;

    quant_block<k_bits>(k_in + sub_pos * 32, k_out + sub_pos * k_bits, k_out_scales + sub_pos);
    quant_block<v_bits>(v_in + sub_pos * 32, v_out + sub_pos * v_bits, v_out_scales + sub_pos);
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
    // int page_size,
    const int pages_per_seq,
    const int warps_per_token,
    const int num_blocks
)
{
    int batch_idx = blockIdx.y;
    int block_id = blockDim.x * (blockIdx.x * ITER_PER_TB);
    int t_warp_id = (block_id + threadIdx.x) / 32;
    int d_warp_id = blockDim.x / 32;
    int max_token_idx = cache_seqlens[batch_idx];
    const uint32_t* b_block_table = block_table + batch_idx * pages_per_seq;

    #pragma unroll 4
    for (int iter = 0; iter < ITER_PER_TB; ++iter)
    {
        int token_idx = t_warp_id / warps_per_token;
        if (token_idx >= max_token_idx) break;
        int page_idx = token_idx / CQ_PAGE_SIZE;
        int page_sub = t_warp_id - page_idx * CQ_PAGE_SIZE * warps_per_token;
        int mapped_page = b_block_table[page_idx];
        int addr = mapped_page * CQ_PAGE_SIZE * warps_per_token + page_sub;

        dequant_block<k_bits>(k_in + addr * k_bits, k_in_scales + addr, k_out + addr * 32);
        dequant_block<v_bits>(v_in + addr * v_bits, v_in_scales + addr, v_out + addr * 32);

        t_warp_id += d_warp_id;
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
