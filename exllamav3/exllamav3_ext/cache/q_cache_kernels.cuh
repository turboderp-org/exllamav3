
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

__device__ inline float shuffle_sum_fx32(float s)
{
    for (int i = 1; i < 32; i <<= 1)
        s += __shfl_xor_sync(0xffffffff, s, i);
    return s;
}

__device__ inline float shuffle_max_fx32(float s)
{
    for (int i = 1; i < 32; i <<= 1)
        s = fmaxf(s, __shfl_xor_sync(0xffffffff, s, i));
    return s;
}

template <int bits>
__device__ inline void quant_block
(
    const half* in,
    uint32_t* out,
    half* out_scales
)
{
    int t = threadIdx.x % 32;

    // Load, rotate and scale 32 values
    float v = __half2float(in[t]);
    v = shuffle_had_fx32(v, t);
    v *= 0.17678f;  // 0.17678 = 1 / sqrt(32)
    float s = shuffle_max_fx32(fabsf(v) + 1e-10);
    half sh = __float2half_rn(s);
    s = __half2float(sh);
    v /= s;

    // Quantize and clamp
    int m = (1 << (bits - 1));
    v *= __int2float_rn(m);
    int q = lrintf(v) + m;
    q = max(min((1 << bits) - 1, q), 0);

    // Pack bits
    register uint32_t bitplanes[bits];
    for (int i = 0, mask = 1; mask <= m; ++i, mask <<= 1)
        bitplanes[i] = __ballot_sync(0xffffffff, q & mask);

    // Write output
    if (t < bits)
        out[t] = bitplanes[t];
    if (t == bits)
        *out_scales = sh;
}

#define MAX_WARPS 32

template <int bits>
__device__ inline void dequant_block
(
    const uint32_t* in,
    const half* in_scales,
    half* out
)
{
    int t = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Load scale and bitplanes
    float s = __half2float(*in_scales);
    __shared__ uint32_t bitplanes[MAX_WARPS][bits];
    if (t < bits)
        bitplanes[warp_id][t] = in[t];
    __syncthreads();

    // Unpack bits
    int m = (1 << (bits - 1));
    uint32_t mask = 1 << t;
    int q = 0;
    for (int i = 0; i < bits; ++i)
        q |= ((bitplanes[warp_id][i] & mask) >> t) << i;

    // Dequantize
    float v = __int2float_rn(q - m);
    v /= __int2float_rn(m);

    // Scale and rotate
    v *= s;
    v = shuffle_had_fx32(v, t);
    v *= 0.17678f;  // 0.17678 = 1 / sqrt(32)

    // Store
    out[t] = __float2half(v);
}

template <int bits>
__global__ __launch_bounds__(1024)
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
__global__ __launch_bounds__(32)
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
__global__ __launch_bounds__(1024)
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
    int page_size,
    int blocks_per_seq,
    int token_dim
)
{
    int batch_idx = blockIdx.z;
    int token_idx = blockIdx.y + cache_seqlens[batch_idx];
    int page_idx = token_idx / page_size;
    int token_pos = block_table[blocks_per_seq * batch_idx + page_idx] * page_size + (token_idx % page_size);
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
__global__ __launch_bounds__(1024)
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
    int page_size,
    int pages_per_seq,
    int warps_per_token
)
{
    int batch_idx = blockIdx.y;
    int t_warp_id = (blockDim.x * blockIdx.x + threadIdx.x) / 32;
    int token_idx = t_warp_id / warps_per_token;
    int max_token_idx = cache_seqlens[batch_idx];
    if (token_idx >= max_token_idx) return;
    int page_idx = token_idx / page_size;
    int page_sub = t_warp_id % (warps_per_token * page_size);
    int mapped_page = block_table[batch_idx * pages_per_seq + page_idx];
    int addr = mapped_page * page_size * warps_per_token + page_sub;

    dequant_block<k_bits>(k_in + addr * k_bits, k_in_scales + addr, k_out + addr * 32);
    dequant_block<v_bits>(v_in + addr * v_bits, v_in_scales + addr, v_out + addr * 32);
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
