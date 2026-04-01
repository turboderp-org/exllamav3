#pragma once
#include "lloyd_max_codebooks.cuh"

// Lloyd-Max quant/dequant block functions.
//
// Interface is identical to quant_block<num_bits> / dequant_block<num_bits>
// from q_cache_kernels.cuh.  The only difference is that the quantization
// step uses boundary-search against the precomputed Lloyd-Max tables instead
// of uniform rounding, and dequantization returns the centroid value instead
// of a uniformly-spaced reconstruction level.
//
// Bit packing layout is IDENTICAL to the uniform kernel so the two codecs
// are wire-compatible (same memory layout for a given num_bits).

// ---------------------------------------------------------------------------
// quant_block_lm
//
// Quantize 32 contiguous fp16 values (one warp) using Lloyd-Max boundaries.
// Each thread handles one element; results are packed as num_bits bitplanes
// of 32 bits each (one bit per lane), plus a half-precision scale word.
//
// Memory layout written to out[0..num_bits-1] (bitplanes) and out_scales[0]:
//   - out[i]       : bit i of the code for all 32 lanes (ballot result)
//   - out_scales[0]: block scale (half), written by lane num_bits
//
// This matches the layout produced by quant_block<num_bits> exactly.
// ---------------------------------------------------------------------------

template <int num_bits>
__device__ __forceinline__ void quant_block_lm
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    int t = threadIdx.x & 31;

    // Load, WHT-rotate, and scale — identical to uniform path
    float v = __half2float(in[t]);
    v = shuffle_had_fx32(v, t);
    v *= 0.17677669529663688110f;  // 1/sqrt(32)
    float s = shuffle_max_fx32(fabsf(v) + 1e-10f);
    half sh = __float2half_rn(s);
    v /= s;  // v is now in [-1, 1]

    // Lloyd-Max boundary search — replaces uniform rounding
    int q = lm_quantize(v, num_bits);

    // Pack bits — identical to uniform path
    register uint32_t bitplanes[num_bits];
    #pragma unroll
    for (int i = 0, mask = 1; i < num_bits; ++i, mask <<= 1)
        bitplanes[i] = __ballot_sync(0xffffffff, q & mask);

    // Write output — identical to uniform path
    if (t < num_bits)
        out[t] = bitplanes[t];
    if (t == num_bits)
        *out_scales = sh;
}

// ---------------------------------------------------------------------------
// dequant_block_lm
//
// Dequantize 32 fp16 values (one warp) stored as num_bits bitplanes.
// Reads the exact same memory layout as quant_block_lm / quant_block.
//
// The uniform dequant reconstructs the float value via an accumulated
// weighted sum:
//
//   v = sum_{i=0}^{num_bits-1}  bit[i] * weight[i]
//
// where the weights form a binary-coded fixed-point representation.
//
// For Lloyd-Max we instead need the integer code so we can do a centroid
// lookup.  We reconstruct q by collecting one bit per bitplane exactly as
// the uniform path collects each bitplane word:
//
//   Each lane `lane` loads up to num_bits words into register `word`
//   (lane i holds bitplane i for i < num_bits, 0 otherwise).
//   __shfl_sync(word, i) broadcasts bitplane i to all lanes.
//   Bit `lane` of that word is bit i of lane's code.
//
// After reconstructing q we look up lm_dequantize(q, num_bits) to get
// the centroid, apply the 1/sqrt(32) scale factor (matching the quant
// path's forward factor), then apply the block scale and inverse WHT.
// ---------------------------------------------------------------------------

template <int num_bits>
__device__ __forceinline__ void dequant_block_lm
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out
)
{
    int lane = threadIdx.x & 31;

    // Each lane loads one bitplane word (lane i -> bitplane i).
    // Lanes beyond num_bits load 0 (never accessed but avoids out-of-bounds).
    uint32_t word = (lane < num_bits) ? in[lane] : 0u;

    // Reconstruct integer code for this lane from the bitplanes.
    // For bit position i: broadcast bitplane-i word, extract bit `lane`.
    int q = 0;
    #pragma unroll
    for (int i = 0; i < num_bits; ++i)
    {
        uint32_t wi = __shfl_sync(0xffffffff, word, i);
        q |= (int)(((wi >> lane) & 1u) << i);
    }

    // Lloyd-Max centroid lookup; result is in [-1, 1] (normalised range)
    float v = lm_dequantize(q, num_bits);

    // Apply the same 1/sqrt(32) factor that was applied before the WHT in
    // the quant path (the inverse WHT will undo this scaling automatically
    // together with the block scale below).
    v *= 0.17677669529663688110f;

    // Apply block scale and inverse WHT — identical to uniform path
    v *= __half2float(*in_scales);
    v = shuffle_had_fx32(v, lane);

    out[lane] = __float2half(v);
}

// ---------------------------------------------------------------------------
// Continuous-layout kernel wrappers (one block per 32-element group)
// ---------------------------------------------------------------------------

template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_lm_cache_cont_kernel
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    in        += 32    * blockIdx.x;
    out       += bits  * blockIdx.x;
    out_scales +=        blockIdx.x;
    quant_block_lm<bits>(in, out, out_scales);
}

#define __(i) quant_lm_cache_cont_kernel<i>
constexpr auto quant_lm_cache_cont_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __


template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void dequant_lm_cache_cont_kernel
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out
)
{
    in        += bits * blockIdx.x;
    in_scales +=        blockIdx.x;
    out       += 32   * blockIdx.x;
    dequant_block_lm<bits>(in, in_scales, out);
}

#define __(i) dequant_lm_cache_cont_kernel<i>
constexpr auto dequant_lm_cache_cont_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

// ---------------------------------------------------------------------------
// Paged-layout kernel wrappers
// ---------------------------------------------------------------------------

template <int k_bits, int v_bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_lm_cache_paged_kernel
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
    const int token_dim
)
{
    int batch_idx  = blockIdx.z;
    int token_idx  = blockIdx.y + cache_seqlens[batch_idx];
    int page_idx   = token_idx / CQ_PAGE_SIZE;
    int token_pos  = block_table[blocks_per_seq * batch_idx + page_idx] * CQ_PAGE_SIZE
                     + (token_idx % CQ_PAGE_SIZE);
    int sub_pos    = (token_pos * token_dim + blockDim.x * blockIdx.x + threadIdx.x) / 32;

    quant_block_lm<k_bits>(k_in + sub_pos * 32, k_out + sub_pos * k_bits, k_out_scales + sub_pos);
    quant_block_lm<v_bits>(v_in + sub_pos * 32, v_out + sub_pos * v_bits, v_out_scales + sub_pos);
}

#define __(i, j) quant_lm_cache_paged_kernel<i, j>
constexpr auto quant_lm_cache_paged_kernel_instances = std::array
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
void dequant_lm_cache_paged_kernel
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
    const int warps_per_token,
    const int num_blocks
)
{
    int batch_idx       = blockIdx.y;
    int block_id        = blockDim.x * (blockIdx.x * ITER_PER_TB);
    int t_warp_id       = (block_id + threadIdx.x) / 32;
    int d_warp_id       = blockDim.x / 32;
    int max_token_idx   = cache_seqlens[batch_idx];
    const uint32_t* b_block_table = block_table + batch_idx * pages_per_seq;

    #pragma unroll 4
    for (int iter = 0; iter < ITER_PER_TB; ++iter)
    {
        int token_idx = t_warp_id / warps_per_token;
        if (token_idx >= max_token_idx) break;
        int page_idx  = token_idx / CQ_PAGE_SIZE;
        int page_sub  = t_warp_id - page_idx * CQ_PAGE_SIZE * warps_per_token;
        int mapped_page = b_block_table[page_idx];
        int addr      = mapped_page * CQ_PAGE_SIZE * warps_per_token + page_sub;

        dequant_block_lm<k_bits>(k_in + addr * k_bits, k_in_scales + addr, k_out + addr * 32);
        dequant_block_lm<v_bits>(v_in + addr * v_bits, v_in_scales + addr, v_out + addr * 32);

        t_warp_id += d_warp_id;
    }
}

#define __(i, j) dequant_lm_cache_paged_kernel<i, j>
constexpr auto dequant_lm_cache_paged_kernel_instances = std::array
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
