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
// shuffle_max_sub
//
// Reduce-max within a sub-group of lanes.  sub_size must be a power of 2
// and <= 32.  The XOR shuffle with offsets strictly less than sub_size only
// communicates between lanes that share the same sub_size-aligned bucket, so
// this is a correct intra-sub-group reduction even with a full-warp mask.
// ---------------------------------------------------------------------------


__device__ __forceinline__ float shuffle_max_sub(float v, int sub_size)
{
    for (int i = 1; i < sub_size; i <<= 1)
    {
        float other = __shfl_xor_sync(0xffffffff, v, i);
        v = fmaxf(v, other);
    }
    return v;
}

// ---------------------------------------------------------------------------
// quant_block_lm_sub
//
// Variant of quant_block_lm that computes one scale per sub-group of
// sub_size lanes (instead of one scale for the full 32-lane block).
//
// Steps:
//   1. Load and full-warp WHT (unchanged — must be 32-wide).
//   2. Apply 1/sqrt(32) normalisation.
//   3. Compute per-sub-group max and normalise within each sub-group.
//   4. Lloyd-Max quantize (unchanged).
//   5. Bitplane-pack (unchanged).
//   6. Write sub-group scales: thread 0 of each sub-group writes its scale.
//
// Memory written:
//   out[0..num_bits-1]              : bitplanes  (same layout as quant_block_lm)
//   out_scales[0..num_subs-1]       : num_subs = 32 / sub_size half-precision scales
// ---------------------------------------------------------------------------

template <int num_bits, int sub_size>
__device__ __forceinline__ void quant_block_lm_sub
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    constexpr int num_subs = 32 / sub_size;
    int t = threadIdx.x & 31;

    // Step 1+2: Load, full-warp WHT, 1/sqrt(32) scale
    float v = __half2float(in[t]);
    v = shuffle_had_fx32(v, t);
    v *= 0.17677669529663688110f;  // 1/sqrt(32)

    // Step 3: Sub-group normalisation
    float s = shuffle_max_sub(fabsf(v) + 1e-10f, sub_size);
    v /= s;

    // Step 4: Lloyd-Max boundary search
    int q = lm_quantize(v, num_bits);

    // Step 5: Bitplane pack
    register uint32_t bitplanes[num_bits];
    #pragma unroll
    for (int i = 0, mask = 1; i < num_bits; ++i, mask <<= 1)
        bitplanes[i] = __ballot_sync(0xffffffff, q & mask);

    // Step 5b: Write bitplanes
    if (t < num_bits)
        out[t] = bitplanes[t];

    // Step 6: Write sub-group scales (one write per sub-group, by lane 0 of each)
    int sub_id   = t / sub_size;
    int sub_lane = t % sub_size;
    if (sub_lane == 0 && sub_id < num_subs)
        out_scales[sub_id] = __float2half_rn(s);
}

// ---------------------------------------------------------------------------
// dequant_block_lm_sub
//
// Inverse of quant_block_lm_sub.  Reads the same bitplane layout, then
// applies per-sub-group scales before the inverse WHT.
// ---------------------------------------------------------------------------

template <int num_bits, int sub_size>
__device__ __forceinline__ void dequant_block_lm_sub
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out
)
{
    int lane = threadIdx.x & 31;

    // Step 1: Load bitplane words (same as dequant_block_lm)
    uint32_t word = (lane < num_bits) ? in[lane] : 0u;

    // Step 2: Reconstruct integer code
    int q = 0;
    #pragma unroll
    for (int i = 0; i < num_bits; ++i)
    {
        uint32_t wi = __shfl_sync(0xffffffff, word, i);
        q |= (int)(((wi >> lane) & 1u) << i);
    }

    // Step 3: Centroid lookup
    float v = lm_dequantize(q, num_bits);

    // Step 4: Apply 1/sqrt(32) factor (matches quant path)
    v *= 0.17677669529663688110f;

    // Step 5: Apply per-sub-group scale
    int sub_id = lane / sub_size;
    v *= __half2float(in_scales[sub_id]);

    // Step 6: Inverse WHT
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

// ---------------------------------------------------------------------------
// Sub-block continuous-layout kernel wrappers (sub_size = 8, 4 scales/block)
// ---------------------------------------------------------------------------

template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_lm_cache_cont_sub_kernel
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4
    in         += 32       * blockIdx.x;
    out        += bits     * blockIdx.x;
    out_scales += num_subs * blockIdx.x;
    quant_block_lm_sub<bits, sub_size>(in, out, out_scales);
}

#define __(i) quant_lm_cache_cont_sub_kernel<i>
constexpr auto quant_lm_cache_cont_sub_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __


template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void dequant_lm_cache_cont_sub_kernel
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out
)
{
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4
    in        += bits     * blockIdx.x;
    in_scales += num_subs * blockIdx.x;
    out       += 32       * blockIdx.x;
    dequant_block_lm_sub<bits, sub_size>(in, in_scales, out);
}

#define __(i) dequant_lm_cache_cont_sub_kernel<i>
constexpr auto dequant_lm_cache_cont_sub_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

// ---------------------------------------------------------------------------
// Sub-block paged-layout kernel wrappers (sub_size = 8, 4 scales/block)
// ---------------------------------------------------------------------------

template <int k_bits, int v_bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_lm_cache_paged_sub_kernel
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
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4

    int batch_idx  = blockIdx.z;
    int token_idx  = blockIdx.y + cache_seqlens[batch_idx];
    int page_idx   = token_idx / CQ_PAGE_SIZE;
    int token_pos  = block_table[blocks_per_seq * batch_idx + page_idx] * CQ_PAGE_SIZE
                     + (token_idx % CQ_PAGE_SIZE);
    int sub_pos    = (token_pos * token_dim + blockDim.x * blockIdx.x + threadIdx.x) / 32;

    quant_block_lm_sub<k_bits, sub_size>(
        k_in  + sub_pos * 32,
        k_out + sub_pos * k_bits,
        k_out_scales + sub_pos * num_subs
    );
    quant_block_lm_sub<v_bits, sub_size>(
        v_in  + sub_pos * 32,
        v_out + sub_pos * v_bits,
        v_out_scales + sub_pos * num_subs
    );
}

#define __(i, j) quant_lm_cache_paged_sub_kernel<i, j>
constexpr auto quant_lm_cache_paged_sub_kernel_instances = std::array
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
void dequant_lm_cache_paged_sub_kernel
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
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4

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

        dequant_block_lm_sub<k_bits, sub_size>(
            k_in  + addr * k_bits,
            k_in_scales + addr * num_subs,
            k_out + addr * 32
        );
        dequant_block_lm_sub<v_bits, sub_size>(
            v_in  + addr * v_bits,
            v_in_scales + addr * num_subs,
            v_out + addr * 32
        );

        t_warp_id += d_warp_id;
    }
}

#define __(i, j) dequant_lm_cache_paged_sub_kernel<i, j>
constexpr auto dequant_lm_cache_paged_sub_kernel_instances = std::array
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

// ===========================================================================
// Asymmetric quantization (scale + zero-point, uniform grid on [min, max])
// ===========================================================================
//
// Motivation: K vectors after WHT have non-zero mean (per-channel bias) and
// heterogeneous variance — the symmetric assumption (range centred at 0) wastes
// bits.  Asymmetric quant stores min (zero-point) + range (scale) per
// sub-group so the full [min, max] interval is covered uniformly.
//
// Convention: all values are in the WHT-rotated, 1/sqrt(32)-scaled domain
// when the scale/zero tensors are written, exactly mirroring quant_block_lm_sub.
// Dequant therefore does NOT apply another 1/sqrt(32) factor — the stored
// scale and zero already absorb it — and then calls the inverse WHT directly.
//
// Zero-point tensor has IDENTICAL shape to the scales tensor.
// ===========================================================================

// ---------------------------------------------------------------------------
// shuffle_min_sub
//
// Reduce-min within a sub-group of lanes (companion to shuffle_max_sub).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float shuffle_min_sub(float v, int sub_size)
{
    for (int i = 1; i < sub_size; i <<= 1)
    {
        float other = __shfl_xor_sync(0xffffffff, v, i);
        v = fminf(v, other);
    }
    return v;
}

// ---------------------------------------------------------------------------
// quant_block_lm_sub_asym
//
// Asymmetric variant of quant_block_lm_sub.  Uses a uniform grid on [min,max]
// instead of the Lloyd-Max symmetric codebook.
//
// Steps:
//   1. Load and full-warp WHT (unchanged — must be 32-wide).
//   2. Apply 1/sqrt(32) normalisation.
//   3. Compute per-sub-group min and max; derive scale = max-min, zero = min.
//   4. Normalise to [0, 1] and uniform-quantize.
//   5. Bitplane-pack (unchanged layout).
//   6. Write sub-group scales and zero-points (lane 0 of each sub-group).
//
// Memory written:
//   out[0..num_bits-1]              : bitplanes
//   out_scales[0..num_subs-1]       : range (max - min) per sub-group
//   out_zeros[0..num_subs-1]        : min per sub-group
// ---------------------------------------------------------------------------

template <int num_bits, int sub_size>
__device__ __forceinline__ void quant_block_lm_sub_asym
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales,
    half* __restrict__ out_zeros
)
{
    constexpr int num_subs = 32 / sub_size;
    int t = threadIdx.x & 31;

    // Step 1+2: Load, full-warp WHT, 1/sqrt(32) scale
    float v = __half2float(in[t]);
    v = shuffle_had_fx32(v, t);
    v *= 0.17677669529663688110f;  // 1/sqrt(32)

    // Step 3: Per-sub-group min and max
    float sub_min = shuffle_min_sub(v,          sub_size);
    float sub_max = shuffle_max_sub(v + 1e-10f, sub_size);

    float range = sub_max - sub_min;
    // Guard against degenerate blocks (all-equal values)
    range = fmaxf(range, 1e-10f);

    // Step 4: Normalise to [0, 1] and uniform quantize
    float v_norm = (v - sub_min) / range;
    v_norm = fminf(fmaxf(v_norm, 0.0f), 1.0f);

    constexpr int num_levels = (1 << num_bits);
    int q = min(max(__float2int_rn(v_norm * (float)(num_levels - 1)), 0), num_levels - 1);

    // Step 5: Bitplane pack (identical layout to symmetric kernels)
    register uint32_t bitplanes[num_bits];
    #pragma unroll
    for (int i = 0, mask = 1; i < num_bits; ++i, mask <<= 1)
        bitplanes[i] = __ballot_sync(0xffffffff, q & mask);

    if (t < num_bits)
        out[t] = bitplanes[t];

    // Step 6: Write scale and zero-point (lane 0 of each sub-group)
    int sub_id   = t / sub_size;
    int sub_lane = t % sub_size;
    if (sub_lane == 0 && sub_id < num_subs)
    {
        out_scales[sub_id] = __float2half_rn(range);
        out_zeros[sub_id]  = __float2half_rn(sub_min);
    }
}

// ---------------------------------------------------------------------------
// dequant_block_lm_sub_asym
//
// Inverse of quant_block_lm_sub_asym.
//
// Reconstruction (in WHT-rotated, 1/sqrt(32)-scaled domain):
//   v_wht_scaled = q / (levels - 1) * scale + zero
//
// No additional 1/sqrt(32) factor is applied (unlike the symmetric path)
// because scale and zero already encode values in the 1/sqrt(32)-scaled
// domain.  The inverse WHT then recovers the original fp16 values.
// ---------------------------------------------------------------------------

template <int num_bits, int sub_size>
__device__ __forceinline__ void dequant_block_lm_sub_asym
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    const half* __restrict__ in_zeros,
    half* __restrict__ out
)
{
    int lane = threadIdx.x & 31;

    // Step 1: Load bitplane words
    uint32_t word = (lane < num_bits) ? in[lane] : 0u;

    // Step 2: Reconstruct integer code
    int q = 0;
    #pragma unroll
    for (int i = 0; i < num_bits; ++i)
    {
        uint32_t wi = __shfl_sync(0xffffffff, word, i);
        q |= (int)(((wi >> lane) & 1u) << i);
    }

    // Step 3: Asymmetric dequantize
    //   v_wht = q / (levels-1) * scale + zero
    // scale and zero are in the 1/sqrt(32)-scaled WHT domain.
    // shuffle_had_fx32 is the UNNORMALIZED WHT (H*H = N*I, not I).
    // To get the correct inverse: multiply by 1/sqrt(32) first, then WHT.
    // This gives: WHT(v/sqrt(32)) = WHT(WHT(x)/32) = x (since WHT(WHT(x))=32*x).
    constexpr int num_levels = (1 << num_bits);
    int sub_id = lane / sub_size;
    float scale = __half2float(in_scales[sub_id]);
    float zero  = __half2float(in_zeros[sub_id]);
    float v = ((float)q / (float)(num_levels - 1)) * scale + zero;

    // Apply 1/sqrt(32) normalization before inverse WHT
    v *= 0.17677669529663688110f;

    // Step 4: Inverse WHT
    v = shuffle_had_fx32(v, lane);

    out[lane] = __float2half(v);
}

// ---------------------------------------------------------------------------
// Continuous-layout asymmetric kernel wrappers (sub_size=8, 4 blocks)
// ---------------------------------------------------------------------------

template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_lm_cache_cont_asym_kernel
(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales,
    half* __restrict__ out_zeros
)
{
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4
    in         += 32       * blockIdx.x;
    out        += bits     * blockIdx.x;
    out_scales += num_subs * blockIdx.x;
    out_zeros  += num_subs * blockIdx.x;
    quant_block_lm_sub_asym<bits, sub_size>(in, out, out_scales, out_zeros);
}

#define __(i) quant_lm_cache_cont_asym_kernel<i>
constexpr auto quant_lm_cache_cont_asym_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __


template <int bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void dequant_lm_cache_cont_asym_kernel
(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    const half* __restrict__ in_zeros,
    half* __restrict__ out
)
{
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4
    in        += bits     * blockIdx.x;
    in_scales += num_subs * blockIdx.x;
    in_zeros  += num_subs * blockIdx.x;
    out       += 32       * blockIdx.x;
    dequant_block_lm_sub_asym<bits, sub_size>(in, in_scales, in_zeros, out);
}

#define __(i) dequant_lm_cache_cont_asym_kernel<i>
constexpr auto dequant_lm_cache_cont_asym_kernel_instances = std::array
{
    __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

// ---------------------------------------------------------------------------
// Paged-layout asymmetric kernel wrappers
//
// The asymmetric (K) and symmetric Lloyd-Max (V) paths are combined in a
// single paged kernel so both caches are updated in one kernel launch.
// k_bits uses the asymmetric path; v_bits uses the existing Lloyd-Max sub path.
// ---------------------------------------------------------------------------

template <int k_bits, int v_bits>
__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_lm_cache_paged_asym_kernel
(
    const half* __restrict__ k_in,
    uint32_t* __restrict__ k_out,
    half* __restrict__ k_out_scales,
    half* __restrict__ k_out_zeros,
    const half* __restrict__ v_in,
    uint32_t* __restrict__ v_out,
    half* __restrict__ v_out_scales,
    const uint32_t* __restrict__ cache_seqlens,
    const uint32_t* __restrict__ block_table,
    const int blocks_per_seq,
    const int token_dim
)
{
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4

    int batch_idx  = blockIdx.z;
    int token_idx  = blockIdx.y + cache_seqlens[batch_idx];
    int page_idx   = token_idx / CQ_PAGE_SIZE;
    int token_pos  = block_table[blocks_per_seq * batch_idx + page_idx] * CQ_PAGE_SIZE
                     + (token_idx % CQ_PAGE_SIZE);
    int sub_pos    = (token_pos * token_dim + blockDim.x * blockIdx.x + threadIdx.x) / 32;

    // K: asymmetric (scale + zero-point)
    quant_block_lm_sub_asym<k_bits, sub_size>(
        k_in  + sub_pos * 32,
        k_out + sub_pos * k_bits,
        k_out_scales + sub_pos * num_subs,
        k_out_zeros  + sub_pos * num_subs
    );

    // V: symmetric Lloyd-Max (unchanged)
    quant_block_lm_sub<v_bits, sub_size>(
        v_in  + sub_pos * 32,
        v_out + sub_pos * v_bits,
        v_out_scales + sub_pos * num_subs
    );
}

#define __(i, j) quant_lm_cache_paged_asym_kernel<i, j>
constexpr auto quant_lm_cache_paged_asym_kernel_instances = std::array
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
void dequant_lm_cache_paged_asym_kernel
(
    const uint32_t* __restrict__ k_in,
    const half* __restrict__ k_in_scales,
    const half* __restrict__ k_in_zeros,
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
    constexpr int sub_size = 8;
    constexpr int num_subs = 32 / sub_size;  // 4

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

        // K: asymmetric dequant
        dequant_block_lm_sub_asym<k_bits, sub_size>(
            k_in  + addr * k_bits,
            k_in_scales + addr * num_subs,
            k_in_zeros  + addr * num_subs,
            k_out + addr * 32
        );

        // V: symmetric Lloyd-Max dequant (unchanged)
        dequant_block_lm_sub<v_bits, sub_size>(
            v_in  + addr * v_bits,
            v_in_scales + addr * num_subs,
            v_out + addr * 32
        );

        t_warp_id += d_warp_id;
    }
}

#define __(i, j) dequant_lm_cache_paged_asym_kernel<i, j>
constexpr auto dequant_lm_cache_paged_asym_kernel_instances = std::array
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
