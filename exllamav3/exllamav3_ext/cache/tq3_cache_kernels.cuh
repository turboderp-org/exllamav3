#pragma once

// Lloyd-Max boundaries for 3-level Gaussian quantizer
// Boundary = +/- 0.6123724 * scale (but we work in normalized [-1,1] space)
// After WHT + divide by max, values are in [-1, 1]
// Decision boundary at +/- 0.5 (approximation that works well in practice
// since WHT output is approximately Gaussian)

#define TQ3_BOUNDARY 0.5f
#define TQ3_CENTROID 1.0f  // centroids are {-1, 0, +1} after normalization
#define TQ3_INV_SQRT32 0.17677669529663688110f  // 1/sqrt(32)

// ============================================================================
// TQ3 quantize block: 32 fp16 values -> 2 bitplane uint32s + 1 fp16 scale
//
// Encoding: trit 0 = -1, trit 1 = 0, trit 2 = +1
// Bitplane 0: set if trit != 1 (i.e., non-zero)
// Bitplane 1: set if trit == 2 (i.e., positive)
// ============================================================================

__device__ __forceinline__ void quant_tq3_block(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    int t = threadIdx.x & 31;

    // Load, rotate (WHT) and scale
    float v = __half2float(in[t]);
    v = shuffle_had_fx32(v, t);
    v *= TQ3_INV_SQRT32;

    // Find scale (max absolute value)
    float s = shuffle_max_fx32(fabsf(v) + 1e-10f);
    half sh = __float2half_rn(s);
    v /= s;

    // Lloyd-Max ternary quantization
    // v is in [-1, 1] after normalization
    // Trit encoding: -1 -> (nonzero=1, positive=0)
    //                 0 -> (nonzero=0, positive=0)
    //                +1 -> (nonzero=1, positive=1)
    int nonzero = (fabsf(v) >= TQ3_BOUNDARY) ? 1 : 0;
    int positive = (v > 0.0f && nonzero) ? 1 : 0;

    // Pack via ballot
    uint32_t bp0 = __ballot_sync(0xffffffff, nonzero);   // bitplane 0: is-nonzero
    uint32_t bp1 = __ballot_sync(0xffffffff, positive);  // bitplane 1: is-positive

    // Write output (2 uint32 bitplanes + 1 fp16 scale)
    if (t == 0) out[0] = bp0;
    if (t == 1) out[1] = bp1;
    if (t == 2) *out_scales = sh;
}


// ============================================================================
// TQ3 dequantize block: 2 bitplane uint32s + 1 fp16 scale -> 32 fp16 values
// ============================================================================

__device__ __forceinline__ void dequant_tq3_block(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out
)
{
    int lane = threadIdx.x & 31;

    // Load bitplanes
    uint32_t bp0 = in[0];  // nonzero mask
    uint32_t bp1 = in[1];  // positive mask

    // Decode trit for this lane
    int is_nonzero = (bp0 >> lane) & 1u;
    int is_positive = (bp1 >> lane) & 1u;

    // Reconstruct: nonzero=0 -> 0.0, nonzero=1 && positive=0 -> -1.0, nonzero=1 && positive=1 -> +1.0
    // Using the Lloyd-Max centroid spacing:
    float v = is_nonzero ? (is_positive ? TQ3_INV_SQRT32 : -TQ3_INV_SQRT32) : 0.0f;

    // Scale and inverse WHT
    v *= __half2float(*in_scales);
    v = shuffle_had_fx32(v, lane);

    // Store
    out[lane] = __float2half(v);
}


// ============================================================================
// Contiguous kernels
// ============================================================================

__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_tq3_cache_cont_kernel(
    const half* __restrict__ in,
    uint32_t* __restrict__ out,
    half* __restrict__ out_scales
)
{
    in += 32 * blockIdx.x;
    out += 2 * blockIdx.x;
    out_scales += blockIdx.x;
    quant_tq3_block(in, out, out_scales);
}

__global__ __launch_bounds__(MAX_WARPS * 32)
void dequant_tq3_cache_cont_kernel(
    const uint32_t* __restrict__ in,
    const half* __restrict__ in_scales,
    half* __restrict__ out
)
{
    in += 2 * blockIdx.x;
    in_scales += blockIdx.x;
    out += 32 * blockIdx.x;
    dequant_tq3_block(in, in_scales, out);
}


// ============================================================================
// Paged kernels (follow same pattern as q_cache_kernels.cuh)
// ============================================================================

__global__ __launch_bounds__(MAX_WARPS * 32)
void quant_tq3_cache_paged_kernel(
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
    int batch_idx = blockIdx.z;
    int token_idx = blockIdx.y + cache_seqlens[batch_idx];
    int page_idx = token_idx / CQ_PAGE_SIZE;
    int token_pos = block_table[blocks_per_seq * batch_idx + page_idx] * CQ_PAGE_SIZE
                    + (token_idx % CQ_PAGE_SIZE);
    int sub_pos = (token_pos * token_dim + blockDim.x * blockIdx.x + threadIdx.x) / 32;

    quant_tq3_block(k_in + sub_pos * 32, k_out + sub_pos * 2, k_out_scales + sub_pos);
    quant_tq3_block(v_in + sub_pos * 32, v_out + sub_pos * 2, v_out_scales + sub_pos);
}


__global__ __launch_bounds__(MAX_WARPS * 32)
void dequant_tq3_cache_paged_kernel(
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

        dequant_tq3_block(k_in + addr * 2, k_in_scales + addr, k_out + addr * 32);
        dequant_tq3_block(v_in + addr * 2, v_in_scales + addr, v_out + addr * 32);

        t_warp_id += d_warp_id;
    }
}
