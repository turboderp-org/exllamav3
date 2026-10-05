#pragma once

// Launch-count folds for the mHC decode sites, used by hc_mix_fused (hc_mix.cu).
//
// A decode step runs two hyper-connection sites per layer, each an apply_ (one hc_apply launch, in place on the
// residual streams) followed by the next site's mix (partials + finalize) and the RMSNorm the block runs on the
// collapsed output. Every launch pays a dispatch gap that is a large part of these small kernels' cost (most of
// it on RDNA, where the folds are on by default), so:
//   - the pending apply runs inside the next mix's partials kernel (hc_apply_partials_kernel), and
//   - the norm runs at the end of the finalize kernel (hc_finalize_norm_row), one block per row.
// Both reproduce the unfused kernels' arithmetic in the same order, so the results are bit-identical.

#include "hc_lanes.cuh"

// The RMSNorm folded into the finalize: rms_norm with RES_NONE, half in and out, half or bf16 weight
struct HcNormArgs
{
    const void* w;          // (D) half or bf16 weight, or null (unweighted)
    bool w_bf16;
    half* y;                // (R, D) normed output
    float eps;
    float constant_bias;
    float constant_scale;
};

// A deferred hc_apply on the streams of the next mix: x <- post * y + comb^T x
struct HcPendingApply
{
    const void* y;          // (R, D)
    bool y_half;
    const float* post;      // (R, H)
    const float* comb;      // (R, H, H)
};

// Phase C of hc_mix_finalize_kernel for a whole row in one block, followed by the RMSNorm of that row. Every
// thread of the block enters (the sink warp included: it stores nothing in phase C but takes part in the
// norm's reductions). The collapsed row is written to global memory as in the unfused kernel and kept in LDS
// for the norm, which replays rms_norm_kernel's single-pass form exactly: virtual thread t < T = min(1024,
// ceil(D / 4 / 32) * 32), the norm's block size for this row length, owns float4 column t; the sum of squares
// in the same fma order, reduce_dyn's two xor-butterfly stages, rmf, then (x * w) * rmf and RNE to half.
// Needs D / 4 <= 1024 (the norm's single-pass regime, and five column quads per thread here)
template <int H>
__device__ __forceinline__ void hc_finalize_norm_row
(
    const float4* __restrict__ s4,       // this row's streams
    const float (&pre_r)[H],
    const bool sink_warp,
    const int tid,
    const int nth,
    const int D,
    half* __restrict__ collapsed_row,
    const HcNormArgs& norm,
    const int r
)
{
    __shared__ half2 sh_norm_x[2048];
    __shared__ float sh_sums[32];
    const int D4 = D / 4;

    // Issue every stream load of this thread first, then the unfused kernel's arithmetic
    constexpr int NIT = 5;
    float4 sv[NIT][H];
    #pragma unroll
    for (int it = 0; it < NIT; ++it)
    {
        const int c = tid + it * nth;
        if (!sink_warp && c < D4)
        {
            #pragma unroll
            for (int h = 0; h < H; ++h) sv[it][h] = s4[(size_t) h * D4 + c];
        }
    }
    half2* out2 = (half2*) collapsed_row;
    #pragma unroll
    for (int it = 0; it < NIT; ++it)
    {
        const int c = tid + it * nth;
        if (sink_warp || c >= D4) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            o.x = fmaf(pre_r[h], sv[it][h].x, o.x);
            o.y = fmaf(pre_r[h], sv[it][h].y, o.y);
            o.z = fmaf(pre_r[h], sv[it][h].z, o.z);
            o.w = fmaf(pre_r[h], sv[it][h].w, o.w);
        }
        half2 o01 = __floats2half2_rn(o.x, o.y);
        half2 o23 = __floats2half2_rn(o.z, o.w);
        out2[c * 2] = o01;
        out2[c * 2 + 1] = o23;
        sh_norm_x[c * 2] = o01;
        sh_norm_x[c * 2 + 1] = o23;
    }
    __syncthreads();

    const int columns = D4;
    const int T = min(1024, ((columns + 31) / 32) * 32);
    const int n_vw = T / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const half4* xin = (const half4*) sh_norm_x;

    auto read_x = [&] (int col) -> float4
    {
        half4 h4 = xin[col];
        float4 f4;
        f4.x = CLAMP_FP16(LOW_TO_FLOAT(h4.x));
        f4.y = CLAMP_FP16(HIGH_TO_FLOAT(h4.x));
        f4.z = CLAMP_FP16(LOW_TO_FLOAT(h4.y));
        f4.w = CLAMP_FP16(HIGH_TO_FLOAT(h4.y));
        return f4;
    };
    // xor butterfly 16, 8, 4, 2, 1 (reduce_dyn's order; every lane ends with the same sum)
    auto bfly = [&] (float v) -> float
    {
        v += hc_xor<16>(v);
        v += hc_xor<8>(v);
        v += hc_xor<4>(v);
        v += hc_xor<2>(v);
        v += hc_xor<1>(v);
        return v;
    };

    for (int vw = warp; vw < n_vw; vw += blockDim.x / 32)
    {
        const int col = vw * 32 + lane;
        float sum = 0.0f;
        if (col < columns)
        {
            float4 x4 = read_x(col);
            sum = fma(x4.x, x4.x, sum);
            sum = fma(x4.y, x4.y, sum);
            sum = fma(x4.z, x4.z, sum);
            sum = fma(x4.w, x4.w, sum);
        }
        sum = bfly(sum);
        if (lane == 0) sh_sums[vw] = sum;
    }
    __syncthreads();
    const float total = n_vw == 1 ? sh_sums[0] : bfly(lane < n_vw ? sh_sums[lane] : 0.0f);
    const float rmf = rsqrtf(total / (float) D + norm.eps) * norm.constant_scale;

    half4* yout = (half4*) (norm.y + (size_t) r * D);
    for (int col = threadIdx.x; col < columns; col += blockDim.x)
    {
        float4 x4 = read_x(col);
        if (norm.w)
        {
            float4 w4;
            if (norm.w_bf16)
            {
                bfloat164 b4;
                READ64(b4, ((const bfloat164*) norm.w) + col);
                w4.x = __bfloat162float(__low2bfloat16(b4.x));
                w4.y = __bfloat162float(__high2bfloat16(b4.x));
                w4.z = __bfloat162float(__low2bfloat16(b4.y));
                w4.w = __bfloat162float(__high2bfloat16(b4.y));
            }
            else
            {
                half4 h4;
                READ64(h4, ((const half4*) norm.w) + col);
                w4.x = LOW_TO_FLOAT(h4.x);
                w4.y = HIGH_TO_FLOAT(h4.x);
                w4.z = LOW_TO_FLOAT(h4.y);
                w4.w = HIGH_TO_FLOAT(h4.y);
            }
            if (norm.constant_bias != 0.0f)
            {
                w4.x += norm.constant_bias;
                w4.y += norm.constant_bias;
                w4.z += norm.constant_bias;
                w4.w += norm.constant_bias;
            }
            x4.x = x4.x * w4.x * rmf;
            x4.y = x4.y * w4.y * rmf;
            x4.z = x4.z * w4.z * rmf;
            x4.w = x4.w * w4.w * rmf;
        }
        else
        {
            x4.x = x4.x * rmf;
            x4.y = x4.y * rmf;
            x4.z = x4.z * rmf;
            x4.w = x4.w * rmf;
        }
        half4 h4
        (
            __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y)),
            __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w))
        );
        WRITE64(yout + col, h4);
    }
}

// hc_apply_kernel's arithmetic for stream g, then hc_mix_partials_kernel's for the chunk of stream g this group
// of NUM_THREADS_A threads owns. A block is H groups over the same column range of the H streams, so every
// stream value the apply needs is in the block: the old values go through LDS before anyone writes, then each
// group stores its stream in place (no other block touches these columns). Chunk g * cps + blockIdx.x gets
// exactly the per-thread accumulation sequence and the two-warp reduce of the unfused partials kernel (same
// chunk_cols, same thread -> column map), so the partials, and with them post / comb / collapsed, are
// bit-identical to hc_apply followed by hc_mix. Needs each stream to be a whole number of chunks
template <int H, int M_, typename FN_T, typename Y_T>
__global__ __launch_bounds__(H * NUM_THREADS_A, 1)
void hc_apply_partials_kernel
(
    float* __restrict__ streams,         // (R, H * D), updated in place
    const Y_T* __restrict__ y,           // (R, D)
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H)
    const FN_T* __restrict__ fn,         // (M, H * D)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols,
    const int n_chunks_a
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int g = threadIdx.x / NUM_THREADS_A;
    const int t = threadIdx.x % NUM_THREADS_A;
    const int D = row_len / H;
    const int D4 = D / 4;
    const int cps = n_chunks_a / H;
    const int chunk = g * cps + blockIdx.x;
    const int c0 = chunk * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        #pragma unroll
        for (int q = 0; q < H; ++q)
            comb_r[h][q] = __ldg(comb + ((size_t) r * H + h) * H + q);
    }

    float4* s4 = (float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    __shared__ float4 xs[H][NUM_THREADS_A];

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    // Iteration i: every group at in-stream float4 column (c0 - g * D) / 4 + t + i * NUM_THREADS_A
    const int n_it = (chunk_cols / 4 + NUM_THREADS_A - 1) / NUM_THREADS_A;
    for (int i = 0; i < n_it; ++i)
    {
        const int c = c0 / 4 + t + i * NUM_THREADS_A;     // flattened float4 index
        const bool valid = c < c1 / 4;
        const int cd = c - g * D4;                        // column within stream g
        // fn rows first: independent of the apply, so all M loads are in flight together with the stream load
        float4 wv[M];
        if (valid)
        {
            #pragma unroll
            for (int j = 0; j < M; ++j)
            {
                if constexpr (std::is_same_v<FN_T, half>)
                {
                    int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                    float2 lo = __half22float2(*(const half2*) &pk.x);
                    float2 hi = __half22float2(*(const half2*) &pk.y);
                    wv[j] = make_float4(lo.x, lo.y, hi.x, hi.y);
                }
                else
                    wv[j] = ((const float4*) fn)[(size_t) j * row_len4 + c];
            }
            xs[g][t] = s4[c];
        }
        __syncthreads();
        if (valid)
        {
            float4 xv[H];
            #pragma unroll
            for (int h = 0; h < H; ++h) xv[h] = xs[h][t];
            float4 yv;
            if constexpr (std::is_same_v<Y_T, half>)
            {
                half2 y01 = ((const half2*) (y + (size_t) r * D))[cd * 2];
                half2 y23 = ((const half2*) (y + (size_t) r * D))[cd * 2 + 1];
                float2 lo = __half22float2(y01);
                float2 hi = __half22float2(y23);
                yv = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                yv = ((const float4*) (y + (size_t) r * D))[cd];
            float4 o;
            o.x = post_r[g] * yv.x;
            o.y = post_r[g] * yv.y;
            o.z = post_r[g] * yv.z;
            o.w = post_r[g] * yv.w;
            #pragma unroll
            for (int q = 0; q < H; ++q)
            {
                o.x = fmaf(comb_r[q][g], xv[q].x, o.x);
                o.y = fmaf(comb_r[q][g], xv[q].y, o.y);
                o.z = fmaf(comb_r[q][g], xv[q].z, o.z);
                o.w = fmaf(comb_r[q][g], xv[q].w, o.w);
            }
            s4[c] = o;

            const float4 sv = o;
            acc[M] = fmaf(sv.x, sv.x, acc[M]);
            acc[M] = fmaf(sv.y, sv.y, acc[M]);
            acc[M] = fmaf(sv.z, sv.z, acc[M]);
            acc[M] = fmaf(sv.w, sv.w, acc[M]);
            #pragma unroll
            for (int j = 0; j < M; ++j)
            {
                const float4 w = wv[j];
                float d = fmaf(sv.x, w.x, fmaf(sv.y, w.y, fmaf(sv.z, w.z, sv.w * w.w)));
                acc[j] += d;
            }
        }
        __syncthreads();
    }

    // The unfused kernel's reduce, per group (its two warps)
    __shared__ float red[H][NUM_THREADS_A / 32][M + 1];
    const int lane = t % 32;
    const int warp = t / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = hc_warp_sum_lane0(acc[k]);
        if (lane == 0) red[g][warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * n_chunks_a + chunk) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[g][w][k];
            out[k] = v;
        }
    }
}
