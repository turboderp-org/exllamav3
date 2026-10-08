#include <cuda_fp16.h>
#include "hc_mix.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"

/*

Fused mHC HyperConnection mix() kernel

mix(streams (R, H, D) fp32) -> post (R, H), comb (R, H, H), collapsed (R, D):

  flat = rmsnorm_unweighted(streams.flatten)          (row of H * D values)
  mixv = flat @ fn.T                                  (M = 2H + H^2 outputs)
  pre  = sigmoid(mixv[0:H]  * s0 + base[0:H]) + eps
  post = 2 sigmoid(mixv[H:2H] * s1 + base[H:2H])
  comb = sinkhorn(softmax(mixv[2H:] * s2 + base[2H:]))   (iters alternating row/col norms)
  collapsed = sum_h pre[h] * streams[h]

Two launches, no grid-wide sync, deterministic (fixed reduction order, no atomics):
  K1 partials: grid (chunksA, R); each block reduces its column chunk to M + 1 partials.
  K2 finalize: grid (chunksC, R); EVERY block re-reduces the tiny partial matrix and
    derives rmr + pre redundantly (removes the cross-block dependency), then streams its
    chunk of collapsed; the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0
    (row sums: shfl_xor 1|2, col sums: shfl_xor 4|8 for H = 4) and writes post/comb.

*/

// Partials blocks are small: at R = 1 the grid is the only parallelism, so favor many
// blocks (row_len / (4 * 64) chunks) over wide ones; the M + 1 block reduce also shrinks
#define NUM_THREADS 256
#define NUM_THREADS_A 64

#include "hc_lanes.cuh"
#include "hc_fuse.cuh"

__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

template <int H, int M_, typename FN_T>
__global__ __launch_bounds__(NUM_THREADS_A)
void hc_mix_partials_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const FN_T* __restrict__ fn,         // (M, H * D) float, or half (opt-in, halves traffic)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols                 // multiple of 4 * NUM_THREADS_A
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
    {
        float4 s = s4[c];
        acc[M] = fmaf(s.x, s.x, acc[M]);
        acc[M] = fmaf(s.y, s.y, acc[M]);
        acc[M] = fmaf(s.z, s.z, acc[M]);
        acc[M] = fmaf(s.w, s.w, acc[M]);
        #pragma unroll
        for (int j = 0; j < M; ++j)
        {
            float4 w;
            if constexpr (std::is_same_v<FN_T, half>)
            {
                // Single vectorized 8-byte load (two half2 loads would double the LDGs)
                int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                float2 lo = __half22float2(*(const half2*) &pk.x);
                float2 hi = __half22float2(*(const half2*) &pk.y);
                w = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                w = ((const float4*) fn)[(size_t) j * row_len4 + c];
            float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
            acc[j] += d;
        }
    }

    // Block reduce M + 1 lanes' accumulators
    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = hc_warp_sum_lane0(acc[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * gridDim.x + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[w][k];
            out[k] = v;
        }
    }
}

// NORM (decode rows): one block per row covers all D columns and runs the following RMSNorm on the collapsed
// row (hc_fuse.cuh)
template <int H, int M_, bool HEAD, bool HALF_OUT, bool NORM = false>
__global__ __launch_bounds__(NUM_THREADS)
void hc_mix_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    void* __restrict__ collapsed,        // (R, D) float or half
    const int D,
    const int chunksA,
    const int chunk_cols_c,              // multiple of 4
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters,
    const HcNormArgs norm = {}
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;

    // Re-reduce this row's partials (tiny, L2-resident). Every block does this to avoid
    // cross-block dependencies. Warp-split over the chunk axis: the serial loop sits
    // at the head of the kernel's critical path, so with many partials chunks (small
    // NUM_THREADS_A blocks) a single-thread-per-quantity loop is too long
    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane <= M)
        {
            const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
            float v = 0.0f;
            for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
                v += p[(size_t) i * (M + 1)];
            red_s[warp][lane] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w)
            v += red_s[w][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();

    // Sinkhorn + post/comb writes: chunk-0 block only, one lane per comb element. Runs in
    // a dedicated warp, concurrent with the other warps' phase C: the ~20-iteration
    // normalization is a serial shfl/div latency chain that only needs the M + 1 reduced
    // scalars, so at small R it sets the kernel's critical path; don't stack phase C
    // work in front of it.
    //
    // Lane l = i * H + j; row sums reduce over j (xor 1..H/2), col sums over i (xor H..).
    const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
    if (sink_warp)
    {
        if (threadIdx.x < H)
            post[(size_t) r * H + threadIdx.x] =
                2.0f * sigmoidf_(fmaf(mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x]));

        if (threadIdx.x < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float v = fmaf(mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x]);

            // softmax over rows
            float m = hc_xor_max<1, H>(v, mask);
            v = __expf(v - m);
            float s = hc_xor_sum<1, H>(v, mask);
            v = __fdividef(v, s) + hc_eps;

            // column normalize, then (iters - 1) x (row, column)
            float cs = hc_xor_sum<H, H * H>(v, mask);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = hc_xor_sum<1, H>(v, mask);
                v = __fdividef(v, rs + hc_eps);
                cs = hc_xor_sum<H, H * H>(v, mask);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * H * H + threadIdx.x] = v;
        }
        // the folded norm needs every thread of the block for its reductions
        if constexpr (!NORM)
            return;
    }

    // Phase C: collapsed chunk, weighted sum over the H stream rows. In the sinkhorn
    // block the first warp is excluded, so the remaining threads re-cover its lanes
    float pre_r[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) pre_r[h] = pre_s[h];

    const bool shrunk = !HEAD && blockIdx.x == 0;
    const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
    const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
    const int c0 = blockIdx.x * chunk_cols_c;
    const int c1 = min(c0 + chunk_cols_c, D);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int D4 = D / 4;
    if constexpr (NORM)
    {
        static_assert(HALF_OUT && !HEAD, "hc_mix_finalize_kernel: the folded norm is for the half-output mix");
        hc_finalize_norm_row<H>(s4, pre_r, sink_warp, tid, nth, D, (half*) collapsed + (size_t) r * D, norm, r);
        return;
    }
    for (int c = c0 / 4 + tid; c < c1 / 4; c += nth)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) collapsed + (size_t) r * D))[c] = o;
    }
}


/*

GatedResidual (Qwen4Exp, the low-rank elementwise cousin of mHC) fused mix, decode form:

  mix(streams (R, H, D) fp32) -> post (R, H), mixed (R, D):
    normed[h] = rmsnorm(streams[h]) * w[h]           (per-STREAM norm, weighted, w incl +1)
    dots      = cat(down, inject) @ normed.flatten   (M = LR + H outputs, LR = low rank ~320)
    t         = silu(dots[:LR] / H)
    post      = 2 sigmoid(dots[LR:] / H)
    g[h, d]   = sigmoid(up[h * D + d, :] @ t)        (per-CHANNEL gate through the low rank)
    mixed[d]  = mean_h g[h, d] * normed[h, d]

  Same two-launch, no-grid-sync, deterministic shape as hc_mix, restructured for the wide low
  rank (LR >> mHC's M = 24, so per-thread accumulator arrays don't fit):
    K1 gr_dots: one block per (fn row | sum-of-squares), computing per-STREAM partial dots
      against the UNNORMALIZED streams -- by linearity the per-stream rms scale applies in K2.
      The norm weight is per (h, d), so it belongs on the stream side of the dot: the kernel
      takes the pre-weighted stream copy that the previous site's hc_apply emitted (wstreams,
      the per-element factor applied while those values were in registers anyway) and falls
      back to multiplying the raw streams by w in the inner loop when no such copy exists
      (the first site after the stream expansion, a device boundary, a PLE layer in between).
      Either way the fn table is the plain fp16 projection, shared with the tiled prefill path.
    K2 gr_finalize: every block re-derives rmr / t redundantly from the K1 output (the mHC
      finalize pattern), then streams its chunk of mixed, evaluating the per-channel up-gate
      inline -- upT is laid out (LR, H * D) so the serial rank loop reads coalesced and stays
      L2-resident at decode R. NOT for large R (the untiled up/down reads defeat the L2);
      the python side runs a plain half-GEMM path for prefill.

  apply_ is hc_apply with no comb: x[h] += post[h] * y, optionally also writing the next
  site's weighted copy xw[h] = x[h] * w_next[h] for its K1.

*/

#define GR_THREADS_A 128

template <int H, bool WEIGHTED>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_kernel
(
    const float* __restrict__ streams,   // (R, H, D) raw (sum of squares)
    const float* __restrict__ wstreams,  // (R, H, D) streams * w, or null (WEIGHTED = false)
    const half* __restrict__ fn,         // (M, H * D) half projection
    const half* __restrict__ wn,         // (H * D) half norm weight, applied here when not WEIGHTED
    float* __restrict__ dots,            // (R, M + 1, H): per-stream dots, row M = sum sq
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;            // fn row, or M for the sum-of-squares row
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    const float4* d4 = (const float4*) ((WEIGHTED ? wstreams : streams) + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 16-byte fn loads (8 halves) against two 16-byte stream quads
            const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
            const int4* w8 = (const int4*) (wn + (size_t) h * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = d4[(size_t) h * D4 + 2 * c];
                float4 s1 = d4[(size_t) h * D4 + 2 * c + 1];
                if constexpr (!WEIGHTED)
                {
                    int4 wk = w8[c];
                    half4 v0 = *(half4*) &wk.x;
                    half4 v1 = *(half4*) &wk.z;
                    s0.x = __fmul_rn(s0.x, LOW_TO_FLOAT(v0.x)); s0.y = __fmul_rn(s0.y, HIGH_TO_FLOAT(v0.x));
                    s0.z = __fmul_rn(s0.z, LOW_TO_FLOAT(v0.y)); s0.w = __fmul_rn(s0.w, HIGH_TO_FLOAT(v0.y));
                    s1.x = __fmul_rn(s1.x, LOW_TO_FLOAT(v1.x)); s1.y = __fmul_rn(s1.y, HIGH_TO_FLOAT(v1.x));
                    s1.z = __fmul_rn(s1.z, LOW_TO_FLOAT(v1.y)); s1.w = __fmul_rn(s1.w, HIGH_TO_FLOAT(v1.y));
                }
                int4 pk = f8[c];
                half4 w0 = *(half4*) &pk.x;
                half4 w1 = *(half4*) &pk.z;
                a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots
    const half* __restrict__ upt,        // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,                        // low rank; M = LR + (post ? H : 0)
    const int chunk_cols,                // multiple of 4
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    // Redundant per-block head derivation (no cross-block dependency): rmr from the sumsq
    // row, then the silu'd low-rank activations into shared memory
    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    // Streamed mixed chunk: one WARP per column quad, the LR-long up-gate dots split across
    // the lanes (lane l covers ranks l, l+32, ...) and shfl-reduced -- at decode R the total
    // column count is small (R * D / 4), so per-thread columns would leave the GPU nearly
    // idle with each thread serializing the rank loop
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // upx layout (H, D / 4, LR, 4): consecutive lanes (consecutive i) read
                // consecutive 8-byte quads, so the rank loop is fully coalesced
                half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                g[h].x = fmaf(ti, LOW_TO_FLOAT(u.x), g[h].x);
                g[h].y = fmaf(ti, HIGH_TO_FLOAT(u.x), g[h].y);
                g[h].z = fmaf(ti, LOW_TO_FLOAT(u.y), g[h].z);
                g[h].w = fmaf(ti, HIGH_TO_FLOAT(u.y), g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}



// ---------------------------------------------------------------------------------------------
// Decode mix, bytes-in-flight version (benchmarks/gr_mix_decode): the fn table is read once per
// site for all rows, with each lane's chunks of GR_J rows issued together, one warp per stream;
// the finalize issues all its 16-byte up loads before use and runs rows in groups. Cold on a PRO
// 6000: 16.1 -> 14.9 us at one row, 33 -> 26 at eight. Shape-templated on the chunks per lane
// (D / 256) and rank iterations (LR / 64); other shapes take the generic kernels above.
#define GR_THREADS_A2 128           // 4 warps: one per stream
#define GR_THREADS_C 128            // 4 warps: one column quad each
#define GR_RB 4                     // rows per accumulator pass
#define GR_MAX_R 32
#define GR_J 2                      // fn rows per dots block (share the stream loads)

// Phase A: dots[r, j, h] = <wstreams[r, h, :], fn[j, h, :]> for every row r (row M: sum of
// squares of the raw streams). One block per fn row j, one warp per stream h: each lane holds
// its 10 x 16-byte chunks of the fn row in registers (issued together: the whole 20 KB row is in
// flight per block), then runs the rows in groups of GR_RB against the L2-resident streams. The
// table is read once per site regardless of R. Without a weighted copy (WEIGHTED = false) the
// lane also holds its chunks of w and scales the raw stream values before the FMAs.
template <int H, int NCH, bool WEIGHTED>
__device__ __forceinline__ void gr_dots_block
(
    const float* __restrict__ streams, const float* __restrict__ wstreams, const half* __restrict__ fn,
    const half* __restrict__ wn, float* __restrict__ dots, const int M, const int D, const int R, const int block
)
{
    const int j0 = block * GR_J;
    const int h = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int D8 = D / 8;                // chunks per stream row
    const float* dsrc = WEIGHTED ? wstreams : streams;
    if (j0 < M)
    {
        int4 f[GR_J][NCH];
        int4 wc[WEIGHTED ? 1 : NCH];
        if constexpr (!WEIGHTED)
        {
            const int4* w8 = (const int4*) (wn + (size_t) h * D);
            #pragma unroll
            for (int k = 0; k < NCH; ++k)
            {
                const int c = lane + k * 32;
                wc[k] = (c < D8) ? w8[c] : make_int4(0, 0, 0, 0);
            }
        }
        #pragma unroll
        for (int jj = 0; jj < GR_J; ++jj)
        {
            const int4* f8 = (const int4*) (fn + ((size_t) min(j0 + jj, M - 1) * H + h) * D);
            #pragma unroll
            for (int k = 0; k < NCH; ++k)
            {
                const int c = lane + k * 32;
                f[jj][k] = (c < D8) ? f8[c] : make_int4(0, 0, 0, 0);
            }
        }
        for (int r0 = 0; r0 < R; r0 += GR_RB)
        {
            float acc[GR_RB][GR_J];
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
                #pragma unroll
                for (int jj = 0; jj < GR_J; ++jj) acc[q][jj] = 0.0f;
            #pragma unroll
            for (int k = 0; k < NCH; ++k)
            {
                const int c = lane + k * 32;
                const bool ok = c < D8;
                #pragma unroll
                for (int q = 0; q < GR_RB; ++q)
                {
                    const int r = r0 + q;
                    const float4* s4 = (const float4*) (dsrc + ((size_t) min(r, R - 1) * H + h) * D + (ok ? c : 0) * 8);
                    float4 s0 = s4[0], s1 = s4[1];
                    if constexpr (!WEIGHTED)
                    {
                        const half2* v2 = (const half2*) &wc[k];
                        float2 v0 = __half22float2(v2[0]), v1 = __half22float2(v2[1]), v2_ = __half22float2(v2[2]), v3 = __half22float2(v2[3]);
                        s0.x = __fmul_rn(s0.x, v0.x); s0.y = __fmul_rn(s0.y, v0.y); s0.z = __fmul_rn(s0.z, v1.x); s0.w = __fmul_rn(s0.w, v1.y);
                        s1.x = __fmul_rn(s1.x, v2_.x); s1.y = __fmul_rn(s1.y, v2_.y); s1.z = __fmul_rn(s1.z, v3.x); s1.w = __fmul_rn(s1.w, v3.y);
                    }
                    #pragma unroll
                    for (int jj = 0; jj < GR_J; ++jj)
                    {
                        const half2* w2 = (const half2*) &f[jj][k];
                        float2 w0 = __half22float2(w2[0]), w1 = __half22float2(w2[1]), w2_ = __half22float2(w2[2]), w3 = __half22float2(w2[3]);
                        float a = acc[q][jj];
                        a = fmaf(s0.x, w0.x, a); a = fmaf(s0.y, w0.y, a); a = fmaf(s0.z, w1.x, a); a = fmaf(s0.w, w1.y, a);
                        a = fmaf(s1.x, w2_.x, a); a = fmaf(s1.y, w2_.y, a); a = fmaf(s1.z, w3.x, a); a = fmaf(s1.w, w3.y, a);
                        acc[q][jj] = a;
                    }
                }
            }
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
                #pragma unroll
                for (int jj = 0; jj < GR_J; ++jj)
                {
                    float a = acc[q][jj];
                    for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
                    if (lane == 0 && r0 + q < R && j0 + jj < M) dots[((size_t) (r0 + q) * (M + 1) + j0 + jj) * H + h] = a;
                }
        }
    }
    else
    {
        // sum of squares per (row, stream)
        for (int r = 0; r < R; ++r)
        {
            const float4* s4 = (const float4*) (streams + ((size_t) r * H + h) * D);
            float a = 0.0f;
            for (int c = lane; c < D / 4; c += 32)
            {
                float4 s = s4[c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
            for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
            if (lane == 0) dots[((size_t) r * (M + 1) + M) * H + h] = a;
        }
    }
}

// Phase B: rmr and the silu'd latent per row (redundantly per block, into shared memory), then
// one warp per column quad: the up-gate dots over the low rank with 16-byte loads (two ranks per
// lane, LR / 64 unrolled iterations, all four streams issued together), rows in groups of GR_RB
// re-reading the (L2-resident) up columns; epilogue: sigmoid gates, stream mean, half/float out.
template <int H, bool HALF_OUT, int NIT>
__device__ __forceinline__ void gr_finalize_block
(
    const float* __restrict__ streams, const float* __restrict__ dots, const half* __restrict__ upt,
    const half* __restrict__ w, float* __restrict__ post, void* __restrict__ mixed,
    const int D, const int LR, const int R, const float rms_eps, const int block
)
{
    const int M = LR + (post ? H : 0);
    extern __shared__ float t_s[];       // (R, LR) then rmr (R, H)
    float* rmr_s = t_s + (size_t) R * LR;
    const int tid = threadIdx.x;
    if (tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        rmr_s[tid] = rsqrtf(dots[((size_t) r * (M + 1) + M) * H + h] / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int idx = tid; idx < R * LR; idx += GR_THREADS_C)
    {
        const int r = idx / LR, i = idx % LR;
        const float* dr = dots + ((size_t) r * (M + 1) + i) * H;
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr_s[r * H + h], dr[h], v);
        v *= inv_h;
        t_s[idx] = v * sigmoidf_(v);
    }
    if (post && block == 0 && tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        const float* dr = dots + ((size_t) r * (M + 1) + LR + h) * H;
        float v = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh) v = fmaf(rmr_s[r * H + hh], dr[hh], v);
        post[tid] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    const int lane = tid % 32, warp = tid / 32;
    const int c = block * (GR_THREADS_C / 32) + warp;           // column quad
    if (c >= D / 4) return;
    const int D4 = D / 4;
    for (int r0 = 0; r0 < R; r0 += GR_RB)
    {
        float4 g[GR_RB][H];
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h) g[q][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int4 u4[NIT][H];
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int i0 = it * 64 + lane * 2;
            #pragma unroll
            for (int h = 0; h < H; ++h)
                u4[it][h] = (i0 < LR) ? *(const int4*) (upt + ((((size_t) h * D4 + c) * LR + i0) * 4)) : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int it = 0; it < NIT; ++it)
        {
            const int i0 = it * 64 + lane * 2;              // ranks i0, i0 + 1
            #pragma unroll
            for (int q = 0; q < GR_RB; ++q)
            {
                const int r = r0 + q;
                if (r >= R) continue;
                const float t0 = i0 < LR ? t_s[(size_t) r * LR + i0] : 0.0f;
                const float t1 = i0 + 1 < LR ? t_s[(size_t) r * LR + i0 + 1] : 0.0f;
                #pragma unroll
                for (int h = 0; h < H; ++h)
                {
                    const half2* u2 = (const half2*) &u4[it][h];
                    float2 a0 = __half22float2(u2[0]), a1 = __half22float2(u2[1]);   // rank i0: 4 columns
                    float2 b0 = __half22float2(u2[2]), b1 = __half22float2(u2[3]);   // rank i0 + 1
                    g[q][h].x = fmaf(t0, a0.x, fmaf(t1, b0.x, g[q][h].x));
                    g[q][h].y = fmaf(t0, a0.y, fmaf(t1, b0.y, g[q][h].y));
                    g[q][h].z = fmaf(t0, a1.x, fmaf(t1, b1.x, g[q][h].z));
                    g[q][h].w = fmaf(t0, a1.y, fmaf(t1, b1.y, g[q][h].w));
                }
            }
        }
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
            #pragma unroll
            for (int h = 0; h < H; ++h)
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    g[q][h].x += __shfl_xor_sync(0xffffffffu, g[q][h].x, offset);
                    g[q][h].y += __shfl_xor_sync(0xffffffffu, g[q][h].y, offset);
                    g[q][h].z += __shfl_xor_sync(0xffffffffu, g[q][h].z, offset);
                    g[q][h].w += __shfl_xor_sync(0xffffffffu, g[q][h].w, offset);
                }
        if (lane != 0) continue;
        #pragma unroll
        for (int q = 0; q < GR_RB; ++q)
        {
            const int r = r0 + q;
            if (r >= R) continue;
            const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
            float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float4 sv = s4[(size_t) h * D4 + c];
                const half2* wq = (const half2*) (w + (size_t) h * D + 4 * c);
                float2 w0 = __half22float2(wq[0]), w1 = __half22float2(wq[1]);
                const float coef = rmr_s[r * H + h] * inv_h;
                o.x = fmaf(sigmoidf_(g[q][h].x) * coef * w0.x, sv.x, o.x);
                o.y = fmaf(sigmoidf_(g[q][h].y) * coef * w0.y, sv.y, o.y);
                o.z = fmaf(sigmoidf_(g[q][h].z) * coef * w1.x, sv.z, o.z);
                o.w = fmaf(sigmoidf_(g[q][h].w) * coef * w1.y, sv.w, o.w);
            }
            if (HALF_OUT)
            {
                half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
                out2[c * 2] = __floats2half2_rn(o.x, o.y);
                out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
            }
            else
                ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
        }
    }
}


template <int H, int NCH, bool WEIGHTED>
__global__ __launch_bounds__(GR_THREADS_A2)
void gr_dots_kernel2(const float* __restrict__ streams, const float* __restrict__ wstreams, const half* __restrict__ fn, const half* __restrict__ wn,
                     float* __restrict__ dots, const int M, const int D, const int R)
{
    gr_dots_block<H, NCH, WEIGHTED>(streams, wstreams, fn, wn, dots, M, D, R, blockIdx.x);
}
template <int H, bool HALF_OUT, int NIT>
__global__ __launch_bounds__(GR_THREADS_C)
void gr_finalize_kernel2(const float* __restrict__ streams, const float* __restrict__ dots, const half* __restrict__ upt, const half* __restrict__ w,
                         float* __restrict__ post, void* __restrict__ mixed, const int D, const int LR, const int R, const float rms_eps)
{
    gr_finalize_block<H, HALF_OUT, NIT>(streams, dots, upt, w, post, mixed, D, LR, R, rms_eps, blockIdx.x);
}

// Residual update for one sublayer site: x[h, d] <- post[h] * y[d] + sum_h' comb[h', h] *
// x[h', d]. Without comb (GatedResidual): x[h, d] <- post[h] * y[d] + x[h, d]. Pure per-column
// mix of the H stream rows, so it runs in place: each thread loads all H values of its columns
// into registers before writing any back. With xw (GatedResidual decode): also writes the next
// site's weighted stream copy xw[h, d] = x[h, d] * wn[h, d] for its gr_dots, while the values
// are in registers.

template <int H, typename Y_T, bool HAS_COMB, bool XW>
__global__ __launch_bounds__(NUM_THREADS)
void hc_apply_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ y,           // (R, D) float or half
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H), or null
    const half* __restrict__ wn,         // (H, D) next site's norm weight, or null
    float* __restrict__ xw,              // (R, H, D) weighted copy out, or null
    const int D,
    const int chunk_cols                 // multiple of 4
)
{
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        if (HAS_COMB)
        {
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb + ((size_t) r * H + h) * H + g);
        }
    }

    float4* x4 = (float4*) (x + (size_t) r * H * D);
    float4* xw4 = XW ? (float4*) (xw + (size_t) r * H * D) : nullptr;
    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS)
    {
        float4 xv[H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            xv[h] = x4[(size_t) h * D4 + c];

        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[c * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[c * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[c];

        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 o;
            o.x = post_r[h] * yv.x;
            o.y = post_r[h] * yv.y;
            o.z = post_r[h] * yv.z;
            o.w = post_r[h] * yv.w;
            if (HAS_COMB)
            {
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
            }
            else
            {
                o.x += xv[h].x;
                o.y += xv[h].y;
                o.z += xv[h].z;
                o.w += xv[h].w;
            }
            x4[(size_t) h * D4 + c] = o;
            if constexpr (XW)
            {
                const half2* w2 = (const half2*) (wn + (size_t) h * D + c * 4);
                float2 w01 = __half22float2(w2[0]), w23 = __half22float2(w2[1]);
                xw4[(size_t) h * D4 + c] = make_float4(__fmul_rn(o.x, w01.x), __fmul_rn(o.y, w01.y),
                                                       __fmul_rn(o.z, w23.x), __fmul_rn(o.w, w23.y));
            }
        }
    }
}

// Shared launch logic. mode: fn rows M = 2H + H^2 (mix) or H (head)

static void hc_mix_launch
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    at::Tensor& partials,
    at::Tensor* post,
    at::Tensor* comb,
    at::Tensor& collapsed,
    Graph* graph,
    const HcPendingApply* pend = nullptr,      // fold a preceding hc_apply on these streams into the partials
    const HcNormArgs* nrm = nullptr            // fold the following RMSNorm into the finalize
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous(), "hc_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int row_len = H * D;
    int M = fn.size(0);
    bool head = M == H;
    TORCH_CHECK(H == 4, "hc_mix: H = 4 only");
    TORCH_CHECK(head || M == 2 * H + H * H, "hc_mix: fn rows must be H or 2H + H^2");
    TORCH_CHECK(fn.size(1) == row_len && D % 4 == 0, "hc_mix: dims");
    bool fn_half = fn.dtype() == at::kHalf;
    if (!fn_half) TORCH_CHECK_DTYPE(fn, kFloat);
    TORCH_CHECK_DTYPE(base, kFloat);
    TORCH_CHECK_DTYPE(scale, kFloat);
    TORCH_CHECK(D > 0, "hc_mix: norm over an empty stream dimension");
    TORCH_CHECK(partials.size(1) > 0, "hc_mix: partials workspace too small");

    bool half_out = collapsed.dtype() == at::kHalf;
    if (pend || nrm) TORCH_CHECK(!head && half_out, "hc_mix: folds are for the half-output mix");
    if (nrm) TORCH_CHECK(D / 4 <= 1024, "hc_mix: folded norm needs D <= 4096");
    if (!R) return;

    int n_chunks_a = partials.size(1);
    int chunk_cols = ((row_len / n_chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    n_chunks_a = (row_len + chunk_cols - 1) / chunk_cols;
    TORCH_CHECK(n_chunks_a <= partials.size(1), "hc_mix: partials workspace too small");

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols_c = ((D / chunks_c + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks_c = (D + chunk_cols_c - 1) / chunk_cols_c;

    dim3 grid_a(n_chunks_a, R);
    dim3 grid_c(n_chunks_c, R);
    #define ARGS_A(FN_T) \
        (const float*) streams.data_ptr(), (const FN_T*) fn.data_ptr(), \
        (float*) partials.data_ptr(), row_len, chunk_cols
    #define ARGS_C(POST, COMB) \
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        POST, COMB, collapsed.data_ptr(), \
        D, n_chunks_a, chunk_cols_c, rms_eps, hc_eps, sinkhorn_iters
    if (pend || nrm)
    {
        // The pending apply runs inside the partials kernel when each stream is a whole number of
        // partials chunks (always at decode row counts); otherwise as its own launch first
        const bool fusable = pend && (D % chunk_cols) == 0 && n_chunks_a % H == 0;
        if (pend && !fusable)
        {
            int chunks_p = std::min(32, std::max(1, 256 / R));
            int chunk_cols_p = ((D / chunks_p + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
            dim3 grid_p((D + chunk_cols_p - 1) / chunk_cols_p, R);
            #define ARGS_P(Y_T) \
                (float*) streams.data_ptr(), (const Y_T*) pend->y, pend->post, pend->comb, nullptr, nullptr, D, chunk_cols_p
            if (pend->y_half) hc_apply_kernel<4, half, true, false><<<grid_p, NUM_THREADS, 0, stream>>>(ARGS_P(half));
            else              hc_apply_kernel<4, float, true, false><<<grid_p, NUM_THREADS, 0, stream>>>(ARGS_P(float));
            #undef ARGS_P
            cuda_check(cudaPeekAtLastError());
        }
        if (fusable)
        {
            dim3 grid_f(n_chunks_a / H, R);
            #define ARGS_F(FN_T, Y_T) \
                (float*) streams.data_ptr(), (const Y_T*) pend->y, pend->post, pend->comb, \
                (const FN_T*) fn.data_ptr(), (float*) partials.data_ptr(), row_len, chunk_cols, n_chunks_a
            #define LAUNCH_F(FN_T, Y_T) \
                hc_apply_partials_kernel<4, 24, FN_T, Y_T><<<grid_f, 4 * NUM_THREADS_A, 0, stream>>>(ARGS_F(FN_T, Y_T));
            if (fn_half) { if (pend->y_half) { LAUNCH_F(half, half) } else { LAUNCH_F(half, float) } }
            else         { if (pend->y_half) { LAUNCH_F(float, half) } else { LAUNCH_F(float, float) } }
            #undef LAUNCH_F
            #undef ARGS_F
        }
        else
        {
            if (fn_half) hc_mix_partials_kernel<4, 24, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
            else         hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        }
        cuda_check(cudaPeekAtLastError());
        float* post_p = (float*) post->data_ptr();
        float* comb_p = (float*) comb->data_ptr();
        if (nrm)
        {
            // One block per row covering all D columns
            const int chunk_cols_c_n = ((D + 3) / 4) * 4;
            hc_mix_finalize_kernel<4, 24, false, true, true><<<dim3(1, R), NUM_THREADS, 0, stream>>>
            (
                (const float*) streams.data_ptr(), (const float*) partials.data_ptr(),
                (const float*) base.data_ptr(), (const float*) scale.data_ptr(),
                post_p, comb_p, collapsed.data_ptr(),
                D, n_chunks_a, chunk_cols_c_n, rms_eps, hc_eps, sinkhorn_iters, *nrm
            );
        }
        else
            hc_mix_finalize_kernel<4, 24, false, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
        cuda_check(cudaPeekAtLastError());
        return;
    }
    if (!head)
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 24, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        float* post_p = (float*) post->data_ptr();
        float* comb_p = (float*) comb->data_ptr();
        if (half_out)
            hc_mix_finalize_kernel<4, 24, false, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
        else
            hc_mix_finalize_kernel<4, 24, false, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
    }
    else
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 4, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 4, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        if (half_out)
            hc_mix_finalize_kernel<4, 4, true, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
        else
            hc_mix_finalize_kernel<4, 4, true, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
    }
    #undef ARGS_A
    #undef ARGS_C
    cuda_check(cudaPeekAtLastError());
}

int hc_mix_num_chunks(int R, int row_len)
{
    if (row_len <= 0) return 0;
    int chunks_a = std::min(128, std::max(1, 512 / std::max(R, 1)));
    int chunk_cols = ((row_len / chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    return (row_len + chunk_cols - 1) / chunk_cols;
}

void hc_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (2H + H^2, H * D) float
    const at::Tensor& base,              // (2H + H^2) float
    const at::Tensor& scale,             // (3) float
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                 // (R, chunks, M + 1) float workspace
    at::Tensor post,                     // (R, H) float out
    at::Tensor comb,                     // (R, H, H) float out
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        (int) sinkhorn_iters,
        partials,
        &post,
        &comb,
        collapsed,
        nullptr
    );
}

// hc_mix with the launch-count folds (hc_fuse.cuh): y / post_a / comb_a describe a pending hc_apply
// on these streams (all or none), run inside the partials kernel; norm_y receives the RMSNorm of the collapsed
// output, run inside the finalize (norm_w may be none for an unweighted norm). collapsed is always written
void hc_mix_fused
(
    at::Tensor x,                        // (R, 4, D) float, updated in place by the pending apply
    const c10::optional<at::Tensor>& y,
    const c10::optional<at::Tensor>& post_a,
    const c10::optional<at::Tensor>& comb_a,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor collapsed,
    const c10::optional<at::Tensor>& norm_w,
    const c10::optional<at::Tensor>& norm_y,
    double norm_eps,
    double norm_bias,
    double norm_scale
)
{
    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 4 && x.is_contiguous(), "hc_mix_fused: x must be (R, 4, D) contiguous");
    const int R = x.size(0);
    const int D = x.size(2);

    HcPendingApply pend {};
    const HcPendingApply* pend_p = nullptr;
    if (y.has_value())
    {
        TORCH_CHECK(post_a.has_value() && comb_a.has_value(), "hc_mix_fused: a pending apply needs y, post and comb");
        TORCH_CHECK_DTYPE(post_a.value(), kFloat);
        TORCH_CHECK_DTYPE(comb_a.value(), kFloat);
        TORCH_CHECK(y->is_contiguous() && post_a->is_contiguous() && comb_a->is_contiguous(), "hc_mix_fused: contiguous inputs required");
        TORCH_CHECK(y->size(0) == R && y->size(-1) == D && y->numel() == (int64_t) R * D &&
                    post_a->size(0) == R && post_a->numel() == (int64_t) R * 4 &&
                    comb_a->size(0) == R && comb_a->numel() == (int64_t) R * 16, "hc_mix_fused: shapes");
        const bool y_half = y->dtype() == at::kHalf;
        if (!y_half) TORCH_CHECK_DTYPE(y.value(), kFloat);
        pend = { y->data_ptr(), y_half, (const float*) post_a->data_ptr(), (const float*) comb_a->data_ptr() };
        pend_p = &pend;
    }

    HcNormArgs nrm {};
    const HcNormArgs* nrm_p = nullptr;
    if (norm_y.has_value())
    {
        TORCH_CHECK_DTYPE(norm_y.value(), kHalf);
        TORCH_CHECK(norm_y->is_contiguous() && norm_y->numel() == (int64_t) R * D, "hc_mix_fused: norm_y shape");
        bool w_bf16 = false;
        const void* w = nullptr;
        if (norm_w.has_value())
        {
            TORCH_CHECK(norm_w->numel() == D && norm_w->is_contiguous(), "hc_mix_fused: norm weight shape");
            w_bf16 = norm_w->dtype() == at::kBFloat16;
            if (!w_bf16) TORCH_CHECK_DTYPE(norm_w.value(), kHalf);
            w = norm_w->data_ptr();
        }
        nrm = { w, w_bf16, (half*) norm_y->data_ptr(), (float) norm_eps, (float) norm_bias, (float) norm_scale };
        nrm_p = &nrm;
    }

    hc_mix_launch(x, fn, base, scale, (float) rms_eps, (float) hc_eps, (int) sinkhorn_iters,
                  partials, &post, &comb, collapsed, nullptr, pend_p, nrm_p);
}

void hc_head
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (H, H * D) float
    const at::Tensor& base,              // (H) float
    const at::Tensor& scale,             // (1) float
    double rms_eps,
    double hc_eps,
    at::Tensor partials,                 // (R, chunks, H + 1) float workspace
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        0,
        partials,
        nullptr,
        nullptr,
        collapsed,
        nullptr
    );
}

void hc_apply
(
    at::Tensor x,                        // (R, H, D) float, updated IN PLACE
    const at::Tensor& y,                 // (R, D) float or half
    const at::Tensor& post,              // (R, H) float
    const c10::optional<at::Tensor>& comb,  // (R, H, H) float, or none (x[h] += post[h] * y)
    const c10::optional<at::Tensor>& wn,    // (H * D) half: next site's norm weight, with xw
    c10::optional<at::Tensor> xw         // (R, H, D) float out: updated x * wn, or none
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(post, kFloat);
    TORCH_CHECK_FLOAT_HALF(y);
    TORCH_CHECK_DIM(x, 3);
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && post.is_contiguous(), "hc_apply: contiguous inputs required");
    int R = x.size(0);
    int H = x.size(1);
    int D = x.size(2);
    TORCH_CHECK(H == 4, "hc_apply: H = 4 only");
    TORCH_CHECK(D % 4 == 0, "hc_apply: dims");
    TORCH_CHECK(y.size(0) == R && y.size(-1) == D && y.numel() == (int64_t) R * D, "hc_apply: y shape");
    TORCH_CHECK(post.size(0) == R && post.numel() == (int64_t) R * H, "hc_apply: gate shapes");
    const float* comb_p = nullptr;
    if (comb)
    {
        TORCH_CHECK_DTYPE(comb.value(), kFloat);
        TORCH_CHECK(comb.value().is_contiguous() && comb.value().size(0) == R && comb.value().numel() == (int64_t) R * H * H,
                    "hc_apply: comb shape");
        comb_p = (const float*) comb.value().data_ptr();
    }
    const half* wn_p = nullptr;
    float* xw_p = nullptr;
    if (xw)
    {
        TORCH_CHECK(wn, "hc_apply: xw needs wn");
        TORCH_CHECK_DTYPE(wn.value(), kHalf);
        TORCH_CHECK_DTYPE(xw.value(), kFloat);
        TORCH_CHECK(wn.value().is_contiguous() && wn.value().numel() == H * D, "hc_apply: wn shape");
        TORCH_CHECK(xw.value().is_contiguous() && xw.value().sizes() == x.sizes(), "hc_apply: xw shape");
        wn_p = (const half*) wn.value().data_ptr();
        xw_p = (float*) xw.value().data_ptr();
    }
    TORCH_CHECK(!(comb_p && xw_p), "hc_apply: xw is for the comb-less (GatedResidual) form");
    if (!R || !D) return;

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols = ((D / chunks_c + 4 * NUM_THREADS - 1)
                      / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;

    dim3 grid(n_chunks, R);
    #define ARGS(Y_T) \
        (float*) x.data_ptr(), (const Y_T*) y.data_ptr(), \
        (const float*) post.data_ptr(), comb_p, wn_p, xw_p, D, chunk_cols
    #define LAUNCH(Y_T) \
        if (comb_p)      hc_apply_kernel<4, Y_T, true, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(Y_T)); \
        else if (xw_p)   hc_apply_kernel<4, Y_T, false, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(Y_T)); \
        else           { hc_apply_kernel<4, Y_T, false, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(Y_T)); }
    if (y.dtype() == at::kHalf) { LAUNCH(half) }
    else                        { LAUNCH(float) }
    #undef LAUNCH
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

#if defined(USE_ROCM)
    #include "rocm/gr_dots_rdna.cuh"
#endif

void gr_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const c10::optional<at::Tensor>& wstreams,   // (R, H, D) float: streams * w (from hc_apply), or none
    const at::Tensor& fn,                // (M, H * D) half: cat(down, inject)
    const at::Tensor& upt,               // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half or float out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn, kHalf);
    TORCH_CHECK_DTYPE(upt, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && upt.is_contiguous() &&
                w.is_contiguous() && dots.is_contiguous(), "gr_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int M = fn.size(0);
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix: upt must be the (H, D / 4, LR, 4) repacked layout");
    int LR = upt.size(2);
    TORCH_CHECK(H == 4, "gr_mix: H = 4 only");
    TORCH_CHECK(D % 8 == 0, "gr_mix: dims");
    TORCH_CHECK(M == LR + (post ? H : 0), "gr_mix: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == H * D && w.numel() == H * D, "gr_mix: dims");
    TORCH_CHECK(dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H, "gr_mix: dots shape");
    const float* ws_p = nullptr;
    if (wstreams)
    {
        TORCH_CHECK_DTYPE(wstreams.value(), kFloat);
        TORCH_CHECK(wstreams.value().is_contiguous() && wstreams.value().sizes() == streams.sizes(), "gr_mix: wstreams shape");
        ws_p = (const float*) wstreams.value().data_ptr();
    }
    TORCH_CHECK(D > 0, "gr_mix: norm over an empty stream dimension");
    if (!R) return;
    const half* w_p = (const half*) w.data_ptr();

    const int nch = D / 256, nit = (LR + 63) / 64;
#if defined(USE_ROCM)
    // RDNA: rows dots (rocm/gr_dots_rdna.cuh) and the generic finalize below; the bytes-in-flight pair runs far
    // off the bandwidth bound on these parts
    const bool fast = false;
    (void) nch; (void) nit;
#else
    const bool fast = R <= GR_MAX_R && D % 256 == 0 && (nch == 10 || nch == 16 || nch == 20) && (nit == 5 || nit == 8 || nit == 16) && LR % 2 == 0;
#endif
    if (fast)
    {
        const float* s_p = (const float*) streams.data_ptr(); const half* fn_p = (const half*) fn.data_ptr(); float* dots_p = (float*) dots.data_ptr();
        const int grid_a2 = (M + GR_J - 1) / GR_J + 1;
        #define DOTS2(NCH, WTD) gr_dots_kernel2<4, NCH, WTD><<<grid_a2, GR_THREADS_A2, 0, stream>>>(s_p, ws_p, fn_p, w_p, dots_p, M, D, R)
        if (ws_p) switch (nch)
        {
            case 10: DOTS2(10, true); break;
            case 16: DOTS2(16, true); break;
            case 20: DOTS2(20, true); break;
        }
        else switch (nch)
        {
            case 10: DOTS2(10, false); break;
            case 16: DOTS2(16, false); break;
            case 20: DOTS2(20, false); break;
        }
        #undef DOTS2
        cuda_check(cudaPeekAtLastError());
        const int grid_c2 = D / 4 / (GR_THREADS_C / 32);
        const int smem2 = (R * LR + R * 4) * sizeof(float);
        float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
        #define ARGS2 \
            (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
            post_p, mixed.data_ptr(), D, LR, R, (float) rms_eps
        const bool hout = mixed.dtype() == at::kHalf;
        switch (nit)
        {
            case 5:  if (hout) gr_finalize_kernel2<4, true, 5><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else { gr_finalize_kernel2<4, false, 5><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); } break;
            case 8:  if (hout) gr_finalize_kernel2<4, true, 8><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else { gr_finalize_kernel2<4, false, 8><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); } break;
            case 16: if (hout) gr_finalize_kernel2<4, true, 16><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); else { gr_finalize_kernel2<4, false, 16><<<grid_c2, GR_THREADS_C, smem2, stream>>>(ARGS2); } break;
        }
        #undef ARGS2
        cuda_check(cudaPeekAtLastError());
        return;
    }

    dim3 grid_a(M + 1, R);
    #define DOTS_ARGS (const float*) streams.data_ptr(), ws_p, (const half*) fn.data_ptr(), w_p, (float*) dots.data_ptr(), M, D
#if defined(USE_ROCM)
    if (gr_dots_rows_try((const float*) streams.data_ptr(), ws_p, (const half*) fn.data_ptr(), w_p, (float*) dots.data_ptr(), M, D, R, stream)) {}
    else
#endif
    if (ws_p) gr_dots_kernel<4, true><<<grid_a, GR_THREADS_A, 0, stream>>>(DOTS_ARGS);
    else    { gr_dots_kernel<4, false><<<grid_a, GR_THREADS_A, 0, stream>>>(DOTS_ARGS); }
    #undef DOTS_ARGS
    cuda_check(cudaPeekAtLastError());

    // Phase C is warp-per-column-quad: chunk at warp granularity (4 * NUM_THREADS / 32
    // columns) so small R still fills the device
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    if (mixed.dtype() == at::kHalf)
        gr_finalize_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    else
        gr_finalize_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}
