#pragma once

// GatedResidual decode dots for RDNA:
//
// gr_dots_kernel with RB fn rows per block instead of one. The thread -> column map (8-half column c = tid +
// GR_THREADS_A k), each per-stream fmaf chain, the warp tree and the 4-warp sum are gr_dots_kernel's, so every
// dot is bit-identical to it; what changes is that a block loads its stream values once into registers (weighted
// there when no pre-weighted copy exists) and then streams RB fn rows with all of a row's loads in flight, where
// the one-row kernel re-reads the stream stack in every block. The sum-of-squares row runs in its own block.
// gr_dots_kernel2 (the bytes-in-flight form) runs far off the bandwidth bound on these parts; both finalize
// kernels read the dots in the same layout.

template <int H, int RB, int NI, bool WEIGHTED>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_rows_kernel
(
    const float* __restrict__ streams,   // (R, H, D) raw (sum of squares)
    const float* __restrict__ wstreams,  // (R, H, D) streams * w, or null (WEIGHTED = false)
    const half* __restrict__ fn,         // (M, H * D) half projection
    const half* __restrict__ wn,         // (H * D) half norm weight, applied here when not WEIGHTED
    float* __restrict__ dots,            // (R, M + 1, H)
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int D4 = D / 4;
    const int D8 = D / 8;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    const float4* d4 = (const float4*) ((WEIGHTED ? wstreams : streams) + (size_t) r * H * D);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    __shared__ float red[RB][H][GR_THREADS_A / 32];

    const int nrb = (M + RB - 1) / RB;
    if (blockIdx.x == nrb)
    {
        // Sum-of-squares row: gr_dots_kernel's j == M branch
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
            a = hc_warp_sum_lane0(a);
            if (lane == 0) red[0][h][warp] = a;
        }
        __syncthreads();
        if (threadIdx.x < H)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_THREADS_A / 32; ++w)
                v += red[0][threadIdx.x][w];
            dots[((size_t) r * (M + 1) + M) * H + threadIdx.x] = v;
        }
        return;
    }

    const int j0 = blockIdx.x * RB;

    // This thread's (weighted) stream values, every stream: 8 columns at c = tid + GR_THREADS_A k
    float4 sa[H][NI], sb[H][NI];
    #pragma unroll
    for (int h = 0; h < H; ++h)
        #pragma unroll
        for (int k = 0; k < NI; ++k)
        {
            const int c = threadIdx.x + k * GR_THREADS_A;
            if (c < D8)
            {
                float4 s0 = d4[(size_t) h * D4 + 2 * c];
                float4 s1 = d4[(size_t) h * D4 + 2 * c + 1];
                if constexpr (!WEIGHTED)
                {
                    int4 wk = ((const int4*) (wn + (size_t) h * D))[c];
                    half4 v0 = *(half4*) &wk.x;
                    half4 v1 = *(half4*) &wk.z;
                    s0.x = __fmul_rn(s0.x, LOW_TO_FLOAT(v0.x)); s0.y = __fmul_rn(s0.y, HIGH_TO_FLOAT(v0.x));
                    s0.z = __fmul_rn(s0.z, LOW_TO_FLOAT(v0.y)); s0.w = __fmul_rn(s0.w, HIGH_TO_FLOAT(v0.y));
                    s1.x = __fmul_rn(s1.x, LOW_TO_FLOAT(v1.x)); s1.y = __fmul_rn(s1.y, HIGH_TO_FLOAT(v1.x));
                    s1.z = __fmul_rn(s1.z, LOW_TO_FLOAT(v1.y)); s1.w = __fmul_rn(s1.w, HIGH_TO_FLOAT(v1.y));
                }
                sa[h][k] = s0;
                sb[h][k] = s1;
            }
        }

    #pragma unroll
    for (int rr = 0; rr < RB; ++rr)
    {
        const int j = j0 + rr;
        if (j >= M) break;                                   // block-uniform
        int4 f[H][NI];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                const int c = threadIdx.x + k * GR_THREADS_A;
                if (c < D8) f[h][k] = ((const int4*) (fn + ((size_t) j * H + h) * D))[c];
            }
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float a = 0.0f;
            #pragma unroll
            for (int k = 0; k < NI; ++k)
            {
                const int c = threadIdx.x + k * GR_THREADS_A;
                if (c < D8)
                {
                    const float4 s0 = sa[h][k];
                    const float4 s1 = sb[h][k];
                    half4 w0 = *(half4*) &f[h][k].x;
                    half4 w1 = *(half4*) &f[h][k].z;
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
            a = hc_warp_sum_lane0(a);
            if (lane == 0) red[rr][h][warp] = a;
        }
    }
    __syncthreads();
    if (threadIdx.x < RB * H)
    {
        const int rr = threadIdx.x / H;
        const int h = threadIdx.x % H;
        const int j = j0 + rr;
        if (j < M)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < GR_THREADS_A / 32; ++w)
                v += red[rr][h][w];
            dots[((size_t) r * (M + 1) + j) * H + h] = v;
        }
    }
}

// Launches the rows dots for D up to 3 * 8 * GR_THREADS_A columns; false otherwise (nothing launched)
static bool gr_dots_rows_try
(
    const float* s_p,
    const float* ws_p,
    const half* fn_p,
    const half* w_p,
    float* dots_p,
    const int M,
    const int D,
    const int R,
    cudaStream_t stream
)
{
    constexpr int RB = 4;
    const int ni = (D / 8 + GR_THREADS_A - 1) / GR_THREADS_A;
    if (ni < 1 || ni > 3 || D % 8) return false;
    dim3 grid((M + RB - 1) / RB + 1, R);
    #define ROWS(NI, WTD) gr_dots_rows_kernel<4, RB, NI, WTD><<<grid, GR_THREADS_A, 0, stream>>>(s_p, ws_p, fn_p, w_p, dots_p, M, D)
    if (ws_p) { if (ni == 1) ROWS(1, true); else if (ni == 2) ROWS(2, true); else ROWS(3, true); }
    else      { if (ni == 1) ROWS(1, false); else if (ni == 2) ROWS(2, false); else ROWS(3, false); }
    #undef ROWS
    cuda_check(cudaPeekAtLastError());
    return true;
}
