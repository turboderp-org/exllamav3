#pragma once

// RDNA router GEMV for m = 1..8 rows, taken by routing_gemv (routing.cu) on ROCm. Same arithmetic as
// routing_gemv_kernel (lane l owns the half2 columns j = l mod 32, accumulated in increasing j with the same
// two fmaf per column, then the same shfl_down tree), so the scores are bit-identical to it; what changes is
// the scheduling: several column loads in flight per lane, two-wave blocks spread over every CU, and, at
// m == 1, 16-byte weight loads staged through LDS. Rows m = 2..8 (speculative verify steps) read the weights
// once for all rows.

#include <cuda_fp16.h>
#include "../util.cuh"

#define RG_WARPS 2
#define RG_U 16

template <int M>
__global__ __launch_bounds__(RG_WARPS * 32)
void routing_gemv_rdna_kernel
(
    const half* __restrict__ x,         // (m, k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (m, E)
    const int k,
    const int E,
    const int m
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * RG_WARPS + warp;
    if (row >= E) return;

    const int k2 = k / 2;
    const half2* x2 = (const half2*) x;
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);

    float sum[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) sum[r] = 0.0f;

    int j0 = lane;
    for (; j0 + 32 * (RG_U - 1) < k2; j0 += 32 * RG_U)
    {
        half2 w[RG_U];
        #pragma unroll
        for (int u = 0; u < RG_U; ++u) w[u] = w2[j0 + 32 * u];
        #pragma unroll
        for (int u = 0; u < RG_U; ++u)
        {
            float2 wf = __half22float2(w[u]);
            #pragma unroll
            for (int r = 0; r < M; ++r)
            {
                if (r >= m) break;
                float2 xf = __half22float2(x2[(size_t) r * k2 + j0 + 32 * u]);
                sum[r] = fmaf(xf.x, wf.x, sum[r]);
                sum[r] = fmaf(xf.y, wf.y, sum[r]);
            }
        }
    }
    for (int j = j0; j < k2; j += 32)
    {
        float2 wf = __half22float2(w2[j]);
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            if (r >= m) break;
            float2 xf = __half22float2(x2[(size_t) r * k2 + j]);
            sum[r] = fmaf(xf.x, wf.x, sum[r]);
            sum[r] = fmaf(xf.y, wf.y, sum[r]);
        }
    }

    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        if (r >= m) break;
        float s = sum[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, offset);
        if (lane == 0)
            scores[(size_t) r * E + row] = __float2half_rn(s);
    }
}

// m == 1 form with 16-byte weight loads. The kernel above issues one dword load per lane per column step,
// which leaves it bound by the load-instruction rate rather than by bandwidth. Here each lane loads 16 bytes
// (4 adjacent half2 columns) of a 128-column block, the wave writes the block to its own LDS slice, and each
// lane reads back its own columns (l, l + 32, l + 64, l + 96), so the chain is still routing_gemv_kernel's,
// column for column, and the result bit-identical. x is staged once per block in LDS. Whole 128-column blocks go
// through LDS and the tail (k / 2 % 128 columns) is read per lane directly. Needs k % 8 == 0, 16-byte aligned rows and
// the LDS budget (x plus U blocks per wave), all checked by the host; otherwise the dword-load kernel runs.
#define RGW_WARPS 4

// Stages x into LDS (every thread of the block must enter), then returns this
// warp's row dot product in lane 0 (the shfl_down tree). Rows >= E return 0.
template <int U>
__device__ __forceinline__ float routing_gemv_rdna_w16_row
(
    const half* __restrict__ x,
    const half* __restrict__ gate_t,
    const int k,
    const int row,
    const bool valid,
    uint32_t* rg_sh
)
{
    const int k2 = k / 2;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    uint32_t* xs = rg_sh;
    uint32_t* wt = rg_sh + k2 + warp * U * 128;

    for (int i = threadIdx.x; i < k2 / 4; i += RGW_WARPS * 32)
        ((uint4*) xs)[i] = ((const uint4*) x)[i];
    __syncthreads();
    if (!valid) return 0.0f;

    const uint4* w4 = (const uint4*) (gate_t + (size_t) row * k);
    const half2* xs2 = (const half2*) xs;
    const half2* wt2 = (const half2*) wt;
    const int nb = k2 / 128;
    float sum = 0.0f;

    int b0 = 0;
    for (; b0 + U <= nb; b0 += U)
    {
        uint4 v[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) v[u] = w4[(b0 + u) * 32 + lane];
        #pragma unroll
        for (int u = 0; u < U; ++u) ((uint4*) (wt + u * 128))[lane] = v[u];
        __syncwarp();
        #pragma unroll
        for (int u = 0; u < U; ++u)
        {
            #pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                float2 wf = __half22float2(wt2[u * 128 + q * 32 + lane]);
                float2 xf = __half22float2(xs2[(b0 + u) * 128 + q * 32 + lane]);
                sum = fmaf(xf.x, wf.x, sum);
                sum = fmaf(xf.y, wf.y, sum);
            }
        }
        __syncwarp();
    }
    for (; b0 < nb; ++b0)
    {
        ((uint4*) wt)[lane] = w4[b0 * 32 + lane];
        __syncwarp();
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float2 wf = __half22float2(wt2[q * 32 + lane]);
            float2 xf = __half22float2(xs2[b0 * 128 + q * 32 + lane]);
            sum = fmaf(xf.x, wf.x, sum);
            sum = fmaf(xf.y, wf.y, sum);
        }
        __syncwarp();
    }
    const half2* w2 = (const half2*) (gate_t + (size_t) row * k);
    for (int j = nb * 128 + lane; j < k2; j += 32)
    {
        float2 wf = __half22float2(w2[j]);
        float2 xf = __half22float2(xs2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    return sum;
}

template <int U>
__global__ __launch_bounds__(RGW_WARPS * 32)
void routing_gemv_rdna_w16_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const int k,
    const int E
)
{
    extern __shared__ uint32_t rg_sh[];
    const int row = blockIdx.x * RGW_WARPS + threadIdx.x / 32;
    float sum = routing_gemv_rdna_w16_row<U>(x, gate_t, k, row, row < E, rg_sh);
    if (row < E && threadIdx.x % 32 == 0)
        scores[row] = __float2half_rn(sum);
}

#define RGW_U 8

// Launches the router GEMV for hidden (m, k) against gate_t (E, k) into scores (m, E); false when the call is
// outside its bounds (nothing launched)
static inline bool routing_gemv_rdna_try(const at::Tensor& hidden, const at::Tensor& gate_t, at::Tensor& scores, cudaStream_t stream)
{
    const int k = (int) hidden.size(-1);
    const int E = (int) scores.size(-1);
    if ((k & 1) || !hidden.is_contiguous() || !gate_t.is_contiguous()) return false;
    const int m = (int) (hidden.numel() / k);
    if (m < 1 || m > 8 || scores.numel() != (int64_t) m * E) return false;

    const half* xp = (const half*) hidden.data_ptr();
    const half* wp = (const half*) gate_t.data_ptr();
    half* sp = (half*) scores.data_ptr();
    const size_t smem_w = (size_t) (k / 2 + RGW_WARPS * RGW_U * 128) * 4;
    if (m == 1 && !(k & 7) && !(((uintptr_t) xp | (uintptr_t) wp) & 15) && smem_w <= 48 * 1024)
    {
        routing_gemv_rdna_w16_kernel<RGW_U><<<CEIL_DIVIDE(E, RGW_WARPS), RGW_WARPS * 32, smem_w, stream>>>(xp, wp, sp, k, E);
        return true;
    }
    const dim3 grid(CEIL_DIVIDE(E, RG_WARPS));
    const dim3 block(RG_WARPS * 32);
    if (m == 1)      routing_gemv_rdna_kernel<1><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
    else if (m <= 2) routing_gemv_rdna_kernel<2><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
    else if (m <= 4) routing_gemv_rdna_kernel<4><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
    else             routing_gemv_rdna_kernel<8><<<grid, block, 0, stream>>>(xp, wp, sp, k, E, m);
    return true;
}
