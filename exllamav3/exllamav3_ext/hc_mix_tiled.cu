#include <cuda_fp16.h>
#include "hc_mix.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "det_gemm.cuh"

/*

Tiled, deterministic GatedResidual mix for prefill-sized row counts (hyperconnections.py
GatedResidual._mix, R > FUSED_MAX_R). Same math as gr_mix:

  rmr[r, h]     = 1 / sqrt(mean_d s[r, h, d]^2 + eps)
  normed[r, hd] = s[r, hd] * rmr[r, h] * w[hd]
  dm[r, j]      = sum_hd normed[r, hd] * proj[j, hd]      j < LR: low-rank latent, j >= LR: post
  t[r, i]       = silu(dm[r, i] / H)                       i < LR
  post[r, h']   = 2 sigmoid(dm[r, LR + h'] / H)
  g[r, hd]      = sum_i t[r, i] * up[hd, i]
  mixed[r, d]   = half( sum_h sigmoid(g[r, hd]) * normed[r, hd] / H )

The two GEMMs run the int8 Ozaki-style scheme of det_gemm.cuh, so the result is bit-identical
on every GPU architecture (fp16 tensor cores are not: their inner accumulation differs between
Blackwell and earlier parts, and cuBLAS additionally picks device-dependent kernels), which is
what lets tensor-parallel ranks replicate decisions such as MoE routing. Everything else is
fixed-order fp32 with explicit non-contracting intrinsics and FMA-only transcendentals.

Three launches:
  1. gr_dots_i8    dm_part[slice] = normed_q @ proj_q^T over one K slice of one stream, plus the
                   slice's sum of squares. A block owns 64 rows x 128 proj columns (16 warps of
                   32 x 16); the fp32 stream chunk is loaded straight into registers, scaled by
                   the norm weight, quantized per (row, 128-wide chunk) and staged as int8 hi/lo
                   while the proj slices arrive by cp.async (2-stage ring). Column blocks of the
                   same rows are adjacent in the grid so the stream read stays one HBM pass.
  2. gr_latent_i8  per row: rmr from the slice sums, dm = sum over slices of rmr[h] * dm_part
                   (fixed order), t = silu(dm / H) quantized per 64-wide chunk for stage 3, post
  3. gr_gate_i8    g = t_q @ up_q^T for a (64 rows x [H x 32 d]) tile, K = LR; the epilogue
                   reduces the H streams in fixed order against the normed operand -> mixed
Requirements: H == 4, D % 128 == 0, LR % 64 == 0, proj padded to Mpad = roundup(M, 64) <= 512
rows (M = LR (+ H)); proj and up arrive pre-quantized per row ((2, N, K) int8 + (N) fp32 scales,
det_quant_weight). Workspaces are sized by the caller from gr_mix_tiled_slices().

*/

#define HC 4
#define DOTS_BM 64
#define DOTS_BN 128
#define DOTS_KCH 128
#define DOTS_THREADS 512
#define DOTS_MAXN 512
#define DOTS_A_BYTES (2 * DOTS_BM * DOTS_KCH)
#define DOTS_B_BYTES (2 * DOTS_BN * DOTS_KCH)
#define DOTS_STAGE_BYTES (DOTS_A_BYTES + DOTS_B_BYTES)
#define DOTS_SMEM (2 * DOTS_STAGE_BYTES)
#define GATE_BM 64
#define GATE_TD 32
#define GATE_KCH 64
#define GATE_THREADS 256
#define GATE_A_BYTES (2 * GATE_BM * GATE_KCH)
#define GATE_B_BYTES (2 * HC * GATE_TD * GATE_KCH)
#define GATE_STAGE_BYTES (GATE_A_BYTES + GATE_B_BYTES)
#define GATE_SMEM (2 * GATE_STAGE_BYTES)

// ---------------------------------------------------------------------------------------------
// 1. dots

__global__ __launch_bounds__(DOTS_THREADS)
void gr_dots_i8
(
    const float* __restrict__ s,        // (R, HC * D)
    const half* __restrict__ w,         // (HC * D)
    const signed char* __restrict__ phi,// (Mpad, HC * D) proj hi
    const signed char* __restrict__ plo,// (Mpad, HC * D) proj lo
    const float* __restrict__ psb,      // (Mpad) proj row scales
    float* __restrict__ dm_part,        // (S, Rpad, Mpad)
    float* __restrict__ ss_part,        // (S, Rpad)
    const int R, const int D, const int Mpad, const int kslice
)
{
    extern __shared__ __align__(128) unsigned char dsm[];
    __shared__ float sa_s[2][DOTS_BM];
    const int HD = HC * D;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int wm = warp / 8, wn = warp % 8;
    const int g = lane / 4, tg = lane % 4;
    const int n0 = blockIdx.x * DOTS_BN, r0 = blockIdx.y * DOTS_BM;
    const int slice = blockIdx.z;
    const int k_beg = slice * kslice, k_end = k_beg + kslice;
    const int n_stages = kslice / DOTS_KCH;
    const size_t Rpad = gridDim.y * DOTS_BM;

    auto st_ahi = [&](int st) { return dsm + st * DOTS_STAGE_BYTES; };
    auto st_alo = [&](int st) { return dsm + st * DOTS_STAGE_BYTES + DOTS_BM * DOTS_KCH; };
    auto st_bhi = [&](int st) { return dsm + st * DOTS_STAGE_BYTES + DOTS_A_BYTES; };
    auto st_blo = [&](int st) { return dsm + st * DOTS_STAGE_BYTES + DOTS_A_BYTES + DOTS_BN * DOTS_KCH; };

    // A: row t / 8, 16 consecutive k at (t % 8) * 16 of each 128-wide stage
    const int a_row = t / 8, a_c16 = (t % 8) * 16, a_gr = r0 + a_row;
    float ss = 0.0f;
    float av[16];
    auto fetch_a = [&](int k0)
    {
        const int k = k0 + a_c16;
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            float4 v = make_float4(0, 0, 0, 0);
            if (a_gr < R) v = *(const float4*) (s + (size_t) a_gr * HD + k + q * 4);
            av[q * 4] = v.x; av[q * 4 + 1] = v.y; av[q * 4 + 2] = v.z; av[q * 4 + 3] = v.w;
        }
    };
    auto stage_a = [&](int k0, int st)
    {
        const int k = k0 + a_c16;
        float v[16];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            const half2* wq = (const half2*) (w + k + q * 4);
            float2 w0 = __half22float2(wq[0]), w1 = __half22float2(wq[1]);
            ss = __fmaf_rn(av[q * 4], av[q * 4], ss); ss = __fmaf_rn(av[q * 4 + 1], av[q * 4 + 1], ss);
            ss = __fmaf_rn(av[q * 4 + 2], av[q * 4 + 2], ss); ss = __fmaf_rn(av[q * 4 + 3], av[q * 4 + 3], ss);
            v[q * 4] = __fmul_rn(av[q * 4], w0.x); v[q * 4 + 1] = __fmul_rn(av[q * 4 + 1], w0.y);
            v[q * 4 + 2] = __fmul_rn(av[q * 4 + 2], w1.x); v[q * 4 + 3] = __fmul_rn(av[q * 4 + 3], w1.y);
        }
        float m = 0.0f;
        #pragma unroll
        for (int q = 0; q < 16; ++q) m = fmaxf(m, fabsf(v[q]));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
        const float scale = m > 0.0f ? m : 1.0f;
        int4 hi4, lo4;
        det_quant16(v, __fdiv_rn(DET_QMAX, scale), hi4, lo4);
        *(int4*) (st_ahi(st) + det_swz8(a_row, t % 8)) = hi4;
        *(int4*) (st_alo(st) + det_swz8(a_row, t % 8)) = lo4;
        if ((t % 8) == 0) sa_s[st][a_row] = __fdiv_rn(scale, DET_QMAX);
    };
    // B: 128 proj rows x 128 k int8 per slice = 1024 pieces per slice, two per thread
    auto issue_b = [&](int k0, int st)
    {
        #pragma unroll
        for (int q = 0; q < 2; ++q)
        {
            const int i = t + q * DOTS_THREADS, row = i / 8, cc = i % 8;
            const int gn = n0 + row, k = k0 + cc * 16;
            const int bytes = (gn < Mpad) ? 16 : 0;
            const size_t off = bytes ? (size_t) gn * HD + k : 0;
            det_cp_async16(det_smem_u32(st_bhi(st) + det_swz8(row, cc)), phi + off, bytes);
            det_cp_async16(det_smem_u32(st_blo(st) + det_swz8(row, cc)), plo + off, bytes);
        }
    };

    float facc[2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) facc[i][j][q] = 0.0f;

    issue_b(k_beg, 0); det_cp_async_commit();
    fetch_a(k_beg); stage_a(k_beg, 0);
    for (int stg = 0; stg < n_stages; ++stg)
    {
        const int k0 = k_beg + stg * DOTS_KCH;
        const bool more = stg + 1 < n_stages;
        det_cp_async_wait<0>();
        __syncthreads();                     // this stage complete for everyone; stage - 1 consumed
        if (more) { issue_b(k0 + DOTS_KCH, (stg + 1) & 1); fetch_a(k0 + DOTS_KCH); }
        det_cp_async_commit();
        const int st = stg & 1;
        const unsigned a_hi = det_smem_u32(st_ahi(st)), a_lo = det_smem_u32(st_alo(st));
        const unsigned b_hi = det_smem_u32(st_bhi(st)), b_lo = det_smem_u32(st_blo(st));

        int acc_hh[2][2][4], acc_x[2][2][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) { acc_hh[i][j][q] = 0; acc_x[i][j][q] = 0; }
        #pragma unroll
        for (int ks = 0; ks < DOTS_KCH; ks += 32)
        {
            unsigned ah[2][4], al[2][4], bh[2][2], bl[2][2], r4[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                det_load_a<128>(ah[i], a_hi, wm * 32 + i * 16, ks / 16, lane);
                det_load_a<128>(al[i], a_lo, wm * 32 + i * 16, ks / 16, lane);
            }
            det_load_b2<128>(r4, b_hi, wn * 16, ks / 16, lane);
            bh[0][0] = r4[0]; bh[0][1] = r4[1]; bh[1][0] = r4[2]; bh[1][1] = r4[3];
            det_load_b2<128>(r4, b_lo, wn * 16, ks / 16, lane);
            bl[0][0] = r4[0]; bl[0][1] = r4[1]; bl[1][0] = r4[2]; bl[1][1] = r4[3];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    det_mma3(acc_hh[i][j], acc_x[i][j], ah[i], al[i], bh[j], bl[j]);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const float s0 = sa_s[st][wm * 32 + i * 16 + g], s1 = sa_s[st][wm * 32 + i * 16 + g + 8];
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    facc[i][j][q] = det_flush(acc_hh[i][j][q], acc_x[i][j][q], q >= 2 ? s1 : s0, facc[i][j][q]);
        }
        // The other buffer's A tiles were consumed in stage - 1 (everyone passed this stage's
        // barrier): quantize the prefetched next chunk into them
        if (more) stage_a(k0 + DOTS_KCH, st ^ 1);
    }
    det_cp_async_wait<0>();
    __syncthreads();
    // Epilogue: apply proj row scales; stage through smem (aliases the ring) to coalesce
    float (*cs_)[DOTS_BN + 4] = (float (*)[DOTS_BN + 4]) dsm;
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                cs_[wm * 32 + i * 16 + g + (q >= 2 ? 8 : 0)][wn * 16 + j * 8 + tg * 2 + (q & 1)] = facc[i][j][q];
    __syncthreads();
    float* out = dm_part + (size_t) slice * Rpad * Mpad;
    #pragma unroll
    for (int q = 0; q < (DOTS_BM * DOTS_BN) / DOTS_THREADS; ++q)
    {
        const int i = (t + q * DOTS_THREADS) / DOTS_BN, j = (t + q * DOTS_THREADS) % DOTS_BN;
        if (n0 + j < Mpad) out[(size_t) (r0 + i) * Mpad + n0 + j] = __fmul_rn(cs_[i][j], psb[n0 + j]);
    }
    // Row sum of squares over this slice: the 8 consecutive lanes of a row, fixed-order tree
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 1));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 2));
    ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, 4));
    if ((t % 8) == 0 && blockIdx.x == 0) ss_part[(size_t) slice * Rpad + r0 + a_row] = ss;
}

// ---------------------------------------------------------------------------------------------
// 2. latent: one block per row, Mpad threads

__global__ __launch_bounds__(DOTS_MAXN)
void gr_latent_i8
(
    const float* __restrict__ dm_part,  // (S, Rpad, Mpad)
    const float* __restrict__ ss_part,  // (S, Rpad)
    float* __restrict__ rmr,            // (R, HC)
    signed char* __restrict__ thi,      // (R, LR)
    signed char* __restrict__ tlo,      // (R, LR)
    float* __restrict__ ts,             // (R, LR / 64)
    float* __restrict__ post,           // (R, HC) or nullptr
    const int R, const int D, const int LR, const int M, const int Mpad, const int S, const int Rpad, const float eps
)
{
    __shared__ float rmr_s[HC];
    __shared__ float wmax[DOTS_MAXN / 32];
    const int r = blockIdx.x, j = threadIdx.x;
    const int q = S / HC;               // slices per stream
    if (j < HC)
    {
        float a = 0.0f;
        for (int sl = j * q; sl < (j + 1) * q; ++sl) a = __fadd_rn(a, ss_part[(size_t) sl * Rpad + r]);
        float v = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(a, (float) D), eps)));
        rmr_s[j] = v;
        rmr[r * HC + j] = v;
    }
    __syncthreads();
    float v = 0.0f;
    if (j < M)
    {
        for (int sl = 0; sl < S; ++sl) v = __fmaf_rn(rmr_s[sl / q], dm_part[((size_t) sl * Rpad + r) * Mpad + j], v);
        v = __fmul_rn(v, 1.0f / (float) HC);
    }
    if (j >= LR)
    {
        if (post && j < M) post[r * HC + (j - LR)] = __fmul_rn(2.0f, sigmoid_det(v));
    }
    const float tv = j < LR ? silu_det(v) : 0.0f;
    // 64-wide chunk max: two warps per chunk
    float m = fabsf(tv);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((j & 31) == 0) wmax[j / 32] = m;
    __syncthreads();
    if (j < LR)
    {
        const int chunk = j / 64;
        const float cm = fmaxf(wmax[chunk * 2], wmax[chunk * 2 + 1]);
        const float scale = cm > 0.0f ? cm : 1.0f;
        signed char hi, lo;
        det_quant_split(tv, __fdiv_rn(DET_QMAX, scale), hi, lo);
        thi[(size_t) r * LR + j] = hi;
        tlo[(size_t) r * LR + j] = lo;
        if ((j & 63) == 0) ts[(size_t) r * (LR / 64) + chunk] = __fdiv_rn(scale, DET_QMAX);
    }
}

// ---------------------------------------------------------------------------------------------
// 3. gate: block = 64 rows x [HC x 32 d], 8 warps (wm, h) with 32 x 32 warp tiles, K = LR

__global__ __launch_bounds__(GATE_THREADS)
void gr_gate_i8
(
    const float* __restrict__ s,        // (R, HC * D)
    const float* __restrict__ rmr,      // (R, HC)
    const half* __restrict__ w,         // (HC * D)
    const signed char* __restrict__ thi,// (R, LR)
    const signed char* __restrict__ tlo,// (R, LR)
    const float* __restrict__ ts,       // (R, LR / 64)
    const signed char* __restrict__ uhi,// (HC * D, LR) up hi
    const signed char* __restrict__ ulo,// (HC * D, LR) up lo
    const float* __restrict__ usb,      // (HC * D) up row scales
    half* __restrict__ mixed,           // (R, D)
    const int R, const int D, const int LR
)
{
    extern __shared__ __align__(128) unsigned char dsm[];
    __shared__ float ts_s[2][GATE_BM];
    const int HD = HC * D;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int wm = warp / 4, h_w = warp % 4;
    const int g = lane / 4, tg = lane % 4;
    const int r0 = blockIdx.y * GATE_BM, d0 = blockIdx.x * GATE_TD;
    const int n_stages = LR / GATE_KCH;
    const float inv_h = 1.0f / (float) HC;

    auto st_ahi = [&](int st) { return dsm + st * GATE_STAGE_BYTES; };
    auto st_alo = [&](int st) { return dsm + st * GATE_STAGE_BYTES + GATE_BM * GATE_KCH; };
    auto st_bhi = [&](int st) { return dsm + st * GATE_STAGE_BYTES + GATE_A_BYTES; };
    auto st_blo = [&](int st) { return dsm + st * GATE_STAGE_BYTES + GATE_A_BYTES + HC * GATE_TD * GATE_KCH; };

    // A: 64 rows x 4 pieces = 256 -> one per thread per slice; B: 128 rows x 4 = 512 -> two
    auto issue = [&](int k0, int st)
    {
        {
            const int row = t / 4, cc = t % 4, gr = r0 + row;
            const int bytes = (gr < R) ? 16 : 0;
            const size_t off = bytes ? (size_t) gr * LR + k0 + cc * 16 : 0;
            det_cp_async16(det_smem_u32(st_ahi(st) + det_swz4(row, cc)), thi + off, bytes);
            det_cp_async16(det_smem_u32(st_alo(st) + det_swz4(row, cc)), tlo + off, bytes);
        }
        #pragma unroll
        for (int q = 0; q < 2; ++q)
        {
            const int i = t + q * GATE_THREADS, row = i / 4, cc = i % 4;
            const int h = row / GATE_TD, dd = row % GATE_TD;
            const size_t off = (size_t) (h * D + d0 + dd) * LR + k0 + cc * 16;
            det_cp_async16(det_smem_u32(st_bhi(st) + det_swz4(row, cc)), uhi + off, 16);
            det_cp_async16(det_smem_u32(st_blo(st) + det_swz4(row, cc)), ulo + off, 16);
        }
        if (t < GATE_BM)
        {
            const int gr = r0 + t;
            ts_s[st][t] = gr < R ? ts[(size_t) gr * (LR / GATE_KCH) + k0 / GATE_KCH] : 0.0f;
        }
    };

    float facc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) facc[i][j][q] = 0.0f;

    issue(0, 0); det_cp_async_commit();
    for (int stg = 0; stg < n_stages; ++stg)
    {
        det_cp_async_wait<0>();
        __syncthreads();
        if (stg + 1 < n_stages) issue((stg + 1) * GATE_KCH, (stg + 1) & 1);
        det_cp_async_commit();
        const int st = stg & 1;
        const unsigned a_hi = det_smem_u32(st_ahi(st)), a_lo = det_smem_u32(st_alo(st));
        const unsigned b_hi = det_smem_u32(st_bhi(st)), b_lo = det_smem_u32(st_blo(st));
        int acc_hh[2][4][4], acc_x[2][4][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) { acc_hh[i][j][q] = 0; acc_x[i][j][q] = 0; }
        #pragma unroll
        for (int ks = 0; ks < GATE_KCH; ks += 32)
        {
            unsigned ah[2][4], al[2][4], bh[4][2], bl[4][2], r4[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                det_load_a<64>(ah[i], a_hi, wm * 32 + i * 16, ks / 16, lane);
                det_load_a<64>(al[i], a_lo, wm * 32 + i * 16, ks / 16, lane);
            }
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj)
            {
                det_load_b2<64>(r4, b_hi, h_w * GATE_TD + jj * 16, ks / 16, lane);
                bh[jj * 2][0] = r4[0]; bh[jj * 2][1] = r4[1]; bh[jj * 2 + 1][0] = r4[2]; bh[jj * 2 + 1][1] = r4[3];
                det_load_b2<64>(r4, b_lo, h_w * GATE_TD + jj * 16, ks / 16, lane);
                bl[jj * 2][0] = r4[0]; bl[jj * 2][1] = r4[1]; bl[jj * 2 + 1][0] = r4[2]; bl[jj * 2 + 1][1] = r4[3];
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    det_mma3(acc_hh[i][j], acc_x[i][j], ah[i], al[i], bh[j], bl[j]);
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
        {
            const float s0 = ts_s[st][wm * 32 + i * 16 + g], s1 = ts_s[st][wm * 32 + i * 16 + g + 8];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    facc[i][j][q] = det_flush(acc_hh[i][j][q], acc_x[i][j][q], q >= 2 ? s1 : s0, facc[i][j][q]);
        }
    }
    det_cp_async_wait<0>();
    __syncthreads();
    // Gate tile (fp32, with the up row scales) to smem: Gs[h][64][36], aliases the ring
    float (*Gs)[GATE_BM][GATE_TD + 4] = (float (*)[GATE_BM][GATE_TD + 4]) dsm;
    static_assert(HC * GATE_BM * (GATE_TD + 4) * 4 <= GATE_SMEM, "gate staging must fit the ring");
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q)
            {
                const int row = wm * 32 + i * 16 + g + (q >= 2 ? 8 : 0);
                const int col = j * 8 + tg * 2 + (q & 1);          // d within the stream's 32
                Gs[h_w][row][col] = __fmul_rn(facc[i][j][q], usb[h_w * D + d0 + col]);
            }
    __syncthreads();
    // mixed[r, d] = sum_h sigmoid(g) * normed(r, h, d) / H: 64 x 32 outputs, 8 per thread
    {
        const int row = t / 4, c = (t % 4) * 8;
        const int gr = r0 + row;
        if (gr < R)
        {
            float o[8];
            #pragma unroll
            for (int q = 0; q < 8; ++q) o[q] = 0.0f;
            #pragma unroll
            for (int h = 0; h < HC; ++h)
            {
                const float sc = rmr[gr * HC + h];
                const float4* sp = (const float4*) (s + (size_t) gr * HD + h * D + d0 + c);
                const half2* wp = (const half2*) (w + h * D + d0 + c);
                float4 v0 = sp[0], v1 = sp[1];
                float2 w0 = __half22float2(wp[0]), w1 = __half22float2(wp[1]), w2 = __half22float2(wp[2]), w3 = __half22float2(wp[3]);
                const float sv[8] = { v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w };
                const float wv[8] = { w0.x, w0.y, w1.x, w1.y, w2.x, w2.y, w3.x, w3.y };
                #pragma unroll
                for (int q = 0; q < 8; ++q)
                {
                    // the normed operand rounded to half, as the decode-path kernels see it
                    const float nv = __half2float(__float2half_rn(__fmul_rn(__fmul_rn(sv[q], sc), wv[q])));
                    o[q] = __fmaf_rn(__fmul_rn(sigmoid_det(Gs[h][row][c + q]), inv_h), nv, o[q]);
                }
            }
            half2* outp = (half2*) (mixed + (size_t) gr * D + d0 + c);
            outp[0] = __floats2half2_rn(o[0], o[1]); outp[1] = __floats2half2_rn(o[2], o[3]);
            outp[2] = __floats2half2_rn(o[4], o[5]); outp[3] = __floats2half2_rn(o[6], o[7]);
        }
    }
}

// Slices per stream for the dots GEMM: a function of the row count only (device-independent),
// from a fixed ladder of divisors of D that keep slices a multiple of the 128-wide stage, sized
// so the grid has a few hundred blocks at small R
static int gr_dots_slices_per_stream(int R, int D, int Mpad)
{
    static const int ladder[] = { 1, 2, 4, 5, 8, 10, 16, 20 };
    const int rb = (R + DOTS_BM - 1) / DOTS_BM;
    const int nb = (Mpad + DOTS_BN - 1) / DOTS_BN;
    int q = 1;
    for (int cand : ladder)
    {
        if (D % cand != 0 || (D / cand) % DOTS_KCH != 0) continue;
        q = cand;
        if (rb * nb * HC * q >= 256) break;
    }
    return q;
}

int gr_mix_tiled_slices(int R, int D, int Mpad)
{
    return HC * gr_dots_slices_per_stream(R, D, Mpad);
}

void gr_mix_tiled
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    const at::Tensor& proj_i8,           // (2, Mpad, H * D) int8 hi / lo, rows >= M zero
    const at::Tensor& proj_sb,           // (Mpad) fp32
    const at::Tensor& up_i8,             // (2, H * D, LR) int8
    const at::Tensor& up_sb,             // (H * D) fp32
    double rms_eps,
    int M,                               // proj rows in use: LR (+ H with post)
    at::Tensor dm_part,                  // (S, Rpad, Mpad) float workspace
    at::Tensor ss_part,                  // (S, Rpad) float workspace
    at::Tensor rmr,                      // (R, H) float workspace
    at::Tensor t_i8,                     // (2, R, LR) int8 workspace
    at::Tensor t_s,                      // (R, LR / 64) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(proj_i8, kChar);
    TORCH_CHECK_DTYPE(up_i8, kChar);
    TORCH_CHECK_DTYPE(proj_sb, kFloat);
    TORCH_CHECK_DTYPE(up_sb, kFloat);
    TORCH_CHECK_DTYPE(dm_part, kFloat);
    TORCH_CHECK_DTYPE(ss_part, kFloat);
    TORCH_CHECK_DTYPE(rmr, kFloat);
    TORCH_CHECK_DTYPE(t_i8, kChar);
    TORCH_CHECK_DTYPE(t_s, kFloat);
    TORCH_CHECK_DTYPE(mixed, kHalf);
    TORCH_CHECK(streams.is_contiguous() && w.is_contiguous() && proj_i8.is_contiguous() && up_i8.is_contiguous() &&
                proj_sb.is_contiguous() && up_sb.is_contiguous() && dm_part.is_contiguous() && ss_part.is_contiguous() &&
                rmr.is_contiguous() && t_i8.is_contiguous() && t_s.is_contiguous() && mixed.is_contiguous(),
                "gr_mix_tiled: contiguous tensors required");
    const int R = streams.size(0);
    const int H = streams.size(1);
    const int D = streams.size(2);
    const int HD = H * D;
    const int Mpad = proj_i8.size(1);
    const int LR = up_i8.size(2);
    const int Rpad = (R + DOTS_BM - 1) / DOTS_BM * DOTS_BM;
    const int q = gr_dots_slices_per_stream(R, D, Mpad), S = HC * q;
    TORCH_CHECK(H == HC, "gr_mix_tiled: H = 4 only");
    TORCH_CHECK(D % DOTS_KCH == 0 && LR % GATE_KCH == 0, "gr_mix_tiled: D must be a multiple of 128 and LR of 64");
    TORCH_CHECK(proj_i8.size(0) == 2 && proj_i8.size(2) == HD && up_i8.size(0) == 2 && up_i8.size(1) == HD && w.numel() == HD, "gr_mix_tiled: dims");
    TORCH_CHECK(proj_sb.numel() == Mpad && up_sb.numel() == HD, "gr_mix_tiled: scales");
    TORCH_CHECK(M == LR + (post ? H : 0) && M <= Mpad, "gr_mix_tiled: proj rows must be LR (+ H with post)");
    TORCH_CHECK(Mpad % 64 == 0 && Mpad <= DOTS_MAXN, "gr_mix_tiled: proj rows must be padded to a multiple of 64, at most 512");
    TORCH_CHECK(dm_part.size(0) >= S && dm_part.size(1) >= Rpad && dm_part.size(2) == Mpad, "gr_mix_tiled: dm_part workspace");
    TORCH_CHECK(ss_part.size(0) >= S && ss_part.size(1) >= Rpad, "gr_mix_tiled: ss_part workspace");
    TORCH_CHECK(rmr.size(0) == R && rmr.size(1) == H, "gr_mix_tiled: rmr workspace");
    TORCH_CHECK(t_i8.size(0) == 2 && t_i8.size(1) == R && t_i8.size(2) == LR && t_s.size(0) == R && t_s.size(1) == LR / 64, "gr_mix_tiled: latent workspace");
    TORCH_CHECK(mixed.size(0) == R && mixed.size(1) == D, "gr_mix_tiled: mixed shape");
    if (post) { TORCH_CHECK_DTYPE(post.value(), kFloat); TORCH_CHECK(post.value().is_contiguous() && post.value().size(0) == R && post.value().size(1) == H, "gr_mix_tiled: post shape"); }

    int dev = 0;
    cudaGetDevice(&dev);
    static bool attr[32] = {};
    if (!attr[dev])
    {
        cudaFuncSetAttribute(gr_dots_i8, cudaFuncAttributeMaxDynamicSharedMemorySize, DOTS_SMEM);
        cudaFuncSetAttribute(gr_gate_i8, cudaFuncAttributeMaxDynamicSharedMemorySize, GATE_SMEM);
        attr[dev] = true;
    }

    const float* s_p = (const float*) streams.data_ptr();
    const half* w_p = (const half*) w.data_ptr();
    const signed char* phi = (const signed char*) proj_i8.data_ptr();
    const signed char* plo = phi + (size_t) Mpad * HD;
    const signed char* uhi = (const signed char*) up_i8.data_ptr();
    const signed char* ulo = uhi + (size_t) HD * LR;
    float* dm_p = (float*) dm_part.data_ptr();
    float* ss_p = (float*) ss_part.data_ptr();
    float* rmr_p = (float*) rmr.data_ptr();
    signed char* thi = (signed char*) t_i8.data_ptr();
    signed char* tlo = thi + (size_t) R * LR;
    float* ts_p = (float*) t_s.data_ptr();
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    half* mixed_p = (half*) mixed.data_ptr();

    dim3 grid_a((Mpad + DOTS_BN - 1) / DOTS_BN, Rpad / DOTS_BM, S);
    gr_dots_i8<<<grid_a, DOTS_THREADS, DOTS_SMEM, stream>>>(s_p, w_p, phi, plo, (const float*) proj_sb.data_ptr(), dm_p, ss_p, R, D, Mpad, D / q);
    cuda_check(cudaPeekAtLastError());

    gr_latent_i8<<<R, Mpad, 0, stream>>>(dm_p, ss_p, rmr_p, thi, tlo, ts_p, post_p, R, D, LR, M, Mpad, S, Rpad, (float) rms_eps);
    cuda_check(cudaPeekAtLastError());

    dim3 grid_c(D / GATE_TD, Rpad / GATE_BM);
    gr_gate_i8<<<grid_c, GATE_THREADS, GATE_SMEM, stream>>>(s_p, rmr_p, w_p, thi, tlo, ts_p, uhi, ulo, (const float*) up_sb.data_ptr(), mixed_p, R, D, LR);
    cuda_check(cudaPeekAtLastError());
}
