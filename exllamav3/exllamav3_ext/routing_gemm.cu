#include <cuda_fp16.h>
#include "routing.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "det_gemm.cuh"

/*

Deterministic router projection: scores (R, E) half = hidden (R, K) half @ gate^T with the int8
Ozaki-style scheme of det_gemm.cuh, so every tensor-parallel rank of ANY architecture that
routes on identical streams produces identical logits and top-k selections. cuBLAS picks
split-K kernels for this skinny shape (E of 64..512, K of thousands) with a device-dependent
split factor, and fp16 tensor cores accumulate differently per architecture.

Two launches: quant_a_kernel quantizes the activations per (row, 128-wide K chunk); the GEMM
runs a 128 x 64 block tile with 16 warps (4 x 4, 32 x 16 warp tiles, ~120 registers so all 16
warps fit an SM), a 2-stage cp.async ring of 128-wide chunks (97 KB), exact int32 sums per chunk
flushed to fp32 with the chunk scale, and a deterministic split-K (slice count from the shape
only) whose fp32 partials are summed in slice order by a third launch. The weights come
pre-quantized per row over all of K ((2, E, K) int8 hi/lo + (E) fp32 scales, built lazily on
the Python side). Rows >= R, columns >= E and K past the end are zero-filled; K % 16 == 0.

Decode-class calls (R <= 64) use per-device static workspaces (small buffers, no per-call
allocation on the host-bound decode path); larger calls allocate per call.

*/

#define RG_BM 128
#define RG_BN 64
#define RG_THREADS 512
#define RG_STAGES 2
#define RG_KCH 128
#define RG_A_BYTES (2 * RG_BM * RG_KCH)
#define RG_B_BYTES (2 * RG_BN * RG_KCH)
#define RG_S_BYTES (RG_BM * 4)
#define RG_STAGE_BYTES (RG_A_BYTES + RG_B_BYTES + RG_S_BYTES)
#define RG_SMEM (RG_STAGES * RG_STAGE_BYTES)
#define RG_STATIC_ROWS 64

// A (R, K) half -> hi/lo (R, K) int8, sa (R, KC) fp32. One thread per 16 k of one row; the 8
// threads of a 128-chunk share the max by shuffle
__global__ __launch_bounds__(256)
void quant_a_kernel(const half* __restrict__ a, signed char* __restrict__ ahi, signed char* __restrict__ alo,
                    float* __restrict__ sa, const int R, const int K, const int KC)
{
    const int idx = blockIdx.x * 256 + threadIdx.x;
    const int eighth = idx & 7, chunk = (idx >> 3) % KC, row = (idx >> 3) / KC;
    if (row >= R) return;
    const int k = chunk * RG_KCH + eighth * 16;
    float v[16];
    #pragma unroll
    for (int q = 0; q < 2; ++q)
    {
        int4 raw = make_int4(0, 0, 0, 0);
        if (k + q * 8 < K) raw = *(const int4*) (a + (size_t) row * K + k + q * 8);
        const half2* h2 = (const half2*) &raw;
        #pragma unroll
        for (int u = 0; u < 4; ++u) { float2 f = __half22float2(h2[u]); v[q * 8 + u * 2] = f.x; v[q * 8 + u * 2 + 1] = f.y; }
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
    if (k < K)
    {
        *(int4*) (ahi + (size_t) row * K + k) = hi4;
        *(int4*) (alo + (size_t) row * K + k) = lo4;
    }
    if (eighth == 0) sa[(size_t) row * KC + chunk] = __fdiv_rn(scale, DET_QMAX);
}

__global__ __launch_bounds__(RG_THREADS)
void routing_gemm_i8_kernel
(
    const signed char* __restrict__ ahi, const signed char* __restrict__ alo, const float* __restrict__ sa,   // (R, K), (R, K), (R, KC)
    const signed char* __restrict__ bhi, const signed char* __restrict__ blo, const float* __restrict__ sb,   // (E, K), (E, K), (E)
    half* __restrict__ c, float* __restrict__ part,                                                           // (R, E) half, or (S, R, E) fp32 partials
    const int R, const int E, const int K, const int KC, const int kslice
)
{
    extern __shared__ __align__(128) unsigned char dsm[];
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    const int wm = warp / 4, wn = warp % 4;
    const int g = lane / 4, tg = lane % 4;
    const int r0 = blockIdx.y * RG_BM, e0 = blockIdx.x * RG_BN;
    const int k_beg = blockIdx.z * kslice, k_end = min(k_beg + kslice, K);
    const int n_chunks = (k_end - k_beg + RG_KCH - 1) / RG_KCH;

    auto st_ahi = [&](int st) { return dsm + st * RG_STAGE_BYTES; };
    auto st_alo = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_BM * RG_KCH; };
    auto st_bhi = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_A_BYTES; };
    auto st_blo = [&](int st) { return dsm + st * RG_STAGE_BYTES + RG_A_BYTES + RG_BN * RG_KCH; };
    auto st_sa  = [&](int st) { return (float*) (dsm + st * RG_STAGE_BYTES + RG_A_BYTES + RG_B_BYTES); };

    auto issue = [&](int chunk, int st)
    {
        const int k0 = k_beg + chunk * RG_KCH;
        #pragma unroll
        for (int q = 0; q < 2; ++q)
        {
            const int i = t + q * RG_THREADS, row = i / 8, cc = i % 8;
            const int gr = r0 + row, k = k0 + cc * 16;
            const int bytes = (gr < R) ? max(0, min(16, k_end - k)) : 0;
            const size_t off = bytes ? (size_t) gr * K + k : 0;
            det_cp_async16(det_smem_u32(st_ahi(st) + det_swz8(row, cc)), ahi + off, bytes);
            det_cp_async16(det_smem_u32(st_alo(st) + det_swz8(row, cc)), alo + off, bytes);
        }
        {
            const int row = t / 8, cc = t % 8;
            const int ge = e0 + row, k = k0 + cc * 16;
            const int bytes = (ge < E) ? max(0, min(16, k_end - k)) : 0;
            const size_t off = bytes ? (size_t) ge * K + k : 0;
            det_cp_async16(det_smem_u32(st_bhi(st) + det_swz8(row, cc)), bhi + off, bytes);
            det_cp_async16(det_smem_u32(st_blo(st) + det_swz8(row, cc)), blo + off, bytes);
        }
        if (t < RG_BM / 4)
        {
            float v4[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u)
            {
                const int gr = r0 + t * 4 + u;
                v4[u] = (gr < R) ? sa[(size_t) gr * KC + (k0 / RG_KCH)] : 0.0f;
            }
            *(float4*) (st_sa(st) + t * 4) = make_float4(v4[0], v4[1], v4[2], v4[3]);
        }
    };

    float facc[2][2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q) facc[i][j][q] = 0.0f;

    issue(0, 0); det_cp_async_commit();
    for (int chunk = 0; chunk < n_chunks; ++chunk)
    {
        det_cp_async_wait<0>();
        __syncthreads();
        if (chunk + 1 < n_chunks) issue(chunk + 1, (chunk + 1) % RG_STAGES);
        det_cp_async_commit();
        const int st = chunk % RG_STAGES;
        const unsigned a_hi = det_smem_u32(st_ahi(st)), a_lo = det_smem_u32(st_alo(st));
        const unsigned b_hi = det_smem_u32(st_bhi(st)), b_lo = det_smem_u32(st_blo(st));
        const float* sa_st = st_sa(st);

        int acc_hh[2][2][4], acc_x[2][2][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q) { acc_hh[i][j][q] = 0; acc_x[i][j][q] = 0; }

        #pragma unroll
        for (int ks = 0; ks < RG_KCH; ks += 32)
        {
            unsigned ah[2][4], al[2][4], bh[2][2], bl[2][2], r4[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
            {
                det_load_a(ah[i], a_hi, wm * 32 + i * 16, ks / 16, lane);
                det_load_a(al[i], a_lo, wm * 32 + i * 16, ks / 16, lane);
            }
            det_load_b2(r4, b_hi, wn * 16, ks / 16, lane);
            bh[0][0] = r4[0]; bh[0][1] = r4[1]; bh[1][0] = r4[2]; bh[1][1] = r4[3];
            det_load_b2(r4, b_lo, wn * 16, ks / 16, lane);
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
            const float s0 = sa_st[wm * 32 + i * 16 + g], s1 = sa_st[wm * 32 + i * 16 + g + 8];
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                #pragma unroll
                for (int q = 0; q < 4; ++q)
                    facc[i][j][q] = det_flush(acc_hh[i][j][q], acc_x[i][j][q], q >= 2 ? s1 : s0, facc[i][j][q]);
        }
    }
    det_cp_async_wait<0>();
    __syncthreads();
    float (*cs_)[RG_BN + 4] = (float (*)[RG_BN + 4]) dsm;
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int q = 0; q < 4; ++q)
                cs_[wm * 32 + i * 16 + g + (q >= 2 ? 8 : 0)][wn * 16 + j * 8 + tg * 2 + (q & 1)] = facc[i][j][q];
    __syncthreads();
    if (part) part += (size_t) blockIdx.z * R * E;
    #pragma unroll
    for (int q = 0; q < (RG_BM * RG_BN) / RG_THREADS; ++q)
    {
        const int i = (t + q * RG_THREADS) / RG_BN, j = (t + q * RG_THREADS) % RG_BN;
        if (r0 + i < R && e0 + j < E)
        {
            float v = __fmul_rn(cs_[i][j], sb[e0 + j]);
            if (part) part[(size_t) (r0 + i) * E + e0 + j] = v;
            else c[(size_t) (r0 + i) * E + e0 + j] = __float2half_rn(v);
        }
    }
}

__global__ __launch_bounds__(256)
void routing_gemm_reduce_kernel(const float* __restrict__ part, half* __restrict__ c, const int n, const int S)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    for (int s = 0; s < S; ++s) v = __fadd_rn(v, part[(size_t) s * n + i]);
    c[i] = __float2half_rn(v);
}

// Split-K slices: a function of the shape only
static int rg_slices(int R, int E, int KC)
{
    const int tiles = CEIL_DIVIDE(E, RG_BN) * CEIL_DIVIDE(R, RG_BM);
    int S = 1;
    while (tiles * S < 256 && S < 8 && S * 2 <= KC) S *= 2;
    return S;
}

// Per-device static workspaces for decode-class row counts
struct RGWorkspace { at::Tensor ahi, alo, sa, part; int K = 0, E = 0; };
static RGWorkspace g_rg_ws[32];

bool routing_gemm_det_fits(const at::Tensor& hidden, const at::Tensor& gate_i8, const at::Tensor& gate_sb, const at::Tensor& scores)
{
    if (hidden.dtype() != at::kHalf || gate_i8.dtype() != at::kChar || gate_sb.dtype() != at::kFloat || scores.dtype() != at::kHalf) return false;
    if (!hidden.is_contiguous() || !gate_i8.is_contiguous() || !gate_sb.is_contiguous() || !scores.is_contiguous()) return false;
    const int K = hidden.size(-1);
    return K % 16 == 0 && gate_i8.dim() == 3 && gate_i8.size(0) == 2 && gate_i8.size(2) == K && gate_sb.numel() == gate_i8.size(1);
}

void routing_gemm_det_(const at::Tensor& hidden, const at::Tensor& gate_i8, const at::Tensor& gate_sb, at::Tensor& scores, cudaStream_t stream)
{
    const int K = hidden.size(-1);
    const int R = hidden.numel() / K;
    const int E = gate_i8.size(1);
    const int KC = CEIL_DIVIDE(K, RG_KCH);
    TORCH_CHECK(scores.numel() == (int64_t) R * E, "routing_gemm_det: scores shape");
    const int S = rg_slices(R, E, KC);
    const int kslice = CEIL_DIVIDE(KC, S) * RG_KCH;
    const int S_eff = CEIL_DIVIDE(K, kslice);

    int dev = 0;
    cudaGetDevice(&dev);
    at::Tensor ahi, alo, sa, part;
    if (R <= RG_STATIC_ROWS)
    {
        RGWorkspace& ws = g_rg_ws[dev];
        if (ws.K != K || ws.E != E || !ws.ahi.defined() || ws.ahi.device() != hidden.device())
        {
            const int KCs = CEIL_DIVIDE(K, RG_KCH);
            const int Smax = rg_slices(1, E, KCs);                 // the smallest R gives the most slices
            auto opt = hidden.options();
            ws.ahi = at::empty({RG_STATIC_ROWS, K}, opt.dtype(at::kChar));
            ws.alo = at::empty({RG_STATIC_ROWS, K}, opt.dtype(at::kChar));
            ws.sa = at::empty({RG_STATIC_ROWS, KCs}, opt.dtype(at::kFloat));
            ws.part = at::empty({Smax, RG_STATIC_ROWS, E}, opt.dtype(at::kFloat));
            ws.K = K; ws.E = E;
        }
        ahi = ws.ahi; alo = ws.alo; sa = ws.sa; part = ws.part;
    }
    else
    {
        auto opt = hidden.options();
        ahi = at::empty({R, K}, opt.dtype(at::kChar));
        alo = at::empty({R, K}, opt.dtype(at::kChar));
        sa = at::empty({R, KC}, opt.dtype(at::kFloat));
        if (S_eff > 1) part = at::empty({S_eff, R, E}, opt.dtype(at::kFloat));
    }
    TORCH_CHECK(S_eff == 1 || part.numel() >= (int64_t) S_eff * R * E, "routing_gemm_det: partials workspace");

    quant_a_kernel<<<CEIL_DIVIDE(R * KC * 8, 256), 256, 0, stream>>>((const half*) hidden.data_ptr(), (signed char*) ahi.data_ptr(),
        (signed char*) alo.data_ptr(), (float*) sa.data_ptr(), R, K, KC);
    cuda_check(cudaPeekAtLastError());

    static bool attr[32] = {};
    if (!attr[dev]) { cudaFuncSetAttribute(routing_gemm_i8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, RG_SMEM); attr[dev] = true; }
    dim3 grid(CEIL_DIVIDE(E, RG_BN), CEIL_DIVIDE(R, RG_BM), S_eff);
    const signed char* bhi = (const signed char*) gate_i8.data_ptr();
    const signed char* blo = bhi + (size_t) E * K;
    half* c = (half*) scores.data_ptr();
    float* part_p = S_eff > 1 ? (float*) part.data_ptr() : nullptr;
    routing_gemm_i8_kernel<<<grid, RG_THREADS, RG_SMEM, stream>>>(
        (const signed char*) ahi.data_ptr(), (const signed char*) alo.data_ptr(), (const float*) sa.data_ptr(),
        bhi, blo, (const float*) gate_sb.data_ptr(), c, part_p, R, E, K, KC, kslice);
    cuda_check(cudaPeekAtLastError());
    if (S_eff > 1)
    {
        const int n = R * E;
        routing_gemm_reduce_kernel<<<CEIL_DIVIDE(n, 256), 256, 0, stream>>>(part_p, c, n, S_eff);
        cuda_check(cudaPeekAtLastError());
    }
}

void routing_gemm_det
(
    const at::Tensor& hidden,           // (R, K) or (..., K) half
    const at::Tensor& gate_i8,          // (2, E, K) int8: hi, lo slices
    const at::Tensor& gate_sb,          // (E) fp32 row scales
    at::Tensor scores                   // (R, E) half
)
{
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(routing_gemm_det_fits(hidden, gate_i8, gate_sb, scores), "routing_gemm_det: half activations, (2, E, K) int8 gate, fp32 scales, contiguous, K % 16 == 0");
    routing_gemm_det_(hidden, gate_i8, gate_sb, scores, stream);
}

// Weight pre-quantization (load time): gate_t (E, K) half -> (2, E, K) int8 hi/lo, (E) fp32 scales
__global__ __launch_bounds__(256)
void quant_gate_kernel(const half* __restrict__ w, signed char* __restrict__ hi, signed char* __restrict__ lo, float* __restrict__ sb, const int E, const int K)
{
    const int row = blockIdx.x;
    float m = 0.0f;
    for (int k = threadIdx.x; k < K; k += 256) m = fmaxf(m, fabsf(__half2float(w[(size_t) row * K + k])));
    __shared__ float red[8];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x / 32] = m;
    __syncthreads();
    m = red[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    const float scale = m > 0.0f ? m : 1.0f;
    const float inv = __fdiv_rn(DET_QMAX, scale);
    for (int k = threadIdx.x; k < K; k += 256)
    {
        signed char h, l;
        det_quant_split(__half2float(w[(size_t) row * K + k]), inv, h, l);
        hi[(size_t) row * K + k] = h;
        lo[(size_t) row * K + k] = l;
    }
    if (threadIdx.x == 0) sb[row] = __fdiv_rn(scale, DET_QMAX);
}

void det_quant_weight
(
    const at::Tensor& w,                // (N, K) half, contiguous
    at::Tensor w_i8,                    // (2, N, K) int8 out
    at::Tensor w_s                      // (N) fp32 out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(w.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK(w.dim() == 2 && w.is_contiguous() && w_i8.is_contiguous() && w_s.is_contiguous(), "det_quant_weight: layout");
    const int N = w.size(0), K = w.size(1);
    TORCH_CHECK(w_i8.size(0) == 2 && w_i8.size(1) == N && w_i8.size(2) == K && w_s.numel() == N, "det_quant_weight: shapes");
    signed char* hi = (signed char*) w_i8.data_ptr();
    quant_gate_kernel<<<N, 256, 0, stream>>>((const half*) w.data_ptr(), hi, hi + (size_t) N * K, (float*) w_s.data_ptr(), N, K);
    cuda_check(cudaPeekAtLastError());
}

// Test hook: exp_det / log_det / softplus_det over a tensor (elementwise, fp32)
__global__ void det_math_kernel(const float* __restrict__ x, float* __restrict__ y, const int n)
{
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    y[i] = exp_det(x[i]); y[n + i] = log_det(fabsf(x[i]) + 1e-30f); y[2 * n + i] = softplus_det(x[i]);
}
void det_math_test(const at::Tensor& x, at::Tensor y)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int n = x.numel();
    TORCH_CHECK(y.numel() == 3 * n, "det_math_test: y must hold 3 x n");
    det_math_kernel<<<CEIL_DIVIDE(n, 256), 256, 0, stream>>>((const float*) x.data_ptr(), (float*) y.data_ptr(), n);
    cuda_check(cudaPeekAtLastError());
}
