#include <cuda_fp16.h>
#include "sparse_head.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "reduction.cuh"

// Sparse draft head: x (m, k) fp16 times the column buffer w (k, n_alloc) fp16, fp32 accumulation, plus bias,
// columns at or past n_active left out, argmax per row, the winning column mapped to its vocabulary id and
// optionally its embedding row gathered. Replaces a cuBLAS GEMM, its split-K reduce, the mask fill, the max
// reduction and the index gather with two launches.

#define SH_THREADS 256
#define SH_WARPS 8
#define SH_COLS 128
#define SH_MAX_ROWS 4
#define SH_MAX_KSPLIT 16
#define SH_MAX_KBLOCK 1024
#define SH_UNROLL 8

// Kernel 1: block (tile, ks) owns 128 columns (four per lane, 8-byte loads, so a warp streams 256 contiguous
// bytes per k row) and a 1/ksplit share of k; its eight warps split that share further and keep eight
// independent loads in flight per lane (the kernel is memory-latency bound otherwise). The x slice sits in
// shared memory. The warps' sums meet in shared memory and the block writes one fp32 partial per (ks, row,
// column). Rows are capped at 4: beyond that the scalar FMA count per weight makes a tensor-core GEMM the
// better tool, and the caller routes larger batches to the torch path
__launch_bounds__(SH_THREADS)
__global__ void sparse_head_partial_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const int m,
    const int k,
    const int n_alloc,
    const int ksplit,
    float* __restrict__ part
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col0 = blockIdx.x * SH_COLS + lane * 4;
    const int ks = blockIdx.y;
    const int kb = k / ksplit;              // this block's k range
    const int kbase = ks * kb;
    const int kw = kb / SH_WARPS;           // this warp's slice of it
    const int i0 = warp * kw;

    __shared__ half xs[SH_MAX_ROWS * SH_MAX_KBLOCK];
    for (int idx = threadIdx.x; idx < m * kb; idx += SH_THREADS)
    {
        int r = idx / kb;
        int i = idx - r * kb;
        xs[r * kb + i] = x[r * k + kbase + i];
    }
    __syncthreads();

    float acc[SH_MAX_ROWS][4];
    #pragma unroll
    for (int r = 0; r < SH_MAX_ROWS; ++r)
        #pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;

    const uint2* wp = reinterpret_cast<const uint2*>(w + (size_t)kbase * n_alloc + col0);
    const size_t wstride = (size_t)n_alloc / 4;         // uint2 (four halves) per row
    int i = i0;
    for (; i + SH_UNROLL <= i0 + kw; i += SH_UNROLL)
    {
        uint2 wv[SH_UNROLL];
        #pragma unroll
        for (int u = 0; u < SH_UNROLL; ++u) wv[u] = wp[(size_t)(i + u) * wstride];
        #pragma unroll
        for (int u = 0; u < SH_UNROLL; ++u)
        {
            half2 w01 = *reinterpret_cast<half2*>(&wv[u].x);
            half2 w23 = *reinterpret_cast<half2*>(&wv[u].y);
            float w0 = __low2float(w01), w1 = __high2float(w01);
            float w2 = __low2float(w23), w3 = __high2float(w23);
            #pragma unroll
            for (int r = 0; r < SH_MAX_ROWS; ++r)
            {
                if (r < m)
                {
                    float xv = __half2float(xs[r * kb + i + u]);
                    acc[r][0] += xv * w0;
                    acc[r][1] += xv * w1;
                    acc[r][2] += xv * w2;
                    acc[r][3] += xv * w3;
                }
            }
        }
    }
    for (; i < i0 + kw; ++i)
    {
        uint2 wv = wp[(size_t)i * wstride];
        half2 w01 = *reinterpret_cast<half2*>(&wv.x);
        half2 w23 = *reinterpret_cast<half2*>(&wv.y);
        float w0 = __low2float(w01), w1 = __high2float(w01);
        float w2 = __low2float(w23), w3 = __high2float(w23);
        #pragma unroll
        for (int r = 0; r < SH_MAX_ROWS; ++r)
        {
            if (r < m)
            {
                float xv = __half2float(xs[r * kb + i]);
                acc[r][0] += xv * w0;
                acc[r][1] += xv * w1;
                acc[r][2] += xv * w2;
                acc[r][3] += xv * w3;
            }
        }
    }

    __shared__ float4 red[SH_WARPS][SH_MAX_ROWS][32];
    for (int r = 0; r < m; ++r)
        red[warp][r][lane] = make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]);
    __syncthreads();
    if (warp < m)
    {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int s = 0; s < SH_WARPS; ++s)
        {
            float4 t = red[s][warp][lane];
            v.x += t.x; v.y += t.y; v.z += t.z; v.w += t.w;
        }
        *reinterpret_cast<float4*>(part + ((size_t)ks * m + warp) * n_alloc + col0) = v;
    }
}

// Kernel 2: one block per row sums the k-split partials per column, adds the bias, skips the unfilled
// columns, takes the argmax and maps it to the vocabulary id (and the embedding row)
__launch_bounds__(SH_THREADS)
__global__ void sparse_head_final_kernel
(
    const float* __restrict__ part,
    const int ksplit,
    const int m,
    const int n_alloc,
    const int n_active,
    const float* __restrict__ bias,
    const int64_t* __restrict__ cols,
    int64_t* __restrict__ out_ids,
    float* __restrict__ out_conf,
    const half* __restrict__ emb_rows,
    half* __restrict__ out_emb,
    const int hidden
)
{
    const int r = blockIdx.x;
    ValIdx best;
    best.val = -INFINITY;
    best.idx = 0;
    for (int c = threadIdx.x; c < n_active; c += SH_THREADS)
    {
        float v = 0.0f;
        for (int ks = 0; ks < ksplit; ++ks)
            v += part[((size_t)ks * m + r) * n_alloc + c];
        if (bias) v += bias[c];
        if (v > best.val) { best.val = v; best.idx = c; }
    }
    // Block argmax over the eight warps (reduction.cuh's block_reduce_argmax assumes 32 warps)
    best = warp_reduce_argmax(best);
    __shared__ ValIdx warp_best[SH_WARPS];
    __shared__ int s_idx;
    if ((threadIdx.x & 31) == 0) warp_best[threadIdx.x >> 5] = best;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        best = warp_best[0];
        #pragma unroll
        for (int s = 1; s < SH_WARPS; ++s)
            if (warp_best[s].val > best.val) best = warp_best[s];
        s_idx = best.idx;
        out_conf[r] = best.val;
        out_ids[r] = cols[best.idx];
    }
    __syncthreads();
    if (emb_rows)
    {
        const half* src = emb_rows + (size_t)s_idx * hidden;
        half* dst = out_emb + (size_t)r * hidden;
        for (int h = threadIdx.x; h < hidden; h += SH_THREADS)
            dst[h] = src[h];
    }
}

void sparse_head_argmax
(
    at::Tensor x,
    at::Tensor w,
    const c10::optional<at::Tensor>& bias,
    int64_t n_active,
    at::Tensor cols,
    at::Tensor out_ids,
    at::Tensor out_conf,
    at::Tensor scratch,
    const c10::optional<at::Tensor>& emb_rows,
    const c10::optional<at::Tensor>& out_emb
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(cols, kLong);
    TORCH_CHECK_DTYPE(out_ids, kLong);
    TORCH_CHECK_DTYPE(out_conf, kFloat);
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "sparse_head_argmax: x and w must be contiguous");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2, "sparse_head_argmax: x (m, k), w (k, n)");
    int m = x.size(0);
    int k = x.size(1);
    int n_alloc = w.size(1);
    TORCH_CHECK(m >= 1 && m <= SH_MAX_ROWS, "sparse_head_argmax: at most 4 rows");
    TORCH_CHECK(w.size(0) == k && k % SH_WARPS == 0, "sparse_head_argmax: k mismatch or not divisible by 8");
    TORCH_CHECK(n_alloc % SH_COLS == 0 && n_alloc >= SH_COLS, "sparse_head_argmax: n must be a multiple of 128");
    TORCH_CHECK(n_active >= 1 && n_active <= n_alloc, "sparse_head_argmax: n_active out of range");
    TORCH_CHECK(cols.size(0) >= n_alloc, "sparse_head_argmax: cols too short");
    int tiles = n_alloc / SH_COLS;
    // k split: warp slices of at least 64 rows (so the streaming loop, not the per-block setup, is the work),
    // whole slices, and an x slice that fits the shared buffer (which forces a larger split for long k)
    int ksplit = 0;
    for (int c = SH_MAX_KSPLIT; c >= 1; c >>= 1)
    {
        if (k % (c * SH_WARPS) != 0) continue;
        if (k / c > SH_MAX_KBLOCK) break;
        ksplit = c;
        if (k / (c * SH_WARPS) >= 64) break;
    }
    TORCH_CHECK(ksplit > 0, "sparse_head_argmax: unsupported k");
    TORCH_CHECK(scratch.numel() * (int64_t)scratch.element_size() >= (int64_t)ksplit * m * n_alloc * (int64_t)sizeof(float),
        "sparse_head_argmax: scratch too small");
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias->defined())
    {
        TORCH_CHECK_DTYPE((*bias), kFloat);
        TORCH_CHECK(bias->size(0) >= n_alloc, "sparse_head_argmax: bias too short");
        bias_ptr = (const float*)bias->data_ptr();
    }
    const half* emb_ptr = nullptr;
    half* out_emb_ptr = nullptr;
    int hidden = 0;
    if (emb_rows.has_value() && emb_rows->defined())
    {
        TORCH_CHECK(out_emb.has_value() && out_emb->defined(), "sparse_head_argmax: out_emb required with emb_rows");
        TORCH_CHECK_DTYPE((*emb_rows), kHalf);
        TORCH_CHECK_DTYPE((*out_emb), kHalf);
        TORCH_CHECK(emb_rows->is_contiguous() && out_emb->is_contiguous(), "sparse_head_argmax: embedding tensors must be contiguous");
        hidden = emb_rows->size(1);
        TORCH_CHECK(emb_rows->size(0) >= n_alloc && out_emb->size(0) >= m && out_emb->size(1) == hidden,
            "sparse_head_argmax: embedding shapes");
        emb_ptr = (const half*)emb_rows->data_ptr();
        out_emb_ptr = (half*)out_emb->data_ptr();
    }

    dim3 grid1(tiles, ksplit);
    sparse_head_partial_kernel<<<grid1, SH_THREADS, 0, stream>>>
    (
        (const half*)x.data_ptr(), (const half*)w.data_ptr(), m, k, n_alloc, ksplit, (float*)scratch.data_ptr()
    );
    sparse_head_final_kernel<<<m, SH_THREADS, 0, stream>>>
    (
        (const float*)scratch.data_ptr(), ksplit, m, n_alloc, (int)n_active, bias_ptr, (const int64_t*)cols.data_ptr(),
        (int64_t*)out_ids.data_ptr(), (float*)out_conf.data_ptr(), emb_ptr, out_emb_ptr, hidden
    );
    cuda_check(cudaPeekAtLastError());
}
