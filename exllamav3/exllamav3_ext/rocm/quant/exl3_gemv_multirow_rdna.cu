// =============================================================================
// Multi-row GEMV for RDNA, m = 2..8
// =============================================================================
//
// The GEMV fast paths take m == 1 only; at m = 2..8 -- speculative-decoding
// verify steps (MTP, DFlash, n-gram drafts) -- every dense linear used to fall
// to the cooperative tile GEMM, which pads the rows to a 16-row tile and walks
// the weights at a fraction of the GEMV's bandwidth, making the dense linears
// the largest kernel class of a verify step.
//
// This path reads the weights once for up to 8 rows. The dot core is the
// barrier-free m == 1 core (exl3_gemv_dot_tile_direct) with a row tile M in
// {2, 4, 8} chosen as the smallest >= m: one B dequant per k-tile feeds M
// pairs of fdot2 accumulators, and each row's chain is exactly the m == 1
// chain, so row r of an m-row call is bit-identical to a separate m == 1 call
// on that row. Rows past m are
// computed on row m - 1's data and never stored.
//
// Structure is the pre-fusion three-kernel form, deliberately. The fused
// m == 1 kernels rotate the input into LDS in their prologue, which does not
// extend to m rows on a 64 KB-per-workgroup part (8 x 12288 halves is 196 KB,
// and the rotation is redundant across a matrix's N-tile blocks). So:
//
//   1. exl3_gemv_mr_had_in_{single,multi}   rotates the m rows of every slab
//      into A_had (global; L2-resident thereafter) and hosts the graph patch
//      sites, republishing the patchable pointers the dot kernel needs through
//      a per-device Exl3GemvMrParams block (the pattern of exl3_mgemv_rdna.cu)
//   2. exl3_gemv_mr_dot_{single,multi}      split-K GEMV, one block per
//      (slot, N-tile), M rows per warp; the fused output epilogue (last-
//      arriving warp per 128-wide segment, exl3_gemv_kernel_rdna.cuh)
//      rotates the m row segments in place
//
// Two launches per call at m > 1 is irrelevant: the verify step is GPU-bound.
// A part with more LDS could rotate the rows in-kernel (the
// helpers are shared), gated on a runtime LDS budget; not done here.
//
// Routing: exl3_gemm_gr and exl3_mgemm_gr call the two try_launch entry
// points for 2 <= m <= EXL3_GEMV_MAX_M (default 8; 1 switches the path off)
// before the cooperative kernels, in and out of graph capture. The
// multi-matrix form declines sliced mode.
//
// Routing weights (the MoE down projection's weighted reduce):
// at m == 1 the multi-matrix form takes weighted calls too, with the
// exl3_mgemv fused epilogue verbatim (exl3_gemv_fused_epilogue: rotate the
// segment with scale * weights[orig_pos], then the last rotated slot runs the
// grouped num_tokens reduction in exl3_mgemv_reduce_kernel's order), so the
// output is bit-identical to the LDS-prologue exl3_mgemv_dot_kernel_splitk it
// replaces there -- the dot core, wave count and N-tiles per block are the
// same rule on the same shape; only where the rotated input lives differs
// (A_had in L2 instead of per-block LDS).
// EXL3_ROCM_MR_WEIGHTED=0 restores the decline (weighted calls go back to
// exl3_mgemv_try_launch). num_tokens > 1 is accepted without range packing
// (min_index < 0), the case the grouped reduction divides evenly.
// =============================================================================

#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_mgemv_rdna.cuh"
#include "exl3_gemv_kernel_rdna.cuh"
#include "exl3_gemv_multirow_rdna.cuh"

#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <atomic>

int exl3_gemv_max_m()
{
    const char* env = std::getenv("EXL3_GEMV_MAX_M");
    if (!env) return 8;
    int v = atoi(env);
    return v < 1 ? 1 : (v > 8 ? 8 : v);
}

// m == 1 goes through this structure too, with a row tile of 1 -- rows read
// from L2, no rotated row in LDS -- ahead of the fused m == 1 kernels
// (EXL3_GEMV_MR_M1=0 restores those): the fused kernels' LDS prologue costs
// occupancy, which outweighs the saved launch. Logits are bit-identical
// either way. The fused kernels keep the weighted-reduce MoE down projection
// (this path declines weights) and lm_head-scale widths (the m == 1 single-warp form; leaving them there
// keeps the two routes bit-identical where both split K).
static bool exl3_gemv_mr_m1_enabled()
{
    const char* env = std::getenv("EXL3_GEMV_MR_M1");
    if (!env) return true;
    return atoi(env) != 0;
}

static bool exl3_gemv_mr_weighted_enabled()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_ROCM_MR_WEIGHTED");
        return !env || atoi(env) != 0;
    }();
    return on;
}

static inline int exl3_gemv_mr_min_m(int n_tiles)
{
    return (exl3_gemv_mr_m1_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES) ? 1 : 2;
}

// Half-integer bitrates (EXL3_HALF_BITS(K), K = 1..3, mul1): the tiles core's
// dq8_half decoder (exl3_gemv_tiles_rdna.cuh) on the same bodies. Callers
// pass the pseudo width only for half-rate weights and only while this is on;
// re-read per call (half-rate calls only -- integer calls never reach it).
bool exl3_rocm_half_gemv_enabled()
{
    const char* env = std::getenv("EXL3_ROCM_HALF_GEMV");
    return !env || atoi(env) != 0;
}

// Widths this file dispatches: integer 1..8 (any codebook), half 1.5 / 2.5 / 3.5 (mul1)
static inline bool exl3_gemv_mr_k_ok(int K, int cb)
{
    if (K >= 1 && K <= 8) return true;
    return K >= EXL3_HALF_BITS(1) && K <= EXL3_HALF_BITS(3) && cb == 2;
}

// -----------------------------------------------------------------------------
// Per-device parameter block: the patchable pointers the dot kernels read
// (republished by the rotation kernels, which host the graph sites) and the
// fused epilogue's arrival counters, [slot][segment], self-resetting.
// -----------------------------------------------------------------------------
struct Exl3GemvMrParams
{
    const uint16_t* B;      // single-matrix
    void* C;
    half* A_had;
    const half* svh;
    void* Cm;               // multi-matrix
    const int64_t* indices;
    const half* weights;
    int seg_counters[EXL3_MGEMV_SEG_COUNTERS];
    int red_counters[EXL3_MGEMV_RED_COUNTERS];   // weighted reduce, per output segment
};

static Exl3GemvMrParams* exl3_gemv_mr_param_block(int device, bool allow_alloc)
{
    static std::mutex mtx;
    static Exl3GemvMrParams* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device] && allow_alloc)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(Exl3GemvMrParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return nullptr;
        }
        if (cudaMemset(p, 0, sizeof(Exl3GemvMrParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            cudaFree(p);
            return nullptr;
        }
        blocks[device] = (Exl3GemvMrParams*) p;
    }
    return blocks[device];
}

void exl3_gemv_multirow_prewarm(int device)
{
    (void) exl3_gemv_mr_param_block(device, true);
}

// =============================================================================
// Multi-row dot core: exl3_gemv_dot_tile_direct with M rows per warp
// =============================================================================
// Same B path, same per-row fdot2 chain and the same quad reduction as the
// m == 1 core, so each row is bit-identical to it. A is the slab's rotated
// rows, lda halves apart, rows_valid of them (<= M): rows past that re-read
// the last valid row rather than memory beyond the slab (their outputs are
// discarded). out[r] is row r's column-lane value (lanes 0-15).

template <int bits, int cb, int M>
__device__ __forceinline__ void exl3_gemv_dot_tile_direct_mr
(
    const half* __restrict__ A,
    const int lda,
    const uint16_t* __restrict__ B,
    const int n_tiles,
    const int tile_n,
    const int lane,
    const int kb_begin,
    const int kb_end,
    float* out,
    const int rows_valid = M
)
{
    constexpr int tile_elements = Exl3Width<bits>::tile_u16;   // 16 * bits (integer K)
    const int r0 = (lane & 3) * 2;
    const int row_last = rows_valid - 1;
    int a_row[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) a_row[r] = (r < row_last ? r : row_last) * lda;

    float accA[M], accB[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) { accA[r] = 0.0f; accB[r] = 0.0f; }

    for (int k_tile = kb_begin; k_tile < kb_end; k_tile++)
    {
        const uint32_t* b_ptr = (const uint32_t*)
            (B + (k_tile * n_tiles + tile_n) * tile_elements);

        FragB frag0, frag1;
        dq_dispatch<Exl3Width<bits>::ka, cb, Exl3Width<bits>::half>(b_ptr, lane << 3, frag0, frag1);

        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            const half2* a2 = (const half2*) (A + a_row[r] + k_tile * 16);
            half2 a01 = a2[r0 >> 1];
            half2 a89 = a2[(r0 >> 1) + 4];
            accA[r] = exl3_fdot2(a01, frag0[0], accA[r]);
            accA[r] = exl3_fdot2(a89, frag0[1], accA[r]);
            accB[r] = exl3_fdot2(a01, frag1[0], accB[r]);
            accB[r] = exl3_fdot2(a89, frag1[1], accB[r]);
        }
    }

    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        accA[r] += __shfl_xor(accA[r], 1, 32);
        accA[r] += __shfl_xor(accA[r], 2, 32);
        accB[r] += __shfl_xor(accB[r], 1, 32);
        accB[r] += __shfl_xor(accB[r], 2, 32);
        float vA = __shfl(accA[r], (lane & 7) * 4, 32);
        float vB = __shfl(accB[r], (lane & 7) * 4, 32);
        out[r] = (lane & 8) ? vB : vA;
    }
}

// Split-K form: the m == 1 chunking and reduction order (exl3_gemv_dot_tile_
// splitk), M rows wide. sh_red holds WARPS_PER_BLOCK * M * 16 floats. Every
// thread of the block enters; out is meaningful for warp 0, lanes 0-15.
template <int bits, int cb, int WARPS_PER_BLOCK, int M>
__device__ __forceinline__ void exl3_gemv_dot_tile_splitk_mr
(
    const half* __restrict__ A,
    const int lda,
    const uint16_t* __restrict__ B,
    const int size_k,
    const int n_tiles,
    const int tile_n,
    const int warp_id,
    const int lane,
    float* sh_red,
    float* out
)
{
    const int num_k_tiles = size_k / 16;
    const int chunk = (num_k_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int kb0 = warp_id * chunk;
    const int kb1 = kb0 + chunk < num_k_tiles ? kb0 + chunk : num_k_tiles;

    float acc[M];
    #pragma unroll
    for (int r = 0; r < M; ++r) acc[r] = 0.0f;
    if (kb0 < kb1)
        exl3_gemv_dot_tile_direct_mr<bits, cb, M>(A, lda, B, n_tiles, tile_n, lane, kb0, kb1, acc);

    if (lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r) sh_red[(warp_id * M + r) * 16 + lane] = acc[r];
    }
    __syncthreads();

    if (warp_id == 0 && lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w)
                total += sh_red[(w * M + r) * 16 + lane];
            out[r] = total;
        }
    }
}

static inline size_t exl3_gemv_mr_smem_bytes(int warps, int M)
{
    return (size_t) warps * M * EXL3_GEMV_TILES_TMAX * 16 * sizeof(float);
}

// =============================================================================
// Multi-row fused output epilogue: the m == 1 handshake, m row segments
// =============================================================================
// Entered by the warp holding the tile's M rows in acc (lanes 0-15); stores
// rows < rows_valid, arrives once per tile, and as the last of the segment's
// 8 tiles rotates each valid row's segment in place with the shared helper --
// per row the arithmetic of the m == 1 epilogue exactly.

template <bool c_fp32, int M>
__device__ __forceinline__ void exl3_gemv_fused_epilogue_mr
(
    const float* acc,
    void* Cb,
    const int64_t row_off,       // slab base within Cb (j * m * n, or 0 for list outputs)
    const int ldc,               // row stride of C
    const int rows_valid,        // m (<= M)
    const int tile_n,
    const int lane,
    const half* __restrict__ svh,
    int* seg_counter
)
{
    if (lane < 16)
    {
        #pragma unroll
        for (int r = 0; r < M; ++r)
        {
            if (r >= rows_valid) break;
            const int64_t out_idx = row_off + (int64_t) r * ldc + tile_n * 16 + lane;
            if constexpr (c_fp32)
                ((float*) Cb)[out_idx] = acc[r];
            else
                ((half*) Cb)[out_idx] = __float2half(acc[r]);
        }
    }

    __threadfence();
    int old = 0;
    if (lane == 0) old = atomicAdd(seg_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != 7) return;

    __threadfence();
    if (lane == 0) *seg_counter = 0;
    const int seg = tile_n / 8;
    const float r_scale = 0.088388347648f;  // 1/sqrt(128)
    #pragma unroll
    for (int r = 0; r < M; ++r)
    {
        if (r >= rows_valid) break;
        const int64_t seg_off = row_off + (int64_t) r * ldc + (int64_t) seg * 128;
        if constexpr (c_fp32)
            exl3_gemv_had_out_128_f(((float*) Cb) + seg_off, svh + seg * 128, r_scale, lane);
        else
            exl3_gemv_had_out_128_h(((half*) Cb) + seg_off, svh + seg * 128, r_scale, lane);
    }
}

// =============================================================================
// Kernel 1: input rotation, m rows per slab, and the graph patch sites
// =============================================================================
// One warp per (slab, row, 128-block). The single-matrix form hosts the six
// GP_gemm_* sites (argument order 0-5 load-bearing; B, C, svh are patch hosts
// only) and the multi-matrix form the four GP_mgemm_* sites (0-3; weights is a
// patch host only -- the path declines weighted calls).

static __global__
__launch_bounds__(256)
void exl3_gemv_mr_had_in_single
(
    const half* __restrict__ A,        // 0: GP_gemm_A
    const uint16_t* __restrict__ B,    // 1: GP_gemm_B_trellis (patch host only)
    void* __restrict__ C,              // 2: GP_gemm_C (patch host only)
    const half* __restrict__ suh,      // 3: GP_gemm_B_suh
    half* __restrict__ A_had,          // 4: GP_gemm_A_had
    const half* __restrict__ svh,      // 5: GP_gemm_B_svh (patch host only)
    Exl3GemvMrParams* __restrict__ pb,
    const int size_k,
    const int size_m
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->B = B;
        pb->C = C;
        pb->A_had = A_had;
        pb->svh = svh;
    }
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= size_m * k_segs) return;
    const int row = warp_id / k_segs;
    const int seg = warp_id % k_segs;
    const int64_t off = (int64_t) row * size_k + seg * 128;
    exl3_gemv_had_in_128(A + off, A_had + off, suh + seg * 128, lane);
}

static __global__
__launch_bounds__(256)
void exl3_gemv_mr_had_in_multi
(
    const half* __restrict__ A,          // 0: GP_mgemm_A
    void* __restrict__ C,                // 1: GP_mgemm_C (patch host only)
    const int64_t* __restrict__ indices, // 2: GP_mgemm_indices
    const half* __restrict__ weights,    // 3: GP_mgemm_weights (patch host only)
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ suh_list,
    half* __restrict__ A_had,
    const int size_k,
    const int size_m,
    const int bszm_in,
    const int bszm,
    const int min_index,
    const int max_index
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->Cm = C;
        pb->indices = indices;
        pb->weights = weights;
    }
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= bszm * size_m * k_segs) return;
    const int j   = warp_id / (size_m * k_segs);
    const int row = (warp_id / k_segs) % size_m;
    const int seg = warp_id % k_segs;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0) return;

    const half* suh = (const half*) suh_list[mat_index];
    // A slab j (or the broadcast input), row `row`; A_had slabs follow packed slot numbering
    const half* A_src = (bszm_in == 1 ? A : A + (int64_t) j * size_m * size_k)
                        + (int64_t) row * size_k + seg * 128;
    half* dst = A_had + ((int64_t) j * size_m + row) * size_k + seg * 128;
    exl3_gemv_had_in_128(A_src, dst, suh + seg * 128, lane);
}

// Weighted form, skipped slot: exl3_gemv_fused_epilogue with a zero tile and no
// rotation (the whole segment is zero, and so is its rotation), then the same
// reduction arrival -- and, as the last rotated slot, the same grouped sum.
template <bool c_fp32>
__device__ __forceinline__ void exl3_gemv_mr_zero_epilogue
(
    void* Cb,
    const int64_t row_off,
    const int size_n,
    const int tile_n,
    const int lane,
    int* seg_counter,
    int* red_counter,
    const int red_target,
    const int num_tokens
)
{
    if (lane < 16)
    {
        const int64_t out_idx = row_off + tile_n * 16 + lane;
        if constexpr (c_fp32) ((float*) Cb)[out_idx] = 0.0f;
        else                  ((half*) Cb)[out_idx] = __float2half(0.0f);
    }
    __threadfence();
    int old = 0;
    if (lane == 0) old = atomicAdd(seg_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != 7) return;
    __threadfence();
    if (lane == 0) *seg_counter = 0;

    __threadfence();
    if (lane == 0) old = atomicAdd(red_counter, 1);
    old = __shfl(old, 0, 32);
    if (old != red_target - 1) return;
    __threadfence();
    if (lane == 0) *red_counter = 0;
    const int seg = tile_n / 8;
    const int stride = red_target / num_tokens;
    const int col0 = seg * 128 + lane;
    for (int t = 0; t < num_tokens; ++t)
    {
        if constexpr (c_fp32)
        {
            const float* C_ = ((const float*) Cb) + (int64_t) t * stride * size_n + col0;
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int jj = 0; jj < stride; ++jj)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c) sum[c] += C_[c * 32];
                C_ += size_n;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                ((float*) Cb)[(int64_t) t * size_n + col0 + c * 32] = sum[c];
        }
        else
        {
            const half* C_ = ((const half*) Cb) + (int64_t) t * stride * size_n + col0;
            half sum[4] = {};
            for (int jj = 0; jj < stride; ++jj)
            {
                #pragma unroll
                for (int c = 0; c < 4; ++c) sum[c] = __hadd(sum[c], C_[c * 32]);
                C_ += size_n;
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                ((half*) Cb)[(int64_t) t * size_n + col0 + c * 32] = sum[c];
        }
    }
}

// =============================================================================
// Kernel 2: the split-K GEMV, M rows per warp, fused output epilogue
// =============================================================================
// Grid (n_tiles, bszm); one block per (slot, N-tile). Both forms share the body.

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M, bool A_LDS = false>
__device__ __forceinline__ void exl3_gemv_mr_body
(
    const half* __restrict__ A,     // this slab's rotated rows (LDS when A_LDS)
    const int lda,
    const uint16_t* __restrict__ B,
    const int size_k,
    const int n_tiles_j,
    const int tile_n,               // first of the block's tpb tiles
    void* Cb,
    const int64_t row_off,
    const int ldc,
    const int rows_valid,
    const half* __restrict__ svh,
    int* seg_counters,              // this slot's counters, indexed by segment
    const int warp_id,
    const int lane,
    const int core,
    const int tpb,
    const float w_scale = 0.0f,        // weighted form (M == 1): 1/sqrt(128) * weights[orig_pos]
    int* red_counters = nullptr,       // weighted form: per-segment rotated-slot counters
    const int red_target = 0,          // weighted form: packed slot count
    const int size_n = 0,              // weighted form: row stride of C (reduction)
    const int num_tokens = 1
)
{
    extern __shared__ char shared_mem[];
    float* sh_red = (float*) shared_mem;

    // tpb adjacent N-tiles per block with the tiles core (1 otherwise); each
    // tile's rows keep the one-tile chain and reduction order
    constexpr int T = EXL3_GEMV_TILES_TMAX;
    float acc[T * M];
    exl3_gemv_dot_tile_splitk_t<bits, cb, WARPS_PER_BLOCK, M, A_LDS, T>
    (
        core, tpb, A, lda, B, size_k, n_tiles_j, tile_n, warp_id, lane, sh_red, acc,
        [&](int tn, int k0, int k1, float* o)
        {
            exl3_gemv_dot_tile_direct_mr<bits, cb, M>(A, lda, B, n_tiles_j, tn, lane, k0, k1, o, rows_valid);
        },
        rows_valid
    );
    if (warp_id != 0) return;
    if constexpr (M == 1)
    {
        if (red_counters)
        {
            // Weighted reduce: exactly exl3_mgemv_store_or_fuse's fused call
            for (int t = 0; t < tpb; ++t)
                exl3_gemv_fused_epilogue<c_fp32>
                (
                    acc[t], Cb, row_off, size_n, tile_n + t, lane, svh, w_scale,
                    seg_counters + (tile_n + t) / 8, red_counters + (tile_n + t) / 8,
                    red_target, num_tokens
                );
            return;
        }
    }
    for (int t = 0; t < tpb; ++t)
        exl3_gemv_fused_epilogue_mr<c_fp32, M>
        (
            acc + t * M, Cb, row_off, ldc, rows_valid, tile_n + t, lane, svh,
            seg_counters + (tile_n + t) / 8
        );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_mr_dot_single
(
    Exl3GemvMrParams* __restrict__ pb,
    const int size_k,
    const int size_n,
    const int size_m,
    const int core,
    const int tpb
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;
    exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
    (
        pb->A_had, size_k, pb->B, size_k, n_tiles, tile_n,
        pb->C, 0, size_n, size_m, pb->svh,
        pb->seg_counters, warp_id, lane, core, tpb
    );
}

// -----------------------------------------------------------------------------
// Single-matrix form with the input rotation fused in (EXL3_GEMV_MR_FUSE_IN)
// -----------------------------------------------------------------------------
// One launch instead of exl3_gemv_mr_had_in_single + exl3_gemv_mr_dot_single.
// The rotated rows live in LDS (size_m * size_k halves, row stride size_k --
// the A_had layout), after the split-K reduction slots. There is no block
// barrier: each warp rotates only the 128-segments its own split-K chunk
// touches (the chunk rule of exl3_gemv_dot_tile_splitk_t, restated here) and
// then reads only that chunk, so it consumes nothing but its own LDS writes.
// A segment straddling two chunks is rotated by both warps into the same
// location with the same bits. The rotation is exl3_gemv_had_in_128 and the
// dot core is the same template reading LDS instead of global memory, so the
// output is bit-identical to the two-kernel form. Hosts the six GP_gemm_*
// sites itself (argument order 0-5; A_had is a patch host only).

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_mr_fused_single
(
    const half* __restrict__ A,        // 0: GP_gemm_A
    const uint16_t* __restrict__ B,    // 1: GP_gemm_B_trellis
    void* __restrict__ C,              // 2: GP_gemm_C
    const half* __restrict__ suh,      // 3: GP_gemm_B_suh
    half* __restrict__ A_had,          // 4: GP_gemm_A_had (patch host only)
    const half* __restrict__ svh,      // 5: GP_gemm_B_svh
    Exl3GemvMrParams* __restrict__ pb,
    const int size_k,
    const int size_n,
    const int size_m,
    const int core,
    const int tpb
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;

    extern __shared__ char shared_mem[];
    half* sh_a = (half*) (shared_mem + WARPS_PER_BLOCK * M * EXL3_GEMV_TILES_TMAX * 16 * sizeof(float));

    const int num_k_tiles = size_k / 16;
    const int chunk = (num_k_tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int kb0 = exl3_uni(warp_id) * chunk;
    const int kb1 = kb0 + chunk < num_k_tiles ? kb0 + chunk : num_k_tiles;
    if (kb0 < kb1)
    {
        const int s0 = kb0 / 8;
        const int s1 = (kb1 + 7) / 8;
        for (int row = 0; row < size_m; ++row)
            for (int seg = s0; seg < s1; ++seg)
            {
                const int64_t off = (int64_t) row * size_k + seg * 128;
                exl3_gemv_had_in_128(A + off, sh_a + off, suh + seg * 128, lane);
            }
    }
    // The wave's LDS writes before its reads (cross-lane, same wave)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

    exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M, true>
    (
        sh_a, size_k, B, size_k, n_tiles, tile_n,
        C, 0, size_n, size_m, svh,
        pb->seg_counters, warp_id, lane, core, tpb
    );
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK, int M>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_mr_dot_multi
(
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const uintptr_t* __restrict__ svh_list,
    const half* __restrict__ A_had,
    const int size_k,
    const int size_n,
    const int size_m,
    const int bszm,
    const int min_index,
    const int max_index,
    const int* __restrict__ size_n_list,  // per-matrix widths; size_n is the max
    void* const* __restrict__ c_list,     // per-matrix output bases (m rows, stride n_j)
    const int core,
    const int tpb,
    const int num_tokens,                 // weighted form: reduction groups (0: unweighted)
    const int weighted
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;

    int orig_pos;
    int mat_index = exl3_mgemv_mat_index(pb->indices, j, bszm, min_index, max_index, &orig_pos);
    if (mat_index < 0)   // block-uniform
    {
        // Weighted form: a skipped slot (negative index, no packing) still has to arrive
        // at the grouped reduce, as an all-zero row -- adding +0 leaves every sum exactly
        // what skipping the row gives, which is the cooperative kernel's semantics
        if (M == 1 && weighted && min_index < 0 && warp_id == 0)
        {
            const int n_segs0 = size_n / 128;
            const int red_target = exl3_mgemv_packed_count(pb->indices, bszm, min_index, max_index);
            for (int t = 0; t < tpb; ++t)
                exl3_gemv_mr_zero_epilogue<c_fp32>
                (
                    pb->Cm, (int64_t) j * size_n, size_n, tile_n + t, lane,
                    pb->seg_counters + j * n_segs0 + (tile_n + t) / 8,
                    pb->red_counters + (tile_n + t) / 8, red_target, num_tokens
                );
        }
        return;
    }
    const int n_j = size_n_list ? size_n_list[mat_index] : size_n;
    const int n_tiles_j = n_j / 16;
    if (tile_n >= n_tiles_j) return;   // block-uniform

    void* Cb = c_list ? c_list[mat_index] : pb->Cm;
    const int64_t row_off = c_list ? 0 : (int64_t) j * size_m * size_n;
    const int ldc = c_list ? n_j : size_n;
    const int n_segs = size_n / 128;
    if (M == 1 && weighted)   // launch-uniform; size_m == 1, no lists (entry point)
    {
        const half* weights = pb->weights;
        const float w_scale = 0.088388347648f * __half2float(weights[orig_pos]);
        const int red_target = exl3_mgemv_packed_count(pb->indices, bszm, min_index, max_index);
        exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
        (
            A_had + (int64_t) j * size_m * size_k, size_k,
            (const uint16_t*) B_list[mat_index], size_k, n_tiles_j, tile_n,
            Cb, row_off, ldc, size_m, (const half*) svh_list[mat_index],
            pb->seg_counters + j * n_segs, warp_id, lane, core, tpb,
            w_scale, pb->red_counters, red_target, size_n, num_tokens
        );
        return;
    }
    exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, M>
    (
        A_had + (int64_t) j * size_m * size_k, size_k,
        (const uint16_t*) B_list[mat_index], size_k, n_tiles_j, tile_n,
        Cb, row_off, ldc, size_m, (const half*) svh_list[mat_index],
        pb->seg_counters + j * n_segs, warp_id, lane, core, tpb
    );
}

// =============================================================================
// Host dispatch
// =============================================================================

// EXL3_GEMV_MR_FUSE_IN=1: the single-matrix form rotates its input inside the
// dot kernel (exl3_gemv_mr_fused_single), one launch per call instead of two,
// whenever the rotated rows fit EXL3_GEMV_MR_FUSE_IN_MAX_KB of LDS per block
// (default 32; larger calls keep the two-kernel form). Read once.
static bool exl3_gemv_mr_fuse_in_enabled()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_GEMV_MR_FUSE_IN");
        return env && atoi(env) != 0;
    }();
    return on;
}

// EXL3_GEMV_MR_FUSE_IN_TRACE=1: log the first 200 single-matrix calls (shape, pointers, whether the
// output overlaps the input, which keeps the two-kernel form)
static bool exl3_gemv_mr_fuse_in_trace()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_GEMV_MR_FUSE_IN_TRACE");
        return env && atoi(env) != 0;
    }();
    return on;
}

static size_t exl3_gemv_mr_fuse_in_max_bytes()
{
    static const size_t v = []
    {
        const char* env = std::getenv("EXL3_GEMV_MR_FUSE_IN_MAX_KB");
        int kb = env ? atoi(env) : 8;
        kb = kb < 1 ? 1 : (kb > 64 ? 64 : kb);
        return (size_t) kb * 1024;
    }();
    return v;
}

// Graph patch sites of the single-matrix form, against the launched instance
static void exl3_gemv_mr_record_single(Graph* graph, void* k)
{
    graph->record_param(k, GP_gemm_A, 0);
    graph->record_param(k, GP_gemm_B_trellis, 1);
    graph->record_param(k, GP_gemm_C, 2);
    graph->record_param(k, GP_gemm_B_suh, 3);
    graph->record_param(k, GP_gemm_A_had, 4);
    graph->record_param(k, GP_gemm_B_svh, 5);
    graph->record_param(k, GP_end, 0);
}

// EXL3_GEMV_MR_FUSE_IN_CHECK=1 (diagnostic, synchronizes every call): run the fused form into a scratch
// output first, then the two-kernel form into C, and report calls whose outputs differ in any bit
static bool exl3_gemv_mr_fuse_in_check()
{
    static const bool on = []
    {
        const char* env = std::getenv("EXL3_GEMV_MR_FUSE_IN_CHECK");
        return env && atoi(env) != 0;
    }();
    return on;
}

static __global__ void exl3_gemv_mr_cmp_kernel(const uint16_t* a, const uint16_t* b, int64_t n, int* diff, int* first)
{
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        if (a[i] != b[i]) { atomicAdd(diff, 1); atomicMin(first, (int) i); }
}

static inline int exl3_gemv_mr_row_tile(int size_m)
{
    return size_m <= 1 ? 1 : (size_m <= 2 ? 2 : (size_m <= 4 ? 4 : 8));
}

// Split-K wave count: the m == 1 rule (shape-aware; EXL3_GEMV_SPLITK_WARPS
// overrides), which keeps row r of an m-row call bit-identical to the m == 1
// split-K result for the same shape. Above the m == 1 single-warp threshold
// (lm_head-scale outputs) the m == 1 path uses one warp per tile; here the
// block count alone saturates the device, so the narrowest split is used.
static inline int exl3_gemv_mr_warps(int size_k, int n_tiles, int bszm)
{
    if (n_tiles > EXL3_GEMV_SPLITK_MAX_TILES) return 4;
    return exl3_gemv_splitk_warps(size_k / 16, n_tiles, bszm);
}

#define LAUNCH_MR_M(bits_val, codebook, warps_val, m_tile, kernel_name, ...) \
    if (c_fp32) { \
        hipLaunchKernelGGL((kernel_name<bits_val, true, codebook, warps_val, m_tile>), \
            grid, block, smem, stream, __VA_ARGS__); \
        if (mr_rec) exl3_gemv_mr_record_single(mr_rec, \
            (void*) &kernel_name<bits_val, true, codebook, warps_val, m_tile>); \
    } else { \
        hipLaunchKernelGGL((kernel_name<bits_val, false, codebook, warps_val, m_tile>), \
            grid, block, smem, stream, __VA_ARGS__); \
        if (mr_rec) exl3_gemv_mr_record_single(mr_rec, \
            (void*) &kernel_name<bits_val, false, codebook, warps_val, m_tile>); \
    }
#define LAUNCH_MR_W(bits_val, codebook, warps_val, kernel_name, ...) \
    switch (m_tile) { \
        case 1: LAUNCH_MR_M(bits_val, codebook, warps_val, 1, kernel_name, __VA_ARGS__) break; \
        case 2: LAUNCH_MR_M(bits_val, codebook, warps_val, 2, kernel_name, __VA_ARGS__) break; \
        case 4: LAUNCH_MR_M(bits_val, codebook, warps_val, 4, kernel_name, __VA_ARGS__) break; \
        case 8: LAUNCH_MR_M(bits_val, codebook, warps_val, 8, kernel_name, __VA_ARGS__) break; \
    }
#define LAUNCH_MR_CB(bits_val, codebook, kernel_name, ...) \
    switch (warps) { \
        case 4:  LAUNCH_MR_W(bits_val, codebook, 4,  kernel_name, __VA_ARGS__) break; \
        case 8:  LAUNCH_MR_W(bits_val, codebook, 8,  kernel_name, __VA_ARGS__) break; \
        case 16: LAUNCH_MR_W(bits_val, codebook, 16, kernel_name, __VA_ARGS__) break; \
    }
#define LAUNCH_MR_K(bits_val, kernel_name, ...) \
    switch (cb) { \
        case 0: LAUNCH_MR_CB(bits_val, 0, kernel_name, __VA_ARGS__); break; \
        case 1: LAUNCH_MR_CB(bits_val, 1, kernel_name, __VA_ARGS__); break; \
        case 2: LAUNCH_MR_CB(bits_val, 2, kernel_name, __VA_ARGS__); break; \
    }
#define LAUNCH_MR(kernel_name, ...) \
    switch (K) { \
        case 1: LAUNCH_MR_K(1, kernel_name, __VA_ARGS__); break; \
        case 2: LAUNCH_MR_K(2, kernel_name, __VA_ARGS__); break; \
        case 3: LAUNCH_MR_K(3, kernel_name, __VA_ARGS__); break; \
        case 4: LAUNCH_MR_K(4, kernel_name, __VA_ARGS__); break; \
        case 5: LAUNCH_MR_K(5, kernel_name, __VA_ARGS__); break; \
        case 6: LAUNCH_MR_K(6, kernel_name, __VA_ARGS__); break; \
        case 7: LAUNCH_MR_K(7, kernel_name, __VA_ARGS__); break; \
        case 8: LAUNCH_MR_K(8, kernel_name, __VA_ARGS__); break; \
        case EXL3_HALF_BITS(1): LAUNCH_MR_CB(EXL3_HALF_BITS(1), 2, kernel_name, __VA_ARGS__); break; \
        case EXL3_HALF_BITS(2): LAUNCH_MR_CB(EXL3_HALF_BITS(2), 2, kernel_name, __VA_ARGS__); break; \
        case EXL3_HALF_BITS(3): LAUNCH_MR_CB(EXL3_HALF_BITS(3), 2, kernel_name, __VA_ARGS__); break; \
    }

bool exl3_gemv_multirow_try_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    if (size_m < exl3_gemv_mr_min_m(size_n / 16) || size_m > exl3_gemv_max_m()) return false;
    if (!exl3_gemv_mr_k_ok(K, cb)) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (!suh_ptr || !A_had_ptr || !svh_ptr) return false;
    if (size_n / 128 > EXL3_MGEMV_SEG_COUNTERS) return false;
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, graph == nullptr);
    if (!pb) return false;

    // Fused form: one launch, rotation into LDS (see exl3_gemv_mr_fused_single). Not when the output
    // overlaps the input: other blocks may still be reading A while the first ones store C
    const char* a_lo = (const char*) A_ptr;
    const char* a_hi = a_lo + (size_t) size_m * size_k * sizeof(half);
    const char* c_lo = (const char*) C_ptr;
    const char* c_hi = c_lo + (size_t) size_m * size_n * (c_fp32 ? sizeof(float) : sizeof(half));
    const bool c_over_a = c_lo < a_hi && a_lo < c_hi;
    if (exl3_gemv_mr_fuse_in_trace())
    {
        static std::mutex tmtx;
        static int seen = 0;
        std::lock_guard<std::mutex> lock(tmtx);
        if (seen < 200)
        {
            seen++;
            fprintf(stderr, " -- gemv_mr_fuse_in: dev %d m %d k %d n %d K %d fp32 %d A %p C %p A_had %p%s%s\n",
                    device, size_m, size_k, size_n, K, (int) c_fp32, (const void*) A_ptr, C_ptr,
                    (const void*) A_had_ptr, c_over_a ? " C_OVERLAPS_A" : "",
                    (const void*) A_had_ptr == (const void*) A_ptr ? " A_HAD_IS_A" : "");
        }
    }
    // Check mode: fused into scratch, then fall through to the two-kernel form into C, then compare
    void* chk_c = nullptr;
    int* chk_dev = nullptr;
    const size_t c_bytes = (size_t) (c_hi - c_lo);
    if (exl3_gemv_mr_fuse_in_check() && !graph && !c_over_a)
    {
        static void* scratch[64] = {};
        static size_t scratch_bytes[64] = {};
        static int* cnt[64] = {};
        if (scratch_bytes[device] < c_bytes)
        {
            if (scratch[device]) cudaFree(scratch[device]);
            cudaMalloc(&scratch[device], c_bytes);
            scratch_bytes[device] = c_bytes;
        }
        if (!cnt[device]) cudaMalloc((void**) &cnt[device], 2 * sizeof(int));
        chk_c = scratch[device];
        chk_dev = cnt[device];
    }
    if ((exl3_gemv_mr_fuse_in_enabled() || chk_c) && !c_over_a)
    {
        const int n_tiles = size_n / 16;
        const int warps = exl3_gemv_mr_warps(size_k, n_tiles, 1);
        const int m_tile = exl3_gemv_mr_row_tile(size_m);
        const size_t smem = exl3_gemv_mr_smem_bytes(warps, m_tile)
                          + (size_t) size_m * size_k * sizeof(half);
        if (smem <= exl3_gemv_mr_fuse_in_max_bytes())
        {
            const int core = exl3_gemv_core_mode();
            const int tpb = exl3_gemv_tiles_tpb(core, device, n_tiles, 1, warps);
            dim3 grid(n_tiles / tpb, 1);
            dim3 block(warps * 32);
            Graph* mr_rec = graph;
            void* c_dst = chk_c ? chk_c : C_ptr;
            LAUNCH_MR(exl3_gemv_mr_fused_single, A_ptr, B_ptr, c_dst, suh_ptr, A_had_ptr, svh_ptr,
                      pb, size_k, size_n, size_m, core, tpb)
            if (!chk_c) return true;
        }
        else chk_c = nullptr;   // the fused form declines this call: nothing to compare
    }

    // 1. Rotation + patch sites
    {
        int blocks = CEIL_DIVIDE(size_m * (size_k / 128) * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_gemv_mr_had_in_single,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, B_ptr, C_ptr, suh_ptr, A_had_ptr, svh_ptr, pb, size_k, size_m
        );
        if (graph)
        {
            void* k = (void*) exl3_gemv_mr_had_in_single;
            graph->record_param(k, GP_gemm_A, 0);
            graph->record_param(k, GP_gemm_B_trellis, 1);
            graph->record_param(k, GP_gemm_C, 2);
            graph->record_param(k, GP_gemm_B_suh, 3);
            graph->record_param(k, GP_gemm_A_had, 4);
            graph->record_param(k, GP_gemm_B_svh, 5);
            graph->record_param(k, GP_end, 0);
        }
    }

    // 2. GEMV, M rows per warp, fused output epilogue
    {
        const int n_tiles = size_n / 16;
        const int warps = exl3_gemv_mr_warps(size_k, n_tiles, 1);
        const int m_tile = exl3_gemv_mr_row_tile(size_m);
        const int core = exl3_gemv_core_mode();
        const int tpb = exl3_gemv_tiles_tpb(core, device, n_tiles, 1, warps);
        dim3 grid(n_tiles / tpb, 1);
        dim3 block(warps * 32);
        size_t smem = exl3_gemv_mr_smem_bytes(warps, m_tile);
        Graph* mr_rec = nullptr;
        LAUNCH_MR(exl3_gemv_mr_dot_single, pb, size_k, size_n, size_m, core, tpb)
    }
    if (chk_c)
    {
        int init[2] = { 0, 0x7fffffff };
        cudaMemcpyAsync(chk_dev, init, sizeof(init), cudaMemcpyHostToDevice, stream);
        hipLaunchKernelGGL(exl3_gemv_mr_cmp_kernel, dim3(64), dim3(256), 0, stream,
            (const uint16_t*) C_ptr, (const uint16_t*) chk_c, (int64_t) (c_bytes / 2), chk_dev, chk_dev + 1);
        int res[2];
        cudaMemcpyAsync(res, chk_dev, sizeof(res), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        static std::atomic<long> calls{0}, bad{0};
        long c_n = ++calls;
        if (res[0])
        {
            long b_n = ++bad;
            if (b_n <= 100)
                fprintf(stderr, " -- gemv_mr_fuse_in CHECK MISMATCH: dev %d m %d k %d n %d K %d cb %d fp32 %d: %d of %zu u16 differ, first %d (A %p C %p A_had %p), call %ld\n",
                        device, size_m, size_k, size_n, K, cb, (int) c_fp32, res[0], c_bytes / 2, res[1],
                        (const void*) A_ptr, C_ptr, (const void*) A_had_ptr, c_n);
        }
        if (c_n % 20000 == 0)
            fprintf(stderr, " -- gemv_mr_fuse_in CHECK: dev %d %ld calls, %ld mismatched\n", device, c_n, bad.load());
    }
    return true;
}

bool exl3_mgemv_multirow_try_launch
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list,
    void** c_list,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    if (size_m < exl3_gemv_mr_min_m(size_n / 16) || size_m > exl3_gemv_max_m()) return false;
    if (!exl3_gemv_mr_k_ok(K, cb)) return false;
    if (size_k % 128 || size_n % 128) return false;
    // Weighted calls (MoE down) at m == 1 with the mgemv epilogue; see the header
    const bool weighted = weights_ptr != nullptr;
    if (weighted && (size_m != 1 || !exl3_gemv_mr_weighted_enabled())) return false;
    if (weighted)
    {
        // The weighted form stands in for exl3_mgemv_try_launch, so it honors that
        // path's kill switch (EXL3_MGEMV=0 pins the cooperative kernel)
        const char* mg = std::getenv("EXL3_MGEMV");
        if (mg && atoi(mg) == 0) return false;
    }
    if (num_tokens < 1 || (num_tokens != 1 && min_index >= 0)) return false;
    if (weighted && size_n_list) return false;
    if (weighted && (!exl3_gemv_fuse_out_enabled() || size_n / 128 > EXL3_MGEMV_RED_COUNTERS)) return false;
    if ((size_n_list != nullptr) != (c_list != nullptr)) return false;
    if (size_n_list && min_index >= 0) return false;
    if (!suh_ptr_ptr || !A_had_ptr || !svh_ptr_ptr) return false;
    int bszm = MAX(bszm_in, bszm_out);
    if (bszm < 1 || bszm > 128) return false;
    if (bszm * (size_n / 128) > EXL3_MGEMV_SEG_COUNTERS) return false;
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, graph == nullptr);
    if (!pb) return false;

    // 1. Rotation + patch sites
    {
        int blocks = CEIL_DIVIDE(bszm * size_m * (size_k / 128) * 32, 256);
        hipLaunchKernelGGL
        (
            exl3_gemv_mr_had_in_multi,
            dim3(blocks), dim3(256), 0, stream,
            A_ptr, C_ptr, indices_ptr, weights_ptr, pb, suh_ptr_ptr, A_had_ptr,
            size_k, size_m, bszm_in, bszm, min_index, max_index
        );
        if (graph)
        {
            void* k = (void*) exl3_gemv_mr_had_in_multi;
            graph->record_param(k, GP_mgemm_A, 0);
            graph->record_param(k, GP_mgemm_C, 1);
            graph->record_param(k, GP_mgemm_indices, 2);
            graph->record_param(k, GP_mgemm_weights, 3);
            graph->record_param(k, GP_end, 0);
        }
    }

    // 2. GEMV
    {
        const int n_tiles = size_n / 16;
        // Multi-token calls (num_tokens groups of slots, the batched MoE route) size
        // the split-K from one token's slot count, so every slot keeps the reduction
        // order of a single-token call on the same experts (bit-identical rows)
        const int bszm_rule = (num_tokens > 1 && bszm % num_tokens == 0) ? bszm / num_tokens : bszm;
        const int warps = exl3_gemv_mr_warps(size_k, n_tiles, bszm_rule);
        const int m_tile = exl3_gemv_mr_row_tile(size_m);
        const int core = exl3_gemv_core_mode();
        const int tpb = exl3_gemv_tiles_tpb(core, device, n_tiles, bszm, warps);
        dim3 grid(n_tiles / tpb, bszm);
        dim3 block(warps * 32);
        size_t smem = exl3_gemv_mr_smem_bytes(warps, m_tile);
        Graph* mr_rec = nullptr;
        LAUNCH_MR(exl3_gemv_mr_dot_multi, pb, B_ptr_ptr, svh_ptr_ptr, A_had_ptr,
                  size_k, size_n, size_m, bszm, min_index, max_index, size_n_list, c_list, core, tpb,
                  num_tokens, weighted ? 1 : 0)
    }
    return true;
}

#undef LAUNCH_MR
#undef LAUNCH_MR_K
#undef LAUNCH_MR_CB
#undef LAUNCH_MR_W
#undef LAUNCH_MR_M

// =============================================================================
// Routed-MoE decode in four launches: exl3_rocm::moe_decode
// =============================================================================
//
// The per-token MoE route runs a routed expert block as exl3_mgemm(gate) +
// exl3_mgemm(up) + silu_mul + exl3_mgemm(down, weighted): seven dependent
// launches per layer at m == 1 (each mgemm is a rotation + a dot kernel here),
// and every one of them costs a command-processor gap on top of its run time --
// decode is GPU-bound (the host runs ahead), so launches are paid in GPU time.
// This op runs the same arithmetic in four:
//
//   1. exl3_moe_dec_had_gu    rotate each token's input once per (slot, gate|up)
//                             into yh -- slot j < S is gate[sel[j]], j >= S is
//                             up[sel[j - S]]; token = (j mod S) / top_k, read in
//                             place (no replicated input rows at bsz > 1)
//   2. exl3_moe_dec_dot       gate and up as one 2S-slot split-K GEMV (the
//                             multi-row body, row tile 1, fused output rotation)
//   3. exl3_moe_dec_had_act   silu(gate) * up in registers (act_mul_kernel_h's
//                             half2 arithmetic and act_limit clamp, verbatim),
//                             rotated with the down projection's suh into A_had
//   4. exl3_moe_dec_dot       down, weighted, with the mgemv fused epilogue's
//                             grouped reduce: token t's routed sum -> out row t
//
// Every value is the one the four-mgemm route produces: the rotations, dot core
// and epilogues are the same functions; the split-K wave count is sized from one
// token's slot count per matrix (top_k), exactly the rule the separate calls
// apply; the activation is elementwise in the same half2 ops. Pointer tables:
// gate and up concatenated (2E entries, built once at load). SILU with a gated
// MLP only; the caller falls back to the mgemm route otherwise.

// Expert-parallel slot mapping: a global pick -> local table index, or -1 for a pick on another rank
// (min_e < 0: no range, the pick is already local)
__device__ __forceinline__ int exl3_moe_local(int64_t e, int min_e, int max_e)
{
    if (min_e < 0) return (int) e;
    return (e >= min_e && e < max_e) ? (int) (e - min_e) : -1;
}

__device__ __forceinline__ half2 exl3_moe_silu_h2(half2 x)   // activation_kernels.cuh _silu(half2)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}

// exl3_gemv_had_in_128 with the 128 input halves already in registers (v)
__device__ __forceinline__ void exl3_moe_had_in_128_v
(
    half4 v,
    half* output_ptr,
    const half* __restrict__ scale,
    const int lane
)
{
    half4 scales = ((const half4*) scale)[lane];
    v.x = exl3_pin_h2(__hmul2(v.x, scales.x));
    v.y = exl3_pin_h2(__hmul2(v.y, scales.y));

    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    const float r_scale = 0.088388347648f;  // 1/sqrt(128)
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    ((half4*) output_ptr)[lane] = v;
}

static __global__
__launch_bounds__(256)
void exl3_moe_dec_had_gu
(
    const half* __restrict__ y,            // (bsz, size_k)
    const int64_t* __restrict__ sel,       // (S)
    const uintptr_t* __restrict__ suh_gu,  // (2E)
    half* __restrict__ yh,                 // (2S, size_k)
    const int size_k,
    const int S,
    const int top_k,
    const int E,
    const int min_e,
    const int max_e
)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= 2 * S * k_segs) return;
    const int j = warp_id / k_segs;
    const int seg = warp_id % k_segs;
    const int js = j < S ? j : j - S;
    const int e_loc = exl3_moe_local(sel[js], min_e, max_e);
    if (e_loc < 0) return;   // warp-uniform: a pick on another rank
    const int mat = e_loc + (j < S ? 0 : E);
    const int token = js / top_k;
    const half* suh = (const half*) suh_gu[mat];
    exl3_gemv_had_in_128(y + (int64_t) token * size_k + seg * 128,
                         yh + (int64_t) j * size_k + seg * 128, suh + seg * 128, lane);
}

static __global__
__launch_bounds__(256)
void exl3_moe_dec_had_act
(
    const half* __restrict__ gu,           // (2S, size_k): gate rows, then up rows
    const int64_t* __restrict__ sel,       // (S)
    const uintptr_t* __restrict__ suh_d,   // (E)
    half* __restrict__ a_had,              // (S, size_k)
    const int size_k,
    const int S,
    const float act_limit,
    const int min_e,
    const int max_e
)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int k_segs = size_k / 128;
    if (warp_id >= S * k_segs) return;
    const int j = warp_id / k_segs;
    const int seg = warp_id % k_segs;
    const int64_t off = (int64_t) j * size_k + seg * 128;
    half4 g = ((const half4*) (gu + off))[lane];
    half4 u = ((const half4*) (gu + (int64_t) S * size_k + off))[lane];

    // act_mul_kernel_h<ACT_SILU>, per half2
    half2 xg[2] = { g.x, g.y };
    half2 yu[2] = { u.x, u.y };
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        half2 x2 = exl3_moe_silu_h2(xg[i]);
        half2 y2 = yu[i];
        if (act_limit != 0.0f)
        {
            y2 = __hmax2(y2, __float2half2_rn(-act_limit));
            y2 = __hmin2(y2, __float2half2_rn(act_limit));
            x2 = __hmin2(x2, __float2half2_rn(act_limit));
        }
        xg[i] = __hmul2(x2, y2);
    }
    half4 a;
    a.x = xg[0];
    a.y = xg[1];
    const int e_loc = exl3_moe_local(sel[j], min_e, max_e);
    if (e_loc < 0) return;   // warp-uniform: a pick on another rank
    const half* suh = (const half*) suh_d[e_loc];
    exl3_moe_had_in_128_v(a, a_had + off, suh + seg * 128, lane);
}

// Gate|up (weighted = false: 2S slots, table split at S) or down (weighted: S slots)
template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_moe_dec_dot
(
    Exl3GemvMrParams* __restrict__ pb,
    const uintptr_t* __restrict__ B_list,
    const uintptr_t* __restrict__ svh_list,
    const int64_t* __restrict__ sel,
    const half* __restrict__ weights,      // (S) for the weighted form, else nullptr
    const half* __restrict__ A_had,        // (slots, size_k)
    void* __restrict__ C,                  // (slots, size_n)
    const int size_k,
    const int size_n,
    const int S,
    const int E,
    const int num_tokens,
    const int core,
    const int tpb,
    const int min_e,
    const int max_e
)
{
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int j = blockIdx.y;
    const int tile_n = blockIdx.x * tpb;
    const int n_tiles = size_n / 16;
    const int n_segs = size_n / 128;
    const int js = j < S ? j : j - S;
    const int e_loc = exl3_moe_local(sel[js], min_e, max_e);
    const int64_t row_off = (int64_t) j * size_n;
    if (e_loc < 0)   // block-uniform: a pick on another rank
    {
        // Weighted (down): the slot still arrives at the grouped reduce as an all-zero row,
        // exl3_gemv_mr_dot_multi's rule for skipped slots. Unweighted (gate|up): nothing to do
        if (weights && warp_id == 0)
            for (int t = 0; t < tpb; ++t)
                exl3_gemv_mr_zero_epilogue<c_fp32>
                (
                    C, row_off, size_n, tile_n + t, lane,
                    pb->seg_counters + j * n_segs + (tile_n + t) / 8,
                    pb->red_counters + (tile_n + t) / 8, S, num_tokens
                );
        return;
    }
    const int mat = e_loc + (j < S ? 0 : E);

    if (weights)
    {
        const float w_scale = 0.088388347648f * __half2float(weights[j]);
        exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, 1>
        (
            A_had + (int64_t) j * size_k, size_k, (const uint16_t*) B_list[mat], size_k, n_tiles, tile_n,
            C, row_off, size_n, 1, (const half*) svh_list[mat],
            pb->seg_counters + j * n_segs, warp_id, lane, core, tpb,
            w_scale, pb->red_counters, S, size_n, num_tokens
        );
    }
    else
    {
        exl3_gemv_mr_body<bits, c_fp32, cb, WARPS_PER_BLOCK, 1>
        (
            A_had + (int64_t) j * size_k, size_k, (const uint16_t*) B_list[mat], size_k, n_tiles, tile_n,
            C, row_off, size_n, 1, (const half*) svh_list[mat],
            pb->seg_counters + j * n_segs, warp_id, lane, core, tpb
        );
    }
}

static void exl3_moe_dec_launch_dot
(
    Exl3GemvMrParams* pb, const uintptr_t* B_list, const uintptr_t* svh_list, const int64_t* sel,
    const half* weights, const half* A_had, void* C, int size_k, int size_n, int S, int E,
    int num_tokens, int slots, int top_k, int K, int cb, bool c_fp32, int device, cudaStream_t stream,
    int min_e, int max_e
)
{
    const int n_tiles = size_n / 16;
    // One token's slots per matrix: the rule the separate per-matrix calls apply
    const int warps = exl3_gemv_mr_warps(size_k, n_tiles, top_k);
    const int core = exl3_gemv_core_mode();
    const int tpb = exl3_gemv_tiles_tpb(core, device, n_tiles, slots, warps);
    dim3 grid(n_tiles / tpb, slots);
    dim3 block(warps * 32);
    size_t smem = exl3_gemv_mr_smem_bytes(warps, 1);

    #define MOE_DEC_L3(bits_val, codebook, warps_val) \
        if (c_fp32) hipLaunchKernelGGL((exl3_moe_dec_dot<bits_val, true, codebook, warps_val>), grid, block, smem, stream, \
            pb, B_list, svh_list, sel, weights, A_had, C, size_k, size_n, S, E, num_tokens, core, tpb, min_e, max_e); \
        else hipLaunchKernelGGL((exl3_moe_dec_dot<bits_val, false, codebook, warps_val>), grid, block, smem, stream, \
            pb, B_list, svh_list, sel, weights, A_had, C, size_k, size_n, S, E, num_tokens, core, tpb, min_e, max_e);
    #define MOE_DEC_L2(bits_val, codebook) \
        switch (warps) { \
            case 4:  MOE_DEC_L3(bits_val, codebook, 4)  break; \
            case 8:  MOE_DEC_L3(bits_val, codebook, 8)  break; \
            case 16: MOE_DEC_L3(bits_val, codebook, 16) break; \
        }
    #define MOE_DEC_L1(bits_val) \
        switch (cb) { \
            case 0: MOE_DEC_L2(bits_val, 0) break; \
            case 1: MOE_DEC_L2(bits_val, 1) break; \
            case 2: MOE_DEC_L2(bits_val, 2) break; \
        }
    switch (K)
    {
        case 1: MOE_DEC_L1(1) break;
        case 2: MOE_DEC_L1(2) break;
        case 3: MOE_DEC_L1(3) break;
        case 4: MOE_DEC_L1(4) break;
        case 5: MOE_DEC_L1(5) break;
        case 6: MOE_DEC_L1(6) break;
        case 7: MOE_DEC_L1(7) break;
        case 8: MOE_DEC_L1(8) break;
        case EXL3_HALF_BITS(1): MOE_DEC_L2(EXL3_HALF_BITS(1), 2) break;
        case EXL3_HALF_BITS(2): MOE_DEC_L2(EXL3_HALF_BITS(2), 2) break;
        case EXL3_HALF_BITS(3): MOE_DEC_L2(EXL3_HALF_BITS(3), 2) break;
    }
    #undef MOE_DEC_L1
    #undef MOE_DEC_L2
    #undef MOE_DEC_L3
}

// Whether exl3_rocm_moe_decode takes a call of S slots (bsz * top_k) at these widths: the same bounds the
// op checks, for callers that fall back instead of raising
bool exl3_rocm_moe_decode_fits(int S, int Hi, int I, int Ho, int K_gu, int cb_gu, int K_d, int cb_d)
{
    return Hi % 128 == 0 && I % 128 == 0 && Ho % 128 == 0
        && 2 * S <= 128 && 2 * S * (I / 128) <= EXL3_MGEMV_SEG_COUNTERS && Ho / 128 <= EXL3_MGEMV_RED_COUNTERS
        && S * (Ho / 128) <= EXL3_MGEMV_SEG_COUNTERS
        && I / 16 <= EXL3_GEMV_SPLITK_MAX_TILES && Ho / 16 <= EXL3_GEMV_SPLITK_MAX_TILES
        && cb_gu >= 0 && cb_gu <= 2 && cb_d >= 0 && cb_d <= 2
        && exl3_gemv_mr_k_ok(K_gu, cb_gu) && exl3_gemv_mr_k_ok(K_d, cb_d);
}

// cb: 0 = 3inst, 1 = mcg, 2 = mul1 (exl3_mgemm_gr's encoding). K: integer 1..8, or
// EXL3_HALF_BITS(k) = 16 + k for a k + 0.5 bpw mul1 matrix (the caller passes it only for
// half-rate layers, and only while EXL3_ROCM_HALF_GEMV is on)
void exl3_rocm_moe_decode_ep
(
    const at::Tensor& y,            // (bsz, Hi) half
    const at::Tensor& sel,          // (bsz, top_k) int64
    const at::Tensor& weights,      // (bsz, top_k) half
    const at::Tensor& gu_trellis,   // (2E) int64 pointer tables, gate then up
    const at::Tensor& gu_suh,
    const at::Tensor& gu_svh,
    const at::Tensor& d_trellis,    // (E)
    const at::Tensor& d_suh,
    const at::Tensor& d_svh,
    at::Tensor& yh,                 // (>= 2S, Hi) half scratch
    at::Tensor& gu,                 // (>= 2S, I) half scratch
    at::Tensor& a_had,              // (>= S, I) half scratch
    at::Tensor& out,                // (>= S, Ho) float: rows 0..bsz-1 = routed sums
    int64_t K_gu,
    int64_t cb_gu,
    int64_t K_d,
    int64_t cb_d,
    double act_limit,
    int64_t min_expert,
    int64_t max_expert
)
{
    const int min_e = (int) min_expert;
    const int max_e = (int) max_expert;
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(weights, kHalf);
    TORCH_CHECK_DTYPE(yh, kHalf);
    TORCH_CHECK_DTYPE(gu, kHalf);
    TORCH_CHECK_DTYPE(a_had, kHalf);
    TORCH_CHECK_DTYPE(out, kFloat);
    TORCH_CHECK(y.is_contiguous() && sel.is_contiguous() && weights.is_contiguous(), "moe_decode: inputs must be contiguous");
    TORCH_CHECK(yh.is_contiguous() && gu.is_contiguous() && a_had.is_contiguous() && out.is_contiguous(), "moe_decode: buffers must be contiguous");

    const int bsz = (int) y.size(0);
    const int Hi = (int) y.size(-1);
    const int top_k = (int) sel.size(-1);
    const int S = bsz * top_k;
    const int E = (int) d_trellis.size(0);
    const int I = (int) gu.size(-1);
    const int Ho = (int) out.size(-1);
    TORCH_CHECK(sel.numel() == S && weights.numel() == S, "moe_decode: sel/weights shape");
    TORCH_CHECK(gu_trellis.size(0) == 2 * E && gu_suh.size(0) == 2 * E && gu_svh.size(0) == 2 * E, "moe_decode: gate|up tables must hold 2E entries");
    TORCH_CHECK(Hi % 128 == 0 && I % 128 == 0 && Ho % 128 == 0, "moe_decode: dims must be multiples of 128");
    TORCH_CHECK(yh.numel() >= (int64_t) 2 * S * Hi && gu.numel() >= (int64_t) 2 * S * I
                && a_had.numel() >= (int64_t) S * I && out.numel() >= (int64_t) S * Ho, "moe_decode: scratch too small");
    TORCH_CHECK(2 * S <= 128 && 2 * S * (I / 128) <= EXL3_MGEMV_SEG_COUNTERS && Ho / 128 <= EXL3_MGEMV_RED_COUNTERS
                && S * (Ho / 128) <= EXL3_MGEMV_SEG_COUNTERS, "moe_decode: shape exceeds the arrival counters");
    TORCH_CHECK(I / 16 <= EXL3_GEMV_SPLITK_MAX_TILES && Ho / 16 <= EXL3_GEMV_SPLITK_MAX_TILES, "moe_decode: output too wide");
    TORCH_CHECK(cb_gu >= 0 && cb_gu <= 2 && cb_d >= 0 && cb_d <= 2
                && exl3_gemv_mr_k_ok((int) K_gu, (int) cb_gu) && exl3_gemv_mr_k_ok((int) K_d, (int) cb_d), "moe_decode: K/cb");

    int device = y.get_device();
    Exl3GemvMrParams* pb = exl3_gemv_mr_param_block(device, true);
    TORCH_CHECK(pb, "moe_decode: parameter block allocation failed");

    const int64_t* sel_p = (const int64_t*) sel.data_ptr();

    // 1. gate|up input rotation
    {
        int blocks = CEIL_DIVIDE(2 * S * (Hi / 128) * 32, 256);
        hipLaunchKernelGGL(exl3_moe_dec_had_gu, dim3(blocks), dim3(256), 0, stream,
            (const half*) y.data_ptr(), sel_p, (const uintptr_t*) gu_suh.data_ptr(),
            (half*) yh.data_ptr(), Hi, S, top_k, E, min_e, max_e);
    }
    // 2. gate|up GEMV, 2S slots
    exl3_moe_dec_launch_dot(pb, (const uintptr_t*) gu_trellis.data_ptr(), (const uintptr_t*) gu_svh.data_ptr(),
        sel_p, nullptr, (const half*) yh.data_ptr(), gu.data_ptr(), Hi, I, S, E, 1, 2 * S, top_k,
        (int) K_gu, (int) cb_gu, false, device, stream, min_e, max_e);
    // 3. activation + down input rotation
    {
        int blocks = CEIL_DIVIDE(S * (I / 128) * 32, 256);
        hipLaunchKernelGGL(exl3_moe_dec_had_act, dim3(blocks), dim3(256), 0, stream,
            (const half*) gu.data_ptr(), sel_p, (const uintptr_t*) d_suh.data_ptr(),
            (half*) a_had.data_ptr(), I, S, (float) act_limit, min_e, max_e);
    }
    // 4. down GEMV, weighted, grouped per token
    exl3_moe_dec_launch_dot(pb, (const uintptr_t*) d_trellis.data_ptr(), (const uintptr_t*) d_svh.data_ptr(),
        sel_p, (const half*) weights.data_ptr(), (const half*) a_had.data_ptr(), out.data_ptr(), I, Ho, S, 0, bsz,
        S, top_k, (int) K_d, (int) cb_d, true, device, stream, min_e, max_e);
    cuda_check(cudaPeekAtLastError());
}


// The no-range form (single GPU, or a caller that already passes local ids)
void exl3_rocm_moe_decode
(
    const at::Tensor& y,
    const at::Tensor& sel,
    const at::Tensor& weights,
    const at::Tensor& gu_trellis,
    const at::Tensor& gu_suh,
    const at::Tensor& gu_svh,
    const at::Tensor& d_trellis,
    const at::Tensor& d_suh,
    const at::Tensor& d_svh,
    at::Tensor& yh,
    at::Tensor& gu,
    at::Tensor& a_had,
    at::Tensor& out,
    int64_t K_gu,
    int64_t cb_gu,
    int64_t K_d,
    int64_t cb_d,
    double act_limit
)
{
    exl3_rocm_moe_decode_ep(y, sel, weights, gu_trellis, gu_suh, gu_svh, d_trellis, d_suh, d_svh,
                            yh, gu, a_had, out, K_gu, cb_gu, K_d, cb_d, act_limit, -1, -1);
}


// =============================================================================
// Shared-expert merge for the RDNA MoE route: dst[t] = routed[t] + g[t] * sh[t], g[t] = sigmoid(y[t] . w)
// (w absent: g = 1). One block per token; the gate dot in fp32 like at::matmul on the fp32 copies
// =============================================================================

static __global__
__launch_bounds__(256)
void exl3_moe_merge_shared_kernel
(
    float* __restrict__ dst, const int64_t ld_dst,
    const float* __restrict__ routed, const int64_t ld_r,
    const float* __restrict__ sh, const int64_t ld_s,
    const half* __restrict__ y, const int64_t ld_y, const int k,
    const half* __restrict__ w,
    const int H
)
{
    const int t = blockIdx.x;
    __shared__ float red[8];
    __shared__ float g_sh;
    float g = 1.0f;
    if (w)
    {
        float acc = 0.0f;
        for (int h = threadIdx.x; h < k; h += blockDim.x)
            acc += __half2float(y[(int64_t) t * ld_y + h]) * __half2float(w[h]);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 32);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            float s = 0.0f;
            for (int i = 0; i < (int) (blockDim.x >> 5); ++i) s += red[i];
            g_sh = 1.0f / (1.0f + expf(-s));
        }
        __syncthreads();
        g = g_sh;
    }
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        dst[(int64_t) t * ld_dst + c] = routed[(int64_t) t * ld_r + c] + g * sh[(int64_t) t * ld_s + c];
}

void exl3_rocm_moe_merge_shared
(
    at::Tensor& dst,                        // (bsz, H) float, row-strided
    const at::Tensor& routed,               // (bsz, H) float, row-strided
    const at::Tensor& sh,                   // (bsz, H) float, row-strided
    const at::Tensor& y,                    // (bsz, k) half, contiguous rows
    const c10::optional<at::Tensor>& gate_w // (k) half, or none
)
{
    const at::cuda::OptionalCUDAGuard device_guard(y.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(dst, kFloat);
    TORCH_CHECK_DTYPE(routed, kFloat);
    TORCH_CHECK_DTYPE(sh, kFloat);
    TORCH_CHECK_DTYPE(y, kHalf);
    TORCH_CHECK(dst.stride(1) == 1 && routed.stride(1) == 1 && sh.stride(1) == 1 && y.stride(1) == 1, "moe_merge_shared: rows must be contiguous");
    const int bsz = (int) dst.size(0);
    const int H = (int) dst.size(1);
    const int k = (int) y.size(1);
    const half* w = nullptr;
    if (gate_w)
    {
        TORCH_CHECK_DTYPE(gate_w.value(), kHalf);
        TORCH_CHECK(gate_w.value().numel() == k && gate_w.value().is_contiguous(), "moe_merge_shared: gate weight shape");
        w = (const half*) gate_w.value().data_ptr();
    }
    if (bsz < 1) return;
    hipLaunchKernelGGL(exl3_moe_merge_shared_kernel, dim3(bsz), dim3(256), 0, stream,
        (float*) dst.data_ptr(), dst.stride(0), (const float*) routed.data_ptr(), routed.stride(0),
        (const float*) sh.data_ptr(), sh.stride(0), (const half*) y.data_ptr(), y.stride(0), k, w, H);
    cuda_check(cudaPeekAtLastError());
}
