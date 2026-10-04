#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dflash2.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/codebook.cuh"
#include "quant/bits_k.cuh"

// DFlash2 grouped dynamic causal convolution over a draft block (dflash.model
// _grouped_dynamic_convolve):
//
//     out[b, t, c] = sum_k (base[k, c] + dyn[b, t, k, c / group_size]) * x[b, t - k, c]
//
// Taps reach back within the block only (t - k < 0 contributes nothing), so the kernel is
// stateless across rounds. dyn is the kernel projection's output viewed as [b, l, K, groups]
// with arbitrary strides (the projection emits both the prepare and the finish deltas in one
// tensor). accumulate: out (fp32 residual stream) += conv(x), fusing the block's residual add
// into the finish() variant; otherwise out = conv(x)

#define NUM_THREADS 256

template <typename T> __device__ __forceinline__ float to_f(T v) { return (float) v; }
template <> __device__ __forceinline__ float to_f(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v) { return (T) v; }
template <> __device__ __forceinline__ half from_f(float v) { return __float2half(v); }

template <typename TX, typename TB, typename TO, bool accumulate>
__global__ __launch_bounds__(NUM_THREADS)
void dflash2_dynconv_kernel
(
    const TX* __restrict__ x,
    const half* __restrict__ dyn,
    const TB* __restrict__ base,
    TO* __restrict__ out,
    const int seqlen,
    const int hidden,
    const int group_size,
    const int taps,
    const int64_t ds_b,
    const int64_t ds_t,
    const int64_t ds_k,
    const int64_t ds_g
)
{
    int c = blockIdx.x * NUM_THREADS + threadIdx.x;
    if (c >= hidden) return;
    int t = blockIdx.y;
    int b = blockIdx.z;
    int g = c / group_size;

    const half* dyn_bt = dyn + b * ds_b + t * ds_t + g * ds_g;
    const TX* x_bt = x + ((int64_t) b * seqlen + t) * hidden + c;

    float acc = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < taps; ++k)
    {
        if (k > t) break;
        float w = to_f(base[k * hidden + c]) + __half2float(dyn_bt[k * ds_k]);
        acc += w * to_f(x_bt[-(int64_t) k * hidden]);
    }

    TO* o = out + ((int64_t) b * seqlen + t) * hidden + c;
    if constexpr (accumulate)
        *o = from_f<TO>(to_f(*o) + acc);
    else
        *o = from_f<TO>(acc);
}

template <typename TX, typename TB, typename TO, bool accumulate>
void launch
(
    const at::Tensor& x, const at::Tensor& dyn, const at::Tensor& base, at::Tensor& out,
    int seqlen, int hidden, int group_size, int taps, int bsz, cudaStream_t stream
)
{
    dim3 grid(CEIL_DIVIDE(hidden, NUM_THREADS), seqlen, bsz);
    dflash2_dynconv_kernel<TX, TB, TO, accumulate><<<grid, NUM_THREADS, 0, stream>>>
    (
        (const TX*) x.data_ptr(),
        (const half*) dyn.data_ptr(),
        (const TB*) base.data_ptr(),
        (TO*) out.data_ptr(),
        seqlen, hidden, group_size, taps,
        dyn.stride(0), dyn.stride(1), dyn.stride(2), dyn.stride(3)
    );
}

/*
x:      (bsz, seqlen, hidden), fp16 or fp32, contiguous
dyn:    (bsz, seqlen, taps, hidden / group_size), fp16, any strides
base:   (taps, hidden), fp16 or bf16, contiguous
out:    (bsz, seqlen, hidden), fp16 or fp32, contiguous; accumulated into when accumulate
*/

void dflash2_dynconv
(
    const at::Tensor& x,
    const at::Tensor& dyn,
    const at::Tensor& base,
    at::Tensor& out,
    int64_t group_size,
    bool accumulate
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(x, 3);
    TORCH_CHECK_DIM(dyn, 4);
    TORCH_CHECK_DIM(base, 2);
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && base.is_contiguous(), "dflash2_dynconv: x, out and base must be contiguous");
    TORCH_CHECK_SHAPES_FULL(x, out);
    TORCH_CHECK_DTYPE(dyn, kHalf);
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kFloat, "dflash2_dynconv: x must be fp16 or fp32");
    TORCH_CHECK(out.dtype() == at::kHalf || out.dtype() == at::kFloat, "dflash2_dynconv: out must be fp16 or fp32");
    TORCH_CHECK(base.dtype() == at::kHalf || base.dtype() == at::kBFloat16, "dflash2_dynconv: base must be fp16 or bf16");
    TORCH_CHECK(!accumulate || out.dtype() == at::kFloat, "dflash2_dynconv: accumulate needs an fp32 residual");

    int bsz = x.size(0);
    int seqlen = x.size(1);
    int hidden = x.size(2);
    int taps = base.size(0);
    TORCH_CHECK(group_size > 0 && hidden % group_size == 0, "dflash2_dynconv: hidden must be a multiple of group_size");
    TORCH_CHECK(base.size(1) == hidden, "dflash2_dynconv: base width mismatch");
    TORCH_CHECK(dyn.size(0) == bsz && dyn.size(1) == seqlen && dyn.size(2) == taps && dyn.size(3) == hidden / group_size,
                "dflash2_dynconv: dyn must be (bsz, seqlen, taps, hidden / group_size)");
    if (!bsz || !seqlen || !hidden) return;

    #define DISPATCH(TX, TB, TO) \
        if (accumulate) launch<TX, TB, TO, true>(x, dyn, base, out, seqlen, hidden, (int) group_size, taps, bsz, stream); \
        else            launch<TX, TB, TO, false>(x, dyn, base, out, seqlen, hidden, (int) group_size, taps, bsz, stream);
    #define DISPATCH_TO(TX, TB) \
        if (out.dtype() == at::kHalf) { DISPATCH(TX, TB, half) } else { DISPATCH(TX, TB, float) }
    #define DISPATCH_TB(TX) \
        if (base.dtype() == at::kHalf) { DISPATCH_TO(TX, half) } else { DISPATCH_TO(TX, __nv_bfloat16) }

    if (x.dtype() == at::kHalf) { DISPATCH_TB(half) } else { DISPATCH_TB(float) }

    #undef DISPATCH_TB
    #undef DISPATCH_TO
    #undef DISPATCH
    cuda_check(cudaPeekAtLastError());
}


// DFlash2 candidate selector walk (dflash.model CandidateSelector, greedy). Per batch row, over
// the block's draft rows i = 0..rows-1, conditioned on the previously chosen token a (the
// verified anchor for row 0):
//
//     score[c] = unary[i, c] + < A[a] * gate[i], B[cands[i, c]] >     (dot over rank)
//     a = cands[i, argmax_c score[c]]                                 (first max on ties)
//
// One block per batch row runs the whole chain: the per-row work (k dots of length rank) is far
// too small to spread over the grid, and keeping it in one block removes the per-row host round
// trip and the ~6 launches per row of the torch formulation. out[:, 0] gets the anchor and
// out[:, 1:] the path, matching the module's [anchor, path...] layout; conf (optional) gets the
// winning score per row with a leading zero

#define WALK_THREADS 256

template <typename TG, typename TC>
__global__ __launch_bounds__(WALK_THREADS)
void dflash2_selector_walk_kernel
(
    const float* __restrict__ unary,        // [bsz, rows, k]
    const int64_t* __restrict__ cands,      // [bsz, rows, k]
    const TG* __restrict__ gate,            // [bsz, rows, rank]
    const TC* __restrict__ pred_cb,         // [vocab, rank]
    const TC* __restrict__ succ_cb,         // [vocab, rank]
    const int64_t* __restrict__ anchor,     // [bsz]
    int64_t* __restrict__ out,              // [bsz, rows + 1]
    float* __restrict__ conf,               // [bsz, rows + 1] or nullptr
    const int rows,
    const int k,
    const int rank
)
{
    extern __shared__ float smem[];
    float* a_g = smem;                      // [rank]: A[pred] * gate[i]
    float* scores = smem + rank;            // [k]
    __shared__ int64_t s_pred;

    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t / 32, lane = t % 32;

    if (t == 0)
    {
        s_pred = anchor[b];
        out[(int64_t) b * (rows + 1)] = s_pred;
        if (conf) conf[(int64_t) b * (rows + 1)] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < rows; ++i)
    {
        const int64_t pred = s_pred;
        const TC* a_row = pred_cb + pred * rank;
        const TG* g_row = gate + ((int64_t) b * rows + i) * rank;
        for (int r = t; r < rank; r += WALK_THREADS)
            a_g[r] = to_f(a_row[r]) * to_f(g_row[r]);
        __syncthreads();

        const int64_t* c_row = cands + ((int64_t) b * rows + i) * k;
        const float* u_row = unary + ((int64_t) b * rows + i) * k;
        for (int c = warp; c < k; c += WALK_THREADS / 32)
        {
            const TC* b_row = succ_cb + c_row[c] * rank;
            float dot = 0.0f;
            for (int r = lane; r < rank; r += 32)
                dot += a_g[r] * to_f(b_row[r]);
            for (int offset = 16; offset > 0; offset /= 2)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            if (lane == 0) scores[c] = u_row[c] + dot;
        }
        __syncthreads();

        if (t == 0)
        {
            int best = 0;
            float best_score = scores[0];
            for (int c = 1; c < k; ++c)
                if (scores[c] > best_score) { best_score = scores[c]; best = c; }
            s_pred = c_row[best];
            out[(int64_t) b * (rows + 1) + i + 1] = s_pred;
            if (conf) conf[(int64_t) b * (rows + 1) + i + 1] = best_score;
        }
        __syncthreads();
    }
}

/*
unary:   (bsz, rows, k) fp32 top-k draft logits per row, contiguous
cands:   (bsz, rows, k) int64 token ids of those logits, contiguous
gate:    (bsz, rows, rank) fp16 or fp32 projected draft states, contiguous
pred_cb, succ_cb: (vocab, rank) fp16 or bf16 codebooks, contiguous (same dtype)
anchor:  (bsz,) int64 verified anchor token per row
out:     (bsz, rows + 1) int64 -> [anchor, path...]
conf:    (bsz, rows + 1) fp32 -> [0, winning score...] (optional)
*/

void dflash2_selector_walk
(
    const at::Tensor& unary,
    const at::Tensor& cands,
    const at::Tensor& gate,
    const at::Tensor& pred_cb,
    const at::Tensor& succ_cb,
    const at::Tensor& anchor,
    at::Tensor& out,
    const c10::optional<at::Tensor>& conf
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unary.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(unary, 3);
    TORCH_CHECK_DIM(cands, 3);
    TORCH_CHECK_DIM(gate, 3);
    TORCH_CHECK_DIM(pred_cb, 2);
    TORCH_CHECK_DIM(succ_cb, 2);
    TORCH_CHECK_DIM(anchor, 1);
    TORCH_CHECK_DIM(out, 2);
    TORCH_CHECK_DTYPE(unary, kFloat);
    TORCH_CHECK_DTYPE(cands, kLong);
    TORCH_CHECK_DTYPE(anchor, kLong);
    TORCH_CHECK_DTYPE(out, kLong);
    TORCH_CHECK_DTYPE_OPT(conf, kFloat);
    TORCH_CHECK(gate.dtype() == at::kHalf || gate.dtype() == at::kFloat, "dflash2_selector_walk: gate must be fp16 or fp32");
    TORCH_CHECK(pred_cb.dtype() == succ_cb.dtype() && (pred_cb.dtype() == at::kHalf || pred_cb.dtype() == at::kBFloat16),
                "dflash2_selector_walk: codebooks must both be fp16 or both bf16");
    TORCH_CHECK(unary.is_contiguous() && cands.is_contiguous() && gate.is_contiguous() && pred_cb.is_contiguous() &&
                succ_cb.is_contiguous() && anchor.is_contiguous() && out.is_contiguous(), "dflash2_selector_walk: inputs must be contiguous");

    int bsz = unary.size(0);
    int rows = unary.size(1);
    int k = unary.size(2);
    int rank = gate.size(2);
    TORCH_CHECK_SHAPES_FULL(unary, cands);
    TORCH_CHECK(gate.size(0) == bsz && gate.size(1) == rows, "dflash2_selector_walk: gate must be (bsz, rows, rank)");
    TORCH_CHECK(pred_cb.size(1) == rank && succ_cb.sizes() == pred_cb.sizes(), "dflash2_selector_walk: codebooks must be (vocab, rank)");
    TORCH_CHECK(anchor.size(0) == bsz, "dflash2_selector_walk: anchor must be (bsz,)");
    TORCH_CHECK(out.size(0) == bsz && out.size(1) == rows + 1, "dflash2_selector_walk: out must be (bsz, rows + 1)");
    TORCH_CHECK(!conf.has_value() || (conf.value().is_contiguous() && conf.value().sizes() == out.sizes()),
                "dflash2_selector_walk: conf must be (bsz, rows + 1), contiguous");
    TORCH_CHECK(k >= 1 && rank >= 1, "dflash2_selector_walk: empty candidate list or rank");
    if (!bsz || !rows) return;

    size_t smem = (rank + k) * sizeof(float);
    float* conf_ptr = conf.has_value() ? (float*) conf.value().data_ptr() : nullptr;

    #define LAUNCH(TG, TC) \
        dflash2_selector_walk_kernel<TG, TC><<<bsz, WALK_THREADS, smem, stream>>> \
        ( \
            (const float*) unary.data_ptr(), (const int64_t*) cands.data_ptr(), (const TG*) gate.data_ptr(), \
            (const TC*) pred_cb.data_ptr(), (const TC*) succ_cb.data_ptr(), (const int64_t*) anchor.data_ptr(), \
            (int64_t*) out.data_ptr(), conf_ptr, rows, k, rank \
        );
    if (gate.dtype() == at::kHalf) { if (pred_cb.dtype() == at::kHalf) { LAUNCH(half, half) } else { LAUNCH(half, __nv_bfloat16) } }
    else                           { if (pred_cb.dtype() == at::kHalf) { LAUNCH(float, half) } else { LAUNCH(float, __nv_bfloat16) } }
    #undef LAUNCH
    cuda_check(cudaPeekAtLastError());
}


// ============================ quantized (integer) codebooks ============================
//
// A codebook may be stored as a per-32-block integer tensor: packed integer values
// [vocab, qbytes] uint8 + per-block fp16 scales [vocab, rank/32] (+ per-block fp16 mins for the
// asymmetric _1 forms). The serving gather reads one vocab row, dequantizes it to fp16
// (qv * scale + min) and hands the raw row to the staged walk. Row-based (not tile-based), so a
// single row is a contiguous read with no cross-row traffic:
//
//     Q8_0: 8-bit signed,   per-block scale          (qv in [-128,127], out = qv * d)
//     Q4_1: 4-bit unsigned, per-block scale + min    (qv in [0,15],     out = qv * d + m)
//     Q3_1: 3-bit unsigned, per-block scale + min    (qv in [0,7],      out = qv * d + m)
//     Q2_1: 2-bit unsigned, per-block scale + min    (qv in [0,3],      out = qv * d + m)
//
// The q buffer is padded by one byte per row so the Q3 cross-byte read never runs past the row.

#define CB_RANK 256           // codebook rank (WALK_THREADS wide)
#define CB_BLOCKS (CB_RANK / 32)   // 8 per-32 blocks per row

template <int FORMAT>
__global__ __launch_bounds__(CB_RANK)
void dflash2_cb_gather_int_kernel
(
    const uint8_t* __restrict__ q,        // [vocab, qbytes+1]
    const half* __restrict__ scales,      // [vocab, CB_BLOCKS]
    const half* __restrict__ mins,        // [vocab, CB_BLOCKS] (null for Q8_0)
    const int64_t* __restrict__ ids,      // [n]
    half* __restrict__ out,               // [n, CB_RANK]
    const int qstride                     // qbytes + 1
)
{
    const int row = blockIdx.x;
    const int c = threadIdx.x;            // 0..CB_RANK-1
    const int64_t v = ids[row];
    const int blk = c >> 5;               // c / 32
    const float d = __half2float(scales[v * CB_BLOCKS + blk]);
    const float m = mins ? __half2float(mins[v * CB_BLOCKS + blk]) : 0.0f;
    const uint8_t* qr = q + v * qstride;
    float qv;
    if constexpr (FORMAT == 0)            // Q8_0: 8-bit signed
        qv = (float) (int8_t) qr[c];
    else if constexpr (FORMAT == 1)       // Q4_1: 4-bit unsigned
        qv = (float) ((qr[c >> 1] >> (4 * (c & 1))) & 0xF);
    else if constexpr (FORMAT == 2)       // Q4_0: 4-bit signed
    {
        const int u = (qr[c >> 1] >> (4 * (c & 1))) & 0xF;
        qv = (float) (u >= 8 ? u - 16 : u);
    }
    else if constexpr (FORMAT == 3)       // Q3_1: 3-bit unsigned
    {
        const int bit = 3 * c;
        const uint16_t two = (uint16_t) (qr[bit >> 3] | (qr[(bit >> 3) + 1] << 8));
        qv = (float) ((two >> (bit & 7)) & 0x7);
    }
    else if constexpr (FORMAT == 4)       // Q3_0: 3-bit signed
    {
        const int bit = 3 * c;
        const uint16_t two = (uint16_t) (qr[bit >> 3] | (qr[(bit >> 3) + 1] << 8));
        const int u = (two >> (bit & 7)) & 0x7;
        qv = (float) (u >= 4 ? u - 8 : u);
    }
    else if constexpr (FORMAT == 5)       // Q2_1: 2-bit unsigned
        qv = (float) ((qr[c >> 2] >> (2 * (c & 3))) & 0x3);
    else                                  // FORMAT == 6: Q2_0: 2-bit signed
    {
        const int u = (qr[c >> 2] >> (2 * (c & 3))) & 0x3;
        qv = (float) (u >= 2 ? u - 4 : u);
    }
    out[(int64_t) row * CB_RANK + c] = __float2half_rn(qv * d + m);
}

/*
q:      (vocab, qbytes+1) uint8 packed integer values (1 pad byte per row)
scales: (vocab, rank/32) fp16 per-block scales
mins:   (vocab, rank/32) fp16 per-block mins (optional; absent for Q8_0)
ids:    (n,) int64 rows to gather
out:    (n, rank) fp16 dequantized rows
*/
void dflash2_cb_gather_int
(
    const at::Tensor& q,
    const at::Tensor& scales,
    const c10::optional<at::Tensor>& mins,
    const at::Tensor& ids,
    at::Tensor& out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(q, 2);
    TORCH_CHECK_DIM(scales, 2);
    TORCH_CHECK_DIM(ids, 1);
    TORCH_CHECK_DIM(out, 2);
    TORCH_CHECK_DTYPE(q, kByte);
    TORCH_CHECK_DTYPE(scales, kHalf);
    TORCH_CHECK_DTYPE_OPT(mins, kHalf);
    TORCH_CHECK_DTYPE(ids, kLong);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(q.is_contiguous() && scales.is_contiguous() && ids.is_contiguous() && out.is_contiguous(),
                "dflash2_cb_gather_int: inputs must be contiguous");
    TORCH_CHECK(out.size(0) == ids.numel() && out.size(1) == CB_RANK,
                "dflash2_cb_gather_int: out must be (n, ", CB_RANK, ")");
    TORCH_CHECK(scales.size(1) == CB_BLOCKS,
                "dflash2_cb_gather_int: scales must be (vocab, ", CB_BLOCKS, ")");

    const int64_t n = ids.numel();
    if (!n) return;

    const int qstride = (int) q.size(1);
    const int qbytes = qstride - 1;
    const half* mins_p = mins.has_value() ? (const half*) mins.value().data_ptr() : nullptr;

    #define GATHER_INT(F) \
        dflash2_cb_gather_int_kernel<F><<<n, CB_RANK, 0, stream>>>( \
            (const uint8_t*) q.data_ptr(), (const half*) scales.data_ptr(), mins_p, \
            (const int64_t*) ids.data_ptr(), (half*) out.data_ptr(), qstride);
    switch (qbytes)
    {
        case 256: GATHER_INT(0) break;                                  // Q8_0
        case 128: if (mins_p) GATHER_INT(1) else GATHER_INT(2) break;   // Q4_1 / Q4_0
        case 96:  if (mins_p) GATHER_INT(3) else GATHER_INT(4) break;   // Q3_1 / Q3_0
        case 64:  if (mins_p) GATHER_INT(5) else GATHER_INT(6) break;   // Q2_1 / Q2_0
        default: TORCH_CHECK(false, "dflash2_cb_gather_int: unsupported codebook width ", qbytes);
    }
    #undef GATHER_INT
    cuda_check(cudaPeekAtLastError());
}


// Staged variant of the selector walk for quantized codebooks. The Python side dequantizes the rows the
// walk can touch (dflash2_cb_gather_int; both id sets are known before the walk starts) and passes them
// here; only the per-position pred row selection stays in-kernel (slot addressing below). The integer
// gather folds the per-32-block scale+min into each row, so the staged rows are the full fp16 codebook
// values and the walk's column scale svh_AB is a no-op (ones). The fp32 walk score, per row i:
//
//     score[c] = unary[i, c] + < p_a * (gate[i] * svh_AB), p_b[c] >
//
// stagedA:   [bsz, 1 + (rows-1)*k, rank] fp16, pred_cb rows: slot 0 = anchor, slot 1 + (i-1)*k + c = row
//            of cands[i-1, c]
// stagedB:   [bsz, rows, k, rank] fp16, succ_cb rows
// svh_AB:    [rank] fp16, ones (the integer dequant already folded the per-block scale+min in)

template <typename TG>
__global__ __launch_bounds__(WALK_THREADS)
void dflash2_selector_walk_staged_kernel
(
    const float* __restrict__ unary,
    const int64_t* __restrict__ cands,
    const TG* __restrict__ gate,
    const half* __restrict__ stagedA,
    const half* __restrict__ stagedB,
    const half* __restrict__ svh_AB,
    const int64_t* __restrict__ anchor,
    int64_t* __restrict__ out,
    float* __restrict__ conf,
    const int rows,
    const int k
)
{
    extern __shared__ float smem[];               // [CB_RANK + k]: a_g + scores
    float* a_g = smem;                            // [CB_RANK]
    float* scores = smem + CB_RANK;               // [k]
    __shared__ int64_t s_pred;
    __shared__ int s_slot;

    const int b = blockIdx.x;
    const int t = threadIdx.x;
    const int warp = t / 32, lane = t % 32;
    const int a_slots = 1 + (rows - 1) * k;

    if (t == 0)
    {
        s_pred = anchor[b];
        s_slot = 0;
        out[(int64_t) b * (rows + 1)] = s_pred;
        if (conf) conf[(int64_t) b * (rows + 1)] = 0.0f;
    }
    __syncthreads();

    const half* stA = stagedA + (int64_t) b * a_slots * CB_RANK;
    const half* stB = stagedB + (int64_t) b * rows * k * CB_RANK;

    for (int i = 0; i < rows; ++i)
    {
        const half* a_row = stA + (int64_t) s_slot * CB_RANK;
        const TG* g_row = gate + ((int64_t) b * rows + i) * CB_RANK;
        a_g[t] = to_f(a_row[t]) * to_f(g_row[t]) * to_f(svh_AB[t]);
        __syncthreads();

        const int64_t* c_row = cands + ((int64_t) b * rows + i) * k;
        const float* u_row = unary + ((int64_t) b * rows + i) * k;
        for (int c = warp; c < k; c += WALK_THREADS / 32)
        {
            const half* b_row = stB + ((int64_t) i * k + c) * CB_RANK;
            float dot = 0.0f;
            for (int r = lane; r < CB_RANK; r += 32)
                dot += a_g[r] * to_f(b_row[r]);
            for (int offset = 16; offset > 0; offset /= 2)
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            if (lane == 0) scores[c] = u_row[c] + dot;
        }
        __syncthreads();

        if (t == 0)
        {
            int best = 0;
            float best_score = scores[0];
            for (int c = 1; c < k; ++c)
                if (scores[c] > best_score) { best_score = scores[c]; best = c; }
            s_pred = c_row[best];
            s_slot = 1 + i * k + best;
            out[(int64_t) b * (rows + 1) + i + 1] = s_pred;
            if (conf) conf[(int64_t) b * (rows + 1) + i + 1] = best_score;
        }
        __syncthreads();
    }
}

/*
unary, cands, gate, anchor, out, conf: as dflash2_selector_walk
stagedA, stagedB, svh_AB: see dflash2_selector_walk_staged_kernel
*/

void dflash2_selector_walk_staged
(
    const at::Tensor& unary,
    const at::Tensor& cands,
    const at::Tensor& gate,
    const at::Tensor& stagedA,
    const at::Tensor& stagedB,
    const at::Tensor& svh_AB,
    const at::Tensor& anchor,
    at::Tensor& out,
    const c10::optional<at::Tensor>& conf
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unary.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(unary, 3);
    TORCH_CHECK_DIM(cands, 3);
    TORCH_CHECK_DIM(gate, 3);
    TORCH_CHECK_DIM(stagedA, 3);
    TORCH_CHECK_DIM(stagedB, 4);
    TORCH_CHECK_DIM(svh_AB, 1);
    TORCH_CHECK_DIM(anchor, 1);
    TORCH_CHECK_DIM(out, 2);
    TORCH_CHECK_DTYPE(unary, kFloat);
    TORCH_CHECK_DTYPE(cands, kLong);
    TORCH_CHECK_DTYPE(anchor, kLong);
    TORCH_CHECK_DTYPE(out, kLong);
    TORCH_CHECK_DTYPE_OPT(conf, kFloat);
    TORCH_CHECK_DTYPE(stagedA, kHalf);
    TORCH_CHECK_DTYPE(stagedB, kHalf);
    TORCH_CHECK_DTYPE(svh_AB, kHalf);
    TORCH_CHECK(gate.dtype() == at::kHalf || gate.dtype() == at::kFloat, "dflash2_selector_walk_staged: gate must be fp16 or fp32");
    TORCH_CHECK(unary.is_contiguous() && cands.is_contiguous() && gate.is_contiguous() && stagedA.is_contiguous() &&
                stagedB.is_contiguous() && svh_AB.is_contiguous() && anchor.is_contiguous() && out.is_contiguous(),
                "dflash2_selector_walk_staged: inputs must be contiguous");

    int bsz = unary.size(0);
    int rows = unary.size(1);
    int k = unary.size(2);
    TORCH_CHECK_SHAPES_FULL(unary, cands);
    TORCH_CHECK(gate.size(0) == bsz && gate.size(1) == rows && gate.size(2) == CB_RANK,
                "dflash2_selector_walk_staged: gate must be (bsz, rows, ", CB_RANK, ")");
    TORCH_CHECK(svh_AB.numel() == CB_RANK, "dflash2_selector_walk_staged: svh_AB must be (", CB_RANK, ",)");
    TORCH_CHECK(stagedA.size(0) == bsz && stagedA.size(1) == 1 + (rows - 1) * k && stagedA.size(2) == CB_RANK,
                "dflash2_selector_walk_staged: stagedA must be (bsz, 1 + (rows-1)*k, rank)");
    TORCH_CHECK(stagedB.size(0) == bsz && stagedB.size(1) == rows && stagedB.size(2) == k && stagedB.size(3) == CB_RANK,
                "dflash2_selector_walk_staged: stagedB must be (bsz, rows, k, rank)");
    TORCH_CHECK(anchor.size(0) == bsz, "dflash2_selector_walk_staged: anchor must be (bsz,)");
    TORCH_CHECK(out.size(0) == bsz && out.size(1) == rows + 1, "dflash2_selector_walk_staged: out must be (bsz, rows + 1)");
    TORCH_CHECK(!conf.has_value() || (conf.value().is_contiguous() && conf.value().sizes() == out.sizes()),
                "dflash2_selector_walk_staged: conf must be (bsz, rows + 1), contiguous");
    TORCH_CHECK(k >= 1 && rows >= 1, "dflash2_selector_walk_staged: empty candidate list or rows");
    if (!bsz || !rows) return;

    size_t smem = (CB_RANK + k) * sizeof(float);
    float* conf_ptr = conf.has_value() ? (float*) conf.value().data_ptr() : nullptr;

    #define LAUNCH(TG) \
        dflash2_selector_walk_staged_kernel<TG><<<bsz, WALK_THREADS, smem, stream>>> \
        ( \
            (const float*) unary.data_ptr(), (const int64_t*) cands.data_ptr(), (const TG*) gate.data_ptr(), \
            (const half*) stagedA.data_ptr(), (const half*) stagedB.data_ptr(), (const half*) svh_AB.data_ptr(), \
            (const int64_t*) anchor.data_ptr(), \
            (int64_t*) out.data_ptr(), conf_ptr, rows, k \
        );
    if (gate.dtype() == at::kHalf) { LAUNCH(half) } else { LAUNCH(float) }
    #undef LAUNCH
    cuda_check(cudaPeekAtLastError());
}


// Top-k over the draft block's logits for the selector, one launch for all rows (torch.topk
// spends ~18 launches on its radix select). Each row is split over blocks of 256 threads x E
// elements; every thread holds its E elements in registers and tracks its running maximum,
// and a warp produces its top-K in K rounds of warp argmax where only the winning lane drops
// its element and rescans (E predicated compares). Warp 0 merges the 8 warp lists the same
// way, and the last-arriving block of a row merges the block partials. Per-thread sorted lists
// were tried first and cost ~4x the instructions (a 16-deep shift per insert). scale and
// softcap (Gemma-class heads) are applied on the fly: both are monotonic, so the selection
// equals transforming the whole row first, and only the kept values transform. Output order is
// descending; ties resolve to the lowest lane / earliest element

#define TOPK_THREADS 256

// x[N] per lane, ids id[N]; K rounds; lane 0 writes round k's winner to out_v[k] / out_ix[k].
// Consumes the lanes' elements (winners are replaced by -inf)
template <int N, int K>
__device__ __forceinline__ void warp_topk(float (&x)[N], int (&id)[N], float* out_v, int* out_ix)
{
    const int lane = threadIdx.x % 32;
    float cur = -INFINITY; int arg = 0;
    #pragma unroll
    for (int i = 0; i < N; ++i) if (x[i] > cur) { cur = x[i]; arg = i; }
    for (int k = 0; k < K; ++k)
    {
        float best = cur; int best_lane = lane;
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float o_best = __shfl_xor_sync(0xffffffff, best, offset);
            int o_lane = __shfl_xor_sync(0xffffffff, best_lane, offset);
            if (o_best > best || (o_best == best && o_lane < best_lane)) { best = o_best; best_lane = o_lane; }
        }
        int best_id = -1;
        #pragma unroll
        for (int i = 0; i < N; ++i) if (i == arg) best_id = id[i];
        best_id = __shfl_sync(0xffffffff, best_id, best_lane);
        if (lane == 0) { out_v[k] = best; out_ix[k] = best_id; }
        if (lane == best_lane)
        {
            #pragma unroll
            for (int i = 0; i < N; ++i) if (i == arg) x[i] = -INFINITY;
            cur = -INFINITY; arg = 0;
            #pragma unroll
            for (int i = 0; i < N; ++i) if (x[i] > cur) { cur = x[i]; arg = i; }
        }
    }
}

// Block top-K: warp lists into shared memory, warp 0 merges them (8 * K entries, K / 4 per
// lane). Two barriers
template <int N, int K>
__device__ __forceinline__ void block_topk(float (&x)[N], int (&id)[N], float* out_v, int* out_ix, float* s_wv, int* s_wix)
{
    constexpr int M = K / 4;    // (TOPK_THREADS / 32) * K / 32 entries per lane in the second stage
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    warp_topk<N, K>(x, id, s_wv + warp * K, s_wix + warp * K);
    __syncthreads();
    if (warp == 0)
    {
        float y[M]; int yid[M];
        #pragma unroll
        for (int i = 0; i < M; ++i) { y[i] = s_wv[lane * M + i]; yid[i] = s_wix[lane * M + i]; }
        warp_topk<M, K>(y, yid, out_v, out_ix);
    }
    __syncthreads();
}

template <typename T, int K, int E>
__global__ __launch_bounds__(TOPK_THREADS)
void dflash2_topk_kernel
(
    const T* __restrict__ logits,
    const int64_t stride_b,
    const int64_t stride_r,
    const int rows,
    const int splits,
    const int vocab,
    const float scale,
    const float softcap,
    float* __restrict__ part_v,       // [bsz * rows, splits, K] scratch
    int* __restrict__ part_ix,        // [bsz * rows, splits, K] scratch
    int* __restrict__ counters,       // [bsz * rows], zero on entry, zero on exit
    float* __restrict__ values,       // [bsz, rows, K]
    int64_t* __restrict__ indices     // [bsz, rows, K]
)
{
    constexpr int VEC = 16 / sizeof(T);        // elements per 16-byte load
    constexpr int LOADS = E / VEC;             // vector loads per thread
    constexpr int SPAN = TOPK_THREADS * VEC;   // elements covered by one load across the block
    static_assert(E % VEC == 0, "E must be a multiple of the vector width");

    __shared__ float s_wv[(TOPK_THREADS / 32) * K];
    __shared__ int s_wix[(TOPK_THREADS / 32) * K];
    __shared__ bool s_last;

    const int row_id = blockIdx.x / splits;      // b * rows + r
    const int split = blockIdx.x % splits;
    const int b = row_id / rows, r = row_id % rows;
    const int t = threadIdx.x;
    const T* row = logits + b * stride_b + r * stride_r;
    const int c0 = split * (TOPK_THREADS * E);

    // This thread's E elements: LOADS coalesced 16-byte loads, each covering SPAN elements
    // across the block. Out-of-range elements read as -inf
    float x[E]; int id[E];
    #pragma unroll
    for (int l = 0; l < LOADS; ++l)
    {
        const int c = c0 + l * SPAN + t * VEC;
        if (c + VEC <= vocab)
        {
            int4 raw = *((const int4*) (row + c));
            const T* xs = (const T*) &raw;
            #pragma unroll
            for (int i = 0; i < VEC; ++i)
            {
                float xv = to_f(xs[i]) * scale;
                if (softcap > 0.0f) xv = tanhf(xv / softcap) * softcap;
                x[l * VEC + i] = xv; id[l * VEC + i] = c + i;
            }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < VEC; ++i)
            {
                float xv = -INFINITY;
                if (c + i < vocab)
                {
                    xv = to_f(row[c + i]) * scale;
                    if (softcap > 0.0f) xv = tanhf(xv / softcap) * softcap;
                }
                x[l * VEC + i] = xv; id[l * VEC + i] = c + i;
            }
        }
    }

    // Block partial -> scratch
    float* pv = part_v + ((int64_t) row_id * splits + split) * K;
    int* pix = part_ix + ((int64_t) row_id * splits + split) * K;
    block_topk<E, K>(x, id, pv, pix, s_wv, s_wix);

    // Last-arriving block of the row merges the partials (splits * K entries, M per thread)
    __threadfence();
    if (t == 0)
    {
        int arrived = atomicAdd(&counters[row_id], 1);
        s_last = arrived == splits - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();

    constexpr int M = K / 4;    // holds splits * K entries when splits <= 64
    float y[M]; int yid[M];
    const float* rv = part_v + (int64_t) row_id * splits * K;
    const int* rix = part_ix + (int64_t) row_id * splits * K;
    #pragma unroll
    for (int i = 0; i < M; ++i)
    {
        int j = t * M + i;
        bool ok = j < splits * K;
        y[i] = ok ? __ldcg(rv + j) : -INFINITY;
        yid[i] = ok ? __ldcg(rix + j) : -1;
    }
    __shared__ float o_v[K];
    __shared__ int o_ix[K];
    block_topk<M, K>(y, yid, o_v, o_ix, s_wv, s_wix);
    if (t < K)
    {
        int64_t o = (int64_t) row_id * K + t;
        values[o] = o_v[t];
        indices[o] = o_ix[t];
    }
    if (t == 0) counters[row_id] = 0;
}

/*
logits:  (bsz, rows, >= vocab) fp16 or fp32, unit stride along the vocab dim (views allowed)
vocab:   number of valid columns
values:  (bsz, rows, K) fp32, indices: (bsz, rows, K) int64; K in {8, 16, 32}
*/

void dflash2_topk
(
    const at::Tensor& logits,
    int64_t vocab,
    double scale,
    double softcap,
    at::Tensor& values,
    at::Tensor& indices
)
{
    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(logits, 3);
    TORCH_CHECK_DIM(values, 3);
    TORCH_CHECK_DIM(indices, 3);
    TORCH_CHECK_DTYPE(values, kFloat);
    TORCH_CHECK_DTYPE(indices, kLong);
    TORCH_CHECK(logits.dtype() == at::kHalf || logits.dtype() == at::kFloat, "dflash2_topk: logits must be fp16 or fp32");
    TORCH_CHECK(logits.stride(2) == 1, "dflash2_topk: logits must have unit stride along the vocab dim");
    TORCH_CHECK(values.is_contiguous() && indices.is_contiguous(), "dflash2_topk: outputs must be contiguous");
    TORCH_CHECK_SHAPES_FULL(values, indices);

    int bsz = logits.size(0);
    int rows = logits.size(1);
    int k = values.size(2);
    TORCH_CHECK(values.size(0) == bsz && values.size(1) == rows, "dflash2_topk: output shape mismatch");
    TORCH_CHECK(vocab >= k && vocab <= logits.size(2), "dflash2_topk: vocab out of range");
    TORCH_CHECK(k == 8 || k == 16 || k == 32, "dflash2_topk: k must be 8, 16 or 32");
    if (!bsz || !rows) return;

    // Elements per thread: 16 covers rows up to 64 splits * 4096 = 262144 columns (the final
    // merge holds splits * K entries at K / 4 per thread), 32 up to 524288
    int E = vocab <= 64 * TOPK_THREADS * 16 ? 16 : 32;
    TORCH_CHECK(vocab <= 64 * TOPK_THREADS * 32, "dflash2_topk: vocab too large (max 524288)");
    TORCH_CHECK(((uintptr_t) logits.data_ptr() & 15) == 0 && (logits.stride(0) * logits.element_size()) % 16 == 0 &&
                (logits.stride(1) * logits.element_size()) % 16 == 0, "dflash2_topk: logits rows must be 16-byte aligned");
    int splits = CEIL_DIVIDE(vocab, TOPK_THREADS * E);
    int num_rows = bsz * rows;
    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(logits.device());
    auto opts_i = at::TensorOptions().dtype(at::kInt).device(logits.device());
    at::Tensor part_v = at::empty({(int64_t) num_rows * splits * k}, opts_f);
    at::Tensor part_ix = at::empty({(int64_t) num_rows * splits * k}, opts_i);
    at::Tensor counters = at::zeros({num_rows}, opts_i);

    #define LAUNCH(T, K, E) \
        dflash2_topk_kernel<T, K, E><<<num_rows * splits, TOPK_THREADS, 0, stream>>> \
        ((const T*) logits.data_ptr(), logits.stride(0), logits.stride(1), rows, splits, (int) vocab, (float) scale, (float) softcap, \
         (float*) part_v.data_ptr(), (int*) part_ix.data_ptr(), (int*) counters.data_ptr(), \
         (float*) values.data_ptr(), (int64_t*) indices.data_ptr());
    #define LAUNCH_E(T, K) if (E == 16) { LAUNCH(T, K, 16) } else { LAUNCH(T, K, 32) }
    #define LAUNCH_K(T) \
        if (k == 8) { LAUNCH_E(T, 8) } else if (k == 16) { LAUNCH_E(T, 16) } else { LAUNCH_E(T, 32) }
    if (logits.dtype() == at::kHalf) { LAUNCH_K(half) } else { LAUNCH_K(float) }
    #undef LAUNCH_K
    #undef LAUNCH_E
    #undef LAUNCH
    cuda_check(cudaPeekAtLastError());
}
