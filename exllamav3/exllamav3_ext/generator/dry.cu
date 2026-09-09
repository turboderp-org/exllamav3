#include <cuda_fp16.h>
#include "dry.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"

// DRY repetition penalty (llama_sampler_dry_apply semantics, single-token breakers), one launch
// per sampling step. Grid is (blocks per row, rows):
//
//   1. every block finds rep_limit, the distance from the end of the window to the nearest
//      breaker (matches cannot extend past it; llama.cpp gives up entirely when it is below
//      allowed_length);
//   2. every block scans a strided slice of offsets k: the run of tokens the window suffix
//      shares with the sequence ending k tokens earlier, capped at min(rep_limit, cap) where
//      cap is the length beyond which the clamped exponent no longer changes. The token that
//      followed the earlier sequence is charged with the longest such run (atomicMax into the
//      per-row workspace);
//   3. the last block to finish a row (atomic counter) writes the output logits: input minus
//      multiplier * base ** min(len - allowed_length, max_exponent) for charged, non-breaker
//      tokens, plain copy otherwise.
//
// The workspace (bsz * dim match lengths) and the counters (bsz) are one temporary int32
// buffer filled with -1 by the caller, so the counters count up from -1. Penalties are computed
// in double and rounded once to float, matching the torch reference; a penalty beyond float
// range lands the logit at -inf.

// TODO: Profiling and optimization

#define NUM_THREADS 1024
#define MAX_SCAN_BLOCKS 32

template <bool input_fp16>
__global__ __launch_bounds__(NUM_THREADS)
void dry_penalty_kernel
(
    const void* __restrict__ in_logits,
    float* __restrict__ out_logits,
    const int64_t* __restrict__ past_ids,
    const int past_len,
    const bool* __restrict__ breakers,
    int* __restrict__ workspace,
    unsigned int* __restrict__ counters,
    const int dim,
    const float multiplier,
    const float base,
    const int allowed_length,
    const int range,
    const int max_exponent,
    const int match_cap
)
{
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int m = range > 0 ? MIN(past_len, range) : past_len;
    // reversed window: r(j) is the j-th most recent token
    const int64_t* window = past_ids + (size_t) row * past_len + (past_len - m);
    #define R(j) (window[m - 1 - (j)])
    int* lmax = workspace + (size_t) row * dim;

    __shared__ int s_rep_limit;
    __shared__ bool s_last;

    if (m > allowed_length)
    {
        if (tid == 0) s_rep_limit = m;
        __syncthreads();
        if (breakers)
        {
            int local = m;
            for (int j = tid; j < m; j += NUM_THREADS)
            {
                int64_t t = R(j);
                if (t >= 0 && t < dim && breakers[t]) { local = j; break; }
            }
            if (local < m) atomicMin(&s_rep_limit, local);
        }
        __syncthreads();
        const int rep_limit = s_rep_limit;

        if (rep_limit >= allowed_length)
        {
            int cap = max_exponent > 0 ? allowed_length + max_exponent : match_cap;
            cap = MIN(cap, rep_limit);
            const int stride = gridDim.x * NUM_THREADS;
            for (int k = 1 + blockIdx.x * NUM_THREADS + tid; k < m; k += stride)
            {
                int len = 0;
                const int lim = MIN(cap, m - k);
                while (len < lim && R(len) == R(len + k)) ++len;
                if (len >= allowed_length)
                {
                    int64_t follower = R(k - 1);
                    if (follower >= 0 && follower < dim)
                        atomicMax(lmax + follower, len);
                }
            }
        }
    }

    // Last block for the row applies the penalties
    __threadfence();
    __syncthreads();
    if (tid == 0)
    {
        int prev = (int) atomicAdd(counters + row, 1u);     // counts up from -1
        s_last = (prev + 1 == (int) gridDim.x - 1);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();

    const size_t row_off = (size_t) row * dim;
    for (int v = tid; v < dim; v += NUM_THREADS)
    {
        int len = lmax[v];
        float x;
        if constexpr (input_fp16)
            x = __half2float(((const half*) in_logits)[row_off + v]);
        else
            x = ((const float*) in_logits)[row_off + v];
        if (len >= allowed_length && !(breakers && breakers[v]))
        {
            int e = len - allowed_length;
            if (max_exponent > 0) e = MIN(e, max_exponent);
            double pen = (double) multiplier * pow((double) base, (double) e);
            x -= (float) pen;
        }
        out_logits[row_off + v] = x;
    }
    #undef R
}

void dry_penalty
(
    const at::Tensor& in_logits,
    const at::Tensor& out_logits,
    const at::Tensor& past_ids,
    const std::optional<at::Tensor>& breakers,
    const at::Tensor& workspace,
    const at::Tensor& counters,
    float multiplier,
    float base,
    int allowed_length,
    int range,
    int max_exponent,
    int match_cap
)
{
    const at::cuda::OptionalCUDAGuard device_guard(in_logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(out_logits, kFloat);
    TORCH_CHECK_DTYPE(past_ids, kLong);
    TORCH_CHECK_DTYPE(workspace, kInt);
    TORCH_CHECK_DTYPE(counters, kInt);
    TORCH_CHECK_SHAPES_FULL(in_logits, out_logits);
    TORCH_CHECK_SHAPES(past_ids, 0, in_logits, 0, 1);
    TORCH_CHECK(in_logits.dim() == 2 && past_ids.dim() == 2, "dry_penalty: expected (bsz, dim) logits and (bsz, len) ids");
    TORCH_CHECK(in_logits.is_contiguous() && out_logits.is_contiguous() && past_ids.is_contiguous(), "dry_penalty: inputs must be contiguous");
    TORCH_CHECK(past_ids.device() == in_logits.device(), "dry_penalty: past_ids must be on the logits device");
    int bsz = in_logits.size(0);
    int dim = in_logits.size(1);
    int past_len = past_ids.size(1);
    TORCH_CHECK(workspace.numel() >= (int64_t) bsz * dim && counters.numel() >= bsz, "dry_penalty: workspace too small");
    const bool* breakers_ptr = nullptr;
    if (breakers.has_value())
    {
        TORCH_CHECK_DTYPE(breakers.value(), kBool);
        TORCH_CHECK(breakers.value().numel() == dim, "dry_penalty: breakers mask must cover the vocabulary");
        breakers_ptr = (const bool*) breakers.value().data_ptr();
    }
    int m = range > 0 ? MIN(past_len, range) : past_len;
    int scan_blocks = MIN(MAX_SCAN_BLOCKS, MAX(1, CEIL_DIVIDE(m, NUM_THREADS)));
    dim3 grid(scan_blocks, bsz);
    #define kernel_args \
        (const void*) in_logits.data_ptr(), \
        (float*) out_logits.data_ptr(), \
        (const int64_t*) past_ids.data_ptr(), \
        past_len, \
        breakers_ptr, \
        (int*) workspace.data_ptr(), \
        (unsigned int*) counters.data_ptr(), \
        dim, multiplier, base, allowed_length, range, max_exponent, match_cap
    if (in_logits.dtype() == at::kHalf)
        dry_penalty_kernel<true><<<grid, NUM_THREADS, 0, stream>>>(kernel_args);
    else
    {
        TORCH_CHECK_DTYPE(in_logits, kFloat);
        dry_penalty_kernel<false><<<grid, NUM_THREADS, 0, stream>>>(kernel_args);
    }
    #undef kernel_args
    cuda_check(cudaPeekAtLastError());
}
