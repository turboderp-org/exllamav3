#include "rope.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "util.h"
#include "util.cuh"
#include "reduction.cuh"

#define ROPESTYLE_NONE 0
#define ROPESTYLE_GPTJ 1
#define ROPESTYLE_NEOX 2
#define MAX_NUM_THREADS 1024

template <int rope_mode>
__global__
void rope_kernel
(
    const half* __restrict__ q,
    half* __restrict__ out_q,
    const half* __restrict__ k,
    half* __restrict__ out_k,
    const float* __restrict__ inv_freq,
    int bsz,
    int seq_len,
    int num_heads_q,
    int num_heads_k,
    int head_dim,
    int partial_head_dim,
    int position,
    const uint32_t* __restrict__ positions,
    const uint32_t* __restrict__ position_ids,
    float attn_factor,
    const half* __restrict__ q_norm,
    const half* __restrict__ k_norm,
    float norm_eps,
    float norm_constant_bias,
    bool inv_freq_table
)
{
    // Get position
    int batch = blockIdx.y;
    int token_pos = blockIdx.x;
    int pos = token_pos + position;
    if (positions)
        pos = token_pos + positions[batch];
    else if (position_ids)
        pos = position_ids[batch * seq_len + token_pos];

    // Load inv_freq, compute sin/cos
    int t = threadIdx.x;
    int t_head = threadIdx.y;

    float sin;
    float cos;
    if (!inv_freq_table)
    {
        float fr = inv_freq[t];
        float pf = __int2float_rn(pos);
        sin = __sinf(fr * pf) * attn_factor;
        cos = __cosf(fr * pf) * attn_factor;
    }
    else
    {
        float fr = inv_freq[pos * partial_head_dim / 2 + t];
        sin = __sinf(fr) * attn_factor;
        cos = __cosf(fr) * attn_factor;
    }

    // Shared buffer
    __shared__ half norm_head[MAX_NUM_THREADS * 2];
    __shared__ float sums[MAX_NUM_THREADS / 32];

    // Prep
    half2 norm_constant_bias_h2 = __float2half2_rn(norm_constant_bias);
    int head_dim_pad = (head_dim + 63) / 64 * 64;

    // Loop over heads
    for (int head_idx = threadIdx.y; head_idx < num_heads_q + num_heads_k; head_idx += blockDim.y)
    {
        const half* g_head_in_ptr;
        half* g_head_out_ptr;
        const half* norm_weight;
        if (head_idx < num_heads_q)
        {
            g_head_in_ptr = q + ((batch * seq_len + token_pos) * num_heads_q + head_idx) * head_dim;
            g_head_out_ptr = out_q + ((batch * seq_len + token_pos) * num_heads_q + head_idx) * head_dim;
            norm_weight = q_norm;
        }
        else if (head_idx < num_heads_q + num_heads_k)
        {
            g_head_in_ptr = k + ((batch * seq_len + token_pos) * num_heads_k + head_idx - num_heads_q) * head_dim;
            g_head_out_ptr = out_k + ((batch * seq_len + token_pos) * num_heads_k + head_idx - num_heads_q) * head_dim;
            norm_weight = k_norm;
        }

        // Temp storage
        half* sh_head = norm_head + t_head * head_dim_pad;
        auto load_head = [&] ()
        {
            if (t < head_dim / 2)
                ((half2*) sh_head)[t] = ((half2*)g_head_in_ptr)[t];
            else
                ((half2*) sh_head)[t] = {};
            __syncthreads();
        };
        auto store_head = [&] ()
        {
            if (t < head_dim / 2)
                ((half2*) g_head_out_ptr)[t] = ((half2*) sh_head)[t];
            __syncthreads();
        };

        // Apply embeddings
        auto apply_rope = [&] ()
        {
            if (t < partial_head_dim / 2)
            {
                if constexpr (rope_mode == ROPESTYLE_NEOX)
                {
                    float v1 = __half2float(sh_head[t]);
                    float v2 = __half2float(sh_head[t + partial_head_dim / 2]);
                    float r1 = v1 * cos - v2 * sin;
                    float r2 = v2 * cos + v1 * sin;
                    sh_head[t] = __float2half_rn(r1);
                    sh_head[t + partial_head_dim / 2] = __float2half_rn(r2);
                }
                else if constexpr (rope_mode == ROPESTYLE_GPTJ)
                {
                    half2 *tptr = (half2*)(sh_head + t * 2);
                    half2 v = *tptr;
                    float v1 = __low2float(v);
                    float v2 = __high2float(v);
                    float r1 = v1 * cos - v2 * sin;
                    float r2 = v2 * cos + v1 * sin;
                    v = __floats2half2_rn(r1, r2);
                    *tptr = v;
                }
            }
            __syncthreads();
        };

        // RMS Norm
        auto apply_norm = [&] ()
        {
            half2 *tptr = (half2*)(sh_head + t * 2);
            half2 *wptr = (half2*)(norm_weight + t * 2);
            // int lane_id = threadIdx.x % 32;
            int warp_id = threadIdx.x / 32;
            int warps = blockDim.x / 32;

            // Sum of squares
            half2 v = *tptr;
            float v1 = __low2float(v);
            float v2 = __high2float(v);
            float sum = v1 * v1 + v2 * v2;
            sums[warps * t_head + warp_id] = warp_reduce_sum_f(sum);
            __syncthreads();

            sum = sums[warps * t_head];
            for (int i = 1; i < warps; ++i) sum += sums[warps * t_head + i];

            // Normalize and downcast
            float rmf = rsqrtf(sum / (float) head_dim + norm_eps);
            v1 *= rmf;
            v2 *= rmf;
            v = __floats2half2_rn(v1, v2);

            // Apply weight and store
            half2 w = __hadd2(*wptr, norm_constant_bias_h2);
            v = __hmul2(w, v);
            *tptr = v;
            __syncthreads();
        };

        // Do the things
        load_head();
        if (q_norm) apply_norm();
        apply_rope();
        store_head();
    }
}

/*

Apply position embeddings, works in-place

- q: tensor of shape (bsz, seq_len, num_heads_q, head_dim), float16
- k: tensor of shape (bsz, seq_len, num_heads_k, head_dim), float16, optional
- out_q: output for queries, may be == q
- out_k: output for keys, may be == k
- inv_freq: tensor of shape (head_dim / 2), float32
- position: int, constant position offset (position ID of first token across batch)
- positions: tensor of shape (bsz), (position ID of first token per seq), int, optional
- position_ids: tensor of shape (bsz, seq_len), int, optional
- rope_mode: ROPESTYLE_NEOX
- attn_factor: scale for sin/cos factors
- q_norm: optional RMS norm weight, must be supplied with k_norm
- k_norm: optional RMS norm weight, must be supplied with q_norm
- norm_eps
- norm_constant_bias

Either positions or position_ids overrides position
*/

void rope
(
    const at::Tensor& q,
    at::Tensor& out_q,
    const c10::optional<at::Tensor>& k,
    c10::optional<at::Tensor>& out_k,
    const at::Tensor& inv_freq,
    uint32_t position,
    const c10::optional<at::Tensor>& positions,
    const c10::optional<at::Tensor>& position_ids,
    int rope_mode,
    float attn_factor,
    const c10::optional<at::Tensor>& q_norm,
    const c10::optional<at::Tensor>& k_norm,
    float norm_eps,
    float norm_constant_bias
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = q.size(0);
    int seq_len = q.size(1);
    int num_heads_q = q.size(2);
    int num_heads_k = 0;
    int head_dim = q.size(3);
    int partial_head_dim = inv_freq.size(-1) * 2;

    const half* q_ptr = (half*) q.data_ptr();
    half* out_q_ptr = (half*) out_q.data_ptr();
    const half* k_ptr = (const half*) OPTPTR(k);
    half* out_k_ptr = (half*) OPTPTR(out_k);
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE_OPT(k, kHalf);
    TORCH_CHECK_DIM(q, 4);
    TORCH_CHECK_DIM_OPT(k, 4);

    if (k_ptr)
    {
        num_heads_k = k.value().size(2);
        TORCH_CHECK(k.value().size(0) == bsz, "k is incorrect shape");
        TORCH_CHECK(k.value().size(1) == seq_len, "k is incorrect shape");
        TORCH_CHECK(k.value().size(3) == head_dim, "k is incorrect shape");
    }

    const float* inv_freq_ptr = (const float*) inv_freq.data_ptr();
    TORCH_CHECK_DTYPE(inv_freq, kFloat);
    bool inv_freq_table = false;
    if (inv_freq.dim() > 1)
    {
        TORCH_CHECK_DIM(inv_freq, 2);
        TORCH_CHECK_SHAPES(q, 3, inv_freq, 1, 2);
        inv_freq_table = true;
    }

    uint32_t* positions_ptr = (uint32_t*) OPTPTR(positions);
    uint32_t* position_ids_ptr = (uint32_t*) OPTPTR(position_ids);
    TORCH_CHECK_DTYPE_OPT(positions, kInt);
    TORCH_CHECK_DTYPE_OPT(position_ids, kInt);
    TORCH_CHECK((positions_ptr != nullptr) + (position_ids_ptr != nullptr) <= 1, "invalid arguments")

    if (positions_ptr)
    {
        TORCH_CHECK_DIM(positions.value(), 1)
        TORCH_CHECK(positions.value().size(0) == bsz, "positions is incorrect shape");
    }

    if (position_ids_ptr)
    {
        TORCH_CHECK_DIM(position_ids.value(), 2)
        TORCH_CHECK(position_ids.value().size(0) == bsz, "position_ids is incorrect shape");
        TORCH_CHECK(position_ids.value().size(1) == seq_len, "position_ids is incorrect shape");
    }

    half* q_norm_ptr = (half*) OPTPTR(q_norm);
    half* k_norm_ptr = (half*) OPTPTR(k_norm);
    TORCH_CHECK_DTYPE_OPT(q_norm, kHalf);
    TORCH_CHECK_DTYPE_OPT(k_norm, kHalf);
    if (q_norm_ptr)
    {
        TORCH_CHECK_DIM(q_norm.value(), 1);
        TORCH_CHECK(q_norm.value().size(0) == head_dim, "q_norm is incorrect size");
    }

    dim3 blocks(seq_len, bsz);
    int warps = CEIL_DIVIDE(head_dim / 2, 32);
    int thr = warps * 32;
    int parallel_heads = MIN((MAX_NUM_THREADS / thr), num_heads_q + num_heads_k);
    dim3 threads(thr, parallel_heads);

    #define ARGS q_ptr, out_q_ptr, k_ptr, out_k_ptr, inv_freq_ptr, bsz, seq_len, num_heads_q, num_heads_k, \
                 head_dim, partial_head_dim, position, positions_ptr, position_ids_ptr, attn_factor, \
                 q_norm_ptr, k_norm_ptr, norm_eps, norm_constant_bias, inv_freq_table

    if      (rope_mode == ROPESTYLE_GPTJ) rope_kernel<ROPESTYLE_GPTJ><<<blocks, threads, 0, stream>>>(ARGS);
    else if (rope_mode == ROPESTYLE_NEOX) rope_kernel<ROPESTYLE_NEOX><<<blocks, threads, 0, stream>>>(ARGS);
}
