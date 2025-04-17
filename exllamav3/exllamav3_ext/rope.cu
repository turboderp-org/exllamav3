#include "rope.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "util.h"
#include "util.cuh"

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
    float attn_factor
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
    float fr = inv_freq[t];
    float pf = __int2float_rn(pos);
    float sin = __sinf(fr * pf) * attn_factor;
    float cos = __cosf(fr * pf) * attn_factor;

    if constexpr (rope_mode == ROPESTYLE_NEOX)
    {
        auto apply = [&] (const half* ptr, half* out_ptr)
        {
            float v1 = __half2float(ptr[0]);
            float v2 = __half2float(ptr[partial_head_dim / 2]);
            float r1 = v1 * cos - v2 * sin;
            float r2 = v2 * cos + v1 * sin;
            out_ptr[0] = __float2half_rn(r1);
            out_ptr[partial_head_dim / 2] = __float2half_rn(r2);
        };

        auto copy = [&] (const half* ptr, half* out_ptr)
        {
            *((half2*) out_ptr) = *((half2*) ptr);
        };

        int head = threadIdx.y;
        while (head < num_heads_q + num_heads_k)
        {
            if (head < num_heads_q)
            {
                int iq = ((batch * seq_len + token_pos) * num_heads_q + head) * head_dim + t;
                if (t < partial_head_dim / 2)
                    apply(q + iq, out_q + iq);
                else
                    copy(q + iq + t, out_q + iq + t);
            }
            else
            {
                int khead = head - num_heads_q;
                int ik = ((batch * seq_len + token_pos) * num_heads_k + khead) * head_dim + t;
                if (t < partial_head_dim / 2)
                    apply(k + ik, out_k + ik);
                else
                    copy(k + ik + t, out_k + ik + t);
            }
            head += blockDim.y;
        }
    }
    else if constexpr (rope_mode == ROPESTYLE_GPTJ)
    {
        auto apply = [&] (const half* ptr, half* out_ptr)
        {
            float v1 = __half2float(ptr[0]);
            float v2 = __half2float(ptr[1]);
            float r1 = v1 * cos - v2 * sin;
            float r2 = v2 * cos + v1 * sin;
            out_ptr[0] = __float2half_rn(r1);
            out_ptr[1] = __float2half_rn(r2);
        };

        auto copy = [&] (const half* ptr, half* out_ptr)
        {
            *((half2*) out_ptr) = *((half2*) ptr);
        };

        int head = threadIdx.y;
        while (head < num_heads_q + num_heads_k)
        {
            if (head < num_heads_q)
            {
                int iq = ((batch * seq_len + token_pos) * num_heads_q + head) * head_dim + t * 2;
                if (t < partial_head_dim / 2)
                    apply(q + iq, out_q + iq);
                else
                    copy(q + iq, out_q + iq);
            }
            else
            {
                int khead = head - num_heads_q;
                int ik = ((batch * seq_len + token_pos) * num_heads_k + khead) * head_dim + t * 2;
                if (t < partial_head_dim / 2)
                    apply(k + ik, out_k + ik);
                else
                    copy(k + ik, out_k + ik);
            }
            head += blockDim.y;
        }
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
    float attn_factor
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const half* q_ptr = (half*) q.data_ptr();
    half* out_q_ptr = (half*) out_q.data_ptr();
    const half* k_ptr = (const half*) OPTPTR(k);
    half* out_k_ptr = (half*) OPTPTR(out_k);
    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE_OPT(k, kHalf);
    TORCH_CHECK_DIM(q, 4);
    TORCH_CHECK_DIM_OPT(k, 4);

    int bsz = q.size(0);
    int seq_len = q.size(1);
    int num_heads_q = q.size(2);
    int num_heads_k = 0;
    int head_dim = q.size(3);
    int partial_head_dim = inv_freq.size(0) * 2;
    if (k_ptr)
    {
        num_heads_k = k.value().size(2);
        TORCH_CHECK(k.value().size(0) == bsz, "k is incorrect shape");
        TORCH_CHECK(k.value().size(1) == seq_len, "k is incorrect shape");
        TORCH_CHECK(k.value().size(3) == head_dim, "k is incorrect shape");
    }

    const float* inv_freq_ptr = (const float*) inv_freq.data_ptr();
    TORCH_CHECK_DTYPE(inv_freq, kFloat);
    TORCH_CHECK_DIM(inv_freq, 1);
//    TORCH_CHECK(inv_freq.size(0) == head_dim / 2, "inv_freq is incorrect shape");

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

    dim3 blocks(seq_len, bsz);
    int parallel_heads = MIN((MAX_NUM_THREADS / (head_dim / 2)), num_heads_q + num_heads_k);
    dim3 threads(head_dim / 2, parallel_heads);

    #define ARGS q_ptr, out_q_ptr, k_ptr, out_k_ptr, inv_freq_ptr, bsz, seq_len, num_heads_q, num_heads_k, \
                 head_dim, partial_head_dim, position, positions_ptr, position_ids_ptr, attn_factor
    if      (rope_mode == ROPESTYLE_GPTJ) rope_kernel<ROPESTYLE_GPTJ><<<blocks, threads, 0, stream>>>(ARGS);
    else if (rope_mode == ROPESTYLE_NEOX) rope_kernel<ROPESTYLE_NEOX><<<blocks, threads, 0, stream>>>(ARGS);
}
