#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "activation.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "compat.cuh"
#include <cmath>

using bfloat16 = __nv_bfloat16;
#define MAX_K_HEADS 32
#define MAX_V_HEADS 64

#define SUBK 4

#define FUSED_OP_2_THREADS 512

__device__ __forceinline__ float _sigmoid_fast_exp(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ bfloat16 trunc_bf16(float x)
{
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float untrunc_bf16(bfloat16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float as_float(bfloat16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float as_float(float x)
{
    return x;
}

__device__ __forceinline__ float softplus(float x)  // beta=1.0, linear threshold=20.0
{
    if (x > 20.0f) return x;
    return log1pf(__expf(x));
}

template<int MAX_HEAD_DIM>
__global__ __launch_bounds__(MAX_HEAD_DIM)
void gated_delta_net_fused_op_kernel
(
    const float* __restrict__ in_qkvz,          // [B,S,Nk, Fseg], float32
    const float* __restrict__ in_ba,            // [B,S,Nk, 2*Ng], float32
    const bfloat16* __restrict__ dt_bias,       // [Nv], bfloat16
    const bfloat16* __restrict__ a_log,         // [Nv], bfloat16
    bfloat16* __restrict__ out_qkv,             // [B, 2*Nk*Hk + Nv*Hv, S], bfloat16
    bfloat16* __restrict__ out_z,               // [B, S, Nv, Hv], bfloat16
    bfloat16* __restrict__ out_beta,            // [B, S, Nv], bfloat16
    float* __restrict__ out_g,                  // [B, S, Nv], float32
    const size_t B,
    const size_t S,
    const size_t Nk,
    const size_t Ng,
    const size_t Hk,
    const size_t Hv,
    const float beta_scale
)
{
    const size_t Nv   = Nk * Ng;
    const size_t Fseg = 2 * Hk + 2 * Ng * Hv;   // per-khead segment in mixed_qkvz
    const size_t Fba  = 2 * Ng;                 // per-khead segment in mixed_ba
    const size_t Nlin = B * S * Nk;
    const size_t Fout = 2 * Nk * Hk + Nv * Hv;  // feature dim in mixed_qkv

    int t = threadIdx.x;

    for (size_t linear = blockIdx.x; linear < Nlin; linear += (size_t) gridDim.x)
    {
        size_t kh = linear % Nk;
        size_t s = (linear / Nk) % S;
        size_t b = (linear / Nk) / S;

        // Base offsets into inputs for this (b,s,kh)
        const size_t base_qkvz = (((b * S) + s) * Nk + kh) * Fseg;
        const size_t base_ba   = (((b * S) + s) * Nk + kh) * Fba;

        // q block: length Hk, source offset 0..Hk-1
        // feature range in out_qkv: [kh*Hk, kh*Hk + Hk)
        const size_t q_feat0 = kh * Hk;
        if (t < Hk)
        {
            const float vq = in_qkvz[base_qkvz + t];
            const size_t f = q_feat0 + t;               // feature index in [0 .. Nk*Hk)
            const size_t out_off = ((b * Fout) + f) * S + s;
            out_qkv[out_off] = trunc_bf16(vq);
        }

        // k block: length Hk, source offset [Hk .. 2*Hk)
        // feature range in out_qkv: [Nk*Hk + kh*Hk, Nk*Hk + kh*Hk + Hk)
        const size_t k_in0 = Hk;
        const size_t k_feat0 = Nk*Hk + kh*Hk;
        if (t < Hk)
        {
            const float vk = in_qkvz[base_qkvz + k_in0 + t];
            const size_t f = k_feat0 + t;
            const size_t out_off = ((b * Fout) + f) * S + s;
            out_qkv[out_off] = trunc_bf16(vk);
        }

        // v and z blocks: each length Ng*Hv
        // v source offset: [2*Hk .. 2*Hk + Ng*Hv)
        // z source offset: [2*Hk + Ng*Hv .. 2*Hk + 2*Ng*Hv)
        const size_t v_in0 = 2*Hk;
        const size_t z_in0 = 2*Hk + Ng*Hv;
        const size_t v_feat_base = 2*Nk*Hk; // start of v block in feature dim

        if (t < Hv)
        {
            for (size_t g = 0; g < Ng; ++g)
            {
                const size_t vhead = kh * Ng + g; // global v-head index in [0..Nv)

                // v -> out_qkv (feature block)
                const float vv = in_qkvz[base_qkvz + v_in0 + g*Hv + t];
                const size_t f = v_feat_base + vhead*Hv + t;
                const size_t out_v_off = ((b * (size_t)Fout) + f) * S + s;
                out_qkv[out_v_off] = trunc_bf16(vv);

                // z -> out_z
                const float vz = in_qkvz[base_qkvz + z_in0 + g*Hv + t];
                const size_t out_z_off = ((((b * S) + s) * Nv) + vhead) * Hv + t;
                out_z[out_z_off] = trunc_bf16(vz);
            }
        }

        // b and a from mixed_ba (each Ng long) -> [B,S,Nv]
        if (t < Ng)
        {
            const size_t vhead = kh * Ng + t;
            const size_t out_va_off = ((b * S) + s) * Nv + vhead;

            // beta = sigmoid(b).bfloat16()
            float b = in_ba[base_ba + t];
            out_beta[out_va_off] = trunc_bf16(_sigmoid_fast_exp(b) * beta_scale);

            // g = -self.a_log.float().exp() * F.softplus(a + self.dt_bias.float())
            float g = in_ba[base_ba + Ng + t];
            float bi = untrunc_bf16(dt_bias[out_va_off % Nv]);
            float al = untrunc_bf16(a_log[out_va_off % Nv]);
            out_g[out_va_off] = -softplus(g + bi) * __expf(al);
        }
    }
}

/*
Single kernel for splitting projected qkvz + ba GDN inputs and producing gate + beta tensors
Also downcasts from float32 to bfloat16
*/

void gated_delta_net_fused_op
(
    const at::Tensor& mixed_qkvz,   // [B,S, Nk*(2*Hk + 2*Ng*Hv)]
    const at::Tensor& mixed_ba,     // [B,S, Nk*(2*Ng)]
    const at::Tensor& dt_bias,      // Nv
    const at::Tensor& a_log,        // Nv
    at::Tensor& mixed_qkv,          // out [B, 2*Nk*Hk + Nv*Hv, S]
    at::Tensor& z,                  // out [B, S, Nv, Hv]
    at::Tensor& beta,               // out [B, S, Nv]
    at::Tensor& g,                  // out [B, S, Nv]
    size_t num_k_heads,
    size_t num_v_heads,
    size_t k_head_dim,
    size_t v_head_dim,
    const float beta_scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(mixed_qkvz.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const auto B = mixed_qkvz.size(0);
    const auto S = mixed_qkvz.size(1);
    const auto Nk = num_k_heads;
    const auto Hk = k_head_dim;
    const auto Hv = v_head_dim;
    const auto Nv = num_v_heads;

    TORCH_CHECK(Nk > 0 && Nv > 0 && Hk > 0 && Hv > 0, "invalid sizes");
    TORCH_CHECK(Nv % Nk == 0, "num_v_heads must be divisible by num_k_heads");
    const size_t Ng = Nv / Nk;

    const size_t Fseg = 2*Hk + 2*Ng*Hv;
    TORCH_CHECK(mixed_qkvz.size(2) == Nk * Fseg, "mixed_qkvz last dim should be Nk*(2*Hk + 2*Ng*Hv)");
    TORCH_CHECK(mixed_ba.size(2) == Nk * (2*Ng), "mixed_ba last dim should be Nk*(2*Ng)");
    TORCH_CHECK(mixed_qkv.size(1) == 2*Nk*Hk + Nv*Hv, "mixed_qkv must be [B, 2*Nk*Hk + Nv*Hv, S]");
    TORCH_CHECK(mixed_qkv.size(2) == S, "mixed_qkv must be [B, 2*Nk*Hk + Nv*Hv, S]");

    TORCH_CHECK_DTYPE(mixed_qkvz, kFloat);
    TORCH_CHECK_DTYPE(mixed_ba, kFloat);
    TORCH_CHECK_DTYPE(dt_bias, kBFloat16);
    TORCH_CHECK_DTYPE(a_log, kBFloat16);
    TORCH_CHECK_DTYPE(mixed_qkv, kBFloat16);
    TORCH_CHECK_DTYPE(z, kBFloat16);
    TORCH_CHECK_DTYPE(beta, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);

    const int blocks = B * S * Nk;
    const int threads = MAX(Hk, Hv);

    #define KERNEL_ARGS                         \
        (const float*) mixed_qkvz.data_ptr(),   \
        (const float*) mixed_ba.data_ptr(),     \
        (const bfloat16*) dt_bias.data_ptr(),   \
        (const bfloat16*) a_log.data_ptr(),     \
        (bfloat16*) mixed_qkv.data_ptr(),       \
        (bfloat16*) z.data_ptr(),               \
        (bfloat16*) beta.data_ptr(),            \
        (float*) g.data_ptr(),                  \
        B, S, Nk, Ng, Hk, Hv,                   \
        beta_scale

    if (threads <= 128)
        gated_delta_net_fused_op_kernel<128><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
    else if (threads <= 256)
        gated_delta_net_fused_op_kernel<256><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
    else TORCH_CHECK(false, "Max head dim exceeded");

    #undef KERNEL_ARGS

    cuda_check(cudaPeekAtLastError());
}

template <typename a_log_T>
__global__ void gated_delta_net_fused_op_2_kernel
(
    const float* __restrict__ in_b,             // [B,S,H]
    const float* __restrict__ in_a,             // [B,S,H]
    const bfloat16* __restrict__ in_dt_bias,    // [H]
    const a_log_T* __restrict__ in_a_log,       // [H]
    bfloat16* __restrict__ out_beta,            // [B,S,H]
    float* __restrict__ out_g,                  // [B,S,H]
    int B,
    int S,
    int H,
    int rows_per_block,
    const float beta_scale
)
{
    int t = threadIdx.x % H;
    int row = blockIdx.x * rows_per_block + threadIdx.x / H;
    if (row >= B * S) return;

    in_b += row * H + t;
    in_a += row * H + t;
    in_dt_bias += t;
    in_a_log += t;
    out_beta += row * H + t;
    out_g += row * H + t;

    float beta = _sigmoid_fast_exp(*in_b) * beta_scale;
    float dt_bias = as_float(*in_dt_bias);
    float g = -softplus(*in_a + dt_bias) * __expf(as_float(*in_a_log));

    *out_beta = trunc_bf16(beta);
    *out_g = g;
}

/*
For Qwen3.5, producing gate + beta tensors, downcast to bfloat16
Transpose and qkv/z cast handled by Torch
*/

void gated_delta_net_fused_op_2
(
    const at::Tensor& b,            // [B,S,H] float
    const at::Tensor& a,            // [B,S,H] float
    const at::Tensor& dt_bias,      // [H] bfloat16
    const at::Tensor& a_log,        // [H] float
    at::Tensor& beta,               // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H] float
    const float beta_scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(b.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(b, kFloat);
    TORCH_CHECK_DTYPE(a, kFloat);
    TORCH_CHECK_DTYPE(dt_bias, kBFloat16);
    TORCH_CHECK_DTYPE(beta, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);

    bool a_log_fp32 = a_log.dtype() == at::kFloat;
    bool a_log_bf16 = a_log.dtype() == at::kBFloat16;

    TORCH_CHECK_SHAPES_FULL(b, a);
    TORCH_CHECK_SHAPES(b, 2, dt_bias, 0, 1);
    TORCH_CHECK_SHAPES(b, 2, a_log, 0, 1);
    TORCH_CHECK_SHAPES_FULL(b, beta);
    TORCH_CHECK_SHAPES_FULL(b, g);

    size_t B = b.size(0);
    size_t S = b.size(1);
    size_t H = b.size(2);

    int rows_per_block = FUSED_OP_2_THREADS / H;
    int threads = rows_per_block * H;
    int blocks = CEIL_DIVIDE(B * S, rows_per_block);

    #define ARGS(a_log_T)                       \
        (const float*) b.data_ptr(),            \
        (const float*) a.data_ptr(),            \
        (const bfloat16*) dt_bias.data_ptr(),   \
        (const a_log_T*) a_log.data_ptr(),      \
        (bfloat16*) beta.data_ptr(),            \
        (float*) g.data_ptr(),                  \
        B,                                      \
        S,                                      \
        H,                                      \
        rows_per_block,                         \
        beta_scale

    if (a_log_fp32)
        gated_delta_net_fused_op_2_kernel<<<blocks, threads, 0, stream>>>(ARGS(float));
    else if (a_log_bf16)
        gated_delta_net_fused_op_2_kernel<<<blocks, threads, 0, stream>>>(ARGS(bfloat16));
    else TORCH_CHECK(false, "gated_delta_net_fused_op_2: unsupported dtype");

    #undef ARGS

    cuda_check(cudaPeekAtLastError());
}


template <int MAX_HEAD_DIM, bool save_history, int V_SPLIT>
__global__ __launch_bounds__(MAX_HEAD_DIM * SUBK)
void cuda_recurrent_gated_delta_rule_kernel
(
                                                // k_dim = num_k_heads * k_head_dim
                                                // v_dim = num_v_heads * v_head_dim
    const bfloat16* __restrict__ mixed_qkv,     // [bsz, seqlen, (k_dim + k_dim + v_dim)]
    const float* __restrict__ g,                // [bsz, seqlen, (group * num_k_heads)]
    const bfloat16* __restrict__ beta,          // [bsz, seqlen, (group * num_k_heads)]
    float* __restrict__ recurrent_state,        // [num_slots, max_history + 1, (group * num_k_heads), k_head_dim, v_head_dim]
    bfloat16* __restrict__ core_attn_out,       // [bsz, seqlen, num_v_heads, v_head_dim]
    const int bsz,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const int k_head_dim,
    const int v_head_dim,
    const float scale,
    const int* __restrict__ slots,              // [bsz]
    const int history_stride                    // max_history + 1
)
{
    int group = num_v_heads / num_k_heads;
    const size_t state_size = group * num_k_heads * k_head_dim * v_head_dim;
    const size_t slot_size = (size_t) history_stride * state_size;

    // Advance to batch item
    int bi = blockIdx.x;
    mixed_qkv +=        bi * seqlen * (2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads);
    g +=                bi * seqlen * (group * num_k_heads);
    beta +=             bi * seqlen * (group * num_k_heads);
    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;
    float* final_state = slot_state;
    core_attn_out +=    bi * seqlen * num_v_heads * v_head_dim;

    // Indexing
    int t = threadIdx.x;
    int bt = threadIdx.y;
    int bts = k_head_dim / SUBK;
    int lane = t % 32;
    int warp = t / 32;
    int head = blockIdx.y;
    int k_head = head / group;
    int v_chunk = blockIdx.z;
    int v_chunk_dim = v_head_dim / V_SPLIT;
    int v_start = v_chunk * v_chunk_dim;

    // Shared buffers
    __shared__ float sh_red[2][MAX_HEAD_DIM / 32];
    __shared__ float sh_k[MAX_HEAD_DIM];
    __shared__ float sh_q[MAX_HEAD_DIM];
    __shared__ float sh_dot1[MAX_HEAD_DIM];
    __shared__ float sh_dot2[MAX_HEAD_DIM];

    // Iterate over sequence dim
    for (int s = 0; s < seqlen; ++s)
    {
        // Advance to q/k head
        const bfloat16* gl_q = mixed_qkv + k_head * k_head_dim;
        const bfloat16* gl_k = mixed_qkv + (num_k_heads + k_head) * k_head_dim;
        const bfloat16* gl_v = mixed_qkv + (2 * num_k_heads * k_head_dim) + head * v_head_dim + v_start;
        bfloat16* out = core_attn_out + head * v_head_dim + v_start;

        float* gl_rs_r;
        float* gl_rs_w;
        if constexpr (save_history)
        {
            bool first = (s == 0);
            bool last = (s == seqlen - 1);
            float* history_r = first ? nullptr : slot_state + (size_t) s * state_size;
            float* history_w = last  ? final_state : slot_state + (size_t) (s + 1) * state_size;
            gl_rs_r = first ? final_state + head * (k_head_dim * v_head_dim)
                            : history_r   + head * (k_head_dim * v_head_dim);
            gl_rs_w = history_w           + head * (k_head_dim * v_head_dim);
        }
        else
        {
            gl_rs_r = final_state + head * (k_head_dim * v_head_dim);
            gl_rs_w = gl_rs_r;
        }

        // Read q/k heads and apply L2 norm
        float q, k;
        if (t < k_head_dim && bt == 0)
        {
            q = __bfloat162float(gl_q[t]);
            k = __bfloat162float(gl_k[t]);

            float sumq = q * q;
            float sumk = k * k;
            #pragma unroll
            for(int offset = 16; offset > 0; offset /= 2)
            {
                sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
                sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
            }
            if (lane == 0)
            {
                sh_red[0][warp] = sumq;
                sh_red[1][warp] = sumk;
            }
        }
        __syncthreads();

        if (t < k_head_dim && bt == 0)
        {
            float sumq = lane < k_head_dim / 32 ? sh_red[0][lane] : 0.0f;
            float sumk = lane < k_head_dim / 32 ? sh_red[1][lane] : 0.0f;
            #pragma unroll
            for(int offset = 16; offset > 0; offset /= 2)
            {
                sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
                sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
            }

            q = q * rsqrtf(sumq + 1e-6f);
            k = k * rsqrtf(sumk + 1e-6f);

            // Write q, k to shmem
            sh_k[t] = k;
            sh_q[t] = q;
        }

        if (t < v_chunk_dim && bt == 0)
        {
            sh_dot1[t] = 0.0f;
            sh_dot2[t] = 0.0f;
        }
        __syncthreads();

        if (t < v_chunk_dim)
        {
            // Dot products with last state
            float sum = 0.0f;
            float* sh_k_rd = sh_k + bt * bts;
            float* rs_rd = gl_rs_r + v_start + t + bt * bts * v_head_dim;

            for (int i = 0; i < k_head_dim / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_rd += v_head_dim, sh_k_rd++)
                    sum = sum + *sh_k_rd * *rs_rd;
            }
            atomicAdd(sh_dot1 + t, sum);
        }
        __syncthreads();

        if (t < v_chunk_dim)
        {
            float g_h = __expf(g[head]);
            float beta_h = __bfloat162float(beta[head]);

            float sum = sh_dot1[t];

            // Read v head and update
            float v = __bfloat162float(gl_v[t]) - sum * g_h;

            // Update step
            float v_out = 0.0f;
            float* sh_k_rd = sh_k + bt * bts;
            float* sh_q_rd = sh_q + bt * bts;
            float* rs_r = gl_rs_r + v_start + t + bt * bts * v_head_dim;
            float* rs_w = gl_rs_w + v_start + t + bt * bts * v_head_dim;

            for (int i = 0; i < k_head_dim / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_r += v_head_dim, rs_w += v_head_dim, sh_k_rd++, sh_q_rd++)
                {
                    // State update step, k x v
                    float state = *rs_r;
                    state = state * g_h + *sh_k_rd * v * beta_h;
                    *rs_w = state;

                    // Accumulate attn output
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            atomicAdd(sh_dot2 + t, v_out);
        }
        __syncthreads();

        if (t < v_chunk_dim && bt == 0)
        {
            float v_out = sh_dot2[t];

            // Store attn output
            out[t] = __float2bfloat16_rz(v_out * scale);
        }

        // Next seq index
        mixed_qkv +=        2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads;
        g +=                num_v_heads;
        beta +=             num_v_heads;
        core_attn_out +=    num_v_heads * v_head_dim;
    }
}

template <bool save_history, int V_SPLIT>
__global__ __launch_bounds__(128 * SUBK)
void cuda_recurrent_gated_delta_rule_kernel_128
(
                                                // k_head_dim = v_head_dim = 128
    const bfloat16* __restrict__ mixed_qkv,     // [bsz, seqlen, (k_dim + k_dim + v_dim)]
    const float* __restrict__ g,                // [bsz, seqlen, (group * num_k_heads)]
    const bfloat16* __restrict__ beta,          // [bsz, seqlen, (group * num_k_heads)]
    float* __restrict__ recurrent_state,        // [num_slots, max_history + 1, (group * num_k_heads), 128, 128]
    bfloat16* __restrict__ core_attn_out,       // [bsz, seqlen, num_v_heads, 128]
    const int bsz,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const int k_head_dim,
    const int v_head_dim,
    const float scale,
    const int* __restrict__ slots,              // [bsz]
    const int history_stride                    // max_history + 1
)
{
    constexpr int HEAD_DIM = 128;
    constexpr int V_CHUNK_DIM = HEAD_DIM / V_SPLIT;
    constexpr int BTS = HEAD_DIM / SUBK;

    int group = num_v_heads / num_k_heads;
    constexpr size_t HEAD_STATE_SIZE = HEAD_DIM * HEAD_DIM;
    const size_t state_size = group * num_k_heads * HEAD_STATE_SIZE;
    const size_t slot_size = (size_t) history_stride * state_size;

    int bi = blockIdx.x;
    mixed_qkv +=        bi * seqlen * (3 * HEAD_DIM * num_k_heads + HEAD_DIM * (num_v_heads - num_k_heads));
    g +=                bi * seqlen * (group * num_k_heads);
    beta +=             bi * seqlen * (group * num_k_heads);
    int state_slot = slots ? slots[bi] : bi;
    float* slot_state = recurrent_state + (size_t) state_slot * slot_size;
    float* final_state = slot_state;
    core_attn_out +=    bi * seqlen * num_v_heads * HEAD_DIM;

    int t = threadIdx.x;
    int bt = threadIdx.y;
    int lane = t % 32;
    int warp = t / 32;
    int head = blockIdx.y;
    int k_head = head / group;
    int v_chunk = blockIdx.z;
    int v_start = v_chunk * V_CHUNK_DIM;

    __shared__ float sh_red[2][HEAD_DIM / 32];
    __shared__ float sh_k[HEAD_DIM];
    __shared__ float sh_q[HEAD_DIM];
    __shared__ float sh_dot1[HEAD_DIM];
    __shared__ float sh_dot2[HEAD_DIM];

    for (int s = 0; s < seqlen; ++s)
    {
        const bfloat16* gl_q = mixed_qkv + k_head * HEAD_DIM;
        const bfloat16* gl_k = mixed_qkv + (num_k_heads + k_head) * HEAD_DIM;
        const bfloat16* gl_v = mixed_qkv + (2 * num_k_heads * HEAD_DIM) + head * HEAD_DIM + v_start;
        bfloat16* out = core_attn_out + head * HEAD_DIM + v_start;

        float* gl_rs_r;
        float* gl_rs_w;
        if constexpr (save_history)
        {
            bool first = (s == 0);
            bool last = (s == seqlen - 1);
            float* history_r = first ? nullptr : slot_state + (size_t) s * state_size;
            float* history_w = last  ? final_state : slot_state + (size_t) (s + 1) * state_size;
            gl_rs_r = first ? final_state + head * HEAD_STATE_SIZE
                            : history_r   + head * HEAD_STATE_SIZE;
            gl_rs_w = history_w           + head * HEAD_STATE_SIZE;
        }
        else
        {
            gl_rs_r = final_state + head * HEAD_STATE_SIZE;
            gl_rs_w = gl_rs_r;
        }

        float q = __bfloat162float(gl_q[t]);
        float k = __bfloat162float(gl_k[t]);

        float sumq = q * q;
        float sumk = k * k;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }
        if (lane == 0)
        {
            sh_red[0][warp] = sumq;
            sh_red[1][warp] = sumk;
        }
        __syncthreads();

        sumq = lane < HEAD_DIM / 32 ? sh_red[0][lane] : 0.0f;
        sumk = lane < HEAD_DIM / 32 ? sh_red[1][lane] : 0.0f;
        #pragma unroll
        for(int offset = 16; offset > 0; offset /= 2)
        {
            sumq += __shfl_xor_sync(0xffffffff, sumq, offset);
            sumk += __shfl_xor_sync(0xffffffff, sumk, offset);
        }

        q = q * rsqrtf(sumq + 1e-6f);
        k = k * rsqrtf(sumk + 1e-6f);
        sh_k[t] = k;
        sh_q[t] = q;

        if (t < V_CHUNK_DIM && bt == 0)
        {
            sh_dot1[t] = 0.0f;
            sh_dot2[t] = 0.0f;
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float sum = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* rs_rd = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_rd += HEAD_DIM, sh_k_rd++)
                    sum = sum + *sh_k_rd * *rs_rd;
            }
            atomicAdd(sh_dot1 + t, sum);
        }
        __syncthreads();

        if (t < V_CHUNK_DIM)
        {
            float g_h = __expf(g[head]);
            float beta_h = __bfloat162float(beta[head]);
            float v = __bfloat162float(gl_v[t]) - sh_dot1[t] * g_h;
            float v_out = 0.0f;
            float* sh_k_rd = sh_k + bt * BTS;
            float* sh_q_rd = sh_q + bt * BTS;
            float* rs_r = gl_rs_r + v_start + t + bt * BTS * HEAD_DIM;
            float* rs_w = gl_rs_w + v_start + t + bt * BTS * HEAD_DIM;

            #pragma unroll
            for (int i = 0; i < HEAD_DIM / 8 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 8; ++j, rs_r += HEAD_DIM, rs_w += HEAD_DIM, sh_k_rd++, sh_q_rd++)
                {
                    float state = *rs_r;
                    state = state * g_h + *sh_k_rd * v * beta_h;
                    *rs_w = state;
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            atomicAdd(sh_dot2 + t, v_out);
        }
        __syncthreads();

        if (t < V_CHUNK_DIM && bt == 0)
            out[t] = __float2bfloat16_rz(sh_dot2[t] * scale);

        mixed_qkv +=        2 * HEAD_DIM * num_k_heads + HEAD_DIM * num_v_heads;
        g +=                num_v_heads;
        beta +=             num_v_heads;
        core_attn_out +=    num_v_heads * HEAD_DIM;
    }
}

void cuda_recurrent_gated_delta_rule
(
    const at::Tensor& mixed_qkv,
    const at::Tensor& g,
    const at::Tensor& beta,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history
)
{
    const at::cuda::OptionalCUDAGuard device_guard(mixed_qkv.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = mixed_qkv.size(0);
    int seqlen = mixed_qkv.size(1);
    int qkv_dim = mixed_qkv.size(2);

    TORCH_CHECK(num_v_heads % num_k_heads == 0, "num_v_heads must be divisible by num_k_heads");
    TORCH_CHECK(k_head_dim >= 32 && k_head_dim % (8 * SUBK) == 0, "k_head_dim must be a multiple of 32");
    TORCH_CHECK(MAX(k_head_dim, v_head_dim) <= 256, "Max head dim exceeded");

    TORCH_CHECK(qkv_dim == 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim,
                "mixed_qkv must be [bsz, seqlen, 2*num_k_heads*k_head_dim + num_v_heads*v_head_dim]");
    TORCH_CHECK(g.dim() == 3 && g.size(0) == bsz && g.size(1) == seqlen && g.size(2) == num_v_heads,
                "g must be [bsz, seqlen, num_v_heads]");
    TORCH_CHECK(beta.dim() == 3 && beta.size(0) == bsz && beta.size(1) == seqlen && beta.size(2) == num_v_heads,
                "beta must be [bsz, seqlen, num_v_heads]");
    TORCH_CHECK(recurrent_state.dim() == 5 &&
                recurrent_state.size(0) >= (slots.has_value() ? 1 : bsz) &&
                recurrent_state.size(1) >= (history ? seqlen : 1) &&
                recurrent_state.size(2) == num_v_heads &&
                recurrent_state.size(3) == k_head_dim &&
                recurrent_state.size(4) == v_head_dim,
                "recurrent_state must be [num_slots, max_history + 1, num_v_heads, k_head_dim, v_head_dim]");
    TORCH_CHECK(core_attn_out.dim() == 4 &&
                core_attn_out.size(0) == bsz &&
                core_attn_out.size(1) == seqlen &&
                core_attn_out.size(2) == num_v_heads &&
                core_attn_out.size(3) == v_head_dim,
                "core_attn_out must be [bsz, seqlen, num_v_heads, v_head_dim]");

    TORCH_CHECK_DTYPE(mixed_qkv, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);
    TORCH_CHECK_DTYPE(beta, kBFloat16);
    TORCH_CHECK_DTYPE(recurrent_state, kFloat);
    TORCH_CHECK_DTYPE(core_attn_out, kBFloat16);
    TORCH_CHECK_DTYPE_OPT(slots, kInt);

    const int* slots_ptr = (const int*) OPTPTR(slots);
    if (slots_ptr)
    {
        TORCH_CHECK(slots.value().dim() == 1 &&
                    slots.value().size(0) == bsz,
                    "slots must be [bsz]");
        TORCH_CHECK(slots.value().device() == mixed_qkv.device(),
                    "slots must be on the same device as mixed_qkv");
    }

    int v_split = (bsz == 1 && k_head_dim <= 128 && v_head_dim == 128 && num_v_heads <= 64) ? 4 : 1;
    TORCH_CHECK(v_head_dim % v_split == 0, "v_head_dim must be divisible by v_split");

    dim3 blocks(bsz, num_v_heads, v_split);  // group * num_k_heads
    dim3 threads(MAX(k_head_dim, v_head_dim / v_split), SUBK);
    int history_stride = recurrent_state.size(1);

    float scale = 1.0f / sqrtf(k_head_dim);

    #define KERNEL_ARGS                         \
        (const bfloat16*) mixed_qkv.data_ptr(), \
        (const float*) g.data_ptr(),            \
        (const bfloat16*) beta.data_ptr(),      \
        (float*) recurrent_state.data_ptr(),    \
        (bfloat16*) core_attn_out.data_ptr(),   \
        bsz,                                    \
        seqlen,                                 \
        num_k_heads,                            \
        num_v_heads,                            \
        k_head_dim,                             \
        v_head_dim,                             \
        scale,                                  \
        slots_ptr,                              \
        history_stride

    if (!history)
    {
        if (k_head_dim == 128 && v_head_dim == 128)
        {
            if (v_split == 4) cuda_recurrent_gated_delta_rule_kernel_128<false, 4><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
            else              cuda_recurrent_gated_delta_rule_kernel_128<false, 1><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
        }
        else if (threads.x <= 128)
        {
            if (v_split == 4) cuda_recurrent_gated_delta_rule_kernel<128, false, 4><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
            else              cuda_recurrent_gated_delta_rule_kernel<128, false, 1><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
        }
        else if (threads.x <= 256)
                              cuda_recurrent_gated_delta_rule_kernel<256, false, 1><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
        else TORCH_CHECK(false, "Max head dim exceeded");
    }
    else
    {
        if (k_head_dim == 128 && v_head_dim == 128)
        {
            if (v_split == 4) cuda_recurrent_gated_delta_rule_kernel_128<true, 4><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
            else              cuda_recurrent_gated_delta_rule_kernel_128<true, 1><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
        }
        else if (threads.x <= 128)
        {
            if (v_split == 4) cuda_recurrent_gated_delta_rule_kernel<128, true, 4><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
            else              cuda_recurrent_gated_delta_rule_kernel<128, true, 1><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
        }
        else if (threads.x <= 256)
                              cuda_recurrent_gated_delta_rule_kernel<256, true, 1><<<blocks, threads, 0, stream>>>(KERNEL_ARGS);
        else TORCH_CHECK(false, "Max head dim exceeded");
    }
    #undef KERNEL_ARGS

    cuda_check(cudaPeekAtLastError());
}

#define CONV1D_MAX_K 16
#define CONV1D_NUM_THREADS 256

// Causal conv1d update for short decode steps. Equivalent to the triton kernel in
// modules/gated_delta_net_fn/conv1d.py but launchable in ~4us instead of ~50us of host time.
//
// out[b,s,d] = act(bias[d] + sum_k w[d,k] * in(b,d,s+k+1)) where the input sequence is the
// concatenation of conv_state[slot,d,0:K] and x[b,d,0:seqlen]. Without history, the last K
// inputs are written back to conv_state[slot,d,0:K]; with history, the last
// min(state_size, K+seqlen) inputs are written to the tail of the state buffer (rewindable).

template <bool ACT, bool HISTORY>
__global__ __launch_bounds__(CONV1D_NUM_THREADS)
void conv1d_update_kernel
(
    const bfloat16* __restrict__ x,           // (bsz, dim, seqlen)
    bfloat16* __restrict__ conv_state,        // (num_slots, dim, state_size)
    const int* __restrict__ slots,            // (bsz) or null (identity)
    const bfloat16* __restrict__ weight,      // (dim, K)
    const bfloat16* __restrict__ bias,        // (dim) or null
    bfloat16* __restrict__ out,               // (bsz, seqlen, dim)
    const int dim,
    const int seqlen,
    const int state_size,
    const int K
)
{
    int d = blockIdx.x * CONV1D_NUM_THREADS + threadIdx.x;
    if (d >= dim) return;
    int b = blockIdx.y;
    int slot = slots ? slots[b] : b;

    const bfloat16* x_d = x + ((size_t) b * dim + d) * seqlen;
    bfloat16* state_d = conv_state + ((size_t) slot * dim + d) * state_size;

    float w[CONV1D_MAX_K];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < K) w[k] = __bfloat162float(weight[(size_t) d * K + k]);

    float bias_d = bias ? __bfloat162float(bias[d]) : 0.0f;

    // Previous window; win[K-1] slot is filled with the current input each step
    float old_state[CONV1D_MAX_K];
    float win[CONV1D_MAX_K];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < K) old_state[k] = __bfloat162float(state_d[k]);
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K - 1; ++k)
        if (k < K - 1) win[k] = old_state[k + 1];

    for (int s = 0; s < seqlen; ++s)
    {
        win[K - 1] = __bfloat162float(x_d[s]);

        float acc = bias_d;
        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K; ++k)
            if (k < K) acc = fmaf(w[k], win[k], acc);

        if constexpr (ACT)
            acc *= _sigmoid_fast_exp(acc);

        out[((size_t) b * seqlen + s) * dim + d] = __float2bfloat16_rn(acc);

        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K - 1; ++k)
            if (k < K - 1) win[k] = win[k + 1];
    }

    if constexpr (!HISTORY)
    {
        #pragma unroll
        for (int k = 0; k < CONV1D_MAX_K; ++k)
        {
            if (k < K)
            {
                int src_t = seqlen + k;
                float v = (src_t < K) ? old_state[src_t] : __bfloat162float(x_d[src_t - K]);
                state_d[k] = __float2bfloat16_rn(v);
            }
        }
    }
    else
    {
        int total = K + seqlen;
        int write_size = state_size < total ? state_size : total;
        int dst_start = state_size - write_size;
        int src_start = total - write_size;
        for (int j = 0; j < write_size; ++j)
        {
            int src_t = src_start + j;
            float v = (src_t < K) ? old_state[src_t] : __bfloat162float(x_d[src_t - K]);
            state_d[dst_start + j] = __float2bfloat16_rn(v);
        }
    }
}

void cuda_causal_conv1d_update
(
    const at::Tensor& x,
    at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    bool activation,
    bool history
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = x.size(0);
    int dim = x.size(1);
    int seqlen = x.size(2);
    int state_size = conv_state.size(2);
    int K = weight.size(1);

    TORCH_CHECK(K <= CONV1D_MAX_K, "conv kernel size exceeds CONV1D_MAX_K");
    TORCH_CHECK(state_size >= K, "conv_state must have at least K entries");
    TORCH_CHECK(x.is_contiguous() && conv_state.is_contiguous() && weight.is_contiguous() && out.is_contiguous(),
                "x, conv_state, weight and out must be contiguous");
    TORCH_CHECK(conv_state.dim() == 3 && conv_state.size(1) == dim,
                "conv_state must be (num_slots, dim, state_size)");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == dim,
                "weight must be (dim, K)");
    TORCH_CHECK(out.dim() == 3 && out.size(0) == bsz && out.size(1) == seqlen && out.size(2) == dim,
                "out must be (bsz, seqlen, dim)");
    TORCH_CHECK_DTYPE(x, kBFloat16);
    TORCH_CHECK_DTYPE(conv_state, kBFloat16);
    TORCH_CHECK_DTYPE(weight, kBFloat16);
    TORCH_CHECK_DTYPE_OPT(bias, kBFloat16);
    TORCH_CHECK_DTYPE(out, kBFloat16);
    TORCH_CHECK_DTYPE_OPT(slots, kInt);

    const int* slots_ptr = (const int*) OPTPTR(slots);
    if (slots_ptr)
    {
        TORCH_CHECK(slots.value().dim() == 1 && slots.value().size(0) == bsz, "slots must be (bsz)");
    }
    else
    {
        TORCH_CHECK(conv_state.size(0) >= bsz, "conv_state too small for batch without slots");
    }
    const bfloat16* bias_ptr = (const bfloat16*) OPTPTR(bias);

    dim3 blocks(CEIL_DIVIDE(dim, CONV1D_NUM_THREADS), bsz);

    #define KERNEL_ARGS                             \
        (const bfloat16*) x.data_ptr(),             \
        (bfloat16*) conv_state.data_ptr(),          \
        slots_ptr,                                  \
        (const bfloat16*) weight.data_ptr(),        \
        bias_ptr,                                   \
        (bfloat16*) out.data_ptr(),                 \
        dim, seqlen, state_size, K

    if (activation)
    {
        if (history) conv1d_update_kernel<true, true><<<blocks, CONV1D_NUM_THREADS, 0, stream>>>(KERNEL_ARGS);
        else         conv1d_update_kernel<true, false><<<blocks, CONV1D_NUM_THREADS, 0, stream>>>(KERNEL_ARGS);
    }
    else
    {
        if (history) conv1d_update_kernel<false, true><<<blocks, CONV1D_NUM_THREADS, 0, stream>>>(KERNEL_ARGS);
        else         conv1d_update_kernel<false, false><<<blocks, CONV1D_NUM_THREADS, 0, stream>>>(KERNEL_ARGS);
    }
    #undef KERNEL_ARGS

    cuda_check(cudaPeekAtLastError());
}
