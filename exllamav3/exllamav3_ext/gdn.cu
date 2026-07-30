#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "activation.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "compat.cuh"
#include "graph.cuh"
#include "gdn.cuh"
#include <cmath>

using bfloat16 = __nv_bfloat16;
#define MAX_K_HEADS 32
#define MAX_V_HEADS 64

#define SUBK 4

#define FUSED_OP_2_THREADS 512
#define FUSED_OP_3_THREADS 256

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


__global__ void mamba2_dt_op_kernel
(
    const float* __restrict__ in_dt,            // [B,S,H]
    const float* __restrict__ in_dt_bias,       // [H]
    const float* __restrict__ in_a_log,         // [H]
    bfloat16* __restrict__ out_dt,              // [B,S,H]
    float* __restrict__ out_g,                  // [B,S,H]
    int rows,
    int H,
    int rows_per_block,
    float dt_min,
    float dt_max
)
{
    int t = threadIdx.x % H;
    int row = blockIdx.x * rows_per_block + threadIdx.x / H;
    if (row >= rows) return;

    float dtv = softplus(in_dt[row * H + t] + in_dt_bias[t]);
    dtv = fminf(fmaxf(dtv, dt_min), dt_max);
    out_g[row * H + t] = -__expf(in_a_log[t]) * dtv;
    out_dt[row * H + t] = trunc_bf16(dtv);
}

/*
Mamba2 discretization: dt = clamp(softplus(dt_raw + dt_bias)), then g = dt * A with
A = -exp(A_log) as the per-head log decay, and dt itself as the input scale (the beta slot of the
recurrent rule kernel)
*/

void mamba2_dt_op
(
    const at::Tensor& dt_raw,       // [B,S,H] float
    const at::Tensor& dt_bias,      // [H] float
    const at::Tensor& a_log,        // [H] float
    at::Tensor& dt,                 // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H] float
    float dt_min,
    float dt_max
)
{
    const at::cuda::OptionalCUDAGuard device_guard(dt_raw.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(dt_raw, kFloat);
    TORCH_CHECK_DTYPE(dt_bias, kFloat);
    TORCH_CHECK_DTYPE(a_log, kFloat);
    TORCH_CHECK_DTYPE(dt, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);

    TORCH_CHECK_SHAPES(dt_raw, 2, dt_bias, 0, 1);
    TORCH_CHECK_SHAPES(dt_raw, 2, a_log, 0, 1);
    TORCH_CHECK_SHAPES_FULL(dt_raw, dt);
    TORCH_CHECK_SHAPES_FULL(dt_raw, g);

    size_t B = dt_raw.size(0);
    size_t S = dt_raw.size(1);
    size_t H = dt_raw.size(2);
    TORCH_CHECK(H <= FUSED_OP_2_THREADS, "mamba2_dt_op: too many heads");

    int rows_per_block = FUSED_OP_2_THREADS / H;
    int threads = rows_per_block * H;
    int blocks = CEIL_DIVIDE(B * S, rows_per_block);

    mamba2_dt_op_kernel<<<blocks, threads, 0, stream>>>
    (
        (const float*) dt_raw.data_ptr(),
        (const float*) dt_bias.data_ptr(),
        (const float*) a_log.data_ptr(),
        (bfloat16*) dt.data_ptr(),
        (float*) g.data_ptr(),
        B * S,
        H,
        rows_per_block,
        dt_min,
        dt_max
    );

    cuda_check(cudaPeekAtLastError());
}


// MAMBA2 mode computes the Mamba2 (SSD) recurrence, which is the gated delta rule minus the
// delta-correction readback: no q/k L2 norm, v used raw (beta = dt scales it in the update),
// output y = q.S + D*v with no 1/sqrt(dk) scale. Input layout is the conv channel order
// [x (v_dim), B (k_dim), C (k_dim)] with x->v, B->k, C->q
template <int MAX_HEAD_DIM, bool save_history, int V_SPLIT, bool MAMBA2 = false>
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
    const int history_stride,                   // max_history + 1
    const float* __restrict__ D                 // [num_v_heads], MAMBA2 only, else nullptr
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
        const bfloat16* gl_q;
        const bfloat16* gl_k;
        const bfloat16* gl_v;
        if constexpr (MAMBA2)
        {
            gl_v = mixed_qkv + head * v_head_dim + v_start;
            gl_k = mixed_qkv + num_v_heads * v_head_dim + k_head * k_head_dim;
            gl_q = mixed_qkv + num_v_heads * v_head_dim + num_k_heads * k_head_dim + k_head * k_head_dim;
        }
        else
        {
            gl_q = mixed_qkv + k_head * k_head_dim;
            gl_k = mixed_qkv + (num_k_heads + k_head) * k_head_dim;
            gl_v = mixed_qkv + (2 * num_k_heads * k_head_dim) + head * v_head_dim + v_start;
        }
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

            if constexpr (MAMBA2)
            {
                sh_k[t] = k;
                sh_q[t] = q;
            }
            else
            {
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
        }

        if constexpr (!MAMBA2)
        {
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
        }

        if (t < v_chunk_dim && bt == 0)
        {
            if constexpr (!MAMBA2)
                sh_dot1[t] = 0.0f;
            sh_dot2[t] = 0.0f;
        }
        __syncthreads();

        if constexpr (!MAMBA2)
        {
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
        }

        if (t < v_chunk_dim)
        {
            float g_h = __expf(g[head]);
            float beta_h = __bfloat162float(beta[head]);

            // Read v head; delta rule subtracts the decayed state readback, Mamba2 injects raw v
            float v = __bfloat162float(gl_v[t]);
            if constexpr (!MAMBA2)
                v -= sh_dot1[t] * g_h;

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
            if constexpr (MAMBA2)
                out[t] = __float2bfloat16_rz(v_out + D[head] * __bfloat162float(gl_v[t]));
            else
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
    const int history_stride,                   // max_history + 1
    const float* __restrict__ D                 // unused, matches the generic kernel signature
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

void cuda_recurrent_gated_delta_rule_gr
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
    bool history,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(mixed_qkv.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(!graph || slots.has_value(), "cuda_recurrent_gated_delta_rule: graph capture requires slots");

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
        history_stride,                         \
        nullptr

    // recurrent_state is kernel param 3 and slots is param 12, patched when running in a graph
    #define LAUNCH_RULE(...)                                                              \
    {                                                                                     \
        __VA_ARGS__<<<blocks, threads, 0, stream>>>(KERNEL_ARGS);                         \
        if (graph)                                                                        \
        {                                                                                 \
            graph->record_param((void*) &__VA_ARGS__, GP_gdn_rule_state, 3);              \
            graph->record_param((void*) &__VA_ARGS__, GP_gdn_rule_slots, 12);             \
            graph->record_param((void*) &__VA_ARGS__, GP_end, 0);                         \
        }                                                                                 \
    }

    if (!history)
    {
        if (k_head_dim == 128 && v_head_dim == 128)
        {
            if (v_split == 4) LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel_128<false, 4>)
            else              LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel_128<false, 1>)
        }
        else if (threads.x <= 128)
        {
            if (v_split == 4) LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<128, false, 4>)
            else              LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<128, false, 1>)
        }
        else if (threads.x <= 256)
                              LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<256, false, 1>)
        else TORCH_CHECK(false, "Max head dim exceeded");
    }
    else
    {
        if (k_head_dim == 128 && v_head_dim == 128)
        {
            if (v_split == 4) LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel_128<true, 4>)
            else              LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel_128<true, 1>)
        }
        else if (threads.x <= 128)
        {
            if (v_split == 4) LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<128, true, 4>)
            else              LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<128, true, 1>)
        }
        else if (threads.x <= 256)
                              LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<256, true, 1>)
        else TORCH_CHECK(false, "Max head dim exceeded");
    }
    #undef LAUNCH_RULE
    #undef KERNEL_ARGS

    cuda_check(cudaPeekAtLastError());
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
    cuda_recurrent_gated_delta_rule_gr(
        mixed_qkv, g, beta, recurrent_state, core_attn_out,
        num_k_heads, num_v_heads, k_head_dim, v_head_dim, slots, history, nullptr);
}

// Mamba2 decode helper for the BC graph: reads the in_proj output [z, xBC, dt] (float, row
// stride N >= v_dim + F + dt_first + H) and writes the bf16 conv input (transposed to the conv
// layout [B,F,S], like gated_delta_net_fused_op_3_kernel's out_mixed_qkv), the discretized dt/g
// tensors, and a genuinely contiguous copy of the gate (z) slice. All pointers are statics

__global__ void mamba2_fused_op_kernel
(
    const float* __restrict__ in_proj_out,      // [BS, N], N >= v_dim + F + dt_first + H
    bfloat16* __restrict__ out_xbc,             // [B, F, S]
    bfloat16* __restrict__ out_dt,              // [BS, H]
    float* __restrict__ out_g,                  // [BS, H]
    float* __restrict__ out_z_gate,             // [BS, v_dim], contiguous copy of proj[.., :v_dim]
    const float* __restrict__ dt_bias,          // [H]
    const float* __restrict__ a_log,            // [H]
    const int v_dim,
    const int F,
    const int H,
    const int dt_first,                         // rank's head offset into the (replicated) dt section
    const float dt_min,
    const float dt_max,
    const int BS,
    const int S,
    const int N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cast_elems = BS * F;

    if (idx < cast_elems)
    {
        int f = idx % F;
        int row = idx / F;                      // b * S + s
        int b = row / S;
        int s = row % S;
        out_xbc[((size_t) b * F + f) * S + s] = trunc_bf16(in_proj_out[(size_t) row * N + v_dim + f]);
    }
    else
    {
        idx -= cast_elems;
        if (idx < BS * H)
        {
            int h = idx % H;
            int row = idx / H;
            float dtv = softplus(in_proj_out[(size_t) row * N + v_dim + F + dt_first + h] + dt_bias[h]);
            dtv = fminf(fmaxf(dtv, dt_min), dt_max);
            out_g[(size_t) row * H + h] = -__expf(a_log[h]) * dtv;
            out_dt[(size_t) row * H + h] = trunc_bf16(dtv);
        }
        else
        {
            idx -= BS * H;
            if (idx >= BS * v_dim) return;
            int z = idx % v_dim;
            int row = idx / v_dim;
            out_z_gate[(size_t) row * v_dim + z] = in_proj_out[(size_t) row * N + z];
        }
    }
}

void mamba2_fused_op_gr
(
    const at::Tensor& proj,         // [B,S,N], N >= v_dim + F + dt_first + H, float
    at::Tensor& xbc,                // out [B, F, S] bfloat16
    at::Tensor& dt,                 // out [B, S, H] bfloat16
    at::Tensor& g,                  // out [B, S, H] float
    at::Tensor& z_gate,             // out [B, S, v_dim] float, contiguous
    const at::Tensor& dt_bias,      // [H] float
    const at::Tensor& a_log,        // [H] float
    int v_dim,
    int dt_first,
    float dt_min,
    float dt_max,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(proj.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(proj, kFloat);
    TORCH_CHECK_DTYPE(xbc, kBFloat16);
    TORCH_CHECK_DTYPE(dt, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);
    TORCH_CHECK_DTYPE(z_gate, kFloat);
    TORCH_CHECK_DTYPE(dt_bias, kFloat);
    TORCH_CHECK_DTYPE(a_log, kFloat);

    int B = proj.size(0);
    int S = proj.size(1);
    int N = proj.size(2);
    int F = xbc.numel() / (B * S);
    int H = dt.numel() / (B * S);
    TORCH_CHECK(N >= v_dim + F + dt_first + H, "mamba2_fused_op: proj too narrow");
    TORCH_CHECK(dt_bias.numel() == H && a_log.numel() == H && g.numel() == B * S * H, "mamba2_fused_op: bad H");
    TORCH_CHECK(z_gate.numel() == B * S * v_dim, "mamba2_fused_op: bad z_gate size");

    int BS = B * S;
    int total = BS * (F + H + v_dim);
    int blocks = CEIL_DIVIDE(total, FUSED_OP_3_THREADS);

    mamba2_fused_op_kernel<<<blocks, FUSED_OP_3_THREADS, 0, stream>>>
    (
        (const float*) proj.data_ptr(),
        (bfloat16*) xbc.data_ptr(),
        (bfloat16*) dt.data_ptr(),
        (float*) g.data_ptr(),
        (float*) z_gate.data_ptr(),
        (const float*) dt_bias.data_ptr(),
        (const float*) a_log.data_ptr(),
        v_dim,
        F,
        H,
        dt_first,
        dt_min,
        dt_max,
        BS,
        S,
        N
    );

    cuda_check(cudaPeekAtLastError());
}

/*
Mamba2 (SSD) recurrence over the conv output channels. mixed_xbc packs [x, B, C] in checkpoint
order, mapping x->v, B->k, C->q of the delta rule kernel with the correction term disabled.
num_k_heads = n_groups, k_head_dim = ssm_state_size, num_v_heads/v_head_dim = mamba heads.
dt (precomputed by mamba2_dt_op along with g = dt * A) takes the beta slot as the input scale,
and D adds the per-head skip connection y += D * x
*/

void cuda_recurrent_mamba2_gr
(
    const at::Tensor& mixed_xbc,
    const at::Tensor& g,
    const at::Tensor& dt,
    const at::Tensor& D,
    at::Tensor& recurrent_state,
    at::Tensor& core_attn_out,
    int num_k_heads,
    int num_v_heads,
    int k_head_dim,
    int v_head_dim,
    const c10::optional<at::Tensor>& slots,
    bool history,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(mixed_xbc.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(!graph || slots.has_value(), "cuda_recurrent_mamba2: graph capture requires slots");

    int bsz = mixed_xbc.size(0);
    int seqlen = mixed_xbc.size(1);
    int xbc_dim = mixed_xbc.size(2);

    TORCH_CHECK(num_v_heads % num_k_heads == 0, "num_v_heads must be divisible by num_k_heads");
    TORCH_CHECK(k_head_dim >= 32 && k_head_dim % (8 * SUBK) == 0, "k_head_dim must be a multiple of 32");
    TORCH_CHECK(MAX(k_head_dim, v_head_dim) <= 256, "Max head dim exceeded");

    TORCH_CHECK(xbc_dim == 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim,
                "mixed_xbc must be [bsz, seqlen, num_v_heads*v_head_dim + 2*num_k_heads*k_head_dim]");
    TORCH_CHECK(g.dim() == 3 && g.size(0) == bsz && g.size(1) == seqlen && g.size(2) == num_v_heads,
                "g must be [bsz, seqlen, num_v_heads]");
    TORCH_CHECK(dt.dim() == 3 && dt.size(0) == bsz && dt.size(1) == seqlen && dt.size(2) == num_v_heads,
                "dt must be [bsz, seqlen, num_v_heads]");
    TORCH_CHECK(D.dim() == 1 && D.size(0) == num_v_heads, "D must be [num_v_heads]");
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

    TORCH_CHECK_DTYPE(mixed_xbc, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);
    TORCH_CHECK_DTYPE(dt, kBFloat16);
    TORCH_CHECK_DTYPE(D, kFloat);
    TORCH_CHECK_DTYPE(recurrent_state, kFloat);
    TORCH_CHECK_DTYPE(core_attn_out, kBFloat16);
    TORCH_CHECK_DTYPE_OPT(slots, kInt);

    const int* slots_ptr = (const int*) OPTPTR(slots);
    if (slots_ptr)
    {
        TORCH_CHECK(slots.value().dim() == 1 &&
                    slots.value().size(0) == bsz,
                    "slots must be [bsz]");
        TORCH_CHECK(slots.value().device() == mixed_xbc.device(),
                    "slots must be on the same device as mixed_xbc");
    }

    dim3 blocks(bsz, num_v_heads, 1);
    dim3 threads(MAX(k_head_dim, v_head_dim), SUBK);
    int history_stride = recurrent_state.size(1);

    float scale = 1.0f;  // unused in MAMBA2 mode

    #define KERNEL_ARGS                         \
        (const bfloat16*) mixed_xbc.data_ptr(), \
        (const float*) g.data_ptr(),            \
        (const bfloat16*) dt.data_ptr(),        \
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
        history_stride,                         \
        (const float*) D.data_ptr()

    // recurrent_state is kernel param 3 and slots is param 12, patched when running in a graph
    #define LAUNCH_RULE(...)                                                              \
    {                                                                                     \
        __VA_ARGS__<<<blocks, threads, 0, stream>>>(KERNEL_ARGS);                         \
        if (graph)                                                                        \
        {                                                                                 \
            graph->record_param((void*) &__VA_ARGS__, GP_gdn_rule_state, 3);              \
            graph->record_param((void*) &__VA_ARGS__, GP_gdn_rule_slots, 12);             \
            graph->record_param((void*) &__VA_ARGS__, GP_end, 0);                         \
        }                                                                                 \
    }

    if (!history)
    {
        if (threads.x <= 128)      LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<128, false, 1, true>)
        else if (threads.x <= 256) LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<256, false, 1, true>)
        else TORCH_CHECK(false, "Max head dim exceeded");
    }
    else
    {
        if (threads.x <= 128)      LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<128, true, 1, true>)
        else if (threads.x <= 256) LAUNCH_RULE(cuda_recurrent_gated_delta_rule_kernel<256, true, 1, true>)
        else TORCH_CHECK(false, "Max head dim exceeded");
    }
    #undef LAUNCH_RULE
    #undef KERNEL_ARGS

    cuda_check(cudaPeekAtLastError());
}

void cuda_recurrent_mamba2
(
    const at::Tensor& mixed_xbc,
    const at::Tensor& g,
    const at::Tensor& dt,
    const at::Tensor& D,
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
    cuda_recurrent_mamba2_gr(
        mixed_xbc, g, dt, D, recurrent_state, core_attn_out,
        num_k_heads, num_v_heads, k_head_dim, v_head_dim, slots, history, nullptr);
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

void cuda_causal_conv1d_update_gr
(
    const at::Tensor& x,
    at::Tensor& conv_state,
    const c10::optional<at::Tensor>& slots,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    bool activation,
    bool history,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(!graph || slots.has_value(), "cuda_causal_conv1d_update: graph capture requires slots");

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

    // conv_state is kernel param 1 and slots is param 2, patched when running in a graph
    #define LAUNCH_CONV(...)                                                              \
    {                                                                                     \
        __VA_ARGS__<<<blocks, CONV1D_NUM_THREADS, 0, stream>>>(KERNEL_ARGS);              \
        if (graph)                                                                        \
        {                                                                                 \
            graph->record_param((void*) &__VA_ARGS__, GP_conv1d_state, 1);                \
            graph->record_param((void*) &__VA_ARGS__, GP_conv1d_slots, 2);                \
            graph->record_param((void*) &__VA_ARGS__, GP_end, 0);                         \
        }                                                                                 \
    }

    if (activation)
    {
        if (history) LAUNCH_CONV(conv1d_update_kernel<true, true>)
        else         LAUNCH_CONV(conv1d_update_kernel<true, false>)
    }
    else
    {
        if (history) LAUNCH_CONV(conv1d_update_kernel<false, true>)
        else         LAUNCH_CONV(conv1d_update_kernel<false, false>)
    }
    #undef LAUNCH_CONV
    #undef KERNEL_ARGS

    cuda_check(cudaPeekAtLastError());
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
    cuda_causal_conv1d_update_gr(x, conv_state, slots, weight, bias, out, activation, history, nullptr);
}

// Split-projection (Qwen3.5) decode helper. Replaces the qkv transpose/cast done in Torch plus
// gated_delta_net_fused_op_2: mixed_qkv[b,f,s] = bf16(qkv[b,s,f]), and beta/g computed from the
// packed ba projection (b = ba[..,:H], a = ba[..,H:])

template <typename a_log_T>
__global__ void gated_delta_net_fused_op_3_kernel
(
    const float* __restrict__ in_qkv,           // [B,S,F]
    const float* __restrict__ in_ba,            // [B,S,2H]
    const bfloat16* __restrict__ in_dt_bias,    // [H]
    const a_log_T* __restrict__ in_a_log,       // [H]
    bfloat16* __restrict__ out_mixed_qkv,       // [B,F,S]
    bfloat16* __restrict__ out_beta,            // [B,S,H]
    float* __restrict__ out_g,                  // [B,S,H]
    const int BS,                               // B * S
    const int S,
    const int F,
    const int H,
    const float beta_scale
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int cast_elems = BS * F;

    if (idx < cast_elems)
    {
        int f = idx % F;
        int row = idx / F;                      // b * S + s
        int b = row / S;
        int s = row % S;
        out_mixed_qkv[((size_t) b * F + f) * S + s] = trunc_bf16(in_qkv[idx]);
    }
    else
    {
        idx -= cast_elems;
        if (idx >= BS * H) return;
        int h = idx % H;
        int row = idx / H;

        float bv = in_ba[(size_t) row * 2 * H + h];
        float av = in_ba[(size_t) row * 2 * H + H + h];
        float beta = _sigmoid_fast_exp(bv) * beta_scale;
        float dt_bias = as_float(in_dt_bias[h]);
        float gv = -softplus(av + dt_bias) * __expf(as_float(in_a_log[h]));

        out_beta[(size_t) row * H + h] = trunc_bf16(beta);
        out_g[(size_t) row * H + h] = gv;
    }
}

void gated_delta_net_fused_op_3_gr
(
    const at::Tensor& qkv,          // [B,S,F] float
    const at::Tensor& ba,           // [B,S,2H] float
    const at::Tensor& dt_bias,      // [H] bfloat16
    const at::Tensor& a_log,        // [H] float or bfloat16
    at::Tensor& mixed_qkv,          // out [B,F,S] bfloat16
    at::Tensor& beta,               // out [B,S,H] bfloat16
    at::Tensor& g,                  // out [B,S,H] float
    const float beta_scale,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(qkv.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(qkv, kFloat);
    TORCH_CHECK_DTYPE(ba, kFloat);
    TORCH_CHECK_DTYPE(dt_bias, kBFloat16);
    TORCH_CHECK_DTYPE(mixed_qkv, kBFloat16);
    TORCH_CHECK_DTYPE(beta, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);

    int B = qkv.size(0);
    int S = qkv.size(1);
    int F = qkv.size(2);
    int H = beta.size(-1);

    TORCH_CHECK(ba.size(-1) == 2 * H, "ba must be [B,S,2H]");
    TORCH_CHECK(ba.numel() == B * S * 2 * H, "ba must be [B,S,2H]");
    TORCH_CHECK(mixed_qkv.numel() == B * S * F, "mixed_qkv must be [B,F,S]");
    TORCH_CHECK(beta.numel() == B * S * H, "beta must be [B,S,H]");
    TORCH_CHECK(g.numel() == B * S * H, "g must be [B,S,H]");

    bool a_log_fp32 = a_log.dtype() == at::kFloat;
    bool a_log_bf16 = a_log.dtype() == at::kBFloat16;

    int BS = B * S;
    int total = BS * (F + H);
    int blocks = CEIL_DIVIDE(total, FUSED_OP_3_THREADS);

    #define ARGS(a_log_T)                       \
        (const float*) qkv.data_ptr(),          \
        (const float*) ba.data_ptr(),           \
        (const bfloat16*) dt_bias.data_ptr(),   \
        (const a_log_T*) a_log.data_ptr(),      \
        (bfloat16*) mixed_qkv.data_ptr(),       \
        (bfloat16*) beta.data_ptr(),            \
        (float*) g.data_ptr(),                  \
        BS, S, F, H,                            \
        beta_scale

    if (a_log_fp32)
        gated_delta_net_fused_op_3_kernel<<<blocks, FUSED_OP_3_THREADS, 0, stream>>>(ARGS(float));
    else if (a_log_bf16)
        gated_delta_net_fused_op_3_kernel<<<blocks, FUSED_OP_3_THREADS, 0, stream>>>(ARGS(bfloat16));
    else TORCH_CHECK(false, "gated_delta_net_fused_op_3: unsupported a_log dtype");

    #undef ARGS

    cuda_check(cudaPeekAtLastError());
}

// Small fp16 GEMV with fp32 accumulation/output for the merged b/a projections. One warp per
// output feature; n is tiny (2 * num_v_heads) so this is launch-bound anyway. Kept out of cublas
// so the x pointer is patchable in captured graphs

#define BA_GEMV_WARPS 8

__global__ __launch_bounds__(BA_GEMV_WARPS * 32)
void gdn_ba_gemv_kernel
(
    const half* __restrict__ x,                 // [rows, k]
    const half* __restrict__ w_t,               // [n, k]
    const half* __restrict__ bias,              // [n] or null
    float* __restrict__ y,                      // [rows, n]
    const int k,
    const int n
)
{
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * BA_GEMV_WARPS + warp;
    if (row >= n) return;
    int r = blockIdx.y;

    const half2* x2 = (const half2*) (x + (size_t) r * k);
    const half2* w2 = (const half2*) (w_t + (size_t) row * k);

    float sum = 0.0f;
    for (int j = lane; j < k / 2; j += 32)
    {
        float2 xf = __half22float2(x2[j]);
        float2 wf = __half22float2(w2[j]);
        sum = fmaf(xf.x, wf.x, sum);
        sum = fmaf(xf.y, wf.y, sum);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0)
    {
        if (bias) sum += __half2float(bias[row]);
        y[(size_t) r * n + row] = sum;
    }
}

void gdn_ba_gemv_gr
(
    const at::Tensor& x,            // [rows, k] half
    const at::Tensor& w_t,          // [n, k] half
    const c10::optional<at::Tensor>& bias,  // [n] half
    at::Tensor& y,                  // [rows, n] float
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w_t, kHalf);
    TORCH_CHECK_DTYPE_OPT(bias, kHalf);
    TORCH_CHECK_DTYPE(y, kFloat);

    int k = x.size(-1);
    int n = w_t.size(0);
    int rows = (int) (x.numel() / k);
    TORCH_CHECK(w_t.dim() == 2 && w_t.size(1) == k, "w_t must be [n, k]");
    TORCH_CHECK(y.numel() == (int64_t) rows * n, "y must be [rows, n]");
    TORCH_CHECK(k % 2 == 0, "k must be even");
    TORCH_CHECK(x.is_contiguous() && w_t.is_contiguous() && y.is_contiguous(), "tensors must be contiguous");

    const half* bias_ptr = (const half*) OPTPTR(bias);
    dim3 blocks(CEIL_DIVIDE(n, BA_GEMV_WARPS), rows);

    gdn_ba_gemv_kernel<<<blocks, BA_GEMV_WARPS * 32, 0, stream>>>
    (
        (const half*) x.data_ptr(),
        (const half*) w_t.data_ptr(),
        bias_ptr,
        (float*) y.data_ptr(),
        k, n
    );

    if (graph)
    {
        graph->record_param((void*) &gdn_ba_gemv_kernel, GP_gdn_ba_x, 0);
        graph->record_param((void*) &gdn_ba_gemv_kernel, GP_end, 0);
    }

    cuda_check(cudaPeekAtLastError());
}

void gdn_ba_gemv
(
    const at::Tensor& x,
    const at::Tensor& w_t,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& y
)
{
    gdn_ba_gemv_gr(x, w_t, bias, y, nullptr);
}

// Batched rewind kernels: collapse the per-recurrent-layer rewind loop (speculative decoding
// draft rejection/commit) into a couple of launches instead of one launch per layer

#define REWIND_MAX_JOBS 64
#define REWIND_CONV_THREADS 256
#define REWIND_STATE_THREADS 256

struct ConvRewindJobBatch { ConvRewindJob jobs[REWIND_MAX_JOBS]; int num_jobs; };
struct StateRewindJobBatch { StateRewindJob jobs[REWIND_MAX_JOBS]; int num_jobs; };

// conv_state[slot, :, :cdim] <- conv_state[slot, :, p-cdim:p], one thread per channel. Reads its
// (up to CONV1D_MAX_K) elements into registers before writing any of them back, so the copy is
// safe even when src/dst windows overlap (num_tokens < conv_kernel_size) -- no synchronization
// needed since channels are independent and a single thread's read-then-write is self-ordered.
__global__ __launch_bounds__(REWIND_CONV_THREADS)
void batched_conv_rewind_kernel(ConvRewindJobBatch batch)
{
    int job_idx = blockIdx.y;
    if (job_idx >= batch.num_jobs) return;
    ConvRewindJob j = batch.jobs[job_idx];

    int d = blockIdx.x * REWIND_CONV_THREADS + threadIdx.x;
    if (d >= j.dim) return;

    const bfloat16* s = (const bfloat16*) j.src + (size_t) d * j.stride;
    bfloat16* t = (bfloat16*) j.dst + (size_t) d * j.stride;

    bfloat16 reg[CONV1D_MAX_K];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < j.cdim) reg[k] = s[k];
    #pragma unroll
    for (int k = 0; k < CONV1D_MAX_K; ++k)
        if (k < j.cdim) t[k] = reg[k];
}

// recurrent_state[slot, 0] <- recurrent_state[slot, last_history+1-num_tokens], flat fp32 copy,
// vectorized as float4. Source and destination never overlap for this one (see ConvRewindJob
// comment in gdn.cuh), so no read-before-write ordering concern here at all.
__global__ __launch_bounds__(REWIND_STATE_THREADS)
void batched_state_rewind_kernel(StateRewindJobBatch batch)
{
    int job_idx = blockIdx.y;
    if (job_idx >= batch.num_jobs) return;
    StateRewindJob j = batch.jobs[job_idx];

    int64_t i4 = (int64_t) blockIdx.x * REWIND_STATE_THREADS + threadIdx.x;
    int64_t n4 = j.num_elements / 4;
    if (i4 >= n4) return;

    ((float4*) j.dst)[i4] = ((const float4*) j.src)[i4];
}

void batched_conv_rewind(std::vector<ConvRewindJob> const& jobs, int device_index)
{
    if (jobs.empty()) return;
    c10::cuda::CUDAGuard device_guard((c10::DeviceIndex) device_index);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    for (size_t base = 0; base < jobs.size(); base += REWIND_MAX_JOBS)
    {
        int n = (int) MIN(jobs.size() - base, (size_t) REWIND_MAX_JOBS);
        ConvRewindJobBatch batch;
        batch.num_jobs = n;
        int max_dim = 0;
        for (int i = 0; i < n; ++i)
        {
            batch.jobs[i] = jobs[base + i];
            TORCH_CHECK(batch.jobs[i].cdim <= CONV1D_MAX_K, "batched_conv_rewind: cdim exceeds CONV1D_MAX_K");
            max_dim = MAX(max_dim, batch.jobs[i].dim);
        }

        dim3 blocks(CEIL_DIVIDE(max_dim, REWIND_CONV_THREADS), n);
        batched_conv_rewind_kernel<<<blocks, REWIND_CONV_THREADS, 0, stream>>>(batch);
        cuda_check(cudaPeekAtLastError());
    }
}

void batched_state_rewind(std::vector<StateRewindJob> const& jobs, int device_index)
{
    if (jobs.empty()) return;
    c10::cuda::CUDAGuard device_guard((c10::DeviceIndex) device_index);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    for (size_t base = 0; base < jobs.size(); base += REWIND_MAX_JOBS)
    {
        int n = (int) MIN(jobs.size() - base, (size_t) REWIND_MAX_JOBS);
        StateRewindJobBatch batch;
        batch.num_jobs = n;
        int64_t max_elems = 0;
        for (int i = 0; i < n; ++i)
        {
            batch.jobs[i] = jobs[base + i];
            TORCH_CHECK(batch.jobs[i].num_elements % 4 == 0, "batched_state_rewind: num_elements must be a multiple of 4");
            max_elems = MAX(max_elems, batch.jobs[i].num_elements);
        }

        dim3 blocks(CEIL_DIVIDE((int)(max_elems / 4), REWIND_STATE_THREADS), n);
        batched_state_rewind_kernel<<<blocks, REWIND_STATE_THREADS, 0, stream>>>(batch);
        cuda_check(cudaPeekAtLastError());
    }
}
