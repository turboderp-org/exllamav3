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
#define MAX_HEAD_DIM 128
#define MAX_K_HEADS 16
#define MAX_V_HEADS 32

#define R_THREADS MAX_HEAD_DIM
#define SUBK 4

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

__device__ __forceinline__ float softplus(float x)  // beta=1.0, linear threshold=20.0
{
    if (x > 20.0f) return x;
    return log1pf(__expf(x));
}

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
    const size_t Hv
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
            out_beta[out_va_off] = trunc_bf16(_sigmoid_fast_exp(b));

            // g = -self.a_log.float().exp() * F.softplus(a + self.dt_bias.float())
            float g = in_ba[base_ba + Ng + t];
            float bi = untrunc_bf16(dt_bias[out_va_off % Nv]);
            float al = untrunc_bf16(a_log[out_va_off % Nv]);
            out_g[out_va_off] = -softplus(g + bi) * expf(al);
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
    size_t v_head_dim
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
    TORCH_CHECK(threads <= MAX_HEAD_DIM, "Max head dim exceeded");

    gated_delta_net_fused_op_kernel<<<blocks, threads, 0, stream>>>
    (
        (const float*) mixed_qkvz.data_ptr(),
        (const float*) mixed_ba.data_ptr(),
        (const bfloat16*) dt_bias.data_ptr(),
        (const bfloat16*) a_log.data_ptr(),
        (bfloat16*) mixed_qkv.data_ptr(),
        (bfloat16*) z.data_ptr(),
        (bfloat16*) beta.data_ptr(),
        (float*) g.data_ptr(),
        B, S, Nk, Ng, Hk, Hv
    );

    cuda_check(cudaPeekAtLastError());
}


__global__ __launch_bounds__(R_THREADS * SUBK)
void cuda_recurrent_gated_delta_rule_kernel
(
    // k_dim = num_k_heads * k_head_dim
    // v_dim = num_v_heads * v_head_dim
    const bfloat16* __restrict__ mixed_qkv,     // [bsz, seqlen, (k_dim + k_dim + v_dim)]
    const float* __restrict__ g,                // [bsz, seqlen, (group * num_k_heads)]
    const bfloat16* __restrict__ beta,          // [bsz, seqlen, (group * num_k_heads)]
    float* __restrict__ recurrent_state,        // [bsz, (group * num_k_heads), k_head_dim, v_head_dim]
    bfloat16* __restrict__ core_attn_out,       // [bsz, seqlen, num_v_heads, v_head_dim]
    const int bsz,
    const int seqlen,
    const int num_k_heads,
    const int num_v_heads,
    const int k_head_dim,
    const int v_head_dim,
    const float scale
)
{
    int group = num_v_heads / num_k_heads;

    // Advance to batch item
    int bi = blockIdx.x;
    mixed_qkv +=        bi * seqlen * (2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads);
    g +=                bi * seqlen * (group * num_k_heads);
    beta +=             bi * seqlen * (group * num_k_heads);
    recurrent_state +=  bi * (group * num_k_heads) * k_head_dim * v_head_dim;
    core_attn_out +=    bi * seqlen * num_v_heads * v_head_dim;

    // Indexing
    int t = threadIdx.x;
    int bt = threadIdx.y;
    int bts = k_head_dim / SUBK;
    int lane = t % 32;
    int warp = t / 32;
    int head = blockIdx.y;
    int k_head = head / group;

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
        const bfloat16* gl_v = mixed_qkv + (2 * num_k_heads * k_head_dim) + head * v_head_dim;
        bfloat16* out = core_attn_out + head * v_head_dim;
        float* gl_rs = recurrent_state + head * (k_head_dim * v_head_dim);

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

        if (t < v_head_dim && bt == 0)
        {
            sh_dot1[t] = 0.0f;
            sh_dot2[t] = 0.0f;
        }
        __syncthreads();

        if (t < v_head_dim)
        {
            // Dot products with last state
            float sum = 0.0f;
            float* sh_k_rd = sh_k + bt * bts;
            float* rs_rd = gl_rs + t + bt * bts * v_head_dim;

            // TODO: Could use tensor cores
            for (int i = 0; i < k_head_dim / 16 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 16; ++j, rs_rd += v_head_dim, sh_k_rd++)
                    sum = sum + *sh_k_rd * *rs_rd;
            }
            atomicAdd(sh_dot1 + t, sum);
        }
        __syncthreads();

        if (t < v_head_dim)
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
            float* rs_rw = gl_rs + t + bt * bts * v_head_dim;

            // TODO: Could use tensor cores
            for (int i = 0; i < k_head_dim / 16 / SUBK; ++i)
            {
                #pragma unroll
                for (int j = 0; j < 16; ++j, rs_rw += v_head_dim, sh_k_rd++, sh_q_rd++)
                {
                    // State update step, k x v
                    float state = *rs_rw;
                    state = state * g_h + *sh_k_rd * v * beta_h;
                    *rs_rw = state;

                    // Accumulate attn output
                    v_out = v_out + *sh_q_rd * state;
                }
            }
            atomicAdd(sh_dot2 + t, v_out);
        }
        __syncthreads();

        if (t < v_head_dim && bt == 0)
        {
            float v_out = sh_dot2[t];

            // Store attn output
            if (bt == 0)
                out[t] = __float2bfloat16_rz(v_out * scale);
        }

        // Next seq index
        mixed_qkv +=        2 * k_head_dim * num_k_heads + v_head_dim * num_v_heads;
        g +=                num_v_heads;
        beta +=             num_v_heads;
        core_attn_out +=    num_v_heads * v_head_dim;
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
    int v_head_dim
)
{
    const at::cuda::OptionalCUDAGuard device_guard(mixed_qkv.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int bsz = mixed_qkv.size(0);
    int seqlen = mixed_qkv.size(1);
    int qkv_dim = mixed_qkv.size(2);

    TORCH_CHECK(num_k_heads <= MAX_K_HEADS, "num_k_heads > MAX_K_HEADS");
    TORCH_CHECK(num_v_heads <= MAX_V_HEADS, "num_v_heads > MAX_V_HEADS");
    TORCH_CHECK(k_head_dim <= MAX_HEAD_DIM, "k_head_dim > MAX_HEAD_DIM");
    TORCH_CHECK(v_head_dim <= MAX_HEAD_DIM, "v_head_dim > MAX_HEAD_DIM");

    TORCH_CHECK_DTYPE(mixed_qkv, kBFloat16);
    TORCH_CHECK_DTYPE(g, kFloat);
    TORCH_CHECK_DTYPE(beta, kBFloat16);
    TORCH_CHECK_DTYPE(recurrent_state, kFloat);
    TORCH_CHECK_DTYPE(core_attn_out, kBFloat16);

    dim3 blocks(bsz, num_v_heads);  // group * num_k_heads
    dim3 threads(MAX(k_head_dim, v_head_dim), SUBK);

    float scale = 1.0f / sqrtf(k_head_dim);

    cuda_recurrent_gated_delta_rule_kernel<<<blocks, threads, 0, stream>>>
    (
        (const bfloat16*) mixed_qkv.data_ptr(),
        (const float*) g.data_ptr(),
        (const bfloat16*) beta.data_ptr(),
        (float*) recurrent_state.data_ptr(),
        (bfloat16*) core_attn_out.data_ptr(),
        bsz,
        seqlen,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        scale
    );
}
