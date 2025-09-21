#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "activation.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "compat.cuh"
#include <cmath>

__device__ __forceinline__ float _sigmoid_fast_exp(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ __nv_bfloat16 trunc_bf16(float x)
{
    return __float2bfloat16_rn(x);
}

__device__ __forceinline__ float untrunc_bf16(__nv_bfloat16 x)
{
    return __bfloat162float(x);
}

__device__ __forceinline__ float softplus(float x)  // beta=1.0, linear threshold=20.0
{
    if (x > 20.0f) return x;
    return log1pf(__expf(x));
}

__global__ void gated_delta_net_fused_op_kernel
(
    const float* __restrict__ in_qkvz,          // [B,S,Nk, Fseg], float32
    const float* __restrict__ in_ba,            // [B,S,Nk, 2*Ng], float32
    const __nv_bfloat16* __restrict__ dt_bias,  // [Nv], bfloat16
    const __nv_bfloat16* __restrict__ a_log,    // [Nv], bfloat16
    __nv_bfloat16* __restrict__ out_qkv,        // [B, 2*Nk*Hk + Nv*Hv, S], bfloat16
    __nv_bfloat16* __restrict__ out_z,          // [B, S, Nv, Hv], bfloat16
    __nv_bfloat16* __restrict__ out_beta,       // [B, S, Nv], bfloat16
    float* __restrict__ out_g,                  // [B, S, Nv], float32
    size_t B,
    size_t S,
    size_t Nk,
    size_t Ng,
    size_t Hk,
    size_t Hv
){
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

    gated_delta_net_fused_op_kernel<<<blocks, threads, 0, stream>>>
    (
        (const float*) mixed_qkvz.data_ptr(),
        (const float*) mixed_ba.data_ptr(),
        (const __nv_bfloat16*) dt_bias.data_ptr(),
        (const __nv_bfloat16*) a_log.data_ptr(),
        (__nv_bfloat16*) mixed_qkv.data_ptr(),
        (__nv_bfloat16*) z.data_ptr(),
        (__nv_bfloat16*) beta.data_ptr(),
        (float*) g.data_ptr(),
        B, S, Nk, Ng, Hk, Hv
    );

    cuda_check(cudaPeekAtLastError());
}