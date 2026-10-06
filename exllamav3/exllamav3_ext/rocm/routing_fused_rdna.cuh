#pragma once

// RDNA: DS3-style no-group routing at bsz 1 as one launch. The two-launch form is the router GEMV and a one-block
// top-k over its scores, small enough that the dispatch gap between them is a large part of the cost, once per MoE
// layer per token. Here the GEMV blocks store their scores, and the last block to arrive (release fence + counter,
// acquire fence) runs routing_ds3_nogroup_topk_kernel's body verbatim on the stored scores: same keys, payloads,
// warp_topk_shared and normalization, so indices and weights are bit-identical to the two-launch form. The counter
// resets itself (no memset, nothing to patch under graph capture), and the last-block pattern is wait-free.
// Included by routing.cu after the top-k helpers it reuses.

__device__ unsigned int g_routing_fused_counter;

template <int ACT, int U>
__global__ __launch_bounds__(RGW_WARPS * 32)
void routing_ds3_nogroup_fused_kernel
(
    const half* __restrict__ x,         // (k)
    const half* __restrict__ gate_t,    // (E, k)
    half* __restrict__ scores,          // (E)
    const half* __restrict__ bias,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const float scaling_factor,
    const int k,
    const int num_experts,
    const int K
)
{
    extern __shared__ uint32_t rg_sh[];
    __shared__ int sh_last;
    const int t = threadIdx.x;
    const int lane_id = t % 32;
    const int warp_id = t / 32;
    const int row = blockIdx.x * RGW_WARPS + warp_id;

    float sum = routing_gemv_rdna_w16_row<U>(x, gate_t, k, row, row < num_experts, rg_sh);
    if (row < num_experts && lane_id == 0)
        scores[row] = __float2half_rn(sum);

    // Arrive: every thread's stores are released before thread 0 counts the block
    __threadfence();
    __syncthreads();
    if (t == 0)
    {
        unsigned int old = atomicAdd(&g_routing_fused_counter, 1u);
        sh_last = (old == gridDim.x - 1);
    }
    __syncthreads();
    if (!sh_last) return;
    __threadfence();
    if (t == 0) g_routing_fused_counter = 0u;

    // routing_ds3_nogroup_topk_kernel, row 0, over the whole block's threads. The keys reuse the x staging
    // area of the LDS (the host checks it holds 2 * E floats)
    float* sh_key = reinterpret_cast<float*>(rg_sh);
    float* sh_payload = sh_key + num_experts;
    __syncthreads();
    for (int e = t; e < num_experts; e += RGW_WARPS * 32)
    {
        float logit = __half2float(scores[e]);
        float act = bias ? routing_act<ACT>(logit) : 0.0f;
        sh_key[e] = bias ? act + __half2float(bias[e]) : logit;
        sh_payload[e] = bias ? act : logit;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float o;
        int out_idx;
        warp_topk_shared(sh_key, sh_payload, num_experts, K, o, out_idx);
        if (lane_id < K && !bias)
            o = routing_act<ACT>(o);

        float s = warp_reduce_sum_first_k(o, K) + 1e-20f;
        if (lane_id < K)
        {
            topk_indices[lane_id] = (int64_t) out_idx;
            topk_weights[lane_id] = __float2half_rn(o * scaling_factor / s);
        }
    }
}

// Runs the fused form when the call fits it (bsz 1, transposed gate, shapes and alignment); false otherwise,
// with nothing launched
static bool routing_ds3_nogroup_fused_try
(
    const at::Tensor& hidden,
    const c10::optional<at::Tensor>& gate_t,
    at::Tensor& scores,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& topk_indices,
    at::Tensor& topk_weights,
    const float scaling_factor,
    const int act_fn,
    cudaStream_t stream
)
{
    if (!gate_t.has_value() || !hidden.is_contiguous() || !gate_t.value().is_contiguous()) return false;
    const int k = hidden.size(-1);
    const int E = scores.size(-1);
    if (hidden.numel() != k || scores.numel() != E || topk_indices.size(0) != 1) return false;
    if (hidden.dtype() != at::kHalf || gate_t.value().dtype() != at::kHalf) return false;
    if (scores.dtype() != at::kHalf || topk_indices.dtype() != at::kLong || topk_weights.dtype() != at::kHalf) return false;
    if (bias.has_value() && bias.value().dtype() != at::kHalf) return false;
    const int K = topk_indices.size(1);
    if (E > MAX_NUM_EXPERTS || K > MAX_K || K > E || K > 32) return false;
    if (k & 7) return false;
    const half* xp = (const half*) hidden.data_ptr();
    const half* wp = (const half*) gate_t.value().data_ptr();
    if (((uintptr_t) xp | (uintptr_t) wp) & 15) return false;
    if (k / 2 < 2 * E) return false;
    const size_t smem = (size_t) (k / 2 + RGW_WARPS * RGW_U * 128) * 4;
    if (smem > 48 * 1024) return false;

    const dim3 grid(CEIL_DIVIDE(E, RGW_WARPS));
    const dim3 block(RGW_WARPS * 32);
    const half* bp = bias.has_value() ? (const half*) bias.value().data_ptr() : nullptr;
    if (act_fn == ROUTING_ACT_SQRTSP)
        routing_ds3_nogroup_fused_kernel<ROUTING_ACT_SQRTSP, RGW_U><<<grid, block, smem, stream>>>
            (xp, wp, (half*) scores.data_ptr(), bp, (int64_t*) topk_indices.data_ptr(), (half*) topk_weights.data_ptr(),
             scaling_factor, k, E, K);
    else
        routing_ds3_nogroup_fused_kernel<ROUTING_ACT_SIGMOID, RGW_U><<<grid, block, smem, stream>>>
            (xp, wp, (half*) scores.data_ptr(), bp, (int64_t*) topk_indices.data_ptr(), (half*) topk_weights.data_ptr(),
             scaling_factor, k, E, K);
    cuda_check(cudaPeekAtLastError());
    return true;
}
