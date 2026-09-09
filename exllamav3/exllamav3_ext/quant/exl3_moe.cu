#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_devctx.cuh"
#include <algorithm>
#include <cstdint>
#include <set>

// The fused route histogram/scan/stable-pack pipeline is adapted from
// @brandonmmusic-max's draft PR #246 (commit 704aefd), with sentinel routing
// added for expert maps and the additive execution path introduced here.
template <typename id_t>
__device__ __forceinline__ int64_t exl3_route_expert
(
    id_t route_id,
    const int64_t* __restrict__ expert_map,
    int64_t expert_map_size,
    int num_buckets
)
{
    int64_t id = static_cast<int64_t>(route_id);
    if (id < 0 || id >= expert_map_size) return num_buckets - 1;
    int64_t expert = expert_map[id];
    if (expert < 0 || expert >= num_buckets) return num_buckets - 1;
    return expert;
}

template <typename id_t>
__global__ void exl3_route_histogram_kernel
(
    const id_t* __restrict__ topk_ids,
    const int64_t* __restrict__ expert_map,
    int64_t* __restrict__ expert_count,
    int64_t num_routes,
    int64_t expert_map_size,
    int num_buckets
)
{
    for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
         r < num_routes; r += (int64_t) blockDim.x * gridDim.x)
    {
        int64_t e = exl3_route_expert(
            topk_ids[r], expert_map, expert_map_size, num_buckets
        );
        atomicAdd(reinterpret_cast<unsigned long long*>(expert_count + e), 1ULL);
    }
}

__global__ void exl3_route_scan_kernel
(
    const int64_t* __restrict__ expert_count,
    int64_t* __restrict__ expert_offsets,
    int num_buckets
)
{
    if (blockIdx.x || threadIdx.x) return;
    int64_t sum = 0;
    for (int e = 0; e < num_buckets; ++e)
    {
        expert_offsets[e] = sum;
        sum += expert_count[e];
    }
}

template <typename weight_t>
__device__ __forceinline__ half exl3_route_to_half(weight_t v)
{
    return __float2half(static_cast<float>(v));
}

template <>
__device__ __forceinline__ half exl3_route_to_half<half>(half v)
{
    return v;
}

template <>
__device__ __forceinline__ half exl3_route_to_half<__nv_bfloat16>
(
    __nv_bfloat16 v
)
{
    return __float2half(__bfloat162float(v));
}

template <typename id_t, typename weight_t>
__global__ void exl3_route_pack_stable_kernel
(
    const id_t* __restrict__ topk_ids,
    const weight_t* __restrict__ topk_weights,
    const int64_t* __restrict__ expert_map,
    const int64_t* __restrict__ expert_offsets,
    int64_t* __restrict__ token_sorted,
    half* __restrict__ weight_sorted,
    int64_t num_routes,
    int topk,
    int64_t expert_map_size,
    int num_buckets
)
{
    // One block per local expert, including the sentinel. Each block scans
    // routes in source order to preserve stable expert-grouped ordering.
    __shared__ int flags[256];
    __shared__ int running;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();
    const int64_t expert = blockIdx.x;
    for (int64_t base = 0; base < num_routes; base += blockDim.x)
    {
        int64_t r = base + threadIdx.x;
        int flag = 0;
        if (r < num_routes)
            flag = exl3_route_expert(
                topk_ids[r], expert_map, expert_map_size, num_buckets
            ) == expert;
        flags[threadIdx.x] = flag;
        __syncthreads();
        for (int stride = 1; stride < blockDim.x; stride <<= 1)
        {
            int v = threadIdx.x >= stride ? flags[threadIdx.x - stride] : 0;
            __syncthreads();
            flags[threadIdx.x] += v;
            __syncthreads();
        }
        if (flag)
        {
            int64_t dst = expert_offsets[expert] + running
                        + flags[threadIdx.x] - 1;
            token_sorted[dst] = r / topk;
            weight_sorted[dst] = exl3_route_to_half(topk_weights[r]);
        }
        __syncthreads();
        if (threadIdx.x == 0) running += flags[blockDim.x - 1];
        __syncthreads();
    }
}

int exl3_moe_max_concurrency(int device)
{
    int num_sms = DevCtx::instance().get_num_sms(device);
    return num_sms / MOE_SMS_PER_EXPERT;
}

std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

fp_exl3_moe_kernel exl3_moe_kernel_instances[] =
{
    // [K][cb - 1][N_off]: K = 0 switches Kg/Ku/Kd at runtime, K > 0 = compile-time Kg = Ku = Kd
    exl3_moe_kernel_k0_n128_cb1(), exl3_moe_kernel_k0_n256_cb1(), exl3_moe_kernel_k0_n128_cb2(), exl3_moe_kernel_k0_n256_cb2(),
    exl3_moe_kernel_k1_n128_cb1(), exl3_moe_kernel_k1_n256_cb1(), exl3_moe_kernel_k1_n128_cb2(), exl3_moe_kernel_k1_n256_cb2(),
    exl3_moe_kernel_k2_n128_cb1(), exl3_moe_kernel_k2_n256_cb1(), exl3_moe_kernel_k2_n128_cb2(), exl3_moe_kernel_k2_n256_cb2(),
    exl3_moe_kernel_k3_n128_cb1(), exl3_moe_kernel_k3_n256_cb1(), exl3_moe_kernel_k3_n128_cb2(), exl3_moe_kernel_k3_n256_cb2(),
    exl3_moe_kernel_k4_n128_cb1(), exl3_moe_kernel_k4_n256_cb1(), exl3_moe_kernel_k4_n128_cb2(), exl3_moe_kernel_k4_n256_cb2(),
    exl3_moe_kernel_k5_n128_cb1(), exl3_moe_kernel_k5_n256_cb1(), exl3_moe_kernel_k5_n128_cb2(), exl3_moe_kernel_k5_n256_cb2(),
    exl3_moe_kernel_k6_n128_cb1(), exl3_moe_kernel_k6_n256_cb1(), exl3_moe_kernel_k6_n128_cb2(), exl3_moe_kernel_k6_n256_cb2(),
    exl3_moe_kernel_k7_n128_cb1(), exl3_moe_kernel_k7_n256_cb1(), exl3_moe_kernel_k7_n128_cb2(), exl3_moe_kernel_k7_n256_cb2(),
    exl3_moe_kernel_k8_n128_cb1(), exl3_moe_kernel_k8_n256_cb1(), exl3_moe_kernel_k8_n128_cb2(), exl3_moe_kernel_k8_n256_cb2()
};

static void check_cuda_contiguous_same_device
(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name
)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.device() == reference.device(),
        name, " must be on the same CUDA device as hidden_state"
    );
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_non_overlapping
(
    const at::Tensor& first,
    const at::Tensor& second,
    const char* first_name,
    const char* second_name
)
{
    const auto first_begin = reinterpret_cast<uintptr_t>(first.data_ptr());
    const auto second_begin = reinterpret_cast<uintptr_t>(second.data_ptr());
    const auto first_end = first_begin + first.nbytes();
    const auto second_end = second_begin + second.nbytes();
    TORCH_CHECK(
        first_end <= second_begin || second_end <= first_begin,
        first_name, " and ", second_name, " must not overlap"
    );
}

/*
Fused mixture-of-experts MLP operation for EXL3 weights

inputs:
    hidden_state:
        input hidden state - shape (bsz, hidden_dim) - fp16

    output_state:
        output hidden state - shape (bsz, hidden_dim) - fp32
        zero-initialized

    expert_count:
        bincount of expert indices across all tokens in batch - shape (num_experts + 1,) - int64
        last item is ignored, used for the case where some tokens may activate less than num_experts_per_token
        experts (specifically in expert split mode)

    token_sorted:
        token indices, sorted by expert - shape (bsz * num_experts_per_tok,)  - int64

    weight_sorted:
        routing weight per token, sorted by expert - shape (bsz * num_experts_per_tok,) - fp16

    temp_state_g:
    temp_state_u:
        temp state storage - shape (concurrency, max_tokens_per_expert, hidden_dim), fp16

    temp_intermediate_g
    temp_intermediate_u:
        temp intermediate storage - shape (concurrency, max_tokens_per_expert, intermediate_dim), fp16

    act_function:
        int, see exl3_moe.cuh

    K_gate
    K_up
    K_down:
        int, bitrates for gate, up, down tensors

    gate_ptrs_trellis
    gate_ptrs_suh
    gate_ptrs_svh
    up_ptrs_trellis
    up_ptrs_suh
    up_ptrs_svh
    down_ptrs_trellis
    down_ptrs_suh
    down_ptrs_svh:
        tensors of data_ptrs to quantized tensor data - each shape (num_experts,) - void*

    gate_mcg
    gate_mul1
    up_mcg
    up_mul1
    down_mcg
    down_mul1:
        bool, codebook flags

    num_active:
        launch-size hint. For exl3_moe this is the number of experts with
        0 < token count <= max_tokens_per_expert. Additive entry points tile
        oversized route spans, so they count every nonempty expert. Pass -1
        when unknown. For additive calls, 0 with nonempty routes is treated as
        unknown rather than dropping work.
*/

static void exl3_moe_impl
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,

    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,

    const int act_function,

    const int K_gate,
    const int K_up,
    const int K_down,

    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,

    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,

    const float act_limit,
    const int num_active,
    const at::Tensor& residual_gate_ptrs_trellis,
    const at::Tensor& residual_up_ptrs_trellis,
    const at::Tensor& residual_down_ptrs_trellis,
    const at::Tensor& residual_gate_scales,
    const at::Tensor& residual_up_scales,
    const at::Tensor& residual_down_scales,
    const at::Tensor& residual_gate_k,
    const at::Tensor& residual_up_k,
    const at::Tensor& residual_down_k,
    const int max_residual_bits,
    const bool tile_overflow,
    const bool validate_only
)
{
    TORCH_CHECK(hidden_state.is_cuda(), "hidden_state must be a CUDA tensor");
    const bool residual_gate_defined = residual_gate_ptrs_trellis.defined();
    const bool any_residual_defined =
        residual_gate_defined ||
        residual_up_ptrs_trellis.defined() ||
        residual_down_ptrs_trellis.defined() ||
        residual_gate_scales.defined() || residual_up_scales.defined() ||
        residual_down_scales.defined() || residual_gate_k.defined() ||
        residual_up_k.defined() || residual_down_k.defined();
    const bool all_residual_defined =
        residual_gate_defined &&
        residual_up_ptrs_trellis.defined() &&
        residual_down_ptrs_trellis.defined() &&
        residual_gate_scales.defined() && residual_up_scales.defined() &&
        residual_down_scales.defined() && residual_gate_k.defined() &&
        residual_up_k.defined() && residual_down_k.defined();
    TORCH_CHECK(
        any_residual_defined == all_residual_defined,
        "Residual pointer, scale, and K tensors must be all defined or all omitted"
    );
    const int num_residual_stages = residual_gate_ptrs_trellis.defined()
        ? residual_gate_ptrs_trellis.size(0)
        : 0;
    TORCH_CHECK(
        (num_residual_stages == 0 && max_residual_bits == 0) ||
        (num_residual_stages > 0 &&
         max_residual_bits >= 1 && max_residual_bits <= 8),
        "max_residual_bits must be zero without residuals or 1..8 with residuals"
    );

    // Validate every tensor before taking a raw data_ptr. This is especially
    // important for the fused entry point, which mutates routing workspaces
    // before launching the MoE kernel.
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DIM(hidden_state, 2);
    TORCH_CHECK(hidden_state.is_contiguous(), "hidden_state must be contiguous");
    size_t bsz = hidden_state.size(0);
    size_t hidden_dim = hidden_state.size(1);

    check_cuda_contiguous_same_device(output_state, hidden_state, "output_state");
    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);

    check_cuda_contiguous_same_device(expert_count, hidden_state, "expert_count");
    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    TORCH_CHECK(
        expert_count.size(0) >= 2,
        "expert_count must contain at least one expert and one sentinel bucket"
    );
    size_t num_experts = expert_count.size(0) - 1;

    check_cuda_contiguous_same_device(token_sorted, hidden_state, "token_sorted");
    check_cuda_contiguous_same_device(weight_sorted, hidden_state, "weight_sorted");
    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DTYPE(weight_sorted, kHalf);
    TORCH_CHECK_DIM(token_sorted, 1);
    TORCH_CHECK_DIM(weight_sorted, 1);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    TORCH_CHECK(
        bsz > 0 || token_sorted.numel() == 0,
        "token_sorted must be empty when hidden_state has no rows"
    );
    TORCH_CHECK(
        bsz == 0 || token_sorted.size(0) % bsz == 0,
        "token_sorted length must be divisible by hidden-state rows"
    );
    size_t num_experts_per_tok = bsz ? token_sorted.size(0) / bsz : 0;

    check_cuda_contiguous_same_device(temp_state_g, hidden_state, "temp_state_g");
    check_cuda_contiguous_same_device(temp_state_u, hidden_state, "temp_state_u");
    TORCH_CHECK_DTYPE(temp_state_g, kHalf);
    TORCH_CHECK_DTYPE(temp_state_u, kHalf);
    TORCH_CHECK_DIM(temp_state_g, 3);
    TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
    TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
    size_t max_tokens_per_expert = temp_state_g.size(1);
    size_t concurrency = temp_state_g.size(0);
    TORCH_CHECK(max_tokens_per_expert > 0, "MoE temp token capacity must be positive");
    TORCH_CHECK(concurrency > 0, "MoE temp concurrency must be positive");

    check_cuda_contiguous_same_device(
        temp_intermediate_g, hidden_state, "temp_intermediate_g"
    );
    check_cuda_contiguous_same_device(
        temp_intermediate_u, hidden_state, "temp_intermediate_u"
    );
    TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf);
    TORCH_CHECK_DTYPE(temp_intermediate_u, kHalf);
    TORCH_CHECK_DIM(temp_intermediate_g, 3);
    TORCH_CHECK_DIM(temp_intermediate_u, 3);
    TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
    TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
    size_t intermediate_dim = temp_intermediate_g.size(2);
    TORCH_CHECK(
        hidden_dim % 128 == 0 && intermediate_dim % 128 == 0,
        "MoE hidden and intermediate dimensions must be multiples of 128"
    );

    TORCH_CHECK(
        K_gate >= 1 && K_gate <= 8 &&
        K_up >= 1 && K_up <= 8 &&
        K_down >= 1 && K_down <= 8,
        "MoE gate/up/down bitrates must be in 1..8"
    );
    TORCH_CHECK(
        num_active >= -1 && num_active <= static_cast<int>(num_experts),
        "num_active must be -1 or in 0..num_experts"
    );
    const int effective_num_active =
        tile_overflow && num_active == 0 && token_sorted.numel() > 0
            ? -1 : num_active;

    // TORCH_CHECK(!(gate_mcg && gate_mul1), "Specified both mcg and mul1 (gate)");
    // TORCH_CHECK(!(up_mcg && up_mul1), "Specified both mcg and mul1 (up)");
    // TORCH_CHECK(!(down_mcg && down_mul1), "Specified both mcg and mul1 (down)");
    TORCH_CHECK(gate_mcg == up_mcg && up_mcg == down_mcg && gate_mul1 == up_mul1 && up_mul1 == down_mul1,
                "MoE kernel: gate/up/down must share the same codebook");
    TORCH_CHECK(gate_mcg != gate_mul1, "MoE kernel: Only mcg and mul1 codebooks are supported");
    const int cb_idx = gate_mul1 ? 1 : 0;

    // TORCH_CHECK(act_function == MOE_ACT_SILU, "MoE kernel: Only SiLU is currently supported");

    int K = 0;
    if (K_gate == K_up && K_up == K_down) K = K_gate;
    // Residual dispatch lives only in the runtime-K (K=0) instances. This
    // keeps the common equal-K legacy kernels free of the additional 1..8
    // residual GEMM specializations and their instruction footprint.
    if (num_residual_stages > 0) K = 0;

    check_cuda_contiguous_same_device(
        gate_ptrs_trellis, hidden_state, "gate_ptrs_trellis"
    );
    check_cuda_contiguous_same_device(gate_ptrs_suh, hidden_state, "gate_ptrs_suh");
    check_cuda_contiguous_same_device(gate_ptrs_svh, hidden_state, "gate_ptrs_svh");
    check_cuda_contiguous_same_device(
        up_ptrs_trellis, hidden_state, "up_ptrs_trellis"
    );
    check_cuda_contiguous_same_device(up_ptrs_suh, hidden_state, "up_ptrs_suh");
    check_cuda_contiguous_same_device(up_ptrs_svh, hidden_state, "up_ptrs_svh");
    check_cuda_contiguous_same_device(
        down_ptrs_trellis, hidden_state, "down_ptrs_trellis"
    );
    check_cuda_contiguous_same_device(down_ptrs_suh, hidden_state, "down_ptrs_suh");
    check_cuda_contiguous_same_device(down_ptrs_svh, hidden_state, "down_ptrs_svh");
    TORCH_CHECK_DTYPE(gate_ptrs_trellis, kLong);
    TORCH_CHECK_DTYPE(gate_ptrs_suh, kLong);
    TORCH_CHECK_DTYPE(gate_ptrs_svh, kLong);
    TORCH_CHECK_DTYPE(up_ptrs_trellis, kLong);
    TORCH_CHECK_DTYPE(up_ptrs_suh, kLong);
    TORCH_CHECK_DTYPE(up_ptrs_svh, kLong);
    TORCH_CHECK_DTYPE(down_ptrs_trellis, kLong);
    TORCH_CHECK_DTYPE(down_ptrs_suh, kLong);
    TORCH_CHECK_DTYPE(down_ptrs_svh, kLong);
    TORCH_CHECK_DIM(gate_ptrs_trellis, 1);
    TORCH_CHECK(gate_ptrs_trellis.size(0) == num_experts, "Number of gate tensors doesn't match num_experts");
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_svh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_trellis);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_svh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_trellis);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_svh);
    if (num_residual_stages > 0)
    {
        // Additive residual trellises use the MCG codebook and reuse the base
        // projection's suh/svh. K metadata is graph-resident and must contain
        // values in 1..max_residual_bits; callers construct and validate it
        // before graph capture.
        check_cuda_contiguous_same_device(
            residual_gate_ptrs_trellis, hidden_state,
            "residual_gate_ptrs_trellis"
        );
        check_cuda_contiguous_same_device(
            residual_up_ptrs_trellis, hidden_state,
            "residual_up_ptrs_trellis"
        );
        check_cuda_contiguous_same_device(
            residual_down_ptrs_trellis, hidden_state,
            "residual_down_ptrs_trellis"
        );
        check_cuda_contiguous_same_device(
            residual_gate_scales, hidden_state, "residual_gate_scales"
        );
        check_cuda_contiguous_same_device(
            residual_up_scales, hidden_state, "residual_up_scales"
        );
        check_cuda_contiguous_same_device(
            residual_down_scales, hidden_state, "residual_down_scales"
        );
        check_cuda_contiguous_same_device(
            residual_gate_k, hidden_state, "residual_gate_k"
        );
        check_cuda_contiguous_same_device(
            residual_up_k, hidden_state, "residual_up_k"
        );
        check_cuda_contiguous_same_device(
            residual_down_k, hidden_state, "residual_down_k"
        );
        TORCH_CHECK_DTYPE(residual_gate_ptrs_trellis, kLong);
        TORCH_CHECK_DTYPE(residual_up_ptrs_trellis, kLong);
        TORCH_CHECK_DTYPE(residual_down_ptrs_trellis, kLong);
        TORCH_CHECK_DIM(residual_gate_ptrs_trellis, 2);
        TORCH_CHECK_SHAPES_FULL(
            residual_gate_ptrs_trellis, residual_up_ptrs_trellis
        );
        TORCH_CHECK_SHAPES_FULL(
            residual_gate_ptrs_trellis, residual_down_ptrs_trellis
        );
        TORCH_CHECK(
            residual_gate_ptrs_trellis.size(1) == num_experts,
            "Residual pointer tables must have shape (num_stages, num_experts)"
        );
        TORCH_CHECK_DTYPE(residual_gate_scales, kFloat);
        TORCH_CHECK_DTYPE(residual_up_scales, kFloat);
        TORCH_CHECK_DTYPE(residual_down_scales, kFloat);
        TORCH_CHECK_SHAPES_FULL(
            residual_gate_ptrs_trellis, residual_gate_scales
        );
        TORCH_CHECK_SHAPES_FULL(
            residual_gate_ptrs_trellis, residual_up_scales
        );
        TORCH_CHECK_SHAPES_FULL(
            residual_gate_ptrs_trellis, residual_down_scales
        );
        TORCH_CHECK_DTYPE(residual_gate_k, kInt);
        TORCH_CHECK_DTYPE(residual_up_k, kInt);
        TORCH_CHECK_DTYPE(residual_down_k, kInt);
        TORCH_CHECK_DIM(residual_gate_k, 1);
        TORCH_CHECK_SHAPES_FULL(residual_gate_k, residual_up_k);
        TORCH_CHECK_SHAPES_FULL(residual_gate_k, residual_down_k);
        TORCH_CHECK(
            residual_gate_k.size(0) == num_residual_stages,
            "Residual K tensors must have shape (num_stages,)"
        );
    }

    if (validate_only || effective_num_active == 0 || bsz == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Launch. All blocks of the grid must be co-resident for the group barriers, so groups * width <= num_sms.
    // With a known number of active experts, launch only as many groups as there are experts and widen them to
    // use the freed SMs, up to MOE_MAX_SMS_PER_EXPERT
    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= num_sms, "Concurrency too high for device num_sms");
    int num_groups = MIN((int) concurrency, MOE_MAX_GROUPS);
    int group_size = MOE_SMS_PER_EXPERT;
    if (effective_num_active > 0)
    {
        num_groups = MIN(num_groups, effective_num_active);
        group_size = MIN(num_sms / num_groups, MOE_MAX_SMS_PER_EXPERT);
    }
    dim3 grid_dim(group_size, 1, num_groups);

    int N_off = 0;
    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;
    fp_exl3_moe_kernel kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];

    if (moe_kernel_attr_set[device].find((void*) kernel) == moe_kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        moe_kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }

    void* _hidden_state = hidden_state.data_ptr();
    void* _temp_state_g = temp_state_g.data_ptr();
    void* _temp_state_u = temp_state_u.data_ptr();
    void* _temp_intermediate_g = temp_intermediate_g.data_ptr();
    void* _temp_intermediate_u = temp_intermediate_u.data_ptr();
    void* _output_state = output_state.data_ptr();

    void* _gate_ptrs_trellis = gate_ptrs_trellis.data_ptr();
    void* _gate_ptrs_suh = gate_ptrs_suh.data_ptr();
    void* _gate_ptrs_svh = gate_ptrs_svh.data_ptr();
    void* _up_ptrs_trellis = up_ptrs_trellis.data_ptr();
    void* _up_ptrs_suh = up_ptrs_suh.data_ptr();
    void* _up_ptrs_svh = up_ptrs_svh.data_ptr();
    void* _down_ptrs_trellis = down_ptrs_trellis.data_ptr();
    void* _down_ptrs_suh = down_ptrs_suh.data_ptr();
    void* _down_ptrs_svh = down_ptrs_svh.data_ptr();
    void* _residual_gate_ptrs_trellis = num_residual_stages
        ? residual_gate_ptrs_trellis.data_ptr()
        : nullptr;
    void* _residual_up_ptrs_trellis = num_residual_stages
        ? residual_up_ptrs_trellis.data_ptr()
        : nullptr;
    void* _residual_down_ptrs_trellis = num_residual_stages
        ? residual_down_ptrs_trellis.data_ptr()
        : nullptr;
    void* _residual_gate_scales = num_residual_stages
        ? residual_gate_scales.data_ptr()
        : nullptr;
    void* _residual_up_scales = num_residual_stages
        ? residual_up_scales.data_ptr()
        : nullptr;
    void* _residual_down_scales = num_residual_stages
        ? residual_down_scales.data_ptr()
        : nullptr;
    void* _residual_gate_k = num_residual_stages
        ? residual_gate_k.data_ptr()
        : nullptr;
    void* _residual_up_k = num_residual_stages
        ? residual_up_k.data_ptr()
        : nullptr;
    void* _residual_down_k = num_residual_stages
        ? residual_down_k.data_ptr()
        : nullptr;

    void* _expert_count = expert_count.data_ptr();
    void* _token_sorted = token_sorted.data_ptr();
    void* _weight_sorted = weight_sorted.data_ptr();

    void* kernelArgs[] =
    {
        &_hidden_state,
        &_temp_state_g,
        &_temp_state_u,
        &_temp_intermediate_g,
        &_temp_intermediate_u,
        &_output_state,
        &_gate_ptrs_trellis,
        &_gate_ptrs_suh,
        &_gate_ptrs_svh,
        &_up_ptrs_trellis,
        &_up_ptrs_suh,
        &_up_ptrs_svh,
        &_down_ptrs_trellis,
        &_down_ptrs_suh,
        &_down_ptrs_svh,
        &_residual_gate_ptrs_trellis,
        &_residual_up_ptrs_trellis,
        &_residual_down_ptrs_trellis,
        &_residual_gate_scales,
        &_residual_up_scales,
        &_residual_down_scales,
        &_residual_gate_k,
        &_residual_up_k,
        &_residual_down_k,
        (void*) &num_residual_stages,
        &_expert_count,
        &_token_sorted,
        &_weight_sorted,
        (void*) &hidden_dim,
        (void*) &intermediate_dim,
        (void*) &num_experts,
        (void*) &num_experts_per_tok,
        (void*) &max_tokens_per_expert,
        (void*) &num_groups,
        (void*) &act_limit,
        (void*) &act_function,
        (void*) &K_gate,
        (void*) &K_up,
        (void*) &K_down,
        (void*) &tile_overflow,
        (void*) &locks
    };

    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );

    cuda_check(cudaPeekAtLastError());
}

void exl3_moe
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,
    const int act_function,
    const int K_gate,
    const int K_up,
    const int K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int num_active
)
{
    const at::Tensor empty;
    exl3_moe_impl
    (
        hidden_state, output_state, expert_count, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
        act_function, K_gate, K_up, K_down,
        gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
        up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
        down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
        gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1,
        act_limit, num_active,
        empty, empty, empty, empty, empty, empty, empty, empty, empty, 0,
        false, false
    );
}

void exl3_moe_additive
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,
    const int act_function,
    const int K_gate,
    const int K_up,
    const int K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const at::Tensor& residual_gate_ptrs_trellis,
    const at::Tensor& residual_up_ptrs_trellis,
    const at::Tensor& residual_down_ptrs_trellis,
    const at::Tensor& residual_gate_scales,
    const at::Tensor& residual_up_scales,
    const at::Tensor& residual_down_scales,
    const at::Tensor& residual_gate_k,
    const at::Tensor& residual_up_k,
    const at::Tensor& residual_down_k,
    const int max_residual_bits,
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int num_active
)
{
    exl3_moe_impl
    (
        hidden_state, output_state, expert_count, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
        act_function, K_gate, K_up, K_down,
        gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
        up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
        down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
        gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1,
        act_limit, num_active,
        residual_gate_ptrs_trellis,
        residual_up_ptrs_trellis,
        residual_down_ptrs_trellis,
        residual_gate_scales,
        residual_up_scales,
        residual_down_scales,
        residual_gate_k,
        residual_up_k,
        residual_down_k,
        max_residual_bits,
        true, false
    );
}

void exl3_moe_additive_fused
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& topk_ids,
    const at::Tensor& topk_weights,
    const at::Tensor& expert_map,
    const at::Tensor& expert_count,
    const at::Tensor& expert_offsets,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,
    const int act_function,
    const int K_gate,
    const int K_up,
    const int K_down,
    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,
    const at::Tensor& residual_gate_ptrs_trellis,
    const at::Tensor& residual_up_ptrs_trellis,
    const at::Tensor& residual_down_ptrs_trellis,
    const at::Tensor& residual_gate_scales,
    const at::Tensor& residual_up_scales,
    const at::Tensor& residual_down_scales,
    const at::Tensor& residual_gate_k,
    const at::Tensor& residual_up_k,
    const at::Tensor& residual_down_k,
    const int max_residual_bits,
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int num_active
)
{
    TORCH_CHECK(hidden_state.is_cuda(), "hidden_state must be a CUDA tensor");
    TORCH_CHECK(hidden_state.is_contiguous(), "hidden_state must be contiguous");
    TORCH_CHECK_DIM(hidden_state, 2);
    check_cuda_contiguous_same_device(topk_ids, hidden_state, "topk_ids");
    check_cuda_contiguous_same_device(topk_weights, hidden_state, "topk_weights");
    check_cuda_contiguous_same_device(expert_map, hidden_state, "expert_map");
    check_cuda_contiguous_same_device(expert_count, hidden_state, "expert_count");
    check_cuda_contiguous_same_device(expert_offsets, hidden_state, "expert_offsets");
    check_cuda_contiguous_same_device(token_sorted, hidden_state, "token_sorted");
    check_cuda_contiguous_same_device(weight_sorted, hidden_state, "weight_sorted");
    TORCH_CHECK(
        topk_ids.scalar_type() == at::kLong ||
        topk_ids.scalar_type() == at::kInt,
        "topk_ids must be int32 or int64"
    );
    TORCH_CHECK(
        topk_weights.scalar_type() == at::kFloat ||
        topk_weights.scalar_type() == at::kHalf ||
        topk_weights.scalar_type() == at::kBFloat16,
        "topk_weights must be float, half, or bfloat16"
    );
    TORCH_CHECK_DIM(topk_ids, 2);
    TORCH_CHECK_DIM(topk_weights, 2);
    TORCH_CHECK_SHAPES_FULL(topk_ids, topk_weights);
    TORCH_CHECK_DTYPE(expert_map, kLong);
    TORCH_CHECK_DIM(expert_map, 1);
    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DTYPE(expert_offsets, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    TORCH_CHECK_DIM(expert_offsets, 1);
    TORCH_CHECK_SHAPES_FULL(expert_count, expert_offsets);
    check_non_overlapping(
        expert_count, expert_offsets, "expert_count", "expert_offsets"
    );
    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DTYPE(weight_sorted, kHalf);
    TORCH_CHECK_DIM(token_sorted, 1);
    TORCH_CHECK_DIM(weight_sorted, 1);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    TORCH_CHECK(
        hidden_state.size(0) == topk_ids.size(0),
        "route rows must equal hidden-state rows"
    );
    const int64_t num_routes = topk_ids.numel();
    const int num_buckets = expert_count.numel();
    TORCH_CHECK(
        num_buckets == gate_ptrs_trellis.numel() + 1,
        "expert_count must include one sentinel bucket"
    );
    TORCH_CHECK(
        token_sorted.numel() >= num_routes,
        "route workspace is too small"
    );

    const at::Tensor routed_tokens = token_sorted.narrow(0, 0, num_routes);
    const at::Tensor routed_weights = weight_sorted.narrow(0, 0, num_routes);
    const int launch_num_active = num_routes == 0 ? 0 : num_active;
    exl3_moe_impl
    (
        hidden_state, output_state, expert_count, routed_tokens, routed_weights,
        temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
        act_function, K_gate, K_up, K_down,
        gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
        up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
        down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
        gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1,
        act_limit, launch_num_active,
        residual_gate_ptrs_trellis,
        residual_up_ptrs_trellis,
        residual_down_ptrs_trellis,
        residual_gate_scales,
        residual_up_scales,
        residual_down_scales,
        residual_gate_k,
        residual_up_k,
        residual_down_k,
        max_residual_bits,
        true, true
    );
    if (num_routes == 0) return;

    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    cuda_check(cudaMemsetAsync(
        expert_count.data_ptr(), 0,
        expert_count.numel() * expert_count.element_size(), stream
    ));
    const int threads = 256;
    const int blocks = std::min<int64_t>(
        1024, (num_routes + threads - 1) / threads
    );
    #define LAUNCH_HIST(ID_T, PTR) \
        exl3_route_histogram_kernel<ID_T><<<blocks, threads, 0, stream>>>( \
            PTR, expert_map.data_ptr<int64_t>(), \
            expert_count.data_ptr<int64_t>(), num_routes, \
            expert_map.numel(), num_buckets)
    if (topk_ids.scalar_type() == at::kInt)
        LAUNCH_HIST(int32_t, topk_ids.data_ptr<int32_t>());
    else
        LAUNCH_HIST(int64_t, topk_ids.data_ptr<int64_t>());
    #undef LAUNCH_HIST

    exl3_route_scan_kernel<<<1, 1, 0, stream>>>
    (
        expert_count.data_ptr<int64_t>(),
        expert_offsets.data_ptr<int64_t>(),
        num_buckets
    );

    #define LAUNCH_PACK(ID_T, ID_PTR, W_T, W_PTR) \
        exl3_route_pack_stable_kernel<ID_T, W_T> \
            <<<num_buckets, threads, 0, stream>>>( \
                ID_PTR, W_PTR, expert_map.data_ptr<int64_t>(), \
                expert_offsets.data_ptr<int64_t>(), \
                token_sorted.data_ptr<int64_t>(), \
                reinterpret_cast<half*>(weight_sorted.data_ptr()), \
                num_routes, topk_ids.size(1), expert_map.numel(), num_buckets)
    #define DISPATCH_WEIGHT(ID_T, ID_PTR) \
        if (topk_weights.scalar_type() == at::kFloat) \
            LAUNCH_PACK( \
                ID_T, ID_PTR, float, topk_weights.data_ptr<float>() \
            ); \
        else if (topk_weights.scalar_type() == at::kHalf) \
            LAUNCH_PACK( \
                ID_T, ID_PTR, half, \
                reinterpret_cast<const half*>(topk_weights.data_ptr()) \
            ); \
        else if (topk_weights.scalar_type() == at::kBFloat16) \
            LAUNCH_PACK( \
                ID_T, ID_PTR, __nv_bfloat16, \
                reinterpret_cast<const __nv_bfloat16*>( \
                    topk_weights.data_ptr() \
                ) \
            ); \
        else TORCH_CHECK( \
            false, "topk_weights must be float, half, or bfloat16" \
        )
    if (topk_ids.scalar_type() == at::kInt)
    {
        DISPATCH_WEIGHT(int32_t, topk_ids.data_ptr<int32_t>());
    }
    else
    {
        DISPATCH_WEIGHT(int64_t, topk_ids.data_ptr<int64_t>());
    }
    #undef DISPATCH_WEIGHT
    #undef LAUNCH_PACK
    cuda_check(cudaPeekAtLastError());

    exl3_moe_additive
    (
        hidden_state,
        output_state,
        expert_count,
        routed_tokens,
        routed_weights,
        temp_state_g,
        temp_state_u,
        temp_intermediate_g,
        temp_intermediate_u,
        act_function,
        K_gate,
        K_up,
        K_down,
        gate_ptrs_trellis,
        gate_ptrs_suh,
        gate_ptrs_svh,
        up_ptrs_trellis,
        up_ptrs_suh,
        up_ptrs_svh,
        down_ptrs_trellis,
        down_ptrs_suh,
        down_ptrs_svh,
        residual_gate_ptrs_trellis,
        residual_up_ptrs_trellis,
        residual_down_ptrs_trellis,
        residual_gate_scales,
        residual_up_scales,
        residual_down_scales,
        residual_gate_k,
        residual_up_k,
        residual_down_k,
        max_residual_bits,
        gate_mcg,
        gate_mul1,
        up_mcg,
        up_mul1,
        down_mcg,
        down_mul1,
        act_limit,
        launch_num_active
    );
}
