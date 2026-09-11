#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_devctx.cuh"
#include <set>

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
        number of experts with 0 < token count <= max_tokens_per_expert, i.e. the number of experts this kernel
        will process. Used to size the launch: fewer, wider expert groups when few experts are active. Pass -1 if
        unknown (defaults to MOE_SMS_PER_EXPERT-wide groups at max concurrency)
*/

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
    const int num_active,
    const c10::optional<at::Tensor>& output_scratch,
    const c10::optional<at::Tensor>& fused_base
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Nothing for the fused kernel to do
    if (num_active == 0) return;
    void* _output_scratch = nullptr;
    void* _fused_base = nullptr;
    if (output_scratch.has_value())
    {
        TORCH_CHECK(fused_base.has_value(), "exl3_moe: output_scratch needs fused_base");
        TORCH_CHECK_DTYPE(output_scratch.value(), kFloat);
        TORCH_CHECK_DTYPE(fused_base.value(), kLong);
        TORCH_CHECK(output_scratch.value().is_contiguous() && output_scratch.value().dim() == 2 &&
                    output_scratch.value().size(1) == hidden_state.size(1), "exl3_moe: output_scratch must be [slots, hidden]");
        _output_scratch = output_scratch.value().data_ptr();
        _fused_base = fused_base.value().data_ptr();
    }

    // Validate args
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DIM(hidden_state, 2);
    size_t bsz = hidden_state.size(0);
    size_t hidden_dim = hidden_state.size(1);

    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);

    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    size_t num_experts = expert_count.size(0) - 1;

    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DIM(token_sorted, 1);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    size_t num_experts_per_tok = token_sorted.size(0) / bsz;

    TORCH_CHECK_DTYPE(temp_state_g, kHalf);
    TORCH_CHECK_DTYPE(temp_state_u, kHalf);
    TORCH_CHECK_DIM(temp_state_g, 3);
    TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
    TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
    size_t max_tokens_per_expert = temp_state_g.size(1);
    size_t concurrency = temp_state_g.size(0);

    TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf);
    TORCH_CHECK_DTYPE(temp_intermediate_u, kHalf);
    TORCH_CHECK_DIM(temp_intermediate_g, 3);
    TORCH_CHECK_DIM(temp_intermediate_u, 3);
    TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
    TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
    size_t intermediate_dim = temp_intermediate_g.size(2);

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
    if (num_active > 0)
    {
        num_groups = MIN(num_groups, num_active);
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
        (void*) &locks,
        &_output_scratch,
        &_fused_base
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


// Deterministic reduction of the fused kernel's per-assignment outputs: for every token and
// column, sum the token's top-k slots in k order and add to the output row. An assignment
// a = token * topk + k is in the fused tier when its expert has 0 < count <= cap (and is a real
// expert, not the sentinel bin); its slot is fused_base[e] + (inv_order[a] - expert_start[e])
#define MOE_GATHER_MAX_TOPK 32

// Deterministic reduction of per-assignment expert outputs: for every token and column, sum the
// token's top-k slots in k order and add to the output row. slot_kind[e]: 0 = expert e has no
// slots (handled elsewhere), 1 = slots hold weighted outputs (fused kernel), 2 = slots hold
// unweighted outputs (batched reconstruct tier), multiplied by the routing weight here. The slot
// of assignment a = token * topk + k is slot_base[e] + (inv_order[a] - expert_start[e]).
__global__ void exl3_moe_gather_kernel
(
    float* __restrict__ output_state,
    const float* __restrict__ output_scratch,
    const int64_t* __restrict__ flat_expert,
    const int64_t* __restrict__ inv_order,
    const int64_t* __restrict__ expert_start,
    const int64_t* __restrict__ slot_base,
    const int64_t* __restrict__ slot_kind,
    const half* __restrict__ weight_sorted,
    const int hidden_dim,
    const int topk,
    const int num_experts
)
{
    // One block per token: the slot list is resolved once into shared memory (threads
    // 0..topk-1), then every column thread streams its column of the listed slots in k order
    __shared__ int64_t slots[MOE_GATHER_MAX_TOPK];
    __shared__ float wts[MOE_GATHER_MAX_TOPK];
    __shared__ int nslots;
    const int token = blockIdx.x;
    if (threadIdx.x < topk)
    {
        int64_t a = (int64_t) token * topk + threadIdx.x;
        int64_t e = flat_expert[a];
        int64_t slot = -1;
        float wt = 1.0f;
        if (e >= 0 && e < num_experts)
        {
            int64_t kind = slot_kind[e];
            if (kind)
            {
                int64_t pos = inv_order[a];
                slot = slot_base[e] + (pos - expert_start[e]);
                if (kind == 2) wt = __half2float(weight_sorted[pos]);
            }
        }
        slots[threadIdx.x] = slot;
        wts[threadIdx.x] = wt;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int n = 0;
        for (int k = 0; k < topk; ++k)
            if (slots[k] >= 0) { slots[n] = slots[k]; wts[n] = wts[k]; ++n; }
        nslots = n;
    }
    __syncthreads();
    const int n = nslots;
    if (n == 0) return;
    for (int col = threadIdx.x; col < hidden_dim; col += blockDim.x)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += output_scratch[slots[k] * hidden_dim + col] * wts[k];
        output_state[(int64_t) token * hidden_dim + col] += sum;
    }
}

void exl3_moe_gather
(
    at::Tensor output_state,
    const at::Tensor& output_scratch,
    const at::Tensor& flat_expert,
    const at::Tensor& inv_order,
    const at::Tensor& expert_start,
    const at::Tensor& slot_base,
    const at::Tensor& slot_kind,
    const at::Tensor& weight_sorted
)
{
    const at::cuda::OptionalCUDAGuard device_guard(output_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_DTYPE(output_scratch, kFloat);
    TORCH_CHECK_DTYPE(flat_expert, kLong);
    TORCH_CHECK_DTYPE(inv_order, kLong);
    TORCH_CHECK_DTYPE(expert_start, kLong);
    TORCH_CHECK_DTYPE(slot_base, kLong);
    TORCH_CHECK_DTYPE(slot_kind, kLong);
    TORCH_CHECK_DTYPE(weight_sorted, kHalf);
    TORCH_CHECK(output_state.is_contiguous() && output_state.dim() == 2, "exl3_moe_gather: output_state");
    TORCH_CHECK(output_scratch.is_contiguous() && output_scratch.dim() == 2 && output_scratch.size(1) == output_state.size(1),
                "exl3_moe_gather: output_scratch must be [slots, hidden]");
    TORCH_CHECK(flat_expert.is_contiguous() && inv_order.is_contiguous() && expert_start.is_contiguous() &&
                slot_base.is_contiguous() && slot_kind.is_contiguous() && weight_sorted.is_contiguous(),
                "exl3_moe_gather: index tensors must be contiguous");
    int tokens = output_state.size(0);
    int hidden_dim = output_state.size(1);
    int num_assign = flat_expert.size(0);
    TORCH_CHECK(num_assign % tokens == 0, "exl3_moe_gather: assignments / tokens");
    int topk = num_assign / tokens;
    int num_experts = slot_kind.size(0);
    TORCH_CHECK(slot_base.size(0) >= num_experts && expert_start.size(0) >= num_experts, "exl3_moe_gather: table sizes");
    if (!tokens) return;
    TORCH_CHECK(topk <= MOE_GATHER_MAX_TOPK, "exl3_moe_gather: top-k too large");
    int threads = MAX(MIN(hidden_dim, 1024), 32);
    exl3_moe_gather_kernel<<<tokens, threads, 0, stream>>>
    (
        (float*) output_state.data_ptr(),
        (const float*) output_scratch.data_ptr(),
        (const int64_t*) flat_expert.data_ptr(),
        (const int64_t*) inv_order.data_ptr(),
        (const int64_t*) expert_start.data_ptr(),
        (const int64_t*) slot_base.data_ptr(),
        (const int64_t*) slot_kind.data_ptr(),
        (const half*) weight_sorted.data_ptr(),
        hidden_dim, topk, num_experts
    );
    cuda_check(cudaPeekAtLastError());
}
