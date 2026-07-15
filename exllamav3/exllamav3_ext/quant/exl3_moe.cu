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
#include <map>
#include <mutex>

template <typename id_t>
__global__ void exl3_route_histogram_kernel
(
    const id_t* __restrict__ topk_ids,
    const int64_t* __restrict__ expert_map,
    int64_t* __restrict__ expert_count,
    int64_t num_routes
)
{
    for (int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
         r < num_routes; r += (int64_t) blockDim.x * gridDim.x)
    {
        int64_t e = expert_map[topk_ids[r]];
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
__device__ __forceinline__ half exl3_route_to_half<half>(half v) { return v; }
template <>
__device__ __forceinline__ half exl3_route_to_half<__nv_bfloat16>(__nv_bfloat16 v)
{
    return __float2half(__bfloat162float(v));
}

template <typename id_t, typename weight_t>
__global__ void exl3_route_pack_stable_kernel
(
    const id_t* __restrict__ topk_ids,
    const weight_t* __restrict__ topk_weights,
    const int64_t* __restrict__ expert_map,
    int64_t* __restrict__ expert_offsets,
    int64_t* __restrict__ token_sorted,
    half* __restrict__ weight_sorted,
    int64_t num_routes,
    int topk
)
{
    // One block per local expert (including the sentinel). Each block scans
    // routes in source order and uses a deterministic block prefix, preserving
    // the exact stable expert-grouped ordering of torch.argsort.
    __shared__ int flags[256];
    __shared__ int running;
    if (threadIdx.x == 0) running = 0;
    __syncthreads();
    const int64_t expert = blockIdx.x;
    for (int64_t base = 0; base < num_routes; base += blockDim.x)
    {
        int64_t r = base + threadIdx.x;
        int flag = 0;
        if (r < num_routes) flag = expert_map[topk_ids[r]] == expert;
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

std::map<void*, int> moe_kernel_attr_smem[MAX_DEVICES] = {};
std::mutex moe_kernel_attr_mutex[MAX_DEVICES];
static thread_local int exl3_moe_dq_threshold = 0;
static thread_local bool exl3_moe_a1_retile = false;

class Exl3MoeA1RetileScope
{
private:
    bool previous;

public:
    Exl3MoeA1RetileScope(): previous(exl3_moe_a1_retile)
    {
        exl3_moe_a1_retile = true;
    }

    ~Exl3MoeA1RetileScope()
    {
        exl3_moe_a1_retile = previous;
    }
};

fp_exl3_moe_kernel exl3_moe_kernel_instances[] =
{
    exl3_moe_kernel_k0_n128(), exl3_moe_kernel_k0_n256(), // Switch Kg, Ku and Kd at runtime
    exl3_moe_kernel_k1_n128(), exl3_moe_kernel_k1_n256(), // Compile-time Kg = Ku = Kd
    exl3_moe_kernel_k2_n128(), exl3_moe_kernel_k2_n256(), // ...
    exl3_moe_kernel_k3_n128(), exl3_moe_kernel_k3_n256(),
    exl3_moe_kernel_k4_n128(), exl3_moe_kernel_k4_n256(),
    exl3_moe_kernel_k5_n128(), exl3_moe_kernel_k5_n256(),
    exl3_moe_kernel_k6_n128(), exl3_moe_kernel_k6_n256(),
    exl3_moe_kernel_k7_n128(), exl3_moe_kernel_k7_n256(),
    exl3_moe_kernel_k8_n128(), exl3_moe_kernel_k8_n256()
};

fp_exl3_moe_kernel exl3_moe_retile_kernel_instances[] =
{
    exl3_moe_retile_kernel_k3_n128(),
    exl3_moe_retile_kernel_k3_n256()
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
        number of nonempty experts handled by this kernel, used to size the
        launch. Pass -1 for the exact legacy 8-SM round-robin schedule.
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
    const int num_active
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(num_active >= -1, "num_active must be -1 or nonnegative");
    if (num_active == 0) return;
    const bool a1_retile = exl3_moe_a1_retile;

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
    if (a1_retile)
        TORCH_CHECK(max_tokens_per_expert >= MOE_TILESIZE_M,
                    "A1 re-tile requires at least one M32 temp row block");

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
    TORCH_CHECK(gate_mcg && !gate_mul1, "MoE kernel: Only mcg codebook is currently supported");
    TORCH_CHECK(up_mcg && !up_mul1, "MoE kernel: Only mcg codebook is currently supported");
    TORCH_CHECK(down_mcg && !down_mul1, "MoE kernel: Only mcg codebook is currently supported");

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

    // Launch. All blocks must be co-resident for the group barriers. With a
    // known active count, use one group per active expert (up to available temp
    // buffers) and widen each group into the otherwise idle SMs. num_active=-1
    // retains the pre-port grid and round-robin assignment exactly. A1 keeps
    // that exact eight-CTA K partition but enables greedy M32 route-block
    // tickets for fused prefill only.
    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= num_sms, "Concurrency too high for device num_sms");
    int num_groups = concurrency;
    int group_size = MOE_SMS_PER_EXPERT;
    int use_ticket_scheduler = 0;
    if (a1_retile)
    {
        num_groups = MIN((int) concurrency, num_sms / MOE_SMS_PER_EXPERT);
        group_size = MOE_SMS_PER_EXPERT;
        use_ticket_scheduler = 1;
    }
    else if (num_active > 0)
    {
        num_groups = MIN(MIN((int) concurrency, MOE_MAX_GROUPS), num_active);
        group_size = MIN(num_sms / num_groups, MOE_MAX_SMS_PER_EXPERT);
        use_ticket_scheduler = 1;
    }
    TORCH_CHECK(num_groups * group_size <= num_sms, "MoE grid exceeds device num_sms");
    dim3 grid_dim(group_size, 1, num_groups);

    int N_off = 0;
    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;
    fp_exl3_moe_kernel kernel;
    if (a1_retile)
    {
        TORCH_CHECK(K_gate == 3 && K_up == 3 && K_down == 3,
                    "A1 re-tile is specialized for TR3 gate/up/down weights");
        kernel = exl3_moe_retile_kernel_instances[N_off];
    }
    else
    {
        kernel = exl3_moe_kernel_instances[2 * K + N_off];
    }

    int moe_smem_bytes = SMEM_MAX;
    int moe_smem_attr_bytes = SMEM_MAX;
    if (a1_retile)
    {
        const int max_bits = MAX(K_gate, MAX(K_up, K_down));
        const int moe_tilesize_n = N_off ? 256 : 128;
        moe_smem_bytes = exl3_moe_smem_bytes
        (
            max_bits, moe_tilesize_n, MOE_A1_SH_STAGES
        );
        moe_smem_attr_bytes = MOE_A1_SMEM_MAX_BYTES;
    }
    #if MOE_SMEMFIX
        if (!a1_retile)
        {
            const int max_bits = MAX(K_gate, MAX(K_up, K_down));
            const int moe_tilesize_n = N_off ? 256 : 128;
            moe_smem_bytes = exl3_moe_smem_bytes(max_bits, moe_tilesize_n);
        }
    #endif

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
        (void*) &exl3_moe_dq_threshold,
        (void*) &use_ticket_scheduler,
        (void*) &locks
    };

    {
        // Function attributes are shared by all host threads. Keep the exact
        // attribute value stable until this launch has consumed it.
        std::lock_guard<std::mutex> attr_lock(moe_kernel_attr_mutex[device]);
        auto& attr_smem = moe_kernel_attr_smem[device];
        auto attr_it = attr_smem.find((void*) kernel);
        if (attr_it == attr_smem.end() || attr_it->second != moe_smem_attr_bytes)
        {
            cuda_check(cudaFuncSetAttribute
            (
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                moe_smem_attr_bytes
            ));
            attr_smem[(void*) kernel] = moe_smem_attr_bytes;
        }

        if (a1_retile)
        {
            cuda_check(cudaLaunchCooperativeKernel
            (
                (void*) kernel,
                grid_dim,
                block_dim,
                kernelArgs,
                moe_smem_bytes,
                stream
            ));
        }
        else
        {
            cuda_check(cudaLaunchKernel
            (
                (void*) kernel,
                grid_dim,
                block_dim,
                kernelArgs,
                moe_smem_bytes,
                stream
            ));
        }
    }

    cuda_check(cudaPeekAtLastError());
}

void exl3_moe_fused
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
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int dq_threshold
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(topk_ids.scalar_type() == at::kLong ||
                topk_ids.scalar_type() == at::kInt,
                "topk_ids must be int32 or int64");
    TORCH_CHECK_DIM(topk_ids, 2);
    TORCH_CHECK_DIM(topk_weights, 2);
    TORCH_CHECK_SHAPES_FULL(topk_ids, topk_weights);
    TORCH_CHECK_DTYPE(expert_map, kLong);
    TORCH_CHECK_DIM(expert_map, 1);
    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DTYPE(expert_offsets, kLong);
    TORCH_CHECK_SHAPES_FULL(expert_count, expert_offsets);
    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DTYPE(weight_sorted, kHalf);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    TORCH_CHECK(hidden_state.size(0) == topk_ids.size(0), "route rows != hidden rows");
    const int64_t num_routes = topk_ids.numel();
    const int num_buckets = expert_count.numel();
    TORCH_CHECK(num_buckets == gate_ptrs_trellis.numel() + 1,
                "expert_count must include one sentinel bucket");
    TORCH_CHECK(token_sorted.numel() >= num_routes, "route workspace too small");

    cudaMemsetAsync(expert_count.data_ptr(), 0,
                    expert_count.numel() * expert_count.element_size(), stream);
    const int threads = 256;
    const int blocks = std::min<int64_t>(1024, (num_routes + threads - 1) / threads);
    #define LAUNCH_HIST(ID_T, PTR) \
        exl3_route_histogram_kernel<ID_T><<<blocks, threads, 0, stream>>>( \
            PTR, expert_map.data_ptr<int64_t>(), \
            expert_count.data_ptr<int64_t>(), num_routes)
    if (topk_ids.scalar_type() == at::kInt)
        LAUNCH_HIST(int32_t, topk_ids.data_ptr<int32_t>());
    else
        LAUNCH_HIST(int64_t, topk_ids.data_ptr<int64_t>());
    #undef LAUNCH_HIST
    exl3_route_scan_kernel<<<1, 1, 0, stream>>>
    (
        expert_count.data_ptr<int64_t>(), expert_offsets.data_ptr<int64_t>(),
        num_buckets
    );
    #define LAUNCH_PACK(ID_T, ID_PTR, W_T, W_PTR) \
        exl3_route_pack_stable_kernel<ID_T, W_T><<<num_buckets, threads, 0, stream>>>( \
            ID_PTR, W_PTR, expert_map.data_ptr<int64_t>(), \
            expert_offsets.data_ptr<int64_t>(), token_sorted.data_ptr<int64_t>(), \
            reinterpret_cast<half*>(weight_sorted.data_ptr()), num_routes, \
            topk_ids.size(1))
    #define DISPATCH_WEIGHT(ID_T, ID_PTR) \
        if (topk_weights.scalar_type() == at::kFloat) \
            LAUNCH_PACK(ID_T, ID_PTR, float, topk_weights.data_ptr<float>()); \
        else if (topk_weights.scalar_type() == at::kHalf) \
            LAUNCH_PACK(ID_T, ID_PTR, half, reinterpret_cast<const half*>(topk_weights.data_ptr())); \
        else if (topk_weights.scalar_type() == at::kBFloat16) \
            LAUNCH_PACK(ID_T, ID_PTR, __nv_bfloat16, reinterpret_cast<const __nv_bfloat16*>(topk_weights.data_ptr())); \
        else TORCH_CHECK(false, "topk_weights must be float, half, or bfloat16")
    if (topk_ids.scalar_type() == at::kInt) {
        DISPATCH_WEIGHT(int32_t, topk_ids.data_ptr<int32_t>());
    } else {
        DISPATCH_WEIGHT(int64_t, topk_ids.data_ptr<int64_t>());
    }
    #undef DISPATCH_WEIGHT
    #undef LAUNCH_PACK
    cuda_check(cudaPeekAtLastError());

    // Narrowed views carry the actual route count into the unchanged ABI; no
    // allocation or synchronization is introduced, so this remains graph-safe.
    exl3_moe_dq_threshold = dq_threshold;
    exl3_moe
    (
        hidden_state, output_state, expert_count,
        token_sorted.narrow(0, 0, num_routes),
        weight_sorted.narrow(0, 0, num_routes),
        temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
        act_function, K_gate, K_up, K_down,
        gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
        up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
        down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
        gate_mcg, gate_mul1, up_mcg, up_mul1, down_mcg, down_mul1,
        act_limit,
        -1
    );
    exl3_moe_dq_threshold = 0;
}

void exl3_moe_fused_retile
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
    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,
    const float act_limit,
    const int dq_threshold
)
{
    Exl3MoeA1RetileScope retile_scope;
    exl3_moe_fused
    (
        hidden_state, output_state, topk_ids, topk_weights, expert_map,
        expert_count, expert_offsets, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_intermediate_g, temp_intermediate_u,
        act_function, K_gate, K_up, K_down,
        gate_ptrs_trellis, gate_ptrs_suh, gate_ptrs_svh,
        up_ptrs_trellis, up_ptrs_suh, up_ptrs_svh,
        down_ptrs_trellis, down_ptrs_suh, down_ptrs_svh,
        gate_mcg, gate_mul1, up_mcg, up_mul1,
        down_mcg, down_mul1, act_limit, dq_threshold
    );
}
