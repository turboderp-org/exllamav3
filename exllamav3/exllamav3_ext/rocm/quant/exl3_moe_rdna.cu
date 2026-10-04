// =============================================================================
// exl3_moe.cu for RDNA -- SMEM_MAX source + runtime LDS budget
// (The 32 / 64-row instances are replaced by a 16-row fallback, marked RDNA
// below.)
// =============================================================================
//
// Generated from quant/exl3_moe.cu. Two changes, both about one constant:
//
// 1. exl3_kernel_map_rdna.cuh is included FIRST. quant/exl3_moe.cu reaches SMEM_MAX
//    through comp_units/exl3_moe_instances.cuh -> exl3_moe_common.cuh, which
//    sets it to 90 KB behind an #ifndef.
//
// 2. The cudaFuncSetAttribute call requests the RUNTIME budget rather than
//    SMEM_MAX.
//
// The CUDA file compiles and links unmodified, but it *reads* SMEM_MAX at the
// launch, so on a 64 KB part it asks for 90 KB of dynamic shared memory and the
// launch fails with "invalid argument". "compiles and links" is not "is
// correct": a header-supplied constant can be wrong in a file that needs no
// other change at all.
//
// exl3_rdna_smem_budget() is MIN(build-time EXL3_RDNA_SMEM_MAX, the device's
// sharedMemPerBlock), so this is right on a 90 KB part too -- the same binary
// asks each device for what that device actually has.
// =============================================================================

#include "../quant/exl3_kernel_map_rdna.cuh"   // FIRST: establishes SMEM_MAX
#include <cuda_fp16.h>
#include "../../quant/exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../../quant/comp_units/exl3_moe_instances.cuh"
#include "exl3_moe_pipe_instances_rdna.cuh"
#include "exl3_moe_shape_rdna.cuh"   // AFTER common (pulled in above): must
                                       // match the kernel's MOE_TILESIZE_K, since
                                       // blockDim is derived from it here
#include "../../quant/exl3_devctx.cuh"
#include "../../quant/bits_k.cuh"
#include "exl3_moe_inner_rdna.cuh"                 // moe_pipe::smem_launch_bytes
#include <set>
#include <map>

// Blocks of `kernel` the runtime can keep resident per SM (WGP) at the MoE launch shape,
// capped at 2. The pipelined kernel is register-budgeted for two (EXL3_MOE_PIPE_WPE); the
// grid is co-resident by design (group barriers spin), so the launch never counts on more
// than the runtime reports. EXL3_ROCM_MOE_BPS=1 forces one block per WGP.
static int moe_pipe_smem(int n_tile)
{
    return n_tile == 256 ? moe_pipe::smem_launch_bytes<MOE_TILESIZE_K, 256>()
                         : moe_pipe::smem_launch_bytes<MOE_TILESIZE_K, 128>();
}

static int moe_blocks_per_sm(fp_exl3_moe_kernel kernel, int device, int smem)
{
    static int forced = -2;
    if (forced == -2)
    {
        const char* e = getenv("EXL3_ROCM_MOE_BPS");
        forced = e ? atoi(e) : -1;
    }
    if (forced == 1) return 1;
    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
    cudaFuncSetAttribute((const void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int) exl3_rdna_smem_budget(device));
    int nb = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (const void*) kernel, block_dim, smem) != hipSuccess)
    {
        (void) hipGetLastError();
        nb = 1;
    }
    return MAX(1, MIN(nb, 2));
}

static bool moe_pipe_enabled();

// Expert group width (blocks per expert): MOE_SMS_PER_EXPERT (8), or
// EXL3_ROCM_MOE_GROUP. Sets the buffer count through exl3_moe_max_concurrency,
// so it must be in the environment before the model loads.
static int moe_group_width()
{
    static int v = -1;
    if (v < 0)
    {
        const char* e = getenv("EXL3_ROCM_MOE_GROUP");
        v = e ? atoi(e) : MOE_SMS_PER_EXPERT;
        if (v < 1) v = MOE_SMS_PER_EXPERT;
    }
    return v;
}
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_pipe[];

int exl3_moe_max_concurrency(int device)
{
    int num_sms = DevCtx::instance().get_num_sms(device);
    // Buffer count for the widest case: the pipelined kernel at its verified blocks per WGP
    // (representative instance K = 2, N = 256, mul1; every pipe instance has the same
    // register budget and LDS request). The launch re-checks the instance it runs.
    int bps = moe_pipe_enabled() ? moe_blocks_per_sm(exl3_moe_kernel_instances_pipe[4 * 2 + 2 * 1 + 1], device, moe_pipe_smem(256)) : 1;
    return MIN(num_sms * bps / moe_group_width(), MOE_MAX_GROUPS);
}

std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

// EXL3_MOE_TILE_N=128 keeps the N = 128 tile shape for dims that are multiples of 256 (which
// otherwise take the N = 256 instances); 0 / unset = automatic
static int moe_tile_n_override()
{
    static int v = -1;
    if (v < 0)
    {
        const char* e = getenv("EXL3_MOE_TILE_N");
        v = e ? atoi(e) : 0;
    }
    return v;
}

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

// Pipelined mainloop instances (exl3_moe_inner_rdna.cuh), same [K][cb - 1][N_off] order
fp_exl3_moe_kernel exl3_moe_kernel_instances_pipe[] =
{
    exl3_moe_kernel_k0_n128_cb1_pipe(), exl3_moe_kernel_k0_n256_cb1_pipe(), exl3_moe_kernel_k0_n128_cb2_pipe(), exl3_moe_kernel_k0_n256_cb2_pipe(),
    exl3_moe_kernel_k1_n128_cb1_pipe(), exl3_moe_kernel_k1_n256_cb1_pipe(), exl3_moe_kernel_k1_n128_cb2_pipe(), exl3_moe_kernel_k1_n256_cb2_pipe(),
    exl3_moe_kernel_k2_n128_cb1_pipe(), exl3_moe_kernel_k2_n256_cb1_pipe(), exl3_moe_kernel_k2_n128_cb2_pipe(), exl3_moe_kernel_k2_n256_cb2_pipe(),
    exl3_moe_kernel_k3_n128_cb1_pipe(), exl3_moe_kernel_k3_n256_cb1_pipe(), exl3_moe_kernel_k3_n128_cb2_pipe(), exl3_moe_kernel_k3_n256_cb2_pipe(),
    exl3_moe_kernel_k4_n128_cb1_pipe(), exl3_moe_kernel_k4_n256_cb1_pipe(), exl3_moe_kernel_k4_n128_cb2_pipe(), exl3_moe_kernel_k4_n256_cb2_pipe(),
    exl3_moe_kernel_k5_n128_cb1_pipe(), exl3_moe_kernel_k5_n256_cb1_pipe(), exl3_moe_kernel_k5_n128_cb2_pipe(), exl3_moe_kernel_k5_n256_cb2_pipe(),
    exl3_moe_kernel_k6_n128_cb1_pipe(), exl3_moe_kernel_k6_n256_cb1_pipe(), exl3_moe_kernel_k6_n128_cb2_pipe(), exl3_moe_kernel_k6_n256_cb2_pipe(),
    exl3_moe_kernel_k7_n128_cb1_pipe(), exl3_moe_kernel_k7_n256_cb1_pipe(), exl3_moe_kernel_k7_n128_cb2_pipe(), exl3_moe_kernel_k7_n256_cb2_pipe(),
    exl3_moe_kernel_k8_n128_cb1_pipe(), exl3_moe_kernel_k8_n256_cb1_pipe(), exl3_moe_kernel_k8_n128_cb2_pipe(), exl3_moe_kernel_k8_n256_cb2_pipe()
};

// Half-integer rates on the pipelined mainloop: [K - 1][N_off], mul1 only, uniform
// gate / up / down (comp_units_rdna/exl3_moe_inst_h*_cb2.cu)
fp_exl3_moe_kernel exl3_moe_kernel_instances_pipe_half[] =
{
    exl3_moe_kernel_h1_n128_cb2_pipe(), exl3_moe_kernel_h1_n256_cb2_pipe(),
    exl3_moe_kernel_h2_n128_cb2_pipe(), exl3_moe_kernel_h2_n256_cb2_pipe(),
    exl3_moe_kernel_h3_n128_cb2_pipe(), exl3_moe_kernel_h3_n256_cb2_pipe()
};

// EXL3_ROCM_HALF_MOE_PIPE: 1 (default) = uniform half-integer rates (1.5 / 2.5 / 3.5 bpw, mul1) take
// the pipelined mainloop through the instances above; 0 = the non-pipelined K = 0 kernel.
// Read per call, and only for half-rate launches (integer-K launches never reach it)
static bool moe_half_pipe_enabled()
{
    const char* e = getenv("EXL3_ROCM_HALF_MOE_PIPE");
    return !(e && e[0] == '0');
}

// EXL3_ROCM_MOE_PIPE: 1 (default) = pipelined mainloop, 0 = the shared exl3_gemm inner.
// Read on every call (getenv is cheap next to the kernel), so one process can switch
// mainloops between calls
static bool moe_pipe_enabled()
{
    const char* e = getenv("EXL3_ROCM_MOE_PIPE");
    return !(e && e[0] == '0');
}

// RDNA: the 32 / 64-row tile instances (exl3_moe_kernel_instances_m32 / _m64 on CUDA)
// are not built here -- exl3_gemm_inner_rdna.cuh implements TILESIZE_M == 16 only.
// Launches asking for m_tile > 16 run the 16-row instance over the same expert range
// (see the kernel selection below): numerically identical, only slower.

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

    count_lo, count_hi:
        experts with token counts outside [count_lo, count_hi] are skipped (they belong to another
        launch's row tile); num_active must count the experts inside the range

    m_tile:
        rows per GEMM tile: 16 (any codebook; N = 128 or 256 tile shape by dims), 32 or 64
        (mul1 codebook, N = 128 instances). Worth it for experts holding more than 16 / 32 rows

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

    const float K_gate,
    const float K_up,
    const float K_down,

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
    const c10::optional<at::Tensor>& fused_base,
    const int count_lo,
    const int count_hi,
    const int m_tile
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

    // Bitrates. Compile-time instances for uniform integer K, the runtime-switch instance (K = 0)
    // otherwise; the kernel receives the rates in half-bit units (see bits_k.cuh), as on CUDA
    const int K2_gate = k2_from_K(K_gate), K2_up = k2_from_K(K_up), K2_down = k2_from_K(K_down);
    TORCH_CHECK(gate_mul1 || (K2_gate % 2 == 0 && K2_up % 2 == 0 && K2_down % 2 == 0),
                "exl3_moe: half-integer bitrates require the mul1 codebook");
    const bool any_half = (K2_gate | K2_up | K2_down) & 1;
    int K = 0;
    if (K2_gate == K2_up && K2_up == K2_down && K2_gate % 2 == 0) K = K2_gate / 2;

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
    // The pipelined mainloop (exl3_moe_inner_rdna.cuh) has integer-K instances and half-rate
    // instances for uniform gate / up / down K + 0.5 (mul1; EXL3_ROCM_HALF_MOE_PIPE). Any other half-integer mix
    // takes the shared exl3_gemm inner (the EXL3_ROCM_MOE_PIPE=0 kernel), whose K = 0 switch has the half_k cases
    const bool pipe_half = any_half && gate_mul1 && K2_gate == K2_up && K2_up == K2_down && K2_gate >= 3 && K2_gate <= 7
                           && moe_pipe_enabled() && moe_half_pipe_enabled();
    const bool pipe = (moe_pipe_enabled() && !any_half) || pipe_half;

    int N_off = 0;
    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0 && moe_tile_n_override() != 128) N_off = 1;
    fp_exl3_moe_kernel kernel;
    if (pipe)
    {
        // The pipelined kernel picks 16 / 32 / 64-row tiles per expert by itself (fixed-K
        // instances), so the caller's m_tile tier split runs through the same instance
        if (m_tile > 16)
            TORCH_CHECK(max_tokens_per_expert >= (size_t) m_tile, "exl3_moe: temp buffers hold fewer rows than the tile");
        kernel = pipe_half ? exl3_moe_kernel_instances_pipe_half[2 * (K2_gate / 2 - 1) + N_off]
                           : exl3_moe_kernel_instances_pipe[4 * K + 2 * cb_idx + N_off];
    }
    else if (m_tile <= 16)
    {
        kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];
    }
    else
    {
        // RDNA: no 32 / 64-row instances (see the note at the instance tables). The 16-row
        // kernel loops over every row of each expert in [count_lo, count_hi], so the caller's
        // tier split is honoured exactly; only the wide tiles' B-dequant amortisation is lost.
        TORCH_CHECK(max_tokens_per_expert >= (size_t) m_tile, "exl3_moe: temp buffers hold fewer rows than the tile");
        kernel = exl3_moe_kernel_instances[4 * K + 2 * cb_idx + N_off];
    }

    // Co-resident slots: SMs x blocks per SM (pipelined kernel: 2 when the runtime confirms
    // it, cached per instance). The buffers were sized by exl3_moe_max_concurrency; a
    // mainloop switch after allocation only lowers the group count, never oversubscribes
    static std::map<void*, int> bps_cache[MAX_DEVICES];
    int bps = 1;
    if (pipe)
    {
        auto it = bps_cache[device].find((void*) kernel);
        if (it == bps_cache[device].end())
        {
            bps = moe_blocks_per_sm(kernel, device, moe_pipe_smem(N_off ? 256 : 128));
            bps_cache[device][(void*) kernel] = bps;
        }
        else bps = it->second;
    }
    const int slots = num_sms * bps;
    int num_groups = MIN((int) concurrency, MOE_MAX_GROUPS);
    num_groups = MIN(num_groups, slots / moe_group_width());
    TORCH_CHECK(num_groups >= 1, "exl3_moe: no co-resident expert group fits the device");
    int group_size = moe_group_width();
    if (num_active > 0)
    {
        num_groups = MIN(num_groups, num_active);
        group_size = MIN(slots / num_groups, MOE_MAX_SMS_PER_EXPERT);
    }
    dim3 grid_dim(group_size, 1, num_groups);

    if (moe_kernel_attr_set[device].find((void*) kernel) == moe_kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) exl3_rdna_smem_budget(device));
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
        (void*) &K2_gate,
        (void*) &K2_up,
        (void*) &K2_down,
        (void*) &locks,
        &_output_scratch,
        &_fused_base,
        (void*) &count_lo,
        (void*) &count_hi
    };

    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        pipe ? moe_pipe_smem(N_off ? 256 : 128) : SMEM_MAX,
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
    if (!tokens) return;
    int hidden_dim = output_state.size(1);
    int num_assign = flat_expert.size(0);
    TORCH_CHECK(num_assign % tokens == 0, "exl3_moe_gather: assignments / tokens");
    int topk = num_assign / tokens;
    int num_experts = slot_kind.size(0);
    TORCH_CHECK(slot_base.size(0) >= num_experts && expert_start.size(0) >= num_experts, "exl3_moe_gather: table sizes");
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
