#include <cuda_fp16.h>
#include "exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"
#include "exl3_gemv_sm70_kernel.cuh"
#include "exl3_devctx.cuh"
#include <map>

/*
QTIP-style small-m GEMV path, kernel in exl3_gemv_kernel.cuh. Dispatched from exl3_gemm via
exl3_gemv_try_launch when the shape heuristic applies, or forced through the exl3_gemv entry
point. Kernel arguments and graph parameter offsets are identical to exl3_gemm_kernel.

Env: EXL3_GEMV = 0 disables the path, 1/unset = heuristic (default), 2 = use wherever the hard
constraints allow (testing).

Heuristic envelope (measured, RTX 3090, 4 bpw, m <= 8, vs the tuned regular kernel): the narrow
config wins 15-60% at attention-projection sizes (n <= 4096), the wide config wins ~8% at
large-n/small-k FFN sizes. Big-k x big-n shapes lose slightly and fall through to the regular
kernel, as do other architectures (Ada/Blackwell are memory-bound here and keep the regular
kernel), bpw != 4, and m > 8.
*/

static int exl3_gemv_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV");
    if (!env) return 1;
    return atoi(env);
}

// -1 = default per bits, 0 = force shuffle extraction, 1 = force smem staging (testing)
static int exl3_gemv_env_smem()
{
    const char* env = std::getenv("EXL3_GEMV_SMEM");
    if (!env) return -1;
    return atoi(env);
}

// -1: not eligible, 0: narrow config, 1: wide config. narrow_coresident = number of narrow-config
// blocks that fit on the device at once (its grid is one block per 32 output columns)
static int exl3_gemv_cfg(int cc, int size_m, int size_k, int size_n, int K, int cb, int mode, int narrow_coresident)
{
    if (mode == 0) return -1;
    if (K < 2 || K > 8) return -1;
    if (K > 4 && cc >= CC_AMPERE) return -1;              // sm70 kernel covers K=5-8
    if (K != 4 && cb == 0 && cc >= CC_AMPERE) return -1;  // sm70 kernel supports K=2,3 cb=0
    if (size_m > EXL3_GEMV_MAX_M) return -1;
    if (size_k % 128 || size_n % 128) return -1;
    //if (cc != CC_AMPERE) return -1;  // measured win on Ampere; Ada/Blackwell are memory-bound here
    if (mode == 2) return size_n <= 8192 ? 0 : 1;
    if (mode == 3) return 0;   // testing: force narrow config
    if (mode == 4) return 1;   // testing: force wide config

    // The narrow config wins (up to ~30%) whenever its grid fits in a single co-resident wave;
    // in the 1..2-wave zone the trailing partial wave costs more than the kernel gains unless
    // per-group work is small (small k). The wide config covers a band of large-n shapes with
    // small-to-mid k. Everything else runs the regular block-pipelined kernel.
    // Per-bits envelopes: 2 bpw is decode-bound and won at every measured shape on both archs;
    // 3 bpw wins everywhere on Ada but only in the narrow envelope on Ampere
    // sm_70: the gemv is the only tensor-core path — the fallback (sm80
    // block-pipelined kernels) is arch-guarded no-ops, so always take it
    // when the shape fits the cooperative launch constraints.
    // Small-n shapes: K-split across blocks (cfg 2) raises occupancy —
    // n <= 2048 gives <= 64 blocks at cfg 0, well under the 80 SMs.
    // Requires co-residency for 4x the grid (all KS chunks resident
    // for the grid.sync); else fall back to cfg 0.
    if (cc < CC_AMPERE)
    {
        if (const char* fc = std::getenv("EXL3_GEMV_FORCE_CFG"))
            return atoi(fc);
        // if (size_n <= 2048 && narrow_coresident >= (size_n / 32) * 4)
        //     return 2;
        return 0;  // cfg2 disabled: illegal-address debug
    }
    if (K == 2) return size_n <= 8192 ? 0 : 1;
    if (K == 3 && cc == CC_ADA) return size_n <= 8192 ? 0 : 1;
    if (size_n / 32 <= narrow_coresident) return 0;
    if (size_k <= 2048 && size_n <= 8192) return 0;
    if (K == 3) return -1;
    if (size_n >= 8192 && size_k <= 4096) return 1;
    if (size_n >= 8192 && size_n <= 10240 && size_k <= 5120 && cc == CC_AMPERE) return 1;
    return -1;
}

static void* exl3_gemv_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    #undef SEL
    return nullptr;
}

// sm_70 variant: same selection signature, different kernel template.
static void* exl3_gemv_sm70_select_kernel(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_sm70_kernel<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    SEL(5, 2, false, 0, 2, false) SEL(5, 2, false, 1, 2, false)
    SEL(5, 2, true,  0, 2, false) SEL(5, 2, true,  1, 2, false)
    SEL(3, 2, false, 0, 2, false) SEL(3, 2, false, 1, 2, false)
    SEL(3, 2, true,  0, 2, false) SEL(3, 2, true,  1, 2, false)
    SEL_GRID(4, 0, false) SEL_GRID(4, 1, false) SEL_GRID(4, 2, false)
    SEL_GRID(2, 0, false) SEL_GRID(2, 1, false) SEL_GRID(2, 2, false) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 0, false) SEL_GRID(3, 1, false) SEL_GRID(3, 2, false) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    SEL_GRID(4, 0, true) SEL_GRID(4, 1, true) SEL_GRID(4, 2, true)
    SEL_GRID(2, 0, true) SEL_GRID(2, 1, true) SEL_GRID(2, 2, true)
    SEL_GRID(3, 0, true) SEL_GRID(3, 1, true) SEL_GRID(3, 2, true)
    SEL_GRID(5, 0, true) SEL_GRID(5, 1, true) SEL_GRID(5, 2, true)
    SEL_GRID(5, 0, false) SEL_GRID(5, 1, false) SEL_GRID(5, 2, false)
    SEL_GRID(6, 0, false) SEL_GRID(6, 1, false) SEL_GRID(6, 2, false)
    SEL_GRID(7, 0, false) SEL_GRID(7, 1, false) SEL_GRID(7, 2, false)
    SEL_GRID(8, 0, false) SEL_GRID(8, 1, false) SEL_GRID(8, 2, false)
    SEL_GRID(6, 0, true) SEL_GRID(6, 1, true) SEL_GRID(6, 2, true)
    SEL_GRID(7, 0, true) SEL_GRID(7, 1, true) SEL_GRID(7, 2, true)
    SEL_GRID(8, 0, true) SEL_GRID(8, 1, true) SEL_GRID(8, 2, true)
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}


// sm_70 dual-matrix selector: same template params as the single-matrix
// sm70 kernel, but instantiates exl3_gemv_sm70_kernel_dual. Shapes in
// use first (K=3 cb=2 MoE gate+up pairs); extend the grid as needed.
static void* exl3_gemv_sm70_select_kernel_dual(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_sm70_kernel_dual<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    #define SEL_GRID(bits_, cb_, sm_) \
        SEL(bits_, cb_, false, 0, 0, sm_) SEL(bits_, cb_, false, 0, 1, sm_) \
        SEL(bits_, cb_, false, 1, 0, sm_) SEL(bits_, cb_, false, 1, 1, sm_) \
        SEL(bits_, cb_, true,  0, 0, sm_) SEL(bits_, cb_, true,  0, 1, sm_) \
        SEL(bits_, cb_, true,  1, 0, sm_) SEL(bits_, cb_, true,  1, 1, sm_)
    SEL_GRID(3, 2, false)
    SEL_GRID(3, 2, true)
    // cfg 2 (K-split 2-ways, 256 threads): K=3 cb=2 MoE gate+up shapes
    SEL(3, 2, false, 0, 2, false) SEL(3, 2, false, 1, 2, false)
    SEL(3, 2, true,  0, 2, false) SEL(3, 2, true,  1, 2, false)
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}

// sm_70 multi-matrix selector: pointer-table variant, N matrices
// per launch. Instantiated for the shapes in use (K=3/5, cb=2).
static void* exl3_gemv_sm70_select_kernel_multi(int bits, int cb, bool c_fp32, int mmode, int cfg, bool smem)
{
    #define SELM(bits_, cb_, fp32_, mm_, cfg_, sm_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_ && smem == sm_) \
            return (void*) exl3_gemv_sm70_kernel_multi<bits_, fp32_, cb_, mm_, cfg_, sm_>;
    SELM(5, 2, false, 0, 0, false) SELM(5, 2, false, 1, 0, false)
    SELM(5, 2, true,  0, 0, false) SELM(5, 2, true,  1, 0, false)
    SELM(3, 2, false, 0, 0, false) SELM(3, 2, false, 1, 0, false)
    SELM(3, 2, true,  0, 0, false) SELM(3, 2, true,  1, 0, false)
    #undef SELM
    return nullptr;
}


bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
)
{
    // Free integer checks first; the env read (~64 ns) and device queries only run for calls
    // that could actually take this path
    if (!has_su_sv) return false;
    if (K < 2 || K > 8) return false;
    if (K > 4 && DevCtx::instance().get_cc(device) >= CC_AMPERE) return false;  // sm70 kernel covers K=5-8
    if (K != 4 && cb == 0 && DevCtx::instance().get_cc(device) >= CC_AMPERE) return false;  // sm70 kernel supports K=2,3 cb=0
    if (size_m > EXL3_GEMV_MAX_M) return false;
    if (size_k % 128 || size_n % 128) return false;

    int mode = force ? 2 : exl3_gemv_env_mode();
    if (mode == 0) return false;
    int cc = DevCtx::instance().get_cc(device);
    // if (cc != CC_AMPERE) return false;
    int mmode = size_m == 1 ? 0 : 1;
    int num_sms = DevCtx::instance().get_num_sms(device);

    // Cooperative launch: grids are capped at full co-residency (cached per kernel), and the
    // narrow config's co-residency also feeds the shape heuristic
    static std::map<void*, int> occ_cache[MAX_DEVICES];
    auto& cache = occ_cache[device];
    auto occupancy = [&] (void* kernel, int block_dim) -> int
    {
        auto it = cache.find(kernel);
        if (it != cache.end()) return it->second;
        int blocks_per_sm;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
        cache[kernel] = blocks_per_sm;
        return blocks_per_sm;
    };

    // Extraction style: shuffle by default, smem staging selectable per call for evaluation
    bool smem = exl3_gemv_env_smem() == 1;
    // sm_70 with K >= 5: default to the register-form shuffle gather
    // (SMEM_STAGE=false) — drops sh_stage (8 KB/block at K>=5), restoring
    // K=4 occupancy. EXL3_GEMV_SMEM=1 forces the SMEM-staged variant.
    // if (cc < 8 && K >= 5) smem = true;

    // sm_70 and older: the sm80 mma kernels are no-ops (arch-guarded) —
    // route to the m8n8k4 variant, which shares the launch signature.
    void* narrow_kernel;
    if (cc < CC_AMPERE)
    {
        narrow_kernel = exl3_gemv_sm70_select_kernel(K, cb, c_fp32, mmode, 0, smem);
    }
    else
    {
        narrow_kernel = exl3_gemv_select_kernel(K, cb, c_fp32, mmode, 0, smem);
    }
    if (!narrow_kernel) return false;
    int narrow_coresident = occupancy(narrow_kernel, 512) * num_sms;

    int cfg = exl3_gemv_cfg(cc, size_m, size_k, size_n, K, cb, mode, narrow_coresident);
    if (cfg < 0) return false;

    void* kernel = nullptr;
    if (cc < CC_AMPERE)
        kernel = cfg == 0 ? narrow_kernel : exl3_gemv_sm70_select_kernel(K, cb, c_fp32, mmode, cfg, smem);
    else
        kernel = cfg == 0 ? narrow_kernel : exl3_gemv_select_kernel(K, cb, c_fp32, mmode, cfg, smem);
    if (!kernel) return false;

    int block_dim = cfg == 1 ? 256 : 512;
    int cols = cfg == 1 ? 64 : 32;
    const int ksplit = cfg == 2 ? 4 : 1;   // must match the kernel's KS

    int max_blocks = occupancy(kernel, block_dim) * num_sms;
    // TEMP: grid-cap sweep for V100 tuning (EXL3_GEMV_GRID_CAP)
    if (const char* cap_env = std::getenv("EXL3_GEMV_GRID_CAP"))
    {
        int cap = atoi(cap_env);
        if (cap > 0) max_blocks = MIN(max_blocks, cap);
    }
    // K-split: grid must cover all groups (>= n/cols) and be a
    // multiple of n/cols so every group gets an equal k-chunk count.
    // Shrink the split factor to fit co-residency; decline if even
    // one block per group doesn't fit.
    int grid;
    if (ksplit > 1)
    {
        // The kernel's KS is compile-time (4): all 4 chunks must be
        // resident. The heuristic guarantees co-residency; double-check.
        int groups = size_n / cols;
        if (max_blocks < groups * ksplit) return false;
        grid = groups * ksplit;
    }
    else
    {
        grid = MIN(size_n / cols, max_blocks);
        if (grid < 1) return false;
    }

    cuda_check(cudaLaunchCooperativeKernel
    (
        kernel,
        dim3(grid),
        dim3(block_dim),
        kernel_args,
        0,
        stream
    ));

    if (launched_kernel) *launched_kernel = kernel;
    return true;
}

void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")

    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = (half*) OPTPTR(A_had);
    const half* svh_ptr = (const half*) OPTPTR(svh);
    TORCH_CHECK(suh_ptr && A_had_ptr && svh_ptr, "exl3_gemv requires suh, A_had and svh");

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int K = B.size(2) / 16;

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int device;
    cudaGetDevice(&device);
    int* locks = DevCtx::instance().get_locks(device);
    float* ws_ptr = (float*) DevCtx::instance().get_ws(device);

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    void* kernel_args[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr,
        (void*)& ws_ptr
    };

    bool ok = exl3_gemv_try_launch
    (
        kernel_args, size_m, size_k, size_n, K, cb, c_fp32,
        true, device, stream, nullptr, true
    );
    TORCH_CHECK(ok, "exl3_gemv: call is not eligible for the GEMV kernel");

    cuda_check(cudaPeekAtLastError());
}

// Dual-matrix entry: fuses two back-to-back GEMV calls (e.g. MoE
// gate+up) into ONE kernel launch. Computes
//   C  = svh-transform(GEMV(B,  suh-transform(A)))   (slot 0)
//   C2 = svh2-transform(GEMV(B2, suh2-transform(A))) (slot 1)
// with the same A on both slots. See docs/sm70_dual_gemv_design.md.
void exl3_gemv2
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& B2,
    at::Tensor& C2,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    const c10::optional<at::Tensor>& suh2,
    const c10::optional<at::Tensor>& A_had2,
    const c10::optional<at::Tensor>& svh2,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_DIM(B2, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    TORCH_CHECK_SHAPES(B2, 0, B, 0, 1);
    TORCH_CHECK_SHAPES(C2, -1, B2, 1, 16);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    TORCH_CHECK_DTYPE(B2, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    TORCH_CHECK(c_fp32 == (C2.dtype() == at::kFloat), "exl3_gemv2: C and C2 must have matching dtypes");
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK(C2.dtype() == (c_fp32 ? at::kFloat : at::kHalf), "exl3_gemv2: C2 dtype must match C");
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")

    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = (half*) OPTPTR(A_had);
    const half* svh_ptr = (const half*) OPTPTR(svh);
    const half* suh2_ptr = (const half*) OPTPTR(suh2);
    half* A_had2_ptr = (half*) OPTPTR(A_had2);
    const half* svh2_ptr = (const half*) OPTPTR(svh2);
    TORCH_CHECK(suh_ptr && A_had_ptr && svh_ptr, "exl3_gemv2 requires suh, A_had and svh");
    TORCH_CHECK(suh2_ptr && A_had2_ptr && svh2_ptr, "exl3_gemv2 requires suh2, A_had2 and svh2");

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int size_n2 = B2.size(1) * 16;
    TORCH_CHECK(size_n == size_n2, "exl3_gemv2: B and B2 must have matching output dims");
    int K = B.size(2) / 16;
    int K2 = B2.size(2) / 16;
    TORCH_CHECK(K == K2, "exl3_gemv2: B and B2 must have matching bitrates");

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int device;
    cudaGetDevice(&device);
    int* locks = DevCtx::instance().get_locks(device);
    float* ws_ptr = (float*) DevCtx::instance().get_ws(device);

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    const uint16_t* B2_ptr = (const uint16_t*) B2.data_ptr();
    void* C2_ptr = (void*) C2.data_ptr();

    // EXL3_GEMM_ARGS_DUAL order: A, B, C, m, k, n, locks, suh, A_had,
    // svh, B2, C2, suh2, A_had2, svh2
    void* kernel_args[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr,
        (void*)& B2_ptr,
        (void*)& C2_ptr,
        (void*)& suh2_ptr,
        (void*)& A_had2_ptr,
        (void*)& svh2_ptr,
        (void*)& ws_ptr   // K-split arg; unused by the dual (CFG 0)
    };

    // cfg 0: 512 threads, 32 cols/block, no K-split. cfg 2 (env-gated):
    // 256 threads, K-split 2-ways — engages all 80 SMs at n=2048
    // (grid 128 vs 64) where cfg 0 leaves 16 SMs idle.
    int cfg = 0;
    if (const char* env = std::getenv("EXL3_GEMV2_CFG")) cfg = atoi(env);

    void* kernel = exl3_gemv_sm70_select_kernel_dual(K, cb, c_fp32, size_m == 1 ? 0 : 1, cfg, false);
    TORCH_CHECK(kernel, "exl3_gemv2: no dual kernel for this shape (K, cb, mmode, cfg, smem)");

    int num_sms = DevCtx::instance().get_num_sms(device);
    int block_dim = cfg == 0 ? 512 : 256;
    int cols = 32;
    int blocks_per_sm;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
    int max_blocks = blocks_per_sm * num_sms;
    int grid;
    if (cfg == 2)
    {
        const int ksplit = 2;
        int groups = size_n / cols;
        TORCH_CHECK(max_blocks >= groups * ksplit,
            "exl3_gemv2: cfg 2 grid does not fit co-residency (", max_blocks, " < ", groups * ksplit, ")");
        grid = groups * ksplit;
    }
    else
    {
        grid = MIN(size_n / cols, max_blocks);
    }
    TORCH_CHECK(grid >= 1, "exl3_gemv2: output too small for one block");

    cuda_check(cudaLaunchCooperativeKernel
    (
        kernel,
        dim3(grid),
        dim3(block_dim),
        kernel_args,
        0,
        stream
    ));

    cuda_check(cudaPeekAtLastError());
}

// Multi-matrix entry: N matrices in ONE launch via pointer tables.
// Tables are device int64 tensors of N pointers each (sm80 mgemm
// contract): A_list/B_list/C_list/suh_list/A_had_list/svh_list.
// All matrices share size_k; each matrix's output width is
// n_list[mi]. See docs/sm70_perf_report.md.
void exl3_gemv_multi
(
    const at::Tensor& A_list,
    const at::Tensor& B_list,
    const at::Tensor& C_list,
    const at::Tensor& suh_list,
    const at::Tensor& A_had_list,
    const at::Tensor& svh_list,
    int64_t n_mat,
    int64_t size_k,
    int64_t size_n,
    int64_t K,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A_list.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int device;
    cudaGetDevice(&device);
    int* locks = DevCtx::instance().get_locks(device);
    float* ws_ptr = (float*) DevCtx::instance().get_ws(device);

    const half* const* A_ptr = (const half* const*) A_list.data_ptr();
    const uint16_t* const* B_ptr = (const uint16_t* const*) B_list.data_ptr();
    void* const* C_ptr = (void* const*) C_list.data_ptr();
    const half* const* suh_ptr = (const half* const*) suh_list.data_ptr();
    half* const* A_had_ptr = (half* const*) A_had_list.data_ptr();
    const half* const* svh_ptr = (const half* const*) svh_list.data_ptr();

    int size_m = 1;
    int nm = (int) n_mat;

    // Arg order must match EXL3_GEMM_ARGS_MULTI: A, B, C, m, k, n,
    // locks, suh, A_had, svh, A_list, B_list, C_list, suh_list,
    // A_had_list, svh_list, n_mat, ws
    void* kernel_args_multi[] =
    {
        (void*)& A_ptr, (void*)& B_ptr, (void*)& C_ptr,
        (void*)& size_m, (void*)& size_k, (void*)& size_n,
        (void*)& locks, (void*)& suh_ptr, (void*)& A_had_ptr,
        (void*)& svh_ptr,
        (void*)& A_ptr, (void*)& B_ptr, (void*)& C_ptr,
        (void*)& suh_ptr, (void*)& A_had_ptr, (void*)& svh_ptr,
        (void*)& nm,
        (void*)& ws_ptr
    };

    void* kernel = exl3_gemv_sm70_select_kernel_multi((int) K, mul1 ? 2 : (mcg ? 1 : 0), false, 0, 0, false);
    TORCH_CHECK(kernel, "exl3_gemv_multi: no multi kernel for this shape");

    int num_sms = DevCtx::instance().get_num_sms(device);
    int block_dim = 512;
    int blocks_per_sm;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, block_dim, 0));
    int max_blocks = blocks_per_sm * num_sms;
    int grid = MIN(size_n / 32, max_blocks);
    TORCH_CHECK(grid >= 1, "exl3_gemv_multi: output too small");

    cuda_check(cudaLaunchCooperativeKernel
    (
        kernel,
        dim3(grid),
        dim3(block_dim),
        kernel_args_multi,
        0,
        stream
    ));

    cuda_check(cudaPeekAtLastError());
}
