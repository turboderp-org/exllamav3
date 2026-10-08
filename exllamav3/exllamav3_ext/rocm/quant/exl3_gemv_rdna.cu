// =============================================================================
// exl3_gemv host dispatch for RDNA
// =============================================================================
//
// Provides quant/exl3_gemv.cu's interface -- exl3_gemv_try_launch and
// exl3_gemv -- on top of an fdot2 GEMV kernel.
//
// What the CUDA version does that this one does not, and why:
//
// - Cooperative launch. The CUDA GEMV kernel fuses the input Hadamard, the
//   matmul and the output Hadamard into one cooperative kernel separated by
//   grid.sync(). The RDNA kernel is a plain launch with the transforms as
//   separate kernels either side, so this is three launches, not one.
//
// - Occupancy-driven grid sizing and the narrow/wide config pair. Those exist to
//   size a cooperative grid; with a plain launch the grid is just one warp per
//   n-tile. Wave count per block is chosen instead.
//
// - The shape heuristic. The CUDA envelope (K == 4 or cb != 0, n <= 8192 bands,
//   co-residency thresholds) was tuned on Ampere against the CUDA GEMM. None of
//   it transfers. See exl3_gemv_rdna_eligible() below for what replaces it.
//
// - m up to 8. This kernel is m == 1 only; m = 2..8 has its own path
//   (exl3_gemv_multirow_rdna.cu).
//
// Bits 1-8 and all three codebooks are supported, which is wider than the CUDA
// GEMV path (K 2-4, and K == 4 only for cb 0).
// =============================================================================

#include <cuda_fp16.h>
#include "../../quant/exl3_gemv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "exl3_gemv_kernel_rdna.cuh"
#include "exl3_mgemv_rdna.cuh"
#include "exl3_gemv_multirow_rdna.cuh"

#include <cstdlib>
#include <mutex>

// Defined in the graph-captured GEMV section at the bottom of this file;
// called from the non-graph dispatch above it.
static void exl3_gemv_graph_prewarm(int device);

// Env: EXL3_GEMV = 0 disables the path (every eligible call falls through to
// the cooperative GEMM); 1 (default) routes shape-aware per
// exl3_gemv_rdna_pays(); 2 takes every eligible call regardless of the
// shape envelope -- the CUDA path's mode-2 semantics. Re-read on every call so
// one process can toggle it.
static int exl3_gemv_env_mode()
{
    const char* env = std::getenv("EXL3_GEMV");
    if (!env) return 1;
    return atoi(env);
}

// EXL3_GEMV_SPLITK = 0 pins the single-warp kernel for every shape -- the A/B
// switch for the in-block split-K form. Re-read per call, like EXL3_GEMV.
bool exl3_gemv_splitk_enabled()
{
    const char* env = std::getenv("EXL3_GEMV_SPLITK");
    if (!env) return true;
    return atoi(env) != 0;
}

// EXL3_GEMV_LDS = 1 pins the LDS dot-tile core in every GEMV form -- the A/B
// and kill switch for the barrier-free core (exl3_gemv_dot_tile_direct).
// Re-read per call; graph captures bake the value read at capture time.
// EXL3_GEMV_FUSE_OUT=0 keeps the fused dot kernels' epilogue to the plain
// store and launches the separate output-rotation (and, multi-matrix,
// reduction) kernels. Re-read per call; shared with exl3_mgemv_rdna.cu.
bool exl3_gemv_fuse_out_enabled()
{
    const char* env = std::getenv("EXL3_GEMV_FUSE_OUT");
    if (!env) return true;
    return atoi(env) != 0;
}

bool exl3_gemv_lds_core()
{
    const char* env = std::getenv("EXL3_GEMV_LDS");
    if (!env) return false;
    return atoi(env) != 0;
}

// EXL3_ROCM_GEMV_TILES=0 selects the barrier-free direct core (bit-identical
// outputs, one n-tile per wave); EXL3_GEMV_LDS=1 still pins the LDS core over
// both. Re-read per call.
int exl3_gemv_core_mode()
{
    if (exl3_gemv_lds_core()) return EXL3_GEMV_CORE_LDS;
    const char* env = std::getenv("EXL3_ROCM_GEMV_TILES");
    if (env && atoi(env) == 0) return EXL3_GEMV_CORE_DIRECT;
    return EXL3_GEMV_CORE_TILES;
}

// N-tiles per wave (see exl3_gemv_kernel_rdna.cuh): 2 on every RDNA part, but
// only while the halved grid still carries enough waves per SIMD: below that
// the block count, not the per-wave instruction count, limits. The wave floor
// scales with the part's multiProcessorCount (1024 waves per 20 WGPs).
// EXL3_GEMV_TILES_T=1|2 overrides.
int exl3_gemv_tiles_tpb(int core, int device, int n_tiles, int bszm, int warps)
{
    if (core != EXL3_GEMV_CORE_TILES) return 1;
    const char* env = std::getenv("EXL3_GEMV_TILES_T");
    if (env)
    {
        int v = atoi(env);
        return v >= 2 ? EXL3_GEMV_TILES_TMAX : 1;
    }
    static int mpc[64] = {};
    int sms = 20;
    if (device >= 0 && device < 64)
    {
        if (!mpc[device])
        {
            int v = 0;
            if (hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess || v <= 0)
            {
                (void) hipGetLastError();
                v = 20;
            }
            mpc[device] = v;
        }
        sms = mpc[device];
    }
    const long waves_t2 = (long) (n_tiles / 2) * bszm * warps;
    return waves_t2 >= 1024L * sms / 20 ? 2 : 1;
}

// -----------------------------------------------------------------------------
// Eligibility
// -----------------------------------------------------------------------------
// These are hard constraints of the kernel, not a performance envelope:
//
// - m == 1: the kernel has no M loop; A is indexed as a bare k-vector.
// - bits 1-8: the full EXL3 range, all covered by dq_dispatch.
// - k % 128, n % 128: the kernel itself only needs multiples of 16, but the
//   Hadamard helpers work on 128-element blocks. The CUDA path applies the same
//   two constraints, so nothing eligible there is rejected here for shape.
//
// Deliberately NOT a constraint: has_su_sv. The CUDA kernel implements the
// transforms internally and so requires all three tensors; here they are
// separate kernels that are simply skipped when absent.
static bool exl3_gemv_rdna_eligible
(
    int size_m, int size_k, int size_n, int K,
    const half* suh_ptr, const half* A_had_ptr
)
{
    if (size_m != 1) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;

    // suh without A_had: the "Must supply A_had with suh" check is commented
    // out in exl3_gemm.cu, so this reaches the kernel, where the CUDA path
    // dereferences the null A_had and faults. This path would instead skip the
    // transform and return a plausible but silently untransformed result, which
    // is strictly worse. Decline and let the GEMM handle it, so the two backends
    // fail the same way on the same malformed call.
    if (suh_ptr && !A_had_ptr) return false;

    return true;
}

// -----------------------------------------------------------------------------
// Wave selection
// -----------------------------------------------------------------------------
// One warp computes one 16-wide output tile regardless; this only sets how many share a block, which trades launch
// granularity against per-block LDS.
static int exl3_gemv_rdna_warps(int n_tiles, int k_blocks)
{
    if (n_tiles <= 32) return 16;
    if (n_tiles <= 96 && k_blocks <= 256) return 8;
    return 4;
}

// Split-K wave count, shared by all three split-K sites (plain, graph, mgemv).
// The best count is shape-dependent: more warps pay while the grid is starved
// for waves (or k is long), fewer warps pay once k is short or the block count
// alone saturates the device (the in-block reduce and the shrinking per-warp
// k-chunk are pure overhead then). blocks = n_tiles x bszm (the mgemv grid
// multiplies by the expert count, which is why bszm is part of the rule, not
// just n_tiles).
// EXL3_GEMV_SPLITK_WARPS forces one count everywhere; re-read per call, baked
// at graph capture like the other switches.
int exl3_gemv_splitk_warps(int k_tiles, int n_tiles, int bszm)
{
    const char* env = std::getenv("EXL3_GEMV_SPLITK_WARPS");
    if (env)
    {
        int w = atoi(env);
        if (w == 4 || w == 8 || w == 16) return w;
    }
    int blocks = n_tiles * (bszm > 0 ? bszm : 1);
    if (k_tiles <= 128) return 4;
    if (blocks >= 1024) return 4;
    if (blocks < 128 || k_tiles >= 1024) return 16;
    return 8;
}

// -----------------------------------------------------------------------------
// Launch
// -----------------------------------------------------------------------------

static void exl3_gemv_rdna_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    int size_k,
    int size_n,
    int bits,
    int cb,
    bool c_fp32,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    cudaStream_t stream
)
{
    // =========================================================================
    // Input Hadamard transform (SUH)
    // =========================================================================
    const half* gemv_input = A_ptr;

    if (suh_ptr && A_had_ptr)
    {
        int warps_needed = size_k / 128;
        int threads_per_block = 256;
        int blocks = (warps_needed + threads_per_block / 32 - 1) / (threads_per_block / 32);

        hipLaunchKernelGGL
        (
            exl3_gemv_rdna_had_in_kernel,
            dim3(blocks), dim3(threads_per_block), 0, stream,
            A_ptr, A_had_ptr, suh_ptr, size_k
        );

        gemv_input = A_had_ptr;
    }

    // =========================================================================
    // GEMV
    // =========================================================================
    int n_tiles = size_n / 16;
    int k_blocks = size_k / 16;

    // Narrow outputs starve the single-warp form (n_tiles warps total); the
    // split-K form gives them WARPS x the wave count. Wide outputs keep the
    // single-warp form. EXL3_GEMV_SPLITK=0 pins the single-warp form
    // everywhere (A/B switch).
    bool splitk = exl3_gemv_splitk_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES;

    int warps_per_block = exl3_gemv_rdna_warps(n_tiles, k_blocks);
    int num_blocks = (n_tiles + warps_per_block - 1) / warps_per_block;

    // Single source for the LDS figure -- see exl3_gemv_kernel_rdna.cuh
    size_t shared_mem_size = exl3_gemv_smem_bytes(warps_per_block);

    if (splitk)
    {
        warps_per_block = exl3_gemv_splitk_warps(k_blocks, n_tiles, 1);
        num_blocks = n_tiles;
        shared_mem_size = exl3_gemv_smem_bytes_splitk(warps_per_block);
    }

    const int lds_core = exl3_gemv_core_mode();

    #define LAUNCH_GEMV_WAVES_K(bits_val, codebook, warps, kernel_name) \
        if (c_fp32) { \
            hipLaunchKernelGGL( \
                (kernel_name<bits_val, true, codebook, warps>), \
                dim3(num_blocks), dim3(warps * 32), shared_mem_size, stream, \
                gemv_input, B_ptr, C_ptr, size_k, size_n, lds_core); \
        } else { \
            hipLaunchKernelGGL( \
                (kernel_name<bits_val, false, codebook, warps>), \
                dim3(num_blocks), dim3(warps * 32), shared_mem_size, stream, \
                gemv_input, B_ptr, C_ptr, size_k, size_n, lds_core); \
        }

    #define LAUNCH_GEMV_WAVES(bits_val, codebook, warps) \
        if (splitk) { \
            LAUNCH_GEMV_WAVES_K(bits_val, codebook, warps, exl3_gemv_dot_kernel_splitk) \
        } else { \
            LAUNCH_GEMV_WAVES_K(bits_val, codebook, warps, exl3_gemv_dot_kernel) \
        }

    #define LAUNCH_GEMV_CB(bits_val, codebook) \
        switch (warps_per_block) { \
            case 4:  LAUNCH_GEMV_WAVES(bits_val, codebook, 4);  break; \
            case 8:  LAUNCH_GEMV_WAVES(bits_val, codebook, 8);  break; \
            case 16: LAUNCH_GEMV_WAVES(bits_val, codebook, 16); break; \
        }

    #define LAUNCH_GEMV(bits_val) \
        switch (cb) { \
            case 0: LAUNCH_GEMV_CB(bits_val, 0); break; \
            case 1: LAUNCH_GEMV_CB(bits_val, 1); break; \
            case 2: LAUNCH_GEMV_CB(bits_val, 2); break; \
        }

    switch (bits) {
        case 1: LAUNCH_GEMV(1); break;
        case 2: LAUNCH_GEMV(2); break;
        case 3: LAUNCH_GEMV(3); break;
        case 4: LAUNCH_GEMV(4); break;
        case 5: LAUNCH_GEMV(5); break;
        case 6: LAUNCH_GEMV(6); break;
        case 7: LAUNCH_GEMV(7); break;
        case 8: LAUNCH_GEMV(8); break;
    }

    #undef LAUNCH_GEMV
    #undef LAUNCH_GEMV_CB
    #undef LAUNCH_GEMV_WAVES
    #undef LAUNCH_GEMV_WAVES_K

    // =========================================================================
    // Output Hadamard transform (SVH), in place on C
    // =========================================================================
    if (svh_ptr)
    {
        int warps_needed = size_n / 128;
        int threads_per_block = 256;
        int blocks = (warps_needed + threads_per_block / 32 - 1) / (threads_per_block / 32);

        if (c_fp32)
        {
            hipLaunchKernelGGL
            (
                exl3_gemv_rdna_had_out_float_kernel,
                dim3(blocks), dim3(threads_per_block), 0, stream,
                (const float*) C_ptr, (float*) C_ptr, svh_ptr, size_n
            );
        }
        else
        {
            hipLaunchKernelGGL
            (
                exl3_gemv_rdna_had_out_half_kernel,
                dim3(blocks), dim3(threads_per_block), 0, stream,
                (const half*) C_ptr, (half*) C_ptr, svh_ptr, size_n
            );
        }
    }
}

// -----------------------------------------------------------------------------
// exl3_gemv_try_launch
// -----------------------------------------------------------------------------
// kernel_args is exl3_gemm's argument array, and its layout is part of the
// contract (exl3_gemm.cu builds it immediately above the call):
//
//   0 A   1 B   2 C   3 size_m   4 size_k   5 size_n   6 locks
//   7 suh   8 A_had   9 svh
//
// Each entry points at the caller's own copy of the argument, so the values are
// read back by dereferencing. locks is unused here: this path has no cross-block
// reduction.
//
// *launched_kernel is set to nullptr on success rather than to a kernel pointer.
// It exists so the caller can record graph parameter offsets, and those offsets
// (7, 8, 9) do not exist on a 5-argument kernel -- patching them would write past
// the node's parameter array. The caller must therefore not take this path while
// capturing; exl3_gemm.cu enforces that with an explicit !graph guard. nullptr
// here means a caller that ignores
// that rule fails loudly in Graph::record() instead of corrupting a node.
bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
)
{
    // Pre-warm the graph path's parameter block while we are guaranteed to be
    // outside capture (this function is only routed to when !graph). See
    // exl3_gemv_graph_param_block for why allocation cannot happen mid-capture.
    exl3_gemv_graph_prewarm(device);
    exl3_gemv_multirow_prewarm(device);   // the m = 2..8 path's block, same reason

    // Half-integer bitrates (1.5 / 2.5 / 3.5 bpw, mul1): the CUDA path has GEMV instances for
    // them (exl3_gemv_half_inst.cu). The RDNA dot cores (direct / LDS / tiles) decode integer K
    // only, so decline: the caller falls through to the cooperative GEMM, which has half_k
    // instances (quant/comp_units/exl3_comp_unit_h*.cu). Correct, not the fast path.
    if (half_k) return false;

    int mode = exl3_gemv_env_mode();
    if (!force && mode == 0) return false;
    // Shape-aware envelope (see exl3_gemv_rdna_pays): below the
    // profitability threshold the cooperative GEMM is faster, so decline
    // unless forced or in take-everything mode.
    if (!force && mode != 2 && !exl3_gemv_rdna_pays(size_k, size_n)) return false;

    const half*     A_ptr     = *(const half* const*)     kernel_args[0];
    const uint16_t* B_ptr     = *(const uint16_t* const*) kernel_args[1];
    void*           C_ptr     = *(void* const*)           kernel_args[2];
    const half*     suh_ptr   = *(const half* const*)     kernel_args[7];
    half*           A_had_ptr = *(half* const*)           kernel_args[8];
    const half*     svh_ptr   = *(const half* const*)     kernel_args[9];

    if (!exl3_gemv_rdna_eligible(size_m, size_k, size_n, K, suh_ptr, A_had_ptr)) return false;

    exl3_gemv_rdna_launch
    (
        A_ptr, B_ptr, C_ptr, size_k, size_n, K, cb, c_fp32,
        suh_ptr, A_had_ptr, svh_ptr, stream
    );

    if (launched_kernel) *launched_kernel = nullptr;
    return true;
}

// -----------------------------------------------------------------------------
// exl3_gemv -- direct entry point (testing), bound as exl3_gemv
// -----------------------------------------------------------------------------

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
    TORCH_CHECK(C.numel() >= (int64_t) size_m * size_n, "exl3_gemv: C must hold one output row per row of A");
    // A half-integer tile (16 * K + 8 uint16) has no RDNA GEMV core; see exl3_gemv_try_launch
    TORCH_CHECK(B.size(2) % 16 == 0, "exl3_gemv: half-integer bitrates are not supported by the RDNA GEMV kernel "
                "(exl3_gemm runs them on the cooperative GEMM)");
    int K = B.size(2) / 16;

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    TORCH_CHECK(exl3_gemv_rdna_eligible(size_m, size_k, size_n, K, suh_ptr, A_had_ptr),
        "exl3_gemv: call is not eligible for the RDNA GEMV kernel "
        "(requires size_m == 1, 1 <= K <= 8, size_k % 128 == 0, size_n % 128 == 0)");

    exl3_gemv_rdna_launch
    (
        (const half*) A.data_ptr(),
        (const uint16_t*) B.data_ptr(),
        (void*) C.data_ptr(),
        size_k, size_n, K, cb, c_fp32,
        suh_ptr, A_had_ptr, svh_ptr,
        stream
    );

    cuda_check(cudaPeekAtLastError());
}

// =============================================================================
// Graph-captured GEMV
// =============================================================================
//
// The plain-launch GEMV above is declined while a graph is capturing, because
// exl3_gemm's six recorded parameter sites (A, B_trellis, C, suh, A_had, svh)
// have no home on a 5-argument kernel. This section gives them one, using the
// pattern proven by the multi-matrix path (exl3_mgemv_rdna.cu): a prologue
// kernel that takes all six as ordinary arguments -- so it is the single node
// Graph::launch() patches -- performs the input rotation, and republishes the
// four pointers the downstream kernels need (B, C, A_had, svh) through a
// per-device Exl3GemvGraphParams block. Graph::launch() patches ONE site per
// caller params entry, so spreading the six GP ids across three kernels
// would leave the later copies stale; the republish is what makes a multi-kernel node sequence patchable at all.
//
// This path requires all three transform tensors (suh, A_had, svh). EXL3
// quant linears always carry su/sv, so nothing real is excluded. Kill switch:
// EXL3_GEMV_GRAPH=0 (the routing site also honours EXL3_GEMV=0/1/2 exactly
// like the non-graph path).
//
// Since the launch-count fusion (see the kernels below) the
// prologue kernel is gone: the dot kernel itself hosts the six patch sites,
// rotates the input in its prologue and the output in its epilogue, so this
// path is ONE launch per call. The parameter block survives for the fallback
// output kernel (EXL3_GEMV_FUSE_OUT=0) and for the epilogue's arrival
// counters. The paragraph above describes the three-kernel form the fusion
// replaced; the patch-site reasoning in it still holds, with "the dot kernel"
// for "the prologue".

static bool exl3_gemv_graph_enabled()
{
    const char* env = std::getenv("EXL3_GEMV_GRAPH");
    if (!env) return true;
    return atoi(env) != 0;
}

// allow_alloc must be false while a graph is capturing: hipMalloc synchronizes
// the device, which INVALIDATES an active stream capture ("operation failed due
// to a previous error during capture" on the next call). The eager first pass
// every BC module runs before its capture is where allocation happens -- the
// non-graph try_launch below pre-warms unconditionally -- and a capture that
// somehow arrives on a cold device declines to the cooperative kernel instead.
static Exl3GemvGraphParams* exl3_gemv_graph_param_block(int device, bool allow_alloc)
{
    static std::mutex mtx;
    static Exl3GemvGraphParams* blocks[64] = {};
    if (device < 0 || device >= 64) return nullptr;
    std::lock_guard<std::mutex> lock(mtx);
    if (!blocks[device] && allow_alloc)
    {
        void* p = nullptr;
        if (cudaMalloc(&p, sizeof(Exl3GemvGraphParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            return nullptr;
        }
        // The fused epilogue's arrival counters must start at zero
        if (cudaMemset(p, 0, sizeof(Exl3GemvGraphParams)) != cudaSuccess)
        {
            (void) cudaGetLastError();
            cudaFree(p);
            return nullptr;
        }
        blocks[device] = (Exl3GemvGraphParams*) p;
    }
    return blocks[device];
}

static void exl3_gemv_graph_prewarm(int device)
{
    (void) exl3_gemv_graph_param_block(device, true);
}

// Launch-count fusion (the design is written up in
// exl3_mgemv_rdna.cu under the same heading, and the helpers live in
// exl3_gemv_kernel_rdna.cuh). The input rotation used to be its own kernel
// (exl3_gemv_graph_had_in_kernel, which also hosted the six graph patch
// sites) and the output rotation another; both are now the dot kernel's
// prologue and epilogue, so a dense linear at m == 1 is one launch instead
// of three. The dot kernel therefore takes the six patchable pointers as its
// first six arguments -- order 0-5 is load-bearing, recorded in the
// cooperative kernel's order so the callers' params subsequences
// match -- and republishes C and svh to the parameter block for the fallback
// output kernel (EXL3_GEMV_FUSE_OUT=0). B, C and svh take no part in the
// prologue's own math; A_had is neither read nor written any more.
//
// Every early exit before the __syncthreads that publishes the rotated input
// is block-uniform; the single-warp form's per-warp tile exit comes after it.

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_graph_dot_kernel
(
    const half* __restrict__ A,        // 0: GP_gemm_A
    const uint16_t* __restrict__ B,    // 1: GP_gemm_B_trellis
    void* __restrict__ C,              // 2: GP_gemm_C
    const half* __restrict__ suh,      // 3: GP_gemm_B_suh
    half* __restrict__ A_had,          // 4: GP_gemm_A_had (patch host only)
    const half* __restrict__ svh,      // 5: GP_gemm_B_svh
    Exl3GemvGraphParams* __restrict__ pb,
    const int size_k,
    const int size_n,
    const int core,
    const bool fuse_out
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->B = B;
        pb->C = C;
        pb->A_had = A_had;
        pb->svh = svh;
    }

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n_tiles = size_n / 16;

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    half* sh_a = (half*) (sh_b_quant + WARPS_PER_BLOCK * EXL3_GEMV_SH_QUANT_U16);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    exl3_gemv_rotate_in<WARPS_PER_BLOCK>(A, suh, sh_a, size_k, warp_id, lane);
    __syncthreads();

    const int tile_n = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (tile_n >= n_tiles) return;

    float accum = exl3_gemv_dot_tile_sel<bits, cb, true, true>
    (
        core, sh_a, B, size_k, n_tiles, tile_n, lane, my_sh_b, my_sh_b_quant, 0, size_k / 16
    );

    if (fuse_out)
        exl3_gemv_fused_epilogue<c_fp32>
        (
            accum, C, 0, size_n, tile_n, lane, svh, 0.088388347648f,  // 1/sqrt(128)
            pb->seg_counters + tile_n / 8, nullptr, 0, 1
        );
    else if (lane < 16)
    {
        const int out_idx = tile_n * 16 + lane;
        if constexpr (c_fp32)
            ((float*) C)[out_idx] = accum;
        else
            ((half*) C)[out_idx] = __float2half(accum);
    }
}

template <int bits, bool c_fp32, int cb, int WARPS_PER_BLOCK>
static __global__
__launch_bounds__(WARPS_PER_BLOCK * 32)
__attribute__((amdgpu_flat_work_group_size(WARPS_PER_BLOCK * 32, WARPS_PER_BLOCK * 32)))
void exl3_gemv_graph_dot_kernel_splitk
(
    const half* __restrict__ A,        // 0: GP_gemm_A
    const uint16_t* __restrict__ B,    // 1: GP_gemm_B_trellis
    void* __restrict__ C,              // 2: GP_gemm_C
    const half* __restrict__ suh,      // 3: GP_gemm_B_suh
    half* __restrict__ A_had,          // 4: GP_gemm_A_had (patch host only)
    const half* __restrict__ svh,      // 5: GP_gemm_B_svh
    Exl3GemvGraphParams* __restrict__ pb,
    const int size_k,
    const int size_n,
    const int core,
    const bool fuse_out
)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        pb->B = B;
        pb->C = C;
        pb->A_had = A_had;
        pb->svh = svh;
    }

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile_n = blockIdx.x;
    const int n_tiles = size_n / 16;

    extern __shared__ char shared_mem[];
    constexpr int SH_STRIDE = EXL3_GEMV_SH_STRIDE;
    half* sh_b_dq = (half*) shared_mem;
    uint16_t* sh_b_quant = (uint16_t*) (sh_b_dq + WARPS_PER_BLOCK * 16 * SH_STRIDE);
    float* sh_red = (float*) (sh_b_quant + WARPS_PER_BLOCK * EXL3_GEMV_SH_QUANT_U16);
    half* sh_a = (half*) (sh_red + WARPS_PER_BLOCK * 16 * EXL3_GEMV_TILES_TMAX);
    half* my_sh_b = sh_b_dq + warp_id * 16 * SH_STRIDE;
    uint16_t* my_sh_b_quant = sh_b_quant + warp_id * EXL3_GEMV_SH_QUANT_U16;

    exl3_gemv_rotate_in<WARPS_PER_BLOCK>(A, suh, sh_a, size_k, warp_id, lane);
    __syncthreads();

    float accum = exl3_gemv_dot_tile_splitk<bits, cb, WARPS_PER_BLOCK, true>
    (
        sh_a, B, size_k, n_tiles, tile_n, warp_id, lane,
        my_sh_b, my_sh_b_quant, sh_red, core
    );

    if (warp_id != 0) return;
    if (fuse_out)
        exl3_gemv_fused_epilogue<c_fp32>
        (
            accum, C, 0, size_n, tile_n, lane, svh, 0.088388347648f,  // 1/sqrt(128)
            pb->seg_counters + tile_n / 8, nullptr, 0, 1
        );
    else if (lane < 16)
    {
        const int out_idx = tile_n * 16 + lane;
        if constexpr (c_fp32)
            ((float*) C)[out_idx] = accum;
        else
            ((half*) C)[out_idx] = __float2half(accum);
    }
}

// Fallback output rotation (EXL3_GEMV_FUSE_OUT=0, or more segments than the
// counters hold); the fused epilogue reproduces it bit for bit.
template <bool c_fp32>
static __global__
__launch_bounds__(256)
void exl3_gemv_graph_had_out_kernel
(
    const Exl3GemvGraphParams* __restrict__ pb,
    const int size_n
)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp_id >= size_n / 128) return;
    int offset = warp_id * 128;
    if constexpr (c_fp32)
        had_ff_r_128_inner<false, true>
        (
            ((const float*) pb->C) + offset,
            ((float*) pb->C) + offset,
            pb->svh + offset,
            0.088388347648f  // 1/sqrt(128)
        );
    else
        had_hf_r_128_inner<false, true>
        (
            ((const half*) pb->C) + offset,
            ((half*) pb->C) + offset,
            pb->svh + offset,
            0.088388347648f  // 1/sqrt(128)
        );
}

// Graph patch sites, recorded against the dot-kernel instantiation that was
// launched, in the cooperative kernel's order.
static void exl3_gemv_graph_record_sites(Graph* graph, void* k)
{
    graph->record_param(k, GP_gemm_A, 0);
    graph->record_param(k, GP_gemm_B_trellis, 1);
    graph->record_param(k, GP_gemm_C, 2);
    graph->record_param(k, GP_gemm_B_suh, 3);
    graph->record_param(k, GP_gemm_A_had, 4);
    graph->record_param(k, GP_gemm_B_svh, 5);
    graph->record_param(k, GP_end, 0);
}

bool exl3_gemv_graph_try_launch
(
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    const half* suh_ptr,
    half* A_had_ptr,
    const half* svh_ptr,
    int size_m,
    int size_k,
    int size_n,
    int K,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    if (size_m != 1) return false;
    if (K < 1 || K > 8) return false;
    if (size_k % 128 || size_n % 128) return false;
    if (!suh_ptr || !A_had_ptr || !svh_ptr) return false;

    int mode = exl3_gemv_env_mode();
    if (mode == 0) return false;
    if (mode != 2 && !exl3_gemv_rdna_pays(size_k, size_n)) return false;
    if (!exl3_gemv_graph_enabled()) return false;

    // Never allocate here: this function only runs during capture, where
    // hipMalloc would invalidate the graph. Cold block -> cooperative kernel.
    Exl3GemvGraphParams* pb = exl3_gemv_graph_param_block(device, false);
    if (!pb) return false;

    // Fused output epilogue unless switched off or the segment counters would
    // not fit this width (lm_head-scale outputs fit: 100352 / 128 = 784)
    const bool fuse_out = exl3_gemv_fuse_out_enabled()
        && size_n / 128 <= EXL3_GEMV_SEG_COUNTERS;

    // 1. GEMV, with the input rotation as its prologue and the output rotation
    //    as its epilogue -- same wave selection and split-K rule as the
    //    non-graph path. The captured kernel choice is baked at capture; shapes
    //    are static per call site, so the heuristic is graph-stable by
    //    construction. The six graph patch sites are recorded against the
    //    instantiation that was launched.
    {
        int n_tiles = size_n / 16;
        int k_blocks = size_k / 16;
        bool splitk = exl3_gemv_splitk_enabled() && n_tiles <= EXL3_GEMV_SPLITK_MAX_TILES;

        int warps_per_block = exl3_gemv_rdna_warps(n_tiles, k_blocks);
        int num_blocks = (n_tiles + warps_per_block - 1) / warps_per_block;
        if (splitk)
        {
            warps_per_block = exl3_gemv_splitk_warps(k_blocks, n_tiles, 1);
            num_blocks = n_tiles;
        }
        size_t smem = exl3_gemv_smem_bytes_fused(warps_per_block, splitk, size_k);

        const int lds_core = exl3_gemv_core_mode();

        #define LAUNCH_GGEMV_WAVES_K(bits_val, codebook, warps, kernel_name) \
            if (c_fp32) { \
                hipLaunchKernelGGL( \
                    (kernel_name<bits_val, true, codebook, warps>), \
                    dim3(num_blocks), dim3(warps * 32), smem, stream, \
                    A_ptr, B_ptr, C_ptr, suh_ptr, A_had_ptr, svh_ptr, pb, \
                    size_k, size_n, lds_core, fuse_out); \
                if (graph) exl3_gemv_graph_record_sites(graph, \
                    (void*) &kernel_name<bits_val, true, codebook, warps>); \
            } else { \
                hipLaunchKernelGGL( \
                    (kernel_name<bits_val, false, codebook, warps>), \
                    dim3(num_blocks), dim3(warps * 32), smem, stream, \
                    A_ptr, B_ptr, C_ptr, suh_ptr, A_had_ptr, svh_ptr, pb, \
                    size_k, size_n, lds_core, fuse_out); \
                if (graph) exl3_gemv_graph_record_sites(graph, \
                    (void*) &kernel_name<bits_val, false, codebook, warps>); \
            }

        #define LAUNCH_GGEMV_WAVES(bits_val, codebook, warps) \
            if (splitk) { \
                LAUNCH_GGEMV_WAVES_K(bits_val, codebook, warps, exl3_gemv_graph_dot_kernel_splitk) \
            } else { \
                LAUNCH_GGEMV_WAVES_K(bits_val, codebook, warps, exl3_gemv_graph_dot_kernel) \
            }

        #define LAUNCH_GGEMV_CB(bits_val, codebook) \
            switch (warps_per_block) { \
                case 4:  LAUNCH_GGEMV_WAVES(bits_val, codebook, 4);  break; \
                case 8:  LAUNCH_GGEMV_WAVES(bits_val, codebook, 8);  break; \
                case 16: LAUNCH_GGEMV_WAVES(bits_val, codebook, 16); break; \
            }

        #define LAUNCH_GGEMV(bits_val) \
            switch (cb) { \
                case 0: LAUNCH_GGEMV_CB(bits_val, 0); break; \
                case 1: LAUNCH_GGEMV_CB(bits_val, 1); break; \
                case 2: LAUNCH_GGEMV_CB(bits_val, 2); break; \
            }

        switch (K) {
            case 1: LAUNCH_GGEMV(1); break;
            case 2: LAUNCH_GGEMV(2); break;
            case 3: LAUNCH_GGEMV(3); break;
            case 4: LAUNCH_GGEMV(4); break;
            case 5: LAUNCH_GGEMV(5); break;
            case 6: LAUNCH_GGEMV(6); break;
            case 7: LAUNCH_GGEMV(7); break;
            case 8: LAUNCH_GGEMV(8); break;
        }

        #undef LAUNCH_GGEMV
        #undef LAUNCH_GGEMV_CB
        #undef LAUNCH_GGEMV_WAVES
        #undef LAUNCH_GGEMV_WAVES_K
    }

    if (fuse_out) return true;

    // 2. Output rotation -- fallback form
    {
        int blocks = CEIL_DIVIDE((size_n / 128) * 32, 256);
        if (c_fp32)
            hipLaunchKernelGGL
            (
                exl3_gemv_graph_had_out_kernel<true>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n
            );
        else
            hipLaunchKernelGGL
            (
                exl3_gemv_graph_had_out_kernel<false>,
                dim3(blocks), dim3(256), 0, stream,
                pb, size_n
            );
    }

    return true;
}
