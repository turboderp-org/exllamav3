#include <cuda_fp16.h>
#include "exl3_gemv_int8.cuh"
#include "exl3_gemv_int8_kernel.cuh"
#include "comp_units/exl3_gemv_int8_instances.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"
#include "hadamard_inner.cuh"
#include <cooperative_groups.h>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <map>


// Mode 0: disabled; 1: int8 + error-feedback residual pass (~15-16 bit effective activation
// precision, KL at parity with fp16 or better); 2: plain int8 (cheaper, ~0.9% output RMS deviation).
static int _exl3_gemv_int8_mode = 0;
bool _exl3_gemv_int8_mode_chk = false;

static int exl3_gemv_int8_mode()
{
    if (_exl3_gemv_int8_mode_chk) return _exl3_gemv_int8_mode;
    const char* e = getenv("EXL3_INT8_GEMV");
    _exl3_gemv_int8_mode = e ? atoi(e) : 2;
    return _exl3_gemv_int8_mode;
}

bool exl3_gemv_int8_enabled()
{
    return exl3_gemv_int8_mode() != 0;
}

// Kill switch for the multi-matrix/sliced path only (EXL3_INT8_MSQ=0), for A/B verification
// against the cooperative mgemm kernel on identical inputs
bool exl3_gemv_int8_msq_enabled()
{
    static const int on = [] { const char* e = getenv("EXL3_INT8_MSQ"); return e ? atoi(e) : 1; }();
    return on != 0;
}

// Highest K the int8 path accepts; above it the regular kernel wins. The fp16 pipeline must be
// compute/latency-limited for the reduced per-weight work to matter, and where that ends is
// per-arch. Ampere is DRAM-bound from K = 6 up (3090: int8 -29/-22/-9/-6% at K=2/3/4/5, then
// -11..-26% on wide shapes at K=6); Ada is marginal at K=6 (4090: -0..+7%, residual mode loses)
// and keeps the conservative gate. Hopper's fp16 kernel is per-SM INT-throughput-bound at K = 6
// (H200, issue #242: +26/+57% per call, +16% e2e), and Blackwell measures the same way (5090:
// +7..+19% at K=6 across shapes, fp16 kernel at only ~65-78% of DRAM peak; K=7/8 are flat).
// RDNA3 (gfx1100) is like Hopper: the K=6 coop fp16 kernel is latency-bound (lm_head 9.18 ms/call
// vs a ~1.0 ms bandwidth floor), and raising the gate to 6 moved the 27B Qwen3.8-3.5bpw e2e decode
// from 22.83 to 31.02 tok/s median (+38.5%) with full golden-token parity (2026-09-17, 4096/256,
// single measured K=6 shape: lm_head k=5120/n=248320). gfx1101 is unmeasured; override with
// EXL3_INT8_GEMV_MAX_K to test there.
// EXL3_INT8_GEMV_MAX_K overrides the per-arch default for testing on unmeasured parts (kernel
// instances exist up to K = 8; at m == 1, K = 7..8 fall through to the cooperative kernel)
int exl3_gemv_int8_max_k(int device)
{
    static const int env_max_k = [] { const char* e = getenv("EXL3_INT8_GEMV_MAX_K"); return e ? atoi(e) : 0; }();
    if (env_max_k) return MIN(env_max_k, 8);
    int cc = DevCtx::instance().get_cc(device);
    return (cc == CC_HOPPER || cc == CC_BLACKWELL || cc == CC_RDNA3) ? 6 : 5;
}

struct GemvInt8Workspace
{
    int* ws = nullptr;
    size_t ws_ints = 0;
};

static GemvInt8Workspace gemv_ws[MAX_DEVICES];
static std::set<void*> gemv_attr_set[MAX_DEVICES];
static std::map<std::pair<void*, size_t>, int> gemv_occ_cache[MAX_DEVICES];

typedef void (*gemv_int8_coop_fn)
    (const half*, const uint16_t*, void*, int, int, int, int*, const half*, half*, const half*);

static void* select_gemv_int8_kernel(int K, bool half_k, bool c_fp32, bool residual)
{
    if (half_k)
    {
        switch (K)
        {
            case 1: return exl3_gemv_int8_coop_sel_h1(c_fp32, residual);
            case 2: return exl3_gemv_int8_coop_sel_h2(c_fp32, residual);
            case 3: return exl3_gemv_int8_coop_sel_h3(c_fp32, residual);
        }
        return nullptr;
    }
    switch (K)
    {
        case 1: return exl3_gemv_int8_coop_sel_k1(c_fp32, residual);
        case 2: return exl3_gemv_int8_coop_sel_k2(c_fp32, residual);
        case 3: return exl3_gemv_int8_coop_sel_k3(c_fp32, residual);
        case 4: return exl3_gemv_int8_coop_sel_k4(c_fp32, residual);
        case 5: return exl3_gemv_int8_coop_sel_k5(c_fp32, residual);
        case 6: return exl3_gemv_int8_coop_sel_k6(c_fp32, residual);
        case 7: return exl3_gemv_int8_coop_sel_k7(c_fp32, residual);
        case 8: return exl3_gemv_int8_coop_sel_k8(c_fp32, residual);
    }
    return nullptr;
}



static void* select_gemv_int8_sq_kernel(int K, bool half_k, int M, bool c_fp32, bool residual)
{
    if (half_k)
    {
        switch (K)
        {
            case 1: return exl3_gemv_int8_sq_sel_h1(M, c_fp32, residual);
            case 2: return exl3_gemv_int8_sq_sel_h2(M, c_fp32, residual);
            case 3: return exl3_gemv_int8_sq_sel_h3(M, c_fp32, residual);
        }
        return nullptr;
    }
    switch (K)
    {
        case 1: return exl3_gemv_int8_sq_sel_k1(M, c_fp32, residual);
        case 2: return exl3_gemv_int8_sq_sel_k2(M, c_fp32, residual);
        case 3: return exl3_gemv_int8_sq_sel_k3(M, c_fp32, residual);
        case 4: return exl3_gemv_int8_sq_sel_k4(M, c_fp32, residual);
        case 5: return exl3_gemv_int8_sq_sel_k5(M, c_fp32, residual);
        case 6: return exl3_gemv_int8_sq_sel_k6(M, c_fp32, residual);
    }
    return nullptr;
}

static void* select_gemv_int8_msq_kernel(int K, bool c_fp32, bool residual)
{
    switch (K)
    {
        case 1: return exl3_gemv_int8_msq_sel_k1(c_fp32, residual);
        case 2: return exl3_gemv_int8_msq_sel_k2(c_fp32, residual);
        case 3: return exl3_gemv_int8_msq_sel_k3(c_fp32, residual);
        case 4: return exl3_gemv_int8_msq_sel_k4(c_fp32, residual);
        case 5: return exl3_gemv_int8_msq_sel_k5(c_fp32, residual);
        case 6: return exl3_gemv_int8_msq_sel_k6(c_fp32, residual);
        case 7: return exl3_gemv_int8_msq_sel_k7(c_fp32, residual);
        case 8: return exl3_gemv_int8_msq_sel_k8(c_fp32, residual);
    }
    return nullptr;
}

// Stage-region bytes for the smem-staged unit: host mirror of the kernel's sh_b layout. ROCm routes
// every K to the narrow unit, so this is 0 there (see gemv_int8_stage_smem).
static size_t gemv_int8_stage_bytes_for(int K, bool half_k)
{
    if (!gemv_int8_stage_smem(K, half_k)) return 0;
    const int per_warp_row = half_k ? 8 * (2 * K + 1) : 16 * K;   // uint32 per pair row per warp
    return (size_t) 8 * GEMV_STAGE_D * per_warp_row * 4;
}

// Fixed-size per-device workspace shared by the sq and coop paths, allocated once and never
// reallocated: the pointer is baked as a kernel argument into captured CUDA graphs, so growing the
// buffer would leave every previously captured graph with a dangling workspace pointer (and let a
// reallocation clobber the self-resetting completion counters at the start of the buffer). Zeroed
// at allocation so the counters begin at zero. Callers must reject work that exceeds the fixed size
// (returns nullptr) and fall through to a non-workspace path.
#define GEMV_INT8_WS_INTS (WORKSPACE_SIZE / sizeof(int))    // 16 MB
static int* gemv_int8_get_ws(int device, size_t ws_ints)
{
    if (ws_ints > GEMV_INT8_WS_INTS) return nullptr;
    GemvInt8Workspace& ws = gemv_ws[device];
    if (!ws.ws)
    {
        cuda_check(cudaMalloc(&ws.ws, GEMV_INT8_WS_INTS * sizeof(int)));
        cuda_check(cudaMemset(ws.ws, 0, GEMV_INT8_WS_INTS * sizeof(int)));
        ws.ws_ints = GEMV_INT8_WS_INTS;
    }
    return ws.ws;
}

// EXL3_SQ_GRID_MULT scales the occupancy-derived sq/msq grid (cap 8).
// WGP: MULT=2 is 35.12 vs 35.67 (miss). CU-mode: sms still 48, MULT=1 is 33.88
// (underfill); MULT=2 is 38.36 vs xmask 35.67 (+7.6%), greedy-identical.
// Hadamard keep (maxb=4): MULT=3 is 38.86 vs 41.30 (−5.9%, grid=576).
// Default 2 only when compiled with -DEXL3_CUMODE.
static int exl3_sq_grid_mult()
{
    static const int v = []
    {
        const char* e = getenv("EXL3_SQ_GRID_MULT");
#if defined(EXL3_CUMODE)
        int n = e ? atoi(e) : 2;
#else
        int n = e ? atoi(e) : 1;
#endif
        return n < 1 ? 1 : (n > 8 ? 8 : n);
    }();
    return v;
}


// EXL3_SQ_ROWS_PER pins the K-slice height (multiple of 8). WGP default 32
// (32.69 vs 48 at 32.00 on this model, pre-CU). CU-mode + MULT=2 + xor-16:
// 40 is 40.79 vs 48 at 40.59 (+0.5%), greedy-identical. 56/64 miss. Hadamard
// had_xmask on rows=40 is 41.30 (maxb 3→4). 48 at maxb=4 is 40.89 (−1.0%).
// Default 40 only with -DEXL3_CUMODE.
static int exl3_sq_rows_per_arg()
{
    static const int v = []
    {
        const char* e = getenv("EXL3_SQ_ROWS_PER");
        int n = e ? atoi(e) : 0;
        if (n > 0) return MAX((n + 7) & ~7, SQ_MINROWS);
#if defined(USE_ROCM)
#if defined(EXL3_CUMODE)
        return 40;
#else
        return 32;
#endif
#else
        return 0;
#endif
    }();
    return v;
}

// EXL3_SQ_LAUNCH_LOG=N prints the first N sq/msq launches (grid/maxb/sms).
// CU-mode A/B: WGP miss on MULT=2 means 1 block/WGP; CU + MULT=1 would fill
// only 48 of 96 CUs if num_sms stays at the WGP count.
static void sq_launch_log(const char* tag, int grid, int maxb, int num_sms,
                          int ksplit, int rows_per, int size_k, int size_n)
{
    static int left = [] { const char* e = getenv("EXL3_SQ_LAUNCH_LOG"); return e ? atoi(e) : 0; }();
    if (left <= 0) return;
    fprintf(stderr, "[%s] grid=%d maxb=%d sms=%d ksplit=%d rows_per=%d k=%d n=%d\n",
            tag, grid, maxb, num_sms, ksplit, rows_per, size_k, size_n);
    --left;
}

// m == 1 fast path: per-slice-scale kernel, regular launch. Returns false to fall through to the
// cooperative kernel (and from there to the regular fp16 kernel).
static bool exl3_gemv_int8_sq
(
    const half* A_ptr, const uint16_t* B_ptr, void* C_ptr,
    int size_m, int size_k, int size_n, int K, bool half_k, bool c_fp32, bool residual,
    const half* suh_ptr, half* A_had_ptr, const half* svh_ptr,
    int device, int num_sms, cudaStream_t stream, Graph* graph
)
{
    if (size_m > 4) return false;
    int M = size_m > 2 ? 4 : size_m;
    void* fn = select_gemv_int8_sq_kernel(K, half_k, M, c_fp32, residual);
    if (!fn) return false;

    int rows_max = gemv_int8_sq_rows_max(M, residual);

    // EXL3_SQ_ROWS_PER pins the slice height (multiple of 8, >= SQ_MINROWS). On RDNA3 the
    // single-wave rule's rows_per = rows_max starves occupancy on wide matrices (lm_head):
    // swept on gfx1101, 64 beats auto by ~19% decode e2e (32/48/96/128/256 all slower).
    // gfx1100 re-sweep (2026-09-17, model tensors, cold rotation): 48 beats 64 by ~6% on the
    // K=3/4 sq kernels and +4.8% decode b1 / +4.2% b4 end-to-end on the 4.0bpw model, so 48 was
    // the ROCm default. Re-measured 2026-09-17 on Qwen3.8-27B-3.5bpw (4096/256, K6+narrow):
    // rows_per=32 = 32.69 tok/s vs 48 = 32.00 (+2.2%), 24 = 31.31 (worse); 32 was the WGP
    // default. CU-mode + MULT=2 + xor-16 (2026-09-23): 40 is 40.79 vs 48 at 40.59;
    // default 40 under -DEXL3_CUMODE (exl3_sq_rows_per_arg).
    int rows_per_arg = exl3_sq_rows_per_arg();


    // Mirror of the kernel's work decomposition (single-wave rule with a half-wave floor)
    auto decomp = [&] (int grid_, int& ksplit, int& rows_per)
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int r = CEIL_DIVIDE(rows_total * nb256, grid_);
        rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
        rows_per = MAX(rows_per, SQ_MINROWS);
        rows_per = MIN(rows_per, rows_max);
        rows_per = MIN(rows_per, (rows_total + 7) & ~7);
        if (rows_per_arg > 0)
            rows_per = MIN(rows_per_arg, MIN(rows_max, (rows_total + 7) & ~7));
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
    };
    auto smem_for = [&] (int rows_per) -> size_t
    {
        size_t stage = gemv_int8_stage_bytes_for(K, half_k);
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * M * (residual ? 2 : 1)
               + stage + (size_t) 2 * M * 128 * 4;
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max));
#if !defined(USE_ROCM)
        // Match the tensor-core kernels' shared-memory carveout: these kernels interleave with
        // them (hundreds of launches per decoded token), and a smaller carveout would make the GPU
        // drain and reconfigure the SMs on every transition - measured at ~4 us per launch in
        // graph replay. No configurable LDS carveout exists on RDNA; the attribute returns
        // hipErrorInvalidValue there.
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
#endif
        gemv_attr_set[device].insert(fn);
        cuda_check(cudaPeekAtLastError());
    }

    int ksplit, rows_per;
    decomp(6 * num_sms, ksplit, rows_per);
    size_t smem_guess = smem_for(rows_per);
    int maxb;
    auto occ_key = std::make_pair(fn, smem_guess);
    auto occ_it = gemv_occ_cache[device].find(occ_key);
    if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
    else
    {
        maxb = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess);
        gemv_occ_cache[device][occ_key] = maxb;
    }
    // Cap raised 1024→2048; EXL3_SQ_GRID_MULT scales occupancy-derived grid (see exl3_sq_grid_mult).
    int grid = MIN(MAX(maxb, 1) * num_sms * exl3_sq_grid_mult(), 2048);
    decomp(grid, ksplit, rows_per);
    size_t smem = smem_for(rows_per);
    sq_launch_log("sq-launch", grid, maxb, num_sms, ksplit, rows_per, size_k, size_n);
    if (ksplit > SQ_KSPLIT_CAP) return false;
    if (size_n / 256 > SQ_COUNTERS_CAP) return false;

    int pstride = size_n * (residual ? 2 : 1);
    int* ws_ptr = gemv_int8_get_ws(device, SQ_WS_RESERVED + (size_t) ksplit * M * pstride);
    if (!ws_ptr) return false;

    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr,
        (void*) &rows_per_arg
    };

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), kernelArgs, smem, stream);
    if (err != cudaSuccess)
    {
        // Nothing was captured: the caller's fallback kernel records its own parameter sites
        cudaGetLastError();
        return false;
    }
    if (graph)
    {
        graph->record_param(fn, GP_gemm_A, 0);
        graph->record_param(fn, GP_gemm_B_trellis, 1);
        graph->record_param(fn, GP_gemm_C, 2);
        graph->record_param(fn, GP_gemm_B_suh, 7);
        graph->record_param(fn, GP_gemm_A_had, 8);
        graph->record_param(fn, GP_gemm_B_svh, 9);
        graph->record_param(fn, GP_end, 0);
    }
    return true;
}

// m == 1 multi-matrix/sliced fast path: per-slice-scale kernel covering a whole mgemm call in one
// regular launch (see exl3_gemv_int8_msq_kernel). Takes the mgemm entry's cooked pointer arguments;
// the kernel signature matches exl3_mgemm_kernel so graph parameter recording is identical.
// Returns false to fall through to the cooperative mgemm kernel. Unlike the single-matrix gate,
// every K is accepted: the alternative here is the cooperative mgemm kernel, which loses to this
// path even at K = 7-8.
bool exl3_gemv_int8_msq
(
    const half* A_ptr,
    const uintptr_t* B_ptr_ptr,
    void* C_ptr,
    int size_m,
    int size_k,
    int size_n,                     // max slice/matrix width
    const uintptr_t* suh_ptr_ptr,
    half* A_had_ptr,
    const uintptr_t* svh_ptr_ptr,
    const int64_t* indices_ptr,
    const half* weights_ptr,
    int bszm_in,
    int bszm_out,
    int min_index,
    int max_index,
    int num_tokens,
    const int* size_n_list_ptr,
    void** c_list_ptr,
    const int* n_stride_list_ptr,
    const int* had_src_list_ptr,
    int num_had_src,
    int K,
    bool c_fp32,
    int device,
    int num_sms,
    cudaStream_t stream,
    Graph* graph
)
{
    if (size_m < 1 || bszm_out < 1) return false;
    bool residual = exl3_gemv_int8_mode() == 1;
    void* fn = select_gemv_int8_msq_kernel(K, c_fp32, residual);
    if (!fn) return false;

    int rows_total = size_k / 16;
    int nb256_max = CEIL_DIVIDE(size_n, 256);
    int rows_max = gemv_int8_sq_rows_max(1, residual);

    // Mirror of the kernel's work decomposition (sq's single-wave rule over the max width);
    // EXL3_SQ_ROWS_PER pins the slice height (ROCm default 32 — see sq path comment above)
    int rows_per_arg = exl3_sq_rows_per_arg();
    auto decomp = [&] (int grid_, int& ksplit, int& rows_per)
    {
        int r = CEIL_DIVIDE(rows_total * nb256_max, grid_);
        rows_per = (MAX(r, MIN(2 * r, 32)) + 7) & ~7;
        rows_per = MAX(rows_per, SQ_MINROWS);
        rows_per = MIN(rows_per, rows_max);
        rows_per = MIN(rows_per, (rows_total + 7) & ~7);
        if (rows_per_arg > 0)
            rows_per = MIN(rows_per_arg, MIN(rows_max, (rows_total + 7) & ~7));
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
    };
    auto smem_for = [&] (int rows_per) -> size_t
    {
        size_t stage = gemv_int8_stage_smem(K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * (residual ? 2 : 1)
               + stage + (size_t) 2 * 128 * 4;
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max));
#if !defined(USE_ROCM)
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
#endif
        gemv_attr_set[device].insert(fn);
        cuda_check(cudaPeekAtLastError());
    }

    int ksplit, rows_per;
    decomp(6 * num_sms, ksplit, rows_per);
    size_t smem_guess = smem_for(rows_per);
    int maxb;
    auto occ_key = std::make_pair(fn, smem_guess);
    auto occ_it = gemv_occ_cache[device].find(occ_key);
    if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
    else
    {
        maxb = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess);
        gemv_occ_cache[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms * exl3_sq_grid_mult(), 2048);
    decomp(grid, ksplit, rows_per);
    size_t smem = smem_for(rows_per);
    sq_launch_log("msq-launch", grid, maxb, num_sms, ksplit, rows_per, size_k, size_n);

    // Workspace: counters share the sq prefix [0..SQ_COUNTERS_CAP); qsums/partials live beyond
    // SQ_WS_RESERVED like the coop kernels. If it doesn't fit, grow rows_per (fewer slices ->
    // less workspace) up to rows_max, then pin the kernel to the same slice height.
    int num_jj = bszm_out * size_m;
    if (num_jj * nb256_max > SQ_COUNTERS_CAP) return false;
    int pstride = nb256_max * 256 * (residual ? 2 : 1);
    int* ws_ptr = nullptr;
    while (true)
    {
        size_t ws_ints = SQ_WS_RESERVED
                       + (size_t) num_jj * ksplit * 4
                       + (size_t) num_jj * ksplit * pstride;
        ws_ptr = gemv_int8_get_ws(device, ws_ints);
        if (ws_ptr) break;
        // Terminate on saturation, not on rows_max: rows_per caps at (rows_total + 7) & ~7
        // which can sit below rows_max forever (lm_head at m=32: n=248320 pushes the partials
        // region past the 16 MB workspace at any slice height) - the old rows_max test spun
        // the host here, hanging the generator on the first mid-length lm_head call
        int next_rows_per = MIN(rows_per * 2, MIN(rows_max, (rows_total + 7) & ~7));
        if (next_rows_per <= rows_per) return false;
        rows_per = next_rows_per;
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
        smem = smem_for(rows_per);
    }
    rows_per_arg = rows_per;    // pin the kernel to the (possibly grown) slice height
    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr_ptr,
        (void*) &indices_ptr,
        (void*) &weights_ptr,
        (void*) &bszm_in,
        (void*) &bszm_out,
        (void*) &min_index,
        (void*) &max_index,
        (void*) &num_tokens,
        (void*) &size_n_list_ptr,
        (void*) &c_list_ptr,
        (void*) &n_stride_list_ptr,
        (void*) &had_src_list_ptr,
        (void*) &num_had_src,
        (void*) &rows_per_arg
    };

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), kernelArgs, smem, stream);
    if (err != cudaSuccess)
    {
        // Nothing was captured: the caller's fallback kernel records its own parameter sites
        fprintf(stderr, "[msq-launch] declined: grid=%d smem=%zu ksplit=%d rows_per=%d err=%s\n",
                grid, smem, ksplit, rows_per, cudaGetErrorString(err));
        cudaGetLastError();
        return false;
    }
    if (graph)
    {
        graph->record_param(fn, GP_mgemm_A, 0);
        graph->record_param(fn, GP_mgemm_C, 2);
        graph->record_param(fn, GP_mgemm_indices, 10);
        graph->record_param(fn, GP_mgemm_weights, 11);
        graph->record_param(fn, GP_end, 0);
    }
    return true;
}

// Exl3_gemm's force_num_sms override reaches the sq/int8 kernels through here so a sweep can scale
// the launch geometry (grid multiplier) and the slice height without a rebuild. 0 = hardware default.
bool exl3_gemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int num_sms_arg,
    cudaStream_t stream,
    Graph* graph
)
{
    if (!suh.has_value() || !A_had.has_value() || !svh.has_value()) return false;

    // Tile width: 16 * K uint16 per 256-weight tile, 16 * K + 8 at the half-integer rates
    // (K + 0.5, mul1 codebook only; the caller gates the codebook)
    const int tile_u16 = (int) B.size(2);
    const int K = tile_u16 / 16;
    const bool half_k = (tile_u16 % 16) != 0;
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int size_m = A.numel() / size_k;
    if (size_n % 256) return false;
    if (size_k % 128) return false;

    int device;
    cudaGetDevice(&device);
    if (K < 1 || K > exl3_gemv_int8_max_k(device)) return false;
    int num_sms = num_sms_arg > 0 ? num_sms_arg : DevCtx::instance().get_num_sms(device);
    bool c_fp32 = C.dtype() == at::kFloat;
    bool residual = exl3_gemv_int8_mode() == 1;

    // Per-slice-scale kernel: m <= 4 in plain int8 mode (rows share the decoded weights and the B
    // stream). Falls through to the cooperative kernel on a constraint miss; batched rows beyond
    // the gate go straight to the regular kernel.
    if (size_m <= (residual ? 1 : 4) && exl3_gemv_int8_sq(
        (const half*) A.data_ptr(), (const uint16_t*) B.data_ptr(), C.data_ptr(),
        size_m, size_k, size_n, K, half_k, c_fp32, residual,
        (const half*) suh->data_ptr(), (half*) A_had->data_ptr(), (const half*) svh->data_ptr(),
        device, num_sms, stream, graph))
        return true;
    if (size_m > 1) return false;

    void* fn = select_gemv_int8_kernel(K, half_k, c_fp32, residual);
    if (!fn) return false;

    // Mirror the kernel's work decomposition for the shared memory size; grid = max co-resident
    // blocks (natural register allocation measures faster than forcing higher occupancy)
    auto smem_for_grid = [&] (int grid_) -> size_t
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int smem_rows_max = residual ? 384 : 768;
        int ksplit = CEIL_DIVIDE(4 * grid_, nb256);
        ksplit = MAX(ksplit, CEIL_DIVIDE(rows_total, smem_rows_max));
        ksplit = MIN(ksplit, rows_total);
        int rows_per = CEIL_DIVIDE(rows_total, ksplit);
        size_t stage = gemv_int8_stage_bytes_for(K, half_k);
        return MAX((size_t) rows_per * 16 * 4 * (residual ? 2 : 1) + stage, (size_t) 8 * 128 * 4);
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        // Upper bound over all shapes: smem_rows_max * 64 B
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 768 * 16 * 4 + GEMV_STAGE_MAX_BYTES);
#if !defined(USE_ROCM)
        // Match the tensor-core kernels' shared-memory carveout: these kernels interleave with
        // them (hundreds of launches per decoded token), and a smaller carveout would make the GPU
        // drain and reconfigure the SMs on every transition - measured at ~4 us per launch in
        // graph replay. No configurable LDS carveout exists on RDNA; the attribute returns
        // hipErrorInvalidValue there.
        cudaFuncSetAttribute((const void*) fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
#endif
        gemv_attr_set[device].insert(fn);
        cuda_check(cudaPeekAtLastError());
    }

    size_t smem_guess = smem_for_grid(6 * num_sms);
    int maxb;
    auto occ_key = std::make_pair(fn, smem_guess);
    auto occ_it = gemv_occ_cache[device].find(occ_key);
    if (occ_it != gemv_occ_cache[device].end()) maxb = occ_it->second;
    else
    {
        maxb = 1;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, NUM_THREADS, smem_guess);
#if defined(USE_ROCM)
        // See exl3_gemv.cu: occupancy overestimates co-residency on RDNA; one fewer per SM.
        if (maxb > 1) maxb -= 1;
#endif
        gemv_occ_cache[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    size_t smem = smem_for_grid(grid);

    // Coop region beyond the sq-reserved prefix: [2n accs][4m qsums][grid partial maxes]
    size_t ws_ints = SQ_WS_RESERVED + (size_t) 2 * size_n + 4 * size_m + 1024;
    int* ws_base = gemv_int8_get_ws(device, ws_ints);
    if (!ws_base) return false;
    int* ws_ptr = ws_base + SQ_WS_RESERVED;

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = C.data_ptr();
    const half* suh_ptr = (const half*) suh->data_ptr();
    half* A_had_ptr = (half*) A_had->data_ptr();   // scratch; used through a raw half* like the regular kernel
    const half* svh_ptr = (const half*) svh->data_ptr();

    void* kernelArgs[] =
    {
        (void*) &A_ptr,
        (void*) &B_ptr,
        (void*) &C_ptr,
        (void*) &size_m,
        (void*) &size_k,
        (void*) &size_n,
        (void*) &ws_ptr,
        (void*) &suh_ptr,
        (void*) &A_had_ptr,
        (void*) &svh_ptr
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
            graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
            graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
            graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    cudaError_t err = cudaLaunchCooperativeKernel(fn, grid, NUM_THREADS, kernelArgs, smem, stream);
    if (err != cudaSuccess)
    {
        // e.g. cooperative launch unsupported or co-residency violated: fall back to the regular kernel
        // (which records its own graph parameter sites)
        cudaGetLastError();
        return false;
    }
    add_graph_args((void*) fn);
    return true;
}
