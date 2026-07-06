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
#include <set>
#include <map>


// Mode 0: disabled; 1: int8 + error-feedback residual pass (~15-16 bit effective activation
// precision, KL at parity with fp16 or better); 2: plain int8 (cheaper, ~0.9% output RMS deviation).
static int _exl3_gemv_int8_mode = 0;
bool _exl3_gemv_int8_mode_chk = false;

static int exl3_gemv_int8_mode()
{
    if (_exl3_gemv_int8_mode_chk) return _exl3_gemv_int8_mode;
    const char* e = getenv("EXL3_INT8_GEMV");
    _exl3_gemv_int8_mode = e ? atoi(e) : 0;
    return _exl3_gemv_int8_mode;
}

bool exl3_gemv_int8_enabled()
{
    return exl3_gemv_int8_mode() != 0;
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

static void* select_gemv_int8_kernel(int K, bool c_fp32, bool residual)
{
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

static void* select_gemv_int8_sq_kernel(int K, int M, bool c_fp32, bool residual)
{
    switch (K)
    {
        case 1: return exl3_gemv_int8_sq_sel_k1(M, c_fp32, residual);
        case 2: return exl3_gemv_int8_sq_sel_k2(M, c_fp32, residual);
        case 3: return exl3_gemv_int8_sq_sel_k3(M, c_fp32, residual);
        case 4: return exl3_gemv_int8_sq_sel_k4(M, c_fp32, residual);
        case 5: return exl3_gemv_int8_sq_sel_k5(M, c_fp32, residual);
    }
    return nullptr;
}

// Grow-only per-device workspace shared by the sq and coop paths. The buffer is zeroed on
// (re)allocation so the sq completion counters (fixed region at the start, self-resetting
// thereafter) begin at zero; the coop kernels only use the region beyond SQ_WS_RESERVED.
static int* gemv_int8_get_ws(int device, size_t ws_ints)
{
    GemvInt8Workspace& ws = gemv_ws[device];
    if (ws.ws_ints < ws_ints)
    {
        if (ws.ws) cudaFree(ws.ws);
        cuda_check(cudaMalloc(&ws.ws, ws_ints * sizeof(int)));
        cuda_check(cudaMemset(ws.ws, 0, ws_ints * sizeof(int)));
        ws.ws_ints = ws_ints;
    }
    return ws.ws;
}

// m == 1 fast path: per-slice-scale kernel, regular launch. Returns false to fall through to the
// cooperative kernel (and from there to the regular fp16 kernel).
static bool exl3_gemv_int8_sq
(
    const half* A_ptr, const uint16_t* B_ptr, void* C_ptr,
    int size_m, int size_k, int size_n, int K, bool c_fp32, bool residual,
    const half* suh_ptr, half* A_had_ptr, const half* svh_ptr,
    int device, int num_sms, cudaStream_t stream, Graph* graph
)
{
    if (size_m > 2) return false;
    int M = size_m;
    void* fn = select_gemv_int8_sq_kernel(K, M, c_fp32, residual);
    if (!fn) return false;

    int rows_max = gemv_int8_sq_rows_max(M, residual);

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
        ksplit = CEIL_DIVIDE(rows_total, rows_per);
    };
    auto smem_for = [&] (int rows_per) -> size_t
    {
        size_t stage = gemv_int8_stage_smem(K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return (size_t) rows_per * 16 * 2 + (size_t) rows_per * 16 * 4 * M * (residual ? 2 : 1)
               + stage + (size_t) 2 * M * 128 * 4;
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_for(rows_max));
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
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    decomp(grid, ksplit, rows_per);
    size_t smem = smem_for(rows_per);
    if (ksplit > SQ_KSPLIT_CAP) return false;
    if (size_n / 256 > SQ_COUNTERS_CAP) return false;

    int pstride = size_n * (residual ? 2 : 1);
    int* ws_ptr = gemv_int8_get_ws(device, SQ_WS_RESERVED + (size_t) ksplit * M * pstride);

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

    cudaError_t err = cudaLaunchKernel(fn, dim3(grid), dim3(NUM_THREADS), kernelArgs, smem, stream);
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
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}

static void* select_mgemv_int8_kernel(int K, bool c_fp32, bool residual)
{
    switch (K)
    {
        case 1: return exl3_mgemv_int8_coop_sel_k1(c_fp32, residual);
        case 2: return exl3_mgemv_int8_coop_sel_k2(c_fp32, residual);
        case 3: return exl3_mgemv_int8_coop_sel_k3(c_fp32, residual);
        case 4: return exl3_mgemv_int8_coop_sel_k4(c_fp32, residual);
        case 5: return exl3_mgemv_int8_coop_sel_k5(c_fp32, residual);
        case 6: return exl3_mgemv_int8_coop_sel_k6(c_fp32, residual);
        case 7: return exl3_mgemv_int8_coop_sel_k7(c_fp32, residual);
        case 8: return exl3_mgemv_int8_coop_sel_k8(c_fp32, residual);
    }
    return nullptr;
}

bool exl3_gemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    cudaStream_t stream,
    Graph* graph
)
{
    if (!suh.has_value() || !A_had.has_value() || !svh.has_value()) return false;

    int K = B.size(2) / 16;
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    int size_m = A.numel() / size_k;
    if (size_n % 256) return false;
    if (size_k % 128) return false;
    // K >= 6 is DRAM-bound and the regular kernel reads the same bytes with less overhead; the int8
    // path only wins where the fp16 pipeline is compute/latency-limited (measured 3090: -29/-22/-9/-6%
    // at K=2/3/4/5, +5% at K=6)
    if (K < 1 || K > GEMV_INT8_MAX_K) return false;

    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    bool c_fp32 = C.dtype() == at::kFloat;
    bool residual = exl3_gemv_int8_mode() == 1;

    // Per-slice-scale kernel: m == 1, plus m == 2 in plain int8 mode (rows share the decoded
    // weights and the B stream; measured on 3090, larger m and batched residual lose to the fp16
    // tensor-core kernel). Falls through to the cooperative kernel on a constraint miss at m == 1;
    // batched rows beyond the gate go straight to the regular kernel.
    if (size_m <= (residual ? 1 : 2) && exl3_gemv_int8_sq(
        (const half*) A.data_ptr(), (const uint16_t*) B.data_ptr(), C.data_ptr(),
        size_m, size_k, size_n, K, c_fp32, residual,
        (const half*) suh->data_ptr(), (half*) A_had->data_ptr(), (const half*) svh->data_ptr(),
        device, num_sms, stream, graph))
        return true;
    if (size_m > 1) return false;

    void* fn = select_gemv_int8_kernel(K, c_fp32, residual);
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
        size_t stage = gemv_int8_stage_smem(K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return MAX((size_t) rows_per * 16 * 4 * (residual ? 2 : 1) + stage, (size_t) 8 * 128 * 4);
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        // Upper bound over all shapes: smem_rows_max * 64 B
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 768 * 16 * 4 + GEMV_STAGE_MAX_BYTES);
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
        gemv_occ_cache[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    size_t smem = smem_for_grid(grid);

    // Coop region beyond the sq-reserved prefix: [2n accs][4m qsums][grid partial maxes]
    size_t ws_ints = SQ_WS_RESERVED + (size_t) 2 * size_n + 4 * size_m + 1024;
    int* ws_ptr = gemv_int8_get_ws(device, ws_ints) + SQ_WS_RESERVED;

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
    add_graph_args((void*) fn);

    if (err != cudaSuccess)
    {
        // e.g. cooperative launch unsupported or co-residency violated: fall back to the regular kernel
        cudaGetLastError();
        return false;
    }
    return true;
}

bool exl3_mgemv_int8
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int min_index,
    int max_index,
    cudaStream_t stream,
    Graph* graph
)
{
    int size_m = A.size(1);
    int size_k = A.size(2);
    int size_n = C.size(2);
    int bszm_in = A.size(0);
    int bszm_out = C.size(0);
    int bszm = MAX(bszm_in, bszm_out);
    if (indices)
    {
        int num_indices = indices->size(1);
        if (bszm_in > num_indices) bszm_in = num_indices;
        if (bszm_out > num_indices) bszm_out = num_indices;
        bszm = MAX(bszm_in, bszm_out);
    }
    if (min_index >= 0 && !indices) return false;
    if (size_n % 256 || size_k % 128 || K < 1 || K > GEMV_INT8_MAX_K || bszm == 0) return false;

    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    bool c_fp32 = C.dtype() == at::kFloat;
    bool residual = exl3_gemv_int8_mode() == 1;
    void* fn = select_mgemv_int8_kernel(K, c_fp32, residual);
    if (!fn) return false;

    auto smem_for_grid = [&] (int grid_) -> size_t
    {
        int rows_total = size_k / 16;
        int nb256 = size_n / 256;
        int smem_rows_max = residual ? 384 : 768;
        int ksplit = CEIL_DIVIDE(4 * grid_, nb256);
        ksplit = MAX(ksplit, CEIL_DIVIDE(rows_total, smem_rows_max));
        ksplit = MIN(ksplit, rows_total);
        int rows_per = CEIL_DIVIDE(rows_total, ksplit);
        size_t stage = gemv_int8_stage_smem(K) ? (size_t) 8 * GEMV_STAGE_D * 16 * K * 4 : 0;
        return MAX((size_t) rows_per * 16 * 4 * (residual ? 2 : 1) + stage, (size_t) 8 * 128 * 4);
    };

    if (gemv_attr_set[device].find(fn) == gemv_attr_set[device].end())
    {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, 768 * 16 * 4 + GEMV_STAGE_MAX_BYTES);
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
        gemv_occ_cache[device][occ_key] = maxb;
    }
    int grid = MIN(MAX(maxb, 1) * num_sms, 1024);
    size_t smem = smem_for_grid(grid);

    size_t ws_ints = SQ_WS_RESERVED + (size_t) 2 * size_n + 4 * size_m + 1024;
    int* ws_ptr = gemv_int8_get_ws(device, ws_ints) + SQ_WS_RESERVED;

    const half* A_ptr = (const half*) A.data_ptr();
    const uintptr_t* B_ptr = (const uintptr_t*) B.data_ptr();
    void* C_ptr = C.data_ptr();
    const uintptr_t* suh_ptr = (const uintptr_t*) suh.data_ptr();
    half* A_had_ptr = (half*) A_had.data_ptr();
    const uintptr_t* svh_ptr = (const uintptr_t*) svh.data_ptr();
    const int64_t* indices_ptr = (const int64_t*) OPTPTR(indices);
    const half* weights_ptr = (const half*) OPTPTR(weights);

    void* kernelArgs[] =
    {
        &A_ptr, &B_ptr, &C_ptr, &size_m, &size_k, &size_n, &ws_ptr, &suh_ptr,
        &A_had_ptr, &svh_ptr, &indices_ptr, &weights_ptr, &bszm_in, &bszm_out,
        &min_index, &max_index
    };
    cudaError_t err = cudaLaunchCooperativeKernel(fn, grid, NUM_THREADS, kernelArgs, smem, stream);
    if (graph)
    {
        graph->record_param(fn, GP_mgemm_A, 0);
        graph->record_param(fn, GP_mgemm_C, 2);
        graph->record_param(fn, GP_mgemm_indices, 10);
        graph->record_param(fn, GP_mgemm_weights, 11);
        graph->record_param(fn, GP_end, 0);
    }
    if (err != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }
    return true;
}
