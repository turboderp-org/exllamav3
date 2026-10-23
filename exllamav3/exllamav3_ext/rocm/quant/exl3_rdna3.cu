#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <mutex>
#include <cstring>
#include "../../util.h"
#include "../../util.cuh"
#include "../../graph.cuh"
#include "../../quant/exl3_devctx.cuh"
#include "exl3_rdna3.cuh"
#include "exl3_rdna3_had.cuh"

namespace
{
    std::mutex g_mtx;
    float* g_ws[MAX_DEVICES] = {};
    int* g_counters[MAX_DEVICES] = {};
    uint2* g_xh[MAX_DEVICES] = {};
    float* g_xcs[MAX_DEVICES] = {};
    int g_enabled = -1;
}

bool exl3_rdna3_enabled()
{
    #ifdef __HIP_PLATFORM_AMD__
        if (g_enabled < 0)
        {
            const char* e = std::getenv("EXL3_RDNA3_GEMM");
            g_enabled = !(e && e[0] == '0');
            // the kernels use gfx11 instructions (v_dot2_f32_f16, v_sad_hi_u8) and are tuned for RDNA3's
            // cache hierarchy; every other architecture keeps the generic RDNA paths
            int device;
            cudaDeviceProp prop;
            if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
                strncmp(prop.gcnArchName, "gfx11", 5) != 0)
                g_enabled = 0;
        }
        return g_enabled;
    #else
        return false;
    #endif
}

// Workspace and counters are allocated once per device, outside of any graph capture (prepare_ctx runs at model
// load), and never move, so captured graphs keep valid pointers. Counters return to zero after every launch
void exl3_rdna3_prepare(int device)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return;
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_ws[device]) return;
        c10::cuda::CUDAGuard guard(device);
        cuda_check(cudaMalloc(&g_ws[device], EXL3_RDNA3_WS_BYTES));
        cuda_check(cudaMalloc(&g_counters[device], EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaMemset(g_counters[device], 0, EXL3_RDNA3_MAX_COUNTERS * sizeof(int)));
        cuda_check(cudaMalloc(&g_xh[device], EXL3_RDNA3_XH_BYTES));
        cuda_check(cudaMalloc(&g_xcs[device], EXL3_RDNA3_XCS_FLOATS * sizeof(float)));
        cuda_check(cudaDeviceSynchronize());
    #endif
}

#ifdef __HIP_PLATFORM_AMD__

// Input transform, once per matmul: x * suh -> 128-point Hadamard -> 1/sqrt(128) -> fp16 in the main kernel's LDS
// pair layout ([m][k/16][4] uint2), plus the per-128-block sums of the rounded values (codebook bias term of the
// raw mul1 decode). One wave per (row, 128-block)
__global__ __launch_bounds__(256)
void exl3_rdna3_had_kernel
(
    const half* __restrict__ A,
    const half* __restrict__ suh,
    uint2* __restrict__ xh,
    float* __restrict__ xcs,
    int size_m,
    int size_k
)
{
    const int lane = threadIdx.x & 31;
    const int kblocks = size_k / 128;
    const int task = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (task >= size_m * kblocks) return;
    const int m = task / kblocks;
    const int c = task % kblocks;
    const half2* ap = (const half2*) (A + (size_t) m * size_k + c * 128 + lane * 4);
    exl3_rdna3_had::transform_block(ap[0], ap[1], suh, xh, xcs, m, c, size_k, lane);
}

// Rows per pass: smallest instantiated MR >= m (MR 16 with multiple row chunks above that)
static fp_exl3_rdna3_kernel select_kernel(int size_m, int K, bool half_k, int cb, bool c_fp32, int& mr)
{
    int mr_idx;
    if      (size_m <= 1)  { mr_idx = 0; mr = 1; }
    else if (size_m <= 2)  { mr_idx = 1; mr = 2; }
    else if (size_m <= 3)  { mr_idx = 5; mr = 3; }
    else if (size_m <= 4)  { mr_idx = 2; mr = 4; }
    else if (size_m <= 5)  { mr_idx = 6; mr = 5; }
    else if (size_m <= 6)  { mr_idx = 7; mr = 6; }
    else if (size_m <= 8)  { mr_idx = 3; mr = 8; }
    else if (size_m <= 12) { mr_idx = 8; mr = 12; }
    else                   { mr_idx = 4; mr = 16; }

    switch (K)
    {
        case 1: return exl3_rdna3_get_k1(half_k, cb, mr_idx, c_fp32);
        case 2: return exl3_rdna3_get_k2(half_k, cb, mr_idx, c_fp32);
        case 3: return exl3_rdna3_get_k3(half_k, cb, mr_idx, c_fp32);
        case 4: return exl3_rdna3_get_k4(half_k, cb, mr_idx, c_fp32);
        case 5: return exl3_rdna3_get_k5(half_k, cb, mr_idx, c_fp32);
        case 6: return exl3_rdna3_get_k6(half_k, cb, mr_idx, c_fp32);
        case 7: return exl3_rdna3_get_k7(half_k, cb, mr_idx, c_fp32);
        case 8: return exl3_rdna3_get_k8(half_k, cb, mr_idx, c_fp32);
    }
    return nullptr;
}

// k-split so the grid (items independent block columns) fills one residency wave: the multiprocessor count is
// WGPs on gfx11 (48 on a 7900 XTX) and LDS allows 6 blocks per WGP; a partial second wave roughly doubles the
// tail. Splits are whole 128-element Hadamard blocks
static void choose_splits(int items, int kblocks, size_t ws_per_split, int device, int& splits, int& ks_per_split)
{
    const int tb = 6 * DevCtx::instance().get_num_sms(device);
    splits = (tb + items - 1) / items;
    splits = MAX(1, MIN(splits, kblocks));
    while (splits > 1 && (size_t) splits * ws_per_split > EXL3_RDNA3_WS_BYTES) splits--;
    if (splits > 1 && items > EXL3_RDNA3_MAX_COUNTERS) splits = 1;
    int kb_per_split = (kblocks + splits - 1) / splits;
    splits = (kblocks + kb_per_split - 1) / kb_per_split;
    ks_per_split = kb_per_split * 8;
}

#endif

bool exl3_rdna3_gemm
(
    const half* A,
    const uint16_t* B,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const half* suh,
    const half* svh,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        if (!suh || !svh) return false;
        if (size_k % 128 || size_n % 128) return false;
        if (K < 1 || K > 8) return false;

        int mr;
        fp_exl3_rdna3_kernel kernel = select_kernel(size_m, K, half_k, cb, c_fp32, mr);
        if (!kernel) return false;

        // Rows per launch pair, bounded by the transformed-input workspace
        const int kblocks = size_k / 128;
        const int max_rows = (int) MIN((size_t) EXL3_RDNA3_XH_BYTES / ((size_t) size_k * 2), (size_t) EXL3_RDNA3_XCS_FLOATS / kblocks);
        if (max_rows < 1 || (graph && size_m > max_rows)) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);

        const int groups = size_n / 128;
        const uint32_t* B32 = (const uint32_t*) B;
        const size_t c_elem = c_fp32 ? 4 : 2;

        for (int r0 = 0; r0 < size_m; r0 += max_rows)
        {
            const int m = MIN(max_rows, size_m - r0);
            const int row_chunks = (m + mr - 1) / mr;

            int splits, ks_per_split;
            choose_splits(groups * row_chunks, kblocks, (size_t) m * size_n * sizeof(float), device, splits, ks_per_split);

            const int had_tasks = m * kblocks;
            exl3_rdna3_had_kernel<<<(had_tasks + 7) / 8, 256, 0, stream>>>
            (
                A + (size_t) r0 * size_k, suh, g_xh[device], g_xcs[device], m, size_k
            );
            kernel<<<dim3(groups * splits, row_chunks), EXL3_RDNA3_THREADS, 0, stream>>>
            (
                g_xh[device], B32, (char*) C + (size_t) r0 * size_n * c_elem, m, size_k, size_n,
                g_counters[device], g_xcs[device], g_ws[device], svh, splits, ks_per_split
            );

            // Graph patching: callers bind the input (A) and output (C) pointers; the transformed input,
            // workspace and counters are static per device
            if (graph)
            {
                graph->record_param((void*) exl3_rdna3_had_kernel, GP_gemm_A, 0);
                graph->record_param((void*) exl3_rdna3_had_kernel, GP_end, 0);
                graph->record_param((void*) kernel, GP_gemm_C, 2);
                graph->record_param((void*) kernel, GP_end, 0);
            }
        }
        cuda_check(cudaPeekAtLastError());
        return true;
    #else
        return false;
    #endif
}
