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
// raw mul1 decode). One wave per (row, 128-block); with suh_tab one source per blockIdx.y, each with its own suh and
// its own xh / xcs slab
__global__ __launch_bounds__(256)
void exl3_rdna3_had_kernel
(
    const half* __restrict__ A,
    const half* __restrict__ suh,
    uint2* __restrict__ xh,
    float* __restrict__ xcs,
    int size_m,
    int size_k,
    const uint64_t* __restrict__ suh_tab,
    const half* __restrict__ A_up,
    Exl3Rdna3GNorm gn
)
{
    const int lane = threadIdx.x & 31;
    const int kblocks = size_k / 128;
    if (suh_tab)
    {
        suh = (const half*) suh_tab[blockIdx.y];
        xh += (size_t) blockIdx.y * size_m * (size_k / 16) * 4;
        xcs += (size_t) blockIdx.y * size_m * kblocks;
    }
    const int task = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (task >= size_m * kblocks) return;
    const int m = task / kblocks;
    const int c = task % kblocks;
    const size_t off = (size_t) m * size_k + c * 128 + lane * 4;

    half2 x01, x23;
    if (gn.flags & GN_ACTIVE)
    {
        // gated_rms_norm (small path, one warp per head): fma sum of squares, xor-reduce 16 .. 1, x * w * rmf,
        // then * act(g), rounded to fp16
        auto bf = [] (uint32_t v, int hi) { return __uint_as_float(hi ? (v & 0xffff0000u) : (v << 16)); };
        const uint2 xr = *(const uint2*) (((const uint16_t*) A) + off);
        float f[4] = { bf(xr.x, 0), bf(xr.x, 1), bf(xr.y, 0), bf(xr.y, 1) };
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) ss = fmaf(f[i], f[i], ss);
        #pragma unroll
        for (int i = 16; i > 0; i >>= 1) ss += __shfl_xor(ss, i);
        const float rmf = rsqrtf(ss / 128.0f + gn.eps);
        float w[4], g[4];
        if (gn.flags & GN_W_BF16)
        {
            const uint2 wr = *(const uint2*) (((const uint16_t*) gn.w) + lane * 4);
            w[0] = bf(wr.x, 0); w[1] = bf(wr.x, 1); w[2] = bf(wr.y, 0); w[3] = bf(wr.y, 1);
        }
        else
        {
            const float4 wr = *(const float4*) (((const float*) gn.w) + lane * 4);
            w[0] = wr.x; w[1] = wr.y; w[2] = wr.z; w[3] = wr.w;
        }
        if (gn.flags & GN_G_BF16)
        {
            const uint2 gr = *(const uint2*) (((const uint16_t*) gn.g) + off);
            g[0] = bf(gr.x, 0); g[1] = bf(gr.x, 1); g[2] = bf(gr.y, 0); g[3] = bf(gr.y, 1);
        }
        else
        {
            const float4 gr = *(const float4*) (((const float*) gn.g) + off);
            g[0] = gr.x; g[1] = gr.y; g[2] = gr.z; g[3] = gr.w;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const float wi = gn.bias != 0.0f ? w[i] + gn.bias : w[i];
            const float r = __fdividef(1.0f, 1.0f + __expf(-g[i]));
            f[i] = f[i] * wi * rmf * ((gn.flags & GN_SIGMOID) ? r : g[i] * r);
        }
        x01 = __floats2half2_rn(f[0], f[1]);
        x23 = __floats2half2_rn(f[2], f[3]);
    }
    else
    {
        const half2* ap = (const half2*) (A + off);
        x01 = ap[0];
        x23 = ap[1];
    }
    if (A_up)
    {
        const half2* up = (const half2*) (A_up + off);
        auto sigmoid2 = [] (half2 x) -> half2
        {
            const half2 one = __float2half2_rn(1.0f);
            return h2rcp(__hadd2(one, h2exp(__hneg2(x))));
        };
        if (gn.flags & GN_UP_SIGMOID)
        {
            x01 = __hmul2(x01, sigmoid2(up[0]));
            x23 = __hmul2(x23, sigmoid2(up[1]));
        }
        else
        {
            x01 = __hmul2(__hmul2(x01, sigmoid2(x01)), up[0]);
            x23 = __hmul2(__hmul2(x23, sigmoid2(x23)), up[1]);
        }
    }
    exl3_rdna3_had::transform_block(x01, x23, suh, xh, xcs, m, c, size_k, lane);
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
    Graph* graph,
    const half* A_up,
    const Exl3Rdna3GNorm* gn
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
        if (max_rows < 1 || ((graph || gn) && size_m > max_rows)) return false;   // the gate is not re-based per pass

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
                A + (size_t) r0 * size_k, suh, g_xh[device], g_xcs[device], m, size_k, nullptr,
                A_up ? A_up + (size_t) r0 * size_k : nullptr, gn ? *gn : Exl3Rdna3GNorm {}
            );
            kernel<<<dim3(groups * splits, row_chunks), EXL3_RDNA3_THREADS, 0, stream>>>
            (
                g_xh[device], B32, (char*) C + (size_t) r0 * size_n * c_elem, m, size_k, size_n,
                g_counters[device], g_xcs[device], g_ws[device], svh, splits, ks_per_split, Exl3Rdna3MTab {}
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

bool exl3_rdna3_mgemm
(
    const half* A,
    const uint64_t* b_tab,
    void* C,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    const uint64_t* suh_tab,
    const uint64_t* svh_tab,
    const uint64_t* c_tab,
    const int* n_stride_tab,
    const int* src_tab,
    int num_entries,
    int num_src,
    int device,
    cudaStream_t stream,
    Graph* graph
)
{
    #ifdef __HIP_PLATFORM_AMD__
        if (!exl3_rdna3_enabled()) return false;
        if (size_k % 128 || size_n % 128) return false;
        if (K < 1 || K > 8) return false;
        if (num_entries < 1 || num_src < 1 || size_m < 1) return false;

        int mr;
        fp_exl3_rdna3_kernel kernel = select_kernel(size_m, K, half_k, cb, c_fp32, mr);
        if (!kernel) return false;

        // Single pass only: every source's transformed input has to fit the workspace
        const int kblocks = size_k / 128;
        if ((size_t) num_src * size_m * size_k * 2 > EXL3_RDNA3_XH_BYTES) return false;
        if ((size_t) num_src * size_m * kblocks > EXL3_RDNA3_XCS_FLOATS) return false;

        const int groups = size_n / 128;
        const int row_chunks = (size_m + mr - 1) / mr;
        const int items = groups * row_chunks * num_entries;
        if (items > EXL3_RDNA3_MAX_COUNTERS) return false;

        if (!g_ws[device]) exl3_rdna3_prepare(device);

        int splits, ks_per_split;
        choose_splits(items, kblocks, (size_t) num_entries * size_m * size_n * sizeof(float), device, splits, ks_per_split);

        const int had_tasks = size_m * kblocks;
        exl3_rdna3_had_kernel<<<dim3((had_tasks + 7) / 8, num_src), 256, 0, stream>>>
        (
            A, nullptr, g_xh[device], g_xcs[device], size_m, size_k, suh_tab, nullptr, Exl3Rdna3GNorm {}
        );
        kernel<<<dim3(groups * splits, row_chunks, num_entries), EXL3_RDNA3_THREADS, 0, stream>>>
        (
            g_xh[device], nullptr, C, size_m, size_k, size_n, g_counters[device], g_xcs[device], g_ws[device], nullptr,
            splits, ks_per_split, Exl3Rdna3MTab { b_tab, svh_tab, c_tab, n_stride_tab, src_tab }
        );

        // Graph patching: the input (A) and output (C) only; the pointer tables are static
        if (graph)
        {
            graph->record_param((void*) exl3_rdna3_had_kernel, GP_mgemm_A, 0);
            graph->record_param((void*) exl3_rdna3_had_kernel, GP_end, 0);
            graph->record_param((void*) kernel, GP_mgemm_C, 2);
            graph->record_param((void*) kernel, GP_end, 0);
        }
        cuda_check(cudaPeekAtLastError());
        return true;
    #else
        return false;
    #endif
}
