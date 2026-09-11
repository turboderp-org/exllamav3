// Dense fp16 GEMM with fp16-accumulator tensor-core MMA and fp32 accumulation across K.
//
// GeForce parts run mma.sync with fp32 accumulators at half the rate of the fp16-accumulator
// form. This kernel runs the MMA with fp16 accumulators and flushes the partial sums into fp32
// registers every KSLICE elements of K, so the accumulation across K stays fp32 and the error
// is that of 32-term fp16 partials. Used by the reconstruct (prefill) paths through hgemm_recon /
// hgemm_batched, which fall back to cuBLAS when the device gains nothing (rate probe) or the
// shape is not covered or too small.
//
// C[b] = A[b] @ B[b], row-major, A [M, K], B [K, N], C [M, N] (fp16 or fp32 output), optional
// strided batch. Requires K % 64 == 0, N % 128 == 0, packed rows; M arbitrary.

#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <mutex>
#include <cstdlib>
#include <cstring>

namespace f16acc
{

constexpr int BM = 128, BN = 128, BK = 64, PAD = 8;
constexpr int WARPS_M = 2, WARPS_N = 4, THREADS = WARPS_M * WARPS_N * 32;
constexpr int STAGES = 2, KSLICE = 32, GROUP_M = 8;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
constexpr int MT = WM / 16, NT = WN / 8;
constexpr int AS_STRIDE = BK + PAD, BS_STRIDE = BN + PAD;
constexpr int A_STAGE = BM * AS_STRIDE, B_STAGE = BK * BS_STRIDE;
constexpr size_t SMEM_BYTES = (size_t) STAGES * (A_STAGE + B_STAGE) * sizeof(half);
constexpr int MIN_ROWS = 384;

static int g_num_sms[MAX_DEVICES];

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred)
{
    int src_size = pred ? 16 : 0;    // 0 -> zero-fill, no global read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_f16(uint32_t* c, const uint32_t* a, const uint32_t* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f32(float* c, const uint32_t* a, const uint32_t* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <bool OUT_F32>
__global__ void __launch_bounds__(THREADS, 1)
gemm_kernel
(
    const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ C,
    int M, int N, int K, int ldc,
    long long strideA, long long strideB, long long strideC
)
{
    extern __shared__ __align__(16) half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * A_STAGE;

    // Grouped raster: consecutive blocks walk GROUP_M M-stripes over one N column before moving
    // on, so the B stripe stays L2-hot across GROUP_M A-stripes
    const int grid_n = gridDim.x, grid_m = gridDim.y;
    const int bid = blockIdx.y * grid_n + blockIdx.x;
    const int group_size = GROUP_M * grid_n;
    const int group = bid / group_size;
    const int first_m = group * GROUP_M;
    const int gm_eff = min(GROUP_M, grid_m - first_m);
    const int in_group = bid - group * group_size;
    const int bm = (first_m + in_group % gm_eff) * BM;
    const int bn = (in_group / gm_eff) * BN;

    const int bz = blockIdx.z;
    A += (long long) bz * strideA;
    B += (long long) bz * strideB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;

    constexpr int A_CPR = BK / 8, B_CPR = BN / 8;
    constexpr int A_ITERS = BM * A_CPR / THREADS, B_ITERS = BK * B_CPR / THREADS;
    auto load_tile = [&](int stage, int kt)
    {
        const int k0 = kt * BK;
        half* as = As + stage * A_STAGE;
        half* bs = Bs + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / A_CPR, chunk = c % A_CPR;
            int grow = bm + row;
            bool pred = grow < M;
            const half* src = A + (long long) (pred ? grow : 0) * K + k0 + chunk * 8;
            cp_async16(as + row * AS_STRIDE + chunk * 8, src, pred);
        }
        #pragma unroll
        for (int i = 0; i < B_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / B_CPR, chunk = c % B_CPR;
            const half* src = B + (long long) (k0 + row) * N + bn + chunk * 8;
            cp_async16(bs + row * BS_STRIDE + chunk * 8, src, true);
        }
    };

    float acc[MT][NT][4];
    uint32_t hacc[MT][NT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
            hacc[i][j][0] = hacc[i][j][1] = 0u;
        }
    auto flush = [&]()
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float2 f0 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][0]));
                float2 f1 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][1]));
                acc[i][j][0] += f0.x; acc[i][j][1] += f0.y;
                acc[i][j][2] += f1.x; acc[i][j][3] += f1.y;
                hacc[i][j][0] = 0u; hacc[i][j][1] = 0u;
            }
    };

    const int KT = K / BK;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    const int a_lrow = lane & 15, a_lcol = (lane >> 4) * 8;
    const int b_lrow = (lane & 7) + ((lane >> 3) & 1) * 8, b_lcol = (lane >> 4) * 8;
    constexpr int KK = BK / 16;
    static_assert(KSLICE == 16 * 2, "flush cadence below assumes kslice 32 = two k16 steps");

    uint32_t af[2][MT][4];
    uint32_t bf[2][NT][2];
    auto load_frags = [&](int buf, const half* as, const half* bs, int kk)
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            ldmatrix_x4(af[buf][i], as + (i * 16 + a_lrow) * AS_STRIDE + kk * 16 + a_lcol);
        #pragma unroll
        for (int j = 0; j < NT; j += 2)
        {
            uint32_t r[4];
            ldmatrix_x4_trans(r, bs + (kk * 16 + b_lrow) * BS_STRIDE + j * 8 + b_lcol);
            bf[buf][j][0] = r[0]; bf[buf][j][1] = r[1]; bf[buf][j + 1][0] = r[2]; bf[buf][j + 1][1] = r[3];
        }
    };

    for (int kt = 0; kt < KT; ++kt)
    {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_tile(nk % STAGES, nk);
            cp_async_commit();
        }
        const half* as = As + (kt % STAGES) * A_STAGE + wm * WM * AS_STRIDE;
        const half* bs = Bs + (kt % STAGES) * B_STAGE + wn * WN;

        load_frags(0, as, bs, 0);
        #pragma unroll
        for (int kk = 0; kk < KK; ++kk)
        {
            if (kk + 1 < KK) load_frags((kk + 1) & 1, as, bs, kk + 1);
            const int cur = kk & 1;
            #pragma unroll
            for (int i = 0; i < MT; ++i)
                #pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_f16(hacc[i][j], af[cur][i], bf[cur][j]);
            if (kk & 1) flush();     // every 32 of K
        }
    }

    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            int col = bn + wn * WN + j * 8 + t * 2;
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                int row = bm + wm * WM + i * 16 + g + h * 8;
                if (row >= M) continue;
                float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
                if constexpr (OUT_F32)
                {
                    float* c = reinterpret_cast<float*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<float2*>(c) = make_float2(v0, v1);
                }
                else
                {
                    half* c = reinterpret_cast<half*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<half2*>(c) = __floats2half2_rn(v0, v1);
                }
            }
        }
}

// Rate probe: independent register-resident MMA chains, no memory traffic
template <bool F16>
__global__ void __launch_bounds__(256) rate_kernel(int iters, float* sink)
{
    uint32_t a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 5, b1 = a0 + 7;
    uint32_t a[4] = { a0, a1, a2, a3 }, b[2] = { b0, b1 };
    uint32_t h[8][2] = {};
    float f[8][4] = {};
    for (int i = 0; i < iters; ++i)
    {
        #pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            if constexpr (F16) mma_f16(h[c], a, b);
            else mma_f32(f[c], a, b);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) s += F16 ? __uint_as_float(h[c][0] ^ h[c][1]) : f[c][0] + f[c][3];
    if (s == 123.456f) sink[threadIdx.x] = s;
}

// Per-device decision: -1 unknown, 0 off, 1 on
static int g_enabled[MAX_DEVICES];
static bool g_init = false;
static std::mutex g_mutex;

static float probe_ms(bool f16, int blocks, int iters, float* sink, cudaStream_t stream)
{
    auto run = [&]() { if (f16) rate_kernel<true><<<blocks, 256, 0, stream>>>(iters, sink);
                       else rate_kernel<false><<<blocks, 256, 0, stream>>>(iters, sink); };
    run();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int r = 0; r < 3; ++r) run();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

bool enabled(int device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_init)
    {
        for (int i = 0; i < MAX_DEVICES; ++i) g_enabled[i] = -1;
        g_init = true;
    }
    if (device < 0 || device >= MAX_DEVICES) return false;
    if (g_enabled[device] >= 0) return g_enabled[device] == 1;

    int on = 0;
    const char* env = std::getenv("EXL3_HGEMM_F16ACC");
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (major < 8)
        on = 0;                                   // cp.async / ldmatrix.x4 / m16n8k16 need sm80+
    else if (env && std::strcmp(env, "0") == 0)
        on = 0;
    else if (env && std::strcmp(env, "1") == 0)
        on = 1;
    else
    {
        // Auto: enable where the fp16-accumulator MMA is at least 1.5x faster than the
        // fp32-accumulator one (GeForce: 2.0x; workstation / datacenter parts: 1.0x)
        const at::cuda::CUDAGuard guard(device);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        int sms = 1;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        float* sink = nullptr;
        cudaMalloc(&sink, 256 * sizeof(float));
        float t32 = probe_ms(false, sms * 4, 512, sink, stream);
        float t16 = probe_ms(true, sms * 4, 512, sink, stream);
        cudaFree(sink);
        cudaGetLastError();
        on = (t32 > 0.f && t16 > 0.f && t32 / t16 >= 1.5f) ? 1 : 0;
    }
    g_enabled[device] = on;
    return on == 1;
}

// Hard shape coverage of the kernel (independent of the device decision)
static bool covered(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c)
{
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kHalf && c.dtype() != at::kFloat) return false;
    if (a.dim() != b.dim() || a.dim() != c.dim() || (a.dim() != 2 && a.dim() != 3)) return false;
    int64_t M = a.size(-2), K = a.size(-1), N = b.size(-1);
    if (b.size(-2) != K || c.size(-2) != M || c.size(-1) != N) return false;
    if (K % BK != 0 || N % BN != 0 || M < 1) return false;
    if (a.stride(-1) != 1 || b.stride(-1) != 1 || c.stride(-1) != 1) return false;
    if (a.stride(-2) != K || b.stride(-2) != N) return false;
    if (c.stride(-2) % 2 != 0) return false;                    // half2 / float2 stores
    if (a.dim() == 3 && (a.size(0) != b.size(0) || a.size(0) != c.size(0))) return false;
    if (((uintptr_t) a.data_ptr() & 15) || ((uintptr_t) b.data_ptr() & 15) || ((uintptr_t) c.data_ptr() & 3)) return false;
    return true;
}

// Heuristic: shapes where the kernel beats cuBLAS (enough rows, at least one block per SM)
static bool worthwhile(const at::Tensor& a, const at::Tensor& b)
{
    int64_t M = a.size(-2), N = b.size(-1);
    if (M < MIN_ROWS) return false;
    int device = a.device().index();
    if (device < 0 || device >= MAX_DEVICES) return false;
    if (!g_num_sms[device])
        cudaDeviceGetAttribute(&g_num_sms[device], cudaDevAttrMultiProcessorCount, device);
    int64_t batch = a.dim() == 3 ? a.size(0) : 1;
    int64_t blocks = ((M + BM - 1) / BM) * (N / BN) * batch;
    return blocks >= g_num_sms[device];
}

template <bool OUT_F32>
static void launch(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, cudaStream_t stream)
{
    auto kern = gemm_kernel<OUT_F32>;
    // The dynamic-smem opt-in is a per-device function attribute: set it once per device
    // (a process-wide flag left the second device of a layer split launching without it:
    // "invalid argument" at the launch)
    static bool attr_set[MAX_DEVICES] = {};
    int device = a.device().index();
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "hgemm_f16acc: device index");
    if (!attr_set[device])
    {
        cuda_check(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) SMEM_BYTES));
        attr_set[device] = true;
    }
    bool batched = a.dim() == 3;
    int batch = batched ? a.size(0) : 1;
    int M = a.size(-2), K = a.size(-1), N = b.size(-1);
    dim3 grid(N / BN, (M + BM - 1) / BM, batch);
    kern<<<grid, THREADS, SMEM_BYTES, stream>>>(
        (const half*) a.data_ptr(), (const half*) b.data_ptr(), c.data_ptr(),
        M, N, K, (int) c.stride(-2),
        batched ? a.stride(0) : 0, batched ? b.stride(0) : 0, batched ? c.stride(0) : 0);
    cuda_check(cudaPeekAtLastError());
}

} // namespace f16acc

// Try the fp16-accumulator kernel; false = caller should use cuBLAS
bool hgemm_f16acc_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c)
{
    if (!f16acc::covered(a, b, c) || !f16acc::worthwhile(a, b)) return false;
    if (!f16acc::enabled(a.device().index())) return false;
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (c.dtype() == at::kFloat) f16acc::launch<true>(a, b, c, stream);
    else f16acc::launch<false>(a, b, c, stream);
    return true;
}

// Force the kernel (tests / benchmarks): errors if the shape is not covered
void hgemm_f16acc(at::Tensor a, at::Tensor b, at::Tensor c)
{
    TORCH_CHECK(f16acc::covered(a, b, c), "hgemm_f16acc: shape not covered (K % 64, N % 128, packed 16-byte aligned rows)");
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (c.dtype() == at::kFloat) f16acc::launch<true>(a, b, c, stream);
    else f16acc::launch<false>(a, b, c, stream);
}

// 1 = fp16-accumulate kernel active on this device, 0 = cuBLAS (probe result or env override)
int hgemm_f16acc_status(int device)
{
    return f16acc::enabled(device) ? 1 : 0;
}

// Reconstruct-path GEMM: the fp16-accumulator kernel where it pays, else cuBLAS
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c)
{
    if (hgemm_f16acc_try(a, b, c)) return;
    hgemm(a, b, c);
}
