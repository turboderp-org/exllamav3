#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include <tuple>
#include <mutex>
#include "exl3_dq.cuh"
#include "hadamard.cuh"

// Constants
#define NUM_THREADS 256
#define SMEM_MAX (90 * 1024)  // max shared memory on compute capability 8.6

// Max allowable output size, in tiles. Used to allocate global lock buffer per device for sync across threadblocks
#define MAX_TILES_C (1024 * 1024)

#include "exl3_gemm_kernel.cuh"

// Singleton to manage context for each device. Stores device attributes and a large-enough lock tensor per device

#define MAX_DEVICES 32
#define CC_OLD        1
#define CC_AMPERE     2
#define CC_ADA        3
#define CC_HOPPER     4
#define CC_BLACKWELL  5

class DevCtx
{
private:
    int num_sms[MAX_DEVICES] = {};
    int cc[MAX_DEVICES] = {};
    void* locks[MAX_DEVICES] = {};
    std::mutex mtx;

public:
    static DevCtx& instance()
    {
        static DevCtx ctx;
        return ctx;
    }

    int get_num_sms(int device)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!num_sms[device])
            cuda_check(cudaDeviceGetAttribute(&num_sms[device], cudaDevAttrMultiProcessorCount, device));
        return num_sms[device];
    }

    int get_cc(int device)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!cc[device])
        {
            cudaDeviceProp prop;
            cuda_check(cudaGetDeviceProperties(&prop, device));
            if (prop.major >= 10) cc[device] = CC_BLACKWELL;
            else if (prop.major >= 9) cc[device] = CC_HOPPER;
            else if (prop.major >= 8 && prop.minor >= 9) cc[device] = CC_ADA;
            else if (prop.major >= 8 && prop.minor >= 6) cc[device] = CC_AMPERE;
            else cc[device] = CC_OLD;
        }
        return cc[device];
    }

    int* get_locks(int device)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!locks[device])
        {
            cudaSetDevice(device);
            cudaMalloc(&locks[device], MAX_TILES_C * sizeof(int));
            cudaMemset(locks[device], 0, MAX_TILES_C * sizeof(int));
        }
        return (int*) locks[device];
    }

private:
    DevCtx() = default;
    DevCtx(const DevCtx&) = delete;
    DevCtx& operator=(const DevCtx&) = delete;
};

// Kernel wrapper for bitrates 1..8

template
<
    int TILESIZE_M,
    int TILESIZE_K,
    int TILESIZE_N,
    int SH_STAGES,
    int FRAG_STAGES
>
bool launch
(
    int K,
    int num_sms,
    const half* A_ptr,
    const uint16_t* B_ptr,
    void* C_ptr,
    bool c_fp32,
    int size_m,
    int size_k,
    int size_n,
    int* locks,
    const uint16_t* sv_ptr,
    cudaStream_t stream
)
{
    if (size_k % TILESIZE_K != 0) return false;
    if (size_n % TILESIZE_N != 0) return false;

    int max_slices = size_k / TILESIZE_K * size_n / TILESIZE_N / 12;  // decided experimentally, TODO: maybe test more
    num_sms = MIN(max_slices, num_sms);  // avoid empty blocks
    int tiles_m = CEIL_DIVIDE(size_m, TILESIZE_M);
    dim3 blocks(num_sms, tiles_m);

    bool launch_ok = false;
    static_for_pack<0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18>
    ([&](auto ic)
    {
        constexpr int i = decltype(ic)::value;
        constexpr int i_b = i & 0x0f;
        constexpr bool fp32 = i & 0x10;

        if (K == i_b && c_fp32 == fp32)
        {
            cudaFuncSetAttribute
            (
                exl3_gemm_kernel<i_b, fp32, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_MAX
            );
            exl3_gemm_kernel<i_b, fp32, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES>
            <<<blocks, NUM_THREADS * TILESIZE_K / 16, SMEM_MAX, stream>>>
            (
                A_ptr,
                B_ptr,
                C_ptr,
                size_m,
                size_k,
                size_n,
                locks,
                sv_ptr
            );
            cuda_check(cudaPeekAtLastError());
            launch_ok = true;
        }
    });
    return launch_ok;
};

int select_kernel(int cc, int size_m, int size_k, int size_n, const uint16_t* sv_ptr);

/*

EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*bits), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float23, contiguous. Does not need to be zero-initialized
- sv: optional, packed output sign flips, shape (n//16), dtype uint16

If temp_A == A and su is not None, input transform is done in-place. EXL3 tensors quantized with the same H (e.g.
Q, K, V projections in normal transformer) will have the same input sign flips.

limitations:
- k % 16 == 0
- n % 128 == 0

*/
int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& sv,
    int force_kernel_idx
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, 1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, 1, B, 1, 16);
    TORCH_CHECK_SHAPES(A, 0, C, 0, 1);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    // TODO: Input scale here to reduce Python overhead?

    // Get SV, optionally
    const uint16_t* sv_ptr = (const uint16_t*) OPTPTR(sv);
    if (sv_ptr) TORCH_CHECK(false, "sv_ptr is disabled");
        // TORCH_CHECK_SHAPES(sv.value(), 0, B, 1, 1);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    int bits = B.size(2) / 16;
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16;

    int selected_kernel;
    if (force_kernel_idx <= 0)
        selected_kernel = select_kernel(cc, size_m, size_k, size_n, sv_ptr);
    else
        selected_kernel = force_kernel_idx;

    if (!selected_kernel)
        TORCH_CHECK(false, "exl3_gemm: no compatible kernel");

    bool launched;
    #define ARGS bits, num_sms, A_ptr, B_ptr, C_ptr, c_fp32, size_m, size_k, size_n, locks, sv_ptr, stream
    switch (selected_kernel)
    {
        //                         tsz_m   tsz_k   tsz_n  sh_st  fr_st  fuse_h
        case 1: launched = launch<    16,     16,    128,     3,     1>(ARGS); break;
        case 2: launched = launch<    16,     32,    256,     4,     2>(ARGS); break;
        case 3: launched = launch<    16,     32,    128,     4,     2>(ARGS); break;
        case 4: launched = launch<    32,     32,    128,     4,     2>(ARGS); break;
        case 5: launched = launch<    64,     16,    128,     4,     2>(ARGS); break;
        case 6: launched = launch<    16,     16,    512,     4,     2>(ARGS); break;
         default:
            launched = false;
            break;
    }

    return launched ? selected_kernel : 0;
}

int exl3_gemm_num_kernel_variants()
{
    return 6;
}

// Select kernel based on tensor shape and device props

int select_kernel(int cc, int size_m, int size_k, int size_n, const uint16_t* sv_ptr)
{
    bool mod_256 = (size_n % 256 == 0);
    bool mod_512 = (size_n % 512 == 0);

    switch(cc)
    {
        case CC_OLD:
        case CC_AMPERE:
            if (size_m > 16) return 4;
            if (size_n < 2048) return 3;
            return mod_256 ? 2 : 3;

        case CC_ADA:
            if (size_m <= 16)
            {
                if (size_k * size_n >= 5e7) return mod_256 ? 2 : 3;
                return 3;
            }
            if (size_n * size_k < 8e6) return mod_256 ? 2 : 4;
            return 4;

        case CC_HOPPER:
        case CC_BLACKWELL:
            if (size_m <= 16)
            {
                if (size_n >= 65536 && mod_512) return 6;
                if (size_k * size_n >= 2e8 && mod_512) return 6;
                if (size_k * size_n >= 5e7) return mod_256 ? 2 : 3;
                return 3;
            }
            if (size_m > 32)
            {
                if (size_n * size_k < 8e6) return 4;
                return 5;
            }
            if (size_n * size_k < 8e6) return 3;
            return 4;
    }
    return 0;
}