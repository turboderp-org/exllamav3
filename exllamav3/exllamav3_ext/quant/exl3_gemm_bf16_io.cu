#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

#include "../util.h"
#include "../util.cuh"
#include "exl3_gemm_bf16_io.cuh"
#include "exl3_bf16_io_common.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_devctx.cuh"

static void check_cuda_tensor
(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name
)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on A.device()");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}
template<int bits, int cb, bool final_grid_sync>
__global__ __launch_bounds__(512)
void exl3_gemm_bf16_io_kernel
(
    const __nv_bfloat16* __restrict__ A,
    const uint16_t* __restrict__ B,
    half* __restrict__ C_scratch,
    __nv_bfloat16* __restrict__ C_bf16,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ suh,
    half* __restrict__ A_had,
    const half* __restrict__ svh
)
{
    auto grid = cg::this_grid();
    int total_warps = size_m * size_k / 128;
    int warps_grid = gridDim.x * blockDim.x / 32;
    int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
    for (; this_warp < total_warps; this_warp += warps_grid)
        had_bh_r_128_inner<true, false>
        (
            A + this_warp * 128,
            A_had + this_warp * 128,
            suh + (this_warp * 128) % size_k,
            0.088388347648f
        );

    grid.sync();
    exl3_gemm_kernel_inner
    <bits, false, cb, 16, 32, 128, 4, 3, true>
    (A_had, B, C_scratch, size_m, size_k, size_n, locks, svh, C_bf16);
    if constexpr (final_grid_sync)
        grid.sync();
}

template<int bits, int cb>
int launch_bf16_io
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    int force_num_sms,
    bool final_grid_sync
)
{
    constexpr int block_dim = 512;
    constexpr int smem_bytes = 90 * 1024;
    int size_m = A.numel() / A.size(-1);
    void* kernel = final_grid_sync
        ? reinterpret_cast<void*>(exl3_gemm_bf16_io_kernel<bits, cb, true>)
        : reinterpret_cast<void*>(exl3_gemm_bf16_io_kernel<bits, cb, false>);
    int device = A.get_device();
    int total_sms = DevCtx::instance().get_num_sms(device);
    int cooperative_capacity = exl3_bf16_cooperative_capacity(
        kernel, device, block_dim, smem_bytes);
    int num_sms = exl3_bf16_select_sms(
        force_num_sms, total_sms, cooperative_capacity, 1);
    int* locks_ptr = DevCtx::instance().get_locks(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const __nv_bfloat16* A_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    const uint16_t* B_ptr = reinterpret_cast<const uint16_t*>(B.data_ptr());
    half* C_scratch_ptr = reinterpret_cast<half*>(C_scratch.data_ptr());
    __nv_bfloat16* C_bf16_ptr = reinterpret_cast<__nv_bfloat16*>(C_bf16.data_ptr());
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;
    const half* suh_ptr = reinterpret_cast<const half*>(suh.data_ptr());
    half* A_had_ptr = reinterpret_cast<half*>(A_had.data_ptr());
    const half* svh_ptr = reinterpret_cast<const half*>(svh.data_ptr());
    void* args[] = {
        &A_ptr, &B_ptr, &C_scratch_ptr, &C_bf16_ptr,
        &size_m, &size_k, &size_n, &locks_ptr, &suh_ptr, &A_had_ptr, &svh_ptr,
    };
    cuda_check(cudaLaunchCooperativeKernel(
        kernel, dim3(num_sms), dim3(block_dim), args, smem_bytes, stream));
    cuda_check(cudaPeekAtLastError());
    return 2;
}

int exl3_gemm_bf16_io
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    int force_shape_idx,
    int force_num_sms,
    bool mcg,
    bool final_grid_sync
)
{
    check_cuda_tensor(A, A, "A");
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    check_cuda_tensor(B, A, "B");
    check_cuda_tensor(C_scratch, A, "C_scratch");
    check_cuda_tensor(C_bf16, A, "C_bf16");
    check_cuda_tensor(suh, A, "suh");
    check_cuda_tensor(A_had, A, "A_had");
    check_cuda_tensor(svh, A, "svh");
    TORCH_CHECK_DTYPE(A, kBFloat16);
    TORCH_CHECK_DTYPE(B, kShort);
    TORCH_CHECK_DTYPE(C_scratch, kHalf);
    TORCH_CHECK_DTYPE(C_bf16, kBFloat16);
    TORCH_CHECK_DTYPE(suh, kHalf);
    TORCH_CHECK_DTYPE(A_had, kHalf);
    TORCH_CHECK_DTYPE(svh, kHalf);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 3,
                "BF16 I/O requires 2D A and 3D encoded B");
    TORCH_CHECK(C_scratch.dim() == 2 && C_bf16.dim() == 2 && A_had.dim() == 2,
                "BF16 I/O scratch, output and Hadamard workspace must be 2D");
    TORCH_CHECK(suh.dim() == 1 && svh.dim() == 1,
                "BF16 I/O SUH and SVH must be one-dimensional");
    TORCH_CHECK(force_shape_idx == 2, "BF16 I/O supports shape 2 only");
    TORCH_CHECK(mcg, "BF16 I/O supports MCG only");
    TORCH_CHECK(A.size(0) > 0 && A.size(0) <= 16, "BF16 I/O supports 1 <= m <= 16");
    TORCH_CHECK(A.size(1) > 0 && A.size(1) % 128 == 0,
                "BF16 I/O requires positive, 128-aligned k");
    TORCH_CHECK(B.size(0) > 0 && B.size(1) > 0,
                "BF16 I/O requires positive encoded k and n dimensions");
    TORCH_CHECK(B.size(2) == 5 * 16 || B.size(2) == 6 * 16,
                "BF16 I/O requires an exact K5 or K6 encoded B width");
    TORCH_CHECK(A.size(1) <= INT_MAX && B.size(1) <= INT_MAX / 16,
                "BF16 I/O dimensions exceed the kernel integer range");
    TORCH_CHECK(A.size(-1) == B.size(0) * 16,
                "BF16 I/O A and encoded B have incompatible k dimensions");
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16;
    TORCH_CHECK(size_n % 128 == 0, "BF16 I/O requires 128-aligned n");
    TORCH_CHECK(size_n / 128 <= MAX_TILES_C,
                "BF16 I/O output exceeds the device lock workspace");
    TORCH_CHECK(A_had.sizes() == A.sizes(),
                "BF16 I/O Hadamard workspace must match A");
    TORCH_CHECK(C_scratch.sizes() == at::IntArrayRef({size_m, size_n}),
                "BF16 I/O FP16 scratch has an incompatible shape");
    TORCH_CHECK(C_bf16.sizes() == C_scratch.sizes(),
                "BF16 I/O output must match the FP16 scratch shape");
    TORCH_CHECK(suh.numel() == size_k && svh.numel() == size_n,
                "BF16 I/O SUH or SVH has an incompatible length");
    int bits = B.size(2) / 16;
    if (bits == 5) return launch_bf16_io<5, 1>(A, B, C_scratch, C_bf16, suh, A_had, svh, force_num_sms, final_grid_sync);
    if (bits == 6) return launch_bf16_io<6, 1>(A, B, C_scratch, C_bf16, suh, A_had, svh, force_num_sms, final_grid_sync);
    TORCH_CHECK(false, "BF16 I/O supports K5/K6 only");
}
