#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

namespace cg = cooperative_groups;

#include "../util.h"
#include "../util.cuh"
#include "exl3_mgemm_bf16_io.cuh"
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

__global__ __launch_bounds__(32)
void exl3_bf16_grouped_had_kernel
(
    const __nv_bfloat16* __restrict__ A,
    half* __restrict__ A_had,
    const half** __restrict__ suh_list,
    const int size_m,
    const int size_k
)
{
    int group = blockIdx.z;
    int row = blockIdx.x;
    int col = blockIdx.y * 128;
    had_bh_r_128_inner<true, false>
    (
        A + row * size_k + col,
        A_had + ((size_t) group * size_m + row) * size_k + col,
        suh_list[group],
        0.088388347648f
    );
}

template<int bits, int cb, bool direct_output, bool final_group_barrier>
__global__ __launch_bounds__(512)
void exl3_mgemm_bf16_io_kernel
(
    const __nv_bfloat16* __restrict__ A,
    const uint16_t** __restrict__ B_list,
    half* __restrict__ C_scratch,
    __nv_bfloat16* __restrict__ C_bf16,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half** __restrict__ suh_list,
    half* __restrict__ A_had,
    const half** __restrict__ svh_list,
    const int count,
    const int output_stride
)
{
    auto grid = cg::this_grid();
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    for (int j = blockIdx.z; j < count; j += gridDim.z)
    {
        const uint16_t* B = B_list[j];
        const half* suh = suh_list[j];
        half* A_had_j = A_had + j * size_m * size_k;

        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for (; this_warp < total_warps; this_warp += warps_grid)
            had_bh_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had_j + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f
            );

        group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);

        half* C_j = C_scratch + j * size_m * size_n;
        int lock_offs = blockIdx.z * size_n / 128;
        const half* svh = svh_list[j];
        __nv_bfloat16* C_bf16_j = C_bf16 + j * size_n;
        if constexpr (direct_output)
        {
            exl3_gemm_kernel_inner
            <bits, false, cb, 16, 32, 128, 4, 3, true, true>
            (A_had_j, B, C_j, size_m, size_k, size_n,
             locks + lock_offs, svh, C_bf16_j, output_stride);
            if constexpr (final_group_barrier)
                group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
        }
        else
        {
            exl3_gemm_kernel_inner
            <bits, false, cb, 16, 32, 128, 4, 3, false>
            (A_had_j, B, C_j, size_m, size_k, size_n,
             locks + lock_offs, nullptr);

            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);

            total_warps = size_m * size_n / 128;
            this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
            for (; this_warp < total_warps; this_warp += warps_grid)
            {
                int blocks_per_row = size_n / 128;
                int row = this_warp / blocks_per_row;
                int col = (this_warp % blocks_per_row) * 128;
                had_hb_r_128_inner<false, true>
                (
                    C_j + this_warp * 128,
                    C_bf16_j + row * output_stride + col,
                    svh + col,
                    0.088388347648f
                );
            }

            if constexpr (final_group_barrier)
                group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
        }
    }
}

template<int bits, int cb, bool direct_output, bool final_group_barrier>
__global__ __launch_bounds__(512)
void exl3_mgemm_bf16_io_grouped_had_kernel
(
    const uint16_t** __restrict__ B_list,
    half* __restrict__ C_scratch,
    __nv_bfloat16* __restrict__ C_bf16,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ A_had,
    const half** __restrict__ svh_list,
    const Exl3Bf16HadGroupIds had_group_ids,
    const int count,
    const int group_count,
    const int output_stride
)
{
    int j = blockIdx.z;
    auto grid = cg::this_grid();
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;
    const uint16_t* B = B_list[j];
    int had_group_id = had_group_ids.values[j];
    const half* A_had_j = A_had + (size_t) had_group_id * size_m * size_k;
    half* C_j = C_scratch + (size_t) j * size_m * size_n;
    int lock_offs = blockIdx.z * size_n / 128;
    const half* svh = svh_list[j];
    __nv_bfloat16* C_bf16_j = C_bf16 + j * size_n;
    if constexpr (direct_output)
    {
        exl3_gemm_kernel_inner
        <bits, false, cb, 16, 32, 128, 4, 3, true, true>
        (A_had_j, B, C_j, size_m, size_k, size_n,
         locks + lock_offs, svh, C_bf16_j, output_stride);
        if constexpr (final_group_barrier)
            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
    }
    else
    {
        exl3_gemm_kernel_inner
        <bits, false, cb, 16, 32, 128, 4, 3, false>
        (A_had_j, B, C_j, size_m, size_k, size_n,
         locks + lock_offs, nullptr);

        group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);

        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for (; this_warp < total_warps; this_warp += warps_grid)
        {
            int blocks_per_row = size_n / 128;
            int row = this_warp / blocks_per_row;
            int col = (this_warp % blocks_per_row) * 128;
            had_hb_r_128_inner<false, true>
            (
                C_j + this_warp * 128,
                C_bf16_j + row * output_stride + col,
                svh + col,
                0.088388347648f
            );
        }

        if constexpr (final_group_barrier)
            group_barrier(blockIdx.z, gridDim.x, barrier_counters_sense);
    }
}

template<int bits, int cb>
int launch_mgemm_bf16_io
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    int force_num_sms,
    int output_stride,
    bool direct_output,
    bool final_group_barrier
)
{
    constexpr int block_dim = 512;
    constexpr int smem_bytes = 90 * 1024;
    void* kernel = direct_output
        ? final_group_barrier
            ? reinterpret_cast<void*>(exl3_mgemm_bf16_io_kernel<bits, cb, true, true>)
            : reinterpret_cast<void*>(exl3_mgemm_bf16_io_kernel<bits, cb, true, false>)
        : final_group_barrier
            ? reinterpret_cast<void*>(exl3_mgemm_bf16_io_kernel<bits, cb, false, true>)
            : reinterpret_cast<void*>(exl3_mgemm_bf16_io_kernel<bits, cb, false, false>);
    int device = A.get_device();
    int total_sms = DevCtx::instance().get_num_sms(device);
    int count = B.numel();
    int cooperative_capacity = exl3_bf16_cooperative_capacity(
        kernel, device, block_dim, smem_bytes);
    int resident_groups = force_num_sms > 0 && final_group_barrier ? 1 : count;
    int num_sms = exl3_bf16_select_sms(
        force_num_sms, total_sms, cooperative_capacity, resident_groups);
    int concurrency = std::min(cooperative_capacity / num_sms, count);
    TORCH_CHECK(concurrency > 0, "invalid BF16 MGEMM concurrency");
    TORCH_CHECK(final_group_barrier || concurrency == count,
                "omitting the terminal BF16 MGEMM barrier requires all matrices resident");
    int* locks_ptr = DevCtx::instance().get_locks(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const __nv_bfloat16* A_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    const uint16_t** B_ptrs = reinterpret_cast<const uint16_t**>(B.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(C_scratch.data_ptr());
    __nv_bfloat16* C_bf16_ptr = reinterpret_cast<__nv_bfloat16*>(C_bf16.data_ptr());
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = C_scratch.size(2);
    const half** suh_ptrs = reinterpret_cast<const half**>(suh.data_ptr());
    half* A_had_ptr = reinterpret_cast<half*>(A_had.data_ptr());
    const half** svh_ptrs = reinterpret_cast<const half**>(svh.data_ptr());
    void* args[] = {
        &A_ptr, &B_ptrs, &C_ptr, &C_bf16_ptr,
        &size_m, &size_k, &size_n, &locks_ptr,
        &suh_ptrs, &A_had_ptr, &svh_ptrs, &count, &output_stride,
    };
    cuda_check(cudaLaunchCooperativeKernel(
        kernel, dim3(num_sms, 1, concurrency),
        dim3(block_dim), args, smem_bytes, stream));
    cuda_check(cudaPeekAtLastError());
    return 2;
}

int exl3_mgemm_bf16_io
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    int bits,
    int force_num_sms,
    int output_stride,
    bool mcg,
    bool direct_output,
    bool final_group_barrier
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
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(C_scratch, kHalf);
    TORCH_CHECK_DTYPE(C_bf16, kBFloat16);
    TORCH_CHECK_DTYPE(suh, kLong);
    TORCH_CHECK_DTYPE(A_had, kHalf);
    TORCH_CHECK_DTYPE(svh, kLong);
    TORCH_CHECK(A.dim() == 2 && C_scratch.dim() == 3 && A_had.dim() == 3,
                "BF16 MGEMM requires 2D A and 3D scratch workspaces");
    TORCH_CHECK(B.dim() == 1 && suh.dim() == 1 && svh.dim() == 1,
                "BF16 MGEMM pointer tables must be one-dimensional");
    TORCH_CHECK(C_bf16.dim() == 2, "BF16 MGEMM output must be 2D");
    TORCH_CHECK(A.size(0) > 0 && A.size(0) <= 16,
                "BF16 MGEMM supports 1 <= m <= 16");
    TORCH_CHECK(A.size(1) > 0 && A.size(1) % 128 == 0,
                "BF16 MGEMM requires positive, 128-aligned k");
    TORCH_CHECK(A.size(1) <= INT_MAX,
                "BF16 MGEMM k exceeds the kernel integer range");
    TORCH_CHECK(B.numel() > 0 && B.numel() <= EXL3_BF16_MAX_GROUPED_MATRICES,
                "BF16 MGEMM matrix count is outside the supported range");
    TORCH_CHECK(C_scratch.size(0) == B.numel() && C_scratch.size(1) == A.size(0),
                "BF16 MGEMM scratch shape does not match count and m");
    TORCH_CHECK(C_scratch.size(2) > 0 && C_scratch.size(2) % 128 == 0,
                "BF16 MGEMM requires positive, 128-aligned n");
    TORCH_CHECK(C_scratch.size(2) <= INT_MAX,
                "BF16 MGEMM n exceeds the kernel integer range");
    TORCH_CHECK(C_bf16.size(0) == A.size(0),
                "BF16 MGEMM output m does not match A");
    TORCH_CHECK(B.numel() == suh.numel() && B.numel() == svh.numel(),
                "BF16 MGEMM pointer table lengths must match");
    TORCH_CHECK(output_stride == C_bf16.size(1),
                "BF16 MGEMM output_stride must match the output width");
    TORCH_CHECK(
        B.numel() * C_scratch.size(2) <= INT_MAX &&
        output_stride >= B.numel() * C_scratch.size(2),
        "BF16 MGEMM output does not hold every matrix"
    );
    TORCH_CHECK(B.numel() * (C_scratch.size(2) / 128) <= MAX_TILES_C,
                "BF16 MGEMM output exceeds the device lock workspace");
    TORCH_CHECK(A_had.sizes() == at::IntArrayRef({B.numel(), A.size(0), A.size(1)}),
                "BF16 MGEMM Hadamard workspace has an incompatible shape");
    TORCH_CHECK(mcg, "BF16 MGEMM supports MCG only");
    if (bits == 5) return launch_mgemm_bf16_io<5, 1>(A, B, C_scratch, C_bf16, suh, A_had, svh, force_num_sms, output_stride, direct_output, final_group_barrier);
    if (bits == 6) return launch_mgemm_bf16_io<6, 1>(A, B, C_scratch, C_bf16, suh, A_had, svh, force_num_sms, output_stride, direct_output, final_group_barrier);
    TORCH_CHECK(false, "BF16 MGEMM supports K5/K6 only");
}

template<int bits, int cb>
int launch_mgemm_bf16_io_grouped_had
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& unique_suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    const Exl3Bf16HadGroupIds& had_group_ids,
    int force_num_sms,
    int output_stride,
    bool direct_output,
    bool final_group_barrier
)
{
    constexpr int block_dim = 512;
    constexpr int smem_bytes = 90 * 1024;
    void* kernel = direct_output
        ? final_group_barrier
            ? reinterpret_cast<void*>(exl3_mgemm_bf16_io_grouped_had_kernel<bits, cb, true, true>)
            : reinterpret_cast<void*>(exl3_mgemm_bf16_io_grouped_had_kernel<bits, cb, true, false>)
        : final_group_barrier
            ? reinterpret_cast<void*>(exl3_mgemm_bf16_io_grouped_had_kernel<bits, cb, false, true>)
            : reinterpret_cast<void*>(exl3_mgemm_bf16_io_grouped_had_kernel<bits, cb, false, false>);
    int device = A.get_device();
    int total_sms = DevCtx::instance().get_num_sms(device);
    int count = B.numel();
    int group_count = unique_suh.numel();
    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = C_scratch.size(2);
    int cooperative_capacity = exl3_bf16_cooperative_capacity(
        kernel, device, block_dim, smem_bytes);
    int num_sms = exl3_bf16_select_sms(
        force_num_sms, total_sms, cooperative_capacity, count);
    int* locks_ptr = DevCtx::instance().get_locks(device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const __nv_bfloat16* A_ptr = reinterpret_cast<const __nv_bfloat16*>(A.data_ptr());
    half* A_had_ptr = reinterpret_cast<half*>(A_had.data_ptr());
    const half** unique_suh_ptrs = reinterpret_cast<const half**>(unique_suh.data_ptr());
    exl3_bf16_grouped_had_kernel<<<dim3(size_m, size_k / 128, group_count), 32, 0, stream>>>
    (A_ptr, A_had_ptr, unique_suh_ptrs, size_m, size_k);
    cuda_check(cudaPeekAtLastError());

    const uint16_t** B_ptrs = reinterpret_cast<const uint16_t**>(B.data_ptr());
    half* C_ptr = reinterpret_cast<half*>(C_scratch.data_ptr());
    __nv_bfloat16* C_bf16_ptr = reinterpret_cast<__nv_bfloat16*>(C_bf16.data_ptr());
    const half** svh_ptrs = reinterpret_cast<const half**>(svh.data_ptr());
    Exl3Bf16HadGroupIds ids = had_group_ids;
    void* args[] = {
        &B_ptrs, &C_ptr, &C_bf16_ptr, &size_m, &size_k, &size_n, &locks_ptr,
        &A_had_ptr, &svh_ptrs, &ids, &count, &group_count, &output_stride,
    };
    cuda_check(cudaLaunchCooperativeKernel(
        kernel, dim3(num_sms, 1, count),
        dim3(block_dim), args, smem_bytes, stream));
    cuda_check(cudaPeekAtLastError());
    return 2;
}

int exl3_mgemm_bf16_io_grouped_had
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C_scratch,
    at::Tensor& C_bf16,
    const at::Tensor& unique_suh,
    at::Tensor& A_had,
    const at::Tensor& svh,
    const at::Tensor& had_group_ids,
    int bits,
    int force_num_sms,
    int output_stride,
    bool mcg,
    bool direct_output,
    bool final_group_barrier
)
{
    check_cuda_tensor(A, A, "A");
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    check_cuda_tensor(B, A, "B");
    check_cuda_tensor(C_scratch, A, "C_scratch");
    check_cuda_tensor(C_bf16, A, "C_bf16");
    check_cuda_tensor(unique_suh, A, "unique_suh");
    check_cuda_tensor(A_had, A, "A_had");
    check_cuda_tensor(svh, A, "svh");
    TORCH_CHECK(had_group_ids.device().is_cpu(), "had_group_ids must be a CPU tensor");
    TORCH_CHECK(had_group_ids.is_contiguous(), "had_group_ids must be contiguous");
    TORCH_CHECK_DTYPE(A, kBFloat16);
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(C_scratch, kHalf);
    TORCH_CHECK_DTYPE(C_bf16, kBFloat16);
    TORCH_CHECK_DTYPE(unique_suh, kLong);
    TORCH_CHECK_DTYPE(A_had, kHalf);
    TORCH_CHECK_DTYPE(svh, kLong);
    TORCH_CHECK_DTYPE(had_group_ids, kInt);
    TORCH_CHECK(A.dim() == 2 && C_scratch.dim() == 3 && A_had.dim() == 3,
                "grouped-Had BF16 MGEMM requires 2D A and 3D scratch workspaces");
    TORCH_CHECK(B.dim() == 1 && unique_suh.dim() == 1 && svh.dim() == 1,
                "grouped-Had BF16 MGEMM pointer tables must be one-dimensional");
    TORCH_CHECK(had_group_ids.dim() == 1,
                "had_group_ids must be one-dimensional");
    TORCH_CHECK(C_bf16.dim() == 2,
                "grouped-Had BF16 MGEMM output must be 2D");
    TORCH_CHECK(A.size(0) > 0 && A.size(0) <= 16,
                "grouped-Had BF16 MGEMM supports 1 <= m <= 16");
    TORCH_CHECK(A.size(1) > 0 && A.size(1) % 128 == 0,
                "grouped-Had BF16 MGEMM requires positive, 128-aligned k");
    TORCH_CHECK(A.size(1) <= INT_MAX,
                "grouped-Had BF16 MGEMM k exceeds the kernel integer range");
    TORCH_CHECK(B.numel() > 0 && B.numel() <= EXL3_BF16_MAX_GROUPED_MATRICES,
                "grouped-Had BF16 MGEMM matrix count is outside the supported range");
    TORCH_CHECK(unique_suh.numel() > 0 &&
                unique_suh.numel() <= EXL3_BF16_MAX_GROUPED_MATRICES,
                "grouped-Had BF16 MGEMM group count is outside the supported range");
    TORCH_CHECK(C_scratch.size(0) == B.numel() && C_scratch.size(1) == A.size(0),
                "grouped-Had BF16 MGEMM scratch shape does not match count and m");
    TORCH_CHECK(C_scratch.size(2) > 0 && C_scratch.size(2) % 128 == 0,
                "grouped-Had BF16 MGEMM requires positive, 128-aligned n");
    TORCH_CHECK(C_scratch.size(2) <= INT_MAX,
                "grouped-Had BF16 MGEMM n exceeds the kernel integer range");
    TORCH_CHECK(C_bf16.size(0) == A.size(0),
                "grouped-Had BF16 MGEMM output m does not match A");
    TORCH_CHECK(B.numel() == svh.numel() && B.numel() == had_group_ids.numel(),
                "grouped-Had BF16 MGEMM table lengths must match matrix count");
    TORCH_CHECK(A_had.size(0) == unique_suh.numel(),
                "grouped-Had BF16 MGEMM workspace does not match group count");
    TORCH_CHECK(A_had.size(1) == A.size(0) && A_had.size(2) == A.size(1),
                "grouped-Had BF16 MGEMM workspace does not match A");
    TORCH_CHECK(output_stride == C_bf16.size(1),
                "grouped-Had BF16 MGEMM output_stride must match the output width");
    TORCH_CHECK(
        B.numel() * C_scratch.size(2) <= INT_MAX &&
        output_stride >= B.numel() * C_scratch.size(2),
        "grouped-Had BF16 MGEMM output does not hold every matrix"
    );
    TORCH_CHECK(B.numel() * (C_scratch.size(2) / 128) <= MAX_TILES_C,
                "grouped-Had BF16 MGEMM output exceeds the device lock workspace");
    TORCH_CHECK(mcg, "grouped-Had BF16 MGEMM supports MCG only");
    TORCH_CHECK(A.size(1) / 128 <= 65535,
                "grouped-Had BF16 MGEMM k exceeds the preprocessing grid limit");
    Exl3Bf16HadGroupIds ids = {};
    const int* ids_ptr = had_group_ids.const_data_ptr<int>();
    for (int j = 0; j < B.numel(); ++j)
    {
        TORCH_CHECK(
            ids_ptr[j] >= 0 && ids_ptr[j] < unique_suh.numel(),
            "had_group_ids[", j, "] is outside [0, ", unique_suh.numel(), ")"
        );
        ids.values[j] = static_cast<uint8_t>(ids_ptr[j]);
    }
    if (bits == 5) return launch_mgemm_bf16_io_grouped_had<5, 1>(A, B, C_scratch, C_bf16, unique_suh, A_had, svh, ids, force_num_sms, output_stride, direct_output, final_group_barrier);
    if (bits == 6) return launch_mgemm_bf16_io_grouped_had<6, 1>(A, B, C_scratch, C_bf16, unique_suh, A_had, svh, ids, force_num_sms, output_stride, direct_output, final_group_barrier);
    TORCH_CHECK(false, "grouped-Had BF16 MGEMM supports K5/K6 only");
}
