#include <cuda_fp16.h>
#include "tq3_cache.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "q_cache_kernels.cuh"     // for shuffle_had_fx32, shuffle_max_fx32, MAX_WARPS, etc.
#include "tq3_cache_kernels.cuh"

// Contiguous quantize
void quant_tq3_cache_cont(
    const at::Tensor& in,
    const at::Tensor& out,
    const at::Tensor& out_scales
)
{
    const at::cuda::OptionalCUDAGuard device_guard(in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(in, kHalf);
    TORCH_CHECK_DTYPE(out, kInt);
    TORCH_CHECK_DTYPE(out_scales, kHalf);

    int bsz = in.numel() / 32;
    int head_dim = in.size(-1);
    int head_blocks = head_dim / 32;
    TORCH_CHECK(head_dim == 32 * head_blocks, "head_dim must be a multiple of 32");
    TORCH_CHECK(out.numel() == bsz * 2, "out is wrong size for TQ3 (2 uint32 per block)");
    TORCH_CHECK(out_scales.numel() == bsz, "out_scales is wrong size");

    quant_tq3_cache_cont_kernel<<<bsz, 32, 0, stream>>>(
        (const half*) in.data_ptr(),
        (uint32_t*) out.data_ptr(),
        (half*) out_scales.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

// Contiguous dequantize
void dequant_tq3_cache_cont(
    const at::Tensor& in,
    const at::Tensor& in_scales,
    const at::Tensor& out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(in, kInt);
    TORCH_CHECK_DTYPE(in_scales, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);

    int bsz = out.numel() / 32;
    TORCH_CHECK(in.numel() == bsz * 2, "in is wrong size for TQ3");
    TORCH_CHECK(in_scales.numel() == bsz, "in_scales is wrong size");

    dequant_tq3_cache_cont_kernel<<<bsz, 32, 0, stream>>>(
        (const uint32_t*) in.data_ptr(),
        (const half*) in_scales.data_ptr(),
        (half*) out.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

// Paged quantize
void quant_tq3_cache_paged(
    const at::Tensor& k_in,
    const at::Tensor& k_out,
    const at::Tensor& k_out_scales,
    const at::Tensor& v_in,
    const at::Tensor& v_out,
    const at::Tensor& v_out_scales,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(k_in, kHalf);
    TORCH_CHECK_DTYPE(k_out, kInt);
    TORCH_CHECK_DTYPE(k_out_scales, kHalf);
    TORCH_CHECK_DTYPE(v_in, kHalf);
    TORCH_CHECK_DTYPE(v_out, kInt);
    TORCH_CHECK_DTYPE(v_out_scales, kHalf);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    int dim;
    if (k_in.dim() == 4)
        dim = k_in.size(2) * k_in.size(3);
    else if (k_in.dim() == 3)
        dim = k_in.size(2);
    else
        TORCH_CHECK(false, "paged cache must be 3D or 4D");

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");
    int tb_per_token = CEIL_DIVIDE(warps_per_token, MAX_WARPS);
    int tb_usage = CEIL_DIVIDE(warps_per_token, tb_per_token);

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    quant_tq3_cache_paged_kernel<<<blocks, threads, 0, stream>>>(
        (const half*) k_in.data_ptr(),
        (uint32_t*) k_out.data_ptr(),
        (half*) k_out_scales.data_ptr(),
        (const half*) v_in.data_ptr(),
        (uint32_t*) v_out.data_ptr(),
        (half*) v_out_scales.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        blocks_per_seq,
        dim
    );
    cuda_check(cudaPeekAtLastError());
}

// Paged dequantize
void dequant_tq3_cache_paged(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(k_in, kInt);
    TORCH_CHECK_DTYPE(k_in_scales, kHalf);
    TORCH_CHECK_DTYPE(k_out, kHalf);
    TORCH_CHECK_DTYPE(v_in, kInt);
    TORCH_CHECK_DTYPE(v_in_scales, kHalf);
    TORCH_CHECK_DTYPE(v_out, kHalf);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    int dim;
    if (k_out.dim() == 4)
        dim = k_out.size(2) * k_out.size(3);
    else if (k_out.dim() == 3)
        dim = k_out.size(2);
    else
        TORCH_CHECK(false, "paged cache must be 3D or 4D");

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");

    int bsz = block_table.size(0);
    int pages_per_seq = block_table.size(1);
    int warps_per_seq = pages_per_seq * page_size * warps_per_token;

    int num_blocks = CEIL_DIVIDE(32 * warps_per_seq, 32 * MAX_WARPS);
    int num_tb = CEIL_DIVIDE(num_blocks, ITER_PER_TB);

    int num_threads = MIN(32 * warps_per_seq, 32 * MAX_WARPS);
    dim3 blocks_dim(num_tb, bsz);
    dim3 threads_dim(num_threads);

    dequant_tq3_cache_paged_kernel<<<blocks_dim, threads_dim, 0, stream>>>(
        (const uint32_t*) k_in.data_ptr(),
        (const half*) k_in_scales.data_ptr(),
        (half*) k_out.data_ptr(),
        (const uint32_t*) v_in.data_ptr(),
        (const half*) v_in_scales.data_ptr(),
        (half*) v_out.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        pages_per_seq,
        warps_per_token,
        num_blocks
    );
    cuda_check(cudaPeekAtLastError());
}
