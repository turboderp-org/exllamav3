#include <cuda_fp16.h>
#include "q_cache.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include <limits>
#include "../quant/codebook.cuh"
#include "q_cache_kernels.cuh"

/*
Quantize contiguous tensor

in: float16, shape (..., dim)
out: int32, shape (..., dim / 32 * bitrate)
out_scales: float16, shape (..., dim / 32)
*/

void quant_cache_cont
(
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
    int bits = out.size(-1) / head_blocks;
    TORCH_CHECK(out.numel() == bsz * bits, "out is wrong size");
    TORCH_CHECK(out_scales.numel() == bsz, "out_scales is wrong size");

    TORCH_CHECK(2 <= bits && bits <= 8, "no kernel for K/V bitrate");

    quant_cache_cont_kernel_instances[bits - 2]<<<bsz, 32, 0, stream>>>
    (
        (const half*) in.data_ptr(),
        (uint32_t*) out.data_ptr(),
        (half*) out_scales.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

/*
Dequantize contiguous tensor

in: int32, shape (..., dim / 32 * bitrate)
in_scales: float16, shape (..., dim / 32)
out: float16, shape (..., dim)
*/

void dequant_cache_cont
(
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
    int head_dim = out.size(-1);
    int head_blocks = head_dim / 32;
    TORCH_CHECK(head_dim == 32 * head_blocks, "head_dim must be a multiple of 32");
    int bits = in.size(-1) / head_blocks;
    TORCH_CHECK(in.numel() == bsz * bits, "in is wrong size");
    TORCH_CHECK(in_scales.numel() == bsz, "in_scales is wrong size");

    TORCH_CHECK(2 <= bits && bits <= 8, "no kernel for K/V bitrate");

    dequant_cache_cont_kernel_instances[bits - 2]<<<bsz, 32, 0, stream>>>
    (
        (const uint32_t*) in.data_ptr(),
        (const half*) in_scales.data_ptr(),
        (half*) out.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

/*
Quantize paged tensor

k_in, v_in: float16, shape (1, cache_size, dim)
k_out, v_out: int32, shape (1, cache_size, dim / 32 * bitrate)
k_out_scales, v_out_scales: float16, shape (1, cache_size, dim / 32)
cache_seqlens: int32, length of each sequence in batch, k_out and v_out are updated _from_ this point
block_table: int32, shape (bsz, blocks_per_seq)
page_size: 256
seq_len: number of positions (size: dim) to update from end of each sequence
*/

void quant_cache_paged
(
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
    TORCH_CHECK_SHAPES_FULL(k_in, v_in);
    TORCH_CHECK_SHAPES_FULL(k_out_scales, v_out_scales);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    int dim;
    if (k_in.dim() == 4)
        dim = k_in.size(2) * k_in.size(3);
    else if (k_in.dim() == 3)
        dim = k_in.size(2);
    else
        TORCH_CHECK(false, "paged cache must be 3D or 4D")

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");
    int tb_per_token = CEIL_DIVIDE(warps_per_token, MAX_WARPS);  // Threadblocks per token position
    int tb_usage = CEIL_DIVIDE(warps_per_token, tb_per_token);   // Number of warps to use per threadblock

    TORCH_CHECK(k_out.dim() == 3 && v_out.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)")
    int k_bits = k_out.size(2) / warps_per_token;
    int v_bits = v_out.size(2) / warps_per_token;

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    quant_cache_paged_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const half*) k_in.data_ptr(),
        (uint32_t*) k_out.data_ptr(),
        (half*) k_out_scales.data_ptr(),
        (const half*) v_in.data_ptr(),
        (uint32_t*) v_out.data_ptr(),
        (half*) v_out_scales.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        // page_size,
        blocks_per_seq,
        dim
    );
    cuda_check(cudaPeekAtLastError());
}

void quant_cache_paged_delta
(
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
    TORCH_CHECK_SHAPES_FULL(k_in, v_in);
    TORCH_CHECK_SHAPES_FULL(k_out_scales, v_out_scales);
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

    TORCH_CHECK(k_out.dim() == 3 && v_out.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)");
    int k_bits = k_out.size(2) / warps_per_token;
    int v_bits = v_out.size(2) / warps_per_token;

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    quant_cache_paged_delta_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const half*) k_in.data_ptr(),
        (uint32_t*) k_out.data_ptr(),
        (half*) k_out_scales.data_ptr(),
        (const half*) v_in.data_ptr(),
        (uint32_t*) v_out.data_ptr(),
        (half*) v_out_scales.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        blocks_per_seq,
        dim,
        seq_len
    );
    cuda_check(cudaPeekAtLastError());
}

/*
Dequantize paged tensor

k_in, v_in: int32, shape (1, cache_size, dim / 32 * bitrate)
k_in_scales, v_in_scales: float16, shape (1, cache_size, dim / 32)
k_out, v_out: float16, shape (1, cache_size, dim)
cache_seqlens: int32, length of each sequence in batch, k_out and v_out are updated _up_to_ this point
block_table: int32, shape (bsz, blocks_per_seq)
page_size: 256
*/

void dequant_cache_paged
(
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
    TORCH_CHECK_SHAPES_FULL(k_in_scales, v_in_scales);
    TORCH_CHECK_SHAPES_FULL(k_out, v_out);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    int dim;
    if (k_out.dim() == 4)
        dim = k_out.size(2) * k_out.size(3);
    else if (k_out.dim() == 3)
        dim = k_out.size(2);
    else
        TORCH_CHECK(false, "paged cache must be 3D or 4D")

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");

    int bsz = block_table.size(0);
    int pages_per_seq = block_table.size(1);
    int warps_per_seq = pages_per_seq * page_size * warps_per_token;

    int num_blocks = CEIL_DIVIDE(32 * warps_per_seq, 32 * MAX_WARPS);
    int num_tb = CEIL_DIVIDE(num_blocks, ITER_PER_TB);

    int num_threads = MIN(32 * warps_per_seq, 32 * MAX_WARPS);
    dim3 blocks(num_tb, bsz);
    dim3 threads(num_threads);

    TORCH_CHECK(k_in.dim() == 3 && v_in.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)")
    int k_bits = k_in.size(2) / warps_per_token;
    int v_bits = v_in.size(2) / warps_per_token;

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    dequant_cache_paged_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const uint32_t*) k_in.data_ptr(),
        (const half*) k_in_scales.data_ptr(),
        (half*) k_out.data_ptr(),
        (const uint32_t*) v_in.data_ptr(),
        (const half*) v_in_scales.data_ptr(),
        (half*) v_out.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        // page_size,
        pages_per_seq,
        warps_per_token,
        num_blocks
    );
    cuda_check(cudaPeekAtLastError());
}

void dequant_cache_paged_gather
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
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
    TORCH_CHECK_SHAPES_FULL(k_in_scales, v_in_scales);
    TORCH_CHECK_SHAPES_FULL(k_out, v_out);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    int dim;
    if (k_out.dim() == 4)
        dim = k_out.size(2) * k_out.size(3);
    else if (k_out.dim() == 3)
        dim = k_out.size(2);
    else
        TORCH_CHECK(false, "paged cache output must be 3D or 4D");

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");
    int tb_per_token = CEIL_DIVIDE(warps_per_token, MAX_WARPS);
    int tb_usage = CEIL_DIVIDE(warps_per_token, tb_per_token);

    TORCH_CHECK(k_in.dim() == 3 && v_in.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)");
    int k_bits = k_in.size(2) / warps_per_token;
    int v_bits = v_in.size(2) / warps_per_token;

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    dequant_cache_paged_gather_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const uint32_t*) k_in.data_ptr(),
        (const half*) k_in_scales.data_ptr(),
        (half*) k_out.data_ptr(),
        (const uint32_t*) v_in.data_ptr(),
        (const half*) v_in_scales.data_ptr(),
        (half*) v_out.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        blocks_per_seq,
        dim,
        seq_len
    );
    cuda_check(cudaPeekAtLastError());
}

void dequant_cache_paged_gather_heads
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len
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
    TORCH_CHECK_SHAPES_FULL(k_in_scales, v_in_scales);
    TORCH_CHECK_SHAPES_FULL(k_out, v_out);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    TORCH_CHECK(k_out.dim() == 4, "heads-first paged cache output must be 4D");
    TORCH_CHECK(k_out.size(2) == seq_len, "heads-first output seq_len mismatch");

    int num_kv_heads = k_out.size(1);
    int head_dim = k_out.size(3);
    int dim = num_kv_heads * head_dim;
    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be a multiple of 32");
    int tb_per_token = CEIL_DIVIDE(warps_per_token, MAX_WARPS);
    int tb_usage = CEIL_DIVIDE(warps_per_token, tb_per_token);

    TORCH_CHECK(k_in.dim() == 3 && v_in.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)");
    int k_bits = k_in.size(2) / warps_per_token;
    int v_bits = v_in.size(2) / warps_per_token;

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    dequant_cache_paged_gather_heads_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const uint32_t*) k_in.data_ptr(),
        (const half*) k_in_scales.data_ptr(),
        (half*) k_out.data_ptr(),
        (const uint32_t*) v_in.data_ptr(),
        (const half*) v_in_scales.data_ptr(),
        (half*) v_out.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        blocks_per_seq,
        dim,
        seq_len,
        num_kv_heads,
        head_dim
    );
    cuda_check(cudaPeekAtLastError());
}

void dequant_cache_paged_gather_delta
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_delta,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_delta,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len,
    int delta_len
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(k_in, kInt);
    TORCH_CHECK_DTYPE(k_in_scales, kHalf);
    TORCH_CHECK_DTYPE(k_delta, kHalf);
    TORCH_CHECK_DTYPE(k_out, kHalf);
    TORCH_CHECK_DTYPE(v_in, kInt);
    TORCH_CHECK_DTYPE(v_in_scales, kHalf);
    TORCH_CHECK_DTYPE(v_delta, kHalf);
    TORCH_CHECK_DTYPE(v_out, kHalf);
    TORCH_CHECK_SHAPES_FULL(k_in_scales, v_in_scales);
    TORCH_CHECK_SHAPES_FULL(k_delta, v_delta);
    TORCH_CHECK_SHAPES_FULL(k_out, v_out);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    int dim;
    if (k_out.dim() == 4)
        dim = k_out.size(2) * k_out.size(3);
    else if (k_out.dim() == 3)
        dim = k_out.size(2);
    else
        TORCH_CHECK(false, "paged cache output must be 3D or 4D");

    int delta_dim;
    if (k_delta.dim() == 4)
        delta_dim = k_delta.size(2) * k_delta.size(3);
    else if (k_delta.dim() == 3)
        delta_dim = k_delta.size(2);
    else
        TORCH_CHECK(false, "delta cache input must be 3D or 4D");

    TORCH_CHECK(delta_dim == dim, "delta dim must match output dim");

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");
    int tb_per_token = CEIL_DIVIDE(warps_per_token, MAX_WARPS);
    int tb_usage = CEIL_DIVIDE(warps_per_token, tb_per_token);

    TORCH_CHECK(k_in.dim() == 3 && v_in.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)");
    int k_bits = k_in.size(2) / warps_per_token;
    int v_bits = v_in.size(2) / warps_per_token;

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    dequant_cache_paged_gather_delta_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const uint32_t*) k_in.data_ptr(),
        (const half*) k_in_scales.data_ptr(),
        (const half*) k_delta.data_ptr(),
        (half*) k_out.data_ptr(),
        (const uint32_t*) v_in.data_ptr(),
        (const half*) v_in_scales.data_ptr(),
        (const half*) v_delta.data_ptr(),
        (half*) v_out.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        blocks_per_seq,
        dim,
        seq_len,
        delta_len
    );
    cuda_check(cudaPeekAtLastError());
}

void dequant_cache_paged_gather_delta_heads
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_delta,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_delta,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    int page_size,
    int seq_len,
    int delta_len
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(k_in, kInt);
    TORCH_CHECK_DTYPE(k_in_scales, kHalf);
    TORCH_CHECK_DTYPE(k_delta, kHalf);
    TORCH_CHECK_DTYPE(k_out, kHalf);
    TORCH_CHECK_DTYPE(v_in, kInt);
    TORCH_CHECK_DTYPE(v_in_scales, kHalf);
    TORCH_CHECK_DTYPE(v_delta, kHalf);
    TORCH_CHECK_DTYPE(v_out, kHalf);
    TORCH_CHECK_SHAPES_FULL(k_in_scales, v_in_scales);
    TORCH_CHECK_SHAPES_FULL(k_delta, v_delta);
    TORCH_CHECK_SHAPES_FULL(k_out, v_out);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    TORCH_CHECK(k_out.dim() == 4, "heads-first paged cache output must be 4D");
    TORCH_CHECK(k_out.size(2) == seq_len, "heads-first output seq_len mismatch");

    int num_kv_heads = k_out.size(1);
    int head_dim = k_out.size(3);
    int dim = num_kv_heads * head_dim;
    int delta_dim;
    if (k_delta.dim() == 4)
        delta_dim = k_delta.size(2) * k_delta.size(3);
    else if (k_delta.dim() == 3)
        delta_dim = k_delta.size(2);
    else
        TORCH_CHECK(false, "delta cache input must be 3D or 4D");

    TORCH_CHECK(delta_dim == dim, "delta dim must match output dim");
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be a multiple of 32");

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");
    int tb_per_token = CEIL_DIVIDE(warps_per_token, MAX_WARPS);
    int tb_usage = CEIL_DIVIDE(warps_per_token, tb_per_token);

    TORCH_CHECK(k_in.dim() == 3 && v_in.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)");
    int k_bits = k_in.size(2) / warps_per_token;
    int v_bits = v_in.size(2) / warps_per_token;

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    dequant_cache_paged_gather_delta_heads_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const uint32_t*) k_in.data_ptr(),
        (const half*) k_in_scales.data_ptr(),
        (const half*) k_delta.data_ptr(),
        (half*) k_out.data_ptr(),
        (const uint32_t*) v_in.data_ptr(),
        (const half*) v_in_scales.data_ptr(),
        (const half*) v_delta.data_ptr(),
        (half*) v_out.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        blocks_per_seq,
        dim,
        seq_len,
        delta_len,
        num_kv_heads,
        head_dim
    );
    cuda_check(cudaPeekAtLastError());
}

void dequant_cache_paged_select_delta_heads
(
    const at::Tensor& k_in,
    const at::Tensor& k_in_scales,
    const at::Tensor& k_delta,
    const at::Tensor& k_out,
    const at::Tensor& v_in,
    const at::Tensor& v_in_scales,
    const at::Tensor& v_delta,
    const at::Tensor& v_out,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    const at::Tensor& selected_positions,
    const at::Tensor& selected_counts,
    int page_size,
    int seq_len,
    int delta_len
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(k_in, kInt);
    TORCH_CHECK_DTYPE(k_in_scales, kHalf);
    TORCH_CHECK_DTYPE(k_delta, kHalf);
    TORCH_CHECK_DTYPE(k_out, kHalf);
    TORCH_CHECK_DTYPE(v_in, kInt);
    TORCH_CHECK_DTYPE(v_in_scales, kHalf);
    TORCH_CHECK_DTYPE(v_delta, kHalf);
    TORCH_CHECK_DTYPE(v_out, kHalf);
    TORCH_CHECK_DTYPE(cache_seqlens, kInt);
    TORCH_CHECK_DTYPE(block_table, kInt);
    TORCH_CHECK_DTYPE(selected_positions, kInt);
    TORCH_CHECK_DTYPE(selected_counts, kInt);
    TORCH_CHECK_SHAPES_FULL(k_in_scales, v_in_scales);
    TORCH_CHECK_SHAPES_FULL(k_delta, v_delta);
    TORCH_CHECK_SHAPES_FULL(k_out, v_out);
    TORCH_CHECK(page_size == CQ_PAGE_SIZE, "Page size mismatch");

    TORCH_CHECK(k_out.dim() == 4, "heads-first paged cache output must be 4D");
    TORCH_CHECK(k_out.size(2) == seq_len, "heads-first output seq_len mismatch");
    TORCH_CHECK(selected_positions.dim() == 2, "selected_positions must have shape (bsz, max_selected)");
    TORCH_CHECK(selected_counts.dim() == 1, "selected_counts must have shape (bsz)");

    int bsz = block_table.size(0);
    TORCH_CHECK(selected_positions.size(0) == bsz, "selected_positions batch mismatch");
    TORCH_CHECK(selected_counts.size(0) == bsz, "selected_counts batch mismatch");

    int max_selected = selected_positions.size(1);
    TORCH_CHECK(seq_len == max_selected, "selected gather seq_len mismatch");

    int num_kv_heads = k_out.size(1);
    int head_dim = k_out.size(3);
    int dim = num_kv_heads * head_dim;
    int delta_dim;
    if (k_delta.dim() == 4)
        delta_dim = k_delta.size(2) * k_delta.size(3);
    else if (k_delta.dim() == 3)
        delta_dim = k_delta.size(2);
    else
        TORCH_CHECK(false, "delta cache input must be 3D or 4D");

    TORCH_CHECK(delta_dim == dim, "delta dim must match output dim");
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be a multiple of 32");

    int warps_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * warps_per_token, "dim must be a multiple of 32");
    int tb_per_token = CEIL_DIVIDE(warps_per_token, MAX_WARPS);
    int tb_usage = CEIL_DIVIDE(warps_per_token, tb_per_token);

    TORCH_CHECK(k_in.dim() == 3 && v_in.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)");
    int k_bits = k_in.size(2) / warps_per_token;
    int v_bits = v_in.size(2) / warps_per_token;
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, max_selected, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    dequant_cache_paged_select_delta_heads_kernel_instances[k_bits - 2][v_bits - 2]<<<blocks, threads, 0, stream>>>
    (
        (const uint32_t*) k_in.data_ptr(),
        (const half*) k_in_scales.data_ptr(),
        (const half*) k_delta.data_ptr(),
        (half*) k_out.data_ptr(),
        (const uint32_t*) v_in.data_ptr(),
        (const half*) v_in_scales.data_ptr(),
        (const half*) v_delta.data_ptr(),
        (half*) v_out.data_ptr(),
        (const uint32_t*) cache_seqlens.data_ptr(),
        (const uint32_t*) block_table.data_ptr(),
        (const uint32_t*) selected_positions.data_ptr(),
        (const uint32_t*) selected_counts.data_ptr(),
        blocks_per_seq,
        dim,
        seq_len,
        delta_len,
        max_selected,
        num_kv_heads,
        head_dim
    );
    cuda_check(cudaPeekAtLastError());
}
