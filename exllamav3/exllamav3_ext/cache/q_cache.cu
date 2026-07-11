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
    const at::Tensor& out_scales,
    float compand_a
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

    int num_blocks = CEIL_DIVIDE(bsz, MAX_WARPS * 4);
    quant_cache_cont_kernel_instances[bits - 2]<<<num_blocks, MAX_WARPS * 32, 0, stream>>>
    (
        (const half*) in.data_ptr(),
        (uint32_t*) out.data_ptr(),
        (half*) out_scales.data_ptr(),
        bsz,
        compand_a
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
    const at::Tensor& out,
    float compand_a
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

    int num_blocks = CEIL_DIVIDE(bsz, MAX_WARPS * 4);
    dequant_cache_cont_kernel_instances[bits - 2]<<<num_blocks, MAX_WARPS * 32, 0, stream>>>
    (
        (const uint32_t*) in.data_ptr(),
        (const half*) in_scales.data_ptr(),
        (half*) out.data_ptr(),
        bsz,
        compand_a
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

void quant_cache_paged_gr
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
    int seq_len,
    float compand_a,
    bool in_contiguous,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

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

    int groups_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * groups_per_token, "dim must be a multiple of 32");
    int chunks_per_token = CEIL_DIVIDE(groups_per_token, 4);     // 4-group warp chunks per token
    int tb_per_token = CEIL_DIVIDE(chunks_per_token, MAX_WARPS); // Threadblocks per token position
    int tb_usage = CEIL_DIVIDE(chunks_per_token, tb_per_token);  // Number of warps to use per threadblock

    TORCH_CHECK(k_out.dim() == 3 && v_out.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)")
    int k_bits = k_out.size(2) / groups_per_token;
    int v_bits = v_out.size(2) / groups_per_token;

    int bsz = block_table.size(0);
    int blocks_per_seq = block_table.size(1);

    dim3 blocks(tb_per_token, seq_len, bsz);
    dim3 threads(32 * tb_usage);

    TORCH_CHECK(2 <= k_bits && k_bits <= 8 && 2 <= v_bits && v_bits <= 8, "no kernel for K/V bitrate");

    void* kernel_ptr = (void*) quant_cache_paged_kernel_instances[k_bits - 2][v_bits - 2];
    const half* k_in_ptr = (const half*) k_in.data_ptr();
    uint32_t* k_out_ptr = (uint32_t*) k_out.data_ptr();
    half* k_out_scales_ptr = (half*) k_out_scales.data_ptr();
    const half* v_in_ptr = (const half*) v_in.data_ptr();
    uint32_t* v_out_ptr = (uint32_t*) v_out.data_ptr();
    half* v_out_scales_ptr = (half*) v_out_scales.data_ptr();
    const uint32_t* cache_seqlens_ptr = (const uint32_t*) cache_seqlens.data_ptr();
    const uint32_t* block_table_ptr = (const uint32_t*) block_table.data_ptr();
    int in_cont = in_contiguous ? 1 : 0;

    void* kernel_args[] =
    {
        &k_in_ptr,
        &k_out_ptr,
        &k_out_scales_ptr,
        &v_in_ptr,
        &v_out_ptr,
        &v_out_scales_ptr,
        &cache_seqlens_ptr,
        &block_table_ptr,
        &blocks_per_seq,
        &groups_per_token,
        &compand_a,
        &in_cont
    };
    cuda_check(cudaLaunchKernel(kernel_ptr, blocks, threads, kernel_args, 0, stream));

    if (graph)
    {
        graph->record_param(kernel_ptr, GP_qcache_seqlens, 6);
        graph->record_param(kernel_ptr, GP_qcache_block_table, 7);
        graph->record_param(kernel_ptr, GP_qcache_blocks_per_seq, 8, 4);
        graph->record_param(kernel_ptr, GP_end, 0);
    }
    cuda_check(cudaPeekAtLastError());
}

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
    int seq_len,
    float compand_a,
    bool in_contiguous
)
{
    quant_cache_paged_gr(k_in, k_out, k_out_scales, v_in, v_out, v_out_scales, cache_seqlens,
                         block_table, page_size, seq_len, compand_a, in_contiguous, nullptr);
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
    int page_size,
    int sliding_window,
    float compand_a
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

    int groups_per_token = dim / 32;
    TORCH_CHECK(dim == 32 * groups_per_token, "dim must be a multiple of 32");
    int chunks_per_token = CEIL_DIVIDE(groups_per_token, 4);

    int bsz = block_table.size(0);
    int pages_per_seq = block_table.size(1);
    int chunks_per_seq = pages_per_seq * page_size * chunks_per_token;

    int num_blocks = CEIL_DIVIDE(chunks_per_seq, MAX_WARPS);
    int num_tb = CEIL_DIVIDE(num_blocks, ITER_PER_TB);

    int num_threads = MIN(32 * chunks_per_seq, 32 * MAX_WARPS);
    dim3 blocks(num_tb, bsz);
    dim3 threads(num_threads);

    TORCH_CHECK(k_in.dim() == 3 && v_in.dim() == 3, "paged q.cache must have shape (num_pages, page_size, dim // 32 * bitrate)")
    int k_bits = k_in.size(2) / groups_per_token;
    int v_bits = v_in.size(2) / groups_per_token;

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
        pages_per_seq,
        groups_per_token,
        chunks_per_token,
        sliding_window,
        compand_a
    );
    cuda_check(cudaPeekAtLastError());
}