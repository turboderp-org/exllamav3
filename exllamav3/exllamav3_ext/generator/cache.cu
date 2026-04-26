#include <cuda_fp16.h>
#include "cache.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"

#define NUM_THREADS 512
#define NUM_BLOCKS 128

__global__ __launch_bounds__(NUM_THREADS)
void cache_rotate_kernel
(
    uint8_t* __restrict__ cache,
    const int32_t* __restrict__ order,
    uint8_t* __restrict__ temp,
    const size_t page_size,
    const size_t rotate_len
)
{
    // Chunk for current CTA
    size_t block_size = CEIL_DIVIDE(page_size, gridDim.x);
    size_t block_beg = blockIdx.x * block_size;
    size_t block_end = MIN(block_beg + block_size, page_size);
    block_size = block_end - block_beg;
    if (block_size <= 0) return;

    // Rotate pages
    for (int i = 0; i < rotate_len; ++i)
    {
        int64_t a = (int64_t) order[2 * i];
        int64_t b = (int64_t) order[2 * i + 1];
        uint8_t* dst = (a >= 0 ? cache + page_size * a : temp) + block_beg;
        uint8_t* src = (b >= 0 ? cache + page_size * b : temp) + block_beg;
        for (int offset = threadIdx.x * 16; offset < block_size; offset += NUM_THREADS * 16)
            *((uint4*) (dst + offset)) = *((uint4*) (src + offset));
        __syncthreads();
    }
}

/*
Reorder cache pages
- cache, paged cache, shape (num_pages, ...), any dtype, contiguous
- order, sequence to rotate, shape (2*n,), dtype int
- temp, temp storage, sized as one cache page

Performs:

for i in range(n):
    a = order[2*i]
    b = order[2*i+1]
    copy: (page[a] if a >= 0 else temp) <- (page[b] if b >= 0 else temp)
*/

void cache_rotate
(
    const at::Tensor& cache,
    const at::Tensor& order,
    const at::Tensor& temp
)
{
    const at::cuda::OptionalCUDAGuard device_guard(cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(cache.dim() >= 2, "cache argument must have dim >= 2")
    TORCH_CHECK(order.dim() == 1, "order argument must have dim == 1")
    TORCH_CHECK_DTYPE(order, kInt);

    size_t num_pages = cache.size(0);
    size_t page_size = cache.nbytes() / num_pages;
    size_t rotate_len = order.size(0) / 2;

    TORCH_CHECK(temp.nbytes() == page_size, "temp tensor incorrect size");

    cache_rotate_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>
    (
        (uint8_t*) cache.data_ptr(),
        (const int32_t*) order.data_ptr(),
        (uint8_t*) temp.data_ptr(),
        page_size,
        rotate_len
    );
}

constexpr int kPageSize = 256;
constexpr int kThreads = 256;

__global__ void paged_kv_update_vec8_kernel
(
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    int B,
    int S,
    int H,
    int D,
    int max_blocks_per_seq,
    int64_t total_vecs
)
{
    const int64_t vecs_per_row = (int64_t) D >> 3;  // 8 halfs = 16 bytes = uint4
    const int64_t row_vecs = (int64_t) H * vecs_per_row;
    const int64_t seq_vecs = (int64_t) S * row_vecs;
    const int64_t stride = (int64_t) blockDim.x * (int64_t) gridDim.x;

    for (int64_t linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; linear < total_vecs; linear += stride)
    {
        const int64_t vec_id = linear % vecs_per_row;
        const int64_t tmp0 = linear / vecs_per_row;
        const int64_t h = tmp0 % H;
        const int64_t tmp1 = tmp0 / H;
        const int64_t s = tmp1 % S;
        const int64_t b = tmp1 / S;

        const int64_t logical_pos = int64_t(cache_seqlens[b]) + s;
        const int64_t logical_block = logical_pos >> 8;
        const int64_t page_offset = logical_pos & (kPageSize - 1);
        const int64_t phys_block = int64_t(block_table[b * max_blocks_per_seq + logical_block]);

        const int64_t src_elem = (((b * S + s) * H + h) * D) + (vec_id << 3);
        const int64_t dst_elem = (((phys_block * kPageSize + page_offset) * H + h) * D) + (vec_id << 3);

        *((uint4*)(k_cache + dst_elem)) = *((uint4*)(k + src_elem));
        *((uint4*)(v_cache + dst_elem)) = *((uint4*)(v + src_elem));
    }
}

void paged_kv_cache_update
(
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& k_cache,
    at::Tensor& v_cache,
    at::Tensor& block_table,
    at::Tensor& cache_seqlens
)
{
    const at::cuda::OptionalCUDAGuard device_guard(k.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(k, kHalf);
    TORCH_CHECK_DTYPE(v, kHalf);
    TORCH_CHECK_DTYPE(k_cache, kHalf);
    TORCH_CHECK_DTYPE(v_cache, kHalf);
    TORCH_CHECK_DTYPE(block_table, kInt);
    TORCH_CHECK_DTYPE(cache_seqlens, kInt);

    TORCH_CHECK(k.dim() == 4, "k must have shape [B, S_new, H_k, D]");
    TORCH_CHECK(v.dim() == 4, "v must have shape [B, S_new, H_k, D]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape [num_blocks, 256, H_k, D]");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must have shape [num_blocks, 256, H_k, D]");
    TORCH_CHECK(block_table.dim() == 2, "block_table must have shape [B, max_blocks_per_seq]");
    TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must have shape [B]");

    const int B = k.size(0);
    const int S = k.size(1);
    const int H = k.size(2);
    const int D = k.size(3);
    const int max_blocks_per_seq = block_table.size(1);

    TORCH_CHECK_SHAPES_FULL(v, k);
    TORCH_CHECK(k_cache.size(1) == 256, "this kernel needs page_size == 256");
    TORCH_CHECK(v_cache.size(1) == 256, "this kernel needs page_size == 256");

    if (B == 0 || S == 0 || H == 0 || D == 0) return;

    const int max_grid = 65535;
    TORCH_CHECK((D & 7) == 0, "dim must be divisible by 8");

    const int64_t total_vecs = (int64_t) B * (int64_t) S * (int64_t) H * (int64_t) (D >> 3);
    const int grid = std::min((int) CEIL_DIVIDE(total_vecs, (int64_t) kThreads), max_grid);
    paged_kv_update_vec8_kernel<<<grid, kThreads, 0, stream>>>
    (
        (const half*) k.data_ptr(),
        (const half*) v.data_ptr(),
        (half*) k_cache.data_ptr(),
        (half*) v_cache.data_ptr(),
        (const int*) block_table.data_ptr(),
        (const int*) cache_seqlens.data_ptr(),
        B, S, H, D,
        max_blocks_per_seq,
        total_vecs
    );
    cuda_check(cudaPeekAtLastError());
}
