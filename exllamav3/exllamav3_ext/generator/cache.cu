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