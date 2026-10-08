#include <cuda_fp16.h>
#include "stloader_cu.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

void inplace_bf16_to_fp16_cpu
(
    void* buffer,
    const size_t numel
)
{
    const __nv_bfloat16* rd = (const __nv_bfloat16*) buffer;
    __half* wr = (__half*) buffer;

    for (size_t i = 0; i < numel; ++i)
    {
        float f32 = __bfloat162float(rd[i]);
        wr[i] = __float2half_rn(f32);
    }
}

#define NUM_THREADS 1024

__global__ __launch_bounds__(NUM_THREADS)
void inplace_bf16_to_fp16_kernel
(
    void* __restrict__ buffer,
    const size_t numel2,
    const bool odd_tail,
    void* __restrict__ head
)
{
    size_t i = (size_t) blockIdx.x * NUM_THREADS + threadIdx.x;

    if (i < numel2)
    {
        const __nv_bfloat162* rd2 = (const __nv_bfloat162*) buffer;
        __half2* wr2 = (__half2*) buffer;

        __nv_bfloat162 b2 = rd2[i];
        float2 f2 = __bfloat1622float2(b2);
        __half2 h2 = __floats2half2_rn(f2.x, f2.y);
        wr2[i] = h2;
    }

    // Odd-numel tensors have one element past the last half2 pair
    else if (i == numel2 && odd_tail)
    {
        const __nv_bfloat16* rd = (const __nv_bfloat16*) buffer;
        __half* wr = (__half*) buffer;

        float f32 = __bfloat162float(rd[numel2 * 2]);
        wr[numel2 * 2] = __float2half_rn(f32);
    }

    // A buffer that starts between two half2 pairs has one element before the first pair
    if (head && i == 0)
    {
        float f32 = __bfloat162float(*(const __nv_bfloat16*) head);
        *(__half*) head = __float2half_rn(f32);
    }
}

cudaError_t inplace_bf16_to_fp16_cuda
(
    void* buffer,
    const size_t numel,
    cudaStream_t stream
)
{
    // The pairwise conversion needs 4-byte alignment; a buffer only 2-byte aligned (an odd element offset into
    // an allocation, or a chunk boundary at an odd element) converts its first element on its own
    uint8_t* pairs = (uint8_t*) buffer;
    size_t n = numel;
    void* head = nullptr;
    if (n && ((uintptr_t) pairs & 2))
    {
        head = pairs;
        pairs += 2;
        n -= 1;
    }
    size_t numel2 = n / 2;
    size_t total_threads = numel2 + (n & 1);
    if (!total_threads && !head) return cudaSuccess;
    size_t blocks = total_threads ? CEIL_DIVIDE(total_threads, NUM_THREADS) : 1;  // head only: one block
    inplace_bf16_to_fp16_kernel<<<blocks, NUM_THREADS, 0, stream>>>(pairs, numel2, (bool) (n & 1), head);
    return cudaPeekAtLastError();
}
