#include "rep_pen.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include <limits>

#define BLOCK_VOCAB_SPAN 4096
#define NUM_THREADS 1024

__device__ __forceinline__
float shmemAtomicMaxF(float* addr, float val)  // val > 0
{
    auto uaddr = reinterpret_cast<unsigned int*>(addr);
    unsigned int old = atomicMax(uaddr, __float_as_uint(val));
    return __uint_as_float(old);
}

template <bool input_fp16>
__global__ __launch_bounds__(NUM_THREADS)
void apply_rep_pens_kernel
(
    const void* __restrict__ in_logits,
    float* __restrict__ out_logits,
    const uint64_t* __restrict__ past_ids,
    int past_len,
    int vocab_size,
    float rep_p,
    int sustain_range,
    int decay_range
)
{
    // Each block processes a range of the logits
    int range_min = blockIdx.x * BLOCK_VOCAB_SPAN;
    int range_max = MIN(range_min + BLOCK_VOCAB_SPAN, vocab_size);

    __shared__ float factors[BLOCK_VOCAB_SPAN];
    for (int i = threadIdx.x; i < BLOCK_VOCAB_SPAN; i += NUM_THREADS)
        factors[i] = 0.0f;
    __syncthreads();

    // Record which tokens from the range appear in past_ids and
    for (int i = threadIdx.x; i < past_len; i += NUM_THREADS)
    {
        if (i < past_len - sustain_range - decay_range)
            continue;

        int tid = (int) past_ids[i];
        if (tid < range_min || tid >= range_max)
            continue;

        int dist = past_len - i;
        if (dist <= sustain_range)
            factors[tid - range_min] = 1.0f;
        else
        {
            float f = MAX(0.0f, 1.0f - ((float)dist - (float)sustain_range) / (float)decay_range);
            shmemAtomicMaxF(factors + tid - range_min, f);
        }
    }
    __syncthreads();

    // Apply penalties to range
    for (int i = threadIdx.x; i < BLOCK_VOCAB_SPAN; i += NUM_THREADS)
    {
        if (i + range_min >= vocab_size)
            break;

        float v;
        if constexpr (input_fp16)
            v = __half2float(((half*) in_logits)[i + range_min]);
        else
            v = ((float*) in_logits)[i + range_min];

        float w = v > 0.0f ? v / rep_p : v * rep_p;
        float f = factors[i] + 1e-30;
        float f1 = (1.0f - f) + 1e-30;
        float o = v * f1 + w * f;
        out_logits[i + range_min] = o;
    }
}

void apply_rep_pens
(
    const at::Tensor& in_logits,
    const at::Tensor& out_logits,
    const at::Tensor& past_ids,
    float rep_p,
    int sustain_range,
    int decay_range
)
{
    const at::cuda::OptionalCUDAGuard device_guard(in_logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(out_logits, kFloat);
    TORCH_CHECK_DTYPE(past_ids, kLong);
    TORCH_CHECK(in_logits.size(0) == 1, "rep. penalties only implemented for bsz 1");  // TODO
    TORCH_CHECK_SHAPES(past_ids, 0, in_logits, 0, 1);
    TORCH_CHECK_SHAPES_FULL(in_logits, out_logits);

    int past_len = past_ids.size(1);
    int vocab_size = in_logits.size(1);
    int num_blocks = CEIL_DIVIDE(vocab_size, BLOCK_VOCAB_SPAN);

    #define kernel_args \
        (const void*) in_logits.data_ptr(), \
        (float*) out_logits.data_ptr(), \
        (const uint64_t*) past_ids.data_ptr(), \
        past_len, \
        vocab_size, \
        rep_p, \
        sustain_range, \
        decay_range

    if (in_logits.dtype() == at::kHalf)
        apply_rep_pens_kernel<true><<<num_blocks, NUM_THREADS, 0, stream>>>(kernel_args);
    else
        apply_rep_pens_kernel<false><<<num_blocks, NUM_THREADS, 0, stream>>>(kernel_args);

    #undef kernel_args

    cuda_check(cudaPeekAtLastError());
}


template <bool input_fp16>
__global__ __launch_bounds__(NUM_THREADS)
void apply_pres_freq_pens_kernel
(
    const void* __restrict__ in_logits,
    float* __restrict__ out_logits,
    const uint64_t* __restrict__ past_ids,
    int past_len,
    int vocab_size,
    float pres_p,
    float freq_p,
    int sustain_range,
    int decay_range
)
{
    // Each block processes a range of the logits
    int range_min = blockIdx.x * BLOCK_VOCAB_SPAN;
    int range_max = MIN(range_min + BLOCK_VOCAB_SPAN, vocab_size);

    __shared__ float frequency[BLOCK_VOCAB_SPAN];
    __shared__ float presence[BLOCK_VOCAB_SPAN];
    for (int i = threadIdx.x; i < BLOCK_VOCAB_SPAN; i += NUM_THREADS)
    {
        frequency[i] = 0.0f;
        presence[i] = 0.0f;
    }
    __syncthreads();

    // Record which tokens from the range appear in past_ids and
    for (int i = threadIdx.x; i < past_len; i += NUM_THREADS)
    {
        if (i < past_len - sustain_range - decay_range)
            continue;

        int tid = (int) past_ids[i];
        if (tid < range_min || tid >= range_max)
            continue;

        int dist = past_len - i;
        float pen = MIN(1.0f, MAX(0.0f, 1.0f - ((float)dist - (float)sustain_range) / (float)decay_range));
        atomicAdd(frequency + tid - range_min, pen * freq_p);
        shmemAtomicMaxF(presence + tid - range_min, pen * pres_p);
    }
    __syncthreads();

    // Apply frequency to range
    for (int i = threadIdx.x; i < BLOCK_VOCAB_SPAN; i += NUM_THREADS)
    {
        if (i + range_min >= vocab_size)
            break;

        float v;
        if constexpr (input_fp16)
            v = __half2float(((half*) in_logits)[i + range_min]);
        else
            v = ((float*) in_logits)[i + range_min];

        v -= frequency[i];
        v -= presence[i];
        out_logits[i + range_min] = v;
    }
}

void apply_pres_freq_pens
(
    const at::Tensor& in_logits,
    const at::Tensor& out_logits,
    const at::Tensor& past_ids,
    float pres_p,
    float freq_p,
    int sustain_range,
    int decay_range
)
{
    const at::cuda::OptionalCUDAGuard device_guard(in_logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(out_logits, kFloat);
    TORCH_CHECK_DTYPE(past_ids, kLong);
    TORCH_CHECK(in_logits.size(0) == 1, "rep. penalties only implemented for bsz 1");  // TODO
    TORCH_CHECK_SHAPES(past_ids, 0, in_logits, 0, 1);
    TORCH_CHECK_SHAPES_FULL(in_logits, out_logits);

    int past_len = past_ids.size(1);
    int vocab_size = in_logits.size(1);
    int num_blocks = CEIL_DIVIDE(vocab_size, BLOCK_VOCAB_SPAN);

    #define kernel_args \
        (const void*) in_logits.data_ptr(), \
        (float*) out_logits.data_ptr(), \
        (const uint64_t*) past_ids.data_ptr(), \
        past_len, \
        vocab_size, \
        pres_p, \
        freq_p, \
        sustain_range, \
        decay_range

    if (in_logits.dtype() == at::kHalf)
        apply_pres_freq_pens_kernel<true><<<num_blocks, NUM_THREADS, 0, stream>>>(kernel_args);
    else
        apply_pres_freq_pens_kernel<false><<<num_blocks, NUM_THREADS, 0, stream>>>(kernel_args);

    #undef kernel_args

    cuda_check(cudaPeekAtLastError());
}