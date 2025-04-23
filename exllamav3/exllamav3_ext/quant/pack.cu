#include "quantize.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"

template <int K>
__global__ __launch_bounds__(128)
void pack_trellis_kernel
(
    uint16_t* __restrict__ g_packed,
    const uint16_t* __restrict__ g_unpacked
)
{
    constexpr int packed_size = 256 * K / 16;
    __shared__ uint16_t s_unpacked[256];
    __shared__ uint16_t s_packed[packed_size];

    int t = threadIdx.x;
    g_packed += (gridDim.x * blockIdx.y + blockIdx.x) * packed_size;
    g_unpacked += (gridDim.x * blockIdx.y + blockIdx.x) * 256;

    ((uint32_t*) s_unpacked)[t] = ((uint32_t*) g_unpacked)[t];
    __syncthreads();

    // 16 spans of 16 weights to guarantee alignment for any K
    const int spans = 16;
    const int len = 256 / spans;
    if (t < spans)
    {
        int i = len * t;
        int j = K * t;
        int k = 32;
        uint32_t buf = 0;
        for (int n = 0; n < len; ++n)
        {
            uint32_t v = (uint32_t) s_unpacked[i];
            v &= ((1 << K) - 1);
            k -= K;
            buf |= (v << k);
            if (k <= 16)
            {
                s_packed[j] = (uint16_t) (buf >> 16);
                buf <<= 16;
                k += 16;
                j++;
            }
            i++;
        }
    }
    __syncthreads();

    if (t < packed_size / 2)
        ((uint32_t*) g_packed)[t] = SWAP16(((uint32_t*) s_packed)[t]);;
}

#define __(i) pack_trellis_kernel<i>
constexpr auto pack_trellis_kernel_instances = std::array
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

void pack_trellis
(
    at::Tensor packed,
    at::Tensor unpacked,
    int K
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1);
    TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);

    int rows = packed.size(0);
    int cols = packed.size(1);

    dim3 blockDim(128);
    dim3 gridDim(rows, cols);

    pack_trellis_kernel_instances[K - 1]<<<gridDim, blockDim, 0, stream>>>
    (
        (uint16_t*) packed.data_ptr(),
        (const uint16_t*) unpacked.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

template <int K>
__global__ __launch_bounds__(128)
void unpack_trellis_kernel
(
    uint16_t* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed
)
{
    constexpr int packed_size = 256 * K / 16;
    __shared__ uint16_t s_packed[packed_size];

    int t = threadIdx.x;
    g_packed += (gridDim.x * blockIdx.y + blockIdx.x) * packed_size;
    g_unpacked += (gridDim.x * blockIdx.y + blockIdx.x) * 256;

    // Read packed tile
    if (t < packed_size / 2)
        ((uint32_t*) s_packed)[t] = ((uint32_t*) g_packed)[t];
    __syncthreads();

    // Index two words
    int b0 = t * 2 * K + K - 16 + 256 * K;          // start of word0
    int b1 = b0 + K;                                // start of word1
    int b2 = b1 + 16;                               // end of word1
    int i0 = b0 / 32;                               // uint32 containing first bit of word0
    int i1 = (b2 - 1) / 32;                         // uint32 containing last bit of word1, may be == i0
    int s1 = (i1 + 1) * 32 - b2;                    // shift to align word1 to 32-bit boundary

    // Load 32-64 bits containing word0 and word1, overlapping by 16-K bits, correct for endianness
    uint32_t a = ((uint32_t*) s_packed)[i0 % (K * 256 / 32)];
    uint32_t b = ((uint32_t*) s_packed)[i1 % (K * 256 / 32)];
//    a = SWAP16(a);
//    b = SWAP16(b);

    // Shift into place
    uint32_t w1 = __funnelshift_r(b, a, s1);
    uint32_t w0 = w1 >> K;
    w0 &= 0xffff;
    w1 &= 0xffff;

    // Store
    uint32_t word01 = (w1 << 16) | w0;
    ((uint32_t*)g_unpacked)[t] = word01;
}

#define __(i) unpack_trellis_kernel<i>
constexpr auto unpack_trellis_kernel_instances = std::array
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

void unpack_trellis
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1);
    TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);

    int rows = packed.size(0);
    int cols = packed.size(1);

    dim3 blockDim(128);
    dim3 gridDim(cols, rows);

    unpack_trellis_kernel_instances[K - 1]<<<gridDim, blockDim, 0, stream>>>
    (
        (uint16_t*) unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr()
    );
    cuda_check(cudaPeekAtLastError());
}

__global__ __launch_bounds__(32)
void pack_signs_kernel
(
    uint16_t* __restrict__ g_packed,
    const uint16_t* __restrict__ g_unpacked,
    int cols
)
{
    int t = threadIdx.x;
    int idx = 32 * blockIdx.x + t;
    if (idx >= cols) return;
    g_unpacked += 16 * idx;
    g_packed += idx;

    // Not efficient but whatever
    uint16_t out = 0;
    for (int i = 0; i < 16; ++i)
    {
        uint16_t v = *g_unpacked++;
        v &= 0x8000;
        out >>= 1;
        out |= v;
    }

    *g_packed = out;
}

void pack_signs
(
    at::Tensor packed,
    at::Tensor unpacked
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(unpacked, kHalf);
    TORCH_CHECK_DTYPE(packed, kShort);

    int cols = packed.size(0);
    dim3 blockDim(32);
    dim3 gridDim(CEIL_DIVIDE(cols, 32));

    pack_signs_kernel<<<gridDim, blockDim, 0, stream>>>
    (
        (uint16_t*) packed.data_ptr(),
        (const uint16_t*) unpacked.data_ptr(),
        cols
    );
    cuda_check(cudaPeekAtLastError());
}

