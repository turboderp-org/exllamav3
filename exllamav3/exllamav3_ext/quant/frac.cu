#include <cuda_fp16.h>
#include "frac.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"

// Fractional-rate trellis tiles. The 256-weight tile is a ring of 16 * K bits: weight i's D(i) new bits,
// D(i) = KA + bit (i mod 16) of MASK, sit at ring position S(i) - D(i) .. S(i) where S(i) is the prefix sum
// of D; its 16-bit window is ring bits [S(i) - 16, S(i)). Bits are stored MSB-first in 32-bit words, the
// same stream convention as the integer pack_trellis output, so a tile is 16 * K uint16 = 2 * K uint32
// words (K half-integer -> whole words). One warp per tile; lane t owns ring positions 8t .. 8t+7 (the
// tensor-core element order the quantizer works in).

__device__ __forceinline__ int frac_d(int i, int ka, uint32_t mask) { return ka + ((mask >> (i & 15)) & 1); }

// Ring position where window i ends, S(i)
__device__ __forceinline__ int frac_s(int i, int ka, uint32_t mask, int bpb)
{
    int s = (i >> 4) * bpb;
    for (int j = 0; j <= (i & 15); ++j) s += frac_d(j, ka, mask);
    return s;
}

// 16-bit window starting at ring bit `start` (mod R), from MSB-first 32-bit words
__device__ __forceinline__ uint32_t frac_window(const uint32_t* words, int nw, int start)
{
    const int R = nw * 32;
    start %= R; if (start < 0) start += R;
    const int wi = start >> 5, o = start & 31;
    const uint64_t v = ((uint64_t) words[wi] << 32) | (uint64_t) words[(wi + 1) % nw];
    return (uint32_t) ((v >> (48 - o)) & 0xFFFFu);
}

// Row-major tile element of ring position i (see tensor_core_perm in exl3_lib/quantize.py)
__device__ __forceinline__ int frac_perm(int i)
{
    const int t = i >> 3, j = i & 7;
    const int r0 = (t & 3) * 2, c0 = t >> 2;
    const int r = r0 + ((j & 1) ? 1 : 0) + ((j & 2) ? 8 : 0);
    const int c = c0 + ((j & 4) ? 8 : 0);
    return r * 16 + c;
}

__global__ __launch_bounds__(128)
void pack_trellis_frac_kernel(uint16_t* __restrict__ g_packed, const uint16_t* __restrict__ g_unpacked,
                              int num_tiles, int ka, uint32_t mask, int bpb)
{
    const int tile = blockIdx.x * 128 + threadIdx.x;
    if (tile >= num_tiles) return;
    const int nw = bpb / 2;
    uint32_t words[32];
    for (int w = 0; w < nw; ++w) words[w] = 0;
    const uint16_t* idx = g_unpacked + (size_t) tile * 256;
    int pos = 0;
    for (int i = 0; i < 256; ++i)
    {
        const int d = frac_d(i, ka, mask);
        const uint32_t v = (uint32_t) idx[i] & ((1u << d) - 1);
        for (int b = d - 1; b >= 0; --b, ++pos)
            if ((v >> b) & 1) words[pos >> 5] |= 1u << (31 - (pos & 31));
    }
    uint32_t* out = (uint32_t*) (g_packed + (size_t) tile * bpb);
    for (int w = 0; w < nw; ++w) out[w] = words[w];
}

__global__ __launch_bounds__(32)
void unpack_trellis_frac_kernel(uint16_t* __restrict__ g_unpacked, const uint16_t* __restrict__ g_packed,
                                int ka, uint32_t mask, int bpb)
{
    __shared__ uint32_t words[32];
    const int tile = blockIdx.x;
    const int t = threadIdx.x;
    const int nw = bpb / 2;
    if (t < nw) words[t] = ((const uint32_t*) (g_packed + (size_t) tile * bpb))[t];
    __syncwarp();
    for (int j = 0; j < 8; ++j)
    {
        const int i = t * 8 + j;
        g_unpacked[(size_t) tile * 256 + i] = (uint16_t) frac_window(words, nw, frac_s(i, ka, mask, bpb) - 16);
    }
}

static int frac_bpb(int KA, int64_t MASK)
{
    TORCH_CHECK(KA >= 1 && KA <= 7 && MASK >= 0 && MASK <= 0xFFFF, "frac: KA must be 1..7, MASK 16 bits");
    // Extra-bit positions among the 16. Counted by hand: __builtin_popcount is GCC/Clang only
    int extra = 0;
    for (int64_t m = MASK; m; m &= m - 1) extra++;
    const int bpb = 16 * KA + extra;
    TORCH_CHECK(bpb % 2 == 0, "frac: bits per 16 weights must be even (whole 32-bit words per tile)");
    return bpb;
}

void pack_trellis_frac(at::Tensor packed, at::Tensor unpacked, int KA, int64_t MASK)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int bpb = frac_bpb(KA, MASK);
    TORCH_CHECK_DIM(packed, 3); TORCH_CHECK_DIM(unpacked, 3);
    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1); TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256); TORCH_CHECK_SIZE(packed, 2, bpb);
    TORCH_CHECK(packed.is_contiguous() && unpacked.is_contiguous(), "frac: contiguous");
    const int num_tiles = packed.size(0) * packed.size(1);
    if (!num_tiles) return;
    pack_trellis_frac_kernel<<<(num_tiles + 127) / 128, 128, 0, stream>>>
        ((uint16_t*) packed.data_ptr(), (const uint16_t*) unpacked.data_ptr(), num_tiles, KA, (uint32_t) MASK, bpb);
    cuda_check(cudaPeekAtLastError());
}

void unpack_trellis_frac(at::Tensor unpacked, at::Tensor packed, int KA, int64_t MASK)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int bpb = frac_bpb(KA, MASK);
    TORCH_CHECK_DIM(packed, 3); TORCH_CHECK_DIM(unpacked, 3);
    TORCH_CHECK_SHAPES(packed, 0, unpacked, 0, 1); TORCH_CHECK_SHAPES(packed, 1, unpacked, 1, 1);
    TORCH_CHECK_SIZE(unpacked, 2, 256); TORCH_CHECK_SIZE(packed, 2, bpb);
    TORCH_CHECK(packed.is_contiguous() && unpacked.is_contiguous(), "frac: contiguous");
    const int num_tiles = packed.size(0) * packed.size(1);
    if (!num_tiles) return;
    unpack_trellis_frac_kernel<<<num_tiles, 32, 0, stream>>>
        ((uint16_t*) unpacked.data_ptr(), (const uint16_t*) packed.data_ptr(), KA, (uint32_t) MASK, bpb);
    cuda_check(cudaPeekAtLastError());
}
