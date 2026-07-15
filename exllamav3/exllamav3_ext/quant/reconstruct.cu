#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "reconstruct.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include <float.h>

template <int K, int cb>
__global__ __launch_bounds__(256)
void reconstruct_kernel
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;  // in uint16s

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int k = blockIdx.y;
    int n = blockIdx.x * 8;
    int tiles_n = gridDim.x;
    int out_blocks_n = tiles_n * 8;

    // Load packed 16*128 tile
    __shared__ uint32_t s_packed[8][packed_size / 2];
    g_packed += (k * packed_blocks_n + packed_n_offset + n) * packed_size;
    if (t < packed_size)
        ((int4*) s_packed)[t] = ((int4*) g_packed)[t];
    __syncthreads();

    // Dequant
    register FragB frag[2];
    dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

    // Shuffle from tensor core layout to row major tile
//    __shared__ half tile[16 * 8 * 16];
    __shared__ half2 tile[16][8][8];

    half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
    half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
    half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
    half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

    if (!(lane_id & 4))
    {
        half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
        half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
        half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
        half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
        half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
        half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
        half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
        half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
        int r0 = (lane_id % 4) * 2;
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r0 + 9;
        int c0 = lane_id / 8;
        int c1 = c0 + 4;
        tile[r0][warp_id][c0] = m0;
        tile[r1][warp_id][c0] = m1;
        tile[r2][warp_id][c0] = m2;
        tile[r3][warp_id][c0] = m3;
        tile[r0][warp_id][c1] = m4;
        tile[r1][warp_id][c1] = m5;
        tile[r2][warp_id][c1] = m6;
        tile[r3][warp_id][c1] = m7;
    }
    __syncthreads();

    // Store unpacked tile
    int r = t / 16;
    int c = t % 16;
    int4* tile_int4 = (reinterpret_cast<int4*> (tile));
    int4* out_int4 = ((int4*) g_unpacked) + (k * 16 + r) * 2 * out_blocks_n + n * 2 + c;
    *out_int4 = tile_int4[t];
}

#define __(i, cb) reconstruct_kernel<i, cb>
constexpr auto reconstruct_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

/*
Reconstruct encoded+packed tensor
*/
void reconstruct_slice
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1,
    int64_t n_offset
)
{
    const at::cuda::OptionalCUDAGuard device_guard(unpacked.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_SHAPES(unpacked, 0, packed, 0, 16);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);
    TORCH_CHECK_DTYPE(unpacked, kHalf);

    int rows = packed.size(0);
    int packed_cols = packed.size(1);
    TORCH_CHECK(unpacked.size(1) % 128 == 0, "unpacked N dimension must be divisible by 128");
    TORCH_CHECK(n_offset % 128 == 0, "n_offset must be divisible by 128");
    TORCH_CHECK(n_offset >= 0, "n_offset must be non-negative");
    TORCH_CHECK(n_offset + unpacked.size(1) <= packed.size(1) * 16, "reconstruct slice exceeds packed tensor bounds");

    int cols = unpacked.size(1) / 16;
    int packed_n_offset = n_offset / 16;

    dim3 blockDim(256);
    dim3 gridDim(cols / 8, rows);

    int cbi = K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;

    reconstruct_kernel_instances[cbi]<<<gridDim, blockDim, 0, stream>>>
    (
        (half*) unpacked.data_ptr(),
        (const uint16_t*) packed.data_ptr(),
        packed_cols,
        packed_n_offset
    );
    cuda_check(cudaPeekAtLastError());
}

template <int K, int cb>
__global__ __launch_bounds__(256)
void reconstruct_fp8dg_nt_kernel
(
    uint8_t* __restrict__ g_q_nt,
    float* __restrict__ g_scales,
    const uint16_t* __restrict__ g_packed,
    int packed_blocks_n,
    int rows_k,
    int cols_n
)
{
    constexpr int packed_size = 256 * K / 16;  // in uint16s
    constexpr float fp8_max = 448.0f;          // torch.float8_e4m3fn finfo.max

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int n128 = blockIdx.x;  // original N block; q_nt row block
    int k128 = blockIdx.y;  // original K block; q_nt col block

    __shared__ uint32_t s_packed[8][packed_size / 2];
    __shared__ half2 tile[16][8][8];
    __shared__ int4 sh_tile_i4[128 * 16];  // 128 rows * (128 half / 8 half per int4)
    __shared__ float sh_amax[256];

    int4* sh_i4 = sh_tile_i4;
    half* sh_half = reinterpret_cast<half*>(sh_tile_i4);

    #pragma unroll
    for (int sub = 0; sub < 8; ++sub)
    {
        int k16 = k128 * 8 + sub;
        int n16 = n128 * 8;
        const uint16_t* packed_tile =
            g_packed + (k16 * packed_blocks_n + n16) * packed_size;

        if (t < packed_size)
            ((int4*) s_packed)[t] = ((const int4*) packed_tile)[t];
        __syncthreads();

        register FragB frag[2];
        dq_dispatch<K, cb>(s_packed[warp_id], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int c1 = c0 + 4;
            tile[r0][warp_id][c0] = m0;
            tile[r1][warp_id][c0] = m1;
            tile[r2][warp_id][c0] = m2;
            tile[r3][warp_id][c0] = m3;
            tile[r0][warp_id][c1] = m4;
            tile[r1][warp_id][c1] = m5;
            tile[r2][warp_id][c1] = m6;
            tile[r3][warp_id][c1] = m7;
        }
        __syncthreads();

        int r = t / 16;
        int c = t % 16;
        int4* tile_int4 = reinterpret_cast<int4*>(tile);
        sh_i4[(sub * 16 + r) * 16 + c] = tile_int4[t];
        __syncthreads();
    }

    float local_amax = 0.0f;
    for (int i = t; i < 128 * 128; i += blockDim.x)
        local_amax = fmaxf(local_amax, fabsf(__half2float(sh_half[i])));
    sh_amax[t] = local_amax;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1)
    {
        if (t < stride)
            sh_amax[t] = fmaxf(sh_amax[t], sh_amax[t + stride]);
        __syncthreads();
    }

    float scale = fmaxf(sh_amax[0], 1.0e-10f) / fp8_max;
    if (t == 0)
        g_scales[n128 * (rows_k / 128) + k128] = scale;
    float inv_scale = 1.0f / scale;

    for (int i = t; i < 128 * 128; i += blockDim.x)
    {
        int kr = i / 128;
        int nc = i - kr * 128;
        int q_row = n128 * 128 + nc;
        int q_col = k128 * 128 + kr;
        float v = __half2float(sh_half[i]) * inv_scale;
        v = fmaxf(-fp8_max, fminf(fp8_max, v));
        g_q_nt[q_row * rows_k + q_col] =
            __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}

#define __(i, cb) reconstruct_fp8dg_nt_kernel<i, cb>
constexpr auto reconstruct_fp8dg_nt_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

/*
Reconstruct encoded+packed tensor directly to DeepGEMM NT FP8 weight layout.
packed represents a row-major [K, N] FP16 matrix. q_nt is [N, K] FP8 and
scales is [N / 128, K / 128] FP32, one scale per DeepGEMM 128x128 tile.
*/
void reconstruct_fp8dg_nt
(
    at::Tensor q_nt,
    at::Tensor scales,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q_nt.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(q_nt, 2);
    TORCH_CHECK_DIM(scales, 2);
    TORCH_CHECK_DIM(packed, 3);
    TORCH_CHECK(q_nt.scalar_type() == c10::ScalarType::Float8_e4m3fn,
                "q_nt must be torch.float8_e4m3fn");
    TORCH_CHECK_DTYPE(scales, kFloat);
    TORCH_CHECK_DTYPE(packed, kShort);
    TORCH_CHECK_SIZE(packed, 2, 256 * K / 16);

    int rows_k = packed.size(0) * 16;
    int cols_n = packed.size(1) * 16;
    TORCH_CHECK(rows_k % 128 == 0, "packed K dimension must reconstruct to a multiple of 128");
    TORCH_CHECK(cols_n % 128 == 0, "packed N dimension must reconstruct to a multiple of 128");
    TORCH_CHECK(q_nt.size(0) == cols_n && q_nt.size(1) == rows_k,
                "q_nt must have shape [N, K] for reconstructed packed [K, N]");
    TORCH_CHECK(scales.size(0) == cols_n / 128 && scales.size(1) == rows_k / 128,
                "scales must have shape [N / 128, K / 128]");
    TORCH_CHECK(q_nt.is_contiguous(), "q_nt must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(packed.is_contiguous(), "packed must be contiguous");

    dim3 blockDim(256);
    dim3 gridDim(cols_n / 128, rows_k / 128);

    int cbi = K - 1;
    if (mcg) cbi += 8;
    else if (mul1) cbi += 16;

    reconstruct_fp8dg_nt_kernel_instances[cbi]<<<gridDim, blockDim, 0, stream>>>
    (
        reinterpret_cast<uint8_t*>(q_nt.data_ptr()),
        scales.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(packed.data_ptr()),
        packed.size(1),
        rows_k,
        cols_n
    );
    cuda_check(cudaPeekAtLastError());
}

void reconstruct
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K,
    bool mcg,
    bool mul1
)
{
    TORCH_CHECK_SHAPES(unpacked, 1, packed, 1, 16);
    reconstruct_slice(unpacked, packed, K, mcg, mul1, 0);
}
