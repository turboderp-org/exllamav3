#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "dequant_blocks.cuh"

// A block covers a 32 (out) x 64 (in) tile and transposes it through shared memory, so the bytes are
// read along the checkpoint's contiguous dimension and written along the output's

#define TILE_O 32
#define TILE_I 64

#define FMT_E4M3 0
#define FMT_E2M1 1
#define FMT_E5M2 2

__device__ __forceinline__ float e2m1_to_float(uint32_t nib)
{
    const float lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    float v = lut[nib & 7];
    return (nib & 8) ? -v : v;
}

__device__ __forceinline__ float e4m3_to_float(uint32_t b)
{
    // e4m3fn: bias 7, no infinities, 0x7f / 0xff are NaN
    uint32_t sign = b >> 7, exp = (b >> 3) & 0xf, man = b & 7;
    float v;
    if (exp == 0) v = ldexpf((float) man, -9);
    else if (exp == 15 && man == 7) v = __int_as_float(0x7fc00000);
    else v = ldexpf(1.0f + (float) man / 8.0f, (int) exp - 7);
    return sign ? -v : v;
}

__device__ __forceinline__ float e5m2_to_float(uint32_t b)
{
    // e5m2: bias 15, IEEE-style infinities and NaNs
    uint32_t sign = b >> 7, exp = (b >> 2) & 0x1f, man = b & 3;
    float v;
    if (exp == 0) v = ldexpf((float) man, -16);
    else if (exp == 31) v = man ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000);
    else v = ldexpf(1.0f + (float) man / 4.0f, (int) exp - 15);
    return sign ? -v : v;
}

template <int FMT, bool SCALE_E8M0, bool HAS_SCALE, bool SRC_T>
__global__ __launch_bounds__(256)
void dequant_blocks_kernel
(
    const uint8_t* __restrict__ packed,
    const void* __restrict__ scale,
    half* __restrict__ out,
    const int rows,           // out features of this segment
    const int cols,           // in features
    const int packed_stride,  // bytes per packed row (row = out feature, or in feature when SRC_T)
    const int out_stride,     // elements per output row
    const int col0,
    const int sc,             // scale columns
    const int br,             // out features per scale row
    const int bc,             // in features per scale column
    const float global_scale
)
{
    __shared__ half tile[TILE_I][TILE_O + 1];

    const int o0 = blockIdx.y * TILE_O;
    const int i0 = blockIdx.x * TILE_I;
    const int tx = threadIdx.x;   // 0..31
    const int ty = threadIdx.y;   // 0..7

    auto decode = [&] (int o, int i) -> float
    {
        uint32_t b;
        if constexpr (FMT == FMT_E2M1)
        {
            b = packed[o * packed_stride + (i >> 1)];
            b = (i & 1) ? (b >> 4) : (b & 0xf);
            return e2m1_to_float(b);
        }
        else
        {
            b = SRC_T ? packed[i * packed_stride + o] : packed[o * packed_stride + i];
            return FMT == FMT_E4M3 ? e4m3_to_float(b) : e5m2_to_float(b);
        }
    };

    auto block_scale = [&] (int o, int i) -> float
    {
        if constexpr (!HAS_SCALE) return 1.0f;
        const int idx = (o / br) * sc + (i / bc);
        if constexpr (SCALE_E8M0) return exp2f((float) ((const uint8_t*) scale)[idx] - 127.0f);
        else return ((const float*) scale)[idx];
    };

    auto round = [&] (float v) -> half
    {
        half h = __float2half(v);
        if (global_scale != 1.0f) h = __float2half(__half2float(h) * global_scale);
        return h;
    };

    for (int r = ty; r < TILE_O; r += 8)
    {
        const int o = o0 + r;
        const int i = i0 + tx * 2;
        half h0 = __float2half(0.0f), h1 = h0;
        if (o < rows && i < cols)
        {
            h0 = round(decode(o, i) * block_scale(o, i));
            if (i + 1 < cols) h1 = round(decode(o, i + 1) * block_scale(o, i + 1));
        }
        tile[tx * 2][r] = h0;
        tile[tx * 2 + 1][r] = h1;
    }
    __syncthreads();

    for (int c = ty; c < TILE_I; c += 8)
    {
        const int i = i0 + c;
        const int o = o0 + tx;
        if (i < cols && o < rows)
            out[(size_t) i * out_stride + col0 + o] = tile[c][tx];
    }
}

void dequant_blocks
(
    const at::Tensor& packed,
    const c10::optional<at::Tensor>& scale,
    at::Tensor out,
    int64_t fmt,
    double global_scale,
    int64_t col0,
    bool src_transposed,
    int64_t block_rows,
    int64_t block_cols
)
{
    const at::cuda::OptionalCUDAGuard device_guard(packed.device());
    TORCH_CHECK_DTYPE(packed, kByte);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK(packed.dim() == 2 && out.dim() == 2, "dequant_blocks: 2D tensors expected");
    TORCH_CHECK(packed.is_contiguous() && out.is_contiguous(), "dequant_blocks: contiguous tensors expected");
    TORCH_CHECK(fmt >= 0 && fmt <= 2, "dequant_blocks: unknown format");
    TORCH_CHECK(!src_transposed || fmt != FMT_E2M1, "dequant_blocks: transposed source is fp8 only");
    const bool fp4 = fmt == FMT_E2M1;
    int rows, cols;
    if (src_transposed) { cols = packed.size(0); rows = packed.size(1); }
    else { rows = packed.size(0); cols = fp4 ? packed.size(1) * 2 : packed.size(1); }
    TORCH_CHECK(out.size(1) >= col0 + rows && out.size(0) >= cols, "dequant_blocks: output too small");

    bool has_scale = scale.has_value();
    bool e8m0 = false;
    int sc = 1, br = rows, bc = cols;
    const void* scale_ptr = nullptr;
    if (has_scale)
    {
        const at::Tensor& s = scale.value();
        TORCH_CHECK(s.is_contiguous(), "dequant_blocks: scale must be contiguous");
        TORCH_CHECK(s.dtype() == at::kByte || s.dtype() == at::kFloat, "dequant_blocks: scale must be uint8 (E8M0) or float32");
        e8m0 = s.dtype() == at::kByte;
        int sr = s.dim() == 0 ? 1 : (int) s.size(0);
        sc = s.dim() < 2 ? 1 : (int) s.size(1);
        // Block sizes come from the format (128 x 128 for weight_scale_inv grids, whose last block may be
        // partial) or, when not given, from an exact division of the weight by the grid
        if (block_rows > 0) br = (int) block_rows;
        else { TORCH_CHECK(rows % sr == 0, "dequant_blocks: rows not divisible by the scale grid"); br = rows / sr; }
        if (block_cols > 0) bc = (int) block_cols;
        else { TORCH_CHECK(cols % sc == 0, "dequant_blocks: cols not divisible by the scale grid"); bc = cols / sc; }
        TORCH_CHECK((int64_t) br * sr >= rows && (int64_t) bc * sc >= cols, "dequant_blocks: scale grid does not cover the weight");
        TORCH_CHECK(!fp4 || (bc % 2 == 0), "dequant_blocks: fp4 scale blocks must cover whole bytes");
        scale_ptr = s.data_ptr();
    }

    dim3 blocks((cols + TILE_I - 1) / TILE_I, (rows + TILE_O - 1) / TILE_O);
    dim3 threads(32, 8);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const uint8_t* p = (const uint8_t*) packed.data_ptr();
    half* o = (half*) out.data_ptr();
    const int pstride = (int) packed.size(1);
    const int ostride = (int) out.size(1);
    const float gs = (float) global_scale;

    #define LAUNCH(F, E, H, T) dequant_blocks_kernel<F, E, H, T><<<blocks, threads, 0, stream>>>( \
        p, scale_ptr, o, rows, cols, pstride, ostride, (int) col0, sc, br, bc, gs)
    #define LAUNCH_FMT(F) \
        if (!has_scale) { if (src_transposed) LAUNCH(F, false, false, true); else LAUNCH(F, false, false, false); } \
        else if (e8m0) { if (src_transposed) LAUNCH(F, true, true, true); else LAUNCH(F, true, true, false); } \
        else { if (src_transposed) LAUNCH(F, false, true, true); else LAUNCH(F, false, true, false); }
    if (fmt == FMT_E4M3) { LAUNCH_FMT(FMT_E4M3) }
    else if (fmt == FMT_E5M2) { LAUNCH_FMT(FMT_E5M2) }
    else
    {
        if (!has_scale) LAUNCH(FMT_E2M1, false, false, false);
        else if (e8m0) LAUNCH(FMT_E2M1, true, true, false);
        else LAUNCH(FMT_E2M1, false, true, false);
    }
    #undef LAUNCH_FMT
    #undef LAUNCH
    cuda_check(cudaPeekAtLastError());
}
