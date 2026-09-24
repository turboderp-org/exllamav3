// Tile-level isolation test: one warp, one 16x16 tile, K=4.
// Runs the EXACT sm70 GEMV dataflow (decode → shuffle → MMA → D store)
// for a single tile and writes the pre-had D values; compared
// host-side against reconstruct + fp32 GEMM restricted to the tile.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "../exllamav3/exllamav3_ext/util.cuh"
#define CEIL_DIVIDE(a, b) ((a) + (b) - 1) / (b)
#include "../exllamav3/exllamav3_ext/quant/exl3_kernel_map.cuh"
#include "../exllamav3/exllamav3_ext/ptx.cuh"
#include "../exllamav3/exllamav3_ext/quant/codebook.cuh"
#include "../exllamav3/exllamav3_ext/quant/exl3_dq.cuh"
#include "../exllamav3/exllamav3_ext/quant/exl3_sm70_map.cuh"
#include "../exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh"

__global__ void tile_test_kernel(const uint32_t* B, const half* x, float* y)
{
    const int lane = threadIdx.x & 31;
    constexpr int TWORDS = 32;  // K=4: 8*4 uint32 per 16x16 tile

    // decode: lane Ls reads word Ls; 8 codes w0..w7
    const uint32_t b = B[lane];
    const uint32_t a = B[(lane + 31) & 31];
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(s) : "r"(b), "r"(a), "n"(20));
    w7 = b & 0xffff;
    w6 = (b >> 4) & 0xffff;
    w5 = (b >> 8) & 0xffff;
    w4 = (b >> 12) & 0xffff;
    w3 = (b >> 16) & 0xffff;
    w2 = s & 0xffff;
    w1 = (s >> 4) & 0xffff;
    w0 = (s >> 8) & 0xffff;
    (void) w0; (void) w1; (void) w2; (void) w3;
    (void) w4; (void) w5; (void) w6; (void) w7;

    // decode via the extension's dq8 (K=4, cb=0)
    uint32_t aw = a;  // (lane + 31) & 31 word
    FragB f0, f1;
    exl3_gemv_ns::dq8_regs_4bits<0>(aw, b, f0, f1);
    uint32_t v[4];
    v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]);
    v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
    v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]);
    v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);

    // software quad loop: 2 mh x 4 qk
    float acc[2][8] = {};
    #pragma unroll
    for (int mh = 0; mh < 2; ++mh)
    #pragma unroll
    for (int qk = 0; qk < 4; ++qk)
    {
        // A fragment: x[k] for k = 4*qk + 0..3 (row 0, m=1)
        const uint32_t a0 = *(const uint32_t*)(x + 4 * qk);
        const uint32_t a1 = *(const uint32_t*)(x + 4 * qk + 2);

        // B fragment via the validated shuffle map
        uint32_t wv[4];
        #pragma unroll
        for (int vi = 0; vi < 4; ++vi)
        {
            const int src = ((2 * qk + (vi >> 1)) & 3)
                + (((lane & 3) + 4 * mh) & 7) * 4;
            const uint32_t pair = __shfl_sync(0xffffffffu,
                v[((qk >> 1) & 1)], src);
            wv[vi] = (vi & 1) ? (pair >> 16) : (pair & 0xffff);
        }
        const uint32_t b0 = wv[0] | (wv[1] << 16);
        const uint32_t b1 = wv[2] | (wv[3] << 16);
        exl3_sm70::mma_m8n8k4_rc_f32(a0, a1, b0, b1, acc[mh]);
    }

    // D store: y[row * 16 + mh*8 + col] per the C map
    for (int mh = 0; mh < 2; ++mh)
        for (int reg = 0; reg < 8; ++reg)
        {
            const int m_row = exl3_sm70::c_row(lane, reg);
            const int n_col = mh * 8 + exl3_sm70::c_col(lane, reg);
            if (m_row == 0)
                y[n_col] = acc[mh][reg];
        }
}

void tile_test(torch::Tensor B, torch::Tensor x, torch::Tensor y)
{
    tile_test_kernel<<<1, 32>>>(
        (const uint32_t*)B.data_ptr(), (const half*)x.data_ptr(), (float*)y.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tile_test", &tile_test, "tile_test");
}