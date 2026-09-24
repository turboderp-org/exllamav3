// Minimal on-device unit test: one warp, controlled A/B, verify
// mma.m8n8k4.row.col.f32.f16.f16.f32 against the ISA fragment maps.
//
// A: 8x4 fp16 (row-major), B: 4x8 fp16. One warp performs the MMA with
// fragments loaded per the ISA maps; D is stored per the C map and
// compared host-side against A @ B.
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

__global__ void mma_test_kernel(const half* A, const half* Bm, float* D)
{
    const int lane = threadIdx.x & 31;
    // A fragment: row = lane%4 (+4 if lane >= 16); a0 = (A[row][0], A[row][1]),
    // a1 = (A[row][2], A[row][3])
    int arow = (lane & 3) + (lane >= 16 ? 4 : 0);
    uint32_t a0 = *(const uint32_t*)(A + arow * 4);
    uint32_t a1 = *(const uint32_t*)(A + arow * 4 + 2);
    // B fragment: col = lane%4 (+4 if high); b0 = (B[0][col], B[1][col]),
    // b1 = (B[2][col], B[3][col]) — B stored row-major (k, n): B[k*8+col]
    int bcol = (lane & 3) + (lane >= 16 ? 4 : 0);
    __half2 h00 = __halves2half2(Bm[0 * 8 + bcol], Bm[1 * 8 + bcol]);
    __half2 h10 = __halves2half2(Bm[2 * 8 + bcol], Bm[3 * 8 + bcol]);
    uint32_t b0 = *reinterpret_cast<uint32_t*>(&h00);
    uint32_t b1 = *reinterpret_cast<uint32_t*>(&h10);
    float d[8] = {0.f};
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    // D store per the C map: row = X (+4 if high), X = (lane&1)+((reg>>1)&1)*2
    // col = ((reg>>2)&1)*4 + (lane&2) + (reg&1)
    for (int reg = 0; reg < 8; ++reg)
    {
        int X = (lane & 1) + ((reg >> 1) & 1) * 2;
        int drow = X + (lane >= 16 ? 4 : 0);
        int dcol = ((reg >> 2) & 1) * 4 + (lane & 2) + (reg & 1);
        D[drow * 8 + dcol] = d[reg];
    }
}

void mma_test(torch::Tensor A, torch::Tensor Bm, torch::Tensor D)
{
    mma_test_kernel<<<1, 32>>>(
        (const half*)A.data_ptr(), (const half*)Bm.data_ptr(), (float*)D.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mma_test", &mma_test, "mma_test");
}