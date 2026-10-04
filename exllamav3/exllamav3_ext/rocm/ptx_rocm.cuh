#pragma once

// ROCm implementations of the tensor-core fragment operations in ptx.cuh (mma.sync m16n8k16, ldmatrix.x4).
// Included by ptx.cuh on ROCm only, after the fragment types are defined.
//
// These keep NVIDIA's per-lane fragment layouts, so every kernel built on them (EXL3 GEMM, MoE, GEMV) runs
// unchanged, and emulate the operation with lane shuffles and FMAs. That is the portable baseline: RDNA's WMMA
// instructions use a different operand layout (each lane holds whole rows of A and columns of B), which needs
// kernels written for it rather than a drop-in replacement here.
//
// Fragment layouts, per the PTX ISA (m16n8k16, .f16), with g = lane / 4 and t = lane % 4:
//   A (16x16, row-major): a[j] = { A[g + 8 * (j % 2)][2t + 8 * (j / 2)], A[...][... + 1] }
//   B (16x8, col-major):  b[i] = { B[2t + 8 * i][g], B[2t + 8 * i + 1][g] }
//   C (16x8):             c[i] = D[g + 8 * (i / 2)][2t + (i % 2)]   (f16: c[0] = row g, c[1] = row g + 8)

__device__ __forceinline__ half2 exl3_shfl_h2(half2 v, int src)
{
    uint32_t u = __builtin_bit_cast(uint32_t, v);
    u = __shfl_sync(0xffffffffu, u, src);
    return __builtin_bit_cast(half2, u);
}

// Gather the two rows of A (g, g + 8) and the two columns of B (2t, 2t + 1) this lane's outputs need, as
// eight k-pairs each. Row m of A lives in lanes 4 * (m % 8) .. + 3 (lane 4g + i holds k = 2i, 2i + 1 in a[0] /
// a[1] and k = 2i + 8, 2i + 9 in a[2] / a[3]); column n of B lives in lanes 4n .. 4n + 3 likewise.

__device__ __forceinline__ void exl3_mma_gather
(
    const FragA& frag_a,
    const FragB& frag_b,
    half2 (&a_rows)[2][8],
    half2 (&b_cols)[2][8]
)
{
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        a_rows[0][i]     = exl3_shfl_h2(frag_a.elems[0], 4 * g + i);
        a_rows[0][4 + i] = exl3_shfl_h2(frag_a.elems[2], 4 * g + i);
        a_rows[1][i]     = exl3_shfl_h2(frag_a.elems[1], 4 * g + i);
        a_rows[1][4 + i] = exl3_shfl_h2(frag_a.elems[3], 4 * g + i);
        b_cols[0][i]     = exl3_shfl_h2(frag_b.elems[0], 8 * t + i);
        b_cols[0][4 + i] = exl3_shfl_h2(frag_b.elems[1], 8 * t + i);
        b_cols[1][i]     = exl3_shfl_h2(frag_b.elems[0], 8 * t + 4 + i);
        b_cols[1][4 + i] = exl3_shfl_h2(frag_b.elems[1], 8 * t + 4 + i);
    }
}

__device__ __forceinline__ float exl3_dot16(const half2 (&a)[8], const half2 (&b)[8])
{
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float2 af = __half22float2(a[i]);
        float2 bf = __half22float2(b[i]);
        s = fmaf(af.x, bf.x, s);
        s = fmaf(af.y, bf.y, s);
    }
    return s;
}

// FP16 @ FP16 + FP32 -> FP32

__device__ __forceinline__ void exl3_mma_m16n8k16_f32(const FragA& frag_a, const FragB& frag_b, FragC& frag_c)
{
    half2 a_rows[2][8], b_cols[2][8];
    exl3_mma_gather(frag_a, frag_b, a_rows, b_cols);
    frag_c.elems[0] += exl3_dot16(a_rows[0], b_cols[0]);
    frag_c.elems[1] += exl3_dot16(a_rows[0], b_cols[1]);
    frag_c.elems[2] += exl3_dot16(a_rows[1], b_cols[0]);
    frag_c.elems[3] += exl3_dot16(a_rows[1], b_cols[1]);
}

// FP16 @ FP16 + FP16 -> FP16. The products are summed in fp32 and the result rounds once into the fp16
// accumulator (the hardware instruction leaves its internal precision unspecified)

__device__ __forceinline__ void exl3_mma_m16n8k16_f16(const FragA& frag_a, const FragB& frag_b, FragC_h& frag_c)
{
    half2 a_rows[2][8], b_cols[2][8];
    exl3_mma_gather(frag_a, frag_b, a_rows, b_cols);
    float2 c0 = __half22float2(frag_c.elems[0]);
    float2 c1 = __half22float2(frag_c.elems[1]);
    c0.x += exl3_dot16(a_rows[0], b_cols[0]);
    c0.y += exl3_dot16(a_rows[0], b_cols[1]);
    c1.x += exl3_dot16(a_rows[1], b_cols[0]);
    c1.y += exl3_dot16(a_rows[1], b_cols[1]);
    frag_c.elems[0] = __float22half2_rn(c0);
    frag_c.elems[1] = __float22half2_rn(c1);
}

// ldmatrix.sync.aligned.m8n8.x4.b16: lanes 8i .. 8i + 7 supply the row addresses of matrix i, and lane l
// receives, for each matrix i, the two b16 values at row l / 4, columns 2 * (l % 4) and + 1

__device__ __forceinline__ void exl3_ldsm4(FragA& frag_a, const void* smem_ptr)
{
    int lane = threadIdx.x & 31;
    unsigned long long base = (unsigned long long) (uintptr_t) smem_ptr;
    int row = lane >> 2;
    int col_bytes = (lane & 3) * 4;
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        unsigned long long row_ptr = __shfl_sync(0xffffffffu, base, 8 * i + row);
        a[i] = *reinterpret_cast<const uint32_t*>((uintptr_t) row_ptr + col_bytes);
    }
}
