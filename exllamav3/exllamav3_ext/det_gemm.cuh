#pragma once
#include <cuda_fp16.h>
#include <mma.h>

/*

Building blocks for GEMMs whose results are bit-identical across GPU architectures.

fp16 tensor-core MMA accumulates its 16-term inner products with an architecture-specific
datapath (alignment, internal width, truncation), so the same mma.sync sequence gives different
low bits on Blackwell than on Ada/Ampere. Integer MMA is exact, so the scheme here is Ozaki-style
splitting: operands are quantized to 14-bit fixed point relative to a scale (per row and K chunk
for activations, per row over K for weights), split as q = hi * 128 + lo with a SIGNED low slice
(lo in [-64, 63]: the dropped lo*lo term then has zero mean; a non-negative lo makes it a bias
10x the fp16 error), and three s8 x s8 -> s32 MMA passes (hi*hi, and both cross terms merged
into one k32 MMA by pairing (A_hi, B_lo) in its first K half and (A_lo, B_hi) in the second)
produce exact int32 sums that are combined in fp32 with explicit non-contracting intrinsics.
Precision matches the fp16 tensor-core path (the error is dominated by the half output rounding).

Transcendentals: the build enables --use_fast_math, which routes expf/sqrtf to MUFU
approximations whose results are not guaranteed identical across architectures. The epilogues of
deterministic kernels use exp_det (range reduction + polynomial, FMAs only) and the correctly
rounded __fsqrt_rn / __fdiv_rn, whose results are unique by definition.

*/

#define DET_QMAX 16319.0f          // |q| <= 16319 keeps hi in [-128, 127] with the shifted split
#define DET_I8_LDS 80

__device__ __forceinline__ void det_quant_split(float v, float inv, signed char& hi, signed char& lo)
{
    int q = __float2int_rn(__fmul_rn(v, inv));
    q = max(-16319, min(16319, q));
    int h = (q + 64) >> 7;             // round(q / 128)
    hi = (signed char) h;
    lo = (signed char) (q - h * 128);  // -64..63
}

// Pack 16 consecutive values (one thread's share of a row chunk) into int8 hi / lo int4 words
__device__ __forceinline__ void det_quant16(const float* v, float inv, int4& hi4, int4& lo4)
{
    unsigned ph[4], pl[4];
    #pragma unroll
    for (int w = 0; w < 4; ++w)
    {
        unsigned h = 0, l = 0;
        #pragma unroll
        for (int b = 0; b < 4; ++b)
        {
            signed char hi, lo;
            det_quant_split(v[w * 4 + b], inv, hi, lo);
            h |= ((unsigned) (unsigned char) hi) << (8 * b);
            l |= ((unsigned) (unsigned char) lo) << (8 * b);
        }
        ph[w] = h; pl[w] = l;
    }
    hi4 = make_int4(ph[0], ph[1], ph[2], ph[3]);
    lo4 = make_int4(pl[0], pl[1], pl[2], pl[3]);
}

__device__ __forceinline__ void det_mma_s8(int* c, const unsigned* a, const unsigned* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// One k32 step of the three-pass product for a 16 x 8 tile: hh += A_hi B_hi; x += A_hi B_lo + A_lo B_hi
__device__ __forceinline__ void det_mma3(int* acc_hh, int* acc_x, const unsigned* ah, const unsigned* al, const unsigned* bh, const unsigned* bl)
{
    det_mma_s8(acc_hh, ah, bh);
    unsigned ax0[4] = { ah[0], ah[1], al[0], al[1] };
    unsigned bx0[2] = { bl[0], bh[0] };
    det_mma_s8(acc_x, ax0, bx0);
    unsigned ax1[4] = { ah[2], ah[3], al[2], al[3] };
    unsigned bx1[2] = { bl[1], bh[1] };
    det_mma_s8(acc_x, ax1, bx1);
}

// Exact chunk sums -> fp32 with the chunk's activation scale, fixed order, no contraction
__device__ __forceinline__ float det_flush(int hh, int x, float scale, float acc)
{
    float sum = __fmaf_rn(16384.0f, __int2float_rn(hh), __fmul_rn(128.0f, __int2float_rn(x)));
    return __fmaf_rn(sum, scale, acc);
}

__device__ __forceinline__ unsigned det_smem_u32(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void det_cp_async16(unsigned dst, const void* src, int src_bytes)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void det_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void det_cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void det_ldmatrix_x4(unsigned* r, unsigned addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// Byte offset of 16-byte piece c (0..7) of row r in a dense 128-byte-row int8 tile, XOR-swizzled
// so that both 16-byte async stores and ldmatrix fragment loads are bank-conflict free
__device__ __forceinline__ int det_swz8(int r, int c) { return r * 128 + ((c ^ (r & 7)) << 4); }

// Same for dense 64-byte rows (four 16-byte pieces): piece c ^ ((r >> 1) & 3)
__device__ __forceinline__ int det_swz4(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }
template <int ROWB> __device__ __forceinline__ int det_swz(int r, int c) { return ROWB == 128 ? det_swz8(r, c) : det_swz4(r, c); }

// A fragment (16 x 32 int8, one m-tile) for lane l via ldmatrix.x4: row (l & 7) + ((l >> 3) & 1) * 8, k piece (l >> 4)
template <int ROWB = 128>
__device__ __forceinline__ void det_load_a(unsigned* frag, unsigned tile_base, int row0, int kpiece0, int lane)
{
    const int row = row0 + (lane & 7) + ((lane >> 3) & 1) * 8;
    det_ldmatrix_x4(frag, tile_base + det_swz<ROWB>(row, kpiece0 + (lane >> 4)));
}
// B fragments for two n-tiles (16 n x 32 k): r[0..1] = n-tile 0, r[2..3] = n-tile 1
template <int ROWB = 128>
__device__ __forceinline__ void det_load_b2(unsigned* r4, unsigned tile_base, int n0, int kpiece0, int lane)
{
    const int n = n0 + (lane & 7) + ((lane >> 4) & 1) * 8;
    det_ldmatrix_x4(r4, tile_base + det_swz<ROWB>(n, kpiece0 + ((lane >> 3) & 1)));
}

// exp with FMAs only: 2^n * p(r), x = n ln2 + r. About 2 ulp; identical on every architecture
__device__ __forceinline__ float exp_det(float x)
{
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    const float n = rintf(__fmul_rn(x, 1.44269504088896341f));
    float r = __fmaf_rn(n, -0.693145751953125f, x);          // ln2 hi
    r = __fmaf_rn(n, -1.428606765330187e-06f, r);            // ln2 lo
    float p = 1.9875691500e-4f;
    p = __fmaf_rn(p, r, 1.3981999507e-3f);
    p = __fmaf_rn(p, r, 8.3334519073e-3f);
    p = __fmaf_rn(p, r, 4.1665795894e-2f);
    p = __fmaf_rn(p, r, 1.6666665459e-1f);
    p = __fmaf_rn(p, r, 5.0000001201e-1f);
    p = __fmaf_rn(p, __fmul_rn(r, r), r);
    p = __fadd_rn(p, 1.0f);
    return __int_as_float(__float_as_int(p) + ((int) n << 23));
}
__device__ __forceinline__ float sigmoid_det(float x) { return __fdiv_rn(1.0f, __fadd_rn(1.0f, exp_det(-x))); }
__device__ __forceinline__ float silu_det(float x) { return __fmul_rn(x, sigmoid_det(x)); }

// log(m 2^e) = e ln2 + 2 atanh(s), s = (m - 1) / (m + 1), FMAs only, about 2 ulp
__device__ __forceinline__ float det_log_series(float s)
{
    const float s2 = __fmul_rn(s, s);
    float p = 0.15313800f;
    p = __fmaf_rn(p, s2, 0.18182640f);
    p = __fmaf_rn(p, s2, 0.22222198f);
    p = __fmaf_rn(p, s2, 0.28571428f);
    p = __fmaf_rn(p, s2, 0.40000000f);
    p = __fmaf_rn(p, s2, 0.66666667f);
    p = __fmaf_rn(p, s2, 2.0f);
    return __fmul_rn(p, s);
}
__device__ __forceinline__ float log_det(float x)
{
    int ix = __float_as_int(x);
    int e = ((ix >> 23) & 0xff) - 127;
    float m = __int_as_float((ix & 0x007fffff) | 0x3f800000);   // [1, 2)
    if (m > 1.41421356f) { m = __fmul_rn(m, 0.5f); e += 1; }
    float r = det_log_series(__fdiv_rn(__fadd_rn(m, -1.0f), __fadd_rn(m, 1.0f)));
    r = __fmaf_rn((float) e, 0.693145751953125f, r);
    r = __fmaf_rn((float) e, 1.428606765330187e-06f, r);
    return r;
}
// log(1 + y) for y >= 0: the series in y / (2 + y) directly while 1 + y needs no exponent step
// (keeps full precision for tiny y), log_det(1 + y) beyond
__device__ __forceinline__ float log1p_det(float y)
{
    if (y < 0.41421356f) return det_log_series(__fdiv_rn(y, __fadd_rn(2.0f, y)));
    return log_det(__fadd_rn(1.0f, y));
}
// softplus matching torch (beta 1, threshold 20)
__device__ __forceinline__ float softplus_det(float x)
{
    if (x > 20.0f) return x;
    return log1p_det(exp_det(x));
}
