#pragma once

// HIP body behind the shared kernel wrappers (exl3_gemm_kernel.cuh,
// exl3_moe_kernel.cuh); dispatcher, kernel tables, autotuner and comp units are
// shared with CUDA. Same contract as the CUDA inner: blockIdx.x slices the
// (k x n) tile space, each column tile assembled by a lock cascade in reverse k
// order (highest-k segment acquires first and accumulates into C in place, the
// k == 0 block writes the final result), no grid-wide sync, and locks used only
// as locks so every caller's disjoint lock range stays disjoint. Decode tiers on
// (bits, cb): 4/6 with mul1 decode lane-local through funnel/byte-dot intrinsics,
// everything else through the shared dq_dispatch. NB: never name a ROCm-side
// header *_hip.cuh - torch's in-place hipify clobbers files with that pattern.

#include "../ptx.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"

// The #undef matters: exl3_moe_common.cuh's 90 KB fallback would exceed
// gfx1100's 64 KB LDS limit.
#undef SMEM_MAX
#define SMEM_MAX (8 * 1024)

// staged in shared memory, declared once per kernel: per-instantiation __shared__
// would request the LDS once per instantiation
#define EXL3_INNER_SH_FLOATS(ts_n) (16 * (ts_n))

namespace exl3_rocm_inner
{

// funnel shift is native on both vendors
__device__ __forceinline__ uint32_t alignbit16(uint32_t hi, uint32_t lo, int imm)
{
    return __funnelshift_r(lo, hi, imm) & 0xFFFFu;
}

// byte lanes sum to <= 1020, so the low 16 bits are exact
__device__ __forceinline__ uint32_t dot4_add(uint32_t x, uint32_t c)
{
    return __builtin_amdgcn_udot4(x, 0x01010101u, c, false);
}

// same arithmetic as codebook.cuh's decode_mul1_product_2
__device__ __forceinline__ float decode_w_mul1(uint32_t code)
{
    uint32_t u = dot4_add(code * 0x83DCD12Du, 0x6400u) & 0xFFFFu;
    __half h = __ushort_as_half((unsigned short) u);
    __half ki = __ushort_as_half((unsigned short) 0x1EEE);
    __half kb = __ushort_as_half((unsigned short) 0xC931);
    float v = __half2float(h) * __half2float(ki) + __half2float(kb);
    return __half2float(__float2half(v));
}

// Column lock (ptx.cuh protocol): counts completed k-tiles of one column tile;
// the topmost block resets it to 0 for the next call

__device__ __forceinline__ void lock_acquire(int* lock, int stage)
{
    if (threadIdx.x == 0)
    {
        unsigned int* a = (unsigned int*) lock;
        unsigned int state;
        do
        {
            state = __hip_atomic_load(a, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
        while (state != (unsigned int) stage);
    }
    __syncthreads();
}

__device__ __forceinline__ void lock_release(int* lock, int val, bool reset)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int* a = (unsigned int*) lock;
        if (reset)
        {
            __hip_atomic_store(a, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            return;
        }
        __hip_atomic_fetch_add(a, (unsigned int) val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// phase functions must stay __forceinline__: a __shared__ array whose address
// escapes into a non-inlined callee is demoted to local memory
struct SegCtx
{
    const half* __restrict__ A;
    const uint32_t* __restrict__ B;
    float* __restrict__ sh_c;       // [16][cols] segment partial
    int size_m;
    int size_k;
    int kt0, kt1;                   // trellis k-subtile range of the segment (16 wide each)
    int nsub_total;                 // subtiles per k row of the whole matrix (size_n / 16)
    int subs_tile;                  // subtiles in this column tile (TILESIZE_N / 16)
    int col;                        // column tile index
};

// Tier A: bits 4, cb 2. 32 words per subtile, 8 lanes per (subtile, m row) item

__device__ __forceinline__ void phase1_b4c2(SegCtx& c)
{
    int lane = threadIdx.x & 7;
    int rem = threadIdx.x >> 3;
    int groups = blockDim.x >> 3;
    int items = c.subs_tile * c.size_m;

    for (int idx = rem; idx < items; idx += groups)
    {
        int s = idx % c.subs_tile;
        int n = c.col * c.subs_tile + s;        // global subtile
        int m = idx / c.subs_tile;

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * 32) + (size_t) n * 32;
        const half* x = c.A + (size_t) m * c.size_k;
        float acc0 = 0.f, acc1 = 0.f;

        #define PAIR4(HI, LO, XK)                                                     \
        acc0 += decode_w_mul1(alignbit16(HI, LO, 28)) * (XK)[0]                       \
              + decode_w_mul1(alignbit16(HI, LO, 24)) * (XK)[1]                       \
              + decode_w_mul1(alignbit16(HI, LO, 20)) * (XK)[8]                       \
              + decode_w_mul1(alignbit16(HI, LO, 16)) * (XK)[9];                      \
        acc1 += decode_w_mul1(alignbit16(HI, LO, 12)) * (XK)[0]                       \
              + decode_w_mul1(alignbit16(HI, LO, 8))  * (XK)[1]                       \
              + decode_w_mul1(alignbit16(HI, LO, 4))  * (XK)[8]                       \
              + decode_w_mul1(alignbit16(HI, LO, 0))  * (XK)[9];

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * 32);
            uint32_t w0 = p32[4 * lane];
            uint32_t w1 = p32[4 * lane + 1];
            uint32_t w2 = p32[4 * lane + 2];
            uint32_t w3 = p32[4 * lane + 3];
            uint32_t prev = p32[(4 * lane + 31) & 31];

            float xf[16];
            const half* xk = x + t * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) xf[i] = __half2float(xk[i]);

            PAIR4(prev, w0, xf)
            PAIR4(w0, w1, xf + 2)
            PAIR4(w1, w2, xf + 4)
            PAIR4(w2, w3, xf + 6)
        }
        #undef PAIR4

        float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) s * 16;
        out[lane] = acc0;
        out[lane + 8] = acc1;
    }
}

// Tier B: bits 6, cb 2. 48 words per subtile, 6-word window algebra

__device__ __forceinline__ void phase1_b6c2(SegCtx& c)
{
    int lane = threadIdx.x & 7;
    int rem = threadIdx.x >> 3;
    int groups = blockDim.x >> 3;
    int items = c.subs_tile * c.size_m;

    for (int idx = rem; idx < items; idx += groups)
    {
        int s = idx % c.subs_tile;
        int n = c.col * c.subs_tile + s;
        int m = idx / c.subs_tile;

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * 48) + (size_t) n * 48;
        const half* x = c.A + (size_t) m * c.size_k;
        float acc0 = 0.f, acc1 = 0.f;

        #define CA(HI, LO, SH) \
            ((SH) < 32 ? alignbit16(HI, LO, (SH)) : ((HI) >> ((SH) - 32)) & 0xFFFFu)

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * 48);
            uint32_t w0 = p32[6 * lane];
            uint32_t w1 = p32[6 * lane + 1];
            uint32_t w2 = p32[6 * lane + 2];
            uint32_t w3 = p32[6 * lane + 3];
            uint32_t w4 = p32[6 * lane + 4];
            uint32_t w5 = p32[6 * lane + 5];
            uint32_t m1 = p32[(6 * lane + 47) % 48];

            float xf[16];
            const half* xk = x + t * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) xf[i] = __half2float(xk[i]);

            acc0 += decode_w_mul1(CA(m1, w0, 26)) * xf[0]
                  + decode_w_mul1(CA(m1, w0, 20)) * xf[1]
                  + decode_w_mul1(CA(m1, w0, 14)) * xf[8]
                  + decode_w_mul1(CA(m1, w0, 8))  * xf[9];
            acc1 += decode_w_mul1(CA(w0, w1, 34)) * xf[0]
                  + decode_w_mul1(CA(w0, w1, 28)) * xf[1]
                  + decode_w_mul1(CA(w0, w1, 22)) * xf[8]
                  + decode_w_mul1(CA(w0, w1, 16)) * xf[9];
            acc0 += decode_w_mul1(CA(w1, w2, 42)) * xf[2]
                  + decode_w_mul1(CA(w1, w2, 36)) * xf[3]
                  + decode_w_mul1(CA(w1, w2, 30)) * xf[10]
                  + decode_w_mul1(CA(w1, w2, 24)) * xf[11];
            acc1 += decode_w_mul1(CA(w1, w2, 18)) * xf[2]
                  + decode_w_mul1(CA(w1, w2, 12)) * xf[3]
                  + decode_w_mul1(CA(w1, w2, 6))  * xf[10]
                  + decode_w_mul1(CA(w1, w2, 0))  * xf[11];
            acc0 += decode_w_mul1(CA(w2, w3, 26)) * xf[4]
                  + decode_w_mul1(CA(w2, w3, 20)) * xf[5]
                  + decode_w_mul1(CA(w2, w3, 14)) * xf[12]
                  + decode_w_mul1(CA(w2, w3, 8))  * xf[13];
            acc1 += decode_w_mul1(CA(w3, w4, 34)) * xf[4]
                  + decode_w_mul1(CA(w3, w4, 28)) * xf[5]
                  + decode_w_mul1(CA(w3, w4, 22)) * xf[12]
                  + decode_w_mul1(CA(w3, w4, 16)) * xf[13];
            acc0 += decode_w_mul1(CA(w4, w5, 42)) * xf[6]
                  + decode_w_mul1(CA(w4, w5, 36)) * xf[7]
                  + decode_w_mul1(CA(w4, w5, 30)) * xf[14]
                  + decode_w_mul1(CA(w4, w5, 24)) * xf[15];
            acc1 += decode_w_mul1(CA(w4, w5, 18)) * xf[6]
                  + decode_w_mul1(CA(w4, w5, 12)) * xf[7]
                  + decode_w_mul1(CA(w4, w5, 6))  * xf[14]
                  + decode_w_mul1(CA(w4, w5, 0))  * xf[15];
        }
        #undef CA

        float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) s * 16;
        out[lane] = acc0;
        out[lane + 8] = acc1;
    }
}

// Tier C: everything else. One warp per subtile through dq_dispatch

template <int bits, int cb>
__device__ __forceinline__ void phase1_dq(SegCtx& c)
{
    int lane_id = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;

    int r0 = 2 * (lane_id & 3);
    int c0 = 2 * (lane_id >> 3) + ((lane_id & 7) >> 2);
    int rows[4] = { r0, r0 + 1, r0 + 8, r0 + 9 };

    for (int idx = warp; idx < c.subs_tile; idx += warps)
    {
        int n = c.col * c.subs_tile + idx;
        const uint32_t* sub_base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * (8 * bits)) + (size_t) n * (8 * bits);

        float acc[16][2];
        #pragma unroll
        for (int m = 0; m < 16; ++m) { acc[m][0] = 0.f; acc[m][1] = 0.f; }

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* ptr = sub_base + (size_t) (t - c.kt0) * (c.nsub_total * (8 * bits));

            FragB frag[2];
            dq_dispatch<bits, cb>(ptr, lane_id * 8, frag[0], frag[1]);

            #pragma unroll
            for (int m = 0; m < 16; ++m)
            {
                if (m < c.size_m)
                {
                    const half* x = c.A + (size_t) m * c.size_k + (size_t) t * 16;
                    float x0 = __half2float(x[rows[0]]);
                    float x1 = __half2float(x[rows[1]]);
                    float x2 = __half2float(x[rows[2]]);
                    float x3 = __half2float(x[rows[3]]);
                    acc[m][0] += __half2float(frag[0][0].x) * x0
                               + __half2float(frag[0][0].y) * x1
                               + __half2float(frag[0][1].x) * x2
                               + __half2float(frag[0][1].y) * x3;
                    acc[m][1] += __half2float(frag[1][0].x) * x0
                               + __half2float(frag[1][0].y) * x1
                               + __half2float(frag[1][1].x) * x2
                               + __half2float(frag[1][1].y) * x3;
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < 16; ++m)
        {
            if (m >= c.size_m) continue;
            float a0 = acc[m][0], a1 = acc[m][1];
            a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
            a0 += __shfl_xor_sync(0xffffffffu, a0, 2);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 2);

            if ((lane_id & 3) == 0)
            {
                float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) idx * 16;
                out[c0] = a0;
                out[c0 + 8] = a1;
            }
        }
    }
}

}  // namespace exl3_rocm_inner

// shared inner entry point; C is [m][n] with row stride size_n. post_scale
// applies only when shmem_out_had is set

template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
__device__ void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale,
    float* __restrict__ sh
)
{
    using namespace exl3_rocm_inner;

    constexpr int TS_N = TILESIZE_N;
    float* sh_c = sh;

    int tiles_k = size_k / 16;                 // trellis k-subtiles (16 wide)
    int tiles_n = size_n / TS_N;
    int units = tiles_k * tiles_n;
    int num_slices = gridDim.x;
    int beg = (int) ((int64_t) units * blockIdx.x / num_slices);
    int end = (int) ((int64_t) units * (blockIdx.x + 1) / num_slices);

    SegCtx c;
    c.A = A;
    c.B = (const uint32_t*) B;
    c.sh_c = sh_c;
    c.size_m = size_m;
    c.size_k = size_k;
    c.nsub_total = size_n / 16;
    c.subs_tile = TS_N / 16;

    while (beg < end)
    {
        int col = beg / tiles_k;
        int seg_k0 = beg % tiles_k;
        int seg_k1 = ((end - 1) / tiles_k == col) ? ((end - 1) % tiles_k) + 1 : tiles_k;

        c.kt0 = seg_k0;
        c.kt1 = seg_k1;
        c.col = col;
        int cols = c.subs_tile * 16;

        if constexpr (cb == 2 && bits == 4)
        {
            phase1_b4c2(c);
        }
        else if constexpr (cb == 2 && bits == 6)
        {
            phase1_b6c2(c);
        }
        else if constexpr (bits > 0)
        {
            phase1_dq<bits, cb>(c);
        }
        // bits == 0 never reaches the inner (the caller switches on K first)

        __syncthreads();

        int lock_i = tiles_k - seg_k1;
        int lock_d = seg_k1 - seg_k0;
        int* lock = &locks[col];
        lock_acquire(lock, lock_i);

        bool first = (lock_i == 0);
        bool last = (lock_i + lock_d == tiles_k);

        if (!first)
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    sh_c[i] += ((const float*) C)[(size_t) m * size_n + (size_t) col * TS_N + n];
                else
                    sh_c[i] += __half2float(((const half*) C)[(size_t) m * size_n + (size_t) col * TS_N + n]);
            }
            __syncthreads();
        }

        if (!last)
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    ((float*) C)[(size_t) m * size_n + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * size_n + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
            }
        }
        else if (shmem_out_had)
        {
            // final block: output Hadamard (without post_scale only the
            // 1/sqrt(128) factor, no per-column scale)
            int nb = cols / 128;
            int warp = threadIdx.x >> 5;
            int lane = threadIdx.x & 31;
            int warps = blockDim.x >> 5;
            int slots = size_m * nb;
            for (int slot = warp; slot < slots; slot += warps)
            {
                int m = slot / nb;
                int b = slot % nb;
                const float* p = sh_c + m * cols + b * 128;
                float v0 = p[lane * 4 + 0];
                float v1 = p[lane * 4 + 1];
                float v2 = p[lane * 4 + 2];
                float v3 = p[lane * 4 + 3];

                float s0 = v0 + v1, d0 = v0 - v1;
                float s1 = v2 + v3, d1 = v2 - v3;
                float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;

                shuffle_had_f4x32(h0, h1, h2, h3, lane);

                const float rs = 0.088388347648f;   // 1/sqrt(128)
                if (post_scale)
                {
                    const half* sb = post_scale + ((size_t) col * TS_N + b * 128) % size_n;
                    h0 *= rs * __half2float(sb[lane * 4 + 0]);
                    h1 *= rs * __half2float(sb[lane * 4 + 1]);
                    h2 *= rs * __half2float(sb[lane * 4 + 2]);
                    h3 *= rs * __half2float(sb[lane * 4 + 3]);
                }
                else
                {
                    h0 *= rs; h1 *= rs; h2 *= rs; h3 *= rs;
                }

                size_t n0 = (size_t) m * size_n + (size_t) col * TS_N + b * 128 + lane * 4;
                if constexpr (c_fp32)
                {
                    float* out = (float*) C + n0;
                    out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3;
                }
                else
                {
                    half* out = (half*) C + n0;
                    out[0] = __float2half(h0); out[1] = __float2half(h1);
                    out[2] = __float2half(h2); out[3] = __float2half(h3);
                }
            }
        }
        else
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    ((float*) C)[(size_t) m * size_n + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * size_n + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
            }
        }

        lock_release(lock, lock_d, last);

        // phase 1 must store fresh values: the accumulate path adds into sh_c
        beg = (col + 1) * tiles_k;
    }
}
