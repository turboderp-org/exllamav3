#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include "../util.h"
#include "../util.cuh"
#include "../triton_kernel.h"
#include "rdna3_attn.h"

// Flash-decoding split kernel for AMD RDNA3 (gfx11, wave32), a drop-in for the Triton
// _paged_attn_decode_split_kernel on the graphed decode/verify path (bc_attn.py): same launch
// arguments (plus Triton's two hidden scratch pointers), same program/split decomposition and
// the same partial_o / partial_ml layout, so the Triton combine kernel reduces its output.
//
// Per (kv head, split) block of 4 waves; each wave walks 32-token blocks of the split:
// - scores: lane <-> token. The lane reads its token's key row, fp16 or 8-bit (midpoint grid,
//   fp16 scale per 32 values, rotated domain); q rows are LDS broadcasts; v_dot2 per 32-group
// - online softmax with a wave-uniform running max per row
// - values: lane <-> dims (8 per lane at head_dim 256), p broadcast through LDS, fp32 FMAs
// - the four waves merge through LDS, then one partial per (row, split) is stored
//
// Scope: causal, no window / softcap / sinks, q_len <= 16, head_dim 128 or 256, cache fp16 or
// 8-bit for both K and V, softmax scale 1/sqrt(head_dim). Other configurations keep Triton.
//
// Adapted from exllamav3-rocm (github.com/phoenixhaxor/exllamav3-rocm, MIT).

#ifdef __HIP_PLATFORM_AMD__

namespace rdna3_attn {

constexpr int PAGE = 256;
constexpr int TB = 32;              // tokens per wave block

typedef _Float16 hv2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c)
{
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(hv2, a), __builtin_bit_cast(hv2, b), c, false);
}

// Two unsigned bytes of w (selected by v_perm; bytes 4-7 of the pair are 0x64) as a half2 of (b - 128):
// 0x64bb is 1024 + b
__device__ __forceinline__ uint32_t bytes_to_h2c(uint32_t w, uint32_t sel)
{
    uint32_t h = __builtin_amdgcn_perm(0x64646464u, w, sel);
    half2 v = *reinterpret_cast<half2*>(&h);
    v = __hsub2(v, __float2half2_rn(1152.0f));
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float wave_max(float v)
{
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) v = fmaxf(v, __shfl_xor(v, i));
    return v;
}

__device__ __forceinline__ float wave_sum(float v)
{
    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) v += __shfl_xor(v, i);
    return v;
}

}  // namespace rdna3_attn

// Block layout: QL == 1 (decode): TG = 4 waves split the tokens, each computes all GROUP rows,
// merged through LDS. QL > 1 (draft verification): one wave per query position (GROUP rows each)
// over all tokens of the split, no merge; the program for h_block 0 of each kv head computes every
// row and stores the partials of all its h_blocks, the other h_block programs exit (K/V read once)
template <int QL> struct AttnGeom
{
    static constexpr int TG = QL == 1 ? 4 : 1;       // token groups (waves) per row group
    static constexpr int RG = QL;                    // row groups (waves): one per query position
    static constexpr int WAVES = TG * RG;
};

template <int QC, int HD, int QL, int GROUP>
__global__ __launch_bounds__(AttnGeom<QL>::WAVES * 32)
void rdna3_attn_decode_split_kernel
(
    const half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    const float* __restrict__ sinks,
    void* scratch0,
    void* scratch1
)
{
    using namespace rdna3_attn;
    constexpr int TG = AttnGeom<QL>::TG;
    constexpr int WAVES = AttnGeom<QL>::WAVES;
    constexpr int NTH = WAVES * 32;
    constexpr int BM = QL <= 1 ? 1 : QL <= 2 ? 2 : QL <= 4 ? 4 : QL <= 8 ? 8 : 16;
    constexpr int BH = 16 / BM;
    constexpr int BLOCK_ROWS = QL * GROUP <= 16 ? 16 : 32;   // partial layout rows (decode_row_layout)
    constexpr int RW = GROUP;                                // rows per wave (one query position)
    constexpr int NROWS = QL * GROUP;                        // rows of this kv head, query-major
    constexpr int G = HD / 32;                               // 32-value groups per head
    constexpr int DPL = HD / 32;                             // value dims per lane
    constexpr int H_BLOCKS = (QL * GROUP + BLOCK_ROWS - 1) / BLOCK_ROWS;

    __shared__ uint32_t qs[NROWS][HD / 2];       // q rows as half2 (rotated for 8-bit keys)
    __shared__ float qsum[NROWS][G];             // per-group sums of qs (midpoint correction)
    __shared__ float pbuf[WAVES][TB][RW];        // softmax weights of the current block
    __shared__ float accm[TG > 1 ? RW : 1][TG > 1 ? HD : 1];    // cross-wave merge (decode)
    __shared__ float mw[WAVES][RW], lw[WAVES][RW];

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int h_block = pid % H_BLOCKS;
    if (h_block != 0) return;
    const int bh = pid / H_BLOCKS;
    const int NKV = (int) (intptr_t) scratch0;   // kv heads, bound at kernel creation (rdna3_attn_split_kernel)
    const int NQH = NKV * GROUP;
    const int batch = bh / NKV;
    const int kvh = bh % NKV;

    const int total_k_len = cache_seqlens[batch] + QL;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    // Stage q: one thread per (row, 32-group); 32-point Hadamard for rotated (8-bit) keys
    for (int task = threadIdx.x; task < NROWS * G; task += NTH)
    {
        const int gr = task / G, g = task % G;
        const int qi = gr / GROUP, hl = gr % GROUP;
        float x[32];
        const half* qp = q + ((size_t) (batch * QL + qi) * NQH + kvh * GROUP + hl) * HD + g * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) x[i] = __half2float(qp[i]);
        if constexpr (QC > 0)
        {
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1)
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    if (!(i & h)) { float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
            #pragma unroll
            for (int i = 0; i < 32; ++i) x[i] *= 0.17677669529663687f;
        }
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            half2 v = __floats2half2_rn(x[2 * i], x[2 * i + 1]);
            s += __low2float(v) + __high2float(v);
            qs[gr][g * 16 + i] = *reinterpret_cast<uint32_t*>(&v);
        }
        qsum[gr][g] = s;
    }
    if constexpr (TG > 1)
        for (int i = threadIdx.x; i < RW * HD; i += NTH) (&accm[0][0])[i] = 0.0f;
    __syncthreads();

    const float sm_scale = QC > 0 ? rsqrtf((float) HD) / 128.0f : rsqrtf((float) HD);

    const int rg = warp / TG;                      // row group = query position
    const int tg = warp % TG;                      // token group
    const int row0 = rg * RW;
    const int q_abs = total_k_len - QL + rg;       // causal bound of this wave's rows

    float m[RW], l[RW], acc[RW][DPL], psum[RW];
    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        m[r] = -INFINITY; l[r] = 0.0f; psum[r] = 0.0f;
        #pragma unroll
        for (int j = 0; j < DPL; ++j) acc[r][j] = 0.0f;
    }

    const size_t row_elems = (size_t) NKV * HD;      // per token row: all kv heads
    const int* bt = block_table + (size_t) batch * num_pages_per_seq;
    const int t_last = min(n_end, q_abs + 1);        // causal: no keys past this wave's position

    for (int b0 = n_start + tg * TB; b0 < t_last; b0 += TG * TB)
    {
        const int t = b0 + lane;
        const bool tvalid = t < t_last;
        const int tc = tvalid ? t : t_last - 1;
        const int tok_row = bt[tc / PAGE] * PAGE + (tc % PAGE);
        const size_t kbase = (size_t) tok_row * row_elems + (size_t) kvh * HD;

        // Scores for this lane's token
        float s[RW];
        #pragma unroll
        for (int r = 0; r < RW; ++r) s[r] = 0.0f;

        #pragma unroll 1
        for (int g = 0; g < G; ++g)
        {
            uint32_t kw[16];
            float gs = 1.0f;
            if constexpr (QC > 0)
            {
                const uint4* kp = (const uint4*) ((const uint8_t*) k_cache + kbase + g * 32);
                uint4 a = kp[0], b = kp[1];
                uint32_t w8[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    kw[2 * i] = rdna3_attn::bytes_to_h2c(w8[i], 0x04010400u);
                    kw[2 * i + 1] = rdna3_attn::bytes_to_h2c(w8[i], 0x04030402u);
                }
                gs = __half2float(k_scales[(size_t) tok_row * (NKV * G) + kvh * G + g]);
            }
            else
            {
                const uint4* kp = (const uint4*) ((const half*) k_cache + kbase + g * 32);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    uint4 v = kp[i];
                    kw[4 * i] = v.x; kw[4 * i + 1] = v.y; kw[4 * i + 2] = v.z; kw[4 * i + 3] = v.w;
                }
            }
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const uint4* qp = (const uint4*) &qs[row0 + r][g * 16];
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    uint4 qv = qp[i];
                    d = rdna3_attn::dot2(kw[4 * i], qv.x, d);
                    d = rdna3_attn::dot2(kw[4 * i + 1], qv.y, d);
                    d = rdna3_attn::dot2(kw[4 * i + 2], qv.z, d);
                    d = rdna3_attn::dot2(kw[4 * i + 3], qv.w, d);
                }
                if constexpr (QC > 0) d = (d + 0.5f * qsum[row0 + r][g]) * gs;   // (b - 128) + 0.5 = b - 127.5
                s[r] += d;
            }
        }

        // Online softmax, wave-uniform running max per row
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            const float sc = tvalid ? s[r] * sm_scale : -INFINITY;
            const float m_new = fmaxf(m[r], rdna3_attn::wave_max(sc));
            const float alpha = m[r] == -INFINITY ? 0.0f : __expf(m[r] - m_new);
            const float p = tvalid ? __expf(sc - m_new) : 0.0f;
            l[r] = l[r] * alpha + p;
            psum[r] *= alpha;
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[r][j] *= alpha;
            m[r] = m_new;
            pbuf[warp][lane][r] = p;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

        // Values: lane <-> dims
        const int nt = min(TB, t_last - b0);
        for (int tt = 0; tt < nt; ++tt)
        {
            const int vrow = __shfl(tok_row, tt);
            const size_t vbase = (size_t) vrow * row_elems + (size_t) kvh * HD + lane * DPL;
            float vf[DPL];
            float vs = 1.0f;
            if constexpr (QC > 0)
            {
                const uint8_t* vp = (const uint8_t*) v_cache + vbase;
                #pragma unroll
                for (int j = 0; j < DPL; j += 4)
                {
                    uint32_t w = *(const uint32_t*) (vp + j);
                    vf[j] = (float) ((w >> 0) & 0xffu);
                    vf[j + 1] = (float) ((w >> 8) & 0xffu);
                    vf[j + 2] = (float) ((w >> 16) & 0xffu);
                    vf[j + 3] = (float) ((w >> 24) & 0xffu);
                }
                vs = __half2float(v_scales[(size_t) vrow * (NKV * G) + kvh * G + (lane * DPL) / 32]) * (1.0f / 128.0f);
            }
            else
            {
                const half2* vp = (const half2*) ((const half*) v_cache + vbase);
                #pragma unroll
                for (int j = 0; j < DPL / 2; ++j)
                {
                    half2 v = vp[j];
                    vf[2 * j] = __low2float(v);
                    vf[2 * j + 1] = __high2float(v);
                }
            }
            #pragma unroll
            for (int r = 0; r < RW; ++r)
            {
                const float pp = pbuf[warp][tt][r] * vs;
                psum[r] += pp;
                #pragma unroll
                for (int j = 0; j < DPL; ++j) acc[r][j] = fmaf(pp, vf[j], acc[r][j]);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }

    #pragma unroll
    for (int r = 0; r < RW; ++r)
    {
        l[r] = rdna3_attn::wave_sum(l[r]);
        if constexpr (QC > 0)
        {
            #pragma unroll
            for (int j = 0; j < DPL; ++j) acc[r][j] -= 127.5f * psum[r];
        }
    }

    // Row (qi, hl) -> partial slot: rows of a kv head are packed head-major (hl * q_len + qi) and cut
    // into BLOCK_ROWS-row programs, as in triton_paged.decode_row_layout
    auto slot = [&] (int qi, int hl, float*& po, float*& pm)
    {
        const int row = hl * QL + qi;
        const int hb = row / BLOCK_ROWS;
        const int r16 = row % BLOCK_ROWS;
        const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
        po = partial_o + (pbase * BLOCK_ROWS + r16) * HD;
        pm = partial_ml + (pbase * BLOCK_ROWS + r16) * 2;
    };

    if constexpr (TG == 1)
    {
        if (split >= num_splits) return;
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            float* po; float* pm;
            slot(rg, r, po, pm);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) po[lane * DPL + j] = acc[r][j];
            if (lane == 0) { pm[0] = m[r]; pm[1] = l[r]; }
        }
        // Padded / out-of-group rows of the partial tiles are never read back into valid outputs
    }
    else
    {
        // Merge the token-group waves through LDS
        #pragma unroll
        for (int r = 0; r < RW; ++r)
            if (lane == 0) { mw[warp][r] = m[r]; lw[warp][r] = l[r]; }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < RW; ++r)
        {
            float mt = -INFINITY;
            #pragma unroll
            for (int w = 0; w < TG; ++w) mt = fmaxf(mt, mw[w][r]);
            const float f = m[r] == -INFINITY ? 0.0f : __expf(m[r] - mt);
            #pragma unroll
            for (int j = 0; j < DPL; ++j) atomicAdd(&accm[r][lane * DPL + j], acc[r][j] * f);
        }
        __syncthreads();
        if (split >= num_splits) return;
        for (int i = threadIdx.x; i < BLOCK_ROWS * HD; i += NTH)
        {
            const int r16 = i / HD;            // QL == 1: r16 = head within h_block 0
            float* po = partial_o + ((size_t) (bh * H_BLOCKS) * num_splits + split) * BLOCK_ROWS * HD;
            po[i] = r16 < RW ? accm[r16][i % HD] : 0.0f;
        }
        if (threadIdx.x < BLOCK_ROWS)
        {
            const int r = threadIdx.x;
            float mt = -INFINITY, lt = 0.0f;
            if (r < RW)
            {
                #pragma unroll
                for (int w = 0; w < TG; ++w) mt = fmaxf(mt, mw[w][r]);
                #pragma unroll
                for (int w = 0; w < TG; ++w)
                    if (mw[w][r] != -INFINITY) lt += lw[w][r] * __expf(mw[w][r] - mt);
            }
            float* pm = partial_ml + ((size_t) (bh * H_BLOCKS) * num_splits + split) * BLOCK_ROWS * 2;
            pm[r * 2] = mt;
            pm[r * 2 + 1] = lt;
        }
    }
}

// Verification (2 <= q_len <= 8), 8-bit cache, matrix cores: S = Q K^T and O += P V as
// v_wmma_f32_16x16x16_f16 tiles. Rows (q_len x GROUP, query-major) pad to 16-row tiles; K/V tiles
// of TB tokens are staged in LDS (V transposed by 4-dim words so a WMMA B fragment is 4 contiguous
// reads), with a register prefetch of the next tile
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <int HD, int QL, int GROUP>
__global__ __launch_bounds__(256)
void rdna3_attn_verify_wmma_kernel
(
    const half* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    const float* __restrict__ sinks,
    void* scratch0,
    void* scratch1
)
{
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__GFX11__)
    // WMMA (v_wmma_f32_16x16x16_f16, wave32) is gfx11 only; the dispatcher never selects this kernel elsewhere
    return;
#else
    using namespace rdna3_attn;
    constexpr int NTH = 256;
    constexpr int W = 8;
    constexpr int BM = QL <= 2 ? 2 : QL <= 4 ? 4 : 8;
    constexpr int BH = 16 / BM;
    constexpr int BLOCK_ROWS = QL * GROUP <= 16 ? 16 : 32;   // partial layout rows (decode_row_layout)
    constexpr int NR = QL * GROUP;
    constexpr int RT = (NR + 15) / 16;              // 16-row tiles
    constexpr int NRP = RT * 16;
    constexpr int G = HD / 32;
    constexpr int KS = HD / 16;                     // k-steps over dims
    constexpr int DT = HD / 16;                     // 16-wide dim tiles
    constexpr int DTW = DT / W;                     // dim tiles per wave (PV)
    constexpr int H_BLOCKS = (QL * GROUP + BLOCK_ROWS - 1) / BLOCK_ROWS;
    constexpr int QSTR = HD + 8;                    // q row stride (halves), conflict-free fragment reads
    constexpr int KSTR = HD + 16;                   // key row stride (bytes)
    constexpr int VSTR = TB + 4;                    // transposed value row stride (words)
    constexpr int PSTR = TB + 8;                    // p row stride (halves)
    constexpr int CPR = HD / 16;
    constexpr int NCHUNK = 2 * TB * CPR;
    constexpr int NSC = 2 * TB * G;
    constexpr int PER_T = (NCHUNK + NTH - 1) / NTH;
    constexpr int PER_S = (NSC + NTH - 1) / NTH;
    static_assert(RT * 2 <= W && DT % W == 0, "tile decomposition");

    __shared__ __attribute__((aligned(16))) _Float16 qh[NRP][QSTR];
    __shared__ float qsum[NRP][G];
    __shared__ __attribute__((aligned(16))) uint8_t kt[TB][KSTR];
    __shared__ __attribute__((aligned(16))) uint32_t vtt[HD / 4][VSTR];
    __shared__ half ksc[TB][G];
    __shared__ __attribute__((aligned(16))) half vsc[G][TB];
    __shared__ float sbuf[NRP][TB];
    __shared__ __attribute__((aligned(16))) _Float16 pbuf[NRP][PSTR];
    __shared__ float mrow[NRP], lrow[NRP], arow[NRP];

    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int l16 = lane & 15;
    const int hi = lane >> 4;

    const int h_block = pid % H_BLOCKS;
    if (h_block != 0) return;
    const int bh = pid / H_BLOCKS;
    const int NKV = (int) (intptr_t) scratch0;   // kv heads, bound at kernel creation (rdna3_attn_split_kernel)
    const int NQH = NKV * GROUP;
    const int batch = bh / NKV;
    const int kvh = bh % NKV;

    const int total_k_len = cache_seqlens[batch] + QL;
    const int n_start = split * split_len;
    const int n_end = min(n_start + split_len, total_k_len);

    // Stage q (rotated, fp16) and per-group sums; zero padding rows
    for (int task = threadIdx.x; task < NRP * G; task += NTH)
    {
        const int gr = task / G, g = task % G;
        float x[32];
        if (gr < NR)
        {
            const int qi = gr / GROUP, hl = gr % GROUP;
            const half* qp = q + ((size_t) (batch * QL + qi) * NQH + kvh * GROUP + hl) * HD + g * 32;
            #pragma unroll
            for (int i = 0; i < 32; ++i) x[i] = __half2float(qp[i]);
            #pragma unroll
            for (int h = 1; h < 32; h <<= 1)
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    if (!(i & h)) { float a = x[i], b = x[i + h]; x[i] = a + b; x[i + h] = a - b; }
        }
        else
        {
            #pragma unroll
            for (int i = 0; i < 32; ++i) x[i] = 0.0f;
        }
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
        {
            const _Float16 v = (_Float16) (x[i] * 0.17677669529663687f);
            sq += (float) v;
            qh[gr][g * 32 + i] = v;
        }
        qsum[gr][g] = sq;
    }
    for (int r = threadIdx.x; r < NRP; r += NTH) { mrow[r] = -INFINITY; lrow[r] = 0.0f; }

    const float sm_scale = rsqrtf((float) HD) / 128.0f;
    const size_t row_elems = (size_t) NKV * HD;
    const int* bt = block_table + (size_t) batch * num_pages_per_seq;
    const uint8_t* kc = (const uint8_t*) k_cache;
    const uint8_t* vc = (const uint8_t*) v_cache;

    uint4 pr[PER_T];
    half ps[PER_S];
    auto fetch = [&] (int tb)
    {
        #pragma unroll
        for (int i = 0; i < PER_T; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NCHUNK)
            {
                const int isv = c / (TB * CPR);
                const int cc = c % (TB * CPR);
                const int tk = cc / CPR, ch = cc % CPR;
                const int t = min(tb + tk, n_end - 1);
                const int tr = bt[t / PAGE] * PAGE + (t % PAGE);
                pr[i] = *(const uint4*) ((isv ? vc : kc) + ((size_t) tr * row_elems + (size_t) kvh * HD) + ch * 16);
            }
        }
        #pragma unroll
        for (int i = 0; i < PER_S; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NSC)
            {
                const int isv = c / (TB * G);
                const int cc = c % (TB * G);
                const int tk = cc / G, g = cc % G;
                const int t = min(tb + tk, n_end - 1);
                const int tr = bt[t / PAGE] * PAGE + (t % PAGE);
                ps[i] = (isv ? v_scales : k_scales)[(size_t) tr * (NKV * G) + kvh * G + g];
            }
        }
    };
    auto commit = [&] ()
    {
        #pragma unroll
        for (int i = 0; i < PER_T; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NCHUNK)
            {
                const int isv = c / (TB * CPR);
                const int cc = c % (TB * CPR);
                const int tk = cc / CPR, ch = cc % CPR;
                if (isv)
                {
                    vtt[ch * 4 + 0][tk] = pr[i].x;
                    vtt[ch * 4 + 1][tk] = pr[i].y;
                    vtt[ch * 4 + 2][tk] = pr[i].z;
                    vtt[ch * 4 + 3][tk] = pr[i].w;
                }
                else *(uint4*) &kt[tk][ch * 16] = pr[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < PER_S; ++i)
        {
            const int c = threadIdx.x + i * NTH;
            if (c < NSC)
            {
                const int isv = c / (TB * G);
                const int cc = c % (TB * G);
                const int tk = cc / G, g = cc % G;
                if (isv) vsc[g][tk] = ps[i]; else ksc[tk][g] = ps[i];
            }
        }
    };

    // Output accumulators: this wave's dim tiles for every row tile, D layout
    v8f oacc[RT][DTW];
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt)
        #pragma unroll
        for (int k = 0; k < DTW; ++k)
            #pragma unroll
            for (int v = 0; v < 8; ++v) oacc[rt][k][v] = 0.0f;

    if (n_start < n_end) fetch(n_start);
    for (int tb = n_start; tb < n_end; tb += TB)
    {
        __syncthreads();
        commit();
        __syncthreads();
        if (tb + TB < n_end) fetch(tb + TB);

        // Scores: wave (rt, tt) computes a 16 x 16 tile of S
        if (warp < RT * 2)
        {
            const int rt = warp >> 1, tt = warp & 1;
            const int tcol = tt * 16 + l16;
            v8f sacc;
            #pragma unroll
            for (int v = 0; v < 8; ++v) sacc[v] = 0.0f;
            #pragma unroll 2
            for (int g = 0; g < G; ++g)
            {
                v8f c;
                #pragma unroll
                for (int v = 0; v < 8; ++v) c[v] = 0.0f;
                #pragma unroll
                for (int h = 0; h < 2; ++h)
                {
                    const int ks = g * 2 + h;
                    v16h a = *(const v16h*) &qh[rt * 16 + l16][ks * 16];
                    const uint4 kb = *(const uint4*) &kt[tcol][ks * 16];
                    const uint32_t w4[4] = {kb.x, kb.y, kb.z, kb.w};
                    uint32_t bw[8];
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                    {
                        bw[2 * i] = rdna3_attn::bytes_to_h2c(w4[i], 0x04010400u);
                        bw[2 * i + 1] = rdna3_attn::bytes_to_h2c(w4[i], 0x04030402u);
                    }
                    v16h b = *(v16h*) bw;
                    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
                }
                const float gs = __half2float(ksc[tcol][g]);
                #pragma unroll
                for (int v = 0; v < 8; ++v)
                    sacc[v] += (c[v] + 0.5f * qsum[rt * 16 + 2 * v + hi][g]) * gs;
            }
            #pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                const int gr = rt * 16 + 2 * v + hi;
                const int t = tb + tcol;
                const int qi = gr / GROUP;
                const bool ok = gr < NR && t < n_end && t <= total_k_len - QL + qi;
                sbuf[gr][tcol] = ok ? sacc[v] * sm_scale : -INFINITY;
            }
        }
        __syncthreads();

        // Online softmax: 8 lanes per row, 4 tokens each
        for (int task = threadIdx.x; task < NRP * 8; task += NTH)
        {
            const int r = task >> 3, sub = task & 7;
            float sv[4];
            float mx = -INFINITY;
            #pragma unroll
            for (int i = 0; i < 4; ++i) { sv[i] = sbuf[r][sub * 4 + i]; mx = fmaxf(mx, sv[i]); }
            #pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor(mx, o));
            const float m_old = mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float ps4 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const float p = sv[i] == -INFINITY ? 0.0f : __expf(sv[i] - m_new);
                ps4 += p;
                pbuf[r][sub * 4 + i] = (_Float16) p;
            }
            #pragma unroll
            for (int o = 1; o < 8; o <<= 1) ps4 += __shfl_xor(ps4, o);
            if (sub == 0)
            {
                const float alpha = m_old == -INFINITY ? 0.0f : __expf(m_old - m_new);
                arow[r] = alpha;
                lrow[r] = lrow[r] * alpha + ps4;
                mrow[r] = m_new;
            }
        }
        __syncthreads();

        // O = O * alpha + P V over this wave's dim tiles
        #pragma unroll
        for (int rt = 0; rt < RT; ++rt)
        {
            float al[8];
            #pragma unroll
            for (int v = 0; v < 8; ++v) al[v] = arow[rt * 16 + 2 * v + hi];
            #pragma unroll
            for (int k = 0; k < DTW; ++k)
                #pragma unroll
                for (int v = 0; v < 8; ++v) oacc[rt][k][v] *= al[v];
        }
        #pragma unroll
        for (int k = 0; k < DTW; ++k)
        {
            const int d = (warp * DTW + k) * 16 + l16;
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                const uint32_t* vr = &vtt[d >> 2][ks * 16];
                const uint4 w0 = *(const uint4*) (vr + 0), w1 = *(const uint4*) (vr + 4);
                const uint4 w2 = *(const uint4*) (vr + 8), w3 = *(const uint4*) (vr + 12);
                const uint32_t wv[16] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w,
                                         w2.x, w2.y, w2.z, w2.w, w3.x, w3.y, w3.z, w3.w};
                // byte (d & 3) of tokens 2i (low half) and 2i + 1 (high half), as 0x64bb halves
                const uint32_t bsel = (uint32_t) (d & 3);
                const half2* vs2 = (const half2*) &vsc[d >> 5][ks * 16];
                uint32_t bw[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    // 0x64bb halves (1024 + b) for tokens 2i, 2i + 1, then (b - 128) + 0.5 exactly, then scale
                    const uint32_t lo = (wv[2 * i] >> (8 * bsel)) & 0xffu;
                    const uint32_t hb = (wv[2 * i + 1] >> (8 * bsel)) & 0xffu;
                    const uint32_t hw = 0x64006400u | lo | (hb << 16);
                    half2 v = *reinterpret_cast<const half2*>(&hw);
                    v = __hsub2(v, __float2half2_rn(1152.0f));
                    v = __hadd2(v, __float2half2_rn(0.5f));
                    v = __hmul2(v, vs2[i]);
                    bw[i] = *reinterpret_cast<uint32_t*>(&v);
                }
                const v16h b = *(v16h*) bw;
                #pragma unroll
                for (int rt = 0; rt < RT; ++rt)
                {
                    const v16h a = *(const v16h*) &pbuf[rt * 16 + l16][ks * 16];
                    oacc[rt][k] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, oacc[rt][k]);
                }
            }
        }
    }

    __syncthreads();
    if (split >= num_splits) return;
    #pragma unroll
    for (int rt = 0; rt < RT; ++rt)
        #pragma unroll
        for (int k = 0; k < DTW; ++k)
            #pragma unroll
            for (int v = 0; v < 8; ++v)
            {
                const int gr = rt * 16 + 2 * v + hi;
                if (gr >= NR) continue;
                const int qi = gr / GROUP, hl = gr % GROUP;
                const int hb = (hl * QL + qi) / BLOCK_ROWS, r16 = (hl * QL + qi) % BLOCK_ROWS;
                const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
                partial_o[(pbase * BLOCK_ROWS + r16) * HD + (warp * DTW + k) * 16 + l16] = oacc[rt][k][v] * (1.0f / 128.0f);   // 8-bit grid step
            }
    for (int gr = threadIdx.x; gr < NR; gr += NTH)
    {
        const int qi = gr / GROUP, hl = gr % GROUP;
        const int hb = (hl * QL + qi) / BLOCK_ROWS, r16 = (hl * QL + qi) % BLOCK_ROWS;
        const size_t pbase = ((size_t) (bh * H_BLOCKS + hb) * num_splits + split);
        partial_ml[(pbase * BLOCK_ROWS + r16) * 2] = mrow[gr];
        partial_ml[(pbase * BLOCK_ROWS + r16) * 2 + 1] = lrow[gr];
    }
#endif
}

typedef void (*fp_rdna3_attn)(const half*, const void*, const void*, const int*, const int*, half*, float*, float*,
                              const half*, const half*, const half*, int, int, int, const float*, void*, void*);

// One instance per (cache, head_dim, q_len, GQA group) with at most 64 query rows per kv head: verification on an
// 8-bit cache (q_len 2..8, up to 48 rows, which bounds its LDS at head_dim 256) on matrix cores, everything else on
// the split kernel
template <int QC, int HD, int QL, int GROUP>
static fp_rdna3_attn pick_one()
{
    if constexpr (QL * GROUP > 64) return nullptr;
    else if constexpr (QC == 8 && QL >= 2 && QL <= 8 && QL * GROUP <= 48) return rdna3_attn_verify_wmma_kernel<HD, QL, GROUP>;
    else return rdna3_attn_decode_split_kernel<QC, HD, QL, GROUP>;
}

template <int QC, int HD, int GROUP>
static fp_rdna3_attn pick_ql(int ql)
{
    switch (ql)
    {
        case 1:  return pick_one<QC, HD, 1, GROUP>();
        case 2:  return pick_one<QC, HD, 2, GROUP>();
        case 3:  return pick_one<QC, HD, 3, GROUP>();
        case 4:  return pick_one<QC, HD, 4, GROUP>();
        case 5:  return pick_one<QC, HD, 5, GROUP>();
        case 6:  return pick_one<QC, HD, 6, GROUP>();
        case 7:  return pick_one<QC, HD, 7, GROUP>();
        case 8:  return pick_one<QC, HD, 8, GROUP>();
        case 9:  return pick_one<QC, HD, 9, GROUP>();
        case 10: return pick_one<QC, HD, 10, GROUP>();
        case 11: return pick_one<QC, HD, 11, GROUP>();
        case 12: return pick_one<QC, HD, 12, GROUP>();
        case 13: return pick_one<QC, HD, 13, GROUP>();
        case 14: return pick_one<QC, HD, 14, GROUP>();
        case 15: return pick_one<QC, HD, 15, GROUP>();
        case 16: return pick_one<QC, HD, 16, GROUP>();
    }
    return nullptr;
}

template <int QC, int HD>
static fp_rdna3_attn pick_group(int group, int ql)
{
    switch (group)
    {
        case 2: return pick_ql<QC, HD, 2>(ql);
        case 4: return pick_ql<QC, HD, 4>(ql);
        case 6: return pick_ql<QC, HD, 6>(ql);
        case 8: return pick_ql<QC, HD, 8>(ql);
    }
    return nullptr;
}

#endif

std::shared_ptr<TritonKernel> rdna3_attn_split_kernel(int qc, int head_dim, int q_len, int n_kv_heads, int group)
{
    #ifdef __HIP_PLATFORM_AMD__
        fp_rdna3_attn k = nullptr;
        if (qc == 8 && head_dim == 256) k = pick_group<8, 256>(group, q_len);
        if (qc == 0 && head_dim == 256) k = pick_group<0, 256>(group, q_len);
        if (qc == 8 && head_dim == 128) k = pick_group<8, 128>(group, q_len);
        if (qc == 0 && head_dim == 128) k = pick_group<0, 128>(group, q_len);
        if (!k || n_kv_heads < 1) return nullptr;
        hipFunction_t fn;
        cuda_check(hipGetFuncBySymbol(&fn, (const void*) k));
        const int waves = q_len == 1 ? 4 : (qc == 8 && q_len <= 8) ? 8 : q_len;
        // The kv head count rides in the first trailing scratch slot (unused by these kernels otherwise)
        return std::make_shared<TritonKernel>((void*) fn, std::string("rdna3_attn_decode_split"), waves, 0,
                                              (void*) (intptr_t) n_kv_heads);
    #else
        return nullptr;
    #endif
}
