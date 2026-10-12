// fa75: flash-attention forward for Turing (sm_75), head_dim 256, fp16 in/out, for the prefill path.
// PyTorch's memory-efficient SDPA (cutlass fmha sm75) reaches 10-14 TFLOPS on these shapes; fa75 reaches
// 45-47 TFLOPS on an RTX 2080 Ti at 1650 MHz (fp32-accumulate HMMA peak ~57 TFLOPS; P V accumulates in fp16).
//
// Layout: q [Tq, Hq, 256], k/v [Tkv, Hkv, 256] (row and head strides given, last dim contiguous), o [Tq, Hq, 256]
// contiguous. Causal is bottom-right aligned (query i sees keys j <= i + Tkv - Tq), as for a prefill chunk
// appended to a cache. GQA: head h reads kv head h / (Hq / Hkv).
//
// Block = 64 query rows of one head, 4 warps x 16 rows, two blocks per SM. Q dims 0..127 live in mma A fragments
// (32 regs) and dims 128..255 in shared memory (read with ldmatrix per tile), O in fp32 accumulators (128 regs);
// S -> P stays in registers (the m16n8 accumulator layout is the m16n8k8 A layout). K and V tiles of 16 keys sit in XOR-swizzled shared memory and are read with ldmatrix(.trans); the
// global loads of V(kt) are in flight during Q K^T and those of K(kt+1) during softmax and P V.
#if !defined(USE_ROCM)

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#define HD 256
#define BM 64
#define BN 16
#define NWARPS 4
#define NTHREADS (NWARPS * 32)
#define ROW_CHUNKS (HD / 8)           // 32 x 16-byte chunks per 512-byte row

// K tile at 0, V tile at 16 KB; each [BN rows][32 chunks], chunk ^= row & 7
static __device__ __forceinline__ uint32_t tile_off(int r, int c) { return (uint32_t)((r * ROW_CHUNKS + (c ^ (r & 7))) * 16); }
// Q dims 128..255 stay in shared memory (upper 16 KB) for the whole kernel: 64 rows x 16 chunks, swizzled
#define QH_OFF 16384
static __device__ __forceinline__ uint32_t qh_off(int r, int c) { return (uint32_t)(QH_OFF + (r * 16 + (c ^ (r & 7))) * 16); }
// Q dims 0..127 staged in the lower 16 KB with the same half-row layout, then pulled into registers
static __device__ __forceinline__ uint32_t ql_off(int r, int c) { return (uint32_t)((r * 16 + (c ^ (r & 7))) * 16); }

static __device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b));
}

// fp16-accumulate HMMA: twice the fp32-accumulate rate on GeForce Turing
static __device__ __forceinline__ void mma1688h(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a0), "r"(a1), "r"(b));
}

static __device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t saddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}

static __device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t saddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}

static __device__ __forceinline__ uint32_t pack_h2(float a, float b)
{
    half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void __launch_bounds__(NTHREADS, 2)
fa75_kernel
(
    const half* __restrict__ q, int64_t q_row, int64_t q_head,
    const half* __restrict__ k, int64_t k_row, int64_t k_head,
    const half* __restrict__ v, int64_t v_row, int64_t v_head,
    half* __restrict__ o,
    int Tq, int Tkv, int Hq, int Hkv,
    float scale_log2, int causal
)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    __trap();
#else
    __shared__ __align__(128) uint8_t sm[BM * HD * 2];   // Q staging; then K tile at 0, V tile at BN * HD * 2
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(sm);
    const uint32_t sK = sbase, sV = sbase + BN * HD * 2;

    const int t_id = threadIdx.x;
    const int warp = t_id >> 5;
    const int lane = t_id & 31;
    const int gq = lane >> 2;
    const int cq = lane & 3;
    const int lr = lane & 7;
    const int lm = lane >> 3;

    const int m_tile = gridDim.x - 1 - blockIdx.x;      // heaviest (last) query tiles first
    const int hq = blockIdx.y;
    const int hk = hq / (Hq / Hkv);
    const int m0 = m_tile * BM;
    const int offs = Tkv - Tq;                            // causal: key j visible to row i iff j <= i + offs

    const half* q_ = q + hq * q_head;
    const half* k_ = k + hk * k_head;
    const half* v_ = v + hk * v_head;

    // Stage Q (64 x 256): dims 0..127 through the tile area into this warp's A fragments (32 regs), dims 128..255
    // into the upper 16 KB, where they stay (read with ldmatrix per tile). Halving the Q fragments frees the
    // registers the pipelined K/V loads need
    uint32_t qf[HD / 16][2];
    {
        #pragma unroll
        for (int j = 0; j < (BM * ROW_CHUNKS) / NTHREADS; ++j)
        {
            int idx = t_id + j * NTHREADS;
            int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (m0 + r < Tq) val = *reinterpret_cast<const uint4*>(q_ + (int64_t)(m0 + r) * q_row + c * 8);
            if (c < ROW_CHUNKS / 2) *reinterpret_cast<uint4*>(sm + ql_off(r, c)) = val;
            else *reinterpret_cast<uint4*>(sm + qh_off(r, c - ROW_CHUNKS / 2)) = val;
        }
        __syncthreads();
        // A (m = row, k = d): matrices (rows 0-7, chunk c), (rows 8-15, c), (rows 0-7, c+1), (rows 8-15, c+1)
        #pragma unroll
        for (int kk = 0; kk < HD / 16; kk += 2)
        {
            uint32_t r4[4];
            int row = 16 * warp + lr + (lm & 1) * 8;
            ldsm_x4(r4, sbase + ql_off(row, kk + (lm >> 1)));
            qf[kk][0] = r4[0]; qf[kk][1] = r4[1];
            qf[kk + 1][0] = r4[2]; qf[kk + 1][1] = r4[3];
        }
        __syncthreads();
    }

    float oacc[HD / 8][4];
    #pragma unroll
    for (int dn = 0; dn < HD / 8; ++dn)
        #pragma unroll
        for (int i = 0; i < 4; ++i) oacc[dn][i] = 0.0f;
    float mrow[2] = { -1e30f, -1e30f };
    float lrow[2] = { 0.0f, 0.0f };

    const int row0 = m0 + 16 * warp + gq;                 // this thread's rows: row0, row0 + 8
    int kv_end = causal ? min(Tkv, m0 + BM - 1 + offs + 1) : Tkv;
    kv_end = max(kv_end, 0);
    const int n_tiles = (kv_end + BN - 1) / BN;

    // Tile loads: BN rows x 32 chunks = 4 x 16 B per thread
    auto load_tile = [&](const half* src, int64_t row_stride, int n0, uint4* reg)
    {
        #pragma unroll
        for (int j = 0; j < (BN * ROW_CHUNKS) / NTHREADS; ++j)
        {
            int idx = t_id + j * NTHREADS;
            int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
            reg[j] = n0 + r < Tkv ? *reinterpret_cast<const uint4*>(src + (int64_t)(n0 + r) * row_stride + c * 8) : make_uint4(0, 0, 0, 0);
        }
    };
    auto store_tile = [&](uint32_t off, const uint4* reg)
    {
        #pragma unroll
        for (int j = 0; j < (BN * ROW_CHUNKS) / NTHREADS; ++j)
        {
            int idx = t_id + j * NTHREADS;
            int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
            *reinterpret_cast<uint4*>(sm + off + tile_off(r, c)) = reg[j];
        }
    };

    uint4 stage[(BN * ROW_CHUNKS) / NTHREADS];
    if (n_tiles > 0)
    {
        load_tile(k_, k_row, 0, stage);
        store_tile(0, stage);
    }
    __syncthreads();

    for (int kt = 0; kt < n_tiles; ++kt)
    {
        const int n0 = kt * BN;

        // V(kt) in flight during Q K^T
        load_tile(v_, v_row, n0, stage);

        // S = Q K^T : 16 x 16 per warp, 2 n8 tiles
        float s[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) s[nt][i] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < HD / 8; kk += 2)
        {
            // matrices: (keys 0-7, chunk kk), (keys 8-15, kk), (keys 0-7, kk+1), (keys 8-15, kk+1)
            uint32_t b[4];
            ldsm_x4(b, sK + tile_off(8 * (lm & 1) + lr, kk + (lm >> 1)));
            uint32_t a[4];
            if (kk < HD / 16)
            {
                a[0] = qf[kk][0]; a[1] = qf[kk][1]; a[2] = qf[kk + 1][0]; a[3] = qf[kk + 1][1];
            }
            else
            {
                ldsm_x4(a, sbase + qh_off(16 * warp + lr + (lm & 1) * 8, kk - HD / 16 + (lm >> 1)));
            }
            mma1688(s[0], a[0], a[1], b[0]);
            mma1688(s[1], a[0], a[1], b[1]);
            mma1688(s[0], a[2], a[3], b[2]);
            mma1688(s[1], a[2], a[3], b[3]);
        }
        store_tile(BN * HD * 2, stage);

        // K(kt+1) in flight during softmax and P V
        const bool more = kt + 1 < n_tiles;
        if (more) load_tile(k_, k_row, n0 + BN, stage);

        // Mask, online softmax (log2 domain)
        const bool need_mask = (n0 + BN > Tkv) || (causal && n0 + BN - 1 > m0 + 16 * warp + offs);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            const int row = row0 + hh * 8;
            float mx = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    float x = s[nt][2 * hh + e] * scale_log2;
                    if (need_mask)
                    {
                        int key = n0 + 8 * nt + 2 * cq + e;
                        if (key >= Tkv || (causal && key > row + offs)) x = -1e30f;
                    }
                    s[nt][2 * hh + e] = x;
                    mx = fmaxf(mx, x);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            const float m_new = fmaxf(mrow[hh], mx);
            const float alpha = exp2f(mrow[hh] - m_new);
            mrow[hh] = m_new;
            float sum = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    float p = exp2f(s[nt][2 * hh + e] - m_new);
                    s[nt][2 * hh + e] = p;
                    sum += p;
                }
            lrow[hh] = lrow[hh] * alpha + sum;
            // Unconditional: a branch on alpha == 1 costs more than the multiplies
            #pragma unroll
            for (int dn = 0; dn < HD / 8; ++dn)
            {
                oacc[dn][2 * hh] *= alpha;
                oacc[dn][2 * hh + 1] *= alpha;
            }
        }
        __syncthreads();                                   // V(kt) visible, K(kt) no longer read

        // O += P V : A = P (key k-step j = n8 tile j of S), B = V (k = key, n = d) via ldmatrix.trans.
        // Each 16-key slice accumulates in fp16 (P <= 1, 16 terms; twice the fp32-accumulate HMMA rate) and is
        // then added into the fp32 O accumulators
        uint32_t pa[BN / 8][2];
        #pragma unroll
        for (int j = 0; j < BN / 8; ++j)
        {
            pa[j][0] = pack_h2(s[j][0], s[j][1]);
            pa[j][1] = pack_h2(s[j][2], s[j][3]);
        }
        #pragma unroll
        for (int dn = 0; dn < HD / 8; dn += 4)
        {
            uint32_t t[4][2] = {};
            #pragma unroll
            for (int j = 0; j < BN / 8; ++j)
            {
                uint32_t b[4];
                ldsm_x4_t(b, sV + tile_off(8 * j + lr, dn + lm));
                #pragma unroll
                for (int x = 0; x < 4; ++x) mma1688h(t[x], pa[j][0], pa[j][1], b[x]);
            }
            #pragma unroll
            for (int x = 0; x < 4; ++x)
            {
                float2 lo = __half22float2(*reinterpret_cast<half2*>(&t[x][0]));
                float2 hi = __half22float2(*reinterpret_cast<half2*>(&t[x][1]));
                oacc[dn + x][0] += lo.x; oacc[dn + x][1] += lo.y;
                oacc[dn + x][2] += hi.x; oacc[dn + x][3] += hi.y;
            }
        }
        if (more) store_tile(0, stage);
        __syncthreads();                                   // K(kt+1) visible, V(kt) no longer read
    }

    // Normalize and store
    #pragma unroll
    for (int hh = 0; hh < 2; ++hh)
    {
        float l = lrow[hh];
        l += __shfl_xor_sync(0xffffffff, l, 1);
        l += __shfl_xor_sync(0xffffffff, l, 2);
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        const int row = row0 + hh * 8;
        if (row < Tq)
        {
            half* o_ = o + ((int64_t) row * Hq + hq) * HD;
            #pragma unroll
            for (int dn = 0; dn < HD / 8; ++dn)
                *reinterpret_cast<uint32_t*>(o_ + 8 * dn + 2 * cq) = pack_h2(oacc[dn][2 * hh] * inv, oacc[dn][2 * hh + 1] * inv);
        }
    }
#endif
}

// q [Tq, Hq, 256], k/v [Tkv, Hkv, 256] (last dim contiguous), o [Tq, Hq, 256] contiguous
void fa75_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, double scale, bool causal)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf && v.dtype() == at::kHalf && o.dtype() == at::kHalf, "fp16");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && o.dim() == 3, "3-d tensors");
    TORCH_CHECK(q.size(2) == HD && k.size(2) == HD && v.size(2) == HD, "head_dim 256");
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && o.is_contiguous(), "contiguous head dim");
    TORCH_CHECK(q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0 && k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 &&
                v.stride(0) % 8 == 0 && v.stride(1) % 8 == 0, "16-byte aligned strides");
    int Tq = q.size(0), Hq = q.size(1), Tkv = k.size(0), Hkv = k.size(1);
    TORCH_CHECK(Hq % Hkv == 0 && v.size(0) == Tkv && v.size(1) == Hkv, "shapes");
    dim3 grid((Tq + BM - 1) / BM, Hq);
    fa75_kernel<<<grid, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) q.data_ptr(), q.stride(0), q.stride(1),
        (const half*) k.data_ptr(), k.stride(0), k.stride(1),
        (const half*) v.data_ptr(), v.stride(0), v.stride(1),
        (half*) o.data_ptr(), Tq, Tkv, Hq, Hkv, (float)(scale * 1.4426950408889634), causal ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#else

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

void fa75_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, double scale, bool causal)
{
    TORCH_CHECK(false, "Turing (sm_75) kernel not available on ROCm");
}

#endif
