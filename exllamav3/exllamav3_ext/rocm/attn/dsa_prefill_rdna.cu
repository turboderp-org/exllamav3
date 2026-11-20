// =============================================================================
// DSA prefill attention for RDNA3.5 (gfx1151), WMMA f16
// =============================================================================
//
// A HIP replacement for the Triton MQA prefill kernel (attention_fn/dsa_mqa.py,
// _dsa_prefill_mqa_kernel) in one case: gathered (top-k) mode with a sliding
// window, sinks, the eq. 26 output de-rotation and an fp16 pool, at the DeepSeek
// V4 Flash / V4.1 Flash shape (64 heads, 448 + 64 latent, V = K). dsa_attn only
// sends it calls that match; everything else stays on the Triton kernels.
//
// One 512-thread block per query row: 4 head groups of 16 heads x 4 quarters of
// the 512 output columns. Keys come in chunks of 32 (the window rows by absolute
// position, oldest first, then the gathered pool entries through the block
// table). Each chunk is staged once in LDS and used for both QK and PV, while the
// next chunk is fetched into registers. Scores are fp32, the softmax is online,
// p is rounded to fp16 for PV (as the Triton kernel does), and the sink logit
// only enters the denominator. The epilogue de-rotates the rope pairs at the
// query position and stores group-major.
//
// The block layout (one block a row, chunks staged once for QK and PV, prefetch
// into registers) follows ds41_sparse_attn_pf in Kyojin (Yamz-Labs/kyojin, MIT).
// This is a rewrite for exllamav3's paged pool and window layout, not a copy.
//
// On gfx1151 the kernel is built for two blocks per WGP (waves_per_eu 8, <= 192
// VGPRs), so one block's barriers overlap the other's math: 1.24-1.37x against the
// Triton kernel, where the one-block build is 1.01-1.02x. Other gfx11 targets
// (gfx11-generic included, which must fit the 1,024-VGPR parts) get one block
// per WGP: at 8 waves per EU there the kernel would spill.
//
// Not bitwise identical to the Triton kernel (different accumulation order).
//
// Fragment layout (wave32): A: lane l, elem r -> A[l & 15][r];
// B: lane l, elem r -> B[r][l & 15]; C: lane l, reg r -> C[2r + (l >> 4)][l & 15]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <string.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../../util.h"
#include "../../util.cuh"
#include "dsa_prefill_rdna.cuh"

// Device code needs gfx11 WMMA; other ROCm targets get an empty kernel, and the host side
// refuses to launch on them
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX11__)
#define DSA_PF_DEVICE_CODE 1
#endif

namespace dsa_pf_rdna {

typedef _Float16 half_t;
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t v4u __attribute__((ext_vector_type(4)));

constexpr int HD = 512;           // latent row: nope + rope
constexpr int DC = 448;           // nope part
constexpr int DR = 64;            // rope part
constexpr int NHEAD = 64;
constexpr int DQ = 128;           // output columns a wave (quarter)
constexpr int LDK = 520;          // LDS pitch of a staged key row in halves (1,040 B, 16-B aligned)

#if defined(__gfx1151__)
#define DSA_PF_OCCUPANCY __attribute__((amdgpu_waves_per_eu(8, 8)))
#else
#define DSA_PF_OCCUPANCY
#endif

#if defined(DSA_PF_DEVICE_CODE)

__device__ __forceinline__ v8f wmma(v16h a, v16h b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ __forceinline__ v16h ld16h(const half_t* p) {
    v4u a = *(const v4u*) p, b = *(const v4u*) (p + 8);
    union { v4u u[2]; v16h h; } x; x.u[0] = a; x.u[1] = b; return x.h;
}

template <int BN>
__device__ __forceinline__ void dsa_pf_body(
    const half_t* __restrict__ q, const half_t* __restrict__ ring, const half_t* __restrict__ kvc,
    const half_t* __restrict__ pool_c, const half_t* __restrict__ pool_r, const int* __restrict__ bt,
    const int* __restrict__ idx, const float* __restrict__ sink, const float* __restrict__ derot, half_t* __restrict__ out,
    int R, int K_pad, int k_len, int win_len, int q_pos0, int win_floor, int ring_beg, int npr, int page_size, int hpg,
    float scale, half_t* sK, float* sS, int* sTok)
{
    constexpr int NT = BN / 16;                 // key tiles a chunk = QK waves of a head group
    constexpr int VPT = BN * 64 / 512;          // 16-B vectors a thread fetches a chunk (64 vectors a 512-half row)
    static_assert(BN % 16 == 0 && NT <= 4 && (BN * 64) % 512 == 0, "BN");
    const int row = blockIdx.x;
    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31;
    const int hg = wv & 3, dq = wv >> 2;
    const int n = lane & 15, hh = lane >> 4;
    const int q_abs = q_pos0 + row;
    const int NK = win_len + k_len;
    const int nch = (NK + BN - 1) / BN;

    v8f acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -__builtin_inff(), l = 0.0f;
    const half_t* qh = q + ((int64_t) row * NHEAD + hg * 16 + n) * HD;

    // Fetch: vector v = tid + 512 i of the chunk is key v >> 6, 16-B part v & 63 (consecutive threads, consecutive bytes)
    v4u pre[VPT];
    bool pok[VPT];
    auto fetch = [&](int c) {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int v = tid + 512 * i, kk = v >> 6, part = v & 63;
            const int j = c * BN + kk;
            const v4u* src = nullptr;
            bool ok = false;
            if (c < nch && j < win_len) {
                const int ap = q_abs - (win_len - 1 - j);          // window, oldest first
                ok = ap >= win_floor;
                if (ok) src = (const v4u*) (ap >= q_pos0 ? kvc + (int64_t) (ap - q_pos0) * HD
                                                        : ring + (int64_t) (ap - ring_beg) * HD) + part;
            } else if (c < nch && j < NK) {
                const int id = idx[(int64_t) row * K_pad + (j - win_len)];
                ok = id >= 0;
                if (ok) {
                    const int64_t tok = (int64_t) bt[(int64_t) row * npr + id / page_size] * page_size + id % page_size;
                    src = part < DC / 8 ? (const v4u*) (pool_c + tok * DC) + part
                                        : (const v4u*) (pool_r + tok * DR) + (part - DC / 8);
                }
            }
            pok[i] = ok;
            pre[i] = ok ? *src : (v4u) {0, 0, 0, 0};
        }
    };

    fetch(0);
    for (int c = 0; c < nch; ++c) {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int v = tid + 512 * i, kk = v >> 6, part = v & 63;
            ((v4u*) (sK + kk * LDK))[part] = pre[i];
            if (part == 0) sTok[kk] = pok[i] ? 1 : -1;
        }
        __syncthreads();
        fetch(c + 1);                                          // global latency hides behind this chunk's math
        const int any = __syncthreads_or(tid < BN && sTok[tid] >= 0);
        if (!any) { __syncthreads(); continue; }
        if (dq < NT) {
            v8f S = {0, 0, 0, 0, 0, 0, 0, 0};
            const half_t* kp = sK + (16 * dq + n) * LDK;
#pragma unroll 4
            for (int ks = 0; ks < HD / 16; ++ks) S = wmma(ld16h(kp + 16 * ks), ld16h(qh + 16 * ks), S);
            float* ps = sS + ((hg * NT + dq) * 8) * 32 + lane;
#pragma unroll
            for (int r = 0; r < 8; ++r) ps[r * 32] = S[r];
        }
        __syncthreads();
        float s[NT][8], mx = -__builtin_inff();
        bool ok[NT][8];
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                ok[t][r] = sTok[16 * t + 2 * r + hh] >= 0;
                s[t][r] = ok[t][r] ? sS[((hg * NT + t) * 8 + r) * 32 + lane] * scale : -__builtin_inff();
                mx = fmaxf(mx, s[t][r]);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16));
        const float m_new = fmaxf(m, mx);
        const float m_exp = (m_new == -__builtin_inff()) ? 0.0f : m_new;
        float sum = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                s[t][r] = ok[t][r] ? __expf(s[t][r] - m_exp) : 0.0f;   // s now holds p
                sum += s[t][r];
            }
        const float alpha = (m == -__builtin_inff()) ? 0.0f : __expf(m - m_exp);
        sum += __shfl_xor(sum, 16);
        l = l * alpha + sum;
        m = m_new;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[nt][i] *= alpha;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            v16h pf;                                           // B operand: P[key 16t + r][head n], 16 keys a lane
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const half_t own = (half_t) s[t][r];
                union { half_t h; uint16_t u; } cv; cv.h = own;
                const uint16_t other = (uint16_t) __shfl_xor((int) cv.u, 16);
                union { uint16_t u; half_t h; } oc; oc.u = other;
                pf[2 * r] = hh == 0 ? own : oc.h;
                pf[2 * r + 1] = hh == 0 ? oc.h : own;
            }
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                v16h A;                                        // A operand: V^T[d][key], d = this lane's column
#pragma unroll
                for (int r = 0; r < 16; ++r) A[r] = sK[(16 * t + r) * LDK + dq * DQ + 16 * nt + n];
                acc[nt] = wmma(A, pf, acc[nt]);
            }
        }
        __syncthreads();
    }

    // Epilogue: sink, normalise, de-rotate the rope pairs, group-major store. Lane (head n, parity hh) holds column
    // d = dq * 128 + 16 nt + 2 r + hh: the pair (even, odd) is (hh 0, hh 1) of the same r, exchanged by one shuffle.
    const float sk = sink[hg * 16 + n];
    const float m_f = fmaxf(m, sk);
    const float w = __expf(m - m_f);                           // m == -inf -> 0
    const float den = l * w + __expf(sk - m_f);
    const float scl = w / den;
    const int h = hg * 16 + n;
    half_t* ob = out + ((int64_t) (h / hpg) * R + row) * (int64_t) (hpg * HD) + (int64_t) (h % hpg) * HD + dq * DQ;
    const float pos = (float) q_abs;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const float own = acc[nt][r] * scl;
            const float other = __shfl_xor(own, 16);
            if ((r & 1) == hh) {
                float lo = hh == 0 ? own : other, hi = hh == 0 ? other : own;
                const int d0 = dq * DQ + 16 * nt + 2 * r;
                if (d0 >= DC) {
                    const float th = derot[(d0 - DC) >> 1] * pos;
                    const float cs = cosf(th), sn = sinf(th);
                    const float lo2 = lo * cs - hi * sn, hi2 = hi * cs + lo * sn;
                    lo = lo2; hi = hi2;
                }
                union { half_t h[2]; uint32_t u; } pk; pk.h[0] = (half_t) lo; pk.h[1] = (half_t) hi;
                *(uint32_t*) (ob + 16 * nt + 2 * r) = pk.u;
            }
        }
}

#endif  // DSA_PF_DEVICE_CODE

// grid = R, block = 512. npr = 0 when one block-table row serves every query row
__global__ __launch_bounds__(512) DSA_PF_OCCUPANCY void dsa_prefill_kernel
(
    const half_t* __restrict__ q, const half_t* __restrict__ ring, const half_t* __restrict__ kvc,
    const half_t* __restrict__ pool_c, const half_t* __restrict__ pool_r, const int* __restrict__ bt,
    const int* __restrict__ idx, const float* __restrict__ sink, const float* __restrict__ derot, half_t* __restrict__ out,
    int R, int K_pad, int k_len, int win_len, int q_pos0, int win_floor, int ring_beg, int npr, int page_size, int hpg, float scale
)
{
#if defined(DSA_PF_DEVICE_CODE)
    __shared__ half_t sK[32 * LDK];
    __shared__ float sS[4 * 2 * 8 * 32];
    __shared__ int sTok[32];
    dsa_pf_body<32>(q, ring, kvc, pool_c, pool_r, bt, idx, sink, derot, out, R, K_pad, k_len, win_len, q_pos0, win_floor,
                    ring_beg, npr, page_size, hpg, scale, sK, sS, sTok);
#endif
}

}  // namespace dsa_pf_rdna

void dsa_prefill_rdna
(
    const at::Tensor& q,            // fp16 (R, 64, 512), rope slice pre-rotated
    const at::Tensor& ring,         // fp16 (ring_rows, 512): window rows below q_pos0 at abs - ring_beg
    const at::Tensor& kv_chunk,     // fp16 (R, 512): this chunk's rows at abs - q_pos0
    const at::Tensor& pool_c,       // fp16 (rows, 448): paged pool, nope part
    const at::Tensor& pool_r,       // fp16 (rows, 64): paged pool, rope part
    const at::Tensor& block_table,  // int32 (1 or R, pages): one row shared by every query row, or one a row
    const at::Tensor& indices,      // int32 (R, K_pad): pool entries, -1 padded
    const at::Tensor& sinks,        // fp32 (64,)
    const at::Tensor& derot,        // fp32 (32,): NEGATED rope inverse frequencies (eq. 26 de-rotation)
    at::Tensor& out,                // fp16 (groups, R, hpg * 512): group-major
    int64_t k_len,
    int64_t win_len,
    int64_t q_pos0,
    int64_t win_floor,
    int64_t ring_beg,
    int64_t page_size,
    double scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    TORCH_CHECK(strncmp(at::cuda::getDeviceProperties(q.get_device())->gcnArchName, "gfx11", 5) == 0,
                "dsa_prefill_rdna: needs a gfx11 device");
    TORCH_CHECK(q.dtype() == at::kHalf && q.is_contiguous() && q.dim() == 3 && q.size(1) == 64 && q.size(2) == 512, "dsa_prefill_rdna: q");
    const int R = (int) q.size(0);
    TORCH_CHECK(kv_chunk.dtype() == at::kHalf && kv_chunk.is_contiguous() && kv_chunk.size(0) >= R && kv_chunk.size(-1) == 512, "dsa_prefill_rdna: kv_chunk");
    TORCH_CHECK(ring.dtype() == at::kHalf && ring.is_contiguous() && ring.size(-1) == 512, "dsa_prefill_rdna: ring");
    TORCH_CHECK(pool_c.dtype() == at::kHalf && pool_c.is_contiguous() && pool_c.size(-1) == 448, "dsa_prefill_rdna: pool_c");
    TORCH_CHECK(pool_r.dtype() == at::kHalf && pool_r.is_contiguous() && pool_r.size(-1) == 64, "dsa_prefill_rdna: pool_r");
    TORCH_CHECK(block_table.dtype() == at::kInt && block_table.is_contiguous() && block_table.dim() == 2, "dsa_prefill_rdna: block_table");
    TORCH_CHECK(block_table.size(0) == 1 || block_table.size(0) == R, "dsa_prefill_rdna: block_table rows");
    TORCH_CHECK(indices.dtype() == at::kInt && indices.is_contiguous() && indices.dim() == 2 && indices.size(0) == R, "dsa_prefill_rdna: indices");
    TORCH_CHECK(k_len >= 0 && k_len <= indices.size(1), "dsa_prefill_rdna: k_len");
    TORCH_CHECK(sinks.dtype() == at::kFloat && sinks.numel() == 64, "dsa_prefill_rdna: sinks");
    TORCH_CHECK(derot.dtype() == at::kFloat && derot.numel() >= 32, "dsa_prefill_rdna: derot");
    TORCH_CHECK(out.dtype() == at::kHalf && out.is_contiguous() && out.dim() == 3 && out.size(1) == R && out.size(2) % 512 == 0, "dsa_prefill_rdna: out");
    const int hpg = (int) (out.size(2) / 512);
    TORCH_CHECK(hpg > 0 && out.size(0) * hpg == 64, "dsa_prefill_rdna: out groups");
    TORCH_CHECK(page_size > 0 && win_len >= 0 && win_len <= 4096, "dsa_prefill_rdna: page_size / win_len");
    if (R == 0) return;
    const int npr = block_table.size(0) > 1 ? (int) block_table.size(1) : 0;
    hipStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    using namespace dsa_pf_rdna;
    hipLaunchKernelGGL(dsa_prefill_kernel, dim3(R), dim3(512), 0, stream,
        (const half_t*) q.data_ptr(), (const half_t*) ring.data_ptr(), (const half_t*) kv_chunk.data_ptr(),
        (const half_t*) pool_c.data_ptr(), (const half_t*) pool_r.data_ptr(), (const int*) block_table.data_ptr(),
        (const int*) indices.data_ptr(), (const float*) sinks.data_ptr(), (const float*) derot.data_ptr(),
        (half_t*) out.data_ptr(), R, (int) indices.size(1), (int) k_len, (int) win_len, (int) q_pos0, (int) win_floor,
        (int) ring_beg, npr, (int) page_size, hpg, (float) scale);
    cuda_check(hipPeekAtLastError());
}
