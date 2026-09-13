#pragma once

// sm_70 GEMV kernel — m8n8k4 tensor-core variant of exl3_gemv_kernel.cuh.
//
// Structure is cloned from the sm80 kernel (warp-K-split, no mainloop sync,
// register-prefetch B streaming, cooperative grid.syncs, identical launch
// signature for graph-parameter compatibility). Differences:
//
// 1. Inner op: mma.m8n8k4.row.col.f32.f16.f16.f32 (native on Volta) instead
//    of mma.m16n8k16 with fp16 accumulation. C fragments stay f32; the
//    fp16 fold cadence disappears.
// 2. Fragment ownership: the sm80 decode produces values in m16n8k16 B
//    fragment lane order; the m8n8k4 consumer needs a different lane→cell
//    map. The exchange is the validated closed-form shuffle pattern in
//    exl3_sm70_map.cuh (shfl_src/shfl_slot): 8 __shfl per lane per tile,
//    after which the same values feed both A (activation) and B (weight)
//    operands.
// 3. Activation fragment: lane loads x[k] for k = 4*kh + i (kh = its MMA
//    computation's k-quad, i = 0..3) directly — 2 .f16x2 per tile.
// 4. Accumulators: 8 f32 per lane per (tile, mh) — 2 MMAs per tile; the
//    fp16 fold cadence is gone (f32 accumulate throughout).
// 5. Cross-warp reduction: same sh_red structure; the per-lane store index
//    follows the m8n8k4 C map (c_row/c_col) instead of the m16n8k16 map.
//
// Template parameters and launch signature match exl3_gemv_kernel so the
// dispatcher can select either implementation per architecture.

#include <cooperative_groups.h>
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_sm70_map.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"

#define EXL3_GEMV_SM70_MAX_M 8

namespace exl3_gemv_sm70_ns {

__device__ __forceinline__ void mma_ab_sm70(
    const uint32_t a0, const uint32_t a1,
    const uint32_t b0, const uint32_t b1,
    float* d)
{
    exl3_sm70::mma_m8n8k4_rc_f32(a0, a1, b0, b1, d);
}

}  // namespace exl3_gemv_sm70_ns

// MMODE: 0 = single row (m == 1), 1 = batched (m <= EXL3_GEMV_SM70_MAX_M)
// CFG: 0 = narrow (512 threads, 2 n-tiles/warp, 16 k-splits),
//      1 = wide   (256 threads, 4 n-tiles/warp, 8 k-splits)
template <int bits, bool c_fp32, int cb, int MMODE, int CFG, bool SMEM_STAGE>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256)
void exl3_gemv_sm70_kernel(EXL3_GEMM_ARGS)
{
    static_assert(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 7 || bits == 8,
        "exl3_gemv_sm70_kernel supports 2-8 bpw");
    constexpr int WK   = CFG == 0 ? 16 : 8;     // k-split (warps per block)
    constexpr int WNT  = CFG == 0 ? 2 : 4;      // adjacent n-tiles per warp
    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth
    // K >= 5: the tile is 8*bits >= 40 words — more than the 32 lanes
    // hold in one load, so each tile takes 2 loads (64 slots >= 8*bits).
    constexpr int LOADS = bits >= 5 ? 2 * WNT : (bits == 2 ? WNT / 2 : WNT);  // warp loads per k-slice
    constexpr int COLS = WNT * 16;

    constexpr int TWORDS = 8 * bits;                        // uint32 per 16x16 tile
    constexpr int THREADS = WK * 32;
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_SM70_MAX_M;
    constexpr int LSTRIDE = bits == 3 ? 24 : 32;            // uint32 per load
    static_assert(bits != 2 || WNT % 2 == 0, "2 bpw packs two tiles per warp load");

    auto grid = cooperative_groups::this_grid();

    // Input scales and Hadamard transform, same as exl3_gemv_kernel
    {
        int total_warps = size_m * size_k / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + this_warp * 128,
                A_had + this_warp * 128,
                suh + (this_warp * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );

        grid.sync();
        A = A_had;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int num_groups = size_n / COLS;

    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;   // uint32 per k-slice row
    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    // Activation fragment: this lane's k-quad base (per MMA computation)
    // and row for the batched mode. kh = (lane >> 2) & 3 selects the k-quad;
    // the lane loads x[k] for k = 4*kh .. 4*kh+3 (2 .f16x2).
    const int kh = (lane >> 2) & 3;
    // Batched mode: m-row this lane's D cells cover (row = lane&1 + quad)
    const int m_row_lo = exl3_sm70::c_row(lane, 0);
    const size_t a_row0 = (size_t) m_row_lo * (size_k / 2);
    // MMODE 0: all lanes compute the same single row — every lane owns
    // D cells (m-row 0) via the C map, so all lanes load the activation.
    // MMODE 1: gate on the lane's m-row being in range.
    const bool row_ok = MMODE == 0 ? true : (m_row_lo < size_m);
    // lanes that own D cells (all 32 do — the C map covers 8 rows with the
    // 8 participating lanes per MMA; the other 24 lanes' accumulators are
    // discarded by the reduction store predicate below).

    // Per-lane extraction constants (see dq8_aligned_2bits / dq8<3, cb, 4> in exl3_dq.cuh)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float sh_red[WK][ROWS][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[SMEM_STAGE ? WK : 1][SMEM_STAGE ? LOADS * LSTRIDE : 1];

    for (int group = blockIdx.x; group < num_groups; group += gridDim.x)
    {
        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;

        // Prefetch ring (indices must be compile-time or pf lands in local memory)
        // K >= 5: each tile spans 2 loads; load l covers tile (l >> 1),
        // words (l & 1) * 32 + lane. K <= 4: one load per tile, offset
        // l * LSTRIDE (== TWORDS).
        auto ld_b = [&] (int i, int l) -> uint32_t
        {
            // bp = tile 0, word lane. For K >= 5 each tile spans 2 loads:
            // load l reads tile (l >> 1), word (l & 1) * 32 + lane.
            if constexpr (bits >= 5)
                return __ldcs(bp + (size_t) i * slice_stride + (l >> 1) * TWORDS + (l & 1) * 32);
            else if constexpr (bits == 3)
                return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
        };

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        // Accumulators: [tile][mh][8 f32]
        float acc[WNT][2][8] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (SMEM_STAGE)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                {
                    // K >= 5: load l holds word (l & 1) * 32 + lane of
                    // tile (l >> 1). K <= 4: load l = tile l, word lane.
                    if constexpr (bits >= 5)
                        sh_stage[warp][(l >> 1) * TWORDS + (l & 1) * 32 + lane] = bw[l];
                    else
                        sh_stage[warp][l * LSTRIDE + lane] = bw[l];
                }
                __syncwarp();
            }

            // A fragment base for this k-slice (half2 units); the quad
            // offset kh*2 is added per software-quad iteration below.
            const size_t a_slice = (size_t) (ks0 + i) * 8;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                // Decode 8 values per lane (m16n8k16 B order) — unchanged
                uint32_t v[8];
                if constexpr (SMEM_STAGE)
                {
                    const uint32_t* tp = &sh_stage[warp][t * TWORDS];
                    if constexpr (bits >= 5)
                    {
                        // Verbatim upstream decode on the shared-memory
                        // tile, mirroring dq_dispatch for bits 5-8:
                        // dq4 pairs for 5/6/8, dq2x2 pairs for 7.
                        FragB f0, f1;
                        if constexpr (bits == 7)
                        {
                            dq2x2<bits, cb>(tp, lane << 3, f0);
                            dq2x2<bits, cb>(tp, (lane << 3) + 4, f1);
                        }
                        else
                        {
                            dq4<bits, cb>(tp, lane << 3, f0);
                            dq4<bits, cb>(tp, (lane << 3) + 4, f1);
                        }
                        v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                        v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                    }
                    else if constexpr (bits == 4)
                    {
                        uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                        FragB f0, f1;
                        exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                        v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                        v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                    }
                    else if constexpr (bits == 2)
                    {
                        FragB f0, f1;
                        exl3_gemv_ns::dq8_regs_2bits<cb>(tp[x_src_a], tp[x_src_b], lane << 3, f0, f1);
                        v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                        v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                    }
                    else
                    {
                        FragB f0, f1;
                        exl3_gemv_ns::dq8_regs_3bits<cb>(tp[x_src_a], tp[x_src_b], x_s2, f0, f1);
                        v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                        v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                    }
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    FragB f0, f1;
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                    v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                    v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                }
                else if constexpr (bits == 2)
                {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    FragB f0, f1;
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                    v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                    v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                }
                else if constexpr (bits >= 5)
                {
                    // Tile t spans loads 2t (words 0..31) and 2t+1 (words
                    // 32..63); dq8_regs_gen gathers via shuffles.
                    FragB f0, f1;
                    exl3_gemv_ns::dq8_regs_gen<bits, cb>(&bw[2 * t], lane << 3, f0, f1);
                    v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                    v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    FragB f0, f1;
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                    v[0] = *reinterpret_cast<const uint32_t*>(&f0[0]); v[1] = *reinterpret_cast<const uint32_t*>(&f0[1]);
                    v[2] = *reinterpret_cast<const uint32_t*>(&f1[0]); v[3] = *reinterpret_cast<const uint32_t*>(&f1[1]);
                }

                // Software quad loop: each lane processes all 4 k-quads,
                // accumulating its own D cells — no cross-group sum.
                //
                // Shuffle map (validated 1024/1024 against reconstruct):
                // for cell (k = 4*kh + vi, col = (lane&3) + 4*mh):
                //   src lane  = ((2*kh + (vi>>1)) & 3) + (((lane&3) + 4*mh) & 7) * 4
                //   v slot    = (kh>>1) & 1          (v[] entry — a .f16x2 pair)
                //   half      = vi & 1               (lo/hi within the pair)
                #pragma unroll
                for (int mh = 0; mh < 2; ++mh)
                #pragma unroll
                for (int qk = 0; qk < 4; ++qk)
                {
                    // A fragment: x[k] for k = 4*qk + 0..3 of this k-slice,
                    // at the lane's A-fragment row (ISA A map: row = lane&3,
                    // +4 for the high group) — the batch row for MMODE 1.
                    const int a_row = MMODE == 0 ? 0
                        : (int)((lane & 3) + 4 * (lane >= 16));
                    const size_t a_base = (size_t) a_row * (size_k / 2)
                        + a_slice + qk * 2;
                    const uint32_t a0 = row_ok ? *(const uint32_t*)(A2 + a_base) : 0u;
                    const uint32_t a1 = row_ok ? *(const uint32_t*)(A2 + a_base + 1) : 0u;

                    // B fragment: 4 cells (k = 4*qk + vi, col = (lane&3)
                    //   + 4*(lane>=16) + 8*mh)  — the ISA B map's col with
                    //   the mh 8-col group shift.
                    // vi {0,1} share one source pair (lo/hi); vi {2,3} share
                    // another — 2 shuffles per (mh, qk), and the pairs ARE
                    // the .f16x2 B operands (no extraction/repacking).
                    const int slot = (qk >> 1) + 2 * mh;
                    const int src_lo = 4 * (lane & 3) + 16 * (lane >= 16)
                        + 2 * (qk & 1);
                    const uint32_t b0 = __shfl_sync(0xffffffffu, v[slot], src_lo);
                    const uint32_t b1 = __shfl_sync(0xffffffffu, v[slot], src_lo + 1);
                    exl3_gemv_sm70_ns::mma_ab_sm70(a0, a1, b0, b1, acc[t][mh]);
                }

            }
        }
        }

        // Cross-warp reduction over the k splits. Per (mh, reg): D cell
        // (row = c_row(lane, reg), col = 8*mh + c_col(lane, reg)) of the tile.
        //
        // The 4 MMA computations (lane groups) each produce a full 8x8 D
        // for their k-quad — the same (row, col) cell is owned by 4 lanes
        // (one per group, differing in lane bits 2-3). Sum the 4 group
        // partials via shuffles, then store from group-0 lanes only.
        {
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                #pragma unroll
                for (int mh = 0; mh < 2; ++mh)
                {
                    #pragma unroll
                    for (int reg = 0; reg < 8; ++reg)
                    {
                        const float v = acc[t][mh][reg];
                        const int m_row = exl3_sm70::c_row(lane, reg);
                        const int n_col = t * 16 + mh * 8 + exl3_sm70::c_col(lane, reg);
                        if (MMODE == 0 ? (m_row == 0) : (m_row < ROWS))
                            sh_red[warp][MMODE == 0 ? 0 : m_row][n_col] = v;
                    }
                }
            }
        }
        __syncthreads();

        const int rows_out = MMODE == 0 ? 1 : min(size_m, ROWS);
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

    // Output scales and Hadamard transform, same semantics as the inner GEMM epilogue
    {
        grid.sync();

        int total_warps = size_m * size_n / 128;
        int warps_grid = gridDim.x * blockDim.x / 32;
        int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

        for(; this_warp < total_warps; this_warp += warps_grid)
        {
            if constexpr (c_fp32)
                had_ff_r_128_inner<false, true>
                (
                    ((const float*) C) + this_warp * 128,
                    ((float*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
            else
                had_hf_r_128_inner<false, true>
                (
                    ((const half*) C) + this_warp * 128,
                    ((half*) C) + this_warp * 128,
                    svh + (this_warp * 128) % size_n,
                    0.088388347648f  // 1/sqrt(128)
                );
        }
    }
}