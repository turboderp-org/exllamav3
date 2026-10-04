// =============================================================================
// exl3_gemm_inner for RDNA 3.5
//
// size_n_stride / blocks_n_full for sliced mgemm are mirrored from
// quant/exl3_gemm_inner.cuh. The 32 / 64-row MoE tiles (TILEBLOCKS_M > 1) are NOT ported; the
// static_assert(TILESIZE_M == 16) below is the guard.
// =============================================================================
//
// This is NOT a port of quant/exl3_gemm_inner.cuh. It keeps that kernel's
// structure -- three-stage software pipeline, split-K with a global lock, the
// FSTAGE unrolled main loop, the optional fused hadamard output -- but the
// tensor-core half is rebuilt on native WMMA, because the PTX primitives it
// uses have no RDNA equivalent:
//
//   mma.sync.m16n8k16  ->  __builtin_amdgcn_wmma_f32_16x16x16_f16_w32
//   ldmatrix (ldsm4)   ->  rdna_wmma::load_matrix_a
//   cp.async           ->  ordinary uint4 copies (RDNA 3.5 has no async
//                          global->LDS at this width)
//
// The fragment layouts do not correspond, so this could not be done by shimming
// primitives. PTX m16n8k16 distributes 8 halves/lane for A and 4 floats/lane
// for C; RDNA WMMA distributes 16 and 8. Everything that touches a C fragment
// -- the threadblock reduction, the partial-sum read/write, the output tile
// write -- is therefore re-derived rather than translated.
//
// Layout facts used here are verified, not inferred; see rocm/rdna_wmma.cuh.
//
// Differences from the CUDA kernel that are deliberate, and why:
//
//  1. No XOR swizzle on the A tile. The CUDA kernel swizzles A in shared memory so
//     that ldmatrix reads are bank-conflict free. We do not use ldmatrix:
//     load_matrix_a has each lane read 16 consecutive halves (32 B), a
//     different access pattern for which that swizzle is meaningless.
//     A is stored plainly and read with a padded stride instead (see
//     SH_A_STRIDE).
//
//  2. B goes through LDS between dequant and the mma. dq_dispatch emits
//     fragments in PTX mma B layout, which the CUDA kernel feeds straight into the
//     instruction. RDNA's B fragment wants lane L to hold *column* L%16 of the
//     16x16 block, so the dequantized values have to be transposed across
//     lanes. This is a shuffle-based unswizzle, matching the pattern in
//     reconstruct.cu.
//
//  3. fp32 accumulation always. The CUDA kernel has an fp16-accumulate path for
//     sm_86, where fp32-accumulate HMMA runs at half rate. RDNA has no such
//     penalty, and gfx11's fp16-accumulate C fragment is the same 8 VGPRs, so
//     fp32 is strictly better here -- same cost, more accuracy.
//
//  4. cp_async_wait/fence collapse. The copies complete before they return, so
//     the pipeline's "wait for stage" becomes an ordinary barrier. This costs
//     the load/compute overlap the CUDA kernel is built around and is the main reason
//     this kernel is expected to be slower than its CUDA counterpart; it is
//     also the only correct option on this hardware.
// =============================================================================

#ifndef EXL3_ROCM_GEMM_INNER_RDNA_H
#define EXL3_ROCM_GEMM_INNER_RDNA_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "../rdna_wmma.cuh"
#include "../../quant/exl3_dq.cuh"
#include "../../quant/hadamard_inner.cuh"
// For EXL3_GEMM_SH_B_DQ_STRIDE. The host-side LDS accounting in
// exl3_gemm_smem_bytes() has to agree with this kernel's indexing exactly, so
// both take the stride from that one macro rather than repeating the literal.
#include "exl3_kernel_map_rdna.cuh"

#define EXL3_GEMM_BASE_THREADS 256

// SMEM_MAX is established by exl3_kernel_map_rdna.cuh (64 KB). It is
// defined unconditionally at quant/exl3_gemm_inner.cuh:7, which is one of
// the reasons this file has to exist rather than being shimmed.
#ifndef SMEM_MAX
#define SMEM_MAX (64 * 1024)
#endif

template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
inline __device__
void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale,
    int size_n_stride = 0     // full width of B and C when computing a column slice (0: = size_n)
)
{
    // Sliced mode (exl3_mgemm): B and C rows are size_n_stride wide while this call
    // covers only size_n columns of them. Mirrors quant/exl3_gemm_inner.cuh.
    if (size_n_stride == 0) size_n_stride = size_n;

    constexpr int TILEBLOCKS_M = TILESIZE_M / 16;
    constexpr int TILEBLOCKS_K = TILESIZE_K / 16;
    constexpr int TILEBLOCKS_N = TILESIZE_N / 16;
    constexpr int NUM_WARPS = EXL3_GEMM_BASE_THREADS / 32;

    // One WMMA N-block per fragment, unlike the CUDA kernel's 2x (PTX m16n8k16
    // has N = 8, so it needs two fragments per 16-wide block; WMMA has N = 16).
    constexpr int FRAGS_N_PER_WARP = TILEBLOCKS_N / NUM_WARPS;

    // A is stored with a padded row stride. Rows are 16 halves (32 B); with 32
    // LDS banks of 4 B, an unpadded stride puts rows 0, 4, 8, 12 on the same
    // bank, so the 16 lanes of a load_matrix_a read collide 4 ways. Padding by
    // one 8-half group breaks that. This replaces the CUDA kernel's XOR swizzle, which
    // solves the same problem for a different access pattern.
    constexpr int SH_A_STRIDE = TILESIZE_K + 8;

    constexpr int sh_a_stage_size = TILESIZE_M * SH_A_STRIDE;                    // halfs
    // uint16 per 16x16 trellis tile; a half-integer rate (bits + 0.5, mul1) carries 16 * bits + 8
    constexpr int TILE_U16 = 16 * bits + (half_k ? 8 : 0);
    constexpr int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * TILE_U16;  // uint16s
    // 18, not 17 -- see EXL3_GEMM_SH_B_DQ_STRIDE in exl3_kernel_map_rdna.cuh
    // for why 18, and why this must not be retyped as a literal
    // here (the host-side LDS accounting drifted from it once already).
    constexpr int SH_B_DQ_STRIDE  = EXL3_GEMM_SH_B_DQ_STRIDE;
    // One 16x16 staging block per warp IN THE BLOCK, which is NUM_WARPS *
    // TILEBLOCKS_K -- not NUM_WARPS. NUM_WARPS counts warps per sub_k group
    // (EXL3_GEMM_BASE_THREADS / 32), but blockDim is
    // EXL3_GEMM_BASE_THREADS * TILEBLOCKS_K, so at TILEBLOCKS_K = 2 the block
    // holds 16 warps. Sizing this for 8 made the sub_k 0 and sub_k 1 warps that
    // share a warp_id stage DIFFERENT B fragments into the SAME buffer, with only
    // a __syncwarp between write and read -- no cross-sub_k ordering at all.
    constexpr int sh_b_dq_size    = NUM_WARPS * TILEBLOCKS_K * 16 * SH_B_DQ_STRIDE;  // halfs

    // sh_c serves two disjoint purposes: the cross-sub_k reduction (only when
    // TILEBLOCKS_K > 1) and the pre-hadamard output tile.
    //
    // The reduction stages one slot of 8 * FRAGS_N_PER_WARP floats per thread, and
    // only ONE sub_k's data is live at a time -- threadblock_reduce() serializes
    // over src with a __syncthreads() on either side of each exchange. So this is
    // sized for EXL3_GEMM_BASE_THREADS slots, not TILEBLOCKS_K * that, and both
    // sides of the exchange must index by t alone. (Sizing it per-sub_k instead
    // would be correct but wasteful: at TILESIZE_N = 256 it costs 32 KB and pushes
    // the 8-bit shape past the 64 KB static_assert below.)
    constexpr int sh_c_reduce = (TILEBLOCKS_K > 1)
        ? (8 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP) : 0;                   // floats
    constexpr int sh_c_size = MAX(sh_c_reduce, shmem_out_had ? TILESIZE_N * TILESIZE_M : 0);

    // Sanity checks. The N-block divisibility assert is the one the CUDA kernel
    // does not need: with TILESIZE_N = 192 the
    // integer division below silently drops a third of the output tile.
    static_assert(EXL3_GEMM_BASE_THREADS == 256, "Invalid kernel params");
    static_assert(TILESIZE_M == 16, "Invalid kernel params");
    static_assert(TILESIZE_K % 16 == 0, "Invalid kernel params");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert(TILEBLOCKS_N % NUM_WARPS == 0,
        "TILESIZE_N/16 must divide the warp count, or N-blocks are dropped");
    static_assert(FRAGS_N_PER_WARP >= 1, "Invalid kernel params");
    static_assert
    (
        SMEM_MAX >= 2 * SH_STAGES * sh_a_stage_size
                  + 2 * SH_STAGES * sh_b_stage_size
                  + 2 * sh_b_dq_size
                  + 4 * sh_c_size,
        "Invalid kernel params (insufficient LDS for shape -- RDNA has 64 KB)"
    );

    // Shared memory
    extern __shared__ half shared[];
    half*     sh_a    = shared;
    uint16_t* sh_b    = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    half*     sh_b_dq = (half*) (sh_b + SH_STAGES * sh_b_stage_size);
    float*    sh_c    = (float*) (sh_b_dq + sh_b_dq_size);

    // Thread index
    const int t       = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    const int sub_k   = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    const int warp_id = t / 32;
    const int lane_id = t % 32;

    // RDNA C fragment mapping, used by every output path below.
    const int c_row      = lane_id % 16;
    const int c_col_base = (lane_id >= 16) ? 1 : 0;

    // Dimensions
    const int tiles_k  = size_k / TILESIZE_K;
    const int tiles_n  = size_n / TILESIZE_N;
    const int blocks_n = tiles_n * TILEBLOCKS_N;
    // Column blocks of the full-width B row: slices index B relative to their own column
    // offset, but a k-tile row still spans the whole matrix (blocks_n keeps the lock index)
    const int blocks_n_full = size_n_stride / 16;

    const int num_slices = gridDim.x;
    const int slice_beg  = tiles_k * tiles_n * blockIdx.x / num_slices;
    const int slice_end  = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    const int slice_len  = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto index_k = [&] (int slice_i) { return (slice_i % tiles_k); };
    auto index_n = [&] (int slice_i) { return (slice_i / tiles_k); };

    const int slice_m = 0;

    // ---- Pipe 0: global -> shared -------------------------------------------
    int slice0_k     = index_k(slice_beg);
    int slice0_n     = index_n(slice_beg);
    int slice0_iters = slice_len;

    const int gl_a_stride_m = TILESIZE_M * size_k;
    const int gl_a_stride_k = TILESIZE_K;
    const half* gl_a_ptr = A + slice_m * gl_a_stride_m + slice0_k * gl_a_stride_k;
    half* sh0_a_ptr = sh_a + (slice0_iters % SH_STAGES) * sh_a_stage_size;

    // A is copied as uint4 (8 halves). Source rows are TILESIZE_K wide and
    // contiguous; destination rows use SH_A_STRIDE, so the row index has to be
    // recovered rather than copying a flat run.
    constexpr int A_VEC_PER_ROW = TILESIZE_K / 8;
    constexpr int load_a_iters  = CEIL_DIVIDE(TILESIZE_M * A_VEC_PER_ROW, EXL3_GEMM_BASE_THREADS);
    bool pred_a_gl[load_a_iters];
    int  load_a_gl[load_a_iters];
    int  load_a_sh[load_a_iters];
    #pragma unroll
    for (int i = 0; i < load_a_iters; ++i)
    {
        int idx = i * EXL3_GEMM_BASE_THREADS + t;
        int k = idx % A_VEC_PER_ROW;
        int m = idx / A_VEC_PER_ROW;
        load_a_gl[i] = m * (size_k / 8) + k;
        load_a_sh[i] = m * (SH_A_STRIDE / 8) + k;
        pred_a_gl[i] = (idx < TILESIZE_M * A_VEC_PER_ROW) && (m < size_m);
    }

    const int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * TILE_U16;
    const int gl_b_stride_n = TILEBLOCKS_N * TILE_U16;
    const uint16_t* gl_b_ptr = B + slice0_k * gl_b_stride_k + slice0_n * gl_b_stride_n;
    uint16_t* sh0_b_ptr = sh_b + (slice0_iters % SH_STAGES) * sh_b_stage_size;

    constexpr int load_b_iters = CEIL_DIVIDE(sh_b_stage_size / 8, EXL3_GEMM_BASE_THREADS);
    bool pred_b_gl[load_b_iters];
    int  load_b_gl[load_b_iters];
    #pragma unroll
    for (int i = 0; i < load_b_iters; ++i)
    {
        int idx = i * EXL3_GEMM_BASE_THREADS + t;
        int n = idx % (gl_b_stride_n / 8);
        int k = idx / (gl_b_stride_n / 8);
        load_b_gl[i] = k * (blocks_n_full * TILE_U16 / 8) + n;
        pred_b_gl[i] = idx < sh_b_stage_size / 8;
    }

    auto advance0 = [&] ()
    {
        slice0_k++;
        slice0_iters--;

        int stage = slice0_iters % SH_STAGES;
        sh0_a_ptr = sh_a + stage * sh_a_stage_size;
        sh0_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice0_k >= tiles_k)
        {
            slice0_k = 0;
            slice0_n++;
            gl_a_ptr = A + slice_m * gl_a_stride_m;
            gl_b_ptr = B + slice0_n * gl_b_stride_n;
        }
        else
        {
            gl_a_ptr += gl_a_stride_k;
            gl_b_ptr += gl_b_stride_k;
        }
    };

    // ---- Pipe 1: shared -> registers ----------------------------------------
    int slice1_k     = slice0_k;
    int slice1_n     = slice0_n;
    int slice1_iters = slice0_iters;

    half*     sh1_a_ptr = sh_a + (slice1_iters % SH_STAGES) * sh_a_stage_size;
    uint16_t* sh1_b_ptr = sh_b + (slice1_iters % SH_STAGES) * sh_b_stage_size;

    auto advance1 = [&] ()
    {
        slice1_k++;
        slice1_iters--;

        int stage = slice1_iters % SH_STAGES;
        sh1_a_ptr = sh_a + stage * sh_a_stage_size;
        sh1_b_ptr = sh_b + stage * sh_b_stage_size;

        if (slice1_k >= tiles_k) { slice1_k = 0; slice1_n++; }
    };

    // ---- Pipe 2: matmul and output ------------------------------------------
    int slice2_k     = slice0_k;
    int slice2_k0    = slice0_k;
    int slice2_n     = slice0_n;
    int slice2_iters = slice0_iters;

    const int gl_c_stride_n = TILESIZE_N;
    const int gl_c_stride_m = TILESIZE_M * size_n_stride;

    half*  gl_c_ptr_16 = ((half*)  C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;
    float* gl_c_ptr_32 = ((float*) C) + slice_m * gl_c_stride_m + slice2_n * gl_c_stride_n;

    WmmaFragA frag_a[FRAG_STAGES][TILEBLOCKS_M];
    WmmaFragB frag_b[FRAG_STAGES][FRAGS_N_PER_WARP];
    WmmaFragC frag_c[TILEBLOCKS_M][FRAGS_N_PER_WARP];

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;

        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            slice2_n++;
            if constexpr (c_fp32) gl_c_ptr_32 += gl_c_stride_n;
            else                  gl_c_ptr_16 += gl_c_stride_n;
        }
    };

    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                frag_c[m][n].clear();
    };

    // Global -> shared. Synchronous: see note 4 in the header comment.
    auto load_gl = [&] ()
    {
        if (sub_k) return;
        if (!slice0_iters) return;

        {
            const uint4* gl = (const uint4*) gl_a_ptr;
            uint4* sh = (uint4*) sh0_a_ptr;
            #pragma unroll
            for (int i = 0; i < load_a_iters; ++i)
                if (pred_a_gl[i]) sh[load_a_sh[i]] = gl[load_a_gl[i]];
        }
        {
            const uint4* gl = (const uint4*) gl_b_ptr;
            uint4* sh = (uint4*) sh0_b_ptr;
            #pragma unroll
            for (int i = 0; i < load_b_iters; ++i)
                if (pred_b_gl[i]) sh[i * EXL3_GEMM_BASE_THREADS + t] = gl[load_b_gl[i]];
        }
        advance0();
    };

    // Shared -> registers.
    auto load_frags = [&] (int buf)
    {
        if (!slice1_iters) return;

        // A: lane L reads row L%16, 16 consecutive halves.
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            const half* a_ptr = sh1_a_ptr + m * 16 * SH_A_STRIDE + sub_k * 16;
            rdna_wmma::load_matrix_a(frag_a[buf][m], a_ptr, SH_A_STRIDE);
        }

        // B: dequantize, transpose across lanes, stage in LDS, then load in
        // WMMA layout. dq_dispatch emits the PTX fragment layout, which does
        // not match WMMA's -- see note 2.
        #pragma unroll
        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
        {
            const int n_idx = warp_id * FRAGS_N_PER_WARP + n;
            // Indexed by the BLOCK-wide warp id, not warp_id (which is t/32 and
            // therefore wraps at NUM_WARPS, aliasing sub_k 0 against sub_k 1).
            // Identical to warp_id whenever TILEBLOCKS_K == 1.
            half* B_lds = sh_b_dq + (warp_id + sub_k * NUM_WARPS) * 16 * SH_B_DQ_STRIDE;
            const uint32_t* b_quant =
                (const uint32_t*) (sh1_b_ptr + (sub_k * TILEBLOCKS_N + n_idx) * TILE_U16);

            FragB frag0, frag1;
            dq_dispatch<bits, cb, half_k>(b_quant, lane_id << 3, frag0, frag1);

            // Same shuffle pattern as reconstruct.cu: lanes with bit 2 clear
            // combine their own values with those of lane+4 and write the
            // 16x16 block row-major, stride 17 to avoid bank conflicts.
            half2 n0 = __shfl_down(frag0[0], 4, 32);
            half2 n1 = __shfl_down(frag0[1], 4, 32);
            half2 n2 = __shfl_down(frag1[0], 4, 32);
            half2 n3 = __shfl_down(frag1[1], 4, 32);

            if (!(lane_id & 4))
            {
                // Store the halves directly. Building m0..m7 with
                // __halves2half2(X, Y) and then pulling them apart with
                // __low2half / __high2half hands back X and Y unchanged -- a
                // round trip that costs 8 VGPRs of live state per lane for
                // nothing. Same as in the RDNA GEMV. Writes are grouped so each (frag, shuffled)
                // pair is fully consumed before the next, which gives the
                // allocator a clean signal about what can die early.
                const int r0 = (lane_id % 4) * 2;
                const int r1 = r0 + 1;
                const int r2 = r0 + 8;
                const int r3 = r0 + 9;
                const int c0 = (lane_id / 8) * 2;
                const int c1 = c0 + 8;

                #define B_IDX(row, col) ((row) * SH_B_DQ_STRIDE + (col))

                B_lds[B_IDX(r0, c0)]     = __low2half (frag0[0]);
                B_lds[B_IDX(r0, c0 + 1)] = __low2half (n0);
                B_lds[B_IDX(r1, c0)]     = __high2half(frag0[0]);
                B_lds[B_IDX(r1, c0 + 1)] = __high2half(n0);

                B_lds[B_IDX(r2, c0)]     = __low2half (frag0[1]);
                B_lds[B_IDX(r2, c0 + 1)] = __low2half (n1);
                B_lds[B_IDX(r3, c0)]     = __high2half(frag0[1]);
                B_lds[B_IDX(r3, c0 + 1)] = __high2half(n1);

                B_lds[B_IDX(r0, c1)]     = __low2half (frag1[0]);
                B_lds[B_IDX(r0, c1 + 1)] = __low2half (n2);
                B_lds[B_IDX(r1, c1)]     = __high2half(frag1[0]);
                B_lds[B_IDX(r1, c1 + 1)] = __high2half(n2);

                B_lds[B_IDX(r2, c1)]     = __low2half (frag1[1]);
                B_lds[B_IDX(r2, c1 + 1)] = __low2half (n3);
                B_lds[B_IDX(r3, c1)]     = __high2half(frag1[1]);
                B_lds[B_IDX(r3, c1 + 1)] = __high2half(n3);

                #undef B_IDX
            }

            // B_lds is warp-private -- now genuinely so, indexed by the
            // block-wide warp id rather than one that wraps at NUM_WARPS -- so
            // ordering
            // only has to hold within the wave. On wave32 the lanes are
            // converged here, so draining LDS is sufficient and a block-wide
            // __syncthreads() would be stronger than required.
            mem_fence();
            rdna_wmma::load_matrix_b(frag_b[buf][n], B_lds, SH_B_DQ_STRIDE);
            mem_fence();
        }
        advance1();
    };

    auto matmul = [&] (int buf)
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                rdna_wmma::mma_sync(frag_c[m][n], frag_a[buf][m], frag_b[buf][n]);
    };

    // Cross-sub_k reduction. Every shape in the RDNA shape *table* uses
    // TILESIZE_K = 16, so TILEBLOCKS_K == 1 and this compiles away -- but the MoE
    // kernel instantiates the same inner with MOE_TILESIZE_K = 32, so for that one
    // caller it is live. It was previously written as though it were unreachable:
    // the writer indexed by t while the reader indexed by (t + src * THREADS), so
    // sub_k 0 summed a region no one had written -- and, with sh_c undersized by
    // the same TILEBLOCKS_K factor, one past the end of the LDS block.
    //
    // Both sides must agree. They index by t alone: the exchange is serialized by
    // the __syncthreads() pair, so sub_k == src writes slot t and sub_k == 0 reads
    // slot t within the same iteration, and the next iteration cannot start writing
    // until the trailing barrier has released. Sizing and indexing this per-sub_k
    // is also correct but costs TILEBLOCKS_K times the LDS.
    auto threadblock_reduce = [&] ()
    {
        if constexpr (TILEBLOCKS_K > 1)
        {
            for (int src = 1; src < TILEBLOCKS_K; ++src)
            {
                if (sub_k == src)
                {
                    float* sh_red = sh_c + 8 * FRAGS_N_PER_WARP * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) *sh_red++ = frag_c[m][n][j];
                }
                __syncthreads();
                if (sub_k == 0)
                {
                    float* sh_red = sh_c + 8 * FRAGS_N_PER_WARP * t;
                    #pragma unroll
                    for (int m = 0; m < TILEBLOCKS_M; ++m)
                        #pragma unroll
                        for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) frag_c[m][n][j] += *sh_red++;
                }
                __syncthreads();
            }
        }
    };

    // Column of the output tile owned by element j of fragment n.
    auto out_col = [&] (int n, int j)
    {
        return (warp_id * FRAGS_N_PER_WARP + n) * 16 + j * 2 + c_col_base;
    };

    auto read_sum_gl = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            int row = m * 16 + c_row;
            if (row >= size_m) continue;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    int col = out_col(n, j);
                    if constexpr (c_fp32)
                        frag_c[m][n][j] += gl_c_ptr_32[row * size_n_stride + col];
                    else
                        frag_c[m][n][j] += __half2float(gl_c_ptr_16[row * size_n_stride + col]);
                }
        }
    };

    auto write_sum_gl = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            int row = m * 16 + c_row;
            if (row >= size_m) continue;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    int col = out_col(n, j);
                    if constexpr (c_fp32)
                        gl_c_ptr_32[row * size_n_stride + col] = frag_c[m][n][j];
                    else
                        gl_c_ptr_16[row * size_n_stride + col] = __float2half(frag_c[m][n][j]);
                }
        }
    };

    // Pre-hadamard: stage the finished tile row-major in shared memory.
    auto write_sum_tile_sh = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            int row = m * 16 + c_row;
            if (row >= size_m) continue;
            #pragma unroll
            for (int n = 0; n < FRAGS_N_PER_WARP; ++n)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    sh_c[row * TILESIZE_N + out_col(n, j)] = frag_c[m][n][j];
        }
    };

    auto output_had_sh_gl = [&] ()
    {
        int sh_warp = warp_id;
        constexpr int active_warps = EXL3_GEMM_BASE_THREADS / 32;
        for (;; sh_warp += active_warps)
        {
            int col = sh_warp % (TILESIZE_N / 128);
            int row = sh_warp / (TILESIZE_N / 128);
            if (row >= size_m) break;

            const float* had_in = sh_c + row * TILESIZE_N + col * 128;
            const half* post_scale_c = post_scale + slice2_n * gl_c_stride_n + col * 128;

            if constexpr (c_fp32)
            {
                float* had_out = gl_c_ptr_32 + row * size_n_stride + col * 128;
                had_ff_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
            else
            {
                half* had_out = gl_c_ptr_16 + row * size_n_stride + col * 128;
                had_fh_r_128_inner<false, true>(had_in, had_out, post_scale_c, 0.088388347648f);
            }
        }
    };

    auto reduce = [&] ()
    {
        threadblock_reduce();

        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[slice_m * blocks_n + slice2_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last  = lock_i + lock_d == tiles_k;

        if (!sub_k && !first) read_sum_gl();
        if (!sub_k && !last)  write_sum_gl();

        if (!sub_k && last)
        {
            if constexpr (shmem_out_had) write_sum_tile_sh();
            else                         write_sum_gl();
        }

        if constexpr (shmem_out_had)
        {
            if (last) __syncthreads();
            if (!sub_k && last) output_had_sh_gl();
        }

        barrier_release(lock, lock_d, last);
        clear_frag_c();
    };

    // With synchronous copies there is nothing outstanding to wait for, so the
    // stage wait is just the barrier that publishes the tile to the block.
    auto wait_stage = [&] () { __syncthreads(); };

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        load_gl();
    wait_stage();

    clear_frag_c();
    if constexpr (FRAG_STAGES > 1)
        load_frags(0);

    #define FSTAGE_OLD(_load, _mul) \
        load_gl(); \
        wait_stage(); \
        load_frags(_load); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \

    #define FSTAGE(_load, _mul) \
        load_gl(); \
        wait_stage(); \
        matmul(_mul); \
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; } \
        advance2(); \
        if (!slice2_iters) break; \
        load_frags(_load); \

    if constexpr (FRAG_STAGES == 1) { while (true) { FSTAGE_OLD(0, 0); } }
    if constexpr (FRAG_STAGES == 2) { while (true) { FSTAGE(1, 0); FSTAGE(0, 1); } }
    if constexpr (FRAG_STAGES == 3) { while (true) { FSTAGE(1, 0); FSTAGE(2, 1); FSTAGE(0, 2); } }
    if constexpr (FRAG_STAGES == 4) { while (true) { FSTAGE(1, 0); FSTAGE(2, 1); FSTAGE(3, 2); FSTAGE(0, 3); } }
    if constexpr (FRAG_STAGES == 5) { while (true) { FSTAGE(1, 0); FSTAGE(2, 1); FSTAGE(3, 2); FSTAGE(4, 3); FSTAGE(0, 4); } }

    #undef FSTAGE
    #undef FSTAGE_OLD
}

#endif // EXL3_ROCM_GEMM_INNER_RDNA_H
