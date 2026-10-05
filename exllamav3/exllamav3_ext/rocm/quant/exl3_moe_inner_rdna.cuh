#pragma once

// =============================================================================
// Pipelined MoE GEMM mainloop for RDNA (EXL3_ROCM_MOE_PIPE, default on)
// =============================================================================
//
// Replaces exl3_gemm_kernel_inner for the fused MoE kernel only. exl3_gemm and
// exl3_mgemm keep the shared inner (exl3_gemm_inner_rdna.cuh) untouched; the old
// MoE mainloop stays reachable in the same build with EXL3_ROCM_MOE_PIPE=0.
//
// The old loop staged every k-tile through LDS behind a block barrier, loaded from
// the sub_k == 0 threads only, and drained vmcnt(0) after every load, paying the
// DRAM round trip per k-tile. What this one does:
//
//   - B (the trellis): every wave loads its own FN 16x16 blocks of each k-tile
//     straight into a register ring DB k-tiles deep, only the R dwords per block
//     its lanes decode from (exl3_lane_plan). The ring is indexed by compile-time
//     slots (the loop is unrolled), so the loads stay in flight across tiles and
//     the compiler's s_waitcnt vmcnt(N) is exact.
//   - Decode: the GEMV tiles core's exact-integer decoder (exl3_dq_tile_decode, bit
//     for bit dq_dispatch), written TRANSPOSED into a wave-private staging block
//     ([n][k], 16 halves per row, halves swizzled against bank conflicts), so the
//     WMMA B fragment (lane L = column L%16, 16 consecutive k) is two ds_load_b128
//     instead of 16 ds_load_u16, 8 shuffles and 16 ds_store_b16. Same halves in the
//     same fragment slots: rdna_wmma.cuh fragment logic is not touched.
//   - A (the gathered expert input, shared by all waves): staged through LDS in
//     chunks of AG k-tiles, one uint4 per thread per chunk and one LDS barrier per
//     chunk. Per-wave global A loads throttled the B stream.
//   - LDS-only fences everywhere in the loop (__builtin_amdgcn_fence(..., "local")):
//     __syncthreads() and s_waitcnt(0) would drain the ring (vmcnt(0)).
//   - Row tiles of 16/32/48/64 (TILEBLOCKS_M = 1..4), chosen per expert by the
//     kernel, share one decoded B fragment across the row blocks.
//   - <= 32 KB of LDS and <= 192 VGPRs, so two 512-thread blocks fit a WGP (the
//     host confirms with the runtime occupancy query before launching two per WGP).
//
// Half-integer rates (EXL3_ROCM_HALF_MOE_PIPE): `bits` may be the pseudo
// width EXL3_HALF_BITS(K) of exl3_gemv_tiles_rdna.cuh -- the lane plan (4 dwords per
// block) and exl3_dq_tile_decode carry dq8_half; only BLK_BYTES depends on it here.
//
// Tile slicing (stream-K over tiles_k * tiles_n tiles per block), the fp16 partial
// sums through global memory, the lock protocol and the sub_k reduction are the old
// inner's: on the same grid the output is bit-identical to EXL3_ROCM_MOE_PIPE=0.
// =============================================================================

#include "exl3_gemm_inner_rdna.cuh"
#include "exl3_gemv_tiles_rdna.cuh"

// EXL3_MOE_PIPE_DB: B register-ring depth, k-tiles in flight per wave (2, 4 or 8)
#ifndef EXL3_MOE_PIPE_DB
#define EXL3_MOE_PIPE_DB 4
#endif
static_assert(EXL3_MOE_PIPE_DB == 2 || EXL3_MOE_PIPE_DB == 4 || EXL3_MOE_PIPE_DB == 8,
    "EXL3_MOE_PIPE_DB must be 2, 4 or 8");

// EXL3_MOE_PIPE_WPE: waves per SIMD the pipelined kernel is register-budgeted for. 8 (= 1536 / 8 = 192
// VGPRs) lets two 512-thread blocks share a WGP; the host still asks the runtime occupancy query before
// using it
#ifndef EXL3_MOE_PIPE_WPE
#define EXL3_MOE_PIPE_WPE 8
#endif

namespace moe_pipe {

typedef __attribute__((address_space(1))) const uint32_t g_u32;
typedef uint32_t u32x4_t __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8_t __attribute__((ext_vector_type(8)));
typedef __attribute__((address_space(1))) const u32x4_t g_u4;

// Wave-private LDS: a wave's own LDS operations execute in order, so only the
// compiler has to be stopped from reordering them. Same pattern as __syncwarp in
// compat.h; emits no s_waitcnt.
__device__ __forceinline__ void wave_lds_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

// Block barrier that orders LDS only: s_waitcnt lgkmcnt(0) + s_barrier, no vmcnt
// drain and no L0 invalidate (unlike __syncthreads() in WGP mode).
__device__ __forceinline__ void block_lds_sync()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// A chunk depth: AG k-tiles per chunk, AG * row blocks <= 8 so a chunk is at most one
// uint4 per thread. 8 rather than 4 halves the chunk barriers.
#ifndef EXL3_MOE_PIPE_AG
#define EXL3_MOE_PIPE_AG 8
#endif
template <int TILEBLOCKS_M> struct PipeAChunk
{
    static constexpr int AG = (EXL3_MOE_PIPE_AG / TILEBLOCKS_M) > 0 ? EXL3_MOE_PIPE_AG / TILEBLOCKS_M : 1;
};

// B staging rows are 16 halves (32 B), not padded to 24 as in the shared inner: the
// staging set plus the A chunks must stay <= 32 KB. The 2-way bank conflict on the
// stores this leaves is broken by swapping the two 16-byte halves of rows whose column
// has bit 2 set (bt_swz below).
constexpr int BT_STRIDE = 16;

// EXL3_MOE_PIPE_AHEAD: 1 = decode one k-tile ahead into a second staging set (hides the
// LDS round trip, but the extra 16 KB of LDS goes over the 32 KB that two blocks per
// WGP need), 0 = one set, decode and use in the same step (default).
#ifndef EXL3_MOE_PIPE_AHEAD
#define EXL3_MOE_PIPE_AHEAD 0
#endif
constexpr int BT_SETS = EXL3_MOE_PIPE_AHEAD ? 2 : 1;
// A chunk rows are TILESIZE_K halves, unpadded; the 16-byte chunks of a row are XOR-
// swizzled by row (a_swz) instead, which removes the ds_load_b128 bank conflicts the
// padding used to (padding costs more LDS, which decides whether two blocks fit a
// WGP).

// LDS needed by moe_gemm_pipe, bytes. The sub_k reduction scratch aliases the B staging
// set that is free at a column end, so it adds nothing. Independent of bits.
template <int bits, int TILEBLOCKS_M, int TILESIZE_K, int TILESIZE_N>
constexpr int smem_bytes()
{
    constexpr int TILEBLOCKS_K = TILESIZE_K / 16;
    constexpr int NUM_WARPS = EXL3_GEMM_BASE_THREADS / 32;
    constexpr int FN = TILESIZE_N / 16 / NUM_WARPS;
    constexpr int SH_A_STRIDE = TILESIZE_K;
    constexpr int AG = PipeAChunk<TILEBLOCKS_M>::AG;
    constexpr int warps = NUM_WARPS * TILEBLOCKS_K;
    return BT_SETS * warps * FN * 16 * BT_STRIDE * 2 + 2 * AG * TILEBLOCKS_M * 16 * SH_A_STRIDE * 2;
}

template <int bits, int cb, int TILEBLOCKS_M, int TILESIZE_K, int TILESIZE_N>
__device__ __forceinline__
void moe_gemm_pipe
(
    const half* __restrict__ A,
    const uint16_t* __restrict__ B,
    half* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks
)
{
    // The 64-row tile carries 64 accumulator VGPRs: a 2-deep B ring keeps it under the
    // 192-VGPR budget of two blocks per WGP (B latency is not its limit, WMMA is). The
    // 48-row tile keeps DB: a 2-deep ring costs it throughput, and at DB = 4 the few
    // spills the budget forces land outside the loops (kernel prologue / expert tail)
    constexpr int DB = (TILEBLOCKS_M >= 4 && EXL3_MOE_PIPE_DB > 2) ? 2 : EXL3_MOE_PIPE_DB;
    constexpr int AG = PipeAChunk<TILEBLOCKS_M>::AG;

    constexpr int TILEBLOCKS_K = TILESIZE_K / 16;
    constexpr int TILEBLOCKS_N = TILESIZE_N / 16;
    constexpr int NUM_WARPS = EXL3_GEMM_BASE_THREADS / 32;       // per sub_k
    constexpr int FN = TILEBLOCKS_N / NUM_WARPS;                 // N-blocks per warp
    constexpr int BLK_BYTES = Exl3Width<bits>::tile_bytes;       // one 16x16 trellis block (32 * bits; half rates 16 * (2K + 1))
    constexpr int SH_A_STRIDE = TILESIZE_K;                     // halves; 16-byte chunks swizzled by row
    constexpr int A_ROWS = TILEBLOCKS_M * 16;
    constexpr int A_VEC = TILESIZE_K / 8;                        // uint4 per A row per k-tile
    constexpr int A_BUF = AG * A_ROWS * SH_A_STRIDE;             // halves per chunk buffer
    constexpr int THREADS = EXL3_GEMM_BASE_THREADS * TILEBLOCKS_K;
    constexpr int WARPS = NUM_WARPS * TILEBLOCKS_K;
    constexpr int BT_WARP = FN * 16 * BT_STRIDE;                 // halves per warp per set
    constexpr int BT_SET = WARPS * BT_WARP;                      // halves per set

    static_assert(TILESIZE_K % 16 == 0 && TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert(TILEBLOCKS_N % NUM_WARPS == 0 && FN >= 1, "Invalid kernel params");
    static_assert(AG * A_ROWS * A_VEC <= THREADS, "A chunk must be at most one uint4 per thread");
    static_assert(TILEBLOCKS_K == 1 || 8 * EXL3_GEMM_BASE_THREADS * FN * 4 <= BT_SET * 2,
                  "sub_k reduction scratch must fit one B staging set");
    static_assert(DB >= 2, "B ring depth");
    static_assert(smem_bytes<bits, TILEBLOCKS_M, TILESIZE_K, TILESIZE_N>() <= SMEM_MAX, "LDS");
    constexpr bool A_ALL = AG * A_ROWS * A_VEC == THREADS;

    extern __shared__ half shared[];
    half* sh_bt = shared;                            // [2 sets][WARPS][FN][16 n][BT_STRIDE k]
    half* sh_a  = sh_bt + BT_SETS * BT_SET;                // [2 bufs][AG][A_ROWS][SH_A_STRIDE]

    const int t       = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    const int sub_k   = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    const int warp_id = t / 32;
    const int lane_id = t % 32;
    const int gw      = threadIdx.x / 32;                        // block-wide warp
    const int c_row      = lane_id % 16;
    const int c_col_base = (lane_id >= 16) ? 1 : 0;

    const int tiles_k  = size_k / TILESIZE_K;
    const int tiles_n  = size_n / TILESIZE_N;
    const int blocks_n = tiles_n * TILEBLOCKS_N;

    const int num_slices = gridDim.x;
    const int slice_beg  = tiles_k * tiles_n * blockIdx.x / num_slices;
    const int slice_end  = tiles_k * tiles_n * (blockIdx.x + 1) / num_slices;
    const int slice_len  = slice_end - slice_beg;
    if (slice_len < 1) return;

    const int k0 = slice_beg % tiles_k;
    const int n0 = slice_beg / tiles_k;

    // ---- B prefetch cursor ---------------------------------------------------
    // Byte offsets into B. Block (k16, n16) starts at (k16 * blocks_n + n16) * BLK_BYTES;
    // this warp's FN blocks are contiguous. Each lane loads only the R dwords of each
    // block that its own 8 weights decode from (exl3_lane_plan, the GEMV tiles core's
    // plan), so the decode reads registers: no LDS staging of the raw words.
    constexpr int R = Exl3LanePlan<bits>::R;
    const Exl3LanePlan<bits> lp = exl3_lane_plan<bits>(lane_id);
    uint32_t b_off[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) b_off[r] = lp.off[r];
    exl3_g_char* b_base = exl3_uni_ptr((exl3_g_char*) B) +
                          exl3_uni((sub_k * blocks_n + warp_id * FN) * BLK_BYTES);
    const int b_step_k = TILEBLOCKS_K * blocks_n * BLK_BYTES;
    const int b_step_n = TILEBLOCKS_N * BLK_BYTES;
    int bp_k = k0;
    int bp_n = n0;
    int bp_j = 0;
    int bp_off = k0 * b_step_k + n0 * b_step_n;

    uint32_t rb[DB][FN][R];

    auto fetch_b = [&] (uint32_t (&dst)[FN][R])
    {
        #pragma unroll
        for (int r = 0; r < R; ++r) exl3_opaque(b_off[r]);
        exl3_g_char* p = b_base + bp_off;
        #pragma unroll
        for (int n = 0; n < FN; ++n)
            #pragma unroll
            for (int r = 0; r < R; ++r)
                dst[n][r] = exl3_ld_g(p + n * BLK_BYTES, b_off[r]);
        // Advance, but never past the last tile: the ring's tail prefetches re-read it
        // (unconditional loads keep the compiler's vmcnt bookkeeping exact)
        if (bp_j < slice_len - 1)
        {
            if (++bp_k == tiles_k) { bp_k = 0; bp_n++; bp_off = bp_n * b_step_n; }
            else bp_off += b_step_k;
        }
        bp_j++;
    };

    // ---- A chunk staging -------------------------------------------------------
    // Thread -> (tile jj of the chunk, row, uint4 column). Rows past size_m read the
    // last valid row (in bounds; those output rows are never stored). A chunk covers
    // slice tiles [c * AG, c * AG + AG); A depends only on k, which wraps at tiles_k.
    const int a_jj  = threadIdx.x / (A_ROWS * A_VEC);
    const int a_row = (threadIdx.x / A_VEC) % A_ROWS;
    const int a_vec = threadIdx.x % A_VEC;
    exl3_g_char* a_base = exl3_uni_ptr((exl3_g_char*) A);
    uint32_t a_goff = (MIN(a_row, size_m - 1) * size_k + a_vec * 8) * 2;
    // Chunk c of row r lives at chunk c ^ a_swz(r)
    auto a_swz = [] (int r) { return ((r >> 1) ^ (r >> 3)) & (A_VEC - 1); };
    const int a_soff = (a_jj * A_ROWS + a_row) * SH_A_STRIDE + (a_vec ^ a_swz(a_row)) * 8;
    int ac_k = k0;                                   // k of the next chunk's first tile
    u32x4_t a_reg;

    auto fetch_a_chunk = [&] ()
    {
        exl3_opaque(a_goff);
        int kk = ac_k + a_jj;
        while (kk >= tiles_k) kk -= tiles_k;
        if (A_ALL || threadIdx.x < AG * A_ROWS * A_VEC)
            a_reg = *(g_u4*) (a_base + kk * (TILESIZE_K * 2) + a_goff);
        ac_k += AG;
        while (ac_k >= tiles_k) ac_k -= tiles_k;
    };
    auto store_a_chunk = [&] (int buf)
    {
        if (A_ALL || threadIdx.x < AG * A_ROWS * A_VEC)
            *((u32x4_t*) (sh_a + buf * A_BUF + a_soff)) = a_reg;
    };

    // ---- compute / output state ----------------------------------------------
    int c_k  = k0;
    int c_k0 = k0;
    int c_n  = n0;
    int c_j  = 0;
    int a_buf = 0;

    WmmaFragC frag_c[TILEBLOCKS_M][FN];
    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
            #pragma unroll
            for (int n = 0; n < FN; ++n)
                frag_c[m][n].clear();
    };

    // B staging, [n][k] per 16x16 block: lane l decodes rows (l%4)*2 + {0,1} (+8) of
    // columns bt_col and bt_col + 8 (the mapping exl3_gemm_inner_rdna.cuh load_frags
    // writes row-major), one half2 per store. Rows (columns n) with bit 2 set keep their
    // two 16-byte halves swapped (bt_swz), which spreads the store banks; the reader
    // un-swaps by address, so the fragment is the same halves in the same slots.
    half* my_bt = sh_bt + gw * BT_WARP;
    const int bt_row = (lane_id % 4) * 2;
    const int bt_col = (lane_id >> 3) * 2 + ((lane_id >> 2) & 1);
    auto bt_swz = [] (int col) { return ((col >> 2) & 1) * 8; };
    const int st_00 = (bt_col    ) * BT_STRIDE + ((bt_row    ) ^ bt_swz(bt_col));
    const int st_08 = (bt_col    ) * BT_STRIDE + ((bt_row + 8) ^ bt_swz(bt_col));
    const int st_80 = (bt_col + 8) * BT_STRIDE + ((bt_row    ) ^ bt_swz(bt_col + 8));
    const int st_88 = (bt_col + 8) * BT_STRIDE + ((bt_row + 8) ^ bt_swz(bt_col + 8));
    const int ld_lo = c_row * BT_STRIDE + (0 ^ bt_swz(c_row));
    const int ld_hi = c_row * BT_STRIDE + (8 ^ bt_swz(c_row));

    // Decode one k-tile's B into staging set par
    auto decode_b = [&] (const uint32_t (&qb)[FN][R], int par)
    {
        #pragma unroll
        for (int n = 0; n < FN; ++n)
        {
            // Bit for bit dq_dispatch<bits, cb>(block, lane << 3): the GEMV tiles core's
            // exact-integer decoder (pk_mul_lo_u16 + mad_u32_u16 hash, sad_u8 byte sums)
            FragB f0, f1;
            exl3_dq_tile_decode<bits, cb>(qb[n], lp, f0, f1);
            half* bt = my_bt + par * BT_SET + n * 16 * BT_STRIDE;
            *((half2*) (bt + st_00)) = f0[0];
            *((half2*) (bt + st_08)) = f0[1];
            *((half2*) (bt + st_80)) = f1[0];
            *((half2*) (bt + st_88)) = f1[1];
        }
    };

    auto load_b_frags = [&] (WmmaFragB (&frag_b)[FN], int par)
    {
        #pragma unroll
        for (int n = 0; n < FN; ++n)
        {
            const half* bt = my_bt + par * BT_SET + n * 16 * BT_STRIDE;
            u32x4_t lo = *(const u32x4_t*) (bt + ld_lo);
            u32x4_t hi = *(const u32x4_t*) (bt + ld_hi);
            u32x8_t v = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            frag_b[n].data = __builtin_bit_cast(half16_t, v);
        }
    };

    auto mma = [&] (const WmmaFragB (&frag_b)[FN], int a_slot)
    {
        const half* a_tile = sh_a + a_buf * A_BUF + a_slot * A_ROWS * SH_A_STRIDE;
        #pragma unroll
        for (int m = 0; m < TILEBLOCKS_M; ++m)
        {
            // rdna_wmma::load_matrix_a's fragment (lane L: row L % 16, this sub_k's 16
            // consecutive k), read as its two 16-byte halves through the swizzle
            WmmaFragA frag_a;
            const half* a_row_p = a_tile + (m * 16 + c_row) * SH_A_STRIDE;
            u32x4_t lo = *(const u32x4_t*) (a_row_p + (((2 * sub_k    ) ^ a_swz(m * 16 + c_row)) * 8));
            u32x4_t hi = *(const u32x4_t*) (a_row_p + (((2 * sub_k + 1) ^ a_swz(m * 16 + c_row)) * 8));
            u32x8_t v = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            frag_a.data = __builtin_bit_cast(half16_t, v);
            #pragma unroll
            for (int n = 0; n < FN; ++n)
                rdna_wmma::mma_sync(frag_c[m][n], frag_a, frag_b[n]);
        }
    };

    auto out_col = [&] (int n, int j)
    {
        return (warp_id * FN + n) * 16 + j * 2 + c_col_base;
    };

    // par: parity of the tile that closed the column. Its B staging set is free once
    // every wave has loaded its fragments, so the sub_k reduction borrows it (the
    // other set already holds the next tile's decoded B)
    auto reduce = [&] (int par)
    {
        half* gl_c = C + c_n * TILESIZE_N;

        if constexpr (TILEBLOCKS_K > 1)
        {
            float* sh_c = (float*) (sh_bt + par * BT_SET);
            block_lds_sync();
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                for (int src = 1; src < TILEBLOCKS_K; ++src)
                {
                    float* sh_red = sh_c + 8 * FN * t;
                    if (sub_k == src)
                    {
                        #pragma unroll
                        for (int n = 0; n < FN; ++n)
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) sh_red[n * 8 + j] = frag_c[m][n][j];
                    }
                    block_lds_sync();
                    if (sub_k == 0)
                    {
                        #pragma unroll
                        for (int n = 0; n < FN; ++n)
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) frag_c[m][n][j] += sh_red[n * 8 + j];
                    }
                    block_lds_sync();
                }
            }
        }

        int lock_i = tiles_k - c_k - 1;
        int lock_d = c_k - c_k0 + 1;
        int* lock = &locks[c_n];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last  = lock_i + lock_d == tiles_k;

        if (!sub_k)
        {
            #pragma unroll
            for (int m = 0; m < TILEBLOCKS_M; ++m)
            {
                const int row = m * 16 + c_row;
                if (row >= size_m) continue;
                half* rp = gl_c + row * size_n;
                if (!first)
                {
                    #pragma unroll
                    for (int n = 0; n < FN; ++n)
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            frag_c[m][n][j] += __half2float(rp[out_col(n, j)]);
                }
                #pragma unroll
                for (int n = 0; n < FN; ++n)
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        rp[out_col(n, j)] = __float2half(frag_c[m][n][j]);
            }
        }

        barrier_release(lock, lock_d, last);
        clear_frag_c();
    };

    // ---- prologue ------------------------------------------------------------
    // B ring: tiles 0 .. DB-1 in slots 0 .. DB-1; tile 0 decoded into set 0.
    // A: chunk 0 staged in buffer 0, chunk 1 in flight.
    #pragma unroll
    for (int s = 0; s < (BT_SETS == 2 ? DB : DB - 1); ++s) fetch_b(rb[s]);
    fetch_a_chunk();
    store_a_chunk(0);
    fetch_a_chunk();
    if constexpr (BT_SETS == 2) decode_b(rb[0], 0);
    clear_frag_c();

    // ---- main loop -------------------------------------------------------------
    // Step for tile j (B slot j % DB, staging set j % 2, A slot j % AG, all compile-time
    // in the unrolled body): issue the loads of tile j's B fragments (decoded last
    // step), refill slot j % DB with tile j + DB, decode tile j + 1 into the other set
    // while those loads land, then the WMMAs. Decoding one tile ahead keeps the LDS
    // round trip off each wave's critical path.
    //
    // At an A chunk boundary the block meets once (LDS-only barrier): that publishes
    // the chunk stored one chunk ago and retires the reads of the buffer about to be
    // refilled. The column-end reduce is inlined in every step behind an unlikely
    // branch: a single out-of-line reduce entered through a switch made clang
    // tail-merge the steps, copy registers through phis and wait vmcnt(0) per tile.
    constexpr int U0 = (DB > AG) ? DB : AG;
    constexpr int U = (U0 > 2) ? U0 : 2;
    static_assert(U % DB == 0 && U % AG == 0 && U % 2 == 0, "unroll");
    while (true)
    {
        #pragma unroll
        for (int s = 0; s < U; ++s)
        {
            if (s % AG == 0)
            {
                block_lds_sync();
                store_a_chunk(a_buf ^ 1);
                fetch_a_chunk();
            }
            WmmaFragB frag_b[FN];
            if constexpr (BT_SETS == 2)
            {
                wave_lds_sync();
                load_b_frags(frag_b, s % 2);
                fetch_b(rb[s % DB]);
                decode_b(rb[(s + 1) % DB], (s + 1) % 2);
            }
            else
            {
                fetch_b(rb[(s + DB - 1) % DB]);
                wave_lds_sync();
                decode_b(rb[s % DB], 0);
                wave_lds_sync();
                load_b_frags(frag_b, 0);
            }
            mma(frag_b, s % AG);
            if (s % AG == AG - 1) a_buf ^= 1;
            if ((c_k == tiles_k - 1) || (c_j == slice_len - 1)) [[unlikely]]
            {
                reduce(BT_SETS == 2 ? s % 2 : 0);
                if (c_j == slice_len - 1) goto done;
                c_j++;
                if (++c_k == tiles_k) { c_k = 0; c_n++; }
                c_k0 = c_k;
            }
            else
            {
                c_k++;
                c_j++;
            }
        }
    }
    done:;
}

// Dynamic LDS the pipelined kernel is launched with: the largest row tile's need. Kept
// <= 32 KB (runtime occupancy: two 512-thread blocks per WGP only at <= 32 KB each)
template <int TILESIZE_K, int TILESIZE_N>
constexpr int smem_launch_bytes()
{
    constexpr int a = smem_bytes<2, 1, TILESIZE_K, TILESIZE_N>();
    constexpr int b = smem_bytes<2, 2, TILESIZE_K, TILESIZE_N>();
    constexpr int c = smem_bytes<2, 3, TILESIZE_K, TILESIZE_N>();
    constexpr int d = smem_bytes<2, 4, TILESIZE_K, TILESIZE_N>();
    constexpr int ab = a > b ? a : b;
    constexpr int cd = c > d ? c : d;
    return ab > cd ? ab : cd;
}

} // namespace moe_pipe
