/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// EXL3 MOD: vendored from rocm_wmma_gemm ea3aa74 into exllamav3 (ROCm port). The only change
// is a separate output type TO for gemm_impl (fp32 accumulate, fp16 or fp32 store); every edit
// is marked "EXL3 MOD". fragment.hpp carries one more marked edit; the other vendored headers
// are byte-identical to upstream.

#ifndef ROCM_WMMA_KERNEL_HPP
#define ROCM_WMMA_KERNEL_HPP

#include "common.hpp"
#include "fragment.hpp"
#include "load.hpp"
#include "mapping.hpp"
#include "wmma.hpp"

namespace rocm_wmma_gemm
{

/**
 * @brief Base configuration struct for wave (warp) settings.
 */
template<int warps_m, int warps_n>
struct wave_config_base
{
    static constexpr int total_warps = warps_m * warps_n;
};

/**
 * @brief Configuration struct for tuning AMD wave bounds.
 *
 * Provides minimum and maximum waves per execution unit to optimize register usage
 * and occupancy for the given matrix layouts.
 */
template<m_layout LAYOUT_A, m_layout LAYOUT_B, int warps_m, int warps_n>
struct wave_config : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 2;
    static constexpr int max = 8;
};

/**
 * @brief Wave configuration specialization for row-major A and col-major B.
 */
template<int warps_m, int warps_n>
struct wave_config<m_layout::row_major, m_layout::col_major, warps_m, warps_n>
    : wave_config_base<warps_m, warps_n>
{
    using base               = wave_config_base<warps_m, warps_n>;
    static constexpr int min = 2;
    static constexpr int max = 4;
};

/**
 * @brief Core device function implementing the WMMA-based GEMM operation.
 *
 * This function computes a block of the output matrix C using the following optimizations:
 *
 * 1. Double Buffering (Software Pipelining): Overlaps global-to-register memory loads with
 *    register-to-LDS and compute stages using two shared memory (LDS) buffers.
 * 2. Multi-Stage Register Prefetching: Stages global loads in register fragments via
 *    `prefetch_fragment` before committing to LDS, hiding instruction and global latency.
 * 3. Hierarchical Warp Tiling: Subdivides the thread block into warps (mapped via warps_m x warps_n)
 *    where each warp registers and computes a specific sub-tile of the block.
 * 4. Dual K-Slices (Layout-Specific): For row-major A and col-major B layouts, traverses K-steps
 *    in nested slices (snake pattern) to interleave LDS read latency directly behind WMMA math execution.
 * 5. Chunked LDS-Buffered Epilogue: Buffer warp register C fragments back to LDS in structured
 *    chunks for coalesced, conflict-free global writes.
 * 6. Memory Load/Commit Interleaving: Interleaves global prefetching and LDS commits across
 *    computation phases to optimize instruction-level parallelism (ILP).
 *
 * @tparam T Data type of output matrix C.
 * @tparam U Data type of input matrices A and B.
 * @tparam LAYOUT_C Memory layout of C.
 * @tparam LAYOUT_A Memory layout of A.
 * @tparam LAYOUT_B Memory layout of B.
 * @tparam warps_m Number of warps mapped to the M dimension.
 * @tparam warps_n Number of warps mapped to the N dimension.
 * @tparam warp_tile_m Number of WMMA tiles per warp in the M dimension.
 * @tparam warp_tile_n Number of WMMA tiles per warp in the N dimension.
 * @tparam k_slices Number of wmma_tile-sized K-slices packed into one block.
 * @tparam single_buffer 1 = single LDS buffer (half LDS, read-sync-write, direct C store)
 *                       for higher occupancy; 0 = double buffer (software-pipelined).
 * @tparam swizzle Swizzle size for the block mapping.
 * @tparam bits Vectorization bit width for memory loads.
 * @tparam is_aligned True if the global matrix dimensions are strictly aligned to the load width.
 */
template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      k_slices,
         int      single_buffer,
         int      swizzle,
         int      bits,
         int      is_aligned,
         class    TO = T> // EXL3 MOD: output type TO, separate from accumulator type T
__device__ __forceinline__ void gemm_impl(
    TO* __restrict__ C, const U* __restrict__ A, const U* __restrict__ B, int M, int N, int K)
{
    static_assert(warps_m != 0 && warps_n != 0);
    static_assert((warp_tile_m * warp_tile_n) > 1);
    static_assert(k_slices > 0);

    constexpr int padding_a = (LAYOUT_A == m_layout::row_major) ? 8 : 0;
    constexpr int padding_b = (LAYOUT_B == m_layout::col_major) ? 8 : 0;

    constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    constexpr int block_n = warps_n * warp_tile_n * wmma_tile;

    // Number of wmma_tile-sized K-slices packed into one block along the contraction
    // dimension. Packing multiple slices lets their LDS reads interleave behind WMMA
    // math and widens each global K-burst; block_k is derived from it.
    constexpr int block_k  = k_slices * wmma_tile;
    constexpr int stride_a = block_k + padding_a;
    constexpr int stride_b = block_k + padding_b;
    constexpr int lds_size = (block_m * stride_a) + (stride_b * block_n);

    // Single buffering halves the LDS tile (one buffer instead of the pipelined pair),
    // holding the next tile in registers (already done by prefetch_fragment) and committing
    // after a read-sync-write barrier — trading one extra __syncthreads per K-iteration and
    // a direct (non-staged) C store for higher occupancy. Whether it wins is config- and
    // size-dependent, so it is a tuned parameter rather than a compile-time heuristic.
    // single_buffer == 0 is byte-identical to the double-buffered software pipeline.
    constexpr bool use_single_buffer = (single_buffer != 0);
    constexpr int  lds_buffers       = use_single_buffer ? 1 : 2;

    const int grid_m  = (M + block_m - 1) / block_m;
    const int grid_n  = (N + block_n - 1) / block_n;
    const int tile_id = blockIdx.x;

    using mapper = tile_mapper<block_m, block_n, LAYOUT_A, LAYOUT_B, swizzle>;

    int block_row, block_col;
    mapper().map_tile(tile_id, grid_m, grid_n, &block_row, &block_col);

    __shared__ U lds_mem[lds_buffers * lds_size];

    U* a_tiles_0 = lds_mem;
    U* b_tiles_0 = lds_mem + (block_m * stride_a);

    constexpr int full_block = warp_size * warps_m * warps_n;
    constexpr int half_block = full_block / 2;
    const int     tid        = threadIdx.x;
    const int     cid        = tid % half_block;

    A += blockIdx.y * M * K;
    B += blockIdx.y * K * N;
    C += blockIdx.y * M * N;

    const U* A_base = A + block_row * ((LAYOUT_A == m_layout::col_major) ? 1 : K);
    const U* B_base = B + block_col * ((LAYOUT_B == m_layout::col_major) ? K : 1);

    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / warps_n;
    const int warp_col = warp_id % warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     lane_id      = tid % warp_size;
    const int     half_warp_id = lane_id / half_warp;
    const int     half_lane    = tid % half_warp;

    const int warp_m_base = warp_row * warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * warp_tile_n * wmma_tile;

    constexpr int a_frag_size = k_slices * warp_tile_m;
    constexpr int b_frag_size = k_slices * warp_tile_n;

    fragment<T, wmma_tile> c_frags[warp_tile_m][warp_tile_n];
    fragment<U, wmma_tile> a_frag[a_frag_size];
    fragment<U, wmma_tile> b_frag[b_frag_size];

    const int lead_a = (LAYOUT_A == m_layout::col_major) ? M : K;
    const int lead_b = (LAYOUT_B == m_layout::col_major) ? K : N;

    // Constructor builds the SRD once from the matrix base + allocation size.
    // All prefetch/partial_prefetch calls reuse it with no per-call overhead.
    prefetch_fragment<LAYOUT_A, bits, full_block, block_m, block_k, padding_a, U> regs_a(
        A,
        static_cast<unsigned>(M) * static_cast<unsigned>(K));
    prefetch_fragment<LAYOUT_B, bits, full_block, block_k, block_n, padding_b, U> regs_b(
        B,
        static_cast<unsigned>(K) * static_cast<unsigned>(N));

    const U* A_tile_ptr = A_base;
    const U* B_tile_ptr = B_base;

    const int global_mult_A = block_k * ((LAYOUT_A == m_layout::col_major) ? M : 1);
    const int global_mult_B = block_k * ((LAYOUT_B == m_layout::col_major) ? 1 : N);

    constexpr int frag_mult_A   = (LAYOUT_A == m_layout::col_major) ? 1 : stride_a;
    constexpr int frag_mult_B   = (LAYOUT_B == m_layout::col_major) ? stride_b : 1;
    constexpr int frag_offset_A = wmma_tile * frag_mult_A;
    constexpr int frag_offset_B = wmma_tile * frag_mult_B;

    const int warp_offset_A = (warp_m_base + half_lane) * frag_mult_A;
    const int warp_offset_B = (warp_n_base + half_lane) * frag_mult_B;

    // LDS stride between consecutive K-slices (from prefetch_fragment::commit layout):
    //   row-major A: m*stride_a + k  => 1
    //   col-major A: k*block_m  + m  => block_m
    //   col-major B: n*stride_b + k  => 1
    //   row-major B: k*block_n  + n  => block_n
    constexpr int slice_stride_A
        = (LAYOUT_A == m_layout::row_major) ? wmma_tile : wmma_tile * block_m;
    constexpr int slice_stride_B
        = (LAYOUT_B == m_layout::col_major) ? wmma_tile : wmma_tile * block_n;

    regs_a.prefetch(A_tile_ptr, lead_a, tid);
    regs_b.prefetch(B_tile_ptr, lead_b, tid);
    regs_a.commit(a_tiles_0, tid);
    regs_b.commit(b_tiles_0, tid);
    __syncthreads();

    // In double-buffer mode current/next alternate between the two LDS halves; in
    // single-buffer mode there is one buffer and next_* alias current_* (unused).
    U* current_a = a_tiles_0;
    U* current_b = b_tiles_0;
    U* next_a    = use_single_buffer ? a_tiles_0 : (lds_mem + lds_size);
    U* next_b    = use_single_buffer ? b_tiles_0 : (lds_mem + lds_size + (block_m * stride_a));

    constexpr bool   warp_m_is_major = warp_tile_m >= warp_tile_n;
    constexpr size_t warp_inner_max  = warp_m_is_major ? warp_tile_n : warp_tile_m;
    // Slice-packing layouts scale the outer dimension by k_slices so every slice is
    // traversed as one contiguous snake. Single-slice keeps the original outer extent.
    constexpr size_t warp_outer_max
        = k_slices * static_cast<size_t>(warp_m_is_major ? warp_tile_m : warp_tile_n);
    constexpr size_t total_combos = warp_outer_max * warp_inner_max;
    // The prefetch of the next tile is split across the fold, alternating A and B every
    // other iteration, so each matrix is staged in total_combos / 2 chunks.
    constexpr size_t prefetch_steps = total_combos / 2;

    // get_wm/get_wn produce indices spanning all k_slices in one contiguous snake:
    // the outer index runs 0..k_slices*wt-1, the inner 0..wt-1 (or vice versa).
    constexpr auto get_wm = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_o : w_i_s;
    };

    constexpr auto get_wn = [](size_t i) constexpr -> size_t
    {
        size_t w_o   = i / warp_inner_max;
        size_t w_i   = i % warp_inner_max;
        size_t w_i_s = (w_o & 1) ? (warp_inner_max - w_i - 1) : w_i;
        return warp_m_is_major ? w_i_s : w_o;
    };

    auto run_fold
        = [&]<bool do_prefetch>(const U* ca0, const U* cb0, const U* next_A, const U* next_B)
    {
        // Returns the LDS pointer for fragment index WM/WN in the flat array. The high
        // part of the index selects the K-slice (WM / warp_tile_m) and the low part the
        // fragment within that slice. The index is a template parameter so the slice and
        // fragment offsets are guaranteed to constant-fold to a single immediate.
        auto a_ptr = [&]<size_t WM>() constexpr -> const U*
        { return ca0 + (WM / warp_tile_m) * slice_stride_A + (WM % warp_tile_m) * frag_offset_A; };
        auto b_ptr = [&]<size_t WN>() constexpr -> const U*
        { return cb0 + (WN / warp_tile_n) * slice_stride_B + (WN % warp_tile_n) * frag_offset_B; };

        if constexpr(warp_m_is_major)
        {
            load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[0],
                                                     a_ptr.template operator()<0>(),
                                                     block_m,
                                                     stride_a);
            load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[0],
                                                     b_ptr.template operator()<0>(),
                                                     stride_b,
                                                     block_n);
        }
        else
        {
            load_matrix<m_input::matrix_b, LAYOUT_B>(b_frag[0],
                                                     b_ptr.template operator()<0>(),
                                                     stride_b,
                                                     block_n);
            load_matrix<m_input::matrix_a, LAYOUT_A>(a_frag[0],
                                                     a_ptr.template operator()<0>(),
                                                     block_m,
                                                     stride_a);
        }

        [&]<size_t... i>(std::index_sequence<i...>)
        {
            (
                [&]()
                {
                    if constexpr(do_prefetch)
                    {
                        constexpr size_t step = i / 2;
                        if constexpr((i % 2) == 0)
                        {
                            regs_a.template partial_prefetch<step, prefetch_steps>(next_A,
                                                                                   lead_a,
                                                                                   tid);
                        }
                        else
                        {
                            regs_b.template partial_prefetch<step, prefetch_steps>(next_B,
                                                                                   lead_b,
                                                                                   tid);
                        }
                    }

                    constexpr size_t wm = get_wm(i);
                    constexpr size_t wn = get_wn(i);

                    constexpr size_t next_i = i + 1;
                    if constexpr(next_i < total_combos)
                    {
                        constexpr size_t next_wm = get_wm(next_i);
                        constexpr size_t next_wn = get_wn(next_i);
                        if constexpr(warp_m_is_major)
                        {
                            if constexpr(next_wm != wm)
                            {
                                load_matrix<m_input::matrix_a, LAYOUT_A>(
                                    a_frag[next_wm],
                                    a_ptr.template operator()<next_wm>(),
                                    block_m,
                                    stride_a);
                            }
                            // B changes every time wm wraps to a new outer row (including slice boundary)
                            if constexpr(next_wm % warp_tile_m == 0)
                            {
                                constexpr size_t b_idx
                                    = next_wn + (next_wm / warp_tile_m) * warp_tile_n;
                                load_matrix<m_input::matrix_b, LAYOUT_B>(
                                    b_frag[b_idx],
                                    b_ptr.template operator()<b_idx>(),
                                    stride_b,
                                    block_n);
                            }
                        }
                        else
                        {
                            if constexpr(next_wn != wn)
                            {
                                load_matrix<m_input::matrix_b, LAYOUT_B>(
                                    b_frag[next_wn],
                                    b_ptr.template operator()<next_wn>(),
                                    stride_b,
                                    block_n);
                            }
                            if constexpr(next_wn % warp_tile_n == 0)
                            {
                                constexpr size_t a_idx
                                    = next_wm + (next_wn / warp_tile_n) * warp_tile_m;
                                load_matrix<m_input::matrix_a, LAYOUT_A>(
                                    a_frag[a_idx],
                                    a_ptr.template operator()<a_idx>(),
                                    block_m,
                                    stride_a);
                            }
                        }
                    }

                    // Map local iteration to flat fragment arrays
                    constexpr size_t a_slice = warp_m_is_major ? 0 : (wn / warp_tile_n);
                    constexpr size_t b_slice = warp_m_is_major ? (wm / warp_tile_m) : 0;
                    constexpr size_t a_wmma  = wm + a_slice * warp_tile_m;
                    constexpr size_t b_wmma  = wn + b_slice * warp_tile_n;

                    __builtin_amdgcn_s_setprio(1);
                    wmma(a_frag[a_wmma],
                         b_frag[b_wmma],
                         c_frags[wm % warp_tile_m][wn % warp_tile_n]);
                    __builtin_amdgcn_s_setprio(0);
                }(),
                ...);
        }(std::make_index_sequence<total_combos>{});
    };

    for(int k_tile = 0; k_tile < K - block_k; k_tile += block_k)
    {
        const U* ca0    = current_a + warp_offset_A;
        const U* cb0    = current_b + warp_offset_B;
        const U* next_A = A_tile_ptr + global_mult_A;
        const U* next_B = B_tile_ptr + global_mult_B;

        // Stages the next tile global->regs (partial_prefetch) while computing the current
        // tile from LDS (load_matrix).
        run_fold.template operator()<true>(ca0, cb0, next_A, next_B);

        A_tile_ptr += global_mult_A;
        B_tile_ptr += global_mult_B;

        if constexpr(use_single_buffer)
        {
            // Read-sync-write: the next commit overwrites the same buffer run_fold just
            // read, so every load_matrix read must retire before the commit stores begin.
            __syncthreads();
            regs_a.commit(current_a, tid);
            regs_b.commit(current_b, tid);
            __syncthreads();
        }
        else
        {
            // Double buffer: commit the next tile into the free half while the current
            // half is still being read, then swap. One barrier suffices.
            regs_a.commit(next_a, tid);
            regs_b.commit(next_b, tid);

            U* temp_a = current_a;
            U* temp_b = current_b;
            current_a = next_a;
            current_b = next_b;
            next_a    = temp_a;
            next_b    = temp_b;
            __syncthreads();
        }
    }

    const U* ca0 = current_a + warp_offset_A;
    const U* cb0 = current_b + warp_offset_B;

    run_fold.template operator()<false>(ca0, cb0, nullptr, nullptr);

    __syncthreads();

    // EXL3 MOD begin: accumulate in T (float), store as TO (float or half). A half fragment
    // holds its 8 values in the even slots (the f16 WMMA opsel=0 layout store_matrix reads),
    // a float fragment holds them densely. With TO == T this is a register copy.
    fragment<TO, wmma_tile> o_frags[warp_tile_m][warp_tile_n];
    [&]<size_t... f>(std::index_sequence<f...>)
    {
        (
            [&]()
            {
                constexpr int fm = static_cast<int>(f) / warp_tile_n;
                constexpr int fn = static_cast<int>(f) % warp_tile_n;
                if constexpr(std::is_same<T, TO>::value)
                {
                    o_frags[fm][fn] = c_frags[fm][fn];
                }
                else
                {
                    static_assert(std::is_same<T, float>::value && std::is_same<TO, half>::value);
                    [&]<size_t... e>(std::index_sequence<e...>)
                    {
                        ((o_frags[fm][fn].get()[2 * e]
                          = static_cast<_Float16>(c_frags[fm][fn].get()[e])),
                         ...);
                    }(std::make_index_sequence<wmma_tile / 2>{});
                }
            }(),
            ...);
    }(std::make_index_sequence<warp_tile_m * warp_tile_n>{});
    // EXL3 MOD end

    // Single-buffer halves lds_mem, which is too small to stage the full C tile for the
    // chunked LDS epilogue (the chunk staging assumes the 2x buffer). Write fragments
    // straight to global instead. The chunked (coalesced) path is kept for double buffer.
    if constexpr(use_single_buffer)
    {
        // One row of the fragment grid: fold over wn with wm fixed as a template param
        // (avoids nesting a pack expansion inside another pack's expansion).
        auto store_row = [&]<size_t wm>()
        {
            [&]<size_t... wn>(std::index_sequence<wn...>)
            {
                (store_matrix<LAYOUT_C, false>(
                     C,
                     o_frags[wm][wn],
                     block_row + warp_m_base + static_cast<int>(wm) * wmma_tile + half_warp_id,
                     block_col + warp_n_base + static_cast<int>(wn) * wmma_tile + half_lane,
                     M,
                     N),
                 ...);
            }(std::make_index_sequence<warp_tile_n>{});
        };

        [&]<size_t... wm>(std::index_sequence<wm...>)
        { (store_row.template operator()<wm>(), ...); }(std::make_index_sequence<warp_tile_m>{});
    }
    else
    {
        constexpr bool is_col_major = (LAYOUT_C == m_layout::col_major);

        constexpr int chunk_rows = wmma_tile;
        constexpr int chunk_cols = wmma_tile;

        // Pad the col-major-C staging pitch (LDS column stride = block_m) to break the bank
        // conflict where consecutive lanes' fragment stores land in the same bank (block_m a
        // multiple of 32 banks). +2 staggers them. Row-major C is unaffected. Accounted for in
        // chunk_unit_bytes below so the group-fit never overflows the LDS buffer.
        constexpr int c_pad         = is_col_major ? 2 : 0;
        constexpr int c_store_pitch = block_m + c_pad; // padded LDS column stride for col-major C

        constexpr int num_chunks = is_col_major ? (warps_n * warp_tile_n) : (warps_m * warp_tile_m);
        constexpr int available_bytes = lds_buffers * lds_size * static_cast<int>(sizeof(U));
        constexpr int chunk_unit_bytes
            = is_col_major ? (c_store_pitch * chunk_cols * static_cast<int>(sizeof(TO)))
                           : (chunk_rows * block_n * static_cast<int>(sizeof(TO)));

        // Group multiple chunks into one store/flush pass to reduce barriers
        constexpr int chunk_multiple = []() constexpr
        {
            for(int m = num_chunks; m >= 1; --m)
            {
                if(num_chunks % m == 0 && m * chunk_unit_bytes <= available_bytes)
                {
                    return m;
                }
            }
            return 1;
        }();

        constexpr int num_groups = num_chunks / chunk_multiple;
        constexpr int group_cols = is_col_major ? chunk_multiple * chunk_cols : chunk_cols;
        constexpr int group_rows = is_col_major ? chunk_rows : chunk_multiple * chunk_rows;
        constexpr int group_step = is_col_major ? group_cols : group_rows;
        constexpr int transfer_m = is_col_major ? block_m : group_step;
        constexpr int transfer_n = is_col_major ? group_step : block_n;

        TO* c_buf = reinterpret_cast<TO*>(lds_mem); // EXL3 MOD: T -> TO
        shared_to_global_store<LAYOUT_C,
                               bits,
                               full_block,
                               transfer_m,
                               transfer_n,
                               is_aligned,
                               TO, // EXL3 MOD: T -> TO
                               c_pad>
            store_c(C, static_cast<unsigned>(M) * static_cast<unsigned>(N));

        // Stores a chunk to LDS before block-wide flush
        auto store_chunk = [&]<size_t ci>(TO* buf, int local_offset) // EXL3 MOD: T -> TO
        {
            if constexpr(is_col_major)
            {
                constexpr int responsible_warp_col = static_cast<int>(ci) / warp_tile_n;
                constexpr int wn_idx               = static_cast<int>(ci) % warp_tile_n;

                if(warp_col == responsible_warp_col)
                {
                    [&]<size_t... wm>(std::index_sequence<wm...>)
                    {
                        (store_matrix<LAYOUT_C, true>(buf,
                                                      o_frags[wm][wn_idx],
                                                      warp_m_base + static_cast<int>(wm) * wmma_tile
                                                          + half_warp_id,
                                                      local_offset + half_lane,
                                                      c_store_pitch,
                                                      group_cols),
                         ...);
                    }(std::make_index_sequence<warp_tile_m>{});
                }
            }
            else
            {
                constexpr int responsible_warp_row = static_cast<int>(ci) / warp_tile_m;
                constexpr int wm_idx               = static_cast<int>(ci) % warp_tile_m;

                if(warp_row == responsible_warp_row)
                {
                    [&]<size_t... wn>(std::index_sequence<wn...>)
                    {
                        (store_matrix<LAYOUT_C, true>(buf,
                                                      o_frags[wm_idx][wn],
                                                      local_offset + half_warp_id,
                                                      warp_n_base + static_cast<int>(wn) * wmma_tile
                                                          + half_lane,
                                                      group_rows,
                                                      block_n),
                         ...);
                    }(std::make_index_sequence<warp_tile_n>{});
                }
            }
        };

        auto flush_group = [&]<size_t gi>(TO* buf) // EXL3 MOD: T -> TO
        {
            if constexpr(is_col_major)
            {
                store_c.store(buf,
                              block_row,
                              block_col + static_cast<int>(gi) * group_step,
                              M,
                              N,
                              tid);
            }
            else
            {
                store_c.store(buf,
                              block_row + static_cast<int>(gi) * group_step,
                              block_col,
                              M,
                              N,
                              tid);
            }
        };

        // Store all chunks for group g
        auto store_group = [&]<size_t g>()
        {
            [&]<size_t... li>(std::index_sequence<li...>)
            {
                (store_chunk.template operator()<g * chunk_multiple + li>(c_buf,
                                                                          static_cast<int>(li)
                                                                              * wmma_tile),
                 ...);
            }(std::make_index_sequence<chunk_multiple>{});
        };

        [&]<size_t... gi>(std::index_sequence<gi...>)
        {
            (
                [&]()
                {
                    store_group.template operator()<gi>();
                    __syncthreads();
                    flush_group.template operator()<gi>(c_buf);
                    __syncthreads();
                }(),
                ...);
        }(std::make_index_sequence<num_groups>{});
    } // else: chunked double-buffer epilogue
}

/**
 * @brief Functor wrapping the GEMM kernel launch.
 *
 * @tparam T Data type of output matrix C.
 * @tparam U Data type of input matrices A and B.
 * @tparam LAYOUT_C Memory layout of C.
 * @tparam LAYOUT_A Memory layout of A.
 * @tparam LAYOUT_B Memory layout of B.
 * @tparam warps_m Number of warps mapped to the M dimension.
 * @tparam warps_n Number of warps mapped to the N dimension.
 * @tparam warp_tile_m Number of WMMA tiles per warp in the M dimension.
 * @tparam warp_tile_n Number of WMMA tiles per warp in the N dimension.
 * @tparam k_slices Number of wmma_tile-sized K-slices packed into one block.
 * @tparam single_buffer 1 = single LDS buffer (half LDS, read-sync-write, direct C store)
 *                       for higher occupancy; 0 = double buffer (software-pipelined).
 * @tparam swizzle Swizzle size for the block mapping.
 * @tparam bits Vectorization bit width for memory loads.
 * @tparam is_aligned True if the global matrix dimensions are strictly aligned to the load width.
 */
template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      k_slices,
         int      single_buffer,
         int      swizzle,
         int      bits,
         int      is_aligned>
struct kernel_gemm_impl
{
    using config = wave_config<LAYOUT_A, LAYOUT_B, warps_m, warps_n>;

    /**
     * @brief The global entry point for the GEMM kernel.
     *
     * @param C Output matrix C pointer.
     * @param A Input matrix A pointer.
     * @param B Input matrix B pointer.
     * @param M Number of rows in matrices A and C.
     * @param N Number of columns in matrices B and C.
     * @param K Number of columns in A and rows in B.
     */
    __global__ __launch_bounds__(warp_size* warps_m* warps_n) static void run(
        T* __restrict__ C, const U* __restrict__ A, const U* __restrict__ B, int M, int N, int K)
    {
        gemm_impl<T,
                  U,
                  LAYOUT_C,
                  LAYOUT_A,
                  LAYOUT_B,
                  warps_m,
                  warps_n,
                  warp_tile_m,
                  warp_tile_n,
                  k_slices,
                  single_buffer,
                  swizzle,
                  bits,
                  is_aligned>(C, A, B, M, N, K);
    }
};

template<class T,
         class U,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m,
         int      warp_tile_n,
         int      single_buffer,
         int      swizzle,
         int      bits,
         int      is_aligned>
struct kernel_gemm_impl<T,
                        U,
                        LAYOUT_C,
                        LAYOUT_A,
                        LAYOUT_B,
                        warps_m,
                        warps_n,
                        warp_tile_m,
                        warp_tile_n,
                        1,
                        single_buffer,
                        swizzle,
                        bits,
                        is_aligned>
{
    using config = wave_config<LAYOUT_A, LAYOUT_B, warps_m, warps_n>;

    /**
     * @brief The global entry point for the specialized GEMM kernel (k_slices = 1).
     */
    __global__ __launch_bounds__(warp_size* warps_m* warps_n)
        __attribute__((amdgpu_waves_per_eu(config::min,
                                           config::max))) static void run(T* __restrict__ C,
                                                                          const U* __restrict__ A,
                                                                          const U* __restrict__ B,
                                                                          int M,
                                                                          int N,
                                                                          int K)
    {
        gemm_impl<T,
                  U,
                  LAYOUT_C,
                  LAYOUT_A,
                  LAYOUT_B,
                  warps_m,
                  warps_n,
                  warp_tile_m,
                  warp_tile_n,
                  1, // k_slices = 1
                  single_buffer,
                  swizzle,
                  bits,
                  is_aligned>(C, A, B, M, N, K);
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_KERNEL_HPP
