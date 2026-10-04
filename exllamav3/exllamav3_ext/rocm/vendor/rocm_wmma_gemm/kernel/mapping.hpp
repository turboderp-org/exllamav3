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

#ifndef ROCM_WMMA_GEMM_MAPPING_HPP
#define ROCM_WMMA_GEMM_MAPPING_HPP

namespace rocm_wmma_gemm
{

/**
 * @brief Row-major tile mapping
 */
template<int BLOCK_M, int BLOCK_N>
class row_major_mapping
{
public:
    static __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
    {
        int row = tile_id / grid_n;
        int col = tile_id % grid_n;

        *block_row = row * BLOCK_M;
        *block_col = col * BLOCK_N;
    }
};

/**
 * @brief Column-major tile mapping
 */
template<int BLOCK_M, int BLOCK_N>
class col_major_mapping
{
public:
    static __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
    {
        int col = tile_id / grid_m;
        int row = tile_id % grid_m;

        *block_row = row * BLOCK_M;
        *block_col = col * BLOCK_N;
    }
};

/**
 * @brief Fast Snake Row-Swizzle with XOR Skew
 *
 * This mapping optimizes L2 cache hit rates for GEMM operations by altering the order
 * in which thread blocks process output tiles. It uses three main techniques:
 *
 * 1. 2D Swizzling (Intra-band locality):
 *    Instead of processing a thin 1D row of tiles, consecutive blocks process a 2D
 *    patch (height = SWIZZLE_M). This allows concurrent blocks to share loads from
 *    Matrix A (same M) and Matrix B (same N) in the L2 cache.
 *
 * 2. Snake Traversal (Inter-band locality):
 *    When finishing one horizontal band and moving to the next, a standard mapping
 *    jumps from N=max back to N=0, causing Matrix B to be evicted from L2.
 *    The snake traversal reverses the N-direction on odd bands, keeping Matrix B
 *    in the L2 cache across band boundaries.
 *
 *    Macro Traversal Visualization (Grid divided into bands of SWIZZLE_M):
 *    Band 0 (m_outer=0): [0] -> [1] -> [2] -> [3]  (N increases)
 *                                               |
 *    Band 1 (m_outer=1): [7] <- [6] <-[5] <- [4]  (N decreases, reuses B at edge)
 *                         |
 *    Band 2 (m_outer=2): [8] -> [9] -> [10]-> [11] (N increases)
 *
 * 3. XOR Skew (Partition Camping Avoidance):
 *    If concurrent blocks process the exact same M-coordinates simultaneously, they
 *    may hammer the same memory partitions/banks, causing queuing delays. The XOR
 *    skew staggers the inner M traversal based on the N coordinate.
 *
 *    Micro Traversal Visualization (SWIZZLE_M = 4):
 *    Without XOR (inner_m):          With XOR (inner_m ^ (final_n % 4)):
 *    final_n:  0   1   2   3         final_n:  0   1   2   3
 *           +----------------               +----------------
 *    pos 0  |  0   0   0   0         pos 0  |  0   1   2   3
 *    pos 1  |  1   1   1   1         pos 1  |  1   0   3   2
 *    pos 2  |  2   2   2   2         pos 2  |  2   3   0   1
 *    pos 3  |  3   3   3   3         pos 3  |  3   2   1   0
 *
 *    Result: Concurrent blocks (e.g., final_n 0,1,2,3 at pos 0) access M-indices
 *    0, 1, 2, and 3 respectively, perfectly distributing the load across memory banks.
 */
template<int BLOCK_M, int BLOCK_N, int SWIZZLE_M = 8>
class xor_snake_row_swizzle_mapping
{
    static_assert((SWIZZLE_M & (SWIZZLE_M - 1)) == 0, "SWIZZLE_M must be a power of 2");

public:
    static __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
    {
        const int idx_n = tile_id % grid_n;
        const int idx_m = tile_id / grid_n;

        const int m_outer = idx_m / SWIZZLE_M;
        const int m_inner = idx_m % SWIZZLE_M;

        int final_m, final_n;

        // Fast path for main complete groups
        if(m_outer < grid_m / SWIZZLE_M)
        {
            const int local = idx_n + m_inner * grid_n;
            final_n         = local / SWIZZLE_M;

            // XOR SKEW: Stagger the inner M traversal based on the N coordinate
            int inner_m = local % SWIZZLE_M;
            inner_m     = inner_m ^ (final_n & (SWIZZLE_M - 1));

            final_m = inner_m + m_outer * SWIZZLE_M;
        }
        // Slow path for the remainder edge (No XOR to prevent out-of-bounds)
        else
        {
            const int m_rem = grid_m % SWIZZLE_M;
            const int local = idx_n + m_inner * grid_n;
            final_n         = local / m_rem;
            final_m         = (local % m_rem) + m_outer * SWIZZLE_M;
        }

        // MACRO TRAVERSAL: Snake (Boustrophedon)
        if(m_outer & 1)
        {
            final_n = grid_n - 1 - final_n;
        }

        if(final_m >= grid_m || final_n >= grid_n)
        {
            *block_row = (grid_m - 1) * BLOCK_M;
            *block_col = (grid_n - 1) * BLOCK_N;
            return;
        }

        *block_row = final_m * BLOCK_M;
        *block_col = final_n * BLOCK_N;
    }
};

/**
 * @brief Fast Snake Column-Swizzle with XOR Skew
 *
 * This is the transposed equivalent of the row-swizzle mapping, optimized for
 * column-major grid layouts. It groups tiles into vertical bands of width SWIZZLE_N.
 *
 * 1. 2D Swizzling: Groups tiles into SWIZZLE_N wide vertical bands to maximize
 *    L2 cache sharing of Matrix A and Matrix B among concurrent thread blocks.
 *
 * 2. Snake Traversal:
 *    Reverses the M-direction on odd vertical bands to ensure Matrix A remains
 *    in the L2 cache when transitioning between bands.
 *
 *    Macro Traversal Visualization (Grid divided into bands of SWIZZLE_N):
 *    Band 0       Band 1       Band 2
 *    (n_outer=0)  (n_outer=1)  (n_outer=2)
 *       [0]          [7]          [8]
 *        |            ^            |
 *       [1]          [6]          [9]
 *        |            ^            |
 *       [2]          [5]         [10]
 *        |            ^            |
 *       [3] -> -> -> [4]         [11]
 *
 * 3. XOR Skew:
 *    Staggers the inner N traversal based on the M coordinate to prevent partition
 *    camping. Without this, concurrent blocks might access the same N-coordinates
 *    simultaneously, causing memory bank conflicts.
 *
 *    Micro Traversal Visualization (SWIZZLE_N = 4):
 *    Without XOR (inner_n):          With XOR (inner_n ^ (final_m % 4)):
 *    final_m:  0   1   2   3         final_m:  0   1   2   3
 *           +----------------               +----------------
 *    pos 0  |  0   0   0   0         pos 0  |  0   1   2   3
 *    pos 1  |  1   1   1   1         pos 1  |  1   0   3   2
 *    pos 2  |  2   2   2   2         pos 2  |  2   3   0   1
 *    pos 3  |  3   3   3   3         pos 3  |  3   2   1   0
 */
template<int BLOCK_M, int BLOCK_N, int SWIZZLE_N = 8>
class xor_snake_col_swizzle_mapping
{
    static_assert((SWIZZLE_N & (SWIZZLE_N - 1)) == 0, "SWIZZLE_N must be a power of 2");

public:
    static __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
    {
        const int idx_m = tile_id % grid_m;
        const int idx_n = tile_id / grid_m;

        const int n_outer = idx_n / SWIZZLE_N;
        const int n_inner = idx_n % SWIZZLE_N;

        int final_m, final_n;

        // Fast path for main complete groups
        if(n_outer < grid_n / SWIZZLE_N)
        {
            const int local = idx_m + n_inner * grid_m;
            final_m         = local / SWIZZLE_N;

            // XOR SKEW: Stagger the inner N traversal based on the M coordinate
            int inner_n = local % SWIZZLE_N;
            inner_n     = inner_n ^ (final_m & (SWIZZLE_N - 1));

            final_n = inner_n + n_outer * SWIZZLE_N;
        }
        // Slow path for the remainder edge (No XOR to prevent out-of-bounds)
        else
        {
            const int n_rem = grid_n % SWIZZLE_N;
            const int local = idx_m + n_inner * grid_m;
            final_m         = local / n_rem;
            final_n         = (local % n_rem) + n_outer * SWIZZLE_N;
        }

        // MACRO TRAVERSAL: Snake (Boustrophedon)
        if(n_outer & 1)
        {
            final_m = grid_m - 1 - final_m;
        }

        if(final_m >= grid_m || final_n >= grid_n)
        {
            *block_row = (grid_m - 1) * BLOCK_M;
            *block_col = (grid_n - 1) * BLOCK_N;
            return;
        }

        *block_row = final_m * BLOCK_M;
        *block_col = final_n * BLOCK_N;
    }
};

/**
 * @brief Selector for tile mapping implementation based on layouts
 */
template<m_layout LAYOUT_A, m_layout LAYOUT_B, int SWIZZLE>
struct select_tile_mapping_impl
{
    template<int BLOCK_M, int BLOCK_N>
    using type = row_major_mapping<BLOCK_M, BLOCK_N>;
};

template<int SWIZZLE>
struct select_tile_mapping_impl<m_layout::col_major, m_layout::row_major, SWIZZLE>
{
    template<int BLOCK_M, int BLOCK_N>
    using type = xor_snake_col_swizzle_mapping<BLOCK_M, BLOCK_N, SWIZZLE>;
};

template<int SWIZZLE>
struct select_tile_mapping_impl<m_layout::row_major, m_layout::col_major, SWIZZLE>
{
    template<int BLOCK_M, int BLOCK_N>
    using type = xor_snake_row_swizzle_mapping<BLOCK_M, BLOCK_N, SWIZZLE>;
};

template<int SWIZZLE>
struct select_tile_mapping_impl<m_layout::col_major, m_layout::col_major, SWIZZLE>
{
    template<int BLOCK_M, int BLOCK_N>
    using type = xor_snake_col_swizzle_mapping<BLOCK_M, BLOCK_N, SWIZZLE>;
};

/**
 * @brief Dispatcher for mapping a 1D tile ID to 2D block coordinates.
 *
 * Selects the optimal mapping strategy (e.g., snake traversal, swizzling)
 * based on the memory layouts of the input matrices to maximize L2 cache
 * reuse and avoid partition camping.
 *
 * @tparam BLOCK_M Block dimension M.
 * @tparam BLOCK_N Block dimension N.
 * @tparam LAYOUT_A Memory layout of Matrix A.
 * @tparam LAYOUT_B Memory layout of Matrix B.
 * @tparam SWIZZLE The swizzle size.
 */
template<int BLOCK_M, int BLOCK_N, m_layout LAYOUT_A, m_layout LAYOUT_B, int SWIZZLE>
class tile_mapper
{
    using base_type =
        typename select_tile_mapping_impl<LAYOUT_A, LAYOUT_B, SWIZZLE>::template type<BLOCK_M,
                                                                                      BLOCK_N>;

public:
    /**
     * @brief Maps a 1D block index to 2D block coordinates.
     *
     * @param tile_id The 1D block index (e.g., from `blockIdx.x`).
     * @param grid_m The grid dimension in M.
     * @param grid_n The grid dimension in N.
     * @param block_row Output parameter for the mapped row.
     * @param block_col Output parameter for the mapped column.
     */
    __device__ __forceinline__ void
        map_tile(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col) const
    {
        base_type::map_tile(tile_id, grid_m, grid_n, block_row, block_col);
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_MAPPING_HPP
