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

#ifndef ROCM_WMMA_GEMM_LOAD_HPP
#define ROCM_WMMA_GEMM_LOAD_HPP

namespace rocm_wmma_gemm
{

/**
 * @brief OOB-clamped buffer load/store SRD configuration for gfx11/12.
 */
static constexpr unsigned buffer_rsrc_config = 0x31004000u;

/**
 * @brief Creates a 128-bit buffer resource descriptor with OOB clamping.
 *
 * @param ptr Base pointer for the buffer.
 * @param num_bytes Total allocation size in bytes for bounds checking.
 * @return 128-bit SRD for raw buffer instructions.
 */
static __device__ __forceinline__ __amdgpu_buffer_rsrc_t make_buffer_rsrc(const void* ptr,
                                                                          unsigned    num_bytes)
{
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr),
                                             0,
                                             num_bytes,
                                             static_cast<int>(buffer_rsrc_config));
}

/**
 * @brief Unified helper for vectorized buffer loads.
 *
 * @tparam V Vectorized data type.
 * @param rsrc Resource descriptor with clamped bounds.
 * @param byte_offset Flat byte offset from the buffer base.
 * @return The loaded vector (zero-padded if out of bounds).
 */
template<typename V>
static __device__ __forceinline__ V buffer_load(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    if constexpr(sizeof(V) == 32)
    {
        V     result;
        char* res_ptr = reinterpret_cast<char*>(&result);
        using part_t  = float __attribute__((ext_vector_type(4)));

        part_t raw0 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
        part_t raw1 = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset + 16, 0, 0);

        __builtin_memcpy(res_ptr, &raw0, 16);
        __builtin_memcpy(res_ptr + 16, &raw1, 16);
        return result;
    }
    else if constexpr(sizeof(V) == 24)
    {
        V     result;
        char* res_ptr = reinterpret_cast<char*>(&result);
        using part_t  = float __attribute__((ext_vector_type(3)));

        part_t raw0 = __builtin_amdgcn_raw_buffer_load_b96(rsrc, byte_offset, 0, 0);
        part_t raw1 = __builtin_amdgcn_raw_buffer_load_b96(rsrc, byte_offset + 12, 0, 0);

        __builtin_memcpy(res_ptr, &raw0, 12);
        __builtin_memcpy(res_ptr + 12, &raw1, 12);
        return result;
    }
    else if constexpr(sizeof(V) == 16)
    {
        using payload_t = float __attribute__((ext_vector_type(4)));
        payload_t raw   = __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_offset, 0, 0);
        V         result;
        __builtin_memcpy(&result, &raw, 16);
        return result;
    }
    else if constexpr(sizeof(V) == 12)
    {
        using payload_t = float __attribute__((ext_vector_type(3)));
        payload_t raw   = __builtin_amdgcn_raw_buffer_load_b96(rsrc, byte_offset, 0, 0);
        V         result;
        __builtin_memcpy(&result, &raw, 12);
        return result;
    }
    else if constexpr(sizeof(V) == 8)
    {
        using payload_t = float __attribute__((ext_vector_type(2)));
        payload_t raw   = __builtin_amdgcn_raw_buffer_load_b64(rsrc, byte_offset, 0, 0);
        V         result;
        __builtin_memcpy(&result, &raw, 8);
        return result;
    }
    else
    {
        static_assert(sizeof(V) == 4);
        float raw = __builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0);
        V     result;
        __builtin_memcpy(&result, &raw, 4);
        return result;
    }
}

/**
 * @brief Unified helper for vectorized buffer stores.
 *
 * @tparam V Vectorized data type.
 * @param rsrc Resource descriptor with clamped bounds.
 * @param byte_offset Flat byte offset from the buffer base.
 * @param value The vector to store.
 */
template<typename V>
static __device__ __forceinline__ void
    buffer_store(__amdgpu_buffer_rsrc_t rsrc, int byte_offset, const V& value)
{
    if constexpr(sizeof(V) == 32)
    {
        using part_t = unsigned int __attribute__((ext_vector_type(4)));
        part_t      p0, p1;
        const char* val_ptr = reinterpret_cast<const char*>(&value);
        __builtin_memcpy(&p0, val_ptr, 16);
        __builtin_memcpy(&p1, val_ptr + 16, 16);

        __builtin_amdgcn_raw_buffer_store_b128(p0, rsrc, byte_offset, 0, 0);
        __builtin_amdgcn_raw_buffer_store_b128(p1, rsrc, byte_offset + 16, 0, 0);
    }
    else if constexpr(sizeof(V) == 24)
    {
        using part_t = unsigned int __attribute__((ext_vector_type(3)));
        part_t      p0, p1;
        const char* val_ptr = reinterpret_cast<const char*>(&value);
        __builtin_memcpy(&p0, val_ptr, 12);
        __builtin_memcpy(&p1, val_ptr + 12, 12);

        __builtin_amdgcn_raw_buffer_store_b96(p0, rsrc, byte_offset, 0, 0);
        __builtin_amdgcn_raw_buffer_store_b96(p1, rsrc, byte_offset + 12, 0, 0);
    }
    else if constexpr(sizeof(V) == 16)
    {
        using payload_t = unsigned int __attribute__((ext_vector_type(4)));
        payload_t p;
        __builtin_memcpy(&p, &value, 16);
        __builtin_amdgcn_raw_buffer_store_b128(p, rsrc, byte_offset, 0, 0);
    }
    else if constexpr(sizeof(V) == 12)
    {
        using payload_t = unsigned int __attribute__((ext_vector_type(3)));
        payload_t p;
        __builtin_memcpy(&p, &value, 12);
        __builtin_amdgcn_raw_buffer_store_b96(p, rsrc, byte_offset, 0, 0);
    }
    else if constexpr(sizeof(V) == 8)
    {
        using payload_t = unsigned int __attribute__((ext_vector_type(2)));
        payload_t p;
        __builtin_memcpy(&p, &value, 8);
        __builtin_amdgcn_raw_buffer_store_b64(p, rsrc, byte_offset, 0, 0);
    }
    else
    {
        static_assert(sizeof(V) == 4);
        unsigned int p;
        __builtin_memcpy(&p, &value, 4);
        __builtin_amdgcn_raw_buffer_store_b32(p, rsrc, byte_offset, 0, 0);
    }
}

/**
 * @brief Byte-exact vector type selector.
 *
 * `ext_vector_type` rounds a vector's size/alignment up to a power-of-two element
 * count, so non-power-of-2 widths (e.g. 6 for b96, 12 for b192) get a padded
 * `sizeof` larger than their real byte payload. That padding both misroutes the
 * `sizeof(V)` dispatch in buffer_load/buffer_store and makes register->LDS stores
 * overwrite neighboring cells. For power-of-2 widths we keep the native vector
 * type (single-instruction stores); for the rest we use a plain array aggregate
 * whose `sizeof` is exactly `W * sizeof(B)`.
 */
template<class B, int W, bool IS_POW2 = ((W & (W - 1)) == 0)>
struct load_vector
{
    using type = B __attribute__((ext_vector_type(W)));
};

template<class B, int W>
struct load_vector<B, W, false>
{
    struct type
    {
        B data[W];
    };
};

/**
 * @brief Stores shared memory (LDS) tile blocks out to global memory.
 *
 * @tparam ACCESS The layout of the matrix being written to.
 * @tparam MAX_BITS Maximum bit-width for vectorized store operations.
 * @tparam BLOCK_SIZE Number of threads in the block.
 * @tparam BLOCK_M Number of rows in the block.
 * @tparam BLOCK_N Number of columns in the block.
 * @tparam IS_ALIGNED True if the matrix dimensions are aligned to the vector width.
 * @tparam T The data type of the matrix elements.
 */
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         bool     IS_ALIGNED,
         class T,
         int LDS_PAD = 0>
class shared_to_global_store
{
    static constexpr int contig_dim = (ACCESS == m_layout::row_major) ? BLOCK_N : BLOCK_M;
    static constexpr int iter_dim   = (ACCESS == m_layout::row_major) ? BLOCK_M : BLOCK_N;
    // LDS staging pitch along the contiguous dim. Padding it staggers bank access to kill
    // the write/read bank conflict for the col-major-C epilogue; 0 = compact (today).
    static constexpr int lds_pitch    = contig_dim + LDS_PAD;
    static constexpr int contig_bytes = contig_dim * sizeof(T);
    static constexpr int max_bytes    = MAX_BITS / 8;

    static constexpr bool is_m_pow2        = (BLOCK_M & (BLOCK_M - 1)) == 0;
    static constexpr bool is_n_pow2        = (BLOCK_N & (BLOCK_N - 1)) == 0;
    static constexpr bool is_non_pow2_tile = !is_m_pow2 || !is_n_pow2;

    static constexpr int vector_bytes = []() constexpr
    {
        constexpr auto fits = [](int vb) constexpr -> bool
        {
            if(vb > max_bytes)
            {
                return false;
            }
            if((contig_bytes % vb) != 0)
            {
                return false;
            }
            return true;
        };

        if constexpr(is_non_pow2_tile && fits(24))
        {
            return 24;
        }
        else if constexpr(is_non_pow2_tile && fits(12))
        {
            return 12;
        }
        else if constexpr(fits(32))
        {
            return 32;
        }
        else if constexpr(fits(16))
        {
            return 16;
        }
        else if constexpr(fits(12))
        {
            return 12;
        }
        else if constexpr(fits(8))
        {
            return 8;
        }
        else if constexpr(fits(4))
        {
            return 4;
        }
        else
        {
            return 2;
        }
    }();

    static constexpr int actual_load_width = vector_bytes / sizeof(T);

    using type                        = typename type_selector<T>::type;
    using vector_type                 = typename load_vector<type, actual_load_width>::type;
    static constexpr int vector_width = actual_load_width;

    static constexpr int total_elements        = BLOCK_M * BLOCK_N;
    static constexpr int total_vectors         = total_elements / vector_width;
    static constexpr int vectors_per_thread    = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static constexpr int guaranteed_iterations = total_vectors / BLOCK_SIZE;
    static constexpr int remainder_iterations  = vectors_per_thread - guaranteed_iterations;

    static constexpr int step_elems = BLOCK_SIZE * vector_width;
    static constexpr int iter_inc   = step_elems / contig_dim;
    static constexpr int off_inc    = step_elems % contig_dim;

    __amdgpu_buffer_rsrc_t out_rsrc_;
    T*                     output_;

public:
    /**
     * @brief Constructs the store helper and initializes the buffer SRD.
     *
     * @param output Base pointer to the global memory destination.
     * @param alloc_elems Total elements in the global matrix for OOB clamping.
     */
    __device__ __forceinline__ shared_to_global_store(T* output, unsigned alloc_elems)
        : out_rsrc_(make_buffer_rsrc(output, alloc_elems * static_cast<unsigned>(sizeof(T))))
        , output_(output)
    {}

    /**
     * @brief Copies a block of data from LDS to global memory.
     *
     * @param input Pointer to the shared memory source tile.
     * @param row Starting row offset in the global matrix.
     * @param col Starting col offset in the global matrix.
     * @param M Number of rows in the global matrix.
     * @param N Number of columns in the global matrix.
     * @param tid Thread ID within the block.
     */
    __device__ __forceinline__ void store(const T* input, int row, int col, int M, int N, int tid)
    {
        const int lead_dim = (ACCESS == m_layout::row_major) ? N : M;
        const int base_idx = tid * vector_width;
        int       liter    = base_idx / contig_dim;
        int       lcontig  = base_idx % contig_dim;
        // LDS read cursor uses the (possibly padded) staging pitch; the global store cursor
        // (curr_gstore) uses the real lead_dim. They diverge only when LDS_PAD != 0.
        int curr_sload = liter * lds_pitch + lcontig;

        int grow        = (ACCESS == m_layout::row_major) ? (row + liter) : (row + lcontig);
        int gcol        = (ACCESS == m_layout::row_major) ? (col + lcontig) : (col + liter);
        int curr_gstore = (ACCESS == m_layout::row_major) ? (grow * N + gcol) : (gcol * M + grow);

        auto do_store = [&](int gstore, int sload, int local_grow, int local_gcol)
        {
            const auto& val      = *reinterpret_cast<const vector_type*>(input + sload);
            const int   byte_off = gstore * static_cast<int>(sizeof(T));

            const int g_contig   = (ACCESS == m_layout::row_major) ? local_gcol : local_grow;
            const int g_iter     = (ACCESS == m_layout::row_major) ? local_grow : local_gcol;
            const int contig_max = (ACCESS == m_layout::row_major) ? N : M;
            const int iter_max   = (ACCESS == m_layout::row_major) ? M : N;

            if constexpr(IS_ALIGNED)
            {
                buffer_store<vector_type>(out_rsrc_, byte_off, val);
            }
            else if(g_iter < iter_max && (g_contig + vector_width - 1) < contig_max)
            {
                buffer_store<vector_type>(out_rsrc_, byte_off, val);
            }
            else if(g_iter < iter_max && g_contig < contig_max)
            {
                const int valid = contig_max - g_contig;
                [&]<size_t... v>(std::index_sequence<v...>)
                {
                    (((static_cast<int>(v) < valid) ? (void)(output_[gstore + static_cast<int>(v)]
                                                             = input[sload + static_cast<int>(v)])
                                                    : (void)0),
                     ...);
                }(std::make_index_sequence<vector_width>{});
            }
        };

        auto store_vector_unchecked = [&]<size_t>()
        {
            const int local_grow
                = (ACCESS == m_layout::row_major) ? (row + liter) : (row + lcontig);
            const int local_gcol
                = (ACCESS == m_layout::row_major) ? (col + lcontig) : (col + liter);
            do_store(curr_gstore, curr_sload, local_grow, local_gcol);

            curr_sload += iter_inc * lds_pitch;
            curr_gstore += iter_inc * lead_dim;
            liter += iter_inc;
            if constexpr(off_inc != 0)
            {
                lcontig += off_inc;
                curr_sload += off_inc;
                curr_gstore += off_inc;
                if(lcontig >= contig_dim)
                {
                    lcontig -= contig_dim;
                    liter += 1;
                    curr_sload += lds_pitch - contig_dim;
                    curr_gstore += lead_dim - contig_dim;
                }
            }
        };

        auto store_vector_checked = [&]<size_t>()
        {
            if(liter < iter_dim)
            {
                const int local_grow
                    = (ACCESS == m_layout::row_major) ? (row + liter) : (row + lcontig);
                const int local_gcol
                    = (ACCESS == m_layout::row_major) ? (col + lcontig) : (col + liter);
                do_store(curr_gstore, curr_sload, local_grow, local_gcol);
            }

            curr_sload += iter_inc * lds_pitch;
            curr_gstore += iter_inc * lead_dim;
            liter += iter_inc;
            if constexpr(off_inc != 0)
            {
                lcontig += off_inc;
                curr_sload += off_inc;
                curr_gstore += off_inc;
                if(lcontig >= contig_dim)
                {
                    lcontig -= contig_dim;
                    liter += 1;
                    curr_sload += lds_pitch - contig_dim;
                    curr_gstore += lead_dim - contig_dim;
                }
            }
        };

        if constexpr(guaranteed_iterations > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_vector_unchecked.template operator()<i>(), ...);
            }(std::make_index_sequence<guaranteed_iterations>{});
        }

        if constexpr(remainder_iterations > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_vector_checked.template operator()<i>(), ...);
            }(std::make_index_sequence<remainder_iterations>{});
        }
    }
};

/**
 * @brief Prefetches global memory tile blocks into registers.
 *
 * Used to stage loads from global memory into registers before committing them
 * to LDS, enabling software pipelining.
 *
 * @tparam ACCESS The memory layout of the block being prefetched.
 * @tparam MAX_BITS Maximum bit-width for vectorized memory operations.
 * @tparam BLOCK_SIZE Number of threads participating in the prefetch.
 * @tparam BLOCK_M Number of rows in the block.
 * @tparam BLOCK_N Number of columns in the block.
 * @tparam PADDING Padding added to shared memory to avoid bank conflicts.
 * @tparam T Data type of the elements.
 */
template<m_layout ACCESS,
         int      MAX_BITS,
         int      BLOCK_SIZE,
         int      BLOCK_M,
         int      BLOCK_N,
         int      PADDING,
         class T>
class prefetch_fragment
{
    static constexpr int contig_dim   = (ACCESS == m_layout::row_major) ? BLOCK_N : BLOCK_M;
    static constexpr int iter_dim     = (ACCESS == m_layout::row_major) ? BLOCK_M : BLOCK_N;
    static constexpr int contig_bytes = contig_dim * sizeof(T);
    static constexpr int max_bytes    = MAX_BITS / 8;

    static constexpr bool is_m_pow2        = (BLOCK_M & (BLOCK_M - 1)) == 0;
    static constexpr bool is_n_pow2        = (BLOCK_N & (BLOCK_N - 1)) == 0;
    static constexpr bool is_non_pow2_tile = !is_m_pow2 || !is_n_pow2;

    static constexpr int vector_bytes = []() constexpr
    {
        constexpr auto fits = [](int vb) constexpr -> bool
        {
            if(vb > max_bytes)
            {
                return false;
            }
            if((contig_bytes % vb) != 0)
            {
                return false;
            }
            return true;
        };

        if constexpr(is_non_pow2_tile && fits(24))
        {
            return 24;
        }
        else if constexpr(is_non_pow2_tile && fits(12))
        {
            return 12;
        }
        else if constexpr(fits(32))
        {
            return 32;
        }
        else if constexpr(fits(16))
        {
            return 16;
        }
        else if constexpr(fits(12))
        {
            return 12;
        }
        else if constexpr(fits(8))
        {
            return 8;
        }
        else if constexpr(fits(4))
        {
            return 4;
        }
        else
        {
            return 2;
        }
    }();

    static constexpr int vector_width = vector_bytes / sizeof(T);

    static constexpr int total_elements     = BLOCK_M * BLOCK_N;
    static constexpr int total_vectors      = total_elements / vector_width;
    static constexpr int vectors_per_thread = (total_vectors + BLOCK_SIZE - 1) / BLOCK_SIZE;
    static constexpr int guaranteed_iters   = total_vectors / BLOCK_SIZE;
    static constexpr int remainder_iters    = vectors_per_thread - guaranteed_iters;

    using base_type   = typename type_selector<T>::type;
    using vector_type = typename load_vector<base_type, vector_width>::type;

    vector_type regs[vectors_per_thread];

    static constexpr int step_elems = BLOCK_SIZE * vector_width;
    static constexpr int iter_inc   = step_elems / contig_dim;
    static constexpr int off_inc    = step_elems % contig_dim;

    __amdgpu_buffer_rsrc_t rsrc_;
    const T*               base_ptr_;

public:
    /**
     * @brief Constructs the prefetch fragment and initializes the buffer SRD.
     *
     * @param base Base pointer to the global memory matrix.
     * @param alloc_elems Total elements in the global matrix for OOB clamping.
     */
    __device__ __forceinline__ prefetch_fragment(const T* base, unsigned alloc_elems)
        : rsrc_(make_buffer_rsrc(base, alloc_elems * static_cast<unsigned>(sizeof(T))))
        , base_ptr_(base)
    {}

    __device__ __forceinline__ void advance_global(int lead, int& iter, int& off, int& curr)
    {
        curr += iter_inc * lead;
        iter += iter_inc;
        if constexpr(off_inc != 0)
        {
            off += off_inc;
            curr += off_inc;
            if(off >= contig_dim)
            {
                off -= contig_dim;
                iter += 1;
                curr += lead - contig_dim;
            }
        }
    }

    __device__ __forceinline__ void load_global_unchecked(
        int tile_byte_offset, size_t reg, int lead, int& iter, int& off, int& curr)
    {
        const int byte_off = tile_byte_offset + curr * static_cast<int>(sizeof(T));
        regs[reg]          = buffer_load<vector_type>(rsrc_, byte_off);
        advance_global(lead, iter, off, curr);
    }

    __device__ __forceinline__ void load_global_checked(
        int tile_byte_offset, size_t reg, int lead, int& iter, int& off, int& curr)
    {
        if(iter < iter_dim)
        {
            const int byte_off = tile_byte_offset + curr * static_cast<int>(sizeof(T));
            regs[reg]          = buffer_load<vector_type>(rsrc_, byte_off);
        }
        advance_global(lead, iter, off, curr);
    }

    /**
     * @brief Prefetches a full tile block from global memory into registers.
     *
     * @param input Pointer to the global memory tile offset.
     * @param lead Leading dimension size.
     * @param tid Thread ID in the block.
     */
    __device__ __forceinline__ void prefetch(const T* input, int lead, int tid)
    {
        const int tile_byte_offset = static_cast<int>(reinterpret_cast<const char*>(input)
                                                      - reinterpret_cast<const char*>(base_ptr_));
        const int base_idx         = tid * vector_width;
        int       iter             = base_idx / contig_dim;
        int       off              = base_idx % contig_dim;
        int       curr             = iter * lead + off;

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (load_global_unchecked(tile_byte_offset, i, lead, iter, off, curr), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }

        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (load_global_checked(tile_byte_offset, guaranteed_iters + i, lead, iter, off, curr),
                 ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }

    /**
     * @brief Partially prefetches a chunk of data into registers.
     *
     * @tparam STEP Current pipeline step index.
     * @tparam TOTAL_STEPS Total pipeline steps.
     * @param input Pointer to the global memory tile offset.
     * @param lead Leading dimension size.
     * @param tid Thread ID in the block.
     */
    template<size_t STEP, size_t TOTAL_STEPS>
    __device__ __forceinline__ void partial_prefetch(const T* input, int lead, int tid)
    {
        constexpr size_t vpt            = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start          = STEP * loads_per_step;
        constexpr size_t end   = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;
        constexpr size_t count = (end > start) ? (end - start) : 0;

        if constexpr(count > 0)
        {
            const int tile_byte_offset = static_cast<int>(
                reinterpret_cast<const char*>(input) - reinterpret_cast<const char*>(base_ptr_));
            const int base_idx = tid * vector_width;
            const int seed     = base_idx + static_cast<int>(start) * step_elems;
            int       iter     = seed / contig_dim;
            int       off      = seed % contig_dim;
            int       curr     = iter * lead + off;

            [&]<size_t... i>(std::index_sequence<i...>)
            {
                (
                    [&]()
                    {
                        constexpr size_t reg = start + i;
                        if constexpr(reg < static_cast<size_t>(guaranteed_iters))
                        {
                            load_global_unchecked(tile_byte_offset, reg, lead, iter, off, curr);
                        }
                        else
                        {
                            load_global_checked(tile_byte_offset, reg, lead, iter, off, curr);
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<count>{});
        }
    }

    static constexpr int lds_pitch = contig_dim + PADDING;

    __device__ __forceinline__ void advance_lds(int& iter, int& off, int& curr)
    {
        curr += iter_inc * lds_pitch;
        iter += iter_inc;
        if constexpr(off_inc != 0)
        {
            off += off_inc;
            curr += off_inc;
            if(off >= contig_dim)
            {
                off -= contig_dim;
                iter += 1;
                curr += lds_pitch - contig_dim;
            }
        }
    }

    __device__ __forceinline__ void
        store_lds_unchecked(T* output, size_t reg, int& iter, int& off, int& curr)
    {
        *reinterpret_cast<vector_type*>(output + curr) = regs[reg];
        advance_lds(iter, off, curr);
    }

    __device__ __forceinline__ void
        store_lds_checked(T* output, size_t reg, int& iter, int& off, int& curr)
    {
        if(iter < iter_dim)
        {
            *reinterpret_cast<vector_type*>(output + curr) = regs[reg];
        }
        advance_lds(iter, off, curr);
    }

    /**
     * @brief Partially commits prefetched register data to shared memory.
     *
     * @tparam STEP Current pipeline step index.
     * @tparam TOTAL_STEPS Total pipeline steps.
     * @param output Pointer to the LDS destination.
     * @param tid Thread ID in the block.
     */
    template<size_t STEP, size_t TOTAL_STEPS>
    __device__ __forceinline__ void partial_commit(T* output, int tid)
    {
        constexpr size_t vpt            = static_cast<size_t>(vectors_per_thread);
        constexpr size_t loads_per_step = (vpt + TOTAL_STEPS - 1) / TOTAL_STEPS;
        constexpr size_t start          = STEP * loads_per_step;
        constexpr size_t end   = (start + loads_per_step > vpt) ? vpt : start + loads_per_step;
        constexpr size_t count = (end > start) ? (end - start) : 0;

        if constexpr(count > 0)
        {
            const int base_idx = tid * vector_width;
            const int seed     = base_idx + static_cast<int>(start) * step_elems;
            int       iter     = seed / contig_dim;
            int       off      = seed % contig_dim;
            int       curr     = iter * lds_pitch + off;

            [&]<size_t... i>(std::index_sequence<i...>)
            {
                (
                    [&]()
                    {
                        constexpr size_t reg = start + i;
                        if constexpr(reg < static_cast<size_t>(guaranteed_iters))
                        {
                            store_lds_unchecked(output, reg, iter, off, curr);
                        }
                        else
                        {
                            store_lds_checked(output, reg, iter, off, curr);
                        }
                    }(),
                    ...);
            }(std::make_index_sequence<count>{});
        }
    }

    /**
     * @brief Commits all prefetched register data to shared memory.
     *
     * @param output Pointer to the LDS destination.
     * @param tid Thread ID in the block.
     */
    __device__ __forceinline__ void commit(T* output, int tid)
    {
        const int base_idx = tid * vector_width;
        int       iter     = base_idx / contig_dim;
        int       off      = base_idx % contig_dim;
        int       curr     = iter * lds_pitch + off;

        if constexpr(guaranteed_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_lds_unchecked(output, i, iter, off, curr), ...);
            }(std::make_index_sequence<guaranteed_iters>{});
        }

        if constexpr(remainder_iters > 0)
        {
            [&]<size_t... i>(std::index_sequence<i...>) {
                (store_lds_checked(output, guaranteed_iters + i, iter, off, curr), ...);
            }(std::make_index_sequence<remainder_iters>{});
        }
    }
};

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_LOAD_HPP
