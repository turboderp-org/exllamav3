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

// EXL3 MOD: vendored from rocm_wmma_gemm ea3aa74; two marked edits (half bit-casts where
// __HIP_NO_HALF_CONVERSIONS__ removes the implicit conversion).

#ifndef ROCM_WMMA_GEMM_FRAGMENT_HPP
#define ROCM_WMMA_GEMM_FRAGMENT_HPP

namespace rocm_wmma_gemm
{

/**
 * @brief Represents a WMMA register fragment for holding a tile of matrix data.
 *
 * @tparam T The high-level data type of the matrix elements.
 * @tparam TILE The dimension size of the tile (e.g., 16 for a 16x16 tile).
 */
template<class T, int TILE>
class fragment
{
public:
    static constexpr int NEW_TILE = (sizeof(T) == sizeof(float)) ? TILE / 2 : TILE;
    using underlying_type         = T;
    using type                    = typename type_selector<T>::type;
    using frag_vec                = type __attribute__((ext_vector_type(NEW_TILE)));
    using value_type              = type;

private:
    frag_vec _fragment = {};

public:
    /**
     * @brief Proxy class for accessing elements of the fragment.
     */
    class proxy
    {
        frag_vec& vec_ref;
        int       index;

        friend class iterator;

    public:
        __device__ __forceinline__ proxy(frag_vec& v, int i) : vec_ref(v), index(i) {}

        template<typename U = type>
        __device__ __forceinline__ auto operator=(type value) ->
            typename std::enable_if<!std::is_same<U, T>::value, proxy&>::type
        {
            vec_ref[index] = value;
            return *this;
        }

        // This operator handles the T type and also serves as fallback when type == T
        __device__ __forceinline__ proxy& operator=(const T& value)
        {
            if constexpr(std::is_same<T, __hip_bfloat16>::value)
            {
                vec_ref[index] = __bfloat16_as_short(value);
            }
            // EXL3 MOD begin: exllamav3 builds with __HIP_NO_HALF_CONVERSIONS__, which removes
            // __half's implicit conversions; reinterpret the bits instead (same value).
            else if constexpr(std::is_same<T, half>::value && !std::is_same<type, T>::value)
            {
                vec_ref[index] = __builtin_bit_cast(type, value);
            }
            // EXL3 MOD end
            else
            {
                vec_ref[index] = static_cast<type>(value);
            }

            return *this;
        }

        __device__ __forceinline__ operator type() const
        {
            return vec_ref[index];
        }

        proxy*       operator&()       = delete;
        const proxy* operator&() const = delete;
        proxy(const proxy&)            = delete;
    };

    /**
     * @brief Iterator class for traversing elements of the fragment.
     */
    class iterator
    {
        frag_vec& vec_ref;
        int       current_index;

        friend class fragment<T, TILE>;

        __device__ __forceinline__ iterator(frag_vec& v, int i) : vec_ref(v), current_index(i) {}

    public:
        __device__ __forceinline__ proxy operator*() const
        {
            return proxy(vec_ref, current_index);
        }

        __device__ __forceinline__ iterator& operator++()
        {
            ++current_index;
            return *this;
        }

        __device__ __forceinline__ iterator& operator+=(int n)
        {
            current_index += n;
            return *this;
        }

        __device__ __forceinline__ iterator operator+(int n) const
        {
            iterator temp = *this;
            temp += n;
            return temp;
        }

        __device__ __forceinline__ bool operator!=(const iterator& other) const
        {
            return current_index != other.current_index;
        }
    };

public:
    __device__ __forceinline__ iterator begin()
    {
        return iterator(_fragment, 0);
    }

    __device__ __forceinline__ iterator end()
    {
        return iterator(_fragment, TILE);
    }

    __device__ __forceinline__ frag_vec& get()
    {
        return _fragment;
    }

    __device__ __forceinline__ const frag_vec& get() const
    {
        return _fragment;
    }

    __device__ __forceinline__ T operator[](int i) const
    {
        if constexpr(std::is_same<T, __hip_bfloat16>::value)
        {
            return __short_as_bfloat16(_fragment[i]);
        }
        // EXL3 MOD begin: see the proxy's operator= (no implicit __half conversions)
        else if constexpr(std::is_same<T, half>::value && !std::is_same<type, T>::value)
        {
            // copy the element out first: __builtin_bit_cast on a vector-element lvalue
            // reads element 0 (measured: every row of a tile came back as rows 0/1)
            const type v = _fragment[i];
            return __builtin_bit_cast(T, v);
        }
        // EXL3 MOD end
        else
        {
            return _fragment[i];
        }
    }
};

/**
 * @brief Loads a matrix tile into a fragment for natively aligned layouts.
 *
 * Natively aligned layouts mean the requested matrix elements are contiguous in memory.
 * For Matrix A, this is row-major. For Matrix B, this is col-major. In these cases,
 * a single vectorized load instruction can fetch the entire tile row/column.
 *
 * @tparam MATRIX Indicates whether we are loading Matrix A or Matrix B.
 * @tparam ACCESS The layout of the source data.
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param frag The destination fragment to populate.
 * @param data Pointer to the starting position in shared or global memory.
 * @param M Leading dimension (unused for native aligned fast path).
 * @param N Stride (unused for native aligned fast path).
 */
template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::row_major)
                                || (MATRIX == m_input::matrix_b && ACCESS == m_layout::col_major),
                            void>::type
{
    const T* tmp = data;
    auto     it  = frag.begin();

    auto load_element = [&]<size_t i>()
    {
        *it = *tmp;
        ++tmp;
        ++it;
    };

    [&]<size_t... i>(std::index_sequence<i...>)
    { (load_element.template operator()<i>(), ...); }(std::make_index_sequence<TILE>{});
}

/**
 * @brief Loads a Matrix A tile into a fragment from a non-native (col-major) layout.
 *
 * Since Matrix A natively expects row-major (contiguous rows), a col-major access
 * requires strided memory reads (jumping by M elements for each item in the tile row).
 *
 * @tparam MATRIX Must be matrix_a.
 * @tparam ACCESS Must be col_major.
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param frag The destination fragment to populate.
 * @param data Pointer to the starting position in memory.
 * @param M The stride between consecutive elements in the logical row.
 * @param N Unused.
 */
template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::col_major),
                            void>::type
{
    const T* tmp = data;
    auto     it  = frag.begin();

    auto load_element = [&]<size_t i>()
    {
        *it = *tmp;
        tmp += M;
        ++it;
    };

    [&]<size_t... i>(std::index_sequence<i...>)
    { (load_element.template operator()<i>(), ...); }(std::make_index_sequence<TILE>{});
}

/**
 * @brief Loads a Matrix B tile into a fragment from a non-native (row-major) layout.
 *
 * Since Matrix B natively expects col-major (contiguous columns), a row-major access
 * requires strided memory reads (jumping by N elements for each item in the tile col).
 *
 * @tparam MATRIX Must be matrix_b.
 * @tparam ACCESS Must be row_major.
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param frag The destination fragment to populate.
 * @param data Pointer to the starting position in memory.
 * @param M Unused.
 * @param N The stride between consecutive elements in the logical column.
 */
template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::row_major),
                            void>::type
{
    const T* tmp = data;
    auto     it  = frag.begin();

    auto load_element = [&]<size_t i>()
    {
        *it = *tmp;
        tmp += N;
        ++it;
    };

    [&]<size_t... i>(std::index_sequence<i...>)
    { (load_element.template operator()<i>(), ...); }(std::make_index_sequence<TILE>{});
}

/**
 * @brief Stores an accumulator fragment to global memory in row-major layout with bounds checking.
 *
 * @tparam ACCESS Must be row_major.
 * @tparam USE_SHARED False indicating global memory (requires bounds checking).
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param data Pointer to the destination matrix.
 * @param frag The accumulator fragment to store.
 * @param row The starting global row index.
 * @param col The starting global column index.
 * @param M The total number of rows in the global matrix (for bounds checking).
 * @param N The total number of columns in the global matrix (for bounds checking).
 */
template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::row_major && !USE_SHARED, void>::type
{
    auto store_element = [&]<size_t i>()
    {
        constexpr int r = i * 2;
        constexpr int s = (std::is_same<T, float>::value || std::is_same<T, int>::value) ? i : r;
        if((row + r) < M && col < N)
        {
            data[(row + r) * N + col] = frag[s];
        }
    };

    [&]<size_t... i>(std::index_sequence<i...>)
    { (store_element.template operator()<i>(), ...); }(std::make_index_sequence<TILE / 2>{});
}

/**
 * @brief Stores an accumulator fragment to shared memory in row-major layout without bounds checking.
 *
 * @tparam ACCESS Must be row_major.
 * @tparam USE_SHARED True indicating shared memory (no bounds checking needed since LDS is sized perfectly).
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param data Pointer to the shared memory buffer.
 * @param frag The accumulator fragment to store.
 * @param row The local row index within the LDS buffer.
 * @param col The local column index within the LDS buffer.
 * @param M Unused.
 * @param N The row stride (pitch) of the shared memory buffer.
 */
template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::row_major && USE_SHARED, void>::type
{
    auto store_element = [&]<size_t i>()
    {
        constexpr int r = i * 2;
        constexpr int s = (std::is_same<T, float>::value || std::is_same<T, int>::value) ? i : r;
        data[(row + r) * N + col] = frag[s];
    };

    [&]<size_t... i>(std::index_sequence<i...>)
    { (store_element.template operator()<i>(), ...); }(std::make_index_sequence<TILE / 2>{});
}

/**
 * @brief Stores an accumulator fragment to global memory in col-major layout with bounds checking.
 *
 * @tparam ACCESS Must be col_major.
 * @tparam USE_SHARED False indicating global memory (requires bounds checking).
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param data Pointer to the destination matrix.
 * @param frag The accumulator fragment to store.
 * @param row The starting global row index.
 * @param col The starting global column index.
 * @param M The total number of rows in the global matrix (for bounds checking).
 * @param N The total number of columns in the global matrix (for bounds checking).
 */
template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::col_major && !USE_SHARED, void>::type
{
    auto store_element = [&]<size_t i>()
    {
        constexpr int r = i * 2;
        constexpr int s = (std::is_same<T, float>::value || std::is_same<T, int>::value) ? i : r;
        if((row + r) < M && col < N)
        {
            data[col * M + (row + r)] = frag[s];
        }
    };

    [&]<size_t... i>(std::index_sequence<i...>)
    { (store_element.template operator()<i>(), ...); }(std::make_index_sequence<TILE / 2>{});
}

/**
 * @brief Stores an accumulator fragment to shared memory in col-major layout without bounds checking.
 *
 * @tparam ACCESS Must be col_major.
 * @tparam USE_SHARED True indicating shared memory (no bounds checking needed).
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param data Pointer to the shared memory buffer.
 * @param frag The accumulator fragment to store.
 * @param row The local row index within the LDS buffer.
 * @param col The local column index within the LDS buffer.
 * @param M The column stride (pitch) of the shared memory buffer.
 * @param N Unused.
 */
template<m_layout ACCESS, bool USE_SHARED, class T, int TILE>
__device__ __forceinline__ auto
    store_matrix(T* data, fragment<T, TILE>& frag, int row, int col, int M, int N) ->
    typename std::enable_if<ACCESS == m_layout::col_major && USE_SHARED, void>::type
{
    auto store_element = [&]<size_t i>()
    {
        constexpr int r = i * 2;
        constexpr int s = (std::is_same<T, float>::value || std::is_same<T, int>::value) ? i : r;
        data[col * M + (row + r)] = frag[s];
    };

    [&]<size_t... i>(std::index_sequence<i...>)
    { (store_element.template operator()<i>(), ...); }(std::make_index_sequence<TILE / 2>{});
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_FRAGMENT_HPP
