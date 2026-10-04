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

#ifndef ROCM_WMMA_GEMM_COMMON_HPP
#define ROCM_WMMA_GEMM_COMMON_HPP

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace rocm_wmma_gemm
{

/**
 * @brief Enum class defining matrix memory layout options.
 */
enum class m_layout
{
    row_major, ///< Row-major layout (elements consecutive in memory by row)
    col_major ///< Column-major layout (elements consecutive in memory by column)
};

/**
 * @brief Enum to specify which input matrix is being accessed.
 */
enum class m_input
{
    matrix_a, ///< Refers to input matrix A
    matrix_b ///< Refers to input matrix B
};

/**
 * @brief Type selector struct for mapping high-level types to internal types.
 *
 * @tparam T The high-level data type (e.g., float, half, __hip_bfloat16).
 */
template<class T>
struct type_selector
{
    using type = T; ///< The mapped internal type.
};

/**
 * @brief Type selector specialization for half precision.
 */
template<>
struct type_selector<half>
{
    using type = _Float16; ///< Maps to _Float16 internally.
};

/**
 * @brief Type selector specialization for bfloat16 precision.
 */
template<>
struct type_selector<__hip_bfloat16>
{
    using type = short; ///< Maps to short internally.
};

/** @brief The base dimension size of a WMMA tile (16x16). */
constexpr int wmma_tile = 16;

/** @brief The number of threads in a warp for AMD GPUs (Wave32). */
constexpr int warp_size = 32;

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_COMMON_HPP
