#pragma once

// Copyright (c) 2023-2026 The ggml authors
// Minimal FP16-input, FP32-accumulator MMA tile helpers used by the SM120 attention kernel.
// The layouts and instructions are derived from llama.cpp's ggml-cuda/mma.cuh
// under the MIT license. Only the NVIDIA m16n8k16 paths needed here are kept.

#include <cuda_fp16.h>
#include <cstdint>

namespace exl3_sm120 {

enum data_layout {
    DATA_LAYOUT_I_MAJOR = 0,
};

template <int I_, int J_, typename T, data_layout dl_ = DATA_LAYOUT_I_MAJOR>
struct tile;

template <int I_, int J_>
struct tile<I_, J_, float, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    float x[ne] = {0.0f};

    static __device__ __forceinline__ int get_i(const int l) {
        static_assert(I == 16 && (J == 8 || J == 16));
        if constexpr (J == 8) {
            return ((l / 2) * 8) + (threadIdx.x / 4);
        } else {
            return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
        }
    }

    static __device__ __forceinline__ int get_j(const int l) {
        static_assert(I == 16 && (J == 8 || J == 16));
        if constexpr (J == 8) {
            return ((threadIdx.x % 4) * 2) + (l % 2);
        } else {
            return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
        }
    }
};

template <int I_, int J_>
struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR> {
    static constexpr int I = I_;
    static constexpr int J = J_;
    static constexpr int ne = I * J / 32;
    half2 x[ne] = {};

    static __device__ __forceinline__ int get_i(const int l) {
        static_assert(I == 16 && J == 8);
        return ((l % 2) * 8) + (threadIdx.x / 4);
    }

    static __device__ __forceinline__ int get_j(const int l) {
        static_assert(I == 16 && J == 8);
        return ((l / 2) * 4) + (threadIdx.x % 4);
    }
};

static __device__ __forceinline__ half2 make_half2_f(const float x, const float y) {
    return __floats2half2_rn(x, y);
}

template <int I, int J>
static __device__ __forceinline__ tile<I, J / 2, half2> get_half2(const tile<I, J, float>& src) {
    tile<I, J / 2, half2> dst;
#pragma unroll
    for (int l = 0; l < src.ne; l += 2) {
        dst.x[l / 2] = make_half2_f(src.x[l], src.x[l + 1]);
    }
    return dst;
}

static __device__ __forceinline__ void load_ldmatrix(
    tile<16, 8, half2>& dst,
    const half2* src,
    const int stride
) {
    int* d = reinterpret_cast<int*>(dst.x);
    const int* s = reinterpret_cast<const int*>(src) +
        (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "l"(s));
}

static __device__ __forceinline__ void mma(
    tile<16, 16, float>& d,
    const tile<16, 8, half2>& a,
    const tile<16, 8, half2>& b
) {
    const int* ai = reinterpret_cast<const int*>(a.x);
    const int* bi = reinterpret_cast<const int*>(b.x);
    float* di = d.x;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(di[0]), "+f"(di[1]), "+f"(di[2]), "+f"(di[3])
        : "r"(ai[0]), "r"(ai[1]), "r"(ai[2]), "r"(ai[3]), "r"(bi[0]), "r"(bi[2]));
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(di[4]), "+f"(di[5]), "+f"(di[6]), "+f"(di[7])
        : "r"(ai[0]), "r"(ai[1]), "r"(ai[2]), "r"(ai[3]), "r"(bi[1]), "r"(bi[3]));
}

template <int D, int NCOLS>
struct mma_tile_sizes {
    using T_A_KQ  = tile<16, 8, half2>;
    using T_B_KQ  = tile<16, 8, half2>;
    using T_C_KQ  = tile<16, 16, float>;
    using T_A_VKQ = tile<16, 8, half2>;
    using T_B_VKQ = tile<16, 8, half2>;
    using T_C_VKQ = tile<16, 16, float>;
};

static constexpr __device__ int get_cols_per_thread() {
    return 2;
}

} // namespace exl3_sm120
