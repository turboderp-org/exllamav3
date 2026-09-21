#pragma once

// Copyright (c) 2023-2026 The ggml authors
// Producer-consumer TMA attention derived from llama.cpp's
// ggml/src/ggml-cuda/fattn-mma-f16-sm120.cuh under the MIT license.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>

#include "attention_sm120_mma.cuh"

namespace exl3_sm120 {

static constexpr int WARP_SIZE = 32;
static constexpr float SOFTMAX_FTZ_THRESHOLD = -20.0f;
static constexpr float FATTN_KQ_MAX_OFFSET = 3.0f * 0.6931f;

static __device__ __forceinline__ half2 make_half2(const float x, const float y) {
    return make_half2_f(x, y);
}

#define TURING_MMA_AVAILABLE 1
#define EXL3_UNUSED_VARS(...) do {} while (false)
#define EXL3_NO_DEVICE_CODE do {} while (false)

template<int DKQ> struct config;

template<> struct config<128> {
    static constexpr int nbatch_fa      = 64;
    static constexpr int nbatch_combine = 64;
};

template<> struct config<256> {
    static constexpr int nbatch_fa      = 64;
    static constexpr int nbatch_combine = 128;
};

template<> struct config<512> {
    static constexpr int nbatch_fa      = 64;
    static constexpr int nbatch_combine = 128;
};

static constexpr int nwarps       = 4;
static constexpr int depth        = 2;
static constexpr int chunk_h2     = 32;
static constexpr int chunk_ne     = 2*chunk_h2;
static constexpr int barrier_id   = 7;

struct alignas(8) circular_barriers {
    uint64_t produced[depth];
    uint64_t consumed[depth];
};

struct alignas(8) barrier_storage {
    uint64_t q_ready;
    circular_barriers kv;
};

static __device__ __forceinline__ uint32_t smem_addr(const void * ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void barrier_init(uint64_t * barrier, const uint32_t arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(barrier)), "r"(arrive_count) : "memory");
}

static __device__ __forceinline__ void barrier_arrive_expect_tx(uint64_t * barrier, const uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_addr(barrier)), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void barrier_arrive(uint64_t * barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_addr(barrier)) : "memory");
}

static __device__ __forceinline__ void barrier_wait(uint64_t * barrier, const uint32_t phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "L_fattn_tma_wait_%=: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
        "@p bra L_fattn_tma_done_%=;\n\t"
        "bra L_fattn_tma_wait_%=;\n\t"
        "L_fattn_tma_done_%=: \n\t"
        "}"
        :: "r"(smem_addr(barrier)), "r"(phase), "r"(0x989680) : "memory");
}

static __device__ __forceinline__ void tma_load_3d(
        void * dst, const CUtensorMap * map, uint64_t * barrier,
        const int32_t c0, const int32_t c1, const int32_t c2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(smem_addr(dst)), "l"(reinterpret_cast<uint64_t>(map)),
           "r"(c0), "r"(c1), "r"(c2), "r"(smem_addr(barrier)) : "memory");
#endif
}

static __device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync %0, %1;" :: "n"(barrier_id), "n"(nwarps*WARP_SIZE) : "memory");
}

static __device__ __forceinline__ float softmax_rescale(const float diff) {
    return diff >= SOFTMAX_FTZ_THRESHOLD ? expf(diff) : 0.0f;
}

template<data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_swizzle_128(
        tile<16, 8, half2, dl> & t, const half2 * base, const int row0, const int col0, const int stride) {
    int * xi = reinterpret_cast<int *>(t.x);
    const int row = row0 + threadIdx.x % 16;
    const int col = col0 + (threadIdx.x / 16)*4;
    const int col_swizzled = col ^ ((row & 7)*4);
    const int * xs = reinterpret_cast<const int *>(base) + row*stride + col_swizzled;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3]) : "l"(xs));
}

template<data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_trans_swizzle_128(
        tile<16, 8, half2, dl> & t, const half2 * base, const int row0, const int col0, const int stride) {
    int * xi = reinterpret_cast<int *>(t.x);
    const int row = row0 + threadIdx.x % 16;
    const int col = col0 + (threadIdx.x / 16)*4;
    const int col_swizzled = col ^ ((row & 7)*4);
    const int * xs = reinterpret_cast<const int *>(base) + row*stride + col_swizzled;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3]) : "l"(xs));
}

struct writer {
    circular_barriers * barriers;
    uint32_t ptr   = 0;
    uint32_t phase = 0xffffffffu;

    __device__ __forceinline__ int reserve(const bool elected) {
        if (elected) {
            barrier_wait(&barriers->consumed[ptr], (phase >> ptr) & 1u);
        }
        __syncwarp();
        return ptr;
    }

    __device__ __forceinline__ void advance() {
        phase ^= 1u << ptr;
        ptr = ptr + 1 == depth ? 0 : ptr + 1;
    }
};

struct reader {
    circular_barriers * barriers;
    uint32_t ptr   = 0;
    uint32_t phase = 0;

    __device__ __forceinline__ int wait() {
        barrier_wait(&barriers->produced[ptr], (phase >> ptr) & 1u);
        return ptr;
    }

    __device__ __forceinline__ void pop(const int slot) {
        barrier_arrive(&barriers->consumed[slot]);
        phase ^= 1u << ptr;
        ptr = ptr + 1 == depth ? 0 : ptr + 1;
    }
};

template<int DKQ, int ncols1>
struct shared_layout {
    static constexpr int ncols            = 64;
    static constexpr int nbatch_fa        = config<DKQ>::nbatch_fa;
    static constexpr int q_h2             = ncols*(DKQ/2);
    static constexpr int kv_slot_h2       = nbatch_fa*chunk_h2;
    static constexpr int stream_bytes     = depth*kv_slot_h2*sizeof(half2);
    static constexpr int pipeline_bytes   = q_h2*sizeof(half2) + stream_bytes;
    static constexpr int combine_f32      = nwarps*16*(config<DKQ>::nbatch_combine + 4);
    static constexpr int data_bytes       = pipeline_bytes > combine_f32*int(sizeof(float)) ? pipeline_bytes : combine_f32*int(sizeof(float));
    static constexpr int shared_bytes     = data_bytes + sizeof(barrier_storage);
    static_assert(data_bytes % alignof(barrier_storage) == 0, "misaligned SM120 attention barriers");

    half2 * data;
    barrier_storage * barriers;

    __device__ explicit shared_layout(void * smem) {
        data = reinterpret_cast<half2 *>(smem);
        barriers = reinterpret_cast<barrier_storage *>(reinterpret_cast<uint8_t *>(smem) + data_bytes);
    }

    __device__ __forceinline__ half2 * q() const {
        return data;
    }

    __device__ __forceinline__ half2 * kv(const int slot) const {
        return data + q_h2 + slot*kv_slot_h2;
    }

};

template<int DKQ, int ncols1, int ncols2>
static __device__ __forceinline__ void producer(
        const half2 * Q_h2, const float scale,
        const int stride_Q1, const int stride_Q2,
        const int jt, const int zt_gqa, const int gqa_ratio, const int ne01,
        const int sequence, const int z_KV, const int kb0_start, const int kb0_stop,
        const int32_t * block_table, const int num_pages_per_seq,
        const CUtensorMap * map_k, const CUtensorMap * map_v,
        shared_layout<DKQ, ncols1> layout) {
    constexpr int nbatch_fa = config<DKQ>::nbatch_fa;
    constexpr int nchunks   = DKQ/chunk_ne;
    const int lane = threadIdx.x;
    const bool elected = lane == 0;

    writer kvw{&layout.barriers->kv};
    const half2 scale_h2 = make_half2(scale, scale);
    half2 * tile_q = layout.q();

    for (int i = lane; i < 64*(DKQ/2); i += WARP_SIZE) {
        const int jc = i/(DKQ/2);
        const int k = i - jc*(DKQ/2);
        const int j = jc/ncols2;
        const int c = jc - j*ncols2;
        half2 value = make_half2(0.0f, 0.0f);
        if (jt*ncols1 + j < ne01 && zt_gqa*ncols2 + c < gqa_ratio) {
            value = __hmul2(scale_h2, Q_h2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k]);
        }
        const int k_smem = DKQ > 256 ? k ^ ((jc & 7)*4) : k;
        tile_q[jc*(DKQ/2) + k_smem] = value;
    }
    __syncwarp();
    // Each Q-tile writer publishes its own stores before consumers read the tile.
    __threadfence_block();
    barrier_arrive(&layout.barriers->q_ready);

    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        const int logical_token = kb0*nbatch_fa;
        const int logical_page = logical_token >> 8;
        const int page_offset = logical_token & 255;
        const int physical_page = block_table[sequence*num_pages_per_seq + logical_page];
        const int physical_token = physical_page*256 + page_offset;

#pragma unroll
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            const int slot = kvw.reserve(elected);
            if (elected) {
                barrier_arrive_expect_tx(&layout.barriers->kv.produced[slot], nbatch_fa*chunk_ne*sizeof(half));
                tma_load_3d(layout.kv(slot), map_k, &layout.barriers->kv.produced[slot],
                    chunk*chunk_ne, z_KV, physical_token);
            }
            kvw.advance();
        }

#pragma unroll
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            const int slot = kvw.reserve(elected);
            if (elected) {
                barrier_arrive_expect_tx(&layout.barriers->kv.produced[slot], nbatch_fa*chunk_ne*sizeof(half));
                tma_load_3d(layout.kv(slot), map_v, &layout.barriers->kv.produced[slot],
                    chunk*chunk_ne, z_KV, physical_token);
            }
            kvw.advance();
        }
    }
}

template<int DKQ, int ncols1, int ncols2, bool use_logit_softcap, int split_k>
static __device__ __forceinline__ void consumer(
        const float * sinks_f,
        half * dstk,
        float * dst_parts,
        float2 * dst_meta,
        const float logit_softcap,
        const int ne01,
        const int ne02,
        const int gqa_ratio,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop,
        const int q_base,
        const int total_k,
        const bool causal,
        shared_layout<DKQ, ncols1> layout) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int DV             = DKQ;
    constexpr int ncols          = ncols1*ncols2;
    constexpr int nbatch_fa      = config<DKQ>::nbatch_fa;
    constexpr int nbatch_combine = config<DKQ>::nbatch_combine;
    using T_A_KQ  = typename mma_tile_sizes<DV, ncols>::T_A_KQ;
    using T_B_KQ  = typename mma_tile_sizes<DV, ncols>::T_B_KQ;
    using T_C_KQ  = typename mma_tile_sizes<DV, ncols>::T_C_KQ;
    using T_A_VKQ = typename mma_tile_sizes<DV, ncols>::T_A_VKQ;
    using T_B_VKQ = typename mma_tile_sizes<DV, ncols>::T_B_VKQ;
    using T_C_VKQ = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;
    constexpr int cols_per_warp   = T_B_KQ::I;
    constexpr int cols_per_thread = get_cols_per_thread();
    constexpr int np              = nwarps*cols_per_warp/ncols;
    constexpr bool q_in_reg       = DKQ <= 256;
    static_assert(ncols == 64 && cols_per_warp == 16 && np == 1, "bad SM120 attention shape");

    T_B_KQ Q_B[q_in_reg ? DKQ/(2*T_B_KQ::J) : 1];
    T_C_VKQ VKQ_C[DV/T_C_VKQ::J];
    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    barrier_wait(&layout.barriers->q_ready, 0);
    half2 * tile_q = layout.q();
    const int j0 = threadIdx.y*cols_per_warp;
    if constexpr (q_in_reg) {
#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_q + j0*(DKQ/2) + k0, DKQ/2);
        }
    }

    reader kvr{&layout.barriers->kv};

    for (int kb0 = 0; kb0 < kb0_stop; ++kb0) {
        T_C_KQ KQ_C[nbatch_fa/T_C_KQ::J];

#pragma unroll
        for (int chunk = 0; chunk < DKQ/chunk_ne; ++chunk) {
            const int slot = kvr.wait();
            half2 * tile_k = layout.kv(slot);
            T_B_KQ Q_fragment;

#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += T_A_KQ::I) {
#pragma unroll
                for (int k0 = 0; k0 < chunk_h2; k0 += T_A_KQ::J) {
                    if constexpr (q_in_reg) {
                        Q_fragment = Q_B[(chunk*chunk_h2 + k0)/T_B_KQ::J];
                    } else {
                        load_ldmatrix_swizzle_128(Q_fragment, tile_q, j0, chunk*chunk_h2 + k0, DKQ/2);
                    }
                    T_A_KQ K_A;
                    load_ldmatrix_swizzle_128(K_A, tile_k, i00, k0, chunk_h2);
                    mma(KQ_C[i00/T_A_KQ::I], Q_fragment, K_A);
                }
            }
            kvr.pop(slot);
        }

        if constexpr (use_logit_softcap) {
#pragma unroll
            for (int i = 0; i < nbatch_fa/T_C_KQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
                }
            }
        }

        float KQ_max_new[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max_new[col] = KQ_max[col];
        }
        float KQ_rowsum_add[cols_per_thread] = {0.0f};

        const int kv_tile_start = (kb0_start + kb0)*nbatch_fa;
#pragma unroll
        for (int i00 = 0; i00 < nbatch_fa; i00 += T_C_KQ::J) {
#pragma unroll
            for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                const int j = (threadIdx.y*cols_per_warp + T_C_KQ::get_i(l0))/ncols2;
                const int q_idx = jt*ncols1 + j;
                const int kv0 = kv_tile_start + i00 + T_C_KQ::get_j(l0);
                const int kv1 = kv0 + 1;
                const int q_abs = q_base + q_idx;
                const bool q_valid = q_idx < ne01;
                if (!q_valid || kv0 >= total_k || (causal && kv0 > q_abs)) {
                    KQ_C[i00/T_C_KQ::J].x[l0 + 0] = -INFINITY;
                }
                if (!q_valid || kv1 >= total_k || (causal && kv1 > q_abs)) {
                    KQ_C[i00/T_C_KQ::J].x[l0 + 1] = -INFINITY;
                }
            }
        }

        if constexpr (DKQ == 256) {
            constexpr int nbatch_phase = 32;
            static_assert(nbatch_fa == 2*nbatch_phase, "bad D256 logical attention tile");
            float KQ_max_scale_phase_1[cols_per_thread];

#pragma unroll
            for (int phase = 0; phase < 2; ++phase) {
#pragma unroll
                for (int col = 0; col < cols_per_thread; ++col) {
                    KQ_max_new[col] = KQ_max[col];
                    KQ_rowsum_add[col] = 0.0f;
                }
#pragma unroll
                for (int k0 = phase*nbatch_phase; k0 < (phase + 1)*nbatch_phase; k0 += T_C_KQ::J) {
#pragma unroll
                    for (int l = 0; l < T_C_KQ::ne; ++l) {
                        const int KQ_idx = (l/2) % 2;
                        KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/T_C_KQ::J].x[l] + FATTN_KQ_MAX_OFFSET);
                    }
                }
#pragma unroll
                for (int col = 0; col < cols_per_thread; ++col) {
                    KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 2));
                    KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 1));
                }
#pragma unroll
                for (int k0 = phase*nbatch_phase; k0 < (phase + 1)*nbatch_phase; k0 += T_C_KQ::J) {
#pragma unroll
                    for (int l = 0; l < T_C_KQ::ne; ++l) {
                        const int KQ_idx = (l/2) % 2;
                        KQ_C[k0/T_C_KQ::J].x[l] = expf(KQ_C[k0/T_C_KQ::J].x[l] - KQ_max_new[KQ_idx]);
                        KQ_rowsum_add[KQ_idx] += KQ_C[k0/T_C_KQ::J].x[l];
                    }
                }

                float KQ_max_scale[cols_per_thread];
#pragma unroll
                for (int col = 0; col < cols_per_thread; ++col) {
                    const float diff = KQ_max[col] - KQ_max_new[col];
                    KQ_max_scale[col] = softmax_rescale(diff);
                    KQ_max[col] = KQ_max_new[col];
                    KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
                }
                if (phase == 0) {
#pragma unroll
                    for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                        for (int l = 0; l < T_C_VKQ::ne; ++l) {
                            VKQ_C[i].x[l] *= KQ_max_scale[(l/2) % cols_per_thread];
                        }
                    }
                } else {
#pragma unroll
                    for (int col = 0; col < cols_per_thread; ++col) {
                        KQ_max_scale_phase_1[col] = KQ_max_scale[col];
                    }
                }
            }

            T_B_VKQ B[nbatch_fa/(2*T_B_VKQ::J)];
#pragma unroll
            for (int k = 0; k < nbatch_fa/(2*T_B_VKQ::J); ++k) {
                B[k] = get_half2(KQ_C[k]);
            }

#pragma unroll
            for (int chunk = 0; chunk < DV/chunk_ne; ++chunk) {
                const int slot = kvr.wait();
                half2 * tile_v = layout.kv(slot);
                const int i0_start = chunk*chunk_ne;
#pragma unroll
                for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_start + chunk_ne; i_VKQ_0 += T_A_VKQ::I) {
#pragma unroll
                    for (int k00 = 0; k00 < nbatch_phase/2; k00 += T_A_VKQ::J) {
                        T_A_VKQ A;
                        load_ldmatrix_trans_swizzle_128(A, tile_v, 2*k00, (i_VKQ_0 - i0_start)/2, chunk_h2);
                        mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/T_A_VKQ::J], A);
                    }
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        VKQ_C[i_VKQ_0/T_A_VKQ::I].x[l] *= KQ_max_scale_phase_1[(l/2) % cols_per_thread];
                    }
#pragma unroll
                    for (int k00 = nbatch_phase/2; k00 < nbatch_fa/2; k00 += T_A_VKQ::J) {
                        T_A_VKQ A;
                        load_ldmatrix_trans_swizzle_128(A, tile_v, 2*k00, (i_VKQ_0 - i0_start)/2, chunk_h2);
                        mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/T_A_VKQ::J], A);
                    }
                }
                kvr.pop(slot);
            }
        } else {
#pragma unroll
            for (int k0 = 0; k0 < nbatch_fa; k0 += T_C_KQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int KQ_idx = (l/2) % 2;
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/T_C_KQ::J].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 2));
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 1));
            }
#pragma unroll
            for (int k0 = 0; k0 < nbatch_fa; k0 += T_C_KQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int KQ_idx = (l/2) % 2;
                    KQ_C[k0/T_C_KQ::J].x[l] = expf(KQ_C[k0/T_C_KQ::J].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/T_C_KQ::J].x[l];
                }
            }

            float KQ_max_scale[cols_per_thread];
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const float diff = KQ_max[col] - KQ_max_new[col];
                KQ_max_scale[col] = softmax_rescale(diff);
                KQ_max[col] = KQ_max_new[col];
                KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
            }
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[(l/2) % cols_per_thread];
                }
            }

            T_B_VKQ B[nbatch_fa/(2*T_B_VKQ::J)];
#pragma unroll
            for (int k = 0; k < nbatch_fa/(2*T_B_VKQ::J); ++k) {
                B[k] = get_half2(KQ_C[k]);
            }

#pragma unroll
            for (int chunk = 0; chunk < DV/chunk_ne; ++chunk) {
                const int slot = kvr.wait();
                half2 * tile_v = layout.kv(slot);
                const int i0_start = chunk*chunk_ne;
#pragma unroll
                for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_start + chunk_ne; i_VKQ_0 += T_A_VKQ::I) {
#pragma unroll
                    for (int k00 = 0; k00 < nbatch_fa/2; k00 += T_A_VKQ::J) {
                        T_A_VKQ A;
                        load_ldmatrix_trans_swizzle_128(A, tile_v, 2*k00, (i_VKQ_0 - i0_start)/2, chunk_h2);
                        mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/T_A_VKQ::J], A);
                    }
                }
                kvr.pop(slot);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_rowsum[col] += __shfl_xor_sync(0xffffffffu, KQ_rowsum[col], 2);
        KQ_rowsum[col] += __shfl_xor_sync(0xffffffffu, KQ_rowsum[col], 1);
    }

    if (sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = threadIdx.y*cols_per_warp + T_C_KQ::get_i(2*col);
            const float sink = sinks_f[jc % ncols2];
            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = softmax_rescale(diff);
            KQ_max[col] = KQ_max_new;
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + expf(sink - KQ_max_new);
        }
#pragma unroll
        for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale[(l/2) % cols_per_thread];
            }
        }
    }

    // Keep the numerator in FP32 through normalization or split-K combination.
    float * tile_out = reinterpret_cast<float *>(layout.data);
    constexpr int tile_stride = nbatch_combine + 4;

    // Finish all pipeline reads before its shared memory is reused.
    consumer_sync();
    const int jc_meta = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(2*(threadIdx.x % cols_per_thread));
    const float2 meta = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
    if (threadIdx.x % 4 < cols_per_thread) {
        reinterpret_cast<float2 *>(tile_out)[jc_meta*(tile_stride/2) + nbatch_combine/2] = meta;
    }
    consumer_sync();

#pragma unroll
    for (int k00 = 0; k00 < DV; k00 += nbatch_combine) {
        const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
        for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int j = j0 + T_C_VKQ::get_i(l);
                const int k = k1 + T_C_VKQ::get_j(l);
                tile_out[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
            }
        }
        consumer_sync();

#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4, WARP_SIZE/8}) {
            const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
            const int k0_stop   =                             nbatch_combine - nbatch_combine % stride_k;
            const int stride_jc = WARP_SIZE/stride_k;
            if (k0_start == k0_stop) {
                continue;
            }
#pragma unroll
            for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
                const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x/stride_k);
                if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                    break;
                }
                const int j = jc/ncols2;
                const int c = jc - j*ncols2;
                if ((ncols1 > 1 && jt*ncols1 + j >= ne01) || (ncols2 > 1 && zt_gqa*ncols2 + c >= gqa_ratio)) {
                    continue;
                }
                const float2 meta = reinterpret_cast<const float2 *>(tile_out)[jc*(tile_stride/2) + nbatch_combine/2];
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);
                    if constexpr (split_k == 1) {
                        const float value = tile_out[jc*tile_stride + k];
                        dstk[((jt*ncols1 + j)*ne02 + c)*DV + k00 + k] =
                            __float2half_rn(meta.y == 0.0f ? 0.0f : value/meta.y);
                    } else {
                        dst_parts[(j*ne02 + c)*split_k*DV + k00 + k] = tile_out[jc*tile_stride + k];
                        if (k00 == 0 && k == 0) {
                            dst_meta[(j*ne02 + c)*split_k] = meta;
                        }
                    }
                }
            }
        }
        consumer_sync();
    }

#else
    EXL3_UNUSED_VARS(sinks_f, dstk, dst_parts, dst_meta, logit_softcap, ne01, ne02, gqa_ratio, jt, zt_gqa,
        kb0_start, kb0_stop, q_base, total_k, causal, layout);
    EXL3_NO_DEVICE_CODE;
#endif
}

template<int DKQ, int ncols1, int ncols2, bool use_logit_softcap, int split_k>
__global__ __launch_bounds__((nwarps + 1)*WARP_SIZE, 1) void kernel(
        __grid_constant__ const CUtensorMap map_k,
        __grid_constant__ const CUtensorMap map_v,
        const half * q_ptr,
        const int32_t * block_table,
        const int32_t * cache_seqlens,
        half * dst_ptr,
        float * dst_parts_ptr,
        float2 * dst_meta_ptr,
        const float scale,
        const float logit_softcap,
        const int32_t q_len,
        const int32_t n_q_heads,
        const int32_t n_kv_heads,
        const int32_t num_pages_per_seq,
        const int32_t kv_append_len,
        const bool causal) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
    extern __shared__ __align__(1024) uint8_t smem[];
    shared_layout<DKQ, ncols1> layout(smem);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        barrier_init(&layout.barriers->q_ready, WARP_SIZE);
#pragma unroll
        for (int i = 0; i < depth; ++i) {
            barrier_init(&layout.barriers->kv.produced[i], 1);
            barrier_init(&layout.barriers->kv.consumed[i], nwarps*WARP_SIZE);
        }
    }
    __syncthreads();

    constexpr int nbatch_fa = config<DKQ>::nbatch_fa;
    const int jt = blockIdx.x;
    const int split = blockIdx.y % split_k;
    const int tile_y = blockIdx.y / split_k;
    const int gqa_ratio = n_q_heads/n_kv_heads;
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1)/ncols2;
    const int z_KV = tile_y/ntiles_z_gqa;
    const int zt_gqa = tile_y - z_KV*ntiles_z_gqa;
    const int sequence = blockIdx.z;
    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2;
    const int q_base = cache_seqlens[sequence];
    const int total_k = q_base + kv_append_len;
    const int q_tile_stop = min((jt + 1)*ncols1, q_len);
    const int tile_total_k = causal ? q_base + q_tile_stop : total_k;
    const int kb0_total = (tile_total_k + nbatch_fa - 1)/nbatch_fa;
    const int kb0_start = int64_t(kb0_total)*split/split_k;
    const int kb0_stop = int64_t(kb0_total)*(split + 1)/split_k;

    const half2 * q_h2 = reinterpret_cast<const half2 *>(q_ptr) +
        (int64_t(sequence)*q_len*n_q_heads + zt_Q)*(DKQ/2);
    half * dstk = dst_ptr + (int64_t(sequence)*q_len*n_q_heads + zt_Q)*DKQ;
    float * dst_parts = nullptr;
    float2 * dst_meta = nullptr;
    if constexpr (split_k > 1) {
        const int64_t row0 = (int64_t(sequence)*q_len + jt*ncols1)*n_q_heads + zt_Q;
        dst_parts = dst_parts_ptr + (row0*split_k + split)*DKQ;
        dst_meta = dst_meta_ptr + row0*split_k + split;
    }

    if (threadIdx.y == nwarps) {
        producer<DKQ, ncols1, ncols2>(q_h2, scale, n_q_heads*(DKQ/2), DKQ/2,
            jt, zt_gqa, gqa_ratio, q_len, sequence, z_KV, kb0_start, kb0_stop,
            block_table, num_pages_per_seq, &map_k, &map_v, layout);
    } else {
        consumer<DKQ, ncols1, ncols2, use_logit_softcap, split_k>(nullptr, dstk, dst_parts, dst_meta,
            logit_softcap, q_len, n_q_heads, gqa_ratio, jt, zt_gqa, kb0_start,
            kb0_stop - kb0_start, q_base, total_k, causal, layout);
    }
#else
    EXL3_UNUSED_VARS(map_k, map_v, q_ptr, block_table, cache_seqlens, dst_ptr, dst_parts_ptr,
        dst_meta_ptr, scale, logit_softcap, q_len, n_q_heads, n_kv_heads, num_pages_per_seq,
        kv_append_len, causal);
#endif
}

template<int D>
__launch_bounds__(D, 1)
static __global__ void combine_results(
        const float * parts_ptr,
        const float2 * meta_ptr,
        half * dst_ptr) {
    constexpr int split_k = 3;
    const int row = (blockIdx.z*gridDim.x + blockIdx.x)*gridDim.y + blockIdx.y;
    const float * parts = parts_ptr + row*split_k*D;
    const float2 * meta_src = meta_ptr + row*split_k;
    half * dst = dst_ptr + row*D;
    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    if (tid < split_k) {
        meta[tid] = meta_src[tid];
    }
    __syncthreads();

    float kqmax = meta[0].x;
#pragma unroll
    for (int split = 1; split < split_k; ++split) {
        kqmax = fmaxf(kqmax, meta[split].x);
    }

    float numerator = 0.0f;
    float denominator = 0.0f;
#pragma unroll
    for (int split = 0; split < split_k; ++split) {
        const float scale = expf(meta[split].x - kqmax);
        numerator += scale*parts[split*D + tid];
        denominator += scale*meta[split].y;
    }
    dst[tid] = __float2half_rn(denominator == 0.0f ? 0.0f : numerator/denominator);
}

#undef TURING_MMA_AVAILABLE
#undef EXL3_UNUSED_VARS
#undef EXL3_NO_DEVICE_CODE

} // namespace exl3_sm120
