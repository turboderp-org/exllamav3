#pragma once

// Templated trellis-quantization (Viterbi) kernel, instantiated per (K, cb) in
// comp_units/quantize_tiles_inst_k*.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"

#define QUANTIZE_TILES_NUM_THREADS 512
#ifndef H_INF
#define H_INF __ushort_as_half(0x7c00)
#endif

template <int K, int cb>
__global__ __launch_bounds__(QUANTIZE_TILES_NUM_THREADS, 2)
void quantize_tiles_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    half* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr
)
{
    extern __shared__ uint8_t shbuf[];
    uint8_t* sh = shbuf;

    constexpr int Kr = 16 - K;
    constexpr int max_q = 1 << K;
    constexpr int edges = 65536 >> K;

    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + 256 * tile_idx;
    float* output_tile = output_tiles_ptr + 256 * tile_idx;
    uint16_t* output_indices = output_indices_ptr + 256 * tile_idx;
    uint16_t* temp_edges = temp_edges_ptr + 256 * edges * tile_idx;

    half* sh_input_tile = (half*) sh; sh += 256 * sizeof(half);
    half* sh_min = (half*) sh; sh += 32 * sizeof(half);
    int* sh_idx = (int*) sh; sh += 32 * sizeof(int);

    half* sh_temp_costs = (half*) sh;
    half* temp_costs = K >= 2 ? sh_temp_costs : temp_costs_ptr + 2 * edges * tile_idx;
    half* temp_costs_inc = temp_costs + edges;

    if (thread < 256) sh_input_tile[thread] = __float2half_rn(input_tile[thread]);
    __syncthreads();

    auto forward = [&](int roll, int pre_state)
    {
        int ri = roll & 255;
        half* t = temp_costs;
        temp_costs = temp_costs_inc;
        temp_costs_inc = t;

        for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * QUANTIZE_TILES_NUM_THREADS)
        {
            const half2 w2 = __half2half2(sh_input_tile[ri]);
            int in_edge_idx = out_edge_idx >> K;
            uint32_t product0 = 0;
            uint32_t product1 = 0;
            half2 decoded2;
            if constexpr (cb == 1)
            {
                product0 = mul_const_u32<0xCBAC1FEDu>(out_edge_idx);
                product1 = product0 + 0xCBAC1FEDu;
                decoded2 = decode_mcg_product_2(product0, product1);
            }
            else if constexpr (cb == 2)
            {
                product0 = out_edge_idx * 0x83DCD12Du;
                product1 = product0 + 0x83DCD12Du;
                decoded2 = decode_mul1_product_2(product0, product1);
            }
            else
            {
                decoded2 = decode_3inst_2<cb>(out_edge_idx, out_edge_idx + 1);
            }
            half2 dh2 = __hsub2(decoded2, w2);
            half2 min_err2 = __hmul2(dh2, dh2);
            if (pre_state >= 0 && in_edge_idx != pre_state) min_err2 = __half2half2(H_INF);
            int min_in_edge0 = in_edge_idx;
            int min_in_edge1 = in_edge_idx;

            #pragma unroll
            for (int k = 1; k < max_q; ++k)
            {
                const int state0 = (k << Kr) | out_edge_idx;
                in_edge_idx = state0 >> K;
                if constexpr (cb == 1)
                {
                    // MCG multiplication is linear modulo 2^32 across successive branch states.
                    constexpr uint32_t product_step = 0xCBAC1FEDu << Kr;
                    product0 += product_step;
                    product1 += product_step;
                    decoded2 = decode_mcg_product_2(product0, product1);
                }
                else if constexpr (cb == 2)
                {
                    // The mul1 multiplication is equally linear modulo 2^32.
                    constexpr uint32_t product_step = 0x83DCD12Du << Kr;
                    product0 += product_step;
                    product1 += product_step;
                    decoded2 = decode_mul1_product_2(product0, product1);
                }
                else
                {
                    decoded2 = decode_3inst_2<cb>(state0, state0 + 1);
                }
                dh2 = __hsub2(decoded2, w2);
                half2 err2 = __hmul2(dh2, dh2);
                if (pre_state >= 0 && in_edge_idx != pre_state) err2 = __half2half2(H_INF);
                if (__hlt(__low2half(err2), __low2half(min_err2)))
                {
                    min_err2 = __halves2half2(__low2half(err2), __high2half(min_err2));
                    min_in_edge0 = in_edge_idx;
                }
                if (__hlt(__high2half(err2), __high2half(min_err2)))
                {
                    min_err2 = __halves2half2(__low2half(min_err2), __high2half(err2));
                    min_in_edge1 = in_edge_idx;
                }
            }

            reinterpret_cast<half2*>(temp_costs)[out_edge_idx >> 1] = min_err2;
            temp_edges[edges * ri + out_edge_idx] = (uint16_t) min_in_edge0;
            temp_edges[edges * ri + out_edge_idx + 1] = (uint16_t) min_in_edge1;
        }
        __syncthreads();

        for (int i = 1; i < 256; ++i)
        {
            ri = (i + roll) & 255;
            t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;

            for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * QUANTIZE_TILES_NUM_THREADS)
            {
                const half2 w2 = __half2half2(sh_input_tile[ri]);
                int in_edge_idx = out_edge_idx >> K;
                uint32_t product0 = 0;
                uint32_t product1 = 0;
                half2 decoded2;
                if constexpr (cb == 1)
                {
                    product0 = mul_const_u32<0xCBAC1FEDu>(out_edge_idx);
                    product1 = product0 + 0xCBAC1FEDu;
                    decoded2 = decode_mcg_product_2(product0, product1);
                }
                else if constexpr (cb == 2)
                {
                    product0 = out_edge_idx * 0x83DCD12Du;
                    product1 = product0 + 0x83DCD12Du;
                    decoded2 = decode_mul1_product_2(product0, product1);
                }
                else
                {
                    decoded2 = decode_3inst_2<cb>(out_edge_idx, out_edge_idx + 1);
                }
                half2 dh2 = __hsub2(decoded2, w2);
                half2 min_err2 = __hfma2(dh2, dh2, __half2half2(temp_costs_inc[in_edge_idx]));
                int min_in_edge0 = in_edge_idx;
                int min_in_edge1 = in_edge_idx;

                #pragma unroll
                for (int k = 1; k < max_q; ++k)
                {
                    const int state0 = (k << Kr) | out_edge_idx;
                    in_edge_idx = state0 >> K;
                    if constexpr (cb == 1)
                    {
                        // MCG multiplication is linear modulo 2^32 across successive branch states.
                        constexpr uint32_t product_step = 0xCBAC1FEDu << Kr;
                        product0 += product_step;
                        product1 += product_step;
                        decoded2 = decode_mcg_product_2(product0, product1);
                    }
                    else if constexpr (cb == 2)
                    {
                        // The mul1 multiplication is equally linear modulo 2^32.
                        constexpr uint32_t product_step = 0x83DCD12Du << Kr;
                        product0 += product_step;
                        product1 += product_step;
                        decoded2 = decode_mul1_product_2(product0, product1);
                    }
                    else
                    {
                        decoded2 = decode_3inst_2<cb>(state0, state0 + 1);
                    }
                    dh2 = __hsub2(decoded2, w2);
                    half2 err2 = __hfma2(dh2, dh2, __half2half2(temp_costs_inc[in_edge_idx]));
                    if (__hlt(__low2half(err2), __low2half(min_err2)))
                    {
                        min_err2 = __halves2half2(__low2half(err2), __high2half(min_err2));
                        min_in_edge0 = in_edge_idx;
                    }
                    if (__hlt(__high2half(err2), __high2half(min_err2)))
                    {
                        min_err2 = __halves2half2(__low2half(min_err2), __high2half(err2));
                        min_in_edge1 = in_edge_idx;
                    }
                }

                reinterpret_cast<half2*>(temp_costs)[out_edge_idx >> 1] = min_err2;
                temp_edges[edges * ri + out_edge_idx] = (uint16_t) min_in_edge0;
                temp_edges[edges * ri + out_edge_idx + 1] = (uint16_t) min_in_edge1;
            }
            __syncthreads();
        }
    };

    auto argmin_cost = [&]()
    {
        // Preserve the historical 1024-thread tie-breaking order.
        half local_min0 = H_INF;
        half local_min1 = H_INF;
        int local_idx0 = -1;
        int local_idx1 = -1;
        for (int e = thread; e < edges; e += 2 * QUANTIZE_TILES_NUM_THREADS)
        {
            half v = temp_costs_inc[e];
            if (__hlt(v, local_min0)) { local_min0 = v; local_idx0 = e; }
        }
        for (int e = thread + QUANTIZE_TILES_NUM_THREADS; e < edges; e += 2 * QUANTIZE_TILES_NUM_THREADS)
        {
            half v = temp_costs_inc[e];
            if (__hlt(v, local_min1)) { local_min1 = v; local_idx1 = e; }
        }

        const int lane_id = thread & 31;
        const int warp_id = thread >> 5;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            half other_min0 = __shfl_down_sync(0xffffffff, local_min0, offset);
            int other_idx0 = __shfl_down_sync(0xffffffff, local_idx0, offset);
            if (__hlt(other_min0, local_min0)) { local_min0 = other_min0; local_idx0 = other_idx0; }
            half other_min1 = __shfl_down_sync(0xffffffff, local_min1, offset);
            int other_idx1 = __shfl_down_sync(0xffffffff, local_idx1, offset);
            if (__hlt(other_min1, local_min1)) { local_min1 = other_min1; local_idx1 = other_idx1; }
        }
        sh_min[warp_id] = local_min0;
        sh_idx[warp_id] = local_idx0;
        sh_min[16 + warp_id] = local_min1;
        sh_idx[16 + warp_id] = local_idx1;
        __syncthreads();

        int local_idx = 0;
        if (warp_id == 0)
        {
            half local_min = sh_min[lane_id];
            local_idx = sh_idx[lane_id];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                half other_min = __shfl_down_sync(0xffffffff, local_min, offset);
                int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
                if (__hlt(other_min, local_min)) { local_min = other_min; local_idx = other_idx; }
            }
        }
        return local_idx;
    };

    auto backward = [&](int roll, bool write, int edge)
    {
        if (thread == 0)
        {
            for (int i = 255; i >= 0; --i)
            {
                const int ri = (i + roll) & 255;
                const int prev_edge = (int) temp_edges[edges * ri + edge];
                const int encoded = (prev_edge << K) | edge;
                edge = prev_edge;
                if (write)
                {
                    output_indices[ri] = (uint16_t) encoded;
                    output_tile[ri] = __half2float(decode_3inst<cb>(encoded));
                }
                else if (ri == 0) break;
            }
        }
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    forward(128, -1);
    int end_state = backward(128, false, argmin_cost());
    forward(0, end_state);
    backward(0, true, end_state);
}
