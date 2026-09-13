#pragma once

// Dense specializations tuned on sm_120
#include "quantize_tiles_kernel.cuh"

// Block geometry, tuned on sm_120: one resident block per SM
__host__ __device__ constexpr int qt_optimized_threads(int K, int cb)
{
    return K == 5 || K == 6 || (K == 7 && cb != 0) ? 512 : 1024;
}

template <int K, int cb, int L = 256>
__global__ __launch_bounds__(qt_optimized_threads(K, cb), 1) void quantize_tiles_optimized_kernel(
    const float* __restrict__ input_tiles_ptr, float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr, half* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr, const half2* __restrict__ lut)
{
    static_assert(K >= 1 && K <= 8 && L % 2 == 0);
    constexpr int NT = qt_optimized_threads(K, cb);
    constexpr int NW = NT / 32;
    constexpr int Kr = 16 - K;
    constexpr int max_q = 1 << K;
    constexpr int edges = 65536 >> K;
    constexpr int history_bytes = L * edges / (K == 1 ? 8 : 1);
    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    uint8_t* temp_edges = reinterpret_cast<uint8_t*>(temp_edges_ptr) + (int64_t)history_bytes * tile_idx;

    extern __shared__ uint8_t shbuf[];
    half* sh_input_tile = reinterpret_cast<half*>(shbuf);
    // Retain the measured shared-memory layout, including the unused half-minimum area.
    int* sh_idx = reinterpret_cast<int*>(shbuf + L * sizeof(half) + 64);
    half* temp_costs = reinterpret_cast<half*>(shbuf + L * sizeof(half) + 64 + 128);
    // K=1 stages the next costs in registers, then overwrites this single shared buffer.
    half* temp_costs_inc = temp_costs + (K == 1 ? 0 : edges);
    for (int i = thread; i < L; i += NT)
        sh_input_tile[i] = __float2half_rn(input_tile[i]);
    __syncthreads();
    auto ring = [&](int i, int roll)
    {
        int ri = i + roll;
        if (ri >= L)
            ri -= L;
        return ri;
    };

    auto argmin_cost = [&]()
    {
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges; e += NT)
        {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            unsigned key = ((uint32_t)__half_as_ushort(temp_costs[e]) << 16) | rank;
            best = min(best, key);
        }
        best = qt_warp_min(best);
        if ((thread & 31) == 0)
            ((uint32_t*)sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32)
        {
            best = thread < NW ? ((uint32_t*)sh_idx)[thread] : 0x7c00ffffu;
            best = qt_warp_min(best);
        }
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        return best >= 0x7c000000u ? 0 : (int)(((rank & 31) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge)
    {
        if (thread == 0)
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                int branch;
                if constexpr (K == 1)
                {
                    const auto* bits = reinterpret_cast<const uint32_t*>(temp_edges);
                    int offset = ri * (edges / 32) + (edge / 64) * 2 + (edge & 1);
                    branch = (bits[offset] >> ((edge / 2) & 31)) & 1;
                }
                else
                    branch = temp_edges[edges * ri + edge];
                // Keep traceback in bounds even if all candidate costs were NaN and a packed
                // minimum retained its sentinel predecessor. Finite-input choices are unchanged.
                const int prev_edge = ((branch & (max_q - 1)) << (16 - 2 * K)) | (edge >> K);
                const int encoded = (prev_edge << K) | edge;
                edge = prev_edge;
                if (write)
                {
                    output_indices[ri] = (uint16_t)encoded;
                    output_tile[ri] = __half2float(decode_3inst<cb>(encoded));
                }
                else if (ri == 0)
                    break;
            }
        }
        if (thread == 0)
            sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    auto solve = [&](auto& forward)
    {
        forward(L / 2, -1);
        int end_state = backward(L / 2, false, argmin_cost());
        forward(0, end_state);
        backward(0, true, end_state);
    };

    // K=1: procedural decode, register-staged costs, and one traceback bit per state.
    if constexpr (K == 1)
    {
        // One lane handles each output pair.
        constexpr int G = 1;
        constexpr int NP = edges * G / (2 * NT);
        constexpr int NC = max_q / G;
        const int GL = threadIdx.x % G;
        half2 next_cost[NP];
        auto forward = [&](int roll, int pre_state)
        {
            for (int i = 0; i < L; ++i)
            {
                int ri = ring(i, roll);
                half* t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;
                half2 w = __half2half2(sh_input_tile[ri]);

                #pragma unroll
                for (int j = 0; j < NP; ++j)
                {
                    int out = 2 * (thread / G + j * (NT / G));
                    half2 best = __half2half2(H_INF);
                    int b0 = 0x7fffffff, b1 = 0x7fffffff;

                    #pragma unroll
                    for (int c = 0; c < NC; ++c)
                    {
                        int state = ((c * G + GL) << Kr) | out;
                        int pred = state >> K;
                        half2 d = __hsub2(decode_3inst_2<cb>(state, state + 1), w);
                        half2 err;
                        if (i == 0)
                        {
                            err = __hmul2(d, d);
                            if (pre_state >= 0 && pred != pre_state)
                                err = __half2half2(H_INF);
                        }
                        else
                            err = __hfma2(d, d, __half2half2(temp_costs_inc[pred]));
                        if (c == 0 || __hlt(__low2half(err), __low2half(best)))
                        {
                            best = __halves2half2(__low2half(err), __high2half(best));
                            b0 = pred;
                        }
                        if (c == 0 || __hlt(__high2half(err), __high2half(best)))
                        {
                            best = __halves2half2(__low2half(best), __high2half(err));
                            b1 = pred;
                        }
                    }

                    if (GL == 0)
                    {
                        next_cost[j] = best;
                        if (pre_state >= 0 || ri < L / 2)
                        {
                            #pragma unroll
                            for (int bit = 0; bit < K; ++bit)
                            {
                                unsigned a = __ballot_sync(0xffffffff, (b0 >> (16 - 2 * K + bit)) & 1);
                                unsigned b = __ballot_sync(0xffffffff, (b1 >> (16 - 2 * K + bit)) & 1);
                                if ((thread & 31) == 0)
                                {
                                    int offset = (ri * K + bit) * (edges / 32) + (out / 64) * 2;
                                    reinterpret_cast<uint32_t*>(temp_edges)[offset] = a;
                                    reinterpret_cast<uint32_t*>(temp_edges)[offset + 1] = b;
                                }
                            }
                        }
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int j = 0; j < NP; ++j)
                {
                    int out = 2 * (thread / G + j * (NT / G));
                    if (GL == 0)
                        reinterpret_cast<half2*>(temp_costs)[out / 2] = next_cost[j];
                }
                __syncthreads();
            }
        };
        solve(forward);
    }
    // K=2/3: decoded values persist in registers across both passes.
    else if constexpr (K <= 3)
    {
        // One lane handles each output pair.
        constexpr int G = 1;
        constexpr int NP = edges * G / (2 * NT);
        constexpr int NC = max_q / G;
        const int GL = threadIdx.x % G;
        half2 values[NP][NC];
        #pragma unroll
        for (int j = 0; j < NP; ++j)
        {
            int out = 2 * (thread / G + j * (NT / G));
            #pragma unroll
            for (int c = 0; c < NC; ++c)
            {
                int state = ((c * G + GL) << Kr) | out;
                values[j][c] = decode_3inst_2<cb>(state, state + 1);
            }
        }
        auto forward = [&](int roll, int pre_state)
        {
            for (int i = 0; i < L; ++i)
            {
                int ri = ring(i, roll);
                half* t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;
                half2 w = __half2half2(sh_input_tile[ri]);
                #pragma unroll
                for (int j = 0; j < NP; ++j)
                {
                    int out = 2 * (thread / G + j * (NT / G));
                    half2 best = __half2half2(H_INF);
                    int b0 = 0x7fffffff, b1 = 0x7fffffff;
                    #pragma unroll
                    for (int c = 0; c < NC; ++c)
                    {
                        int state = ((c * G + GL) << Kr) | out;
                        int pred = state >> K;
                        half2 d = __hsub2(values[j][c], w);
                        half2 err;
                        if (i == 0)
                        {
                            err = __hmul2(d, d);
                            if (pre_state >= 0 && pred != pre_state)
                                err = __half2half2(H_INF);
                        }
                        else
                            err = __hfma2(d, d, __half2half2(temp_costs_inc[pred]));
                        if (c == 0 || __hlt(__low2half(err), __low2half(best)))
                        {
                            best = __halves2half2(__low2half(err), __high2half(best));
                            b0 = pred;
                        }
                        if (c == 0 || __hlt(__high2half(err), __high2half(best)))
                        {
                            best = __halves2half2(__low2half(best), __high2half(err));
                            b1 = pred;
                        }
                    }

                    if (GL == 0)
                    {
                        reinterpret_cast<half2*>(temp_costs)[out / 2] = best;
                        if (pre_state >= 0 || ri < L / 2)
                        {
                            ((uint16_t*)temp_edges)[(edges * ri + out) / 2] =
                                ((b0 >> (16 - 2 * K)) & 255) | (((b1 >> (16 - 2 * K)) & 255) << 8);
                        }
                    }
                }
                __syncthreads();
            }
        };
        solve(forward);
    }
    // K=4/5: integer cost/predecessor minima and paired byte traceback stores.
    else if constexpr (K <= 5)
    {
        // One lane handles each output pair.
        constexpr int G = 1;
        constexpr int NP = edges * G / (2 * NT);
        constexpr int NC = max_q / G;
        const int GL = threadIdx.x % G;
        half2 values[NP][NC];
        #pragma unroll
        for (int j = 0; j < NP; ++j)
        {
            int out = 2 * (thread / G + j * (NT / G));
            #pragma unroll
            for (int c = 0; c < NC; ++c)
            {
                int state = ((c * G + GL) << Kr) | out;
                values[j][c] = decode_3inst_2<cb>(state, state + 1);
            }
        }
        auto forward = [&](int roll, int pre_state)
        {
            for (int i = 0; i < L; ++i)
            {
                int ri = ring(i, roll);
                half* t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;
                half2 w = __half2half2(sh_input_tile[ri]);
                #pragma unroll
                for (int j = 0; j < NP; ++j)
                {
                    int out = 2 * (thread / G + j * (NT / G));
                    uint32_t best0 = 0x7c00ffffu, best1 = 0x7c00ffffu;
                    #pragma unroll
                    for (int c = 0; c < NC; ++c)
                    {
                        int state = ((c * G + GL) << Kr) | out;
                        int pred = state >> K;
                        half2 d = __hsub2(values[j][c], w);
                        half2 err;
                        if (i == 0)
                        {
                            err = __hmul2(d, d);
                            if (pre_state >= 0 && pred != pre_state)
                                err = __half2half2(H_INF);
                        }
                        else
                            err = __hfma2(d, d, __half2half2(temp_costs_inc[pred]));

                        best0 = min(best0, ((uint32_t)__half_as_ushort(__low2half(err)) << 16) | pred);
                        best1 = min(best1, ((uint32_t)__half_as_ushort(__high2half(err)) << 16) | pred);
                    }

                    half2 best = __halves2half2(__ushort_as_half(best0 >> 16), __ushort_as_half(best1 >> 16));
                    int b0 = best0 & 65535, b1 = best1 & 65535;
                    if (GL == 0)
                    {
                        reinterpret_cast<half2*>(temp_costs)[out / 2] = best;
                        if (pre_state >= 0 || ri < L / 2)
                        {
                            ((uint16_t*)temp_edges)[(edges * ri + out) / 2] =
                                ((b0 >> (16 - 2 * K)) & 255) | (((b1 >> (16 - 2 * K)) & 255) << 8);
                        }
                    }
                }
                __syncthreads();
            }
        };
        solve(forward);
    }
    // K=6: cached global codebook avoids procedural decoding and register-cache pressure.
    else if constexpr (K == 6)
    {
        auto forward = [&](int roll, int pre_state)
        {
            int ri = ring(0, roll);
            half* t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;

            for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * NT)
            {
                const half2 w2 = __half2half2(sh_input_tile[ri]);
                int in_edge_idx = out_edge_idx >> K;
                half2 decoded2;
                decoded2 = __ldg(lut + ((out_edge_idx) >> 1));
                half2 dh2 = __hsub2(decoded2, w2);
                half2 min_err2 = __hmul2(dh2, dh2);
                if (pre_state >= 0 && in_edge_idx != pre_state)
                    min_err2 = __half2half2(H_INF);
                int min_in_edge0 = in_edge_idx;
                int min_in_edge1 = in_edge_idx;
                uint32_t best0 = ((uint32_t)__half_as_ushort(__low2half(min_err2)) << 16) | in_edge_idx;
                uint32_t best1 = ((uint32_t)__half_as_ushort(__high2half(min_err2)) << 16) | in_edge_idx;

                #pragma unroll
                for (int k = 1; k < max_q; ++k)
                {
                    const int state0 = (k << Kr) | out_edge_idx;
                    in_edge_idx = state0 >> K;
                    decoded2 = __ldg(lut + ((state0) >> 1));
                    dh2 = __hsub2(decoded2, w2);
                    half2 err2 = __hmul2(dh2, dh2);
                    if (pre_state >= 0 && in_edge_idx != pre_state)
                        err2 = __half2half2(H_INF);
                    best0 = min(best0, ((uint32_t)__half_as_ushort(__low2half(err2)) << 16) | in_edge_idx);
                    best1 = min(best1, ((uint32_t)__half_as_ushort(__high2half(err2)) << 16) | in_edge_idx);
                }

                min_err2 = __halves2half2(__ushort_as_half(best0 >> 16), __ushort_as_half(best1 >> 16));
                min_in_edge0 = best0 & 65535;
                min_in_edge1 = best1 & 65535;
                reinterpret_cast<half2*>(temp_costs)[out_edge_idx >> 1] = min_err2;
                if (pre_state >= 0 || ri < L / 2)
                {
                    ((uint16_t*)temp_edges)[(edges * ri + out_edge_idx) / 2] =
                        ((min_in_edge0 >> (16 - 2 * K)) & 255) |
                        (((min_in_edge1 >> (16 - 2 * K)) & 255) << 8);
                }
            }
            __syncthreads();

            for (int i = 1; i < L; ++i)
            {
                ri = ring(i, roll);
                t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;

                for (int out_edge_idx = 2 * thread; out_edge_idx < edges; out_edge_idx += 2 * NT)
                {
                    const half2 w2 = __half2half2(sh_input_tile[ri]);
                    int in_edge_idx = out_edge_idx >> K;
                    half2 decoded2;
                    decoded2 = __ldg(lut + ((out_edge_idx) >> 1));
                    half2 dh2 = __hsub2(decoded2, w2);
                    half2 min_err2 = __hfma2(dh2, dh2, __half2half2(temp_costs_inc[in_edge_idx]));
                    int min_in_edge0 = in_edge_idx;
                    int min_in_edge1 = in_edge_idx;
                    uint32_t best0 = ((uint32_t)__half_as_ushort(__low2half(min_err2)) << 16) | in_edge_idx;
                    uint32_t best1 = ((uint32_t)__half_as_ushort(__high2half(min_err2)) << 16) | in_edge_idx;

                    #pragma unroll
                    for (int k = 1; k < max_q; ++k)
                    {
                        const int state0 = (k << Kr) | out_edge_idx;
                        in_edge_idx = state0 >> K;
                        decoded2 = __ldg(lut + ((state0) >> 1));
                        dh2 = __hsub2(decoded2, w2);
                        half2 err2 = __hfma2(dh2, dh2, __half2half2(temp_costs_inc[in_edge_idx]));
                        best0 =
                            min(best0, ((uint32_t)__half_as_ushort(__low2half(err2)) << 16) | in_edge_idx);
                        best1 =
                            min(best1, ((uint32_t)__half_as_ushort(__high2half(err2)) << 16) | in_edge_idx);
                    }

                    min_err2 = __halves2half2(__ushort_as_half(best0 >> 16), __ushort_as_half(best1 >> 16));
                    min_in_edge0 = best0 & 65535;
                    min_in_edge1 = best1 & 65535;
                    reinterpret_cast<half2*>(temp_costs)[out_edge_idx >> 1] = min_err2;
                    if (pre_state >= 0 || ri < L / 2)
                    {
                        ((uint16_t*)temp_edges)[(edges * ri + out_edge_idx) / 2] =
                            ((min_in_edge0 >> (16 - 2 * K)) & 255) |
                            (((min_in_edge1 >> (16 - 2 * K)) & 255) << 8);
                    }
                }
                __syncthreads();
            }
        };
        solve(forward);
    }
    // K=7/8: full-warp minima, with completed outputs distributed over lanes for stores.
    else
    {
        // G lanes cooperate on one output pair; branch values persist in registers.
        constexpr int G = 32;
        constexpr int NP = edges * G / (2 * NT);
        constexpr int NC = max_q / G;
        const int GL = threadIdx.x % G;
        half2 values[NP][NC];
        #pragma unroll
        for (int j = 0; j < NP; ++j)
        {
            int out = 2 * ((thread / 32) * NP + j);
            #pragma unroll
            for (int c = 0; c < NC; ++c)
            {
                int state = ((c * G + GL) << Kr) | out;
                values[j][c] = decode_3inst_2<cb>(state, state + 1);
            }
        }
        auto forward = [&](int roll, int pre_state)
        {
            for (int i = 0; i < L; ++i)
            {
                int ri = ring(i, roll);
                half* t = temp_costs;
                temp_costs = temp_costs_inc;
                temp_costs_inc = t;
                half2 w = __half2half2(sh_input_tile[ri]);
                uint32_t owned0[(NP + 31) / 32], owned1[(NP + 31) / 32];
                #pragma unroll
                for (int j = 0; j < NP; ++j)
                {
                    int out = 2 * ((thread / 32) * NP + j);
                    uint32_t best0 = 0x7c00ffffu, best1 = 0x7c00ffffu;
                    #pragma unroll
                    for (int c = 0; c < NC; ++c)
                    {
                        int state = ((c * G + GL) << Kr) | out;
                        int pred = state >> K;
                        half2 d = __hsub2(values[j][c], w);
                        half2 err;
                        if (i == 0)
                        {
                            err = __hmul2(d, d);
                            if (pre_state >= 0 && pred != pre_state)
                                err = __half2half2(H_INF);
                        }
                        else
                            err = __hfma2(d, d, __half2half2(temp_costs_inc[pred]));

                        best0 = c == 0
                                    ? (((uint32_t)__half_as_ushort(__low2half(err)) << 16) | pred)
                                    : min(best0, ((uint32_t)__half_as_ushort(__low2half(err)) << 16) | pred);
                        best1 = c == 0
                                    ? (((uint32_t)__half_as_ushort(__high2half(err)) << 16) | pred)
                                    : min(best1, ((uint32_t)__half_as_ushort(__high2half(err)) << 16) | pred);
                    }

                    best0 = qt_warp_min(best0);
                    best1 = qt_warp_min(best1);
                    if (GL == (j & 31))
                    {
                        owned0[j / 32] = best0;
                        owned1[j / 32] = best1;
                    }
                }
                #pragma unroll
                for (int z = 0; z < (NP + 31) / 32; ++z)
                {
                    int j = z * 32 + GL;
                    if (j < NP)
                    {
                        int out = 2 * ((thread / 32) * NP + j);
                        uint32_t v0 = owned0[z], v1 = owned1[z];
                        temp_costs[out] = __ushort_as_half(v0 >> 16);
                        temp_costs[out + 1] = __ushort_as_half(v1 >> 16);
                        if (pre_state >= 0 || ri < L / 2)
                            ((uint16_t*)temp_edges)[(edges * ri + out) / 2] =
                                (((v0 & 65535) >> (16 - 2 * K)) & 255) |
                                ((((v1 & 65535) >> (16 - 2 * K)) & 255) << 8);
                    }
                }
                __syncthreads();
            }
        };
        solve(forward);
    }
}
