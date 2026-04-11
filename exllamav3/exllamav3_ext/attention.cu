#include <cuda_fp16.h>
#include <cstdint>
#include "add.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include <algorithm>
#include "quant/exl3_devctx.cuh"

#define G_MAX 8
#define PAGE_SIZE 256

__device__ __forceinline__ uint64_t paged_cache_offset
(
    const int32_t* __restrict__ block_row,  // [num_pages_per_seq]
    int64_t logical_pos,
    int64_t kv_head,
    int64_t n_kv_heads,
    int64_t dim,
    int d
)
{
    const int64_t logical_page   = logical_pos / PAGE_SIZE;
    const int64_t offset_in_page = logical_pos % PAGE_SIZE;
    const int32_t physical_page  = block_row[logical_page];

    return
        ((((uint64_t)physical_page * (uint64_t)PAGE_SIZE +
           (uint64_t)offset_in_page) * (uint64_t)n_kv_heads +
          (uint64_t)kv_head) * (uint64_t)dim) + (uint64_t)d;
}

template<int D>
__global__ void kv_cache_update_kernel_paged
(
    const half*    __restrict__ k,             // [bsz, kv_append_len, n_kv_heads, D]
    const half*    __restrict__ v,             // [bsz, kv_append_len, n_kv_heads, D]
    half*          __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    half*          __restrict__ v_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    int64_t bsz,
    int64_t kv_append_len,
    int64_t n_kv_heads,
    int64_t num_pages_per_seq
)
{
    constexpr int THREADS = D / 2;

    const int64_t bt_idx  = (int64_t) blockIdx.x;  // batch-token flattened
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int tid         = threadIdx.x;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch  = bt_idx / kv_append_len;
    const int64_t kv_pos = bt_idx % kv_append_len;

    const int64_t logical_pos = (int64_t)cache_seqlens[batch] + kv_pos;
    const int32_t* block_row  = block_table + batch * num_pages_per_seq;

    const uint64_t src_off =
        ((((uint64_t)batch * (uint64_t)kv_append_len + (uint64_t)kv_pos) * (uint64_t)n_kv_heads +
          (uint64_t)kv_head) * (uint64_t)D) + (uint64_t)d0;

    const uint64_t dst_off =
        paged_cache_offset(block_row, logical_pos, kv_head, n_kv_heads, D, d0);

    *((half2*)(k_cache + dst_off)) = *((const half2*)(k + src_off));
    *((half2*)(v_cache + dst_off)) = *((const half2*)(v + src_off));
}


template<int G>
__global__ void attn_chunked_paged_kernel_512x256
(
    const half*    __restrict__ q,             // [bsz, q_len, n_q_heads, 512]
    const half*    __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, 512]
    const half*    __restrict__ v_cache,       // [num_cache_pages, 256, n_kv_heads, 512]
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    float*         __restrict__ workspace,     // [bsz, q_len, n_q_heads, n_chunks, 514]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    int64_t num_pages_per_seq,
    bool causal,
    float scale
)
{
    constexpr int D       = 512;
    constexpr int THREADS = 256;
    constexpr int WARPS   = THREADS / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t total_k_len = (int64_t)cache_seqlens[batch] + kv_append_len;
    const int64_t kv_start    = chunk * kv_chunk_size;
    const int64_t kv_end      = min(kv_start + kv_chunk_size, total_k_len);

    // Bottom-right aligned causal masking:
    // causal_limit = q_pos + seqlen_k - seqlen_q
    const int64_t causal_limit = causal ? (total_k_len - q_len + q_pos) : (total_k_len - 1);

    const int32_t* block_row = block_table + batch * num_pages_per_seq;

    __shared__ float partial_smem[G * WARPS];
    __shared__ float score_smem[G];

    half2 q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)d0;

        q_reg[g] = *((half2*)(q + q_off));
    }

    float m_reg[G], l_reg[G], o0_reg[G], o1_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g]  = -INFINITY;
        l_reg[g]  = 0.f;
        o0_reg[g] = 0.f;
        o1_reg[g] = 0.f;
    }

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        const uint64_t k_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, d0);
        const half2 k_reg = *((half2*)(k_cache + k_off));
        const float2 kf   = __half22float2(k_reg);

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float2 qf = __half22float2(q_reg[g]);
            partial[g] = qf.x * kf.x + qf.y * kf.y;
        }

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                partial_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        if (warp_id == 0 && lane_id < G)
        {
            const int g = lane_id;
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; w++)
                s += partial_smem[g * WARPS + w];
            score_smem[g] = s * scale;
        }
        __syncthreads();

        const uint64_t v_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, d0);
        const half2 v_reg = *((half2*)(v_cache + v_off));
        const float2 vf   = __half22float2(v_reg);

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = score_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o0_reg[g] = alpha * o0_reg[g] + exp_s * vf.x;
            o1_reg[g] = alpha * o1_reg[g] + exp_s * vf.y;
            l_reg[g]  = alpha * l_reg[g]  + exp_s;
            m_reg[g]  = m_new;
        }
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks +
              (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }

        ws[2 + d0] = o0_reg[g];
        ws[2 + d1] = o1_reg[g];
    }
}


template<int G>
__global__ void attn_chunked_kernel_512x256
(
    const half* __restrict__ q,         // [bsz, q_len,  n_q_heads, 512]
    const half* __restrict__ k,         // [bsz, kv_len, n_kv_heads, 512]
    const half* __restrict__ v,         // [bsz, kv_len, n_kv_heads, 512]
    float*      __restrict__ workspace, // [bsz, q_len, n_q_heads, n_chunks, 514]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    bool causal,
    float scale
)
{
    constexpr int D       = 512;
    constexpr int THREADS = 256;
    constexpr int WARPS   = THREADS / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t kv_start = chunk * kv_chunk_size;
    const int64_t kv_end   = min(kv_start + kv_chunk_size, kv_len);

    // Lower-right aligned causal masking:
    // q[q_pos] corresponds to absolute position (kv_len - q_len + q_pos).
    const int64_t causal_limit = causal ? (kv_len - q_len + q_pos) : (kv_len - 1);

    __shared__ float partial_smem[G * WARPS];
    __shared__ float score_smem[G];

    // Load q once into per-thread registers.
    half2 q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)d0;

        q_reg[g] = *((half2*)(q + q_off));
    }

    float m_reg[G], l_reg[G], o0_reg[G], o1_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g]  = -INFINITY;
        l_reg[g]  = 0.f;
        o0_reg[g] = 0.f;
        o1_reg[g] = 0.f;
    }

    const half* k_base = k + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);
    const half* v_base = v + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        const uint64_t kv_off = ((uint64_t)kv_pos * (uint64_t)n_kv_heads * (uint64_t)D) + (uint64_t)d0;

        // Load K directly to registers.
        const half2 k_reg = *((const half2*)(k_base + kv_off));
        const float2 kf   = __half22float2(k_reg);

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float2 qf = __half22float2(q_reg[g]);
            partial[g] = qf.x * kf.x + qf.y * kf.y;
        }

        // Intra-warp reduction.
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        // One partial per warp/head.
        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                partial_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        // Warp 0 reduces across warps. Lane g handles head g.
        if (warp_id == 0 && lane_id < G)
        {
            const int g = lane_id;
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; w++)
                s += partial_smem[g * WARPS + w];
            score_smem[g] = s * scale;
        }
        __syncthreads();

        // Load V directly to registers.
        const half2 v_reg = *((half2*)(v_base + kv_off));
        const float2 vf   = __half22float2(v_reg);

        // Online softmax update.
        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = score_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o0_reg[g] = alpha * o0_reg[g] + exp_s * vf.x;
            o1_reg[g] = alpha * o1_reg[g] + exp_s * vf.y;
            l_reg[g]  = alpha * l_reg[g]  + exp_s;
            m_reg[g]  = m_new;
        }
    }

    // Write chunk partials to workspace.
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks +
              (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }

        ws[2 + d0] = o0_reg[g];
        ws[2 + d1] = o1_reg[g];
    }
}


__global__ void attn_reduce_kernel_512x256
(
    const float* __restrict__ workspace,  // [bsz, q_len, n_q_heads, n_chunks, 514]
    half*        __restrict__ output,     // [bsz, q_len, n_q_heads, 512]
    int64_t bsz,
    int64_t q_len,
    int64_t n_q_heads,
    int64_t n_chunks
)
{
    constexpr int D = 512;

    const int64_t bq_idx = (int64_t)(blockIdx.x);
    const int64_t q_head = (int64_t)(blockIdx.y);
    const int tid        = threadIdx.x;

    const int d0 = tid * 2;
    const int d1 = d0 + 1;

    const int64_t batch = bq_idx / q_len;
    const int64_t q_pos = bq_idx % q_len;

    float m_acc  = -INFINITY;
    float l_acc  = 0.f;
    float o0_acc = 0.f;
    float o1_acc = 0.f;

    const float* ws_base = workspace +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
          (uint64_t)q_head) * (uint64_t)n_chunks * (uint64_t)(D + 2));

    for (int64_t c = 0; c < n_chunks; c++)
    {
        const float* ws_c = ws_base + (uint64_t)c * (uint64_t)(D + 2);
        const float m_c   = ws_c[0];
        const float l_c   = ws_c[1];

        if (l_c == 0.f) continue;

        const float o0_c  = ws_c[2 + d0];
        const float o1_c  = ws_c[2 + d1];

        const float m_new = fmaxf(m_acc, m_c);
        const float alpha = __expf(m_acc - m_new);
        const float beta  = __expf(m_c  - m_new);

        o0_acc = alpha * o0_acc + beta * o0_c;
        o1_acc = alpha * o1_acc + beta * o1_c;
        l_acc  = alpha * l_acc  + beta * l_c;
        m_acc  = m_new;
    }

    half* out = output +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
          (uint64_t)q_head) * (uint64_t)D);

    out[d0] = __float2half(l_acc > 0.f ? (o0_acc / l_acc) : 0.f);
    out[d1] = __float2half(l_acc > 0.f ? (o1_acc / l_acc) : 0.f);
}


template<int D, int G>
__global__ void attn_chunked_paged_kernel
(
    const half*    __restrict__ q,             // [bsz, q_len, n_q_heads, D]
    const half*    __restrict__ k_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const half*    __restrict__ v_cache,       // [num_cache_pages, 256, n_kv_heads, D]
    const int32_t* __restrict__ block_table,   // [bsz, num_pages_per_seq]
    const int32_t* __restrict__ cache_seqlens, // [bsz]
    float*         __restrict__ workspace,     // [bsz, q_len, n_q_heads, n_chunks, D+2]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_append_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    int64_t num_pages_per_seq,
    bool causal,
    float scale
)
{
    constexpr int WARPS = D / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t total_k_len = (int64_t)cache_seqlens[batch] + kv_append_len;
    const int64_t kv_start    = chunk * kv_chunk_size;
    const int64_t kv_end      = min(kv_start + kv_chunk_size, total_k_len);

    const int64_t causal_limit = causal ? (total_k_len - q_len + q_pos) : (total_k_len - 1);

    const int32_t* block_row = block_table + batch * num_pages_per_seq;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half*  kv_smem     = (half*) smem_raw;
    float* reduce_smem = (float*) (kv_smem + D);

    register half q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)tid;
        q_reg[g] = q[q_off];
    }

    register float m_reg[G], l_reg[G], o_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g] = -INFINITY;
        l_reg[g] = 0.f;
        o_reg[g] = 0.f;
    }

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        const uint64_t k_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, tid);
        kv_smem[tid] = k_cache[k_off];
        __syncthreads();

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
            partial[g] = __half2float(q_reg[g]) * __half2float(kv_smem[tid]);

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                reduce_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        if (tid == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                float s = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS; w++)
                    s += reduce_smem[g * WARPS + w];
                reduce_smem[g] = s * scale;
            }
        }
        __syncthreads();

        const uint64_t v_off =
            paged_cache_offset(block_row, kv_pos, kv_head, n_kv_heads, D, tid);
        kv_smem[tid] = v_cache[v_off];
        __syncthreads();

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = reduce_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o_reg[g] = alpha * o_reg[g] + exp_s * __half2float(kv_smem[tid]);
            l_reg[g] = alpha * l_reg[g] + exp_s;
            m_reg[g] = m_new;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks +
              (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }
        ws[2 + tid] = o_reg[g];
    }
}


template<int D, int G>
__global__ void attn_chunked_kernel
(
    const half* __restrict__ q,         // [bsz, q_len,  n_q_heads,  D]
    const half* __restrict__ k,         // [bsz, kv_len, n_kv_heads, D]
    const half* __restrict__ v,         // [bsz, kv_len, n_kv_heads, D]
    float*      __restrict__ workspace, // [bsz, q_len, n_q_heads, n_chunks, D+2]
    int64_t bsz,
    int64_t q_len,
    int64_t kv_len,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t n_chunks,
    int64_t kv_chunk_size,
    bool causal,
    float scale
)
{
    constexpr int WARPS = D / 32;

    const int64_t bq_idx  = (int64_t) blockIdx.x;
    const int64_t kv_head = (int64_t) blockIdx.y;
    const int64_t chunk   = (int64_t) blockIdx.z;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / 32;
    const int lane_id     = tid % 32;

    const int64_t batch        = bq_idx / q_len;
    const int64_t q_pos        = bq_idx % q_len;
    const int64_t q_head_start = kv_head * G;

    const int64_t kv_start = chunk * kv_chunk_size;
    const int64_t kv_end   = min(kv_start + kv_chunk_size, kv_len);

    // Standard causal semantics: query position q_pos may attend to keys 0..q_pos.
    // If you later want decode-style bottom-right alignment, make q_start explicit.
    const int64_t seq_pos      = q_pos;
    const int64_t causal_limit = causal ? ((kv_len - q_len) + seq_pos) : (kv_len - 1);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half*  kv_smem     = (half*) smem_raw;
    float* reduce_smem = (float*) (kv_smem + D);
    // reduce_smem has WARPS*G_MAX entries:
    //   reduce_smem[g*WARPS + w] is warp w's partial for head g.

    register half q_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        const uint64_t q_off =
            ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
              (uint64_t)(q_head_start + g)) * (uint64_t)D) + (uint64_t)tid;
        q_reg[g] = q[q_off];
    }

    register float m_reg[G], l_reg[G], o_reg[G];
    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        m_reg[g] = -INFINITY;
        l_reg[g] = 0.f;
        o_reg[g] = 0.f;
    }

    const half* k_base = k + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);
    const half* v_base = v + (((uint64_t)batch * (uint64_t)kv_len * (uint64_t)n_kv_heads + (uint64_t)kv_head) * (uint64_t)D);

    for (int64_t kv_pos = kv_start; kv_pos < kv_end; kv_pos++)
    {
        if (kv_pos > causal_limit) break;

        kv_smem[tid] = k_base[((uint64_t)kv_pos * (uint64_t)n_kv_heads * (uint64_t)D) + (uint64_t)tid];
        __syncthreads();

        float partial[G];
        #pragma unroll
        for (int g = 0; g < G; g++)
            partial[g] = __half2float(q_reg[g]) * __half2float(kv_smem[tid]);

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                partial[g] += __shfl_xor_sync(0xffffffff, partial[g], mask);
        }

        if (lane_id == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
                reduce_smem[g * WARPS + warp_id] = partial[g];
        }
        __syncthreads();

        if (tid == 0)
        {
            #pragma unroll
            for (int g = 0; g < G; g++)
            {
                float s = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS; w++)
                    s += reduce_smem[g * WARPS + w];
                reduce_smem[g] = s * scale;
            }
        }

        kv_smem[tid] = v_base[((uint64_t)kv_pos * (uint64_t)n_kv_heads * (uint64_t)D) + (uint64_t)tid];
        __syncthreads();

        #pragma unroll
        for (int g = 0; g < G; g++)
        {
            const float s     = reduce_smem[g];
            const float m_new = fmaxf(m_reg[g], s);
            const float alpha = __expf(m_reg[g] - m_new);
            const float exp_s = __expf(s - m_new);

            o_reg[g] = alpha * o_reg[g] + exp_s * __half2float(kv_smem[tid]);
            l_reg[g] = alpha * l_reg[g] + exp_s;
            m_reg[g] = m_new;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < G; g++)
    {
        float* ws = workspace +
            (((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
               (uint64_t)(q_head_start + g)) * (uint64_t)n_chunks + (uint64_t)chunk) * (uint64_t)(D + 2));

        if (tid == 0)
        {
            ws[0] = m_reg[g];
            ws[1] = l_reg[g];
        }
        ws[2 + tid] = o_reg[g];
    }
}

template<int D, int G>
__global__ void attn_reduce_kernel
(
    const float* __restrict__ workspace,  // [bsz, q_len, n_q_heads, n_chunks, D+2]
    half*        __restrict__ output,     // [bsz, q_len, n_q_heads, D]
    int64_t bsz,
    int64_t q_len,
    int64_t n_q_heads,
    int64_t n_chunks
)
{
    const int64_t bq_idx = (int64_t)(blockIdx.x);
    const int64_t q_head = (int64_t)(blockIdx.y);
    const int tid        = threadIdx.x;

    const int64_t batch = bq_idx / q_len;
    const int64_t q_pos = bq_idx % q_len;

    float m_acc = -INFINITY;
    float l_acc = 0.f;
    float o_acc = 0.f;

    const float* ws_base = workspace +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads +
          (uint64_t)q_head) * (uint64_t)n_chunks * (uint64_t)(D + 2));

    for (int64_t c = 0; c < n_chunks; c++)
    {
        const float* ws_c = ws_base + (uint64_t)c * (uint64_t)(D + 2);
        const float m_c   = ws_c[0];
        const float l_c   = ws_c[1];
        const float o_c   = ws_c[2 + tid];

        if (l_c == 0.f) continue;

        const float m_new = fmaxf(m_acc, m_c);
        const float alpha = __expf(m_acc - m_new);
        const float beta  = __expf(m_c  - m_new);

        o_acc = alpha * o_acc + beta * o_c;
        l_acc = alpha * l_acc + beta * l_c;
        m_acc = m_new;
    }

    half* out = output +
        ((((uint64_t)batch * (uint64_t)q_len + (uint64_t)q_pos) * (uint64_t)n_q_heads + (uint64_t)q_head) * (uint64_t)D);

    out[tid] = __float2half(l_acc > 0.f ? (o_acc / l_acc) : 0.f);
}

void bighead_attn_paged
(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    const at::Tensor& o,
    // const at::Tensor& workspace,
    int kv_chunk_size,
    bool causal,
    float sm_scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k, kHalf);
    TORCH_CHECK_DTYPE(v, kHalf);
    TORCH_CHECK_DTYPE(k_cache, kHalf);
    TORCH_CHECK_DTYPE(v_cache, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    // TORCH_CHECK_DTYPE(workspace, kFloat);
    TORCH_CHECK_DTYPE(block_table, kInt);
    TORCH_CHECK_DTYPE(cache_seqlens, kInt);

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
    TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
    // TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(cache_seqlens.is_contiguous(), "cache_seqlens must be contiguous");

    TORCH_CHECK(q.dim() == 4, "q must be rank-4");
    TORCH_CHECK(k.dim() == 4, "k must be rank-4");
    TORCH_CHECK(v.dim() == 4, "v must be rank-4");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be rank-4");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be rank-4");
    TORCH_CHECK(o.dim() == 4, "o must be rank-4");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be rank-2");
    TORCH_CHECK(cache_seqlens.dim() == 1, "cache_seqlens must be rank-1");

    const int64_t bsz            = q.size(0);
    const int64_t q_len          = q.size(1);
    const int64_t n_q_heads      = q.size(2);
    const int64_t dim            = q.size(3);

    const int64_t kv_append_len  = k.size(1);
    const int64_t n_kv_heads     = k.size(2);

    const int64_t num_cache_pages   = k_cache.size(0);
    const int64_t page_size         = k_cache.size(1);
    const int64_t num_pages_per_seq = block_table.size(1);

    TORCH_CHECK(k.size(0) == bsz, "k batch mismatch");
    TORCH_CHECK(v.size(0) == bsz, "v batch mismatch");
    TORCH_CHECK(v.size(1) == kv_append_len, "v kv_len mismatch");
    TORCH_CHECK(v.size(2) == n_kv_heads, "v n_kv_heads mismatch");
    TORCH_CHECK(k.size(3) == dim, "k head_dim mismatch");
    TORCH_CHECK(v.size(3) == dim, "v head_dim mismatch");

    TORCH_CHECK(k_cache.size(1) == PAGE_SIZE, "k_cache page size must be 256");
    TORCH_CHECK(v_cache.size(1) == PAGE_SIZE, "v_cache page size must be 256");
    TORCH_CHECK(k_cache.size(0) == num_cache_pages, "internal k_cache shape error");
    TORCH_CHECK(v_cache.size(0) == num_cache_pages, "v_cache num_pages mismatch");
    TORCH_CHECK(k_cache.size(2) == n_kv_heads, "k_cache n_kv_heads mismatch");
    TORCH_CHECK(v_cache.size(2) == n_kv_heads, "v_cache n_kv_heads mismatch");
    TORCH_CHECK(k_cache.size(3) == dim, "k_cache head_dim mismatch");
    TORCH_CHECK(v_cache.size(3) == dim, "v_cache head_dim mismatch");

    TORCH_CHECK(o.size(0) == bsz, "o batch mismatch");
    TORCH_CHECK(o.size(1) == q_len, "o q_len mismatch");
    TORCH_CHECK(o.size(2) == n_q_heads, "o n_q_heads mismatch");
    TORCH_CHECK(o.size(3) == dim, "o head_dim mismatch");

    TORCH_CHECK(block_table.size(0) == bsz, "block_table batch mismatch");
    TORCH_CHECK(cache_seqlens.size(0) == bsz, "cache_seqlens batch mismatch");

    TORCH_CHECK(n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads");
    const int64_t G = n_q_heads / n_kv_heads;
    TORCH_CHECK(G <= G_MAX, "GQA ratio ", G, " exceeds G_MAX=", G_MAX);
    TORCH_CHECK(kv_chunk_size > 0, "kv_chunk_size must be positive");

    const int64_t max_total_k_len = num_pages_per_seq * PAGE_SIZE;

    const uint64_t ws_numel      = WORKSPACE_SIZE / sizeof(float);
    int64_t n_chunks;
    while (true)
    {
        n_chunks                 = (max_total_k_len + kv_chunk_size - 1) / kv_chunk_size;
        const uint64_t ws_needed = (uint64_t)bsz * (uint64_t)q_len * (uint64_t)n_q_heads * (uint64_t)n_chunks * (uint64_t)(dim + 2);
        if (ws_needed > ws_numel)
            kv_chunk_size *= 2;
        else break;
    }
    float* ws_ptr                = (float*) DevCtx::instance().get_ws(q.get_device());
    // DBGI2(kv_chunk_size, n_chunks);

    const half* q_ptr            = (const half*) q.data_ptr();
    const half* k_ptr            = (const half*) k.data_ptr();
    const half* v_ptr            = (const half*) v.data_ptr();
    half* k_cache_ptr            = (half*) k_cache.data_ptr();
    half* v_cache_ptr            = (half*) v_cache.data_ptr();
    half* o_ptr                  = (half*) o.data_ptr();
    const int32_t* block_ptr     = block_table.data_ptr<int32_t>();
    const int32_t* seqlens_ptr   = cache_seqlens.data_ptr<int32_t>();

    dim3 grid_up((uint32_t)std::max<int64_t>(1, bsz * kv_append_len), (uint32_t)n_kv_heads);
    dim3 grid1((uint32_t)(bsz * q_len), (uint32_t)n_kv_heads, (uint32_t)n_chunks);
    dim3 grid2((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

    const float scale = sm_scale == 0.0f ? rsqrtf((float) dim) : sm_scale;

    #define PAGED_UPDATE_ARGS \
        k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, block_ptr, seqlens_ptr, \
        bsz, kv_append_len, n_kv_heads, num_pages_per_seq

    #define PAGED_ARGS1 \
        q_ptr, k_cache_ptr, v_cache_ptr, block_ptr, seqlens_ptr, ws_ptr, \
        bsz, q_len, kv_append_len, n_q_heads, n_kv_heads, n_chunks, \
        (int64_t)kv_chunk_size, num_pages_per_seq, causal, scale

    #define ARGS2 \
        ws_ptr, o_ptr, bsz, q_len, n_q_heads, n_chunks

    #define LAUNCH_PAGED(DIM, GVAL) \
        if (dim == DIM && G == GVAL) { \
            if (kv_append_len > 0) { \
                kv_cache_update_kernel_paged<DIM><<<grid_up, DIM / 2, 0, stream>>>(PAGED_UPDATE_ARGS); \
                cuda_check(cudaPeekAtLastError()); \
            } \
            const size_t smem_bytes = \
                (size_t)DIM * sizeof(half) + (size_t)(DIM / 32) * G * sizeof(float); \
            attn_chunked_paged_kernel<DIM, GVAL><<<grid1, DIM, smem_bytes, stream>>>(PAGED_ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            attn_reduce_kernel<DIM, GVAL><<<grid2, DIM, 0, stream>>>(ARGS2); \
            cuda_check(cudaPeekAtLastError()); \
        }

    #define LAUNCH_PAGED_512(GVAL) \
        if (dim == 512 && G == GVAL) { \
            if (kv_append_len > 0) { \
                kv_cache_update_kernel_paged<512><<<grid_up, 256, 0, stream>>>(PAGED_UPDATE_ARGS); \
                cuda_check(cudaPeekAtLastError()); \
            } \
            attn_chunked_paged_kernel_512x256<GVAL><<<grid1, 256, 0, stream>>>(PAGED_ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            attn_reduce_kernel_512x256<<<grid2, 256, 0, stream>>>(ARGS2); \
            cuda_check(cudaPeekAtLastError()); \
        }

    LAUNCH_PAGED_512(1)
    else LAUNCH_PAGED_512(2)
    else LAUNCH_PAGED_512(4)
    else LAUNCH_PAGED_512(8)
    else LAUNCH_PAGED(64, 1)
    else LAUNCH_PAGED(64, 2)
    else LAUNCH_PAGED(64, 4)
    else LAUNCH_PAGED(64, 8)
    else LAUNCH_PAGED(128, 1)
    else LAUNCH_PAGED(128, 2)
    else LAUNCH_PAGED(128, 4)
    else LAUNCH_PAGED(128, 8)
    else LAUNCH_PAGED(256, 1)
    else LAUNCH_PAGED(256, 2)
    else LAUNCH_PAGED(256, 4)
    else LAUNCH_PAGED(256, 8)
    else TORCH_CHECK(false,
        "head_dim must be 64, 128, 256, or 512, num_kv_groups must be 1, 2, 4 or 8");

    #undef LAUNCH_PAGED_512
    #undef LAUNCH_PAGED
    #undef ARGS2
    #undef PAGED_ARGS1
    #undef PAGED_UPDATE_ARGS
}

void bighead_attn
(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& o,
    // const at::Tensor& workspace,
    int kv_chunk_size,
    bool causal,
    float sm_scale
)
{
    const at::cuda::OptionalCUDAGuard device_guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(q, kHalf);
    TORCH_CHECK_DTYPE(k, kHalf);
    TORCH_CHECK_DTYPE(v, kHalf);
    TORCH_CHECK_DTYPE(o, kHalf);
    // TORCH_CHECK_DTYPE(workspace, kFloat);

    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
    // TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");

    TORCH_CHECK(q.dim() == 4, "q must be rank-4");
    TORCH_CHECK(k.dim() == 4, "k must be rank-4");
    TORCH_CHECK(v.dim() == 4, "v must be rank-4");
    TORCH_CHECK(o.dim() == 4, "o must be rank-4");

    const int64_t bsz        = q.size(0);
    const int64_t q_len      = q.size(1);
    const int64_t n_q_heads  = q.size(2);
    const int64_t dim        = q.size(3);
    const int64_t kv_len     = k.size(1);
    const int64_t n_kv_heads = k.size(2);
    const int64_t G          = n_q_heads / n_kv_heads;

    TORCH_CHECK(k.size(0) == bsz, "k batch mismatch");
    TORCH_CHECK(v.size(0) == bsz, "v batch mismatch");
    TORCH_CHECK(v.size(1) == kv_len, "v kv_len mismatch");
    TORCH_CHECK(v.size(2) == n_kv_heads, "v n_kv_heads mismatch");
    TORCH_CHECK(k.size(3) == dim, "k head_dim mismatch");
    TORCH_CHECK(v.size(3) == dim, "v head_dim mismatch");
    TORCH_CHECK(o.size(0) == bsz, "o batch mismatch");
    TORCH_CHECK(o.size(1) == q_len, "o q_len mismatch");
    TORCH_CHECK(o.size(2) == n_q_heads, "o n_q_heads mismatch");
    TORCH_CHECK(o.size(3) == dim, "o head_dim mismatch");

    TORCH_CHECK(n_q_heads % n_kv_heads == 0, "n_q_heads must be divisible by n_kv_heads");
    TORCH_CHECK(G <= G_MAX, "GQA ratio ", G, " exceeds G_MAX=", G_MAX);
    TORCH_CHECK(kv_chunk_size > 0, "kv_chunk_size must be positive");

    const uint64_t ws_numel      = WORKSPACE_SIZE / sizeof(float);
    int64_t n_chunks;
    while (true)
    {
        n_chunks                 = (kv_len + kv_chunk_size - 1) / kv_chunk_size;
        const uint64_t ws_needed = (uint64_t)bsz * (uint64_t)q_len * (uint64_t)n_q_heads * (uint64_t)n_chunks * (uint64_t)(dim + 2);
        if (ws_needed > ws_numel)
            kv_chunk_size *= 2;
        else break;
    }
    float* ws_ptr                = (float*) DevCtx::instance().get_ws(q.get_device());

    const half* q_ptr = (const half*) q.data_ptr();
    const half* k_ptr = (const half*) k.data_ptr();
    const half* v_ptr = (const half*) v.data_ptr();
    half* o_ptr       = (half*) o.data_ptr();

    dim3 grid1((uint32_t)(bsz * q_len), (uint32_t)n_kv_heads, (uint32_t)n_chunks);
    dim3 grid2((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

    dim3 grid1_512((uint32_t)(bsz * q_len), (uint32_t)n_kv_heads, (uint32_t)n_chunks);
    dim3 grid2_512((uint32_t)(bsz * q_len), (uint32_t)n_q_heads);

    const float scale = sm_scale == 0.0f ? rsqrtf((float) dim) : sm_scale;

    #define ARGS1 \
        q_ptr, k_ptr, v_ptr, ws_ptr, \
        bsz, q_len, kv_len, n_q_heads, n_kv_heads, n_chunks, \
        (int64_t)kv_chunk_size, causal, scale

    #define ARGS2 \
        ws_ptr, o_ptr, bsz, q_len, n_q_heads, n_chunks

    #define LAUNCH(DIM, GVAL) \
        if (dim == DIM && G == GVAL) { \
            const size_t smem_bytes = \
                (size_t)DIM * sizeof(half) + (size_t)(DIM / 32) * G * sizeof(float); \
            attn_chunked_kernel<DIM, GVAL><<<grid1, DIM, smem_bytes, stream>>>(ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            attn_reduce_kernel<DIM, GVAL><<<grid2, DIM, 0, stream>>>(ARGS2); \
            cuda_check(cudaPeekAtLastError()); \
        }

    #define LAUNCH_512(GVAL) \
        if (dim == 512 && G == GVAL) { \
            attn_chunked_kernel_512x256<GVAL><<<grid1_512, 256, 0, stream>>>(ARGS1); \
            cuda_check(cudaPeekAtLastError()); \
            attn_reduce_kernel_512x256<<<grid2_512, 256, 0, stream>>>(ARGS2); \
            cuda_check(cudaPeekAtLastError()); \
        }

    LAUNCH_512(1)
    else LAUNCH_512(2)
    else LAUNCH_512(4)
    else LAUNCH_512(8)
    else LAUNCH(64, 1)
    else LAUNCH(64, 2)
    else LAUNCH(64, 4)
    else LAUNCH(64, 8)
    else LAUNCH(128, 1)
    else LAUNCH(128, 2)
    else LAUNCH(128, 4)
    else LAUNCH(128, 8)
    else LAUNCH(256, 1)
    else LAUNCH(256, 2)
    else LAUNCH(256, 4)
    else LAUNCH(256, 8)
    else TORCH_CHECK(false, "head_dim must be 64, 128, 256, or 512, num_kv_groups must be 1, 2, 4 or 8");

    #undef LAUNCH_512
    #undef LAUNCH
    #undef ARGS2
    #undef ARGS1
}

size_t bighead_attn_workspace_size
(
    int bsz,
    int q_len,
    int n_q_heads,
    int max_kv_len,  // or PAGE_SIZE * max_pages_per_seq
    int kv_chunk_size,
    int dim
)
{
    const int64_t n_chunks = (max_kv_len + kv_chunk_size - 1) / kv_chunk_size;
    return
        (size_t)bsz *
        (size_t)q_len *
        (size_t)n_q_heads *
        (size_t)n_chunks *
        (size_t)(dim + 2);
}
