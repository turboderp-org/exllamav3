#include <cuda_fp16.h>
#include "quantize.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"
#include "exl3_devctx.cuh"
#include <cmath>

#define NUM_THREADS 1024
#define H_INF __ushort_as_half(0x7c00)
#define H_NINF __ushort_as_half(0xfc00)

template <int K, int cb>
__global__ __launch_bounds__(MIN(NUM_THREADS, 65536 >> K))
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

    int tile_idx = blockIdx.x;
    int thread = threadIdx.x;

    constexpr int Kr = 16 - K;
    constexpr int max_q = 1 << K;
    constexpr int edges = 65536 >> K;

    const float* input_tile = input_tiles_ptr + 256 * tile_idx;
    float* output_tile = output_tiles_ptr + 256 * tile_idx;
    uint16_t* output_indices = output_indices_ptr + 256 * tile_idx;
    uint16_t* temp_edges = temp_edges_ptr + 256 * edges * tile_idx;

    // Tile buffer
    half* sh_input_tile = (half*) sh; sh += 256 * sizeof(half);

    half* sh_min = (half*) sh; sh += 32 * sizeof(half);
    int* sh_idx = (int*) sh; sh += 32 * sizeof(int);

    // K >= mshk lets temp_costs fit in shmem, otherwise fall back to global temp buffer
    constexpr int mshk = 2;
    half* sh_temp_costs = (half*) sh;
    half* temp_costs = K >= mshk ? sh_temp_costs : temp_costs_ptr + 2 * edges * tile_idx;
    half* temp_costs_inc = temp_costs + edges;

    // Fetch input tile to shmem
    if (thread < 256) sh_input_tile[thread] = __float2half_rn(input_tile[thread]);
    __syncthreads();

    auto forward = [&](int roll, int pre_state)
    {
        int ri = roll % 256;
        half dh, err, min_err, w;

        // temp_costs_inc[z] is the cost/cumulative error of an incoming edge from state (z & edge_mask)
        half* t = temp_costs;
        temp_costs = temp_costs_inc;
        temp_costs_inc = t;

        for (int out_edge_idx = thread; out_edge_idx < edges; out_edge_idx += NUM_THREADS)
        {
            w = sh_input_tile[ri];

            int state = out_edge_idx;
            int in_edge_idx = state >> K;
            dh = __hsub(decode_3inst<cb>(state), w);
            err = __hmul(dh, dh);
            if (pre_state >= 0 && in_edge_idx != pre_state) err = H_INF;
            min_err = err;
            int min_in_edge = in_edge_idx;

            #pragma unroll
            for (int k = 1; k < max_q; ++k)
            {
                state = (k << Kr) | out_edge_idx;
                in_edge_idx = state >> K;
                dh = __hsub(decode_3inst<cb>(state), w);
                err = __hmul(dh, dh);
                if (pre_state >= 0 && in_edge_idx != pre_state) err = H_INF;
                if (__hlt(err, min_err)) { min_err = err; min_in_edge = in_edge_idx; }
            }

            temp_costs[out_edge_idx] = min_err;
            temp_edges[edges * ri + out_edge_idx] = (uint16_t) min_in_edge;
        }

        // Next iteration depends on costs computed by current iteration
        __syncthreads();

        // Each thread iterates over all weights in the tile
        for (int i = 1; i < 256; ++i)
        {
            ri = (i + roll) % 256;

            // Swap buffers.
            t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;

            for (int out_edge_idx = thread; out_edge_idx < edges; out_edge_idx += NUM_THREADS)
            {
                w = sh_input_tile[ri];

                int state = out_edge_idx;
                int in_edge_idx = state >> K;
                dh = __hsub(decode_3inst<cb>(state), w);
                err = __hfma(dh, dh, temp_costs_inc[in_edge_idx]);
                min_err = err;
                int min_in_edge = in_edge_idx;

                #pragma unroll
                for (int k = 1; k < max_q; ++k)
                {
                    state = (k << Kr) | out_edge_idx;
                    in_edge_idx = state >> K;
                    dh = __hsub(decode_3inst<cb>(state), w);
                    err = __hfma(dh, dh, temp_costs_inc[in_edge_idx]);
                    if (__hlt(err, min_err)) { min_err = err; min_in_edge = in_edge_idx; }
                }

                temp_costs[out_edge_idx] = min_err;
                temp_edges[edges * ri + out_edge_idx] = (uint16_t) min_in_edge;
            }

            // Next iteration depends on costs computed by current iteration
            __syncthreads();
        }
    };

    auto argmin_cost = [&]()
    {
        // Find the final state with the lowest total cost. Return value is only valid in thread 0

        half local_min = H_INF;
        int local_idx = -1;
        #pragma unroll
        for (int e = threadIdx.x; e < edges; e += NUM_THREADS)
        {
            half v = temp_costs_inc[e];
            if (__hlt(v, local_min)) { local_min = v; local_idx = e; }
        }

        // Shuffle reduction
        int lane_id = thread % 32;
        int warp_id = thread / 32;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            half other_min = __shfl_down_sync(0xffffffff, local_min, offset, 32);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset, 32);
            if (__hlt(other_min, local_min))
            {
                local_min = other_min;
                local_idx = other_idx;
            }
        }

        sh_min[warp_id] = local_min;
        sh_idx[warp_id] = local_idx;
        __syncthreads();

        if (warp_id == 0)
        {
            local_min = lane_id * 32 < edges && thread < NUM_THREADS / 32 ? sh_min[lane_id] : H_INF;
            local_idx = thread < NUM_THREADS ? sh_idx[lane_id] : 0;

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                half other_min = __shfl_down_sync(0xffffffff, local_min, offset, 32);
                int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset, 32);
                if (__hlt(other_min, local_min))
                {
                    local_min = other_min;
                    local_idx = other_idx;
                }
            }
        }

        return local_idx;
    };

    auto backward = [&](int roll, bool write, int edge)
    {
        // Construct output tile. Since the graph has to be walked, this will run in a single thread per block.
        // Profiling says this is not a bottleneck

        if (thread == 0)
        {
            for (int i = 255; i >= 0; --i)
            {
                int ri = (i + roll) % 256;

                int prev_edge = (int) temp_edges[edges * ri + edge];
                int encoded = (prev_edge << K) | edge;
                edge = prev_edge;

                if (write)
                {
                    output_indices[ri] = (uint16_t) encoded;
                    output_tile[ri] = __half2float(decode_3inst<cb>(encoded));
                }
                else if (ri == 0) break;
            }
        }

        // Broadcast to block
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        edge = sh_idx[0];

        return edge;
    };

    // Solve starting at position 128 find initial state for second pass
    forward(128, -1);
    int end_state = argmin_cost();
    end_state = backward(128, false, end_state);

    // Solve again from position 0 with tail-biting constraint
    forward(0, end_state);
    backward(0, true, end_state);
}

#define __(i, cb) quantize_tiles_kernel<i, cb>
constexpr auto quantize_tiles_kernel_instances = std::array
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

/*
Quantize batch of tiles

input_tiles: shape (n, 256), float
output_tiles: shape (n, 256), float
output_indices: shape (n, 256), uint16_t (unpacked)
temp_costs: shape (max_bsz, 2, 65536 >> K), float (scratch space for Viterbi algorithm)
temp_edges: shape (max_bsz, 256, 65536 >> K), uint16_t (scratch space for Viterbi algorithm)
K: number of bits per weight (1..8)
*/

void quantize_tiles
(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int K,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(input_tiles, 2);
    TORCH_CHECK_SIZE(input_tiles, 1, 256);
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_indices);
    TORCH_CHECK_DTYPE(input_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_indices, kShort);

    int edges = 65536 >> K;
    int threads = MIN(NUM_THREADS, edges);

    int num_tiles = input_tiles.size(0);
    if (!num_tiles) return;

    TORCH_CHECK_DTYPE(temp_costs, kHalf);
    TORCH_CHECK_DIM(temp_costs, 3);
    TORCH_CHECK_SIZE(temp_costs, 1, 2);
    TORCH_CHECK_SIZE(temp_costs, 2, edges);

    TORCH_CHECK_DTYPE(temp_edges, kShort);
    TORCH_CHECK_DIM(temp_edges, 3);
    TORCH_CHECK_SIZE(temp_edges, 1, 256);
    TORCH_CHECK_SIZE(temp_edges, 2, edges);

    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    int max_batch_size = MIN(temp_costs.size(0), num_sms);

    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int batch_i = 0;
    do
    {
        int batch_j = MIN(batch_i + max_batch_size, num_tiles);

        const float* input_tiles_ptr = ((const float*) input_tiles.data_ptr()) + 256 * batch_i;
        float* output_tiles_ptr = ((float*) output_tiles.data_ptr()) + 256 * batch_i;
        uint16_t* output_indices_ptr = ((uint16_t*) output_indices.data_ptr()) + 256 * batch_i;
        half* temp_costs_ptr = (half*) temp_costs.data_ptr();
        uint16_t* temp_edges_ptr = (uint16_t*) temp_edges.data_ptr();

        int bsz = batch_j - batch_i;
        int kernel_idx = K - 1 + 8 * cb;
        int shmem = 2 * (65536 >> K) * sizeof(half) + 512 + 64 + 128;

        cudaFuncSetAttribute
        (
            quantize_tiles_kernel_instances[kernel_idx],
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem
        );
        cuda_check(cudaPeekAtLastError());

        quantize_tiles_kernel_instances[kernel_idx]<<<bsz, threads, shmem, stream>>>
        (
            input_tiles_ptr,
            output_tiles_ptr,
            output_indices_ptr,
            temp_costs_ptr,
            temp_edges_ptr
        );
        cuda_check(cudaPeekAtLastError());

        batch_i = batch_j;
    }
    while (batch_i < num_tiles);
}

template <typename T>
__global__ //__launch_bounds__(64)
void decode_kernel
(
    const uint16_t* __restrict__ input_tiles_ptr,
    T* __restrict__ output_tiles_ptr,
    int cols,
    bool mcg,
    bool mul1
)
{
    int col = threadIdx.x + blockIdx.x * 64;
    if (col >= cols) return;
    int row = blockIdx.y;
    int idx = row * cols + col;

    uint32_t enc = (uint32_t) input_tiles_ptr[idx];
    half w;
    if (mcg)
        w = decode_3inst<1>(enc);
    else if (mul1)
        w = decode_3inst<2>(enc);
    else
        w = decode_3inst<0>(enc);

    if constexpr (std::is_same_v<T, float>)
        output_tiles_ptr[idx] = __half2float(w);
    else
        output_tiles_ptr[idx] = w;
}

/*
Decode tensor

input_indices: uint16_t
output_tiles: float or half
mcg: use mcg codebook
mul1: use mcg codebook
*/

void decode
(
    at::Tensor input_indices,
    at::Tensor output_tiles,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_indices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(input_indices, 2);
    TORCH_CHECK_SHAPES_FULL(input_indices, output_tiles);
    TORCH_CHECK_DTYPE(input_indices, kShort);

    int rows = input_indices.size(0);
    int cols = input_indices.size(1);

    dim3 blockDim(64);
    dim3 gridDim(cols / 64, rows);

    if (output_tiles.dtype() == at::kFloat)
        decode_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const uint16_t*) input_indices.data_ptr(),
            (float*) output_tiles.data_ptr(),
            cols,
            mcg,
            mul1
        );
    else if (output_tiles.dtype() == at::kHalf)
        decode_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const uint16_t*) input_indices.data_ptr(),
            (half*) output_tiles.data_ptr(),
            cols,
            mcg,
            mul1
        );
}


#define NUM_THREADS_TD 1024
#define MAX_BINS 1024

__global__ __launch_bounds__(NUM_THREADS_TD)
void test_distribution_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ dist_output_ptr,
    float* __restrict__ ref_output_ptr,
    uint64_t numel,
    uint64_t num_bins,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
)
{
    __shared__ int histogram[MAX_BINS];
    auto reset_histogram = [&]()
    {
        for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS_TD)
            histogram[i] = 0;
        __syncthreads();
    };

    auto write_histogram = [&](float* output_ptr, uint64_t sc)
    {
        float scf = (float) sc;
        for (int i = threadIdx.x; i < num_bins; i += NUM_THREADS_TD)
            output_ptr[i] = ((float) histogram[i]) / scf;
        __syncthreads();
    };

    auto count = [&](float val)
    {
        val -= min_value;
        val /= (max_value - min_value);
        val *= (float) num_bins;
        int idx = (int) val;
        if (idx < 0) idx = 0;
        if (idx > num_bins - 1) idx = num_bins - 1;
        atomicAdd(&histogram[idx], 1);
    };

    if (ref_output_ptr)
    {
        reset_histogram();
        for (uint64_t i = threadIdx.x; i < 65536; i += NUM_THREADS_TD)
        {
            if (mcg)
                count(decode_3inst_f<1>((uint16_t) (i & 0xffff)));
            else if (mul1)
                count(decode_3inst_f<2>((uint16_t) (i & 0xffff)));
            else
                count(decode_3inst_f<0>((uint16_t) (i & 0xffff)));
        }
        __syncthreads();
        write_histogram(ref_output_ptr, 65536);
    }

    reset_histogram();
    for (uint64_t i = threadIdx.x; i < numel; i += NUM_THREADS_TD)
        count(input_ptr[i]);
    __syncthreads();
    write_histogram(dist_output_ptr, numel);
}

/*
Compare tensor distribution to codebook (not optimized)

input: tensor, float, any shape
dist_output: (empty) output histogram, float, shape (num_bins,)
ref_output, optional: (empty) output codebook histogram, float, shape (num_bins,)
*/

void test_distribution
(
    at::Tensor& input,
    at::Tensor& dist_output,
    const c10::optional<at::Tensor>& ref_output,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(input, kFloat);

    uint64_t numel = input.numel();
    float* ref_output_ptr = (float*) OPTPTR(ref_output);
    uint64_t num_bins = dist_output.numel();
    TORCH_CHECK(num_bins <= MAX_BINS, "Too many bins");
    if (ref_output_ptr)
        TORCH_CHECK(num_bins == ref_output.value().numel());

    test_distribution_kernel<<<1, NUM_THREADS_TD, 0, stream>>>
    (
        (const float*) input.data_ptr(),
        (float*) dist_output.data_ptr(),
        (float*) ref_output_ptr,
        numel,
        num_bins,
        min_value,
        max_value,
        mcg,
        mul1
    );
}