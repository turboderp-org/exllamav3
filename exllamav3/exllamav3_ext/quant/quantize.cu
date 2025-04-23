#include "quantize.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"
#include <cmath>

#define NUM_THREADS 1024

template <int K>
__global__ __launch_bounds__(1024)
void quantize_tiles_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    float* __restrict__ temp_costs_ptr,
    uint16_t* __restrict__ temp_edges_ptr
)
{
    int tile_idx = blockIdx.x;

    constexpr int Kr = 16 - K;
    constexpr int max_q = 1 << K;
    constexpr int edges = 65536 >> K;

    const float* input_tile = input_tiles_ptr + 256 * tile_idx;
    float* output_tile = output_tiles_ptr + 256 * tile_idx;
    uint16_t* output_indices = output_indices_ptr + 256 * tile_idx;
    float* temp_costs = temp_costs_ptr + 2 * edges * tile_idx;
    float* temp_costs_inc = temp_costs + edges;
    uint16_t* temp_edges = temp_edges_ptr + 256 * edges * tile_idx;

    auto forward = [&](int roll, int pre_state)
    {
        // Each thread iterates over all weights in the tile
        for (int i = 0; i < 256; ++i)
        {
            int ri = (i + roll) % 256;

            // Swap buffers.
            // temp_costs_inc[z] is the cost/cumulative error of an incoming edge from state (z & edge_mask)
            float* t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;

            for (int out_edge_idx = threadIdx.x; out_edge_idx < edges; out_edge_idx += NUM_THREADS)
            {
                float w = input_tile[ri];

                float min_err = INFINITY;
                int min_in_edge = 0;

                #pragma unroll
                for (int k = 0; k < max_q; ++k)
                {
                    int state = (k << Kr) | out_edge_idx;

                    float err = decode_pcb_f_diff(state, w);
                    err = err * err;

                    int in_edge_idx = state >> K;
                    if (i > 0)
                        err += temp_costs_inc[in_edge_idx];
                    else if (pre_state >= 0 && in_edge_idx != pre_state)
                        err = 1e30f;

                    if (err < min_err)
                    {
                        min_err = err;
                        min_in_edge = in_edge_idx;
                    }
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

        float local_min = 1e30f;
        int local_idx = -1;
        for (int e = threadIdx.x; e < edges; e += NUM_THREADS)
        {
            float v = temp_costs_inc[e];
            if (v < local_min)
            {
                local_min = v;
                local_idx = e;
            }
        }

        // Shuffle reduction
        int lane_id = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_min = __shfl_down_sync(0xffffffff, local_min, offset, 32);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset, 32);
            if (other_min < local_min)
            {
                local_min = other_min;
                local_idx = other_idx;
            }
        }

        __shared__ float s_min[32];
        __shared__ int s_idx[32];

        s_min[warp_id] = local_min;
        s_idx[warp_id] = local_idx;
        __syncthreads();

        if (warp_id == 0)
        {
            local_min = lane_id * 32 < edges ? s_min[lane_id] : 1e31f;
            local_idx = s_idx[lane_id];

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                float other_min = __shfl_down_sync(0xffffffff, local_min, offset, 32);
                int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset, 32);
                if (other_min < local_min)
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

        if (threadIdx.x == 0)
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
                    output_tile[ri] = __half2float(decode_pcb(encoded));
                }
                else if (ri == 0) break;
            }
        }

        // Broadcast to block
        __shared__ int broadcast;
        if (threadIdx.x == 0) broadcast = edge;
        __syncthreads();
        edge = broadcast;

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

#define __(i) quantize_tiles_kernel<i>
constexpr auto quantize_tiles_kernel_instances = std::array
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
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
    int K
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

    TORCH_CHECK_DTYPE(temp_costs, kFloat);
    TORCH_CHECK_DIM(temp_costs, 3);
    TORCH_CHECK_SIZE(temp_costs, 1, 2);
    TORCH_CHECK_SIZE(temp_costs, 2, edges);

    TORCH_CHECK_DTYPE(temp_edges, kShort);
    TORCH_CHECK_DIM(temp_edges, 3);
    TORCH_CHECK_SIZE(temp_edges, 1, 256);
    TORCH_CHECK_SIZE(temp_edges, 2, edges);

    int max_batch_size = temp_costs.size(0);

    int batch_i = 0;
    do
    {
        int batch_j = MIN(batch_i + max_batch_size, num_tiles);

        const float* input_tiles_ptr = ((const float*) input_tiles.data_ptr()) + 256 * batch_i;
        float* output_tiles_ptr = ((float*) output_tiles.data_ptr()) + 256 * batch_i;
        uint16_t* output_indices_ptr = ((uint16_t*) output_indices.data_ptr()) + 256 * batch_i;
        float* temp_costs_ptr = (float*) temp_costs.data_ptr();
        uint16_t* temp_edges_ptr = (uint16_t*) temp_edges.data_ptr();

        int bsz = batch_j - batch_i;

        quantize_tiles_kernel_instances[K - 1]<<<bsz, threads, 0, stream>>>
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
    int cols
)
{
    int col = threadIdx.x + blockIdx.x * 64;
    if (col >= cols) return;
    int row = blockIdx.y;
    int idx = row * cols + col;

    uint16_t enc = input_tiles_ptr[idx];
    if constexpr (std::is_same_v<T, float>)
        output_tiles_ptr[idx] = __half2float(decode_pcb((uint64_t) enc));
    else
        output_tiles_ptr[idx] = decode_pcb((uint64_t) enc);
}

/*
Decode tensor

input_indices: uint16_t
output_tiles: float or half
*/

void decode
(
    at::Tensor input_indices,
    at::Tensor output_tiles
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
            cols
        );
    else if (output_tiles.dtype() == at::kHalf)
        decode_kernel<<<gridDim, blockDim, 0, stream>>>
        (
            (const uint16_t*) input_indices.data_ptr(),
            (half*) output_tiles.data_ptr(),
            cols
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
    float max_value
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

    reset_histogram();
    for (uint64_t i = threadIdx.x; i < 65536; i += NUM_THREADS_TD)
        count(decode_3inst_f((uint16_t) (i & 0xffff)));
    __syncthreads();
    write_histogram(ref_output_ptr, 65536);

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
ref_output: (empty) output codebook histogram, float, shape (num_bins,)
*/

void test_distribution
(
    at::Tensor input,
    at::Tensor dist_output,
    at::Tensor ref_output,
    float min_value,
    float max_value
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(input, kFloat);

    uint64_t numel = input.numel();
    uint64_t num_bins = ref_output.numel();
    TORCH_CHECK(num_bins <= MAX_BINS, "Too many bins");

    test_distribution_kernel<<<1, NUM_THREADS_TD, 0, stream>>>
    (
        (const float*) input.data_ptr(),
        (float*) dist_output.data_ptr(),
        (float*) ref_output.data_ptr(),
        numel,
        num_bins,
        min_value,
        max_value
    );
}