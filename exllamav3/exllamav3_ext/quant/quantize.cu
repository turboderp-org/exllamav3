#include <cuda_fp16.h>
#include "quantize.cuh"
#include <array>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <map>
#include <cstdlib>
#include "../util.h"
#include "../util.cuh"
#include "codebook.cuh"
#include "exl3_devctx.cuh"
#include <cmath>

#define H_INF __ushort_as_half(0x7c00)

#include "comp_units/quantize_tiles_instances.cuh"
#include "quantize_tiles_kernel.cuh"

#define __(i, cb) quantize_tiles_kernel_k##i##_cb##cb()
static const std::array<fp_quantize_tiles_kernel, 24> quantize_tiles_kernel_instances
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

// 160-length rows (n-gram embedding vectors), mul1 codebook only
#define __(i) quantize_tiles_kernel_k##i##_cb2_l160()
static const std::array<fp_quantize_tiles_kernel, 8> quantize_tiles_kernel_instances_l160
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __

// Keep original instances for architectures outside the sm_120 tuning target.
#define __(i, cb) quantize_tiles_kernel_k##i##_cb##cb(true)
static const std::array<fp_quantize_tiles_kernel, 24> quantize_tiles_optimized_instances
{
    __(1, 0), __(2, 0), __(3, 0), __(4, 0), __(5, 0), __(6, 0), __(7, 0), __(8, 0),
    __(1, 1), __(2, 1), __(3, 1), __(4, 1), __(5, 1), __(6, 1), __(7, 1), __(8, 1),
    __(1, 2), __(2, 2), __(3, 2), __(4, 2), __(5, 2), __(6, 2), __(7, 2), __(8, 2)
};
#undef __

// 160-length rows (n-gram embedding vectors), mul1 codebook only
#define __(i) quantize_tiles_kernel_k##i##_cb2_l160(true)
static const std::array<fp_quantize_tiles_kernel, 8> quantize_tiles_optimized_instances_l160
{
    __(1), __(2), __(3), __(4), __(5), __(6), __(7), __(8)
};
#undef __


template <int cb>
__global__ void quantize_codebook_kernel(half* table)
{
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    table[state] = decode_3inst<cb>(state);
}

static const half2* quantize_codebook(int device, int cb, const at::Tensor& input)
{
    // Per-host-thread cache; the event permits safe reuse on another CUDA stream.
    struct Entry { at::Tensor table; at::cuda::CUDAEvent ready; };
    thread_local std::map<std::pair<int, int>, Entry> tables;
    auto& entry = tables[{device, cb}];
    auto stream = at::cuda::getCurrentCUDAStream();
    if (!entry.table.defined())
    {
        entry.table = at::empty({65536}, input.options().dtype(at::kHalf));
        auto* ptr = reinterpret_cast<half*>(entry.table.data_ptr());
        if (cb == 0) quantize_codebook_kernel<0><<<256, 256, 0, stream.stream()>>>(ptr);
        if (cb == 1) quantize_codebook_kernel<1><<<256, 256, 0, stream.stream()>>>(ptr);
        if (cb == 2) quantize_codebook_kernel<2><<<256, 256, 0, stream.stream()>>>(ptr);
        cuda_check(cudaPeekAtLastError());
        entry.ready.record(stream);
    }
    else entry.ready.block(stream);
    // Protect outstanding uses if the host thread exits and releases its cached tensor.
    c10::cuda::CUDACachingAllocator::recordStream(entry.table.storage().data_ptr(), stream);
    return reinterpret_cast<const half2*>(entry.table.data_ptr());
}


// Which architectures run the dense specializations (quantize_tiles_optimized.cuh) for a given K
// and codebook. On Ada and Ampere, the register-cached decoded values and the global codebook table
// do not pay off for K=3 and K=6. K=7 depends on the codebook.
// EXL3_QT_OPTIMIZED=1/0 forces the choice for testing purposes. The Python scratch allocation asks the
// extension through quantize_tiles_scratch, so the layouts always agree
bool quantize_tiles_use_optimized(int major, int minor, int K, int cb)
{
    if (const char* env = std::getenv("EXL3_QT_OPTIMIZED"))
        return env[0] == '1';
    if (major == 12) return true;
    if (major == 8 && minor == 9)
        return K == 1 || K == 2 || K == 4 || K == 5 || (K == 7 && cb == 0) || K == 8;
    if (major == 8 && minor == 6)
        return K == 1 || K == 2 || K == 4 || K == 5 || (K == 7 && cb != 2) || K == 8;
    return false;
}

// Kernel and launch geometry for one (K, codebook, L) on the current device: which implementation,
// its block size (read back from the compiled kernel, so a PTX-JIT'd instance launches with the
// size it was built for), dynamic shared memory and the resident blocks per SM
struct QtLaunch
{
    bool optimized;
    fp_quantize_tiles_kernel kernel;
    int num_threads;
    int shmem;
    int blocks_per_sm;
};

static QtLaunch qt_launch(int device, int K, int cb, int L)
{
    const auto* props = at::cuda::getDeviceProperties(device);
    const bool optimized = quantize_tiles_use_optimized(props->major, props->minor, K, cb);
    const int edges = 65536 >> K;
    const int cost_arrays = optimized && K == 1 ? 1 : (K >= 2 ? 2 : 0);
    const int shmem = cost_arrays * edges * sizeof(half) + L * sizeof(half) + 64 + 128;
    const auto& instances = optimized ? quantize_tiles_optimized_instances : quantize_tiles_kernel_instances;
    const auto& instances_l160 = optimized ? quantize_tiles_optimized_instances_l160 : quantize_tiles_kernel_instances_l160;
    auto kernel = L == 256 ? instances[K - 1 + 8 * cb] : instances_l160[K - 1];
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    cudaFuncAttributes attr;
    cuda_check(cudaFuncGetAttributes(&attr, kernel));
    int blocks_per_sm;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, attr.maxThreadsPerBlock, shmem));
    return {optimized, kernel, attr.maxThreadsPerBlock, shmem, blocks_per_sm};
}

// For the Python scratch allocation: whether the dense specialization runs for this configuration
// (byte / bit history layout) and the number of tiles one wave keeps resident
std::tuple<bool, int64_t> quantize_tiles_scratch(int device, int K, bool mcg, bool mul1, int L)
{
    const c10::cuda::CUDAGuard device_guard(device);
    TORCH_CHECK(K >= 1 && K <= 8, "quantize_tiles_scratch: K must be 1..8");
    TORCH_CHECK(L == 256 || (L == 160 && mul1), "quantize_tiles_scratch: length 160 requires the mul1 codebook");
    auto launch = qt_launch(device, K, mul1 ? 2 : mcg ? 1 : 0, L);
    return {launch.optimized, (int64_t) launch.blocks_per_sm * DevCtx::instance().get_num_sms(device)};
}


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
    const int L = input_tiles.size(1);
    TORCH_CHECK(L == 256 || L == 160, "quantize_tiles tile length must be 256 or 160");
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_indices);
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_tiles);
    TORCH_CHECK_DTYPE(input_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_indices, kShort);
    TORCH_CHECK(K >= 1 && K <= 8, "quantize_tiles K must be in range 1..8");

    const int edges = 65536 >> K;
    const int num_tiles = input_tiles.size(0);
    if (!num_tiles) return;

    for (const auto& tensor : {input_tiles, output_tiles, output_indices, temp_costs, temp_edges})
    {
        TORCH_CHECK(tensor.device() == input_tiles.device(), "quantize_tiles tensors must share a device");
        TORCH_CHECK(tensor.is_contiguous(), "quantize_tiles tensors must be contiguous");
    }
    TORCH_CHECK_DTYPE(temp_costs, kHalf);
    TORCH_CHECK(temp_costs.numel() > 0, "quantize_tiles requires nonempty cost scratch");
    int device;
    cuda_check(cudaGetDevice(&device));
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;
    TORCH_CHECK(L == 256 || cb == 2, "quantize_tiles length 160 requires the mul1 codebook");
    const auto launch = qt_launch(device, K, cb, L);
    const bool optimized = launch.optimized;
    const auto kernel = launch.kernel;
    const int num_threads = launch.num_threads;
    const int shmem = launch.shmem;
    const int blocks_per_sm = launch.blocks_per_sm;
    int64_t scratch_tiles;
    if (optimized)
    {
        TORCH_CHECK(temp_edges.scalar_type() == at::kByte || temp_edges.scalar_type() == at::kShort,
                    "quantize_tiles optimized history must be byte or short storage");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(temp_edges.data_ptr()) % (K == 1 ? 4 : 2) == 0,
                    "quantize_tiles history has insufficient alignment");
        const int history_bytes = L * edges / (K == 1 ? 8 : 1);
        scratch_tiles = temp_edges.numel() * temp_edges.element_size() / history_bytes;
    }
    else
    {
        TORCH_CHECK_DIM(temp_costs, 3);
        TORCH_CHECK_SIZE(temp_costs, 1, 2);
        TORCH_CHECK_SIZE(temp_costs, 2, edges);
        TORCH_CHECK_DTYPE(temp_edges, kShort);
        TORCH_CHECK_DIM(temp_edges, 3);
        TORCH_CHECK_SIZE(temp_edges, 1, L);
        TORCH_CHECK_SIZE(temp_edges, 2, edges);
        scratch_tiles = MIN(temp_costs.size(0), temp_edges.size(0));
    }
    const int max_batch_size = (int) MIN(scratch_tiles, (int64_t) blocks_per_sm * DevCtx::instance().get_num_sms(device));
    TORCH_CHECK(max_batch_size > 0, "quantize_tiles scratch must hold at least one tile");
    const half2* lut = optimized && K == 6 ? quantize_codebook(device, cb, input_tiles) : nullptr;

    for (int batch_i = 0; batch_i < num_tiles; batch_i += max_batch_size)
    {
        const int bsz = MIN(max_batch_size, num_tiles - batch_i);
        kernel<<<bsz, num_threads, shmem, stream>>>
        (
            ((const float*) input_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((float*) output_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((uint16_t*) output_indices.data_ptr()) + (int64_t) L * batch_i,
            (half*) temp_costs.data_ptr(),
            (uint16_t*) temp_edges.data_ptr(),
            lut
        );
        cuda_check(cudaPeekAtLastError());
    }
}

// Fractional-rate quantizer, mul1 codebook only: KA bits per position plus one extra bit where the 16-position MASK is set.
// Costs and history live in the caller's scratch: temp_costs (batch, 2, 65536 >> KA) half,
// temp_edges (batch, 256, 65536 >> KA) short
void quantize_tiles_frac
(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int KA,
    int64_t MASK
)
{
    const at::cuda::OptionalCUDAGuard device_guard(input_tiles.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DIM(input_tiles, 2);
    const int L = input_tiles.size(1);
    TORCH_CHECK(L == 256, "quantize_tiles_frac: tile length must be 256");
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_indices);
    TORCH_CHECK_SHAPES_FULL(input_tiles, output_tiles);
    TORCH_CHECK_DTYPE(input_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_tiles, kFloat);
    TORCH_CHECK_DTYPE(output_indices, kShort);
    TORCH_CHECK_DTYPE(temp_costs, kHalf);
    TORCH_CHECK_DTYPE(temp_edges, kShort);
    const int edges_max = 65536 >> KA;
    TORCH_CHECK_DIM(temp_costs, 3);
    TORCH_CHECK_SIZE(temp_costs, 1, 2);
    TORCH_CHECK_SIZE(temp_costs, 2, edges_max);
    TORCH_CHECK_DIM(temp_edges, 3);
    TORCH_CHECK_SIZE(temp_edges, 1, L);
    TORCH_CHECK_SIZE(temp_edges, 2, edges_max);
    for (const auto& tensor : {input_tiles, output_tiles, output_indices, temp_costs, temp_edges})
        TORCH_CHECK(tensor.is_contiguous() && tensor.device() == input_tiles.device(), "quantize_tiles_frac: layout");
    fp_quantize_tiles_kernel kernel = nullptr;
    struct { int ka; uint32_t mask; fp_quantize_tiles_kernel (*fn)(); } const table[] = {
        {1, 0xaaaau, &quantize_tiles_frac_kernel_a1_maaaa},
        {2, 0xaaaau, &quantize_tiles_frac_kernel_a2_maaaa},
        {3, 0xaaaau, &quantize_tiles_frac_kernel_a3_maaaa},
    };
    for (const auto& e : table) if (e.ka == KA && e.mask == (uint32_t) MASK) kernel = e.fn();
    TORCH_CHECK(kernel, "quantize_tiles_frac: no instance for (KA, MASK) = (", KA, ", ", MASK, ")");
    const int num_tiles = input_tiles.size(0);
    if (!num_tiles) return;
    const int shmem = L * sizeof(half) + 32 * sizeof(int) + 128;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
    const int max_batch_size = (int) MIN(temp_costs.size(0), temp_edges.size(0));
    for (int batch_i = 0; batch_i < num_tiles; batch_i += max_batch_size)
    {
        const int bsz = MIN(max_batch_size, num_tiles - batch_i);
        kernel<<<bsz, 512, shmem, stream>>>
        (
            ((const float*) input_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((float*) output_tiles.data_ptr()) + (int64_t) L * batch_i,
            ((uint16_t*) output_indices.data_ptr()) + (int64_t) L * batch_i,
            (half*) temp_costs.data_ptr(),
            (uint16_t*) temp_edges.data_ptr(),
            nullptr
        );
        cuda_check(cudaPeekAtLastError());
    }
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
    dim3 gridDim(CEIL_DIVIDE(cols, 64), rows);

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
