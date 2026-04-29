#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemm_kernel.cuh"
#include "exl3_kernel_map.cuh"
#include "exl3_devctx.cuh"
#include "exl3_gemv.cuh"
#include "coop_autotune.cuh"
#include <set>
#include <vector>

int exl3_gemm_tilesize_k_g[] = {EXL3_GEMM_TILESIZE_K};
int exl3_gemm_tilesize_n_g[] = {EXL3_GEMM_TILESIZE_N};
int exl3_gemm_blockdim_g[] = {EXL3_GEMM_BLOCKDIM};

/*
EXL3 matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float32, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

std::set<void*> kernel_attr_set[MAX_DEVICES] = {};

uint64_t gemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb
)
{
    uint64_t h = 1469598103934665603ull;
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) size_m);
    mix((uint64_t) size_k);
    mix((uint64_t) size_n);
    mix((uint64_t) K);
    mix(c_fp32 ? 1ull : 0ull);
    mix((uint64_t) device);
    mix((uint64_t) cc);
    mix((uint64_t) max_num_sms);
    mix((uint64_t) cb);
    return h;
}

uint64_t mgemm_autotune_hash
(
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool c_fp32,
    int device,
    int cc,
    int max_num_sms,
    int cb,
    int bszm_in,
    int bszm_out
)
{
    uint64_t h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb);
    auto mix = [&] (uint64_t v)
    {
        h ^= v;
        h *= 1099511628211ull;
    };
    mix((uint64_t) bszm_in);
    mix((uint64_t) bszm_out);
    return h;
}

int exl3_gemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DIM(B, 3);
    TORCH_CHECK_SHAPES(A, -1, B, 0, 16);
    TORCH_CHECK_SHAPES(C, -1, B, 1, 16);
    // TORCH_CHECK_SHAPES(A, 0, C, 0, 1);
    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kShort);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);

    // Get SU, optionally
    const half* suh_ptr = (const half*) OPTPTR(suh);
    half* A_had_ptr = nullptr;
    if (suh_ptr)
    {
        // TORCH_CHECK_SHAPES(suh.value(), 0, A, 1, 1);
        A_had_ptr = (half*) OPTPTR(A_had);
        // TORCH_CHECK(A_had_ptr, "Must supply A_had with suh");
        // TORCH_CHECK_SHAPES_FULL(A_had.value(), A);
    }

    // Get SV, optionally
    const half* svh_ptr = (const half*) OPTPTR(svh);
    // if (svh_ptr)
        // TORCH_CHECK_SHAPES(svh.value(), 0, B, 1, 16);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = force_num_sms ? force_num_sms : DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    int K = B.size(2) / 16;
    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();

    int size_m = 1;
    int dim = A.dim();
    for (int d = 0; d < dim - 1; ++d) size_m *= A.size(d);
    int size_k = A.size(-1);
    int size_n = B.size(1) * 16;

    // Select kernel
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int block_dim;
    int shape_idx;
    fp_exl3_gemm_kernel kernel;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_gemm_A, 0);
            graph->record_param(kernel_ptr, GP_gemm_B_trellis, 1);
            graph->record_param(kernel_ptr, GP_gemm_C, 2);
            graph->record_param(kernel_ptr, GP_gemm_B_suh, 7);
            graph->record_param(kernel_ptr, GP_gemm_A_had, 8);
            graph->record_param(kernel_ptr, GP_gemm_B_svh, 9);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = gemm_autotune_hash(MAX(size_m, 2), size_k, size_n, K, c_fp32, device, cc, num_sms, cb);
        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, SMEM_MAX, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        std::vector<CoopAutotuneCandidate> candidates;
        for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
        {
            if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

            fp_exl3_gemm_kernel candidate_kernel = get_gemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb);
            if (!candidate_kernel) continue;

            int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
            int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
            int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
            int max_candidate_sms = MAX(MIN(max_slices, num_sms), 1);

            candidates.push_back
            ({
                (void*) candidate_kernel,
                exl3_gemm_blockdim_g[candidate_shape_idx],
                max_candidate_sms,
                1,
                max_candidate_sms,
                candidate_shape_idx
            });
        }
        TORCH_CHECK(!candidates.empty(), "exl3_gemm autotune: no compatible kernel shapes");

        tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, SMEM_MAX, stream, (size_t) size_k * size_n);
        if (graph)
        add_graph_args((void*) tuned.kernel);
        cuda_check(cudaPeekAtLastError());
        return tuned.tag;
    }

    kernel = select_exl3_gemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb
    );
    if (!kernel) return 0;

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }
    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        num_sms,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
)
{
    return exl3_gemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        force_shape_idx,
        mcg,
        mul1,
        force_num_sms,
        nullptr
    );
}

/*
EXL3 multi matmul, A @ B -> C

- A: row-major A tensor, shape (m, k), dtype float16, contiguous
- B: EXL3-quantized B tensor, shape (k//16, n//16, 16*K), dtype uint16
- C: empty row-major C tensor, shape (m, n), dtype float16 or float23, contiguous. Does not need to be zero-initialized
- suh: optional, packed input scales/flips, shape (k//16), dtype float16
- A_had: required if suh given, may be reference to A, temporary storage for input transform, size and dtype as A
- svh: optional, packed output scales/flips, shape (n//16), dtype float16

limitations:
- k % 16 == 0
- n % 128 == 0
*/

int exl3_mgemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(A.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(A, kHalf);
    TORCH_CHECK_DTYPE(B, kLong);
    TORCH_CHECK_DTYPE(suh, kLong);
    TORCH_CHECK_DTYPE(svh, kLong);
    bool c_fp32 = C.dtype() == at::kFloat;
    if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);
    TORCH_CHECK_DIM(A, 3);
    TORCH_CHECK_DIM(B, 1);
    TORCH_CHECK_DIM(suh, 1);
    TORCH_CHECK_DIM(svh, 1);
    TORCH_CHECK_DIM(C, 3);

    TORCH_CHECK_SHAPES(A, 1, C, 1, 1);
    TORCH_CHECK_SHAPES(B, 0, suh, 0, 1);
    TORCH_CHECK_SHAPES(B, 0, svh, 0, 1);

    int bsz = A.size(1);
    int bszm_in = A.size(0);
    int bszm_out = C.size(0);
    int bszm = MAX(bszm_in, bszm_out);

    const long* indices_ptr = (const long*) OPTPTR(indices);
    const half* weights_ptr = (const half*) OPTPTR(weights);

    if (indices)
    {
        TORCH_CHECK_DIM(indices.value(), 2);
        int num_indices = indices.value().size(1);
        TORCH_CHECK(num_indices <= bszm_in || num_indices <= bszm_out, "mgemm: too many indices for tensor batch");
        if (bszm_in > num_indices) bszm_in = num_indices;
        if (bszm_out > num_indices) bszm_out = num_indices;
    }

    if (weights)
    {
        TORCH_CHECK_DIM(weights.value(), 2);
    }

    int size_m = A.size(1);
    int size_k = A.size(2);
    int size_n = C.size(2);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int total_sms = DevCtx::instance().get_num_sms(device);
    int num_sms = force_num_sms ? force_num_sms : total_sms;
    int cc = DevCtx::instance().get_cc(device);
    int* locks = DevCtx::instance().get_locks(device);

    // Dispatch
    const half* A_ptr = (const half*) A.data_ptr();
    const uintptr_t* B_ptr_ptr = (const uintptr_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    const half* A_had_ptr = (const half*) A_had.data_ptr();
    const uintptr_t* suh_ptr_ptr = (const uintptr_t*) suh.data_ptr();
    const uintptr_t* svh_ptr_ptr = (const uintptr_t*) svh.data_ptr();

    // Select kernel
    TORCH_CHECK(!(mcg && mul1), "Specified both mcg and mul1")
    int cb = 0;
    if (mcg) cb = 1;
    if (mul1) cb = 2;

    int shape_idx;
    int block_dim;
    fp_exl3_mgemm_kernel kernel;
    int concurrency;

    void* kernelArgs[] =
    {
        (void*)& A_ptr,
        (void*)& B_ptr_ptr,
        (void*)& C_ptr,
        (void*)& size_m,
        (void*)& size_k,
        (void*)& size_n,
        (void*)& locks,
        (void*)& suh_ptr_ptr,
        (void*)& A_had_ptr,
        (void*)& svh_ptr_ptr,
        (void*)& indices_ptr,
        (void*)& weights_ptr,
        (void*)& bszm_in,
        (void*)& bszm_out,
        (void*)& min_index,
        (void*)& max_index
    };

    auto add_graph_args = [&](void* kernel_ptr)
    {
        if (graph)
        {
            graph->record_param(kernel_ptr, GP_mgemm_A, 0);
            graph->record_param(kernel_ptr, GP_mgemm_C, 2);
            graph->record_param(kernel_ptr, GP_mgemm_indices, 10);
            graph->record_param(kernel_ptr, GP_mgemm_weights, 11);
            graph->record_param(kernel_ptr, GP_end, 0);
        }
    };

    bool autotune = force_shape_idx <= 0 && force_num_sms <= 0;
    if (autotune)
    {
        uint64_t autotune_key = mgemm_autotune_hash
        (
            size_m, size_k, size_n, K, c_fp32, device, cc, total_sms, cb, bszm_in, bszm_out
        );

        CoopAutotuneLaunch tuned;
        if (CoopKernelAutotuner::launch_locked(autotune_key, kernelArgs, SMEM_MAX, stream, &tuned))
        {
            add_graph_args((void*) tuned.kernel);
            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
        if (!graph)
        {
            std::vector<CoopAutotuneCandidate> candidates;
            for (int candidate_shape_idx = 1; candidate_shape_idx <= EXL3_GEMM_NUM_SHAPES; ++candidate_shape_idx)
            {
                if (!exl3_gemm_shape_compat(candidate_shape_idx, size_m, size_k, size_n, K)) continue;

                fp_exl3_mgemm_kernel candidate_kernel = get_mgemm_kernel_ptr(K, candidate_shape_idx, c_fp32, cb);
                if (!candidate_kernel) continue;

                int tilesize_k = exl3_gemm_tilesize_k_g[candidate_shape_idx];
                int tilesize_n = exl3_gemm_tilesize_n_g[candidate_shape_idx];
                int max_slices = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
                int max_candidate_sms = MAX(MIN(max_slices, total_sms), 1);

                candidates.push_back
                ({
                    (void*) candidate_kernel,
                    exl3_gemm_blockdim_g[candidate_shape_idx],
                    max_candidate_sms,
                    bszm,
                    total_sms,
                    candidate_shape_idx
                });
            }
            TORCH_CHECK(!candidates.empty(), "exl3_mgemm autotune: no compatible kernel shapes");

            tuned = CoopKernelAutotuner::launch(autotune_key, candidates, kernelArgs, SMEM_MAX, stream, (size_t) size_k * size_n * bszm);
            add_graph_args((void*) tuned.kernel);

            // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, tuned.tag, tuned.block_dim, tuned.num_sms, tuned.concurrency);

            cuda_check(cudaPeekAtLastError());
            return tuned.tag;
        }
    }

    kernel = select_exl3_mgemm_kernel
    (
        cc, size_m, size_k, size_n, K, c_fp32,
        force_shape_idx, &block_dim, &shape_idx,
        &num_sms, cb, bszm_in, bszm_out
    );
    int tilesize_k = exl3_gemm_tilesize_k_g[shape_idx];
    int tilesize_n = exl3_gemm_tilesize_n_g[shape_idx];
    int tiles = MAX(size_k / tilesize_k * size_n / tilesize_n, 1);
    num_sms = tiles;
    if (num_sms * bszm > total_sms) num_sms = MAX(total_sms / bszm, 1);
    if (num_sms <= total_sms && tiles / num_sms > 48) num_sms = MIN(total_sms, num_sms * 2);
    concurrency = MIN(total_sms / num_sms, bszm);

    // DBGI10(size_m, size_k, size_n, K, bszm_in, bszm_out, shape_idx, block_dim, num_sms, concurrency);

    // Launch bigger grid if possible
    dim3 block_grid(num_sms, 1, concurrency);

    // Launch
    if (kernel_attr_set[device].find((void*) kernel) == kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        kernel_attr_set[device].insert((void*) kernel);
    }

    cudaLaunchCooperativeKernel
    (
        (void*) kernel,
        block_grid,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );
    add_graph_args((void*) kernel);

    cuda_check(cudaPeekAtLastError());
    return shape_idx;
}

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms
)
{
    return exl3_mgemm_gr
    (
        A,
        B,
        C,
        suh,
        A_had,
        svh,
        indices,
        weights,
        K,
        force_shape_idx,
        mcg_mult,
        mul1_mult,
        min_index,
        max_index,
        force_num_sms,
        nullptr
    );
}
