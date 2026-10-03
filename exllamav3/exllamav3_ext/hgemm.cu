#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <limits>
#include <vector>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

// The fp32-output reconstruct GEMM is the single largest prefill item: the model's q/k/v/gate/up
// projections are built with out_dtype=torch.float (exllamav3/architecture/*.py), and on this stack
// an fp16-input GEMM with an fp32 C/D (Tensile "HSS", MT64x32x8) runs at 18-19 TFLOP/s through
// hipBLAS *and* hipBLASLt, against 92-103 TFLOP/s for the identical call with an fp16 C/D. Measured
// cold-rotation at m=2048, the dominant shape (k=5120, n=17408): 20.07 ms fp32-out vs 4.14 ms
// fp16-out-plus-convert - 4.85x (profiling/f32out_gemm_probe.py, plan doc D1).
//
// EXL3_HGEMM_F16OUT runs the GEMM into an fp16 slab and widens it (the accumulation stays fp32;
// the only numeric change is one rounding of the result to fp16, the same precision the residual
// stream already carries). Off by default: it ships only with the logits KLD gate.
// Default ON: measured +60-73% prefill 2k (bench_lean, two runs per arm on one image: 324.7/345.3
// -> 562.0/511.1 tok/s) with decode and every batch aggregate unchanged, at a model-level KLD of
// 1.9e-3 worst (5.4e-4 on the long-context prompt) and zero greedy divergence in 8 steps - inside
// the 1.1-4.1e-3 band already accepted for the msq/T1 prefill route (findings-log 10.3).
// EXL3_HGEMM_F16OUT=0 restores the fp32-output path exactly.
static bool hgemm_f16out_enabled()
{
    static const bool on = []
    {
        const char* e = getenv("EXL3_HGEMM_F16OUT");
#if defined(USE_ROCM)
        // Measured win on RDNA3 (hipBLAS fp32-out GEMM runs ~5x slower than fp16-out);
        // the fp16 slab rounds each output once to fp16, the precision the residual
        // stream already carries.
        return e ? atoi(e) != 0 : true;
#else
        // On CUDA the fp32-output path is not known to be slow; keep the exact path
        // unless asked.
        return e ? atoi(e) != 0 : false;
#endif
    }();
    return on;
}

// Grow-only fp16 slab, one per device, sized on the first call. Deliberately a plain process-lifetime
// allocation with a stable pointer: it is sized during the first (eager) call for a shape, so a
// captured graph never sees it move.
static at::Tensor hgemm_f16_scratch(const at::Tensor& c, int size_m, int size_n)
{
    static std::vector<at::Tensor> scratch;
    int device = c.get_device();
    if ((int) scratch.size() <= device) scratch.resize(device + 1);
    at::Tensor& s = scratch[device];
    int64_t numel = (int64_t) size_m * (int64_t) size_n;
    if (!s.defined() || s.numel() < numel)
        s = at::empty({numel}, c.options().dtype(at::kHalf));
    return s;
}


static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;

    // c.dim() == 2 only: all in-tree callers pass the (m, n) slab that reconstruct_hgemm builds;
    // a higher-rank c keeps the incumbent path rather than risking a shape mismatch.
    if (output_fp32 && c.dim() == 2 && hgemm_f16out_enabled())
    {
        // Same call, fp16 destination, then widen into c (which may be a strided slice view)
        at::Tensor scratch = hgemm_f16_scratch(c, size_m, size_n);
        auto r16 = cublasGemmEx
        (
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            size_n, size_m, size_k,
            &alpha_, b_ptr, CUDA_R_16F, size_n,
                     a_ptr, CUDA_R_16F, size_k,
            &beta_,  scratch.data_ptr(), CUDA_R_16F, size_n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        cublas_check(r16);
        cuda_check(cudaPeekAtLastError());
        // The slab is grow-only, so it is usually larger than this call needs: narrow it to the
        // exact element count before reshaping (a plain .view() would reject a reused larger slab
        // as soon as a smaller shape follows a larger one - caught by the per-shape probe).
        int64_t numel = (int64_t) size_m * (int64_t) size_n;
        c.copy_(scratch.narrow(0, 0, numel).view({size_m, size_n}));
        return;
    }

    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, size_k,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS
    if (hgemm_f16acc_try(a, w, c)) return;

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}
