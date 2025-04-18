#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include "util.h"
#include "util.cuh"

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16
- if c is float32, operation is float16 @ float16 -> float23
*/

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    bool output_fp32 = c.dtype() == at::kFloat;
    if (!output_fp32)
        TORCH_CHECK_DTYPE(c, kHalf);

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(a, 2);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK_DIM(c, 2);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, 1, 1);

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_m = a.size(0);
    int size_k = a.size(1);
    int size_n = b.size(1);

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);

    if (!output_fp32)
    {
        half alpha_ = __float2half(1.0f);
        half beta_ = __float2half(0.0f);

        half* c_ptr = (half*) c.data_ptr();
        cublasHgemm
        (
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            size_n, size_m, size_k,
            &alpha_, b_ptr, size_n,
                     a_ptr, size_k,
            &beta_,  c_ptr, size_n
        );
    }
    else
    {
        float alpha_ = 1.0f;
        float beta_ = 0.0f;

        float* c_ptr = (float*) c.data_ptr();
        cublasGemmEx
        (
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            size_n, size_m, size_k,
            &alpha_, b_ptr, CUDA_R_16F, size_n,
                     a_ptr, CUDA_R_16F, size_k,
            &beta_,  c_ptr, CUDA_R_32F, size_n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
}
