#pragma once

#include <torch/version.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include "quant/exl3_devctx.cuh"

// Bind this thread/device handle to ExLlama's persistent workspace.
// PyTorch 2.12 introduced the setup flag; keep the 2.6+ no-argument API
// for older versions and unqualified future versions. ROCm owns its setup policy.
inline cublasHandle_t exl3_cublas_handle(cudaStream_t stream)
{
#if defined(USE_ROCM) || TORCH_VERSION_MAJOR != 2 || TORCH_VERSION_MINOR < 12 || TORCH_VERSION_MINOR > 14
    // Preserve backend math/atomics setup. This may reserve a redundant Torch
    // workspace on CUDA, but does not require new ATen APIs.
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
#else
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle(false);
    // Match the verified 2.12-2.14 CUDA math-mode policy while skipping setup.
    const bool tf32 = !at::NoTF32Guard::should_disable_tf32() &&
        at::globalContext().float32Precision(at::Float32Backend::CUDA, at::Float32Op::MATMUL) == at::Float32Precision::TF32;
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
#endif
    TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
    TORCH_CUDABLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    int device;
    AT_CUDA_CHECK(cudaGetDevice(&device));
    void* workspace = DevCtx::instance().get_ws(device);
    TORCH_CUDABLAS_CHECK(cublasSetWorkspace(handle, workspace, WORKSPACE_SIZE));
    return handle;
}
